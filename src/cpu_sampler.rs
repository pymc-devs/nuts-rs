use itertools::Itertools;
use rand::{prelude::StdRng, Rng, SeedableRng};
use std::thread::{spawn, JoinHandle};
use rayon::prelude::*;
use thiserror::Error;

use crate::{
    cpu_potential::EuclideanPotential,
    cpu_state::State,
    mass_matrix::{DiagAdaptExp, MassMatrix},
    nuts::{Collector, NutsOptions, SampleInfo, Potential, SampleStats, NutsSampler, Sampler, AdaptivePotential},
    stepsize::DualAverage,
};

pub use crate::cpu_potential::CpuLogpFunc;
pub use crate::stepsize::DualAverageSettings;
pub use crate::mass_matrix::DiagAdaptExpSettings;
pub use crate::nuts::NutsError;



pub struct EuclideanSampler<F: CpuLogpFunc, M: MassMatrix, C: Collector> {
    sampler: NutsSampler<EuclideanPotential<F, DiagAdaptExp>, rand::rngs::StdRng, GlobalCollector>
}


impl<F: CpuLogpFunc, M: MassMatrix, C: Collector> EuclideanSampler<F, M, C> {
    pub fn new(
        logp: F,
        args: SamplerArgs,
        mass_matrix_settings: DiagAdaptExpSettings,
        chain: u64,
        seed: u64,
    ) -> EuclideanSampler<F, M, C> {
        let dim = logp.dim();

        let mass_matrix = DiagAdaptExp::new(dim, mass_matrix_settings);
        let potential = EuclideanPotential::new(logp, mass_matrix, args.max_energy_error, args.initial_step);
        let collector = GlobalCollector::new(dim);
        let options = NutsOptions {
            step_size: args.initial_step,
            maxdepth: args.maxdepth,
            max_energy_error: args.max_energy_error,
        };
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        EuclideanSampler {
            sampler: NutsSampler::new(potential, collector, options, rng, chain)
        }
    }
}

impl<F: CpuLogpFunc, M: MassMatrix, C: Collector> Sampler for EuclideanSampler<F, M, C> {
    type Potential = EuclideanPotential<F, M>;
    type Collector = C;

    fn set_position(&mut self, position: &[f64]) -> Result<(), F::Err> {
        self.sampler.set_position(position)
    }

    fn draw(&mut self) -> Result<(Box<[f64]>, SampleStats<EuclideanPotential<F, DiagAdaptExp>, GlobalCollector>), NutsError> {
        self.sampler.draw()
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[derive(Clone, Copy)]
pub struct SamplerArgs {
    pub num_tune: u64,
    pub initial_step: f64,
    pub maxdepth: u64,
    pub max_energy_error: f64,
    pub step_size_adapt: DualAverageSettings,
    pub mass_matrix: DiagAdaptExpSettings,
}

impl Default for SamplerArgs {
    fn default() -> Self {
        Self {
            num_tune: 1000,
            initial_step: 0.1,
            maxdepth: 10,
            max_energy_error: 1000f64,
            step_size_adapt: DualAverageSettings::default(),
            mass_matrix: DiagAdaptExpSettings::default(),
        }
    }
}

pub struct AdaptiveSampler<F: CpuLogpFunc> {
    num_tune: u64,
    sampler: EuclideanSampler<F, DiagAdaptExp, GlobalCollector>,
    step_size_adapt: DualAverage,
    draw_count: u64,
}

impl<F: CpuLogpFunc> AdaptiveSampler<F> {
    pub fn new(logp: F, args: SamplerArgs, chain: u64, seed: u64) -> Self {
        let sampler = EuclideanSampler::new(logp, args, args.mass_matrix, chain, seed);
        let step_size_adapt = DualAverage::new(args.step_size_adapt, args.initial_step);

        Self {
            num_tune: args.num_tune,
            sampler,
            step_size_adapt,
            draw_count: 0,
        }
    }

    pub fn set_position(&mut self, position: &[f64]) -> Result<(), F::Err> {
        self.sampler.set_position(position)
    }

    pub fn init_random<R: Rng + ?Sized, I: InitPointFunc>(
        &mut self,
        rng: &mut R,
        func: &mut I,
        n_try: u64,
    ) -> Result<(), F::Err> {
        let mut last_error: Option<F::Err> = None;
        let mut position = vec![0.; self.dim()];
        for _ in 0..n_try {
            func.new_init_point(rng, &mut position);
            match self.set_position(&position) {
                Ok(_) => return Ok(()),
                Err(e) => last_error = Some(e),
            }
        }
        Err(last_error.unwrap())
    }

    pub fn tuning(&self) -> bool {
        self.draw_count <= self.num_tune
    }

    pub fn draw(&mut self) -> Result<(Box<[f64]>, SampleStats<EuclideanPotential<F, DiagAdaptExp>, GlobalCollector>), NutsError> {
        let step_size = if self.tuning() {
            self.step_size_adapt.current_step_size()
        } else {
            self.step_size_adapt.current_step_size_adapted()
        };
        let step_size_bar = self.step_size_adapt.current_step_size_adapted();

        self.sampler.sampler.options.step_size = step_size;

        self.draw_count += 1;
        let (position, mut stats) = self.sampler.draw()?;

        stats.step_size_bar = step_size_bar;

        if self.tuning() {
            self.step_size_adapt.advance(stats.mean_acceptance_rate);
            let reset_step_size = self.sampler
                .potential
                .mass_matrix_mut()
                .adapt(&self.sampler.collector.mass_matrix_tuning);
            if reset_step_size {
                self.step_size_adapt.reset(self.step_size_adapt.current_step_size_adapted())
            }
        }
        Ok((position, stats))
    }

    pub fn dim(&self) -> usize {
        self.sampler.dim()
    }
}

pub trait InitPointFunc {
    fn new_init_point<R: Rng + ?Sized>(&mut self, rng: &mut R, out: &mut [f64]);
}


#[derive(Error, Debug)]
pub enum ParallelSamplingError {
    #[error("Could not send sample to controller thread")]
    ChannelClosed {
        #[from]
        source: flume::SendError<()>
    },
    #[error("Nuts failed because of unrecoverable logp function error")]
    NutsError {
        #[from]
        source: NutsError
    }
}

pub fn sample_parallel<F: CpuLogpFunc + Clone + Send + 'static, I: InitPointFunc>(
    func: F,
    init_point_func: &mut I,
    settings: SamplerArgs,
    n_chains: u64,
    n_draws: u64,
    seed: u64,
    n_try_init: u64,
) -> Result<
    (
        JoinHandle<Vec<Result<(), ParallelSamplingError>>>,
        flume::Receiver<(
            Box<[f64]>,
            SampleStats<EuclideanPotential<F, DiagAdaptExp>, GlobalCollector>
        )>,
    ),
    F::Err,
> {
    let mut func = func;
    let draws = settings.num_tune + n_draws;
    let mut grad = vec![0.; func.dim()];
    let mut rng = StdRng::seed_from_u64(seed.wrapping_sub(1));
    let mut points: Vec<Result<Box<[f64]>, F::Err>> = (0..n_chains)
        .map(|_| {
            let mut position = vec![0.; func.dim()];
            init_point_func.new_init_point(&mut rng, &mut position);

            let mut error = None;
            for _ in 0..n_try_init {
                match func.logp(&mut position, &mut grad) {
                    Err(e) => error = Some(e),
                    Ok(_) => {
                        error = None;
                        break;
                    }
                }
            }
            match error {
                Some(e) => Err(e),
                None => Ok(position.into()),
            }
        })
        .collect();

    let points: Result<Vec<Box<[f64]>>, _> = points.drain(..).collect();
    let points = points?;

    let (sender, receiver) = flume::bounded(128);

    let funcs = (0..n_chains).map(|_| func.clone()).collect_vec();
    let handle = spawn(move || {
        let results: Vec<Result<(), ParallelSamplingError<F::Err>>> = points
            .into_par_iter()
            .zip(funcs)
            //.with_max_len(1)
            .enumerate()
            .map_with(sender, |sender, (chain, (point, func))| {
                let mut sampler = AdaptiveSampler::new(
                    func,
                    settings,
                    chain as u64,
                    (chain as u64).wrapping_add(seed),
                );
                sampler
                    .set_position(&point)
                    .expect("Could not eval logp at initial positon, though we could previously.");
                for _ in 0..draws {
                    sender.send(sampler.draw()?)?;
                }
                Ok(())
            })
            .collect();
        results
    });

    Ok((handle, receiver))
}

pub mod test_logps {
    use crate::{cpu_potential::CpuLogpFunc, nuts::LogpError};
    use thiserror::Error;

    #[derive(Clone)]
    pub struct NormalLogp {
        dim: usize,
        mu: f64,
    }

    impl NormalLogp {
        pub fn new(dim: usize, mu: f64) -> NormalLogp {
            NormalLogp { dim, mu }
        }
    }

    #[derive(Error, Debug)]
    pub enum NormalLogpError {}
    impl LogpError for NormalLogpError {
        fn is_recoverable(&self) -> bool {
            false
        }
    }

    impl CpuLogpFunc for NormalLogp {
        type Err = NormalLogpError;

        fn dim(&self) -> usize {
            self.dim
        }
        fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, NormalLogpError> {
            let n = position.len();
            assert!(gradient.len() == n);

            let mut logp = 0f64;
            for (p, g) in position.iter().zip(gradient.iter_mut()) {
                let val = *p - self.mu;
                logp -= val * val / 2.;
                *g = -val;
            }
            Ok(logp)
        }
    }
}

pub struct JitterInitFunc {}

impl JitterInitFunc {
    pub fn new() -> JitterInitFunc {
        JitterInitFunc {}
    }
}

impl InitPointFunc for JitterInitFunc {
    fn new_init_point<R: Rng + ?Sized>(&mut self, rng: &mut R, out: &mut [f64]) {
        rng.fill(out);
        out.iter_mut().for_each(|val| *val = 2. * *val - 1.);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cpu_sampler::{SamplerArgs, EuclideanSampler},
        mass_matrix::DiagAdaptExpSettings,
    };

    use super::test_logps::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn make_state() {
        /*
        let _ = State::new(10);
        let mut storage = Rc::new(StateStorage::with_capacity(0));
        let a = new_state(&mut storage, 10);
        assert!(storage.free_states.borrow_mut().len() == 0);
        drop(a);
        assert!(storage.free_states.borrow_mut().len() == 1);
        let a = new_state(&mut storage, 10);
        assert!(storage.free_states.borrow_mut().len() == 0);
        drop(a);
        assert!(storage.free_states.borrow_mut().len() == 1);
        */
    }

    #[test]
    fn deterministic() {
        let dim = 3usize;
        let func = NormalLogp::new(dim, 3.);
        let init = vec![3.5; dim];

        let settings = DiagAdaptExpSettings::default();
        let mut sampler = EuclideanSampler::new(func, SamplerArgs::default(), settings, 10, 42);

        sampler.set_position(&init).unwrap();
        let (sample1, stats1) = sampler.draw().unwrap();

        let func = NormalLogp::new(dim, 3.);
        let mut sampler = EuclideanSampler::new(func, SamplerArgs::default(), settings, 10, 42);

        sampler.set_position(&init).unwrap();
        let (sample2, stats2) = sampler.draw().unwrap();

        dbg!(&sample1);
        dbg!(stats1);

        dbg!(&sample2);
        dbg!(stats2);

        assert_eq!(sample1, sample2);
    }
}
