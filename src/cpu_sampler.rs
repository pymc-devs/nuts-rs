use itertools::Itertools;
use rand::{prelude::StdRng, Rng, SeedableRng};
use std::thread::{spawn, JoinHandle};
use rayon::prelude::*;
use thiserror::Error;

use crate::{
    mass_matrix::{DiagAdaptExpSettings, DiagMassMatrix},
    nuts::{SampleStats, Sampler, NutsError, NutsSampler},
    stepsize::DualAverageSettings, CpuLogpFunc, adapt_strategy::{DualAverageStrategy, ExpWindowDiagAdapt, CombinedStrategy}, cpu_potential::EuclideanPotential, NutsOptions
};


#[derive(Clone, Copy)]
pub struct SamplerArgs {
    pub num_tune: u64,
    pub maxdepth: u64,
    pub max_energy_error: f64,
    pub step_size_adapt: DualAverageSettings,
    pub mass_matrix: DiagAdaptExpSettings,
}

impl Default for SamplerArgs {
    fn default() -> Self {
        Self {
            num_tune: 1000,
            maxdepth: 10,
            max_energy_error: 1000f64,
            step_size_adapt: DualAverageSettings::default(),
            mass_matrix: DiagAdaptExpSettings::default(),
        }
    }
}


pub trait InitPointFunc {
    fn new_init_point<R: Rng + ?Sized>(&mut self, rng: &mut R, out: &mut [f64]);
}


#[derive(Error, Debug)]
pub enum ParallelSamplingError {
    #[error("Could not send sample to controller thread")]
    ChannelClosed(),
    #[error("Nuts failed because of unrecoverable logp function error")]
    NutsError {
        #[from]
        source: NutsError
    },
    #[error("Initialization of first point failed")]
    InitError {
        source: NutsError,
    }
}

pub type ParallelChainResult = Result<(), ParallelSamplingError>;

pub fn sample_parallel<F: CpuLogpFunc + Clone + Send + 'static, I: InitPointFunc> (
    func: F,
    init_point_func: &mut I,
    settings: SamplerArgs,
    n_chains: u64,
    n_draws: u64,
    seed: u64,
    n_try_init: u64,
) -> Result<
    (
        JoinHandle<Vec<ParallelChainResult>>,
        flume::Receiver<(Box<[f64]>, impl SampleStats)>,
    ),
    ParallelSamplingError
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
    let points = points.map_err(|e| NutsError::LogpFailure(Box::new(e)))?;

    let (sender, receiver) = flume::bounded(128);

    let funcs = (0..n_chains).map(|_| func.clone()).collect_vec();
    let handle = spawn(move || {
        let results: Vec<Result<(), ParallelSamplingError>> = points
            .into_par_iter()
            .zip(funcs)
            //.with_max_len(1)
            .enumerate()
            .map_with(sender, |sender, (chain, (point, func))| {
                let mut sampler = new_sampler(func, settings, chain as u64, seed);
                sampler.set_position(&point)?;
                sampler
                    .set_position(&point)
                    .expect("Could not eval logp at initial positon, though we could previously.");
                for _ in 0..draws {
                    let (point, info) = sampler.draw()?;
                    sender.send((point, info))
                        .map_err(|_| ParallelSamplingError::ChannelClosed())?;
                }
                Ok(())
            })
            .collect();
        results
    });

    Ok((handle, receiver))
}

pub fn new_sampler<F: CpuLogpFunc>(
    logp: F,
    settings: SamplerArgs,
    chain: u64,
    seed: u64,
) -> impl Sampler {
    use crate::nuts::AdaptStrategy;
    let num_tune = settings.num_tune;
    let step_size_adapt =
        DualAverageStrategy::new(DualAverageSettings::default(), num_tune, logp.dim());
    let mass_matrix_adapt =
        ExpWindowDiagAdapt::new(DiagAdaptExpSettings::default(), num_tune, logp.dim());

    let strategy = CombinedStrategy::new(step_size_adapt, mass_matrix_adapt);

    let mass_matrix = DiagMassMatrix::new(vec![1f64; logp.dim()].into());
    let max_energy_error = settings.max_energy_error;
    let step_size = settings.step_size_adapt.initial_step;

    let potential = EuclideanPotential::new(logp, mass_matrix, max_energy_error, step_size);
    let options = NutsOptions { maxdepth: settings.maxdepth };

    let rng = {
        rand::rngs::StdRng::seed_from_u64(seed)
    };

    NutsSampler::new(potential, strategy, options, rng, chain)
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

/*
#[cfg(test)]
mod tests {
    use crate::{
        cpu_sampler::SamplerArgs,
        mass_matrix::DiagAdaptExpSettings,
    };

    use super::test_logps::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn make_state() {
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
*/
