use itertools::Itertools;
use rand::{prelude::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::thread::{spawn, JoinHandle};

use crate::{
    cpu_potential::Potential,
    cpu_state::{State, StatePool},
    mass_matrix::DiagAdaptExp,
    nuts::{draw, Collector, DivergenceInfo, NutsOptions, SampleInfo},
    stepsize::DualAverage,
};

pub use crate::cpu_potential::CpuLogpFunc;
pub use crate::stepsize::DualAverageSettings;
pub use crate::mass_matrix::DiagAdaptExpSettings;

struct RunningMean {
    sum: f64,
    count: u64,
}

impl RunningMean {
    fn new() -> RunningMean {
        RunningMean { sum: 0., count: 0 }
    }

    fn add(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
    }

    fn current(&self) -> f64 {
        self.sum / self.count as f64
    }

    fn reset(&mut self) {
        self.sum = 0f64;
        self.count = 0;
    }
}

struct AcceptanceRateCollector {
    initial_energy: f64,
    mean: RunningMean,
}

impl AcceptanceRateCollector {
    fn new() -> AcceptanceRateCollector {
        AcceptanceRateCollector {
            initial_energy: 0.,
            mean: RunningMean::new(),
        }
    }
}

impl Collector for AcceptanceRateCollector {
    type State = State;

    fn register_leapfrog(
        &mut self,
        _start: &Self::State,
        end: &Self::State,
        divergence_info: Option<&dyn crate::nuts::DivergenceInfo>,
    ) {
        use crate::nuts::State;
        match divergence_info {
            Some(_) => self.mean.add(0.),
            None => self
                .mean
                .add(end.log_acceptance_probability(self.initial_energy).exp()),
        }
    }

    fn register_init(&mut self, state: &Self::State, _options: &NutsOptions) {
        use crate::nuts::State;
        self.initial_energy = state.energy();
        self.mean.reset();
    }
}

struct GlobalCollector {
    acceptance_rate: AcceptanceRateCollector,
    mass_matrix_tuning: crate::mass_matrix::AdaptCollector,
}

#[derive(Debug)]
struct CollectedStats {
    pub mean_acceptance_rate: f64,
}

impl GlobalCollector {
    fn new(dim: usize) -> GlobalCollector {
        GlobalCollector {
            acceptance_rate: AcceptanceRateCollector::new(),
            mass_matrix_tuning: crate::mass_matrix::AdaptCollector::new(dim),
        }
    }

    fn stats(&self) -> CollectedStats {
        CollectedStats {
            mean_acceptance_rate: self.acceptance_rate.mean.current(),
        }
    }
}

impl Collector for GlobalCollector {
    type State = State;

    fn register_leapfrog(
        &mut self,
        start: &Self::State,
        end: &Self::State,
        divergence_info: Option<&dyn crate::nuts::DivergenceInfo>,
    ) {
        self.acceptance_rate
            .register_leapfrog(start, end, divergence_info);
        self.mass_matrix_tuning
            .register_leapfrog(start, end, divergence_info);
    }

    fn register_draw(&mut self, state: &Self::State, info: &SampleInfo) {
        self.acceptance_rate.register_draw(state, info);
        self.mass_matrix_tuning.register_draw(state, info);
    }

    fn register_init(&mut self, state: &Self::State, options: &NutsOptions) {
        self.acceptance_rate.register_init(state, options);
        self.mass_matrix_tuning.register_init(state, options);
    }
}

#[derive(Debug)]
pub struct SampleStats {
    pub step_size: f64,
    pub step_size_bar: f64,
    pub depth: u64,
    pub maxdepth_reached: bool,
    pub idx_in_trajectory: i64,
    pub logp: f64,
    pub mean_acceptance_rate: f64,
    pub divergence_info: Option<Box<dyn DivergenceInfo>>,
    pub chain: u64,
    pub draw: u64,
    pub tree_size: u64,
    pub first_diag_mass_matrix: f64,
}

pub struct StaticSampler<F: CpuLogpFunc> {
    potential: Potential<F, DiagAdaptExp>,
    state: State,
    pool: StatePool,
    rng: rand::rngs::StdRng,
    collector: GlobalCollector,
    options: NutsOptions,
    draw_count: u64,
    chain: u64,
    dim: usize,
}

struct NullCollector {}

impl Collector for NullCollector {
    type State = State;
}

impl<F: CpuLogpFunc> StaticSampler<F> {
    pub fn new(
        logp: F,
        args: SamplerArgs,
        mass_matrix_settings: DiagAdaptExpSettings,
        chain: u64,
        seed: u64,
    ) -> StaticSampler<F> {
        let dim = logp.dim();

        let mass_matrix = DiagAdaptExp::new(dim, mass_matrix_settings);
        let mut pool = StatePool::new(dim);
        let potential = Potential::new(logp, mass_matrix);
        let state = pool.new_state();
        let collector = GlobalCollector::new(dim);
        let options = NutsOptions {
            step_size: args.initial_step,
            maxdepth: args.maxdepth,
            max_energy_error: args.max_energy_error,
        };
        StaticSampler {
            potential,
            state,
            pool,
            options,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            collector,
            draw_count: 0,
            chain,
            dim,
        }
    }

    pub fn set_position(&mut self, position: &[f64]) -> Result<(), F::Err> {
        use crate::nuts::Potential;
        {
            let inner = self.state.try_mut_inner().expect("State already in use");
            inner.q.copy_from_slice(position);
            inner.p_sum.fill(0.);
        }
        if let Err(err) = self.potential.update_potential_gradient(&mut self.state) {
            return Err(err.logp_function_error.unwrap());
        }
        Ok(())
    }

    pub fn draw(&mut self) -> (Box<[f64]>, SampleStats) {
        use crate::nuts::Potential;
        self.potential
            .randomize_momentum(&mut self.state, &mut self.rng);

        let (state, info) = draw(
            &mut self.pool,
            self.state.clone(),
            &mut self.rng,
            &mut self.potential,
            &self.options,
            &mut self.collector,
        );
        self.state = state;
        let position: Box<[f64]> = self.state.q.clone();
        let stats = SampleStats {
            step_size: self.options.step_size,
            step_size_bar: self.options.step_size,
            divergence_info: info.divergence_info,
            idx_in_trajectory: self.state.idx_in_trajectory,
            depth: info.depth,
            maxdepth_reached: info.maxdepth,
            logp: -self.state.potential_energy,
            mean_acceptance_rate: self.collector.stats().mean_acceptance_rate,
            chain: self.chain,
            draw: self.draw_count,
            tree_size: self.collector.acceptance_rate.mean.count,
            first_diag_mass_matrix: self.potential.mass_matrix_mut().current.variance[0],
        };
        self.draw_count += 1;
        (position, stats)
    }

    pub fn dim(&self) -> usize {
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
    sampler: StaticSampler<F>,
    step_size_adapt: DualAverage,
    draw_count: u64,
}

impl<F: CpuLogpFunc> AdaptiveSampler<F> {
    pub fn new(logp: F, args: SamplerArgs, chain: u64, seed: u64) -> Self {
        let sampler = StaticSampler::new(logp, args, args.mass_matrix, chain, seed);
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

    pub fn draw(&mut self) -> (Box<[f64]>, SampleStats) {
        let step_size = if self.tuning() {
            self.step_size_adapt.current_step_size()
        } else {
            self.step_size_adapt.current_step_size_adapted()
        };
        let step_size_bar = self.step_size_adapt.current_step_size_adapted();

        self.sampler.options.step_size = step_size;

        self.draw_count += 1;
        let (position, mut stats) = self.sampler.draw();

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
        (position, stats)
    }

    pub fn dim(&self) -> usize {
        self.sampler.dim()
    }
}

pub trait InitPointFunc {
    fn new_init_point<R: Rng + ?Sized>(&mut self, rng: &mut R, out: &mut [f64]);
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
        JoinHandle<Result<Vec<()>, ()>>,
        flume::Receiver<(Box<[f64]>, SampleStats)>,
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
        let results: Result<Vec<_>, ()> = points
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
                    sender.send(sampler.draw()).map_err(|_| ())?;
                }
                let result: Result<(), ()> = Ok(());
                result
            })
            .collect();
        results
    });

    Ok((handle, receiver))
}

pub mod test_logps {
    use crate::cpu_potential::CpuLogpFunc;

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

    impl CpuLogpFunc for NormalLogp {
        type Err = ();

        fn dim(&self) -> usize {
            self.dim
        }
        fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, ()> {
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
        cpu_sampler::{SamplerArgs, StaticSampler},
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
        let mut sampler = StaticSampler::new(func, SamplerArgs::default(), settings, 10, 42);

        sampler.set_position(&init).unwrap();
        let (sample1, stats1) = sampler.draw();

        let func = NormalLogp::new(dim, 3.);
        let mut sampler = StaticSampler::new(func, SamplerArgs::default(), settings, 10, 42);

        sampler.set_position(&init).unwrap();
        let (sample2, stats2) = sampler.draw();

        dbg!(&sample1);
        dbg!(stats1);

        dbg!(&sample2);
        dbg!(stats2);

        assert_eq!(sample1, sample2);
    }
}
