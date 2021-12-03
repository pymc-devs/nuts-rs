use crate::{
    cpu_potential::{Potential, UnitMassMatrix},
    cpu_state::{State, StatePool},
    nuts::{draw, Collector, DivergenceInfo, NutsOptions, SampleInfo},
    stepsize::{DualAverage, DualAverageSettings},
};

pub use crate::cpu_potential::CpuLogpFunc;

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
        self.mean.add(1.);
        self.mean.reset();
    }
}

struct StatsCollector {
    acceptance_rate: AcceptanceRateCollector,
}

#[derive(Debug)]
struct CollectedStats {
    pub mean_acceptance_rate: f64,
}

impl StatsCollector {
    fn new() -> StatsCollector {
        StatsCollector {
            acceptance_rate: AcceptanceRateCollector::new(),
        }
    }

    fn stats(&self) -> CollectedStats {
        CollectedStats {
            mean_acceptance_rate: self.acceptance_rate.mean.current(),
        }
    }
}

impl Collector for StatsCollector {
    type State = State;

    fn register_leapfrog(
        &mut self,
        start: &Self::State,
        end: &Self::State,
        divergence_info: Option<&dyn crate::nuts::DivergenceInfo>,
    ) {
        self.acceptance_rate
            .register_leapfrog(start, end, divergence_info);
    }

    fn register_draw(&mut self, state: &Self::State, info: &SampleInfo) {
        self.acceptance_rate.register_draw(state, info);
    }

    fn register_init(&mut self, state: &Self::State, options: &NutsOptions) {
        self.acceptance_rate.register_init(state, options);
    }
}

#[derive(Debug)]
pub struct SampleStats {
    pub step_size: f64,
    pub step_size_bar: f64,
    pub depth: u64,
    pub idx_in_trajectory: i64,
    pub logp: f64,
    pub mean_acceptance_rate: f64,
    pub divergence_info: Option<Box<dyn DivergenceInfo>>,
}

pub struct UnitStaticSampler<F: CpuLogpFunc> {
    potential: Potential<F, UnitMassMatrix>,
    state: State,
    pool: StatePool,
    rng: rand::rngs::StdRng,
    collector: StatsCollector,
    options: NutsOptions,
}

struct NullCollector {}

impl Collector for NullCollector {
    type State = State;
}

impl<F: CpuLogpFunc> UnitStaticSampler<F> {
    pub fn new(logp: F, seed: u64, maxdepth: u64, step_size: f64) -> UnitStaticSampler<F> {
        use rand::SeedableRng;

        let mass_matrix = UnitMassMatrix {};
        let mut pool = StatePool::new(logp.dim());
        let potential = Potential::new(logp, mass_matrix);
        let state = pool.new_state();
        let collector = StatsCollector::new();
        let options = NutsOptions {
            step_size,
            maxdepth,
            max_energy_error: 1000.,
        };
        UnitStaticSampler {
            potential,
            state,
            pool,
            options,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            collector,
        }
    }

    pub fn set_position(&mut self, position: &[f64]) -> Result<(), F::Err> {
        use crate::nuts::Potential;
        {
            let inner = self.state.try_mut_inner().expect("State already in use");
            inner.q.copy_from_slice(position);
        }
        if let Err(err) = self.potential.update_potential_gradient(&mut self.state) {
            return Err(err.logp_function_error.unwrap());
        }
        // TODO check init of p_sum
        Ok(())
    }

    pub fn draw(&mut self) -> (Box<[f64]>, SampleStats) {
        use crate::nuts::Potential;
        self.potential
            .randomize_momentum(&mut self.state, &mut self.rng);
        self.potential.update_velocity(&mut self.state);
        self.potential.update_kinetic_energy(&mut self.state);

        let (state, info) = draw(
            &mut self.pool,
            self.state.clone(),
            &mut self.rng,
            &mut self.potential,
            &self.options,
            &mut self.collector,
        );
        self.state = state;
        let position: Box<[f64]> = self.state.q.clone().into();
        let stats = SampleStats {
            step_size: self.options.step_size,
            step_size_bar: self.options.step_size,
            divergence_info: info.divergence_info,
            idx_in_trajectory: self.state.idx_in_trajectory,
            depth: info.depth,
            logp: -self.state.potential_energy,
            mean_acceptance_rate: self.collector.stats().mean_acceptance_rate,
        };
        (position, stats)
    }
}

pub struct AdaptiveSampler<F: CpuLogpFunc> {
    num_tune: u64,
    sampler: UnitStaticSampler<F>,
    step_size_adapt: DualAverage,
    draw_count: u64,
}

impl<F: CpuLogpFunc> AdaptiveSampler<F> {
    pub fn new(logp: F, num_tune: u64, initial_step: f64, seed: u64) -> Self {
        let maxdepth = 10;
        let sampler = UnitStaticSampler::new(logp, seed, maxdepth, initial_step);
        let settings = DualAverageSettings::default();
        let step_size_adapt = DualAverage::new(settings, initial_step);

        Self {
            num_tune,
            sampler,
            step_size_adapt,
            draw_count: 0,
        }
    }

    pub fn set_position(&mut self, position: &[f64]) -> Result<(), F::Err> {
        self.sampler.set_position(position)
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
            self.step_size_adapt.advance(stats.mean_acceptance_rate)
        }
        (position, stats)
    }
}

pub mod test_logps {
    use crate::cpu_potential::CpuLogpFunc;

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
                logp -= val * val;
                *g = -val;
            }
            Ok(logp)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu_sampler::UnitStaticSampler;

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

        let mut sampler = UnitStaticSampler::new(func, 42, 10, 1e-2);

        sampler.set_position(&init).unwrap();
        let (sample1, stats1) = sampler.draw();

        let func = NormalLogp::new(dim, 3.);
        let mut sampler = UnitStaticSampler::new(func, 42, 10, 1e-2);

        sampler.set_position(&init).unwrap();
        let (sample2, stats2) = sampler.draw();

        dbg!(&sample1);
        dbg!(stats1);

        dbg!(&sample2);
        dbg!(stats2);

        assert_eq!(sample1, sample2);
    }
}
