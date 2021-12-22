use std::marker::PhantomData;

use itertools::izip;

use crate::{
    cpu_potential::{CpuLogpFunc, EuclideanPotential},
    mass_matrix::{
        DiagAdaptExpSettings, DiagMassMatrix, DrawGradCollector, ExpWeightedVariance, MassMatrix,
    },
    nuts::{AdaptStrategy, Collector},
    stepsize::{AcceptanceRateCollector, DualAverage, DualAverageSettings},
};

pub(crate) struct DualAverageStrategy<F, M> {
    step_size_adapt: DualAverage,
    num_tune: u64,
    _phantom1: PhantomData<F>,
    _phantom2: PhantomData<M>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DualAverageStats {
    step_size_bar: f64,
    mean_tree_accept: f64,
}

impl<F: CpuLogpFunc, M: MassMatrix> AdaptStrategy for DualAverageStrategy<F, M> {
    type Potential = EuclideanPotential<F, M>;
    type Collector = AcceptanceRateCollector<crate::cpu_state::State>;
    type Stats = DualAverageStats;
    type Options = DualAverageSettings;

    fn new(options: Self::Options, num_tune: u64, _dim: usize) -> Self {
        Self {
            num_tune,
            step_size_adapt: DualAverage::new(options),
            _phantom1: PhantomData::default(),
            _phantom2: PhantomData::default(),
        }
    }

    fn adapt(&mut self, potential: &mut Self::Potential, draw: u64, collector: &Self::Collector) {
        self.step_size_adapt.advance(collector.mean.current());
        if draw < self.num_tune {
            potential.step_size = self.step_size_adapt.current_step_size()
        } else {
            potential.step_size = self.step_size_adapt.current_step_size_adapted()
        }
    }

    fn new_collector(&self) -> Self::Collector {
        AcceptanceRateCollector::new()
    }

    fn current_stats(&self, collector: &Self::Collector) -> Self::Stats {
        DualAverageStats {
            step_size_bar: self.step_size_adapt.current_step_size_adapted(),
            mean_tree_accept: collector.mean.current(),
        }
    }
}

pub(crate) struct ExpWindowDiagAdapt<F> {
    dim: usize,
    exp_variance_draw: ExpWeightedVariance,
    exp_variance_grad: ExpWeightedVariance,
    settings: DiagAdaptExpSettings,
    _phantom: PhantomData<F>,
}

impl<F: CpuLogpFunc> AdaptStrategy for ExpWindowDiagAdapt<F> {
    type Potential = EuclideanPotential<F, DiagMassMatrix>;
    type Collector = DrawGradCollector;
    type Stats = ();
    type Options = DiagAdaptExpSettings;

    fn new(options: Self::Options, num_tune: u64, dim: usize) -> Self {
        Self {
            dim,
            exp_variance_draw: ExpWeightedVariance::new(dim, options.variance_decay, true),
            exp_variance_grad: ExpWeightedVariance::new(dim, options.variance_decay, true),
            settings: options,
            _phantom: PhantomData::default(),
        }
    }

    fn adapt(&mut self, potential: &mut Self::Potential, draw: u64, collector: &Self::Collector) {
        if draw < self.settings.discard_window {
            return;
        }

        if draw > self.settings.stop_at_draw {
            return;
        }

        if draw == self.settings.discard_window {
            self.exp_variance_draw
                .set_mean(collector.draw.iter().copied());
        }

        self.exp_variance_draw
            .add_sample(collector.draw.iter().copied());
        self.exp_variance_grad
            .add_sample(collector.grad.iter().copied());

        if draw > 2 * self.settings.discard_window {
            potential.mass_matrix.update_diag(
                izip!(
                    self.exp_variance_draw.current(),
                    self.exp_variance_grad.current(),
                )
                .map(|(&draw, &grad)| (draw / grad).sqrt().clamp(1e-12, 1e10)),
            );
        }
    }

    fn new_collector(&self) -> Self::Collector {
        DrawGradCollector::new(self.dim)
    }

    fn current_stats(&self, _collector: &Self::Collector) -> Self::Stats {
        ()
    }
}

pub(crate) struct CombinedStrategy<S1, S2> {
    data1: S1,
    data2: S2,
}

impl<S1, S2> CombinedStrategy<S1, S2> {
    fn new(s1: S1, s2: S2) -> Self {
        Self {
            data1: s1,
            data2: s2,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct CombinedStats<D1: Copy, D2: Copy> {
    stats1: D1,
    stats2: D2,
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct CombinedOptions<O1: Copy + Send + Default, O2: Copy + Send + Default> {
    options1: O1,
    options2: O2,
}

impl<S1, S2> AdaptStrategy for CombinedStrategy<S1, S2>
where
    S1: AdaptStrategy,
    S2: AdaptStrategy<Potential = S1::Potential>,
{
    type Potential = S1::Potential;
    type Collector = CombinedCollector<S1::Collector, S2::Collector>;
    type Stats = CombinedStats<S1::Stats, S2::Stats>;
    type Options = CombinedOptions<S1::Options, S2::Options>;

    fn new(options: Self::Options, num_tune: u64, dim: usize) -> Self {
        Self {
            data1: S1::new(options.options1, num_tune, dim),
            data2: S2::new(options.options2, num_tune, dim),
        }
    }

    fn adapt(&mut self, potential: &mut Self::Potential, draw: u64, collector: &Self::Collector) {
        self.data1.adapt(potential, draw, &collector.collector1);
        self.data2.adapt(potential, draw, &collector.collector2);
    }

    fn new_collector(&self) -> Self::Collector {
        CombinedCollector {
            collector1: self.data1.new_collector(),
            collector2: self.data2.new_collector(),
        }
    }

    fn current_stats(&self, collector: &Self::Collector) -> Self::Stats {
        CombinedStats {
            stats1: self.data1.current_stats(&collector.collector1),
            stats2: self.data2.current_stats(&collector.collector2),
        }
    }
}

pub(crate) struct CombinedCollector<C1: Collector, C2: Collector> {
    collector1: C1,
    collector2: C2,
}

impl<C1, C2> Collector for CombinedCollector<C1, C2>
where
    C1: Collector,
    C2: Collector<State = C1::State>,
{
    type State = C1::State;

    fn register_leapfrog(
        &mut self,
        start: &Self::State,
        end: &Self::State,
        divergence_info: Option<&dyn crate::nuts::DivergenceInfo>,
    ) {
        self.collector1
            .register_leapfrog(start, end, divergence_info);
        self.collector2
            .register_leapfrog(start, end, divergence_info);
    }

    fn register_draw(&mut self, state: &Self::State, info: &crate::nuts::SampleInfo) {
        self.collector1.register_draw(state, info);
        self.collector2.register_draw(state, info);
    }

    fn register_init(&mut self, state: &Self::State, options: &crate::nuts::NutsOptions) {
        self.collector1.register_init(state, options);
        self.collector2.register_init(state, options);
    }
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

#[cfg(test)]
mod test {
    use super::test_logps::NormalLogp;
    use super::*;
    use crate::nuts::{AdaptStrategy, NutsOptions, NutsSampler, Sampler};

    #[test]
    fn instanciate_adaptive_sampler() {
        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.);
        let num_tune = 100;
        let step_size_adapt =
            DualAverageStrategy::new(DualAverageSettings::default(), num_tune, func.dim());
        let mass_matrix_adapt =
            ExpWindowDiagAdapt::new(DiagAdaptExpSettings::default(), num_tune, func.dim());
        let strategy = CombinedStrategy::new(step_size_adapt, mass_matrix_adapt);

        let mass_matrix = DiagMassMatrix::new(vec![1f64; func.dim()].into());
        let max_energy_error = 1000f64;
        let step_size = 0.1f64;

        let potential = EuclideanPotential::new(func, mass_matrix, max_energy_error, step_size);
        let options = NutsOptions { maxdepth: 10u64 };

        let rng = {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(42)
        };
        let chain = 0u64;

        let mut sampler = NutsSampler::new(potential, strategy, options, rng, chain);
        sampler.set_position(&vec![1.5f64; ndim]).unwrap();
        sampler.draw().unwrap();
    }
}
