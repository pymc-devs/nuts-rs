use std::{fmt::Debug, iter, marker::PhantomData};

use itertools::izip;

use crate::{
    cpu_potential::{CpuLogpFunc, EuclideanPotential},
    mass_matrix::{
        DiagAdaptExpSettings, DiagMassMatrix, DrawGradCollector, ExpWeightedVariance, MassMatrix,
    },
    nuts::{
        AdaptStrategy, AsSampleStatVec, Collector, Hamiltonian, NutsOptions, SampleStatItem,
        SampleStatValue,
    },
    stepsize::{AcceptanceRateCollector, DualAverage, DualAverageOptions},
};

const LOWER_LIMIT: f64 = 1e-10f64;
const UPPER_LIMIT: f64 = 1e10f64;

pub(crate) struct DualAverageStrategy<F, M> {
    step_size_adapt: DualAverage,
    options: DualAverageSettings,
    num_tune: u64,
    num_early: u64,
    _phantom1: PhantomData<F>,
    _phantom2: PhantomData<M>,
}

#[derive(Debug, Clone, Copy)]
pub struct DualAverageStats {
    step_size_bar: f64,
    mean_tree_accept: f64,
    n_steps: u64,
}

impl AsSampleStatVec for DualAverageStats {
    fn add_to_vec(&self, vec: &mut Vec<SampleStatItem>) {
        vec.push(("step_size_bar", SampleStatValue::F64(self.step_size_bar)));
        vec.push((
            "mean_tree_accept",
            SampleStatValue::F64(self.mean_tree_accept),
        ));
        vec.push(("n_steps", SampleStatValue::U64(self.n_steps)));
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DualAverageSettings {
    pub early_target_accept: f64,
    pub target_accept: f64,
    pub final_window_ratio: f64,
    pub params: DualAverageOptions,
}

impl Default for DualAverageSettings {
    fn default() -> Self {
        Self {
            early_target_accept: 0.5,
            target_accept: 0.8,
            final_window_ratio: 0.4,
            params: DualAverageOptions::default(),
        }
    }
}

impl<F: CpuLogpFunc, M: MassMatrix> AdaptStrategy for DualAverageStrategy<F, M> {
    type Potential = EuclideanPotential<F, M>;
    type Collector = AcceptanceRateCollector<crate::cpu_state::State>;
    type Stats = DualAverageStats;
    type Options = DualAverageSettings;

    fn new(options: Self::Options, num_tune: u64, _dim: usize) -> Self {
        Self {
            num_tune,
            num_early: ((num_tune as f64) * options.final_window_ratio).ceil() as u64,
            options,
            step_size_adapt: DualAverage::new(options.params),
            _phantom1: PhantomData::default(),
            _phantom2: PhantomData::default(),
        }
    }

    fn init(
        &mut self,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        _state: &<Self::Potential as Hamiltonian>::State,
    ) {
        potential.step_size = self.options.params.initial_step;
    }

    fn adapt(
        &mut self,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: u64,
        collector: &Self::Collector,
    ) {
        let target = if draw >= self.num_early {
            self.options.target_accept
        } else {
            let start = self.options.early_target_accept;
            let end = self.options.target_accept;
            let time = (draw as f64) / (self.num_early as f64);
            start + (end - start) * (1f64 + (6f64 * (time - 0.6)).tanh()) / 2f64
        };
        if draw < self.num_tune {
            self.step_size_adapt
                .advance(collector.mean.current(), target);
            potential.step_size = self.step_size_adapt.current_step_size()
        } else {
            potential.step_size = self.step_size_adapt.current_step_size_adapted()
        }
    }

    fn new_collector(&self) -> Self::Collector {
        AcceptanceRateCollector::new()
    }

    fn current_stats(
        &self,
        _options: &NutsOptions,
        _potential: &Self::Potential,
        collector: &Self::Collector,
    ) -> Self::Stats {
        DualAverageStats {
            step_size_bar: self.step_size_adapt.current_step_size_adapted(),
            mean_tree_accept: collector.mean.current(),
            n_steps: collector.mean.count(),
        }
    }
}

pub(crate) struct ExpWindowDiagAdapt<F> {
    dim: usize,
    num_tune: u64,
    exp_variance_draw: ExpWeightedVariance,
    exp_variance_grad: ExpWeightedVariance,
    exp_variance_draw_bg: ExpWeightedVariance,
    exp_variance_grad_bg: ExpWeightedVariance,
    settings: DiagAdaptExpSettings,
    _phantom: PhantomData<F>,
}

#[derive(Clone, Debug)]
pub struct ExpWindowDiagAdaptStats {
    mass_matrix_inv: Option<Box<[f64]>>,
}

impl AsSampleStatVec for ExpWindowDiagAdaptStats {
    fn add_to_vec(&self, vec: &mut Vec<SampleStatItem>) {
        vec.push((
            "mass_matrix_inv",
            SampleStatValue::OptionArray(self.mass_matrix_inv.clone()),
        ));
    }
}

impl<F: CpuLogpFunc> AdaptStrategy for ExpWindowDiagAdapt<F> {
    type Potential = EuclideanPotential<F, DiagMassMatrix>;
    type Collector = DrawGradCollector;
    type Stats = ExpWindowDiagAdaptStats;
    type Options = DiagAdaptExpSettings;

    fn new(options: Self::Options, num_tune: u64, dim: usize) -> Self {
        Self {
            dim,
            num_tune: num_tune.saturating_sub(options.final_window),
            exp_variance_draw: ExpWeightedVariance::new(dim, options.early_variance_decay, true),
            exp_variance_grad: ExpWeightedVariance::new(dim, options.early_variance_decay, true),
            exp_variance_draw_bg: ExpWeightedVariance::new(dim, options.early_variance_decay, true),
            exp_variance_grad_bg: ExpWeightedVariance::new(dim, options.early_variance_decay, true),
            settings: options,
            _phantom: PhantomData::default(),
        }
    }

    fn init(
        &mut self,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &<Self::Potential as Hamiltonian>::State,
    ) {
        self.exp_variance_draw.set_variance(iter::repeat(1f64));
        self.exp_variance_draw.set_mean(state.q.iter().copied());
        self.exp_variance_grad
            .set_variance(state.grad.iter().map(|&val| {
                let diag = if !self.settings.grad_init {
                    1f64
                } else {
                    let out = val * val;
                    if out == 0f64 {
                        1f64
                    } else {
                        out
                    }
                };
                assert!(diag.is_finite());
                diag
            }));
        self.exp_variance_grad.set_mean(iter::repeat(0f64));

        potential.mass_matrix.update_diag(
            izip!(
                self.exp_variance_draw.current(),
                self.exp_variance_grad.current(),
            )
            .map(|(draw, grad)| {
                let val = (draw / grad).sqrt().clamp(LOWER_LIMIT, UPPER_LIMIT);
                assert!(val.is_finite());
                val
            }),
        );
    }

    fn adapt(
        &mut self,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: u64,
        collector: &Self::Collector,
    ) {
        if draw >= self.num_tune {
            return;
        }

        let count = self.exp_variance_draw_bg.count();

        let early_switch = (count == self.settings.early_window_switch_freq)
            & (draw < self.settings.window_switch_freq);

        if early_switch | ((draw % self.settings.window_switch_freq == 0) & (count > 5)) {
            self.exp_variance_draw = std::mem::replace(
                &mut self.exp_variance_draw_bg,
                ExpWeightedVariance::new(self.dim, self.settings.variance_decay, true),
            );
            self.exp_variance_grad = std::mem::replace(
                &mut self.exp_variance_grad_bg,
                ExpWeightedVariance::new(self.dim, self.settings.variance_decay, true),
            );

            self.exp_variance_draw_bg
                .set_mean(collector.draw.iter().copied());
            self.exp_variance_grad_bg
                .set_mean(collector.grad.iter().copied());
        } else if collector.is_good {
            self.exp_variance_draw
                .add_sample(collector.draw.iter().copied());
            self.exp_variance_grad
                .add_sample(collector.grad.iter().copied());
            self.exp_variance_draw_bg
                .add_sample(collector.draw.iter().copied());
            self.exp_variance_grad_bg
                .add_sample(collector.grad.iter().copied());
        }

        if self.exp_variance_draw.count() > 2 {
            assert!(self.exp_variance_draw.count() == self.exp_variance_grad.count());
            if (self.settings.grad_init) | (draw > self.settings.window_switch_freq) {
                potential.mass_matrix.update_diag(
                    izip!(
                        self.exp_variance_draw.current(),
                        self.exp_variance_grad.current(),
                    )
                    .map(|(draw, grad)| {
                        let mut val = (draw / grad).sqrt().clamp(LOWER_LIMIT, UPPER_LIMIT);
                        if !val.is_finite() {
                            val = 1f64;
                        }
                        val
                    }),
                );
            }
        }
    }

    fn new_collector(&self) -> Self::Collector {
        DrawGradCollector::new(self.dim)
    }

    fn current_stats(
        &self,
        _options: &NutsOptions,
        potential: &Self::Potential,
        _collector: &Self::Collector,
    ) -> Self::Stats {
        let diag = if self.settings.store_mass_matrix {
            Some(potential.mass_matrix.variance.clone())
        } else {
            None
        };
        ExpWindowDiagAdaptStats {
            mass_matrix_inv: diag,
        }
    }
}

pub(crate) struct CombinedStrategy<S1, S2> {
    data1: S1,
    data2: S2,
}

impl<S1, S2> CombinedStrategy<S1, S2> {
    pub(crate) fn new(s1: S1, s2: S2) -> Self {
        Self {
            data1: s1,
            data2: s2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CombinedStats<D1: Debug, D2: Debug> {
    stats1: D1,
    stats2: D2,
}

impl<D1: AsSampleStatVec, D2: AsSampleStatVec> AsSampleStatVec for CombinedStats<D1, D2> {
    fn add_to_vec(&self, vec: &mut Vec<SampleStatItem>) {
        self.stats1.add_to_vec(vec);
        self.stats2.add_to_vec(vec);
    }
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

    fn init(
        &mut self,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &<Self::Potential as Hamiltonian>::State,
    ) {
        self.data1.init(options, potential, state);
        self.data2.init(options, potential, state);
    }

    fn adapt(
        &mut self,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: u64,
        collector: &Self::Collector,
    ) {
        self.data1
            .adapt(options, potential, draw, &collector.collector1);
        self.data2
            .adapt(options, potential, draw, &collector.collector2);
    }

    fn new_collector(&self) -> Self::Collector {
        CombinedCollector {
            collector1: self.data1.new_collector(),
            collector2: self.data2.new_collector(),
        }
    }

    fn current_stats(
        &self,
        options: &NutsOptions,
        potential: &Self::Potential,
        collector: &Self::Collector,
    ) -> Self::Stats {
        CombinedStats {
            stats1: self
                .data1
                .current_stats(options, potential, &collector.collector1),
            stats2: self
                .data2
                .current_stats(options, potential, &collector.collector2),
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
        pub(crate) fn new(dim: usize, mu: f64) -> NormalLogp {
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
    use crate::nuts::{AdaptStrategy, Chain, NutsChain, NutsOptions};

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

        let mass_matrix = DiagMassMatrix::new(ndim);
        let max_energy_error = 1000f64;
        let step_size = 0.1f64;

        let potential = EuclideanPotential::new(func, mass_matrix, max_energy_error, step_size);
        let options = NutsOptions {
            maxdepth: 10u64,
            store_gradient: true,
        };

        let rng = {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(42)
        };
        let chain = 0u64;

        let mut sampler = NutsChain::new(potential, strategy, options, rng, chain);
        sampler.set_position(&vec![1.5f64; ndim]).unwrap();
        for _ in 0..200 {
            sampler.draw().unwrap();
        }
    }
}
