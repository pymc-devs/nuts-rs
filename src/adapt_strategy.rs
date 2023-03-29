use std::{fmt::Debug, marker::PhantomData};

#[cfg(feature = "arrow")]
use arrow2::{
    array::{MutableArray, MutableFixedSizeListArray, MutablePrimitiveArray, StructArray, TryPush},
    datatypes::{DataType, Field},
};
use itertools::izip;

use crate::{
    cpu_potential::{CpuLogpFunc, EuclideanPotential},
    mass_matrix::{DiagMassMatrix, DrawGradCollector, MassMatrix, RunningVariance},
    nuts::{AdaptStrategy, Collector, Hamiltonian, NutsOptions},
    stepsize::{AcceptanceRateCollector, DualAverage, DualAverageOptions},
    DivergenceInfo,
};

#[cfg(feature = "arrow")]
use crate::nuts::{ArrowBuilder, ArrowRow};
#[cfg(feature = "arrow")]
use crate::SamplerArgs;

const LOWER_LIMIT: f64 = 1e-10f64;
const UPPER_LIMIT: f64 = 1e10f64;

pub(crate) struct DualAverageStrategy<F, M> {
    step_size_adapt: DualAverage,
    options: DualAverageSettings,
    enabled: bool,
    finalized: bool,
    _phantom1: PhantomData<F>,
    _phantom2: PhantomData<M>,
}

impl<F, M> DualAverageStrategy<F, M> {
    fn enable(&mut self) {
        self.enabled = true;
    }

    fn finalize(&mut self) {
        self.finalized = true;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DualAverageStats {
    pub step_size_bar: f64,
    pub mean_tree_accept: f64,
    pub n_steps: u64,
}

#[cfg(feature = "arrow")]
pub struct DualAverageStatsBuilder {
    step_size_bar: MutablePrimitiveArray<f64>,
    mean_tree_accept: MutablePrimitiveArray<f64>,
    n_steps: MutablePrimitiveArray<u64>,
}

#[cfg(feature = "arrow")]
impl ArrowBuilder<DualAverageStats> for DualAverageStatsBuilder {
    fn append_value(&mut self, value: &DualAverageStats) {
        self.step_size_bar.push(Some(value.step_size_bar));
        self.mean_tree_accept.push(Some(value.mean_tree_accept));
        self.n_steps.push(Some(value.n_steps));
    }

    fn finalize(mut self) -> StructArray {
        let fields = vec![
            Field::new("step_size_bar", DataType::Float64, false),
            Field::new("mean_tree_accept", DataType::Float64, false),
            Field::new("n_steps", DataType::UInt64, false),
        ];

        let arrays = vec![
            self.step_size_bar.as_box(),
            self.mean_tree_accept.as_box(),
            self.n_steps.as_box(),
        ];

        StructArray::new(DataType::Struct(fields), arrays, None)
    }
}

#[cfg(feature = "arrow")]
impl ArrowRow for DualAverageStats {
    type Builder = DualAverageStatsBuilder;

    fn new_builder(_dim: usize, _settings: &SamplerArgs) -> Self::Builder {
        Self::Builder {
            step_size_bar: MutablePrimitiveArray::new(),
            mean_tree_accept: MutablePrimitiveArray::new(),
            n_steps: MutablePrimitiveArray::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DualAverageSettings {
    pub target_accept: f64,
    pub initial_step: f64,
    pub params: DualAverageOptions,
}

impl Default for DualAverageSettings {
    fn default() -> Self {
        Self {
            target_accept: 0.8,
            initial_step: 0.1,
            params: DualAverageOptions::default(),
        }
    }
}

impl<F: CpuLogpFunc, M: MassMatrix> AdaptStrategy for DualAverageStrategy<F, M> {
    type Potential = EuclideanPotential<F, M>;
    type Collector = AcceptanceRateCollector<crate::cpu_state::State>;
    type Stats = DualAverageStats;
    type Options = DualAverageSettings;

    fn new(options: Self::Options, _num_tune: u64, _dim: usize) -> Self {
        Self {
            options,
            enabled: true,
            step_size_adapt: DualAverage::new(options.params, options.initial_step),
            finalized: false,
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
        potential.step_size = self.options.initial_step;
    }

    fn adapt(
        &mut self,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        _draw: u64,
        collector: &Self::Collector,
    ) {
        if self.finalized {
            self.step_size_adapt
                .advance(collector.mean.current(), self.options.target_accept);
            potential.step_size = self.step_size_adapt.current_step_size_adapted();
            return;
        }
        if !self.enabled {
            return;
        }
        self.step_size_adapt
            .advance(collector.mean.current(), self.options.target_accept);
        potential.step_size = self.step_size_adapt.current_step_size()
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

/// Settings for mass matrix adaptation
#[derive(Clone, Copy, Debug)]
pub struct DiagAdaptExpSettings {
    pub store_mass_matrix: bool,
}

impl Default for DiagAdaptExpSettings {
    fn default() -> Self {
        Self {
            store_mass_matrix: false,
        }
    }
}

pub(crate) struct ExpWindowDiagAdapt<F> {
    dim: usize,
    exp_variance_draw: RunningVariance,
    exp_variance_grad: RunningVariance,
    exp_variance_grad_bg: RunningVariance,
    exp_variance_draw_bg: RunningVariance,
    settings: DiagAdaptExpSettings,
    _phantom: PhantomData<F>,
}

impl<F: CpuLogpFunc> ExpWindowDiagAdapt<F> {
    fn update_estimators(&mut self, collector: &DrawGradCollector) {
        if collector.is_good {
            self.exp_variance_draw
                .add_sample(collector.draw.iter().copied());
            self.exp_variance_grad
                .add_sample(collector.grad.iter().copied());
            self.exp_variance_draw_bg
                .add_sample(collector.draw.iter().copied());
            self.exp_variance_grad_bg
                .add_sample(collector.grad.iter().copied());
        }
    }

    fn switch(&mut self, collector: &DrawGradCollector) {
        self.exp_variance_draw = std::mem::replace(
            &mut self.exp_variance_draw_bg,
            RunningVariance::new(self.dim),
        );
        self.exp_variance_grad = std::mem::replace(
            &mut self.exp_variance_grad_bg,
            RunningVariance::new(self.dim),
        );

        self.update_estimators(collector);
    }

    fn current_count(&self) -> u64 {
        assert!(self.exp_variance_draw.count() == self.exp_variance_grad.count());
        self.exp_variance_draw.count()
    }

    fn background_count(&self) -> u64 {
        assert!(self.exp_variance_draw_bg.count() == self.exp_variance_grad_bg.count());
        self.exp_variance_draw_bg.count()
    }

    fn update_potential(&self, potential: &mut EuclideanPotential<F, DiagMassMatrix>) {
        if self.current_count() < 3 {
            return;
        }
        assert!(self.current_count() > 2);
        potential.mass_matrix.update_diag(
            izip!(
                self.exp_variance_draw.current(),
                self.exp_variance_grad.current(),
            )
            .map(|(draw, grad)| {
                let val = (draw / grad).sqrt().clamp(LOWER_LIMIT, UPPER_LIMIT);
                if !val.is_finite() {
                    None
                } else {
                    Some(val)
                }
            }),
        );
    }
}

#[derive(Clone, Debug)]
pub struct ExpWindowDiagAdaptStats {
    pub mass_matrix_inv: Option<Box<[f64]>>,
}

#[cfg(feature = "arrow")]
pub struct ExpWindowDiagAdaptStatsBuilder {
    mass_matrix_inv: MutableFixedSizeListArray<MutablePrimitiveArray<f64>>,
}

#[cfg(feature = "arrow")]
impl ArrowBuilder<ExpWindowDiagAdaptStats> for ExpWindowDiagAdaptStatsBuilder {
    fn append_value(&mut self, value: &ExpWindowDiagAdaptStats) {
        self.mass_matrix_inv
            .try_push(
                value
                    .mass_matrix_inv
                    .as_ref()
                    .map(|vals| vals.iter().map(|&x| Some(x))),
            )
            .unwrap();
    }

    fn finalize(mut self) -> StructArray {
        let fields = vec![Field::new(
            "mass_matrix_inv",
            self.mass_matrix_inv.data_type().clone(),
            true,
        )];

        let arrays = vec![self.mass_matrix_inv.as_box()];

        StructArray::new(DataType::Struct(fields), arrays, None)
    }
}

#[cfg(feature = "arrow")]
impl ArrowRow for ExpWindowDiagAdaptStats {
    type Builder = ExpWindowDiagAdaptStatsBuilder;

    fn new_builder(dim: usize, _settings: &SamplerArgs) -> Self::Builder {
        let items = MutablePrimitiveArray::new();
        // TODO Add only based on settings
        let values = MutableFixedSizeListArray::new_with_field(items, "item", false, dim);
        Self::Builder {
            mass_matrix_inv: values,
        }
    }
}

impl<F: CpuLogpFunc> AdaptStrategy for ExpWindowDiagAdapt<F> {
    type Potential = EuclideanPotential<F, DiagMassMatrix>;
    type Collector = DrawGradCollector;
    type Stats = ExpWindowDiagAdaptStats;
    type Options = DiagAdaptExpSettings;

    fn new(options: Self::Options, _num_tune: u64, dim: usize) -> Self {
        Self {
            dim,
            exp_variance_draw: RunningVariance::new(dim),
            exp_variance_grad: RunningVariance::new(dim),
            exp_variance_draw_bg: RunningVariance::new(dim),
            exp_variance_grad_bg: RunningVariance::new(dim),
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
        self.exp_variance_draw.add_sample(state.q.iter().copied());
        self.exp_variance_draw_bg
            .add_sample(state.q.iter().copied());
        self.exp_variance_grad
            .add_sample(state.grad.iter().copied());
        self.exp_variance_grad_bg
            .add_sample(state.grad.iter().copied());

        potential.mass_matrix.update_diag(
            state
                .grad
                .iter()
                .map(|&grad| Some((grad).abs().recip().clamp(LOWER_LIMIT, UPPER_LIMIT))),
        );
    }

    fn adapt(
        &mut self,
        _options: &mut NutsOptions,
        _potential: &mut Self::Potential,
        _draw: u64,
        _collector: &Self::Collector,
    ) {
        // Must be controlled from a different meta strategy
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

pub(crate) struct GradDiagStrategy<F: CpuLogpFunc> {
    step_size: DualAverageStrategy<F, DiagMassMatrix>,
    mass_matrix: ExpWindowDiagAdapt<F>,
    options: GradDiagOptions,
    num_tune: u64,
    // The number of draws in the the early window
    early_end: u64,

    // The first draw number for the final step size adaptation window
    final_step_size_window: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct GradDiagOptions {
    pub dual_average_options: DualAverageSettings,
    pub mass_matrix_options: DiagAdaptExpSettings,
    pub early_window: f64,
    pub step_size_window: f64,
    pub mass_matrix_switch_freq: u64,
    pub early_mass_matrix_switch_freq: u64,
}

impl Default for GradDiagOptions {
    fn default() -> Self {
        Self {
            dual_average_options: DualAverageSettings::default(),
            mass_matrix_options: DiagAdaptExpSettings::default(),
            early_window: 0.3,
            step_size_window: 0.2,
            mass_matrix_switch_freq: 60,
            early_mass_matrix_switch_freq: 10,
        }
    }
}

impl<F: CpuLogpFunc> AdaptStrategy for GradDiagStrategy<F> {
    type Potential = EuclideanPotential<F, DiagMassMatrix>;
    type Collector = CombinedCollector<
        AcceptanceRateCollector<<EuclideanPotential<F, DiagMassMatrix> as Hamiltonian>::State>,
        DrawGradCollector,
    >;
    type Stats = CombinedStats<DualAverageStats, ExpWindowDiagAdaptStats>;
    type Options = GradDiagOptions;

    fn new(options: Self::Options, num_tune: u64, dim: usize) -> Self {
        let num_tune_f = num_tune as f64;
        let step_size_window = (options.step_size_window * num_tune_f) as u64;
        let early_end = (options.early_window * num_tune_f) as u64;
        let final_second_step_size = num_tune.saturating_sub(step_size_window);

        assert!(early_end < num_tune);

        Self {
            step_size: DualAverageStrategy::new(options.dual_average_options, num_tune, dim),
            mass_matrix: ExpWindowDiagAdapt::new(options.mass_matrix_options, num_tune, dim),
            options,
            num_tune,
            early_end,
            final_step_size_window: final_second_step_size,
        }
    }

    fn init(
        &mut self,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &<Self::Potential as Hamiltonian>::State,
    ) {
        self.step_size.init(options, potential, state);
        self.mass_matrix.init(options, potential, state);
        self.step_size.enable();
    }

    fn adapt(
        &mut self,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: u64,
        collector: &Self::Collector,
    ) {
        if draw >= self.num_tune {
            return;
        }

        if draw < self.final_step_size_window {
            let is_early = draw < self.early_end;
            let switch_freq = if is_early {
                self.options.early_mass_matrix_switch_freq
            } else {
                self.options.mass_matrix_switch_freq
            };

            if self.mass_matrix.background_count() >= switch_freq {
                self.mass_matrix.switch(&collector.collector2);
            } else {
                self.mass_matrix.update_estimators(&collector.collector2);
            }
            self.mass_matrix.update_potential(potential);
            self.step_size
                .adapt(options, potential, draw, &collector.collector1);
            return;
        }

        if draw == self.num_tune - 1 {
            self.step_size.finalize();
        }
        self.step_size
            .adapt(options, potential, draw, &collector.collector1);
    }

    fn new_collector(&self) -> Self::Collector {
        CombinedCollector {
            collector1: self.step_size.new_collector(),
            collector2: self.mass_matrix.new_collector(),
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
                .step_size
                .current_stats(options, potential, &collector.collector1),
            stats2: self
                .mass_matrix
                .current_stats(options, potential, &collector.collector2),
        }
    }
}

#[cfg(feature = "arrow")]
#[derive(Debug, Clone)]
pub struct CombinedStats<D1: Debug + ArrowRow, D2: Debug + ArrowRow> {
    pub stats1: D1,
    pub stats2: D2,
}

#[cfg(not(feature = "arrow"))]
#[derive(Debug, Clone)]
pub struct CombinedStats<D1: Debug, D2: Debug> {
    pub stats1: D1,
    pub stats2: D2,
}

#[cfg(feature = "arrow")]
pub struct CombinedStatsBuilder<D1: ArrowRow, D2: ArrowRow> {
    stats1: D1::Builder,
    stats2: D2::Builder,
}

#[cfg(feature = "arrow")]
impl<D1: Debug + ArrowRow, D2: Debug + ArrowRow> ArrowRow for CombinedStats<D1, D2> {
    type Builder = CombinedStatsBuilder<D1, D2>;

    fn new_builder(dim: usize, settings: &SamplerArgs) -> Self::Builder {
        Self::Builder {
            stats1: D1::new_builder(dim, settings),
            stats2: D2::new_builder(dim, settings),
        }
    }
}

#[cfg(feature = "arrow")]
impl<D1: Debug + ArrowRow, D2: Debug + ArrowRow> ArrowBuilder<CombinedStats<D1, D2>>
    for CombinedStatsBuilder<D1, D2>
{
    fn append_value(&mut self, value: &CombinedStats<D1, D2>) {
        self.stats1.append_value(&value.stats1);
        self.stats2.append_value(&value.stats2);
    }

    fn finalize(self) -> StructArray {
        let mut data1 = self.stats1.finalize().into_data();
        let data2 = self.stats2.finalize().into_data();

        assert!(data1.2.is_none());
        assert!(data2.2.is_none());

        data1.0.extend(data2.0);
        data1.1.extend(data2.1);

        StructArray::new(DataType::Struct(data1.0), data1.1, None)
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
        divergence_info: Option<&DivergenceInfo>,
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
        let options = GradDiagOptions::default();
        let strategy = GradDiagStrategy::new(options, num_tune, ndim);

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
