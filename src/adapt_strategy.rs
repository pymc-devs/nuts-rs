use std::{fmt::Debug, marker::PhantomData};

#[cfg(feature = "arrow")]
use arrow2::{
    array::{MutableArray, MutableFixedSizeListArray, MutablePrimitiveArray, StructArray, TryPush},
    datatypes::{DataType, Field},
};

use crate::{
    mass_matrix::{DiagMassMatrix, DrawGradCollector, MassMatrix, RunningVariance},
    math_base::Math,
    nuts::{AdaptStrategy, Collector, NutsOptions},
    potential::EuclideanPotential,
    state::State,
    stepsize::{AcceptanceRateCollector, DualAverage, DualAverageOptions},
    DivergenceInfo,
};

#[cfg(feature = "arrow")]
use crate::nuts::{ArrowBuilder, ArrowRow};
#[cfg(feature = "arrow")]
use crate::SamplerArgs;

const LOWER_LIMIT: f64 = 1e-20f64;
const UPPER_LIMIT: f64 = 1e20f64;

const INIT_LOWER_LIMIT: f64 = 1e-20f64;
const INIT_UPPER_LIMIT: f64 = 1e20f64;

pub(crate) struct DualAverageStrategy<F, M> {
    step_size_adapt: DualAverage,
    options: DualAverageSettings,
    enabled: bool,
    use_mean_sym: bool,
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

    fn use_mean_sym(&mut self) {
        self.use_mean_sym = true;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DualAverageStats {
    pub step_size_bar: f64,
    pub mean_tree_accept: f64,
    pub mean_tree_accept_sym: f64,
    pub n_steps: u64,
}

#[cfg(feature = "arrow")]
pub struct DualAverageStatsBuilder {
    step_size_bar: MutablePrimitiveArray<f64>,
    mean_tree_accept: MutablePrimitiveArray<f64>,
    mean_tree_accept_sym: MutablePrimitiveArray<f64>,
    n_steps: MutablePrimitiveArray<u64>,
}

#[cfg(feature = "arrow")]
impl ArrowBuilder<DualAverageStats> for DualAverageStatsBuilder {
    fn append_value(&mut self, value: &DualAverageStats) {
        self.step_size_bar.push(Some(value.step_size_bar));
        self.mean_tree_accept.push(Some(value.mean_tree_accept));
        self.mean_tree_accept_sym
            .push(Some(value.mean_tree_accept_sym));
        self.n_steps.push(Some(value.n_steps));
    }

    fn finalize(mut self) -> Option<StructArray> {
        let fields = vec![
            Field::new("step_size_bar", DataType::Float64, false),
            Field::new("mean_tree_accept", DataType::Float64, false),
            Field::new("mean_tree_accept_sym", DataType::Float64, false),
            Field::new("n_steps", DataType::UInt64, false),
        ];

        let arrays = vec![
            self.step_size_bar.as_box(),
            self.mean_tree_accept.as_box(),
            self.mean_tree_accept_sym.as_box(),
            self.n_steps.as_box(),
        ];

        Some(StructArray::new(DataType::Struct(fields), arrays, None))
    }
}

#[cfg(feature = "arrow")]
impl ArrowRow for DualAverageStats {
    type Builder = DualAverageStatsBuilder;

    fn new_builder(_dim: usize, _settings: &SamplerArgs) -> Self::Builder {
        Self::Builder {
            step_size_bar: MutablePrimitiveArray::new(),
            mean_tree_accept: MutablePrimitiveArray::new(),
            mean_tree_accept_sym: MutablePrimitiveArray::new(),
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

impl<M: Math, Mass: MassMatrix<M>> AdaptStrategy<M> for DualAverageStrategy<M, Mass> {
    type Potential = EuclideanPotential<M, Mass>;
    type Collector = AcceptanceRateCollector<M>;
    type Stats = DualAverageStats;
    type Options = DualAverageSettings;

    fn new(_math: &mut M, options: Self::Options, _num_tune: u64) -> Self {
        Self {
            options,
            enabled: true,
            step_size_adapt: DualAverage::new(options.params, options.initial_step),
            finalized: false,
            use_mean_sym: false,
            _phantom1: PhantomData::default(),
            _phantom2: PhantomData::default(),
        }
    }

    fn init(
        &mut self,
        _math: &mut M,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        _state: &State<M>,
    ) {
        potential.step_size = self.options.initial_step;
    }

    fn adapt(
        &mut self,
        _math: &mut M,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        _draw: u64,
        collector: &Self::Collector,
    ) {
        let current = if self.use_mean_sym {
            collector.mean_sym.current()
        } else {
            collector.mean.current()
        };
        if self.finalized {
            self.step_size_adapt
                .advance(current, self.options.target_accept);
            potential.step_size = self.step_size_adapt.current_step_size_adapted();
            return;
        }
        if !self.enabled {
            return;
        }
        self.step_size_adapt
            .advance(current, self.options.target_accept);
        potential.step_size = self.step_size_adapt.current_step_size()
    }

    fn new_collector(&self, _math: &mut M) -> Self::Collector {
        AcceptanceRateCollector::new()
    }

    fn current_stats(
        &self,
        _math: &mut M,
        _options: &NutsOptions,
        _potential: &Self::Potential,
        collector: &Self::Collector,
    ) -> Self::Stats {
        DualAverageStats {
            step_size_bar: self.step_size_adapt.current_step_size_adapted(),
            mean_tree_accept: collector.mean.current(),
            mean_tree_accept_sym: collector.mean_sym.current(),
            n_steps: collector.mean.count(),
        }
    }
}

/// Settings for mass matrix adaptation
#[derive(Clone, Copy, Debug, Default)]
pub struct DiagAdaptExpSettings {
    pub store_mass_matrix: bool,
}

pub(crate) struct ExpWindowDiagAdapt<M: Math> {
    exp_variance_draw: RunningVariance<M>,
    exp_variance_grad: RunningVariance<M>,
    exp_variance_grad_bg: RunningVariance<M>,
    exp_variance_draw_bg: RunningVariance<M>,
    settings: DiagAdaptExpSettings,
    _phantom: PhantomData<M>,
}

impl<M: Math> ExpWindowDiagAdapt<M> {
    fn update_estimators(&mut self, math: &mut M, collector: &DrawGradCollector<M>) {
        if collector.is_good {
            self.exp_variance_draw.add_sample(math, &collector.draw);
            self.exp_variance_grad.add_sample(math, &collector.grad);
            self.exp_variance_draw_bg.add_sample(math, &collector.draw);
            self.exp_variance_grad_bg.add_sample(math, &collector.grad);
        }
    }

    fn switch(&mut self, math: &mut M) {
        self.exp_variance_draw =
            std::mem::replace(&mut self.exp_variance_draw_bg, RunningVariance::new(math));
        self.exp_variance_grad =
            std::mem::replace(&mut self.exp_variance_grad_bg, RunningVariance::new(math));
    }

    fn current_count(&self) -> u64 {
        assert!(self.exp_variance_draw.count() == self.exp_variance_grad.count());
        self.exp_variance_draw.count()
    }

    fn background_count(&self) -> u64 {
        assert!(self.exp_variance_draw_bg.count() == self.exp_variance_grad_bg.count());
        self.exp_variance_draw_bg.count()
    }

    fn update_potential(
        &self,
        math: &mut M,
        potential: &mut EuclideanPotential<M, DiagMassMatrix<M>>,
    ) {
        if self.current_count() < 3 {
            return;
        }

        let (draw_var, draw_scale) = self.exp_variance_draw.current();
        let (grad_var, grad_scale) = self.exp_variance_grad.current();
        assert!(draw_scale == grad_scale);

        potential.mass_matrix.update_diag_draw_grad(
            math,
            draw_var,
            grad_var,
            None,
            (LOWER_LIMIT, UPPER_LIMIT),
        );
    }
}

#[derive(Clone, Debug)]
pub struct ExpWindowDiagAdaptStats {
    pub mass_matrix_inv: Option<Box<[f64]>>,
}

#[cfg(feature = "arrow")]
pub struct ExpWindowDiagAdaptStatsBuilder {
    mass_matrix_inv: Option<MutableFixedSizeListArray<MutablePrimitiveArray<f64>>>,
}

#[cfg(feature = "arrow")]
impl ArrowBuilder<ExpWindowDiagAdaptStats> for ExpWindowDiagAdaptStatsBuilder {
    fn append_value(&mut self, value: &ExpWindowDiagAdaptStats) {
        if let Some(store) = self.mass_matrix_inv.as_mut() {
            store
                .try_push(
                    value
                        .mass_matrix_inv
                        .as_ref()
                        .map(|vals| vals.iter().map(|&x| Some(x))),
                )
                .unwrap();
        }
    }

    fn finalize(self) -> Option<StructArray> {
        if let Some(mut store) = self.mass_matrix_inv {
            let fields = vec![Field::new(
                "mass_matrix_inv",
                store.data_type().clone(),
                true,
            )];

            let arrays = vec![store.as_box()];

            Some(StructArray::new(DataType::Struct(fields), arrays, None))
        } else {
            None
        }
    }
}

#[cfg(feature = "arrow")]
impl ArrowRow for ExpWindowDiagAdaptStats {
    type Builder = ExpWindowDiagAdaptStatsBuilder;

    fn new_builder(dim: usize, settings: &SamplerArgs) -> Self::Builder {
        if settings
            .mass_matrix_adapt
            .mass_matrix_options
            .store_mass_matrix
        {
            let items = MutablePrimitiveArray::new();
            let values = MutableFixedSizeListArray::new_with_field(items, "item", false, dim);
            Self::Builder {
                mass_matrix_inv: Some(values),
            }
        } else {
            Self::Builder {
                mass_matrix_inv: None,
            }
        }
    }
}

impl<M: Math> AdaptStrategy<M> for ExpWindowDiagAdapt<M> {
    type Potential = EuclideanPotential<M, DiagMassMatrix<M>>;
    type Collector = DrawGradCollector<M>;
    type Stats = ExpWindowDiagAdaptStats;
    type Options = DiagAdaptExpSettings;

    fn new(math: &mut M, options: Self::Options, _num_tune: u64) -> Self {
        Self {
            exp_variance_draw: RunningVariance::new(math),
            exp_variance_grad: RunningVariance::new(math),
            exp_variance_draw_bg: RunningVariance::new(math),
            exp_variance_grad_bg: RunningVariance::new(math),
            settings: options,
            _phantom: PhantomData::default(),
        }
    }

    fn init(
        &mut self,
        math: &mut M,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &State<M>,
    ) {
        self.exp_variance_draw.add_sample(math, &state.q);
        self.exp_variance_draw_bg.add_sample(math, &state.q);
        self.exp_variance_grad.add_sample(math, &state.grad);
        self.exp_variance_grad_bg.add_sample(math, &state.grad);

        potential.mass_matrix.update_diag_grad(
            math,
            &state.grad,
            1f64,
            (INIT_LOWER_LIMIT, INIT_UPPER_LIMIT),
        );
    }

    fn adapt(
        &mut self,
        _math: &mut M,
        _options: &mut NutsOptions,
        _potential: &mut Self::Potential,
        _draw: u64,
        _collector: &Self::Collector,
    ) {
        // Must be controlled from a different meta strategy
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        DrawGradCollector::new(math)
    }

    fn current_stats(
        &self,
        math: &mut M,
        _options: &NutsOptions,
        potential: &Self::Potential,
        _collector: &Self::Collector,
    ) -> Self::Stats {
        let diag = if self.settings.store_mass_matrix {
            Some(math.box_array(&potential.mass_matrix.variance))
        } else {
            None
        };
        ExpWindowDiagAdaptStats {
            mass_matrix_inv: diag,
        }
    }
}

pub(crate) struct GradDiagStrategy<M: Math> {
    step_size: DualAverageStrategy<M, DiagMassMatrix<M>>,
    mass_matrix: ExpWindowDiagAdapt<M>,
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
            step_size_window: 0.15,
            mass_matrix_switch_freq: 80,
            early_mass_matrix_switch_freq: 10,
        }
    }
}

impl<M: Math> AdaptStrategy<M> for GradDiagStrategy<M> {
    type Potential = EuclideanPotential<M, DiagMassMatrix<M>>;
    type Collector = CombinedCollector<M, AcceptanceRateCollector<M>, DrawGradCollector<M>>;
    type Stats = CombinedStats<DualAverageStats, ExpWindowDiagAdaptStats>;
    type Options = GradDiagOptions;

    fn new(math: &mut M, options: Self::Options, num_tune: u64) -> Self {
        let num_tune_f = num_tune as f64;
        let step_size_window = (options.step_size_window * num_tune_f) as u64;
        let early_end = (options.early_window * num_tune_f) as u64;
        let final_second_step_size = num_tune.saturating_sub(step_size_window);

        assert!(early_end < num_tune);

        Self {
            step_size: DualAverageStrategy::new(math, options.dual_average_options, num_tune),
            mass_matrix: ExpWindowDiagAdapt::new(math, options.mass_matrix_options, num_tune),
            options,
            num_tune,
            early_end,
            final_step_size_window: final_second_step_size,
        }
    }

    fn init(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &State<M>,
    ) {
        self.step_size.init(math, options, potential, state);
        self.mass_matrix.init(math, options, potential, state);
        self.step_size.enable();
    }

    fn adapt(
        &mut self,
        math: &mut M,
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

            self.mass_matrix
                .update_estimators(math, &collector.collector2);
            // We only switch if we have switch_freq draws in the background estimate,
            // and if the number of remaining mass matrix steps is larger than
            // the switch frequency.
            let could_switch = self.mass_matrix.background_count() >= switch_freq;
            let is_late = switch_freq + draw > self.final_step_size_window;
            if could_switch && (!is_late) {
                self.mass_matrix.switch(math);
            }
            self.mass_matrix.update_potential(math, potential);
            if is_late {
                self.step_size.use_mean_sym();
            }
            self.step_size
                .adapt(math, options, potential, draw, &collector.collector1);
            return;
        }

        if draw == self.num_tune - 1 {
            self.step_size.finalize();
        }
        self.step_size
            .adapt(math, options, potential, draw, &collector.collector1);
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        CombinedCollector {
            collector1: self.step_size.new_collector(math),
            collector2: self.mass_matrix.new_collector(math),
            _phantom: PhantomData::default(),
        }
    }

    fn current_stats(
        &self,
        math: &mut M,
        options: &NutsOptions,
        potential: &Self::Potential,
        collector: &Self::Collector,
    ) -> Self::Stats {
        CombinedStats {
            stats1: self
                .step_size
                .current_stats(math, options, potential, &collector.collector1),
            stats2: self
                .mass_matrix
                .current_stats(math, options, potential, &collector.collector2),
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

    fn finalize(self) -> Option<StructArray> {
        match (self.stats1.finalize(), self.stats2.finalize()) {
            (None, None) => None,
            (Some(stats1), None) => Some(stats1),
            (None, Some(stats2)) => Some(stats2),
            (Some(stats1), Some(stats2)) => {
                let mut data1 = stats1.into_data();
                let data2 = stats2.into_data();

                assert!(data1.2.is_none());
                assert!(data2.2.is_none());

                data1.0.extend(data2.0);
                data1.1.extend(data2.1);

                Some(StructArray::new(DataType::Struct(data1.0), data1.1, None))
            }
        }
    }
}

pub(crate) struct CombinedCollector<M: Math, C1: Collector<M>, C2: Collector<M>> {
    collector1: C1,
    collector2: C2,
    _phantom: PhantomData<M>,
}

impl<M: Math, C1, C2> Collector<M> for CombinedCollector<M, C1, C2>
where
    C1: Collector<M>,
    C2: Collector<M>,
{
    fn register_leapfrog(
        &mut self,
        math: &mut M,
        start: &State<M>,
        end: &State<M>,
        divergence_info: Option<&DivergenceInfo>,
    ) {
        self.collector1
            .register_leapfrog(math, start, end, divergence_info);
        self.collector2
            .register_leapfrog(math, start, end, divergence_info);
    }

    fn register_draw(&mut self, math: &mut M, state: &State<M>, info: &crate::nuts::SampleInfo) {
        self.collector1.register_draw(math, state, info);
        self.collector2.register_draw(math, state, info);
    }

    fn register_init(
        &mut self,
        math: &mut M,
        state: &State<M>,
        options: &crate::nuts::NutsOptions,
    ) {
        self.collector1.register_init(math, state, options);
        self.collector2.register_init(math, state, options);
    }
}

#[cfg(test)]
pub mod test_logps {
    use crate::{cpu_math::CpuLogpFunc, nuts::LogpError};
    use thiserror::Error;

    #[derive(Clone, Debug)]
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
        type LogpError = NormalLogpError;

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
    use crate::{
        cpu_math::CpuMath,
        nuts::{AdaptStrategy, Chain, NutsChain, NutsOptions},
    };

    #[test]
    fn instanciate_adaptive_sampler() {
        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.);
        let mut math = CpuMath::new(func);
        let num_tune = 100;
        let options = GradDiagOptions::default();
        let strategy = GradDiagStrategy::new(&mut math, options, num_tune);

        let mass_matrix = DiagMassMatrix::new(&mut math);
        let max_energy_error = 1000f64;
        let step_size = 0.1f64;

        let potential = EuclideanPotential::new(mass_matrix, max_energy_error, step_size);
        let options = NutsOptions {
            maxdepth: 10u64,
            store_gradient: true,
            store_unconstrained: true,
        };

        let rng = {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(42)
        };
        let chain = 0u64;

        let mut sampler = NutsChain::new(math, potential, strategy, options, rng, chain);
        sampler.set_position(&vec![1.5f64; ndim]).unwrap();
        for _ in 0..200 {
            sampler.draw().unwrap();
        }
    }
}
