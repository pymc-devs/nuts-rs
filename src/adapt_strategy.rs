use std::{fmt::Debug, marker::PhantomData, ops::Deref};

use arrow::{
    array::{ArrayBuilder, PrimitiveBuilder, StructArray},
    datatypes::{DataType, Field, Float64Type, UInt64Type},
};
use itertools::Itertools;
use rand::Rng;

use crate::{
    mass_matrix::{DiagMassMatrix, DrawGradCollector, MassMatrix, RunningVariance},
    math_base::Math,
    nuts::{AdaptStats, AdaptStrategy, Collector, Direction, Hamiltonian, NutsOptions},
    potential::EuclideanPotential,
    sampler::Settings,
    state::State,
    stepsize::{AcceptanceRateCollector, DualAverage, DualAverageOptions},
    DivergenceInfo,
};

use crate::nuts::{SamplerStats, StatTraceBuilder};

const LOWER_LIMIT: f64 = 1e-20f64;
const UPPER_LIMIT: f64 = 1e20f64;

const INIT_LOWER_LIMIT: f64 = 1e-20f64;
const INIT_UPPER_LIMIT: f64 = 1e20f64;

pub struct DualAverageStrategy<F, M> {
    step_size_adapt: DualAverage,
    options: DualAverageSettings,
    enabled: bool,
    use_mean_sym: bool,
    finalized: bool,
    last_mean_tree_accept: f64,
    last_sym_mean_tree_accept: f64,
    last_n_steps: u64,
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

pub struct DualAverageStatsBuilder {
    step_size_bar: PrimitiveBuilder<Float64Type>,
    mean_tree_accept: PrimitiveBuilder<Float64Type>,
    mean_tree_accept_sym: PrimitiveBuilder<Float64Type>,
    n_steps: PrimitiveBuilder<UInt64Type>,
}

impl StatTraceBuilder<DualAverageStats> for DualAverageStatsBuilder {
    fn append_value(&mut self, value: DualAverageStats) {
        self.step_size_bar.append_value(value.step_size_bar);
        self.mean_tree_accept.append_value(value.mean_tree_accept);
        self.mean_tree_accept_sym
            .append_value(value.mean_tree_accept_sym);
        self.n_steps.append_value(value.n_steps);
    }

    fn finalize(self) -> Option<StructArray> {
        let Self {
            mut step_size_bar,
            mut mean_tree_accept,
            mut mean_tree_accept_sym,
            mut n_steps,
        } = self;

        let fields = vec![
            Field::new("step_size_bar", DataType::Float64, false),
            Field::new("mean_tree_accept", DataType::Float64, false),
            Field::new("mean_tree_accept_sym", DataType::Float64, false),
            Field::new("n_steps", DataType::UInt64, false),
        ];

        let arrays = vec![
            ArrayBuilder::finish(&mut step_size_bar),
            ArrayBuilder::finish(&mut mean_tree_accept),
            ArrayBuilder::finish(&mut mean_tree_accept_sym),
            ArrayBuilder::finish(&mut n_steps),
        ];

        Some(StructArray::new(fields.into(), arrays, None))
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self {
            step_size_bar,
            mean_tree_accept,
            mean_tree_accept_sym,
            n_steps,
        } = self;

        let fields = vec![
            Field::new("step_size_bar", DataType::Float64, false),
            Field::new("mean_tree_accept", DataType::Float64, false),
            Field::new("mean_tree_accept_sym", DataType::Float64, false),
            Field::new("n_steps", DataType::UInt64, false),
        ];

        let arrays = vec![
            ArrayBuilder::finish_cloned(step_size_bar),
            ArrayBuilder::finish_cloned(mean_tree_accept),
            ArrayBuilder::finish_cloned(mean_tree_accept_sym),
            ArrayBuilder::finish_cloned(n_steps),
        ];

        Some(StructArray::new(fields.into(), arrays, None))
    }
}

impl<M: Math, Mass: MassMatrix<M>> SamplerStats<M> for DualAverageStrategy<M, Mass> {
    type Builder = DualAverageStatsBuilder;
    type Stats = DualAverageStats;

    fn new_builder(&self, _settings: &impl Settings, _dim: usize) -> Self::Builder {
        Self::Builder {
            step_size_bar: PrimitiveBuilder::new(),
            mean_tree_accept: PrimitiveBuilder::new(),
            mean_tree_accept_sym: PrimitiveBuilder::new(),
            n_steps: PrimitiveBuilder::new(),
        }
    }

    fn current_stats(&self, _math: &mut M) -> Self::Stats {
        DualAverageStats {
            step_size_bar: self.step_size_adapt.current_step_size_adapted(),
            mean_tree_accept: self.last_mean_tree_accept,
            mean_tree_accept_sym: self.last_sym_mean_tree_accept,
            n_steps: self.last_n_steps,
        }
    }
}

impl<M: Math, Mass: MassMatrix<M>> AdaptStats<M> for DualAverageStrategy<M, Mass> {
    fn num_grad_evals(stats: &Self::Stats) -> usize {
        stats.n_steps as usize
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
    type Options = DualAverageSettings;

    fn new(_math: &mut M, options: Self::Options, _num_tune: u64) -> Self {
        Self {
            options,
            enabled: true,
            step_size_adapt: DualAverage::new(options.params, options.initial_step),
            finalized: false,
            use_mean_sym: false,
            last_n_steps: 0,
            last_sym_mean_tree_accept: 0.0,
            last_mean_tree_accept: 0.0,
            _phantom1: PhantomData,
            _phantom2: PhantomData,
        }
    }

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &State<M>,
        rng: &mut R,
    ) {
        let mut pool = potential.new_pool(math, 1);

        let mut state = potential.copy_state(math, &mut pool, state);
        state
            .try_mut_inner()
            .expect("New state should have only one reference")
            .idx_in_trajectory = 0;
        potential.randomize_momentum(math, &mut state, rng);

        let mut collector = AcceptanceRateCollector::new();

        collector.register_init(math, &state, options);

        potential.step_size = self.options.initial_step;

        let state_next = potential.leapfrog(
            math,
            &mut pool,
            &state,
            Direction::Forward,
            state.energy(),
            &mut collector,
        );

        let Ok(_) = state_next else {
            return;
        };

        let accept_stat = collector.mean.current();
        let dir = if accept_stat > self.options.target_accept {
            Direction::Forward
        } else {
            Direction::Backward
        };

        for _ in 0..100 {
            let mut collector = AcceptanceRateCollector::new();
            collector.register_init(math, &state, options);
            let state_next =
                potential.leapfrog(math, &mut pool, &state, dir, state.energy(), &mut collector);
            let Ok(_) = state_next else {
                potential.step_size = self.options.initial_step;
                return;
            };
            let accept_stat = collector.mean.current();
            match dir {
                Direction::Forward => {
                    if (accept_stat <= self.options.target_accept) | (potential.step_size > 1e5) {
                        self.step_size_adapt =
                            DualAverage::new(self.options.params, potential.step_size);
                        return;
                    }
                    potential.step_size *= 2.;
                }
                Direction::Backward => {
                    if (accept_stat >= self.options.target_accept) | (potential.step_size < 1e-10) {
                        self.step_size_adapt =
                            DualAverage::new(self.options.params, potential.step_size);
                        return;
                    }
                    potential.step_size /= 2.;
                }
            }
        }
        // If we don't find something better, use the specified initial value
        potential.step_size = self.options.initial_step;
    }

    fn adapt<R: Rng + ?Sized>(
        &mut self,
        _math: &mut M,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        _draw: u64,
        collector: &Self::Collector,
        _state: &State<M>,
        _rng: &mut R,
    ) {
        let mean_sym = collector.mean_sym.current();
        let mean = collector.mean.current();
        let n_steps = collector.mean.count();
        self.last_mean_tree_accept = mean;
        self.last_sym_mean_tree_accept = mean_sym;
        self.last_n_steps = n_steps;

        let current = if self.use_mean_sym { mean_sym } else { mean };
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

    fn is_tuning(&self) -> bool {
        self.enabled
    }
}

/// Settings for mass matrix adaptation
#[derive(Clone, Copy, Debug)]
pub struct DiagAdaptExpSettings {
    pub store_mass_matrix: bool,
    pub use_grad_based_estimate: bool,
}

impl Default for DiagAdaptExpSettings {
    fn default() -> Self {
        Self {
            store_mass_matrix: false,
            use_grad_based_estimate: true,
        }
    }
}

pub struct ExpWindowDiagAdapt<M: Math> {
    exp_variance_draw: RunningVariance<M>,
    exp_variance_grad: RunningVariance<M>,
    exp_variance_grad_bg: RunningVariance<M>,
    exp_variance_draw_bg: RunningVariance<M>,
    _settings: DiagAdaptExpSettings,
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

    /// Give the opportunity to update the potential and return if it was changed
    fn update_potential(
        &self,
        math: &mut M,
        potential: &mut EuclideanPotential<M, DiagMassMatrix<M>>,
    ) -> bool {
        if self.current_count() < 3 {
            return false;
        }

        let (draw_var, draw_scale) = self.exp_variance_draw.current();
        let (grad_var, grad_scale) = self.exp_variance_grad.current();
        assert!(draw_scale == grad_scale);

        if self._settings.use_grad_based_estimate {
            potential.mass_matrix.update_diag_draw_grad(
                math,
                draw_var,
                grad_var,
                None,
                (LOWER_LIMIT, UPPER_LIMIT),
            );
        } else {
            let scale = (self.exp_variance_draw.count() as f64).recip();
            potential.mass_matrix.update_diag_draw(
                math,
                draw_var,
                scale,
                None,
                (LOWER_LIMIT, UPPER_LIMIT),
            );
        }

        true
    }
}

pub type ExpWindowDiagAdaptStats = ();
type ExpWindowDiagAdaptStatsBuilder = ();

impl<M: Math> SamplerStats<M> for ExpWindowDiagAdapt<M> {
    type Builder = ExpWindowDiagAdaptStats;
    type Stats = ExpWindowDiagAdaptStatsBuilder;

    fn new_builder(&self, _settings: &impl Settings, _dim: usize) -> Self::Builder {}

    fn current_stats(&self, _math: &mut M) -> Self::Stats {}
}

impl<M: Math> AdaptStats<M> for ExpWindowDiagAdapt<M> {
    // This is never called
    fn num_grad_evals(_stats: &Self::Stats) -> usize {
        unimplemented!()
    }
}

impl<M: Math> AdaptStrategy<M> for ExpWindowDiagAdapt<M> {
    type Potential = EuclideanPotential<M, DiagMassMatrix<M>>;
    type Collector = DrawGradCollector<M>;
    type Options = DiagAdaptExpSettings;

    fn new(math: &mut M, options: Self::Options, _num_tune: u64) -> Self {
        Self {
            exp_variance_draw: RunningVariance::new(math),
            exp_variance_grad: RunningVariance::new(math),
            exp_variance_draw_bg: RunningVariance::new(math),
            exp_variance_grad_bg: RunningVariance::new(math),
            _settings: options,
            _phantom: PhantomData,
        }
    }

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &State<M>,
        _rng: &mut R,
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

    fn adapt<R: Rng + ?Sized>(
        &mut self,
        _math: &mut M,
        _options: &mut NutsOptions,
        _potential: &mut Self::Potential,
        _draw: u64,
        _collector: &Self::Collector,
        _state: &State<M>,
        _rng: &mut R,
    ) {
        // Must be controlled from a different meta strategy
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        DrawGradCollector::new(math)
    }

    fn is_tuning(&self) -> bool {
        todo!()
    }
}

pub struct GradDiagStrategy<M: Math> {
    step_size: DualAverageStrategy<M, DiagMassMatrix<M>>,
    mass_matrix: ExpWindowDiagAdapt<M>,
    options: GradDiagOptions,
    num_tune: u64,
    // The number of draws in the the early window
    early_end: u64,

    // The first draw number for the final step size adaptation window
    final_step_size_window: u64,
    tuning: bool,
    has_initial_mass_matrix: bool,
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

impl<M: Math> SamplerStats<M> for GradDiagStrategy<M> {
    type Stats = CombinedStats<DualAverageStats, ExpWindowDiagAdaptStats>;
    type Builder = CombinedStatsBuilder<DualAverageStatsBuilder, ExpWindowDiagAdaptStatsBuilder>;

    fn current_stats(&self, math: &mut M) -> Self::Stats {
        CombinedStats {
            stats1: self.step_size.current_stats(math),
            stats2: self.mass_matrix.current_stats(math),
        }
    }

    fn new_builder(&self, settings: &impl Settings, dim: usize) -> Self::Builder {
        CombinedStatsBuilder {
            stats1: self.step_size.new_builder(settings, dim),
            stats2: self.mass_matrix.new_builder(settings, dim),
        }
    }
}

impl<M: Math> AdaptStats<M> for GradDiagStrategy<M> {
    fn num_grad_evals(stats: &Self::Stats) -> usize {
        stats.stats1.n_steps as usize
    }
}

impl<M: Math> AdaptStrategy<M> for GradDiagStrategy<M> {
    type Potential = EuclideanPotential<M, DiagMassMatrix<M>>;
    type Collector = CombinedCollector<M, AcceptanceRateCollector<M>, DrawGradCollector<M>>;
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
            tuning: true,
            has_initial_mass_matrix: true,
        }
    }

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &State<M>,
        rng: &mut R,
    ) {
        self.mass_matrix.init(math, options, potential, state, rng);
        self.step_size.init(math, options, potential, state, rng);
        self.step_size.enable();
    }

    fn adapt<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: u64,
        collector: &Self::Collector,
        state: &State<M>,
        rng: &mut R,
    ) {
        if draw >= self.num_tune {
            self.tuning = false;
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
            let did_change = self.mass_matrix.update_potential(math, potential);
            if is_late {
                self.step_size.use_mean_sym();
            }
            // First time we change the mass matrix
            if did_change & self.has_initial_mass_matrix {
                self.has_initial_mass_matrix = false;
                self.step_size.init(math, options, potential, state, rng);
            } else {
                self.step_size.adapt(
                    math,
                    options,
                    potential,
                    draw,
                    &collector.collector1,
                    state,
                    rng,
                );
            }
            return;
        }

        if draw == self.num_tune - 1 {
            self.step_size.finalize();
        }
        self.step_size.adapt(
            math,
            options,
            potential,
            draw,
            &collector.collector1,
            state,
            rng,
        );
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        CombinedCollector {
            collector1: self.step_size.new_collector(math),
            collector2: self.mass_matrix.new_collector(math),
            _phantom: PhantomData,
        }
    }

    fn is_tuning(&self) -> bool {
        self.tuning
    }
}

#[derive(Debug, Clone)]
pub struct CombinedStats<D1, D2> {
    pub stats1: D1,
    pub stats2: D2,
}

#[derive(Clone)]
pub struct CombinedStatsBuilder<B1, B2> {
    stats1: B1,
    stats2: B2,
}

impl<S1, S2, B1, B2> StatTraceBuilder<CombinedStats<S1, S2>> for CombinedStatsBuilder<B1, B2>
where
    B1: StatTraceBuilder<S1>,
    B2: StatTraceBuilder<S2>,
{
    fn append_value(&mut self, value: CombinedStats<S1, S2>) {
        self.stats1.append_value(value.stats1);
        self.stats2.append_value(value.stats2);
    }

    fn finalize(self) -> Option<StructArray> {
        let Self { stats1, stats2 } = self;
        match (stats1.finalize(), stats2.finalize()) {
            (None, None) => None,
            (Some(stats1), None) => Some(stats1),
            (None, Some(stats2)) => Some(stats2),
            (Some(stats1), Some(stats2)) => {
                let mut data1 = stats1.into_parts();
                let data2 = stats2.into_parts();

                assert!(data1.2.is_none());
                assert!(data2.2.is_none());

                let mut fields = data1.0.into_iter().map(|x| x.deref().clone()).collect_vec();

                fields.extend(data2.0.into_iter().map(|x| x.deref().clone()));
                data1.1.extend(data2.1);

                Some(StructArray::new(data1.0, data1.1, None))
            }
        }
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self { stats1, stats2 } = self;
        match (stats1.inspect(), stats2.inspect()) {
            (None, None) => None,
            (Some(stats1), None) => Some(stats1),
            (None, Some(stats2)) => Some(stats2),
            (Some(stats1), Some(stats2)) => {
                let mut data1 = stats1.into_parts();
                let data2 = stats2.into_parts();

                assert!(data1.2.is_none());
                assert!(data2.2.is_none());

                let mut fields = data1.0.into_iter().map(|x| x.deref().clone()).collect_vec();

                fields.extend(data2.0.into_iter().map(|x| x.deref().clone()));
                data1.1.extend(data2.1);

                Some(StructArray::new(data1.0, data1.1, None))
            }
        }
    }
}

pub struct CombinedCollector<M: Math, C1: Collector<M>, C2: Collector<M>> {
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

        let mass_matrix = DiagMassMatrix::new(&mut math, true);
        let max_energy_error = 1000f64;
        let step_size = 0.1f64;

        let potential = EuclideanPotential::new(mass_matrix, max_energy_error, step_size);
        let options = NutsOptions {
            maxdepth: 10u64,
            store_gradient: true,
            store_unconstrained: true,
            check_turning: true,
            store_divergences: false,
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
