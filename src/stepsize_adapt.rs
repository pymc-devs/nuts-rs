use std::marker::PhantomData;

use arrow::{
    array::{ArrayBuilder, PrimitiveBuilder, StructArray},
    datatypes::{DataType, Field, Float64Type, UInt64Type},
};
use rand::Rng;

use crate::{
    mass_matrix_adapt::MassMatrixAdaptStrategy,
    nuts::{
        AdaptStats, AdaptStrategy, Collector, Direction, Hamiltonian, NutsOptions, SamplerStats,
        StatTraceBuilder,
    },
    state::State,
    stepsize::{AcceptanceRateCollector, DualAverage, DualAverageOptions},
    Math, Settings,
};

pub struct Strategy<F, M> {
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

impl<F, M> Strategy<F, M> {
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn finalize(&mut self) {
        self.finalized = true;
    }

    pub fn use_mean_sym(&mut self) {
        self.use_mean_sym = true;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub step_size_bar: f64,
    pub mean_tree_accept: f64,
    pub mean_tree_accept_sym: f64,
    pub n_steps: u64,
}

pub struct StatsBuilder {
    step_size_bar: PrimitiveBuilder<Float64Type>,
    mean_tree_accept: PrimitiveBuilder<Float64Type>,
    mean_tree_accept_sym: PrimitiveBuilder<Float64Type>,
    n_steps: PrimitiveBuilder<UInt64Type>,
}

impl StatTraceBuilder<Stats> for StatsBuilder {
    fn append_value(&mut self, value: Stats) {
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

impl<M: Math, Mass: MassMatrixAdaptStrategy<M>> SamplerStats<M> for Strategy<M, Mass> {
    type Builder = StatsBuilder;
    type Stats = Stats;

    fn new_builder(&self, _settings: &impl Settings, _dim: usize) -> Self::Builder {
        Self::Builder {
            step_size_bar: PrimitiveBuilder::new(),
            mean_tree_accept: PrimitiveBuilder::new(),
            mean_tree_accept_sym: PrimitiveBuilder::new(),
            n_steps: PrimitiveBuilder::new(),
        }
    }

    fn current_stats(&self, _math: &mut M) -> Self::Stats {
        Stats {
            step_size_bar: self.step_size_adapt.current_step_size_adapted(),
            mean_tree_accept: self.last_mean_tree_accept,
            mean_tree_accept_sym: self.last_sym_mean_tree_accept,
            n_steps: self.last_n_steps,
        }
    }
}

impl<M: Math, Mass: MassMatrixAdaptStrategy<M>> AdaptStats<M> for Strategy<M, Mass> {
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

impl<M: Math, Mass: MassMatrixAdaptStrategy<M>> AdaptStrategy<M> for Strategy<M, Mass> {
    type Potential = Mass::Potential;
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

        *potential.stepsize_mut() = self.options.initial_step;

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
                *potential.stepsize_mut() = self.options.initial_step;
                return;
            };
            let accept_stat = collector.mean.current();
            match dir {
                Direction::Forward => {
                    if (accept_stat <= self.options.target_accept) | (potential.stepsize() > 1e5) {
                        self.step_size_adapt =
                            DualAverage::new(self.options.params, potential.stepsize());
                        return;
                    }
                    *potential.stepsize_mut() *= 2.;
                }
                Direction::Backward => {
                    if (accept_stat >= self.options.target_accept) | (potential.stepsize() < 1e-10)
                    {
                        self.step_size_adapt =
                            DualAverage::new(self.options.params, potential.stepsize());
                        return;
                    }
                    *potential.stepsize_mut() /= 2.;
                }
            }
        }
        // If we don't find something better, use the specified initial value
        *potential.stepsize_mut() = self.options.initial_step;
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
            *potential.stepsize_mut() = self.step_size_adapt.current_step_size_adapted();
            return;
        }
        if !self.enabled {
            return;
        }
        self.step_size_adapt
            .advance(current, self.options.target_accept);
        *potential.stepsize_mut() = self.step_size_adapt.current_step_size()
    }

    fn new_collector(&self, _math: &mut M) -> Self::Collector {
        AcceptanceRateCollector::new()
    }

    fn is_tuning(&self) -> bool {
        self.enabled
    }
}
