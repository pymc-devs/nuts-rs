use arrow::{
    array::{ArrayBuilder, PrimitiveBuilder, StructArray},
    datatypes::{DataType, Field, Float64Type, UInt64Type},
};
use rand::Rng;

use crate::{
    nuts::{
        AdaptStats, Collector, Direction, Hamiltonian, NutsOptions, SamplerStats, StatTraceBuilder,
    },
    state::State,
    stepsize::{AcceptanceRateCollector, DualAverage, DualAverageOptions},
    Math, Settings,
};

pub struct Strategy {
    step_size_adapt: DualAverage,
    options: DualAverageSettings,
    last_mean_tree_accept: f64,
    last_sym_mean_tree_accept: f64,
    last_n_steps: u64,
}

impl Strategy {
    pub fn new(options: DualAverageSettings) -> Self {
        Self {
            options,
            step_size_adapt: DualAverage::new(options.params, options.initial_step),
            last_n_steps: 0,
            last_sym_mean_tree_accept: 0.0,
            last_mean_tree_accept: 0.0,
        }
    }

    pub fn init<M: Math, R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        potential: &mut impl Hamiltonian<M>,
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

    pub fn update(&mut self, collector: &AcceptanceRateCollector) {
        let mean_sym = collector.mean_sym.current();
        let mean = collector.mean.current();
        let n_steps = collector.mean.count();
        self.last_mean_tree_accept = mean;
        self.last_sym_mean_tree_accept = mean_sym;
        self.last_n_steps = n_steps;
    }

    pub fn update_estimator_early(&mut self) {
        self.step_size_adapt
            .advance(self.last_mean_tree_accept, self.options.target_accept);
    }

    pub fn update_estimator_late(&mut self) {
        self.step_size_adapt
            .advance(self.last_sym_mean_tree_accept, self.options.target_accept);
    }

    pub fn update_stepsize<M: Math>(
        &mut self,
        potential: &mut impl Hamiltonian<M>,
        use_best_guess: bool,
    ) {
        if use_best_guess {
            *potential.stepsize_mut() = self.step_size_adapt.current_step_size_adapted();
        } else {
            *potential.stepsize_mut() = self.step_size_adapt.current_step_size();
        }
    }

    pub fn new_collector(&self) -> AcceptanceRateCollector {
        AcceptanceRateCollector::new()
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

impl<M: Math> SamplerStats<M> for Strategy {
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

impl<M: Math> AdaptStats<M> for Strategy {
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
