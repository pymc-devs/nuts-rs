use arrow::{
    array::{ArrayBuilder, PrimitiveBuilder, StructArray},
    datatypes::{DataType, Field, Float64Type, UInt64Type},
};
use rand::Rng;

use crate::{
    hamiltonian::{Direction, Hamiltonian, LeapfrogResult, Point},
    nuts::{Collector, NutsOptions},
    sampler_stats::{SamplerStats, StatTraceBuilder},
    stepsize::{AcceptanceRateCollector, DualAverage, DualAverageOptions},
    Math, NutsError, Settings,
};

pub struct Strategy {
    step_size_adapt: DualAverage,
    options: DualAverageSettings,
    pub last_mean_tree_accept: f64,
    pub last_sym_mean_tree_accept: f64,
    pub last_n_steps: u64,
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

    pub fn init<M: Math, R: Rng + ?Sized, P: Point<M>>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut impl Hamiltonian<M, Point = P>,
        position: &[f64],
        rng: &mut R,
    ) -> Result<(), NutsError> {
        if let Some(step_size) = self.options.fixed_step_size {
            *hamiltonian.step_size_mut() = step_size;
            return Ok(());
        }
        let mut state = hamiltonian.init_state(math, position)?;
        hamiltonian.initialize_trajectory(math, &mut state, rng)?;

        let mut collector = AcceptanceRateCollector::new();

        collector.register_init(math, &state, options);

        *hamiltonian.step_size_mut() = self.options.initial_step;

        let state_next = hamiltonian.leapfrog(math, &state, Direction::Forward, &mut collector);

        let LeapfrogResult::Ok(_) = state_next else {
            return Ok(());
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
            let state_next = hamiltonian.leapfrog(math, &state, dir, &mut collector);
            let LeapfrogResult::Ok(_) = state_next else {
                *hamiltonian.step_size_mut() = self.options.initial_step;
                return Ok(());
            };
            let accept_stat = collector.mean.current();
            match dir {
                Direction::Forward => {
                    if (accept_stat <= self.options.target_accept) | (hamiltonian.step_size() > 1e5)
                    {
                        self.step_size_adapt =
                            DualAverage::new(self.options.params, hamiltonian.step_size());
                        return Ok(());
                    }
                    *hamiltonian.step_size_mut() *= 2.;
                }
                Direction::Backward => {
                    if (accept_stat >= self.options.target_accept)
                        | (hamiltonian.step_size() < 1e-10)
                    {
                        self.step_size_adapt =
                            DualAverage::new(self.options.params, hamiltonian.step_size());
                        return Ok(());
                    }
                    *hamiltonian.step_size_mut() /= 2.;
                }
            }
        }
        // If we don't find something better, use the specified initial value
        *hamiltonian.step_size_mut() = self.options.initial_step;
        Ok(())
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
        hamiltonian: &mut impl Hamiltonian<M>,
        use_best_guess: bool,
    ) {
        if let Some(step_size) = self.options.fixed_step_size {
            *hamiltonian.step_size_mut() = step_size;
            return;
        }
        if use_best_guess {
            *hamiltonian.step_size_mut() = self.step_size_adapt.current_step_size_adapted();
        } else {
            *hamiltonian.step_size_mut() = self.step_size_adapt.current_step_size();
        }
    }

    pub fn new_collector(&self) -> AcceptanceRateCollector {
        AcceptanceRateCollector::new()
    }
}

pub struct StatsBuilder {
    step_size_bar: PrimitiveBuilder<Float64Type>,
    mean_tree_accept: PrimitiveBuilder<Float64Type>,
    mean_tree_accept_sym: PrimitiveBuilder<Float64Type>,
    n_steps: PrimitiveBuilder<UInt64Type>,
}

impl<M: Math> StatTraceBuilder<M, Strategy> for StatsBuilder {
    fn append_value(&mut self, _math: Option<&mut M>, value: &Strategy) {
        self.step_size_bar
            .append_value(value.step_size_adapt.current_step_size_adapted());
        self.mean_tree_accept
            .append_value(value.last_mean_tree_accept);
        self.mean_tree_accept_sym
            .append_value(value.last_sym_mean_tree_accept);
        self.n_steps.append_value(value.last_n_steps);
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
    type StatOptions = ();

    fn new_builder(
        &self,
        _stat_options: Self::StatOptions,
        _settings: &impl Settings,
        _dim: usize,
    ) -> Self::Builder {
        Self::Builder {
            step_size_bar: PrimitiveBuilder::new(),
            mean_tree_accept: PrimitiveBuilder::new(),
            mean_tree_accept_sym: PrimitiveBuilder::new(),
            n_steps: PrimitiveBuilder::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DualAverageSettings {
    pub target_accept: f64,
    pub initial_step: f64,
    pub params: DualAverageOptions,
    pub fixed_step_size: Option<f64>,
}

impl Default for DualAverageSettings {
    fn default() -> Self {
        Self {
            target_accept: 0.8,
            initial_step: 0.1,
            params: DualAverageOptions::default(),
            fixed_step_size: None,
        }
    }
}
