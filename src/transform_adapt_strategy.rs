use arrow::array::StructArray;

use crate::adapt_strategy::CombinedCollector;
use crate::chain::AdaptStrategy;
use crate::hamiltonian::{Hamiltonian, Point};
use crate::nuts::{Collector, NutsOptions, SampleInfo};
use crate::sampler_stats::{SamplerStats, StatTraceBuilder};
use crate::state::State;
use crate::stepsize::AcceptanceRateCollector;
use crate::stepsize_adapt::{StatsBuilder as StepSizeStatsBuilder, Strategy as StepSizeStrategy};
use crate::transformed_hamiltonian::TransformedHamiltonian;
use crate::{DualAverageSettings, Math, NutsError, Settings};

#[derive(Clone, Copy, Debug)]
pub struct TransformedSettings {
    pub step_size_window: f64,
    pub transform_update_freq: u64,
    pub use_orbit_for_training: bool,
    pub dual_average_options: DualAverageSettings,
    pub transform_train_max_energy_error: f64,
}

impl Default for TransformedSettings {
    fn default() -> Self {
        Self {
            step_size_window: 0.07f64,
            transform_update_freq: 128,
            use_orbit_for_training: false,
            transform_train_max_energy_error: 20f64,
            dual_average_options: Default::default(),
        }
    }
}

pub struct TransformAdaptation {
    step_size: StepSizeStrategy,
    options: TransformedSettings,
    num_tune: u64,
    final_window_size: u64,
    tuning: bool,
    chain: u64,
}

pub struct Builder {
    step_size: StepSizeStatsBuilder,
}

impl<M: Math> StatTraceBuilder<M, TransformAdaptation> for Builder {
    fn append_value(&mut self, math: Option<&mut M>, value: &TransformAdaptation) {
        let Self { step_size } = self;
        step_size.append_value(math, &value.step_size);
    }

    fn finalize(self) -> Option<StructArray> {
        let Self { step_size } = self;
        <StepSizeStatsBuilder as StatTraceBuilder<M, _>>::finalize(step_size)
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self { step_size } = self;
        <StepSizeStatsBuilder as StatTraceBuilder<M, _>>::inspect(step_size)
    }
}

impl<M: Math> SamplerStats<M> for TransformAdaptation {
    type Builder = Builder;
    type StatOptions = ();

    fn new_builder(
        &self,
        _stat_options: Self::StatOptions,
        settings: &impl Settings,
        dim: usize,
    ) -> Self::Builder {
        let step_size = SamplerStats::<M>::new_builder(&self.step_size, (), settings, dim);
        Builder { step_size }
    }
}

pub struct DrawCollector<M: Math> {
    draws: Vec<M::Vector>,
    grads: Vec<M::Vector>,
    logps: Vec<f64>,
    collect_orbit: bool,
    max_energy_error: f64,
}

impl<M: Math> DrawCollector<M> {
    fn new(_math: &mut M, collect_orbit: bool, max_energy_error: f64) -> Self {
        Self {
            draws: vec![],
            grads: vec![],
            logps: vec![],
            collect_orbit,
            max_energy_error,
        }
    }
}

impl<M: Math, P: Point<M>> Collector<M, P> for DrawCollector<M> {
    fn register_leapfrog(
        &mut self,
        math: &mut M,
        _start: &State<M, P>,
        end: &State<M, P>,
        divergence_info: Option<&crate::DivergenceInfo>,
    ) {
        if divergence_info.is_some() {
            return;
        }

        if self.collect_orbit {
            let point = end.point();
            let energy_error = point.energy_error();
            if !energy_error.is_finite() {
                return;
            }

            if energy_error > self.max_energy_error {
                return;
            }

            if !math.array_all_finite(point.position()) {
                return;
            }
            if !math.array_all_finite(point.gradient()) {
                return;
            }

            self.draws.push(math.copy_array(point.position()));
            self.grads.push(math.copy_array(point.gradient()));
            self.logps.push(point.logp());
        }
    }

    fn register_draw(&mut self, math: &mut M, state: &State<M, P>, _info: &SampleInfo) {
        if !self.collect_orbit {
            let point = state.point();
            let energy_error = point.energy_error();
            if !energy_error.is_finite() {
                return;
            }

            if energy_error > self.max_energy_error {
                return;
            }

            if !math.array_all_finite(point.position()) {
                return;
            }
            if !math.array_all_finite(point.gradient()) {
                return;
            }

            self.draws.push(math.copy_array(point.position()));
            self.grads.push(math.copy_array(point.gradient()));
            self.logps.push(point.logp());
        }
    }
}

impl<M: Math> AdaptStrategy<M> for TransformAdaptation {
    type Hamiltonian = TransformedHamiltonian<M>;

    type Collector = CombinedCollector<
        M,
        <Self::Hamiltonian as Hamiltonian<M>>::Point,
        AcceptanceRateCollector,
        DrawCollector<M>,
    >;

    type Options = TransformedSettings;

    fn new(_math: &mut M, options: Self::Options, num_tune: u64, chain: u64) -> Self {
        let step_size = StepSizeStrategy::new(options.dual_average_options);
        let final_window_size =
            ((num_tune as f64) * (1f64 - options.step_size_window)).floor() as u64;
        Self {
            step_size,
            options,
            num_tune,
            final_window_size,
            tuning: true,
            chain,
        }
    }

    fn init<R: rand::Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        position: &[f64],
        rng: &mut R,
    ) -> Result<(), NutsError> {
        hamiltonian.init_transformation(rng, math, position, self.chain)?;
        self.step_size
            .init(math, options, hamiltonian, position, rng)?;
        Ok(())
    }

    fn adapt<R: rand::Rng + ?Sized>(
        &mut self,
        math: &mut M,
        _options: &mut NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        draw: u64,
        collector: &Self::Collector,
        _state: &State<M, <Self::Hamiltonian as Hamiltonian<M>>::Point>,
        rng: &mut R,
    ) -> Result<(), NutsError> {
        self.step_size.update(&collector.collector1);

        if draw >= self.num_tune {
            self.tuning = false;
            return Ok(());
        }

        if draw < self.final_window_size {
            if draw < 100 {
                if (draw > 0) & (draw % 10 == 0) {
                    hamiltonian.update_params(
                        math,
                        rng,
                        collector.collector2.draws.iter(),
                        collector.collector2.grads.iter(),
                        collector.collector2.logps.iter(),
                    )?;
                }
            } else if (draw > 0) & (draw % self.options.transform_update_freq == 0) {
                hamiltonian.update_params(
                    math,
                    rng,
                    collector.collector2.draws.iter(),
                    collector.collector2.grads.iter(),
                    collector.collector2.logps.iter(),
                )?;
            }
            self.step_size.update_estimator_early();
            self.step_size.update_stepsize(hamiltonian, false);
            return Ok(());
        }

        self.step_size.update_estimator_late();
        let is_last = draw == self.num_tune - 1;
        self.step_size.update_stepsize(hamiltonian, is_last);
        Ok(())
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        Self::Collector::new(
            self.step_size.new_collector(),
            DrawCollector::new(
                math,
                self.options.use_orbit_for_training,
                self.options.transform_train_max_energy_error,
            ),
        )
    }

    fn is_tuning(&self) -> bool {
        self.tuning
    }

    fn last_num_steps(&self) -> u64 {
        self.step_size.last_n_steps
    }
}
