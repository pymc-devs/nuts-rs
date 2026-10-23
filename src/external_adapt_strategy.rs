//! Adaptation strategy for when the coordinate transformation is learned from data rather than computed analytically.

use nuts_derive::Storable;
use nuts_storable::{HasDims, Storable};
use serde::{Deserialize, Serialize};

use crate::adapt_strategy::CombinedCollector;
use crate::chain::AdaptStrategy;
use crate::dynamics::{
    Hamiltonian, KineticEnergyKind, Point, State, TransformedHamiltonian, TransformedPoint,
};
use crate::nuts::{Collector, NutsOptions, SampleInfo};
use crate::sampler_stats::{SamplerStats, StatsDims};
use crate::stepsize::AcceptanceRateCollector;
use crate::stepsize::{StepSizeSettings, Strategy as StepSizeStrategy};
use crate::transform::ExternalTransformation;
use crate::walnuts::WalnutsOptions;
use crate::warmup::{DynamicsOptions, WarmupPhase, WarmupSchedule, set_phase_settings};
use crate::{Math, NutsError};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FlowSettings {
    /// Fraction of the tuning draws that belong to the early warmup phase,
    /// see [`WarmupPhase::Early`].
    pub early_window: f64,
    pub step_size_window: f64,
    pub transform_update_freq: u64,
    /// Number of draws between transformation updates in the early phase.
    pub early_transform_update_freq: u64,
    pub use_orbit_for_training: bool,
    pub step_size_settings: StepSizeSettings,
    pub transform_train_max_energy_error: f64,
    /// See `EuclideanAdaptOptions::trajectory_kind`.
    pub trajectory_kind: KineticEnergyKind,
    /// See `EuclideanAdaptOptions::early_trajectory_kind`.
    pub early_trajectory_kind: Option<KineticEnergyKind>,
    /// See `EuclideanAdaptOptions::gradient_clipping`.
    pub gradient_clipping: Option<f64>,
    /// See `EuclideanAdaptOptions::early_gradient_clipping`.
    pub early_gradient_clipping: Option<f64>,
    /// See `EuclideanAdaptOptions::walnuts`.
    pub walnuts: Option<WalnutsOptions>,
}

/// Backwards-compatible alias for [`FlowSettings`].
#[deprecated(since = "0.0.0", note = "Use FlowSettings instead")]
pub type TransformedSettings = FlowSettings;

impl FlowSettings {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        self.step_size_settings.validate()?;
        if !(0.0..1.0).contains(&self.early_window) {
            anyhow::bail!(
                "early_window must be at least 0 and smaller than 1, got {}",
                self.early_window
            );
        }
        self.validate_dynamics()?;
        if self.early_transform_update_freq == 0 {
            anyhow::bail!("early_transform_update_freq must be positive");
        }
        if !(0.0..=1.0).contains(&self.step_size_window) {
            anyhow::bail!(
                "step_size_window must be between 0 and 1, got {}",
                self.step_size_window
            );
        }
        Ok(())
    }
}

impl Default for FlowSettings {
    fn default() -> Self {
        Self {
            // With the default 3000 tuning draws of the flow NUTS settings,
            // the early phase ends at draw 99.
            early_window: 0.033,
            step_size_window: 0.07f64,
            transform_update_freq: 128,
            early_transform_update_freq: 10,
            use_orbit_for_training: false,
            transform_train_max_energy_error: 20f64,
            step_size_settings: Default::default(),
            trajectory_kind: KineticEnergyKind::default(),
            early_trajectory_kind: None,
            gradient_clipping: None,
            early_gradient_clipping: Some(1e6),
            walnuts: None,
        }
    }
}

impl DynamicsOptions for FlowSettings {
    fn trajectory_kind(&self) -> KineticEnergyKind {
        self.trajectory_kind
    }

    fn early_trajectory_kind(&self) -> Option<KineticEnergyKind> {
        self.early_trajectory_kind
    }

    fn gradient_clipping(&self) -> Option<f64> {
        self.gradient_clipping
    }

    fn early_gradient_clipping(&self) -> Option<f64> {
        self.early_gradient_clipping
    }

    fn walnuts(&self) -> Option<WalnutsOptions> {
        self.walnuts
    }
}

pub struct ExternalTransformAdaptation {
    step_size: StepSizeStrategy,
    options: FlowSettings,
    schedule: WarmupSchedule,
    tuning: bool,
    chain: u64,
}

#[derive(Debug, Storable)]
pub struct Stats<P: HasDims, S: Storable<P>> {
    tuning: bool,
    #[storable(flatten)]
    pub step_size: S,
    #[storable(ignore)]
    _phantom: std::marker::PhantomData<fn() -> P>,
}

impl<M: Math> SamplerStats<M> for ExternalTransformAdaptation {
    type Stats = Stats<StatsDims, <StepSizeStrategy as SamplerStats<M>>::Stats>;
    type StatsOptions = ();

    fn extract_stats(&self, math: &mut M, _opt: Self::StatsOptions) -> Self::Stats {
        Stats {
            tuning: self.tuning,
            step_size: { self.step_size.extract_stats(math, ()) },
            _phantom: std::marker::PhantomData,
        }
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
        _step: &crate::nuts::StepInfo,
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

impl<M: Math> AdaptStrategy<M> for ExternalTransformAdaptation {
    type Hamiltonian = TransformedHamiltonian<M, ExternalTransformation<M>>;

    type Collector =
        CombinedCollector<M, TransformedPoint<M>, AcceptanceRateCollector, DrawCollector<M>>;

    type Options = FlowSettings;

    fn new(_math: &mut M, options: Self::Options, num_tune: u64, chain: u64) -> Self {
        let step_size = StepSizeStrategy::new(options.step_size_settings);
        let final_window_size =
            ((num_tune as f64) * (1f64 - options.step_size_window)).floor() as u64;
        let early_end = ((num_tune as f64) * options.early_window) as u64;
        Self {
            step_size,
            options,
            schedule: WarmupSchedule {
                early_end,
                final_start: final_window_size,
                num_tune,
            },
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
        // The initial step size search should already use the settings of
        // the first draw.
        set_phase_settings(self.schedule.phase(0), &self.options, options, hamiltonian);
        hamiltonian.init_transformation(rng, math, position, self.chain)?;
        self.step_size
            .init(math, options, hamiltonian, position, rng)?;
        Ok(())
    }

    fn adapt<R: rand::Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        draw: u64,
        collector: &Self::Collector,
        _state: &State<M, <Self::Hamiltonian as Hamiltonian<M>>::Point>,
        rng: &mut R,
    ) -> Result<(), NutsError> {
        self.step_size.update(&collector.collector1);

        let phase = self.schedule.phase(draw);
        match phase {
            WarmupPhase::Early | WarmupPhase::Main => {
                self.step_size
                    .update_min_micro_steps(self.options.walnuts.as_ref(), options);
                let update_freq = if phase == WarmupPhase::Early {
                    self.options.early_transform_update_freq
                } else {
                    self.options.transform_update_freq
                };
                if (draw > 0) && draw.is_multiple_of(update_freq) {
                    hamiltonian.update_params(
                        math,
                        rng,
                        collector.collector2.draws.iter(),
                        collector.collector2.grads.iter(),
                        collector.collector2.logps.iter(),
                    )?;
                }
                self.step_size.update_estimator_early();
                // Only the last tuning draw if there is no final step size window.
                let is_last = self.schedule.is_last_tuning_draw(draw);
                self.step_size.update_stepsize(rng, hamiltonian, is_last);
            }
            WarmupPhase::FinalStepSize => {
                self.step_size.update_estimator_late();
                let is_last = self.schedule.is_last_tuning_draw(draw);
                self.step_size.update_stepsize(rng, hamiltonian, is_last);
            }
            WarmupPhase::Sampling => {
                // Needed for step size jitter
                self.step_size.update_stepsize(rng, hamiltonian, true);
                self.tuning = false;
            }
        }
        if self.schedule.is_phase_end(draw) {
            set_phase_settings(
                self.schedule.phase(draw + 1),
                &self.options,
                options,
                hamiltonian,
            );
        }
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

    fn phase(&self, draw: u64) -> WarmupPhase {
        self.schedule.phase(draw)
    }

    fn last_num_steps(&self) -> u64 {
        self.step_size.last_n_steps
    }

    fn last_num_gradients(&self) -> u64 {
        self.step_size.last_n_gradients
    }
}
