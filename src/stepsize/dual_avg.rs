//! Nesterov dual-averaging algorithm for tuning the leapfrog step size toward a target acceptance rate.

use serde::{Deserialize, Serialize};

use crate::{
    dynamics::{DivergenceInfo, Point, State},
    math::Math,
    nuts::{BaseStepOutcome, Collector, NutsOptions, SampleInfo, StepInfo},
};

/// Settings for step size adaptation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DualAverageOptions {
    pub k: f64,
    pub t0: f64,
    pub gamma: f64,
    /// Maximum allowed step size. The dual-averaging update is clamped so that
    /// the step size never exceeds this value. Defaults to π. If the step size
    /// is rescaled during adaptation, the bound is rescaled with it.
    pub max_step_size: f64,
}

impl Default for DualAverageOptions {
    fn default() -> DualAverageOptions {
        DualAverageOptions {
            k: 0.75,
            t0: 10.,
            gamma: 0.05,
            max_step_size: std::f64::consts::PI,
        }
    }
}

#[derive(Clone)]
pub struct DualAverage {
    log_step: f64,
    log_step_adapted: f64,
    hbar: f64,
    mu: f64,
    count: u64,
    /// Log of the product of all factors passed to `rescale`.
    log_scale: f64,
    settings: DualAverageOptions,
}

impl DualAverage {
    pub fn new(settings: DualAverageOptions, initial_step: f64) -> DualAverage {
        DualAverage {
            log_step: initial_step.ln(),
            log_step_adapted: initial_step.ln(),
            hbar: 0.,
            mu: (10. * initial_step).ln(),
            count: 1,
            log_scale: 0.,
            settings,
        }
    }

    pub fn advance(&mut self, accept_stat: f64, target: f64) {
        let w = 1. / (self.count as f64 + self.settings.t0);
        self.hbar = (1. - w) * self.hbar + w * (target - accept_stat);
        self.log_step = self.mu - self.hbar * (self.count as f64).sqrt() / self.settings.gamma;
        self.log_step = self
            .log_step
            .min(self.settings.max_step_size.ln() + self.log_scale);
        let mk = (self.count as f64).powf(-self.settings.k);
        self.log_step_adapted = mk * self.log_step + (1. - mk) * self.log_step_adapted;
        self.count += 1;
    }

    pub fn current_step_size(&self) -> f64 {
        self.log_step.exp()
    }

    pub fn current_step_size_adapted(&self) -> f64 {
        self.log_step_adapted.exp()
    }

    #[allow(dead_code)]
    pub fn reset(&mut self, initial_step: f64, bias_factor: f64) {
        self.log_step = initial_step.ln();
        self.log_step_adapted = initial_step.ln();
        self.hbar = 0f64;
        self.mu = (bias_factor * initial_step).ln();
        self.count = 1;
    }

    pub(crate) fn set_initial_step_size(&mut self, step_size: f64) {
        assert!(step_size > 0.0);
        self.log_step = step_size.ln();
        self.log_step_adapted = step_size.ln();
        self.mu = (10. * step_size).ln();
    }

    /// Multiply the current and future step sizes by `factor`.
    pub(crate) fn rescale(&mut self, factor: f64) {
        assert!(factor > 0.0);
        let shift = factor.ln();
        self.log_step += shift;
        self.log_step_adapted += shift;
        self.mu += shift;
        self.log_scale += shift;
    }
}

pub(crate) struct RunningMean {
    sum: f64,
    count: u64,
}

impl RunningMean {
    fn new() -> RunningMean {
        RunningMean { sum: 0., count: 0 }
    }

    fn add(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
    }

    pub(crate) fn current(&self) -> f64 {
        self.sum / self.count as f64
    }

    pub(crate) fn reset(&mut self) {
        self.sum = 0f64;
        self.count = 0;
    }

    pub(crate) fn count(&self) -> u64 {
        self.count
    }
}

pub struct AcceptanceRateCollector {
    initial_energy: f64,
    pub(crate) mean: RunningMean,
    pub(crate) mean_sym: RunningMean,
    pub(crate) max_energy_error: f64,
    pub(crate) num_gradients: u64,
    /// The depth of the last trajectory.
    pub(crate) depth: u64,
}

impl AcceptanceRateCollector {
    pub(crate) fn new() -> AcceptanceRateCollector {
        AcceptanceRateCollector {
            initial_energy: 0.,
            mean: RunningMean::new(),
            mean_sym: RunningMean::new(),
            max_energy_error: 0.,
            num_gradients: 0,
            depth: 0,
        }
    }

    /// Add the acceptance rate for an energy change of `-diff`, or of a
    /// divergence if `diff` is `None`.
    fn add_accept(&mut self, diff: Option<f64>) {
        match diff {
            Some(diff) => {
                self.mean.add(diff.min(0.).exp());
                self.mean_sym
                    .add(2. * diff.min(0.).exp() / (1. + diff.exp()));
            }
            None => {
                self.mean.add(0.);
                self.mean_sym.add(0.);
            }
        }
    }
}

impl<M: Math, P: Point<M>> Collector<M, P> for AcceptanceRateCollector {
    fn register_leapfrog(
        &mut self,
        _math: &mut M,
        _start: &State<M, P>,
        end: &State<M, P>,
        divergence_info: Option<&DivergenceInfo>,
        step: &StepInfo,
    ) {
        self.num_gradients += step.num_gradients;

        let diff = match divergence_info {
            Some(_) => None,
            None => Some(self.initial_energy - end.energy()),
        };

        match step.base_step {
            // Plain NUTS: acceptance relative to the start of the trajectory.
            BaseStepOutcome::Plain => self.add_accept(diff),
            // WALNUTS: the accepted macro steps are within tolerance by
            // construction, so we use the local energy error of the first
            // attempt at the base step size instead.
            BaseStepOutcome::EnergyError(energy_error) if !energy_error.is_nan() => {
                self.add_accept(Some(-energy_error))
            }
            BaseStepOutcome::EnergyError(_) | BaseStepOutcome::Diverged => self.add_accept(None),
        }

        match diff {
            None => self.max_energy_error = f64::NEG_INFINITY,
            Some(diff) => {
                if diff.abs() > self.max_energy_error.abs() {
                    self.max_energy_error = diff;
                }
            }
        }
    }

    fn register_draw(&mut self, _math: &mut M, _state: &State<M, P>, info: &SampleInfo) {
        self.depth = info.depth;
    }

    fn register_init(&mut self, _math: &mut M, state: &State<M, P>, _options: &NutsOptions) {
        self.initial_energy = state.energy();
        self.mean.reset();
        self.mean_sym.reset();
        self.max_energy_error = 0.;
        self.num_gradients = 0;
    }
}
