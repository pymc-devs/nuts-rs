use std::marker::PhantomData;

use crate::nuts::{Collector, NutsOptions, State};

/// Settings for step size adaptation
#[derive(Clone, Copy)]
pub struct DualAverageSettings {
    /// The target acceptance rate. Must be between 0 and 1, usually >= 0.6.
    /// The large the target acceptance rate the smaller the steps will be
    /// that each leapfrog step takes.
    pub target: f64,
    pub k: f64,
    pub t0: f64,
    pub gamma: f64,
    pub initial_step: f64,
}

impl Default for DualAverageSettings {
    fn default() -> DualAverageSettings {
        DualAverageSettings {
            target: 0.8,
            k: 0.75,
            t0: 10.,
            gamma: 0.05,
            initial_step: 0.1,
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
    settings: DualAverageSettings,
}

impl DualAverage {
    pub fn new(settings: DualAverageSettings) -> DualAverage {
        let initial_step = settings.initial_step;
        DualAverage {
            log_step: initial_step.ln(),
            log_step_adapted: initial_step.ln(),
            hbar: 0.,
            mu: (10. * initial_step).ln(),
            count: 1,
            settings,
        }
    }

    pub fn advance(&mut self, accept_stat: f64) {
        let w = 1. / (self.count as f64 + self.settings.t0);
        self.hbar = (1. - w) * self.hbar + w * (self.settings.target - accept_stat);
        self.log_step = self.mu - self.hbar * (self.count as f64).sqrt() / self.settings.gamma;
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
    pub fn reset(&mut self, initial_step: f64) {
        self.log_step = initial_step.ln();
        self.log_step_adapted = initial_step.ln();
        self.hbar = 0f64;
        self.mu = (10f64 * initial_step).ln();
        self.count = 1;
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
}

pub(crate) struct AcceptanceRateCollector<S: State> {
    initial_energy: f64,
    pub(crate) mean: RunningMean,
    phantom: PhantomData<S>,
}

impl<S: State> AcceptanceRateCollector<S> {
    pub(crate) fn new() -> AcceptanceRateCollector<S> {
        AcceptanceRateCollector {
            initial_energy: 0.,
            mean: RunningMean::new(),
            phantom: PhantomData::default(),
        }
    }
}

impl<S: State> Collector for AcceptanceRateCollector<S> {
    type State = S;

    fn register_leapfrog(
        &mut self,
        _start: &Self::State,
        end: &Self::State,
        divergence_info: Option<&dyn crate::nuts::DivergenceInfo>,
    ) {
        match divergence_info {
            Some(_) => self.mean.add(0.),
            None => self
                .mean
                .add(end.log_acceptance_probability(self.initial_energy).exp()),
        }
    }

    fn register_init(&mut self, state: &Self::State, _options: &NutsOptions) {
        self.initial_energy = state.energy();
        self.mean.reset();
    }
}
