pub struct DualAverageSettings {
    pub target: f64,
    pub k: f64,
    pub t0: f64,
    pub gamma: f64,
}

impl Default for DualAverageSettings {
    fn default() -> DualAverageSettings {
        DualAverageSettings {
            target: 0.8,
            k: 0.75,
            t0: 10.,
            gamma: 0.05,
        }
    }
}

pub struct DualAverage {
    log_step: f64,
    log_step_adapt: f64,
    hbar: f64,
    mu: f64,
    count: u64,
    settings: DualAverageSettings,
}

pub struct StepSizeStats {
    pub step_size: f64,
    pub step_size_adapt: f64,
}

impl DualAverage {
    pub fn new(settings: DualAverageSettings, initial_step: f64) -> DualAverage {
        DualAverage {
            log_step: initial_step.ln(),
            log_step_adapt: initial_step.ln(),
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
        self.log_step_adapt = mk * self.log_step + (1. - mk) * self.log_step_adapt;
    }

    pub fn current_step_size(&self) -> f64 {
        self.log_step.exp()
    }

    pub fn current_step_size_adapt(&self) -> f64 {
        self.log_step_adapt.exp()
    }

    pub fn stats(&self) -> StepSizeStats {
        StepSizeStats {
            step_size: self.current_step_size(),
            step_size_adapt: self.current_step_size_adapt(),
        }
    }
}
