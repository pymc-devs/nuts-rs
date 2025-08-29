//! Adam optimizer for step size adaptation.
//!
//! This implements a single-parameter version of the Adam optimizer
//! for adapting the step size in the NUTS algorithm. Unlike dual averaging,
//! Adam maintains both first and second moment estimates of gradients,
//! which can potentially lead to better adaptation in some scenarios.

use std::f64;

/// Settings for Adam step size adaptation
#[derive(Debug, Clone, Copy)]
pub struct AdamOptions {
    /// First moment decay rate (default: 0.9)
    pub beta1: f64,
    /// Second moment decay rate (default: 0.999)
    pub beta2: f64,
    /// Small constant for numerical stability (default: 1e-8)
    pub epsilon: f64,
    /// Learning rate (default: 0.001)
    pub learning_rate: f64,
}

impl Default for AdamOptions {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            learning_rate: 0.05,
        }
    }
}

/// Adam optimizer for step size adaptation.
///
/// This implements the Adam optimizer for a single parameter (the step size).
/// The adaptation takes the acceptance probability statistic and adjusts
/// the step size to reach the target acceptance rate.
#[derive(Clone)]
pub struct Adam {
    /// Current log step size
    log_step: f64,
    /// First moment estimate
    m: f64,
    /// Second moment estimate
    v: f64,
    /// Iteration counter
    t: u64,
    /// Adam settings
    settings: AdamOptions,
}

impl Adam {
    /// Create a new Adam optimizer with given settings and initial step size
    pub fn new(settings: AdamOptions, initial_step: f64) -> Self {
        Self {
            log_step: initial_step.ln(),
            m: 0.0,
            v: 0.0,
            t: 0,
            settings,
        }
    }

    /// Advance the optimizer by one step using the current acceptance statistic
    ///
    /// This updates the step size to move towards the target acceptance rate.
    /// The error signal is the difference between the target and current acceptance rates.
    pub fn advance(&mut self, accept_stat: f64, target: f64) {
        // Compute the error/gradient - we want to minimize (target - accept_stat)²
        // So gradient is -2 * (target - accept_stat)
        // We simplify and just use (accept_stat - target) as our gradient
        let gradient = accept_stat - target;

        // Increment timestep
        self.t += 1;

        // Update biased first moment estimate
        self.m = self.settings.beta1 * self.m + (1.0 - self.settings.beta1) * gradient;

        // Update biased second moment estimate
        self.v = self.settings.beta2 * self.v + (1.0 - self.settings.beta2) * gradient * gradient;

        // Compute bias-corrected first moment estimate
        let m_hat = self.m / (1.0 - self.settings.beta1.powi(self.t as i32));

        // Compute bias-corrected second moment estimate
        let v_hat = self.v / (1.0 - self.settings.beta2.powi(self.t as i32));

        // Update log step size
        // Note: if gradient is positive (accept_stat > target), we should decrease step size
        // if gradient is negative (accept_stat < target), we should increase step size
        self.log_step +=
            self.settings.learning_rate * m_hat / (v_hat.sqrt() + self.settings.epsilon);
    }

    /// Get the current step size (not adapted)
    pub fn current_step_size(&self) -> f64 {
        self.log_step.exp()
    }

    /// Reset the optimizer with a new initial step size and bias factor
    #[allow(dead_code)]
    pub fn reset(&mut self, initial_step: f64, _bias_factor: f64) {
        self.log_step = initial_step.ln();
        self.m = 0.0;
        self.v = 0.0;
        self.t = 0;
    }
}
