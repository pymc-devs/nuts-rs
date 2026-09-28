//! Adaptation of the number of WALNUTS micro steps per macro step.

use nuts_derive::Storable;

use crate::nuts::NutsOptions;

/// Adapts the WALNUTS `min_micro_steps` during tuning, as in walnutpie.
///
/// We keep a running mean of the number of macro steps `2^depth` of the
/// trajectories, starting with a pseudo-observation of 2, and set
/// `min_micro_steps = round(mean / target_macro_steps)`, clamped to
/// `[min_micro_steps, max_min_micro_steps]` of the settings.
///
/// Despite the name, this does not make the trajectories have
/// `target_macro_steps` macro steps on average. The step size adaptation
/// keeps the micro step size roughly constant, so a trajectory has about
/// `n1 / min_micro_steps` macro steps, where `n1` is the number with
/// `min_micro_steps = 1`. The fixed point is then roughly
/// `min_micro_steps = sqrt(n1 / target)` with `sqrt(n1 * target)` macro
/// steps. Also, the mean includes the early tuning draws.
///
/// Unlike walnutpie, the step size strategy multiplies the step size by the
/// same factor whenever `min_micro_steps` changes, so that the micro step
/// size stays the same.
#[derive(Debug)]
pub(crate) struct MinMicroStepsAdaptation {
    lower_bound: Option<u64>,
    sum: f64,
    count: f64,
    last_unrounded: Option<f64>,
}

impl Default for MinMicroStepsAdaptation {
    fn default() -> Self {
        Self {
            lower_bound: None,
            sum: 2.0,
            count: 1.0,
            last_unrounded: None,
        }
    }
}

/// Sampler stats of [`MinMicroStepsAdaptation`]. `None` if it is disabled.
#[derive(Debug, Storable)]
pub struct MinMicroStepsStats {
    /// Running mean of the number of macro steps `2^depth` that the
    /// adaptation uses, including the pseudo-observation. Frozen after the
    /// adaptation ends.
    pub macro_steps_mean: Option<f64>,
    /// `macro_steps_mean / target_macro_steps`, before rounding and clamping
    /// to get `min_micro_steps`.
    pub min_micro_steps_unrounded: Option<f64>,
}

impl MinMicroStepsAdaptation {
    /// Observe the depth of the last trajectory and update `min_micro_steps`
    /// in `options`. Returns the factor by which it changed, if it did.
    /// Does nothing if the adaptation is disabled.
    pub(crate) fn update(&mut self, options: &mut NutsOptions, depth: u64) -> Option<f64> {
        let walnuts = options.walnuts.as_mut()?;
        let target = walnuts.target_macro_steps?;
        let current = walnuts.min_micro_steps;
        let lower_bound = *self.lower_bound.get_or_insert(current);

        self.sum += 2f64.powi(depth as i32);
        self.count += 1.0;
        let unrounded = self.sum / self.count / target;
        self.last_unrounded = Some(unrounded);
        let new = (unrounded.round() as u64).clamp(lower_bound, walnuts.max_min_micro_steps);
        if new == current {
            return None;
        }
        walnuts.min_micro_steps = new;
        Some(new as f64 / current as f64)
    }

    pub(crate) fn stats(&self) -> MinMicroStepsStats {
        MinMicroStepsStats {
            macro_steps_mean: self.last_unrounded.map(|_| self.sum / self.count),
            min_micro_steps_unrounded: self.last_unrounded,
        }
    }
}
