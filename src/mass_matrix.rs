use itertools::izip;
use multiversion::multiversion;

use crate::{
    cpu_state::{InnerState, State},
    math::{multiply, vector_dot},
    nuts::Collector,
};

pub(crate) trait MassMatrix {
    fn update_velocity(&self, state: &mut InnerState);
    fn update_kinetic_energy(&self, state: &mut InnerState);
    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut InnerState, rng: &mut R);
}

pub(crate) struct NullCollector {}

impl Collector for NullCollector {
    type State = State;
}

#[derive(Debug)]
pub(crate) struct DiagMassMatrix {
    inv_stds: Box<[f64]>,
    pub(crate) variance: Box<[f64]>,
}

impl DiagMassMatrix {
    pub(crate) fn new(ndim: usize) -> Self {
        Self {
            inv_stds: vec![0f64; ndim].into(),
            variance: vec![0f64; ndim].into(),
        }
    }

    pub(crate) fn update_diag(&mut self, new_variance: impl Iterator<Item = f64>) {
        update_diag(&mut self.variance, &mut self.inv_stds, new_variance);
    }
}

#[multiversion]
#[clone(target = "[x64|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse")]
fn update_diag(
    variance_out: &mut [f64],
    inv_std_out: &mut [f64],
    new_variance: impl Iterator<Item = f64>,
) {
    izip!(variance_out, inv_std_out, new_variance,).for_each(|(var, inv_std, x)| {
        assert!(x.is_finite(), "Illegal value on mass matrix: {}", x);
        assert!(x > 0f64, "Illegal value on mass matrix: {}", x);
        *var = x;
        *inv_std = (1. / x).sqrt();
    });
}

impl MassMatrix for DiagMassMatrix {
    fn update_velocity(&self, state: &mut InnerState) {
        multiply(&self.variance, &state.p, &mut state.v);
    }

    fn update_kinetic_energy(&self, state: &mut InnerState) {
        state.kinetic_energy = 0.5 * vector_dot(&state.p, &state.v);
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut InnerState, rng: &mut R) {
        let dist = rand_distr::StandardNormal;
        state
            .p
            .iter_mut()
            .zip(self.inv_stds.iter())
            .for_each(|(p, &s)| {
                let norm: f64 = rng.sample(dist);
                *p = s * norm;
            });
    }
}


#[derive(Debug)]
pub(crate) struct ExpWeightedVariance {
    mean: Box<[f64]>,
    variance: Box<[f64]>,
    count: u64,
    pub(crate) alpha: f64, // TODO
    pub(crate) use_mean: bool,
}

impl ExpWeightedVariance {
    pub(crate) fn new(dim: usize, alpha: f64, use_mean: bool) -> Self {
        ExpWeightedVariance {
            mean: vec![0f64; dim].into(),
            variance: vec![0f64; dim].into(),
            count: 0,
            alpha,
            use_mean,
        }
    }

    pub(crate) fn set_mean(&mut self, values: impl Iterator<Item = f64>) {
        self.mean
            .iter_mut()
            .zip(values)
            .for_each(|(out, val)| *out = val);
    }

    pub(crate) fn set_variance(&mut self, values: impl Iterator<Item = f64>) {
        self.variance
            .iter_mut()
            .zip(values)
            .for_each(|(out, val)| *out = val);
    }

    pub(crate) fn add_sample(&mut self, value: impl Iterator<Item = f64>) {
        add_sample(self, value);
        self.count += 1;
    }

    pub(crate) fn current(&self) -> &[f64] {
        &self.variance
    }

    pub(crate) fn count(&self) -> u64 {
        self.count
    }
}

#[multiversion]
#[clone(target = "[x64|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse")]
fn add_sample(self_: &mut ExpWeightedVariance, value: impl Iterator<Item = f64>) {
    if self_.use_mean {
        izip!(value, self_.mean.iter_mut(), self_.variance.iter_mut()).for_each(
            |(x, mean, var)| {
                //if self_.count > 1 {
                //    assert!(x - *mean != 0f64, "var = {}, mean = {}, x = {}, delta = {}, count = {}", var, mean, x, x - *mean, self_.count);
                //}
                let delta = x - *mean;
                //*mean += self_.alpha * delta;
                *mean = self_.alpha.mul_add(delta, *mean);
                *var = (1f64 - self_.alpha) * (*var + self_.alpha * delta * delta);
            },
        );
    } else {
        izip!(value, self_.mean.iter_mut(), self_.variance.iter_mut()).for_each(
            |(x, _mean, var)| {
                let delta = x;
                *var = (1f64 - self_.alpha) * (*var + self_.alpha * delta * delta);
                //assert!(*var > 0f64, "var = {}, x = {}, delta = {}", var, x, delta);
            },
        );
    }
}

/// Settings for mass matrix adaptation
#[derive(Clone, Copy)]
pub struct DiagAdaptExpSettings {
    /// An exponenital decay parameter for the variance estimator
    pub variance_decay: f64,
    /// Exponenital decay parameter for the variance estimator in the first adaptation window
    pub early_variance_decay: f64,
    /// Stop adaptation `final_window` draws before tuning ends.
    pub final_window: u64,
    /// Save the current adapted mass matrix as sampler stat
    pub store_mass_matrix: bool,
    /// Switch to a new variance estimator every `window_switch_freq` draws.
    pub window_switch_freq: u64,
    pub grad_init: bool,
}

impl Default for DiagAdaptExpSettings {
    fn default() -> Self {
        Self {
            variance_decay: 0.02,
            final_window: 50,
            store_mass_matrix: false,
            window_switch_freq: 50,
            early_variance_decay: 0.8,
            grad_init: true,
        }
    }
}

pub(crate) struct DrawGradCollector {
    pub(crate) draw: Box<[f64]>,
    pub(crate) grad: Box<[f64]>,
    pub(crate) is_good: bool,
}

impl DrawGradCollector {
    pub(crate) fn new(dim: usize) -> Self {
        DrawGradCollector {
            draw: vec![0f64; dim].into(),
            grad: vec![0f64; dim].into(),
            is_good: true,
        }
    }
}

impl Collector for DrawGradCollector {
    type State = State;

    fn register_draw(&mut self, state: &Self::State, info: &crate::nuts::SampleInfo) {
        self.draw.copy_from_slice(&state.q);
        self.grad.copy_from_slice(&state.grad);
        let idx = state.index_in_trajectory();
        if let Some(_) = info.divergence_info {
            self.is_good = (idx <= -4) | (idx >= 4);
        } else {
            self.is_good = idx != 0;
        }
    }
}
