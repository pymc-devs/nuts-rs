use itertools::izip;
use multiversion::multiversion;

use crate::{
    cpu_state::{InnerState, State, AlignedArray},
    math::{vector_dot, multiply},
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
    //inv_stds: Box<[f64]>,
    inv_stds: AlignedArray,
    //pub(crate) variance: Box<[f64]>,
    pub(crate) variance: AlignedArray,
}

impl DiagMassMatrix {
    pub(crate) fn new(variance: Box<[f64]>) -> Self {
        let variance_aligned = AlignedArray::new(variance.len());
        let inv_stds_aligned = AlignedArray::new(variance.len());
        let mut out = Self {
            inv_stds: inv_stds_aligned,
            variance: variance_aligned,
        };
        out.update_diag(variance.iter().copied());
        out
    }

    pub(crate) fn update_diag(&mut self, new_variance: impl Iterator<Item = f64>) {
        update_diag(&mut self.variance, &mut self.inv_stds, new_variance);
        /*
        izip!(
            self.variance.iter_mut(),
            self.inv_stds.iter_mut(),
            new_variance
        )
        .for_each(|(var, inv_std, x)| {
            *var = x;
            *inv_std = (1. / x).sqrt();
        });
        */
    }
}

#[multiversion]
#[clone(target = "[x64|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse")]
fn update_diag(variance_out: &mut [f64], inv_std_out: &mut [f64], new_variance: impl Iterator<Item = f64>) {
    izip!(
        variance_out.iter_mut(),
        inv_std_out.iter_mut(),
        new_variance
    )
    .for_each(|(var, inv_std, x)| {
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
    alpha: f64,
    use_mean: bool,
}

impl ExpWeightedVariance {
    pub(crate) fn new(dim: usize, alpha: f64, use_mean: bool) -> Self {
        ExpWeightedVariance {
            mean: vec![0f64; dim].into(),
            variance: vec![0f64; dim].into(),
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

    pub(crate) fn add_sample(&mut self, value: impl Iterator<Item = f64>) {
        add_sample(self, value);
        /*
        izip!(value, self.mean.iter_mut(), self.variance.iter_mut()).for_each(|(x, mean, var)| {
            let delta = if self.use_mean {
                let delta = x - *mean;
                *mean += self.alpha * delta;
                delta
            } else {
                x
            };
            *var = (1. - self.alpha) * (*var + self.alpha * delta * delta);
        });
        */
    }

    pub(crate) fn current(&self) -> &[f64] {
        &self.variance
    }
}

#[multiversion]
#[clone(target = "[x64|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse")]
fn add_sample(self_: &mut ExpWeightedVariance, value: impl Iterator<Item = f64>) {
    if self_.use_mean {
        izip!(value, self_.mean.iter_mut(), self_.variance.iter_mut()).for_each(|(x, mean, var)| {
            let delta = x - *mean;
            *mean += self_.alpha * delta;
            *var = (1. - self_.alpha) * (*var + self_.alpha * delta * delta);
        });
    }
    else {
        izip!(value, self_.mean.iter_mut(), self_.variance.iter_mut()).for_each(|(x, _mean, var)| {
            let delta = x;
            *var = (1. - self_.alpha) * (*var + self_.alpha * delta * delta);
        });
    }
}

/// Settings for mass matrix adaptation
#[derive(Clone, Copy)]
pub struct DiagAdaptExpSettings {
    /// An exponenital decay parameter for the variance estimator
    pub variance_decay: f64,
    /// The number of initial samples during which no mass matrix adaptation occurs.
    pub discard_window: u64,
    /// Stop adaptation ofter stop_at_draw draws. Should be smaller that `num_tune`.
    pub stop_at_draw: u64,
    /// Save the current adapted mass matrix as sampler stat
    pub save_mass_matrix: bool,
}

impl Default for DiagAdaptExpSettings {
    fn default() -> Self {
        Self {
            variance_decay: 0.05,
            discard_window: 50,
            stop_at_draw: 920,
            save_mass_matrix: false,
        }
    }
}

pub(crate) struct DrawGradCollector {
    pub(crate) draw: Box<[f64]>,
    pub(crate) grad: Box<[f64]>,
}

impl DrawGradCollector {
    pub(crate) fn new(dim: usize) -> Self {
        DrawGradCollector {
            draw: vec![0f64; dim].into(),
            grad: vec![0f64; dim].into(),
        }
    }
}

impl Collector for DrawGradCollector {
    type State = State;

    fn register_draw(&mut self, state: &Self::State, info: &crate::nuts::SampleInfo) {
        if info.divergence_info.is_none() {
            self.draw.copy_from_slice(&state.q);
            self.grad.copy_from_slice(&state.grad);
        }
    }
}
