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

    pub(crate) fn update_diag(&mut self, new_variance: impl Iterator<Item = Option<f64>>) {
        update_diag(&mut self.variance, &mut self.inv_stds, new_variance);
    }
}

#[multiversion]
#[clone(target = "[x64|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse")]
fn update_diag(
    variance_out: &mut [f64],
    inv_std_out: &mut [f64],
    new_variance: impl Iterator<Item = Option<f64>>,
) {
    izip!(variance_out, inv_std_out, new_variance).for_each(|(var, inv_std, x)| {
        if let Some(x) = x {
            assert!(x.is_finite(), "Illegal value on mass matrix: {}", x);
            assert!(x > 0f64, "Illegal value on mass matrix: {}", x);
            //assert!(*var != x, "No change in mass matrix from {} to {}", *var, x);
            *var = x;
            *inv_std = (1. / x).sqrt();
        };
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
pub(crate) struct RunningVariance {
    mean: Box<[f64]>,
    variance: Box<[f64]>,
    count: u64,
}


impl RunningVariance {
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            mean: vec![0f64; dim].into(),
            variance: vec![0f64; dim].into(),
            count: 0,
        }
    }

    pub(crate) fn add_sample(&mut self, value: impl Iterator<Item = f64>) {
        self.count += 1;
        if self.count == 1 {
            izip!(self.mean.iter_mut(), value)
                .for_each(|(mean, val)| {
                    *mean = val;
                });
        } else {
            izip!(self.mean.iter_mut(), self.variance.iter_mut(), value)
                .for_each(|(mean, var, x)| {
                    let diff = x - *mean;
                    *mean += diff / (self.count as f64);
                    *var += diff * diff;
                });
        }
    }

    pub(crate) fn current(&self) -> impl Iterator<Item = f64> + '_ {
        assert!(self.count > 1);
        self.variance.iter().map(|&x| x / ((self.count - 1) as f64))
    }

    pub(crate) fn count(&self) -> u64 {
        self.count
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

    fn register_draw(&mut self, state: &Self::State, _info: &crate::nuts::SampleInfo) {
        self.draw.copy_from_slice(&state.q);
        self.grad.copy_from_slice(&state.grad);
        let idx = state.index_in_trajectory();
        self.is_good = _info.divergence_info.is_none() & (idx != 0);
    }
}
