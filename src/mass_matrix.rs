use itertools::{izip, Itertools};

use crate::{
    cpu_state::{InnerState, State},
    math::vector_dot,
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
    pub(crate) fn new(variance: Box<[f64]>) -> Self {
        DiagMassMatrix {
            inv_stds: variance.iter().map(|x| 1. / x.sqrt()).collect_vec().into(),
            variance,
        }
    }

    pub(crate) fn update_diag(&mut self, new_variance: impl Iterator<Item = f64>) {
        izip!(
            self.variance.iter_mut(),
            self.inv_stds.iter_mut(),
            new_variance
        )
        .for_each(|(var, inv_std, x)| {
            *var = x;
            *inv_std = (1. / x).sqrt();
        });
    }
}

impl MassMatrix for DiagMassMatrix {
    fn update_velocity(&self, state: &mut InnerState) {
        //axpy_out(&self.variance, &state.p, 1., &mut state.v);
        izip!(state.v.iter_mut(), self.variance.iter(), state.p.iter()).for_each(
            |(out, &var, &p)| {
                *out = var * p;
            },
        );
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
    }

    pub(crate) fn current(&self) -> &[f64] {
        &self.variance
    }
}

#[derive(Clone, Copy)]
pub struct DiagAdaptExpSettings {
    pub variance_decay: f64,
    pub discard_window: u64,
    pub stop_at_draw: u64,
}

impl Default for DiagAdaptExpSettings {
    fn default() -> Self {
        Self {
            variance_decay: 0.05,
            discard_window: 50,
            stop_at_draw: 920,
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
