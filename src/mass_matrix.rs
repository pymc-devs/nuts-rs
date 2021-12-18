use itertools::izip;

use crate::{
    cpu_potential::{DiagMassMatrix, MassMatrix},
    cpu_state::State,
    nuts::Collector,
};

#[derive(Debug)]
struct ExpWeightedVariance {
    mean: Box<[f64]>,
    variance: Box<[f64]>,
    alpha: f64,
    use_mean: bool,
}

impl ExpWeightedVariance {
    fn new(dim: usize, alpha: f64, use_mean: bool) -> Self {
        ExpWeightedVariance {
            mean: vec![0f64; dim].into(),
            variance: vec![0f64; dim].into(),
            alpha,
            use_mean,
        }
    }

    fn set_mean(&mut self, values: impl Iterator<Item = f64>) {
        self.mean.iter_mut().zip(values).for_each(|(out, val)| *out = val);
    }

    fn add_sample(&mut self, value: impl Iterator<Item = f64>) {
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

    fn current(&self) -> &[f64] {
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

pub(crate) struct DiagAdaptExp {
    exp_variance_draw: ExpWeightedVariance,
    exp_variance_grad: ExpWeightedVariance,
    pub(crate) current: DiagMassMatrix,
    draw_count: u64,
    settings: DiagAdaptExpSettings,
}

impl DiagAdaptExp {
    pub(crate) fn new(dim: usize, settings: DiagAdaptExpSettings) -> Self {
        DiagAdaptExp {
            exp_variance_draw: ExpWeightedVariance::new(dim, settings.variance_decay, true),
            exp_variance_grad: ExpWeightedVariance::new(dim, settings.variance_decay, false),
            current: DiagMassMatrix::new(vec![1f64; dim].into()),
            draw_count: 0,
            settings,
        }
    }

    pub(crate) fn adapt(&mut self, collector: &AdaptCollector) -> bool {
        if self.draw_count < self.settings.discard_window {
            self.draw_count += 1;
            return false;
        }

        if self.draw_count > self.settings.stop_at_draw {
            self.draw_count += 1;
            return false;
        }

        if self.draw_count == self.settings.discard_window {
            self.exp_variance_draw.set_mean(collector.draw.iter().copied());
        }

        self.exp_variance_draw
            .add_sample(collector.draw.iter().copied());
        self.exp_variance_grad
            .add_sample(collector.grad.iter().copied());

        if self.draw_count > 2 * self.settings.discard_window {
            self.current.update_diag(
                izip!(
                    self.exp_variance_draw.current(),
                    self.exp_variance_grad.current(),
                )
                .map(|(&draw, &grad)| (draw / grad).sqrt().clamp(1e-12, 1e10)),
            );
        }

        self.draw_count += 1;
        (self.draw_count - 1) == 2 * self.settings.discard_window
    }
}

impl MassMatrix for DiagAdaptExp {
    fn update_velocity(&self, state: &mut crate::cpu_state::InnerState) {
        self.current.update_velocity(state)
    }

    fn update_kinetic_energy(&self, state: &mut crate::cpu_state::InnerState) {
        self.current.update_kinetic_energy(state)
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        state: &mut crate::cpu_state::InnerState,
        rng: &mut R,
    ) {
        self.current.randomize_momentum(state, rng)
    }
}

pub(crate) struct AdaptCollector {
    draw: Box<[f64]>,
    grad: Box<[f64]>,
}

impl AdaptCollector {
    pub(crate) fn new(dim: usize) -> Self {
        AdaptCollector {
            draw: vec![0f64; dim].into(),
            grad: vec![0f64; dim].into(),
        }
    }
}

impl Collector for AdaptCollector {
    type State = State;
    fn register_draw(&mut self, state: &Self::State, info: &crate::nuts::SampleInfo) {
        if info.divergence_info.is_none() {
            self.draw.copy_from_slice(&state.q);
            self.grad.copy_from_slice(&state.grad);
        }
    }
}
