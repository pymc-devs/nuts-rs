use itertools::izip;
use multiversion::multiversion;

use crate::{
    math_base::Math,
    nuts::Collector,
    state::{InnerState, State},
};

pub(crate) trait MassMatrix<M: Math> {
    fn update_velocity(&self, math: &mut M, state: &mut InnerState<M>);
    fn update_kinetic_energy(&self, math: &mut M, state: &mut InnerState<M>);
    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut InnerState<M>,
        rng: &mut R,
    );
}

pub(crate) struct NullCollector {}

impl<M: Math> Collector<M> for NullCollector {}

#[derive(Debug)]
pub(crate) struct DiagMassMatrix<M: Math> {
    inv_stds: M::Array,
    pub(crate) variance: M::Array,
}

impl<M: Math> DiagMassMatrix<M> {
    pub(crate) fn new(math: &mut M) -> Self {
        Self {
            inv_stds: math.new_array(),
            variance: math.new_array(),
        }
    }

    pub(crate) fn update_diag_draw_grad(
        &mut self,
        math: &mut M,
        draw_var: &M::Array,
        grad_var: &M::Array,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_draw_grad(
            &mut self.variance,
            &mut self.inv_stds,
            draw_var,
            grad_var,
            fill_invalid,
            clamp,
        );
    }

    pub(crate) fn update_diag_grad(
        &mut self,
        math: &mut M,
        gradient: &M::Array,
        fill_invalid: f64,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_grad(
            &mut self.variance,
            &mut self.inv_stds,
            gradient,
            fill_invalid,
            clamp,
        );
    }
}

#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
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

impl<M: Math> MassMatrix<M> for DiagMassMatrix<M> {
    fn update_velocity(&self, math: &mut M, state: &mut InnerState<M>) {
        math.array_mult(&self.variance, &state.p, &mut state.v);
    }

    fn update_kinetic_energy(&self, math: &mut M, state: &mut InnerState<M>) {
        state.kinetic_energy = 0.5 * math.array_vector_dot(&state.p, &state.v);
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut InnerState<M>,
        rng: &mut R,
    ) {
        math.array_gaussian(rng, &mut state.p, &self.inv_stds);
    }
}

#[derive(Debug)]
pub(crate) struct RunningVariance<M: Math> {
    mean: M::Array,
    variance: M::Array,
    count: u64,
}

impl<M: Math> RunningVariance<M> {
    pub(crate) fn new(math: &mut M) -> Self {
        Self {
            mean: math.new_array(),
            variance: math.new_array(),
            count: 0,
        }
    }

    //pub(crate) fn add_sample(&mut self, value: impl Iterator<Item = f64>) {
    pub(crate) fn add_sample(&mut self, math: &mut M, value: &M::Array) {
        self.count += 1;
        if self.count == 1 {
            math.copy_into(value, &mut self.mean);
        } else {
            math.array_update_variance(&mut self.mean, &mut self.variance, value, (self.count as f64).recip());
        }
    }

    /// Return current variance and scaling factor
    pub(crate) fn current(&self) -> (&M::Array, f64) {
        assert!(self.count > 1);
        (&self.variance, ((self.count - 1) as f64).recip())
    }

    pub(crate) fn count(&self) -> u64 {
        self.count
    }
}

pub(crate) struct DrawGradCollector<M: Math> {
    pub(crate) draw: M::Array,
    pub(crate) grad: M::Array,
    pub(crate) is_good: bool,
}

impl<M: Math> DrawGradCollector<M> {
    pub(crate) fn new(math: &mut M) -> Self {
        DrawGradCollector {
            draw: math.new_array(),
            grad: math.new_array(),
            is_good: true,
        }
    }
}

impl<M: Math> Collector<M> for DrawGradCollector<M> {
    fn register_draw(&mut self, math: &mut M, state: &State<M>, info: &crate::nuts::SampleInfo) {
        math.copy_into(&state.q, &mut self.draw);
        math.copy_into(&state.grad, &mut self.grad);
        let idx = state.index_in_trajectory();
        if info.divergence_info.is_some() {
            self.is_good = idx.abs() > 4;
        } else {
            self.is_good = idx != 0;
        }
    }
}
