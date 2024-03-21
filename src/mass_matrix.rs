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
    inv_stds: M::Vector,
    pub(crate) variance: M::Vector,
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
        draw_var: &M::Vector,
        grad_var: &M::Vector,
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
        gradient: &M::Vector,
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
    mean: M::Vector,
    variance: M::Vector,
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
    pub(crate) fn add_sample(&mut self, math: &mut M, value: &M::Vector) {
        self.count += 1;
        if self.count == 1 {
            math.copy_into(value, &mut self.mean);
        } else {
            math.array_update_variance(&mut self.mean, &mut self.variance, value, (self.count as f64).recip());
        }
    }

    /// Return current variance and scaling factor
    pub(crate) fn current(&self) -> (&M::Vector, f64) {
        assert!(self.count > 1);
        (&self.variance, ((self.count - 1) as f64).recip())
    }

    pub(crate) fn count(&self) -> u64 {
        self.count
    }
}

pub(crate) struct DrawGradCollector<M: Math> {
    pub(crate) draw: M::Vector,
    pub(crate) grad: M::Vector,
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
