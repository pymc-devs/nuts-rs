use nuts_derive::Storable;

use crate::{
    euclidean_hamiltonian::EuclideanPoint, hamiltonian::Point, math_base::Math, nuts::Collector,
    sampler_stats::SamplerStats, state::State,
};

pub trait MassMatrix<M: Math>: SamplerStats<M> {
    fn update_velocity(&self, math: &mut M, state: &mut EuclideanPoint<M>);
    fn update_kinetic_energy(&self, math: &mut M, state: &mut EuclideanPoint<M>);
    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        point: &mut EuclideanPoint<M>,
        rng: &mut R,
    );
}

#[derive(Debug)]
pub struct DiagMassMatrix<M: Math> {
    inv_stds: M::Vector,
    pub(crate) variance: M::Vector,
    store_mass_matrix: bool,
}

#[derive(Debug, Storable)]
pub struct DiagMassMatrixStats {
    #[storable(dims("unconstrained_parameter"))]
    pub mass_matrix_inv: Option<Vec<f64>>,
}

impl<M: Math> SamplerStats<M> for DiagMassMatrix<M> {
    type Stats = DiagMassMatrixStats;
    type StatsOptions = ();

    fn extract_stats(&self, math: &mut M, _opt: Self::StatsOptions) -> Self::Stats {
        if self.store_mass_matrix {
            DiagMassMatrixStats {
                mass_matrix_inv: Some(math.box_array(&self.variance).into_vec()),
            }
        } else {
            DiagMassMatrixStats {
                mass_matrix_inv: None,
            }
        }
    }
}

impl<M: Math> DiagMassMatrix<M> {
    pub(crate) fn new(math: &mut M, store_mass_matrix: bool) -> Self {
        Self {
            inv_stds: math.new_array(),
            variance: math.new_array(),
            store_mass_matrix,
        }
    }

    pub(crate) fn update_diag_draw(
        &mut self,
        math: &mut M,
        draw_var: &M::Vector,
        scale: f64,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_draw(
            &mut self.variance,
            &mut self.inv_stds,
            draw_var,
            scale,
            fill_invalid,
            clamp,
        );
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
    fn update_velocity(&self, math: &mut M, point: &mut EuclideanPoint<M>) {
        math.array_mult(&self.variance, &point.momentum, &mut point.velocity);
    }

    fn update_kinetic_energy(&self, math: &mut M, point: &mut EuclideanPoint<M>) {
        point.kinetic_energy = 0.5 * math.array_vector_dot(&point.momentum, &point.velocity);
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        point: &mut EuclideanPoint<M>,
        rng: &mut R,
    ) {
        math.array_gaussian(rng, &mut point.momentum, &self.inv_stds);
    }
}

#[derive(Debug)]
pub struct RunningVariance<M: Math> {
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

    pub(crate) fn add_sample(&mut self, math: &mut M, value: &M::Vector) {
        self.count += 1;
        if self.count == 1 {
            math.copy_into(value, &mut self.mean);
        } else {
            math.array_update_variance(
                &mut self.mean,
                &mut self.variance,
                value,
                (self.count as f64).recip(),
            );
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

pub struct DrawGradCollector<M: Math> {
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

impl<M: Math> Collector<M, EuclideanPoint<M>> for DrawGradCollector<M> {
    fn register_draw(
        &mut self,
        math: &mut M,
        state: &State<M, EuclideanPoint<M>>,
        info: &crate::nuts::SampleInfo,
    ) {
        math.copy_into(state.point().position(), &mut self.draw);
        math.copy_into(state.point().gradient(), &mut self.grad);
        let idx = state.index_in_trajectory();
        if info.divergence_info.is_some() {
            self.is_good = idx.abs() > 4;
        } else {
            self.is_good = idx != 0;
        }
    }
}
