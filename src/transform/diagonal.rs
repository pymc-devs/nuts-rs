//! Diagonal (per-dimension) affine transformation estimated from draw and gradient variance.

use std::fmt::Debug;

use nuts_derive::Storable;

use crate::{math::Math, sampler_stats::SamplerStats, transform::Transformation};

pub struct DiagMassMatrix<M: Math> {
    mean: M::Vector,
    inv_stds: M::Vector,
    stds: M::Vector,
    logdet: f64,
    store_mass_matrix: bool,
    /// Monotonically increasing id; bumped whenever the matrix changes.
    id: i64,
}

impl<M: Math> Debug for DiagMassMatrix<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiagMassMatrix")
            .field("mean", &self.mean)
            .field("inv_stds", &self.inv_stds)
            .field("stds", &self.stds)
            .field("logdet", &self.logdet)
            .field("store_mass_matrix", &self.store_mass_matrix)
            .field("id", &self.id)
            .finish()
    }
}

#[derive(Debug, Storable)]
pub struct DiagMassMatrixStats {
    /// The transformation version counter at the time of this update.
    /// `Some` only on draws where the transformation changed.
    #[storable(event = "transformation_update")]
    pub transformation_update_id: Option<i64>,
    #[storable(event = "transformation_update", dims("unconstrained_parameter"))]
    pub mass_matrix_inv: Option<Vec<f64>>,
    #[storable(event = "transformation_update", dims("unconstrained_parameter"))]
    pub transformation_mu: Option<Vec<f64>>,
}

impl<M: Math> SamplerStats<M> for DiagMassMatrix<M> {
    type Stats = DiagMassMatrixStats;
    type StatsOptions = i64;

    fn extract_stats(&self, math: &mut M, last_id: Self::StatsOptions) -> Self::Stats {
        if self.id != last_id {
            DiagMassMatrixStats {
                transformation_update_id: Some(self.id),
                mass_matrix_inv: if self.store_mass_matrix {
                    Some(math.box_array(&self.stds).into_vec())
                } else {
                    None
                },
                transformation_mu: if self.store_mass_matrix {
                    Some(math.box_array(&self.mean).into_vec())
                } else {
                    None
                },
            }
        } else {
            DiagMassMatrixStats {
                transformation_update_id: None,
                mass_matrix_inv: None,
                transformation_mu: None,
            }
        }
    }
}

impl<M: Math> DiagMassMatrix<M> {
    pub(crate) fn new(math: &mut M, store_mass_matrix: bool) -> Self {
        Self {
            mean: math.new_array(),
            inv_stds: math.new_array(),
            stds: math.new_array(),
            logdet: 0f64,
            store_mass_matrix,
            id: -1,
        }
    }

    pub(crate) fn update_diag_draw(
        &mut self,
        math: &mut M,
        draw_mean: &M::Vector,
        draw_var: &M::Vector,
        scale: f64,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_draw(
            &mut self.inv_stds,
            &mut self.stds,
            draw_var,
            scale,
            fill_invalid,
            clamp,
        );
        math.copy_into(draw_mean, &mut self.mean);
        self.logdet = math.array_sum_ln(&self.inv_stds);
        self.id += 1;
    }

    pub(crate) fn update_diag_draw_grad(
        &mut self,
        math: &mut M,
        draw_mean: &M::Vector,
        grad_mean: &M::Vector,
        draw_var: &M::Vector,
        grad_var: &M::Vector,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_draw_grad(
            &mut self.inv_stds,
            &mut self.stds,
            draw_var,
            grad_var,
            fill_invalid,
            clamp,
        );
        let mut var = math.new_array();
        math.array_mult(&self.stds, &self.stds, &mut var);
        math.array_mult(&var, grad_mean, &mut self.mean);
        math.axpy(&draw_mean, &mut self.mean, 1.0);
        self.logdet = math.array_sum_ln(&self.inv_stds);
        self.id += 1;
    }

    pub(crate) fn update_diag_grad(
        &mut self,
        math: &mut M,
        position: &M::Vector,
        gradient: &M::Vector,
        fill_invalid: f64,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_grad(
            &mut self.inv_stds,
            &mut self.stds,
            gradient,
            fill_invalid,
            clamp,
        );
        let mut var = math.new_array();
        math.array_mult(&self.stds, &self.stds, &mut var);
        math.array_mult(&var, gradient, &mut self.mean);
        math.axpy(position, &mut self.mean, 1.0);
        self.logdet = math.array_sum_ln(&self.inv_stds);
        self.id += 1;
    }

    pub(crate) fn set_transform(&mut self, math: &mut M, stds: &M::Vector, mean: &M::Vector) {
        math.copy_into(stds, &mut self.stds);
        math.copy_into(mean, &mut self.mean);
        math.array_recip(&self.stds, &mut self.inv_stds);
        self.logdet = math.array_sum_ln(&self.inv_stds);
        self.id += 1;
    }

    pub(crate) fn logdet(&self) -> f64 {
        self.logdet
    }

    pub(crate) fn stds(&self) -> &M::Vector {
        &self.stds
    }

    pub(crate) fn inv_stds(&self) -> &M::Vector {
        &self.inv_stds
    }

    pub(crate) fn mean(&self) -> &M::Vector {
        &self.mean
    }
}

impl<M: Math> Transformation<M> for DiagMassMatrix<M> {
    fn init_from_untransformed_position(
        &self,
        math: &mut M,
        untransformed_position: &M::Vector,
        untransformed_gradient: &mut M::Vector,
        transformed_position: &mut M::Vector,
        transformed_gradient: &mut M::Vector,
    ) -> Result<(f64, f64), M::LogpErr> {
        let logp = math.logp_array(untransformed_position, untransformed_gradient)?;
        self.compute_transformed_position(math, untransformed_position, transformed_position);
        self.compute_transformed_gradient(math, untransformed_gradient, transformed_gradient);
        Ok((logp, self.logdet))
    }

    fn init_from_transformed_position(
        &self,
        math: &mut M,
        untransformed_position: &mut M::Vector,
        untransformed_gradient: &mut M::Vector,
        transformed_position: &M::Vector,
        transformed_gradient: &mut M::Vector,
    ) -> Result<(f64, f64), M::LogpErr> {
        self.compute_untransformed_position(math, transformed_position, untransformed_position);
        let logp = math.logp_array(untransformed_position, untransformed_gradient)?;
        self.compute_transformed_gradient(math, untransformed_gradient, transformed_gradient);
        Ok((logp, self.logdet))
    }

    fn inv_transform_normalize(
        &self,
        math: &mut M,
        untransformed_position: &M::Vector,
        untransformed_gradient: &M::Vector,
        transformed_position: &mut M::Vector,
        transformed_gradient: &mut M::Vector,
    ) -> Result<f64, M::LogpErr> {
        self.compute_transformed_position(math, untransformed_position, transformed_position);
        self.compute_transformed_gradient(math, untransformed_gradient, transformed_gradient);
        Ok(self.logdet)
    }

    fn transformation_id(&self, _math: &mut M) -> i64 {
        self.id
    }

    fn next_stats_options(&self, _math: &mut M, _current: i64) -> i64 {
        self.id
    }
}

impl<M: Math> DiagMassMatrix<M> {
    pub(crate) fn compute_transformed_position(
        &self,
        math: &mut M,
        untransformed_position: &M::Vector,
        transformed_position: &mut M::Vector,
    ) {
        math.axpy_out(
            &self.mean,
            &untransformed_position,
            -1.0,
            transformed_position,
        );
        math.array_mult_inplace(transformed_position, &self.inv_stds);
    }

    pub(crate) fn compute_untransformed_position(
        &self,
        math: &mut M,
        transformed_position: &M::Vector,
        untransformed_position: &mut M::Vector,
    ) {
        math.array_mult(transformed_position, &self.stds, untransformed_position);
        math.axpy(&self.mean, untransformed_position, 1.0);
    }

    pub(crate) fn compute_transformed_gradient(
        &self,
        math: &mut M,
        untransformed_gradient: &M::Vector,
        transformed_gradient: &mut M::Vector,
    ) {
        math.array_mult(untransformed_gradient, &self.stds, transformed_gradient);
    }
}
