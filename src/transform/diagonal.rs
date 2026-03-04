use std::fmt::Debug;

use nuts_derive::Storable;

use crate::{math_base::Math, sampler_stats::SamplerStats, transform::Transformation};

pub struct DiagMassMatrix<M: Math> {
    mean: M::Vector,
    inv_stds: M::Vector,
    pub(crate) variance: M::Vector,
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
            .field("variance", &self.variance)
            .field("logdet", &self.logdet)
            .field("store_mass_matrix", &self.store_mass_matrix)
            .field("id", &self.id)
            .finish()
    }
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
            mean: math.new_array(),
            inv_stds: math.new_array(),
            variance: math.new_array(),
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
            &mut self.variance,
            &mut self.inv_stds,
            draw_var,
            scale,
            fill_invalid,
            clamp,
        );
        // μ* = x̄  (no gradient information available in the draw-only case)
        math.copy_into(draw_mean, &mut self.mean);
        self.logdet = Self::compute_logdet_from_inv_stds(math, &self.inv_stds);
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
            &mut self.variance,
            &mut self.inv_stds,
            draw_var,
            grad_var,
            fill_invalid,
            clamp,
        );
        // μ* = x̄ + σ² ⊙ ᾱ  (equation from the paper)
        // self.variance = σ² is already updated above, so we can use it directly
        math.copy_into(draw_mean, &mut self.mean);
        let mut var_times_grad_mean = math.new_array();
        math.array_mult(&self.variance, grad_mean, &mut var_times_grad_mean);
        math.axpy(&var_times_grad_mean, &mut self.mean, 1.0);
        self.logdet = Self::compute_logdet_from_inv_stds(math, &self.inv_stds);
        self.id += 1;
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
        // Mean stays at zero for the gradient-only initialisation
        math.fill_array(&mut self.mean, 0f64);
        self.logdet = Self::compute_logdet_from_inv_stds(math, &self.inv_stds);
        self.id += 1;
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

        // z = (x - μ) * inv_std
        let mut centered = math.new_array();
        math.copy_into(untransformed_position, &mut centered);
        math.axpy(&self.mean, &mut centered, -1.0);
        math.array_mult(&centered, &self.inv_stds, transformed_position);

        // std = variance * inv_std  (= sqrt(variance) = σ)
        // grad_z = grad_x * std
        let mut std_vec = math.new_array();
        math.array_mult(&self.variance, &self.inv_stds, &mut std_vec);
        math.array_mult(untransformed_gradient, &std_vec, transformed_gradient);

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
        // x = z * std + μ   (std = variance * inv_std = σ)
        let mut std_vec = math.new_array();
        math.array_mult(&self.variance, &self.inv_stds, &mut std_vec);
        math.array_mult(transformed_position, &std_vec, untransformed_position);
        math.axpy(&self.mean, untransformed_position, 1.0);

        let logp = math.logp_array(untransformed_position, untransformed_gradient)?;

        // grad_z = grad_x * std
        math.array_mult(untransformed_gradient, &std_vec, transformed_gradient);

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
        // z = (x - μ) * inv_std
        let mut centered = math.new_array();
        math.copy_into(untransformed_position, &mut centered);
        math.axpy(&self.mean, &mut centered, -1.0);
        math.array_mult(&centered, &self.inv_stds, transformed_position);

        // std = variance * inv_std; grad_z = grad_x * std
        let mut std_vec = math.new_array();
        math.array_mult(&self.variance, &self.inv_stds, &mut std_vec);
        math.array_mult(untransformed_gradient, &std_vec, transformed_gradient);

        Ok(self.logdet)
    }

    fn transformation_id(&self, _math: &mut M) -> i64 {
        self.id
    }
}

impl<M: Math> DiagMassMatrix<M> {
    /// Compute Σ log(inv_std_i) from the device vector.
    /// Called only on adaptation updates, never during leapfrog steps.
    fn compute_logdet_from_inv_stds(math: &mut M, inv_stds: &M::Vector) -> f64 {
        math.array_sum_ln(inv_stds)
    }
}
