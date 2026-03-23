//! Define the backend interface that decouples the sampler from any particular hardware or logp implementation.

use std::{error::Error, fmt::Debug};

use nuts_storable::{HasDims, Storable, Value};
use rand::Rng;

/// Errors that happen when we evaluate the logp and gradient function
pub trait LogpError: std::error::Error + Send {
    /// Unrecoverable errors during logp computation stop sampling,
    /// recoverable errors are seen as divergences.
    fn is_recoverable(&self) -> bool;
}

pub trait Math: HasDims {
    type Vector: Debug;
    type EigVectors: Debug;
    type EigValues: Debug;
    type LogpErr: Debug + Send + Sync + LogpError + Sized + 'static;
    type Err: Debug + Send + Sync + Error + 'static;
    type FlowParameters;
    type ExpandedVector: Storable<Self>;

    fn new_array(&mut self) -> Self::Vector;

    fn copy_array(&mut self, array: &Self::Vector) -> Self::Vector {
        let mut copy = self.new_array();
        self.copy_into(array, &mut copy);
        copy
    }

    fn new_eig_vectors<'a>(
        &'a mut self,
        vals: impl ExactSizeIterator<Item = &'a [f64]>,
    ) -> Self::EigVectors;
    fn new_eig_values(&mut self, vals: &[f64]) -> Self::EigValues;

    /// Compute the unnormalized log probability density of the posterior
    ///
    /// This needs to be implemnted by users of the library to define
    /// what distribution the users wants to sample from.
    ///
    /// Errors during that computation can be recoverable or non-recoverable.
    /// If a non-recoverable error occurs during sampling, the sampler will
    /// stop and return an error.
    fn logp_array(
        &mut self,
        position: &Self::Vector,
        gradient: &mut Self::Vector,
    ) -> Result<f64, Self::LogpErr>;

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpErr>;

    fn init_position<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        position: &mut Self::Vector,
        gradient: &mut Self::Vector,
    ) -> Result<f64, Self::LogpErr>;

    /// Expand a vector into a larger representation, to for instance
    /// compute deterministic values that are to be stored in the trace.
    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        array: &Self::Vector,
    ) -> Result<Self::ExpandedVector, Self::Err>;

    fn dim(&self) -> usize;

    fn vector_coord(&self) -> Option<Value> {
        None
    }

    fn scalar_prods3(
        &mut self,
        positive1: &Self::Vector,
        negative1: &Self::Vector,
        positive2: &Self::Vector,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> (f64, f64);

    fn scalar_prods2(
        &mut self,
        positive1: &Self::Vector,
        positive2: &Self::Vector,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> (f64, f64);

    fn sq_norm_sum(&mut self, x: &Self::Vector, y: &Self::Vector) -> f64;

    fn read_from_slice(&mut self, dest: &mut Self::Vector, source: &[f64]);
    fn write_to_slice(&mut self, source: &Self::Vector, dest: &mut [f64]);
    fn eigs_as_array(&mut self, source: &Self::EigValues) -> Box<[f64]>;
    fn copy_into(&mut self, array: &Self::Vector, dest: &mut Self::Vector);
    fn axpy_out(&mut self, x: &Self::Vector, y: &Self::Vector, a: f64, out: &mut Self::Vector);
    fn axpy(&mut self, x: &Self::Vector, y: &mut Self::Vector, a: f64);

    fn box_array(&mut self, array: &Self::Vector) -> Box<[f64]> {
        let mut data = vec![0f64; self.dim()];
        self.write_to_slice(array, &mut data);
        data.into()
    }

    /// Compute the sum of the natural logarithms of all elements in `array`,
    /// i.e. `Σ ln(array[i])`.
    ///
    /// The default implementation copies into a temporary allocation via
    /// [`write_to_slice`]; backends may override this with a zero-allocation
    /// version.
    fn array_sum_ln(&mut self, array: &Self::Vector) -> f64 {
        let mut data = vec![0f64; self.dim()];
        self.write_to_slice(array, &mut data);
        data.iter().map(|x| x.ln()).sum()
    }

    fn fill_array(&mut self, array: &mut Self::Vector, val: f64);

    fn array_all_finite(&mut self, array: &Self::Vector) -> bool;
    fn array_all_finite_and_nonzero(&mut self, array: &Self::Vector) -> bool;
    fn array_mult(&mut self, array1: &Self::Vector, array2: &Self::Vector, dest: &mut Self::Vector);
    fn array_mult_inplace(&mut self, array1: &mut Self::Vector, array2: &Self::Vector);
    fn array_recip(&mut self, array: &Self::Vector, dest: &mut Self::Vector);

    /// Apply the low-rank linear map `(I + U * (diag(vals) - I) * U^T) * rhs` into `dest`.
    ///
    /// `vecs` is `U` (d × r, orthonormal columns), `vals` is the diagonal vector (length r).
    /// When `vecs` has zero columns the result is just a copy of `rhs`.
    fn apply_lowrank_transform(
        &mut self,
        vecs: &Self::EigVectors,
        vals: &Self::EigValues,
        rhs: &Self::Vector,
        dest: &mut Self::Vector,
    );

    fn apply_lowrank_transform_inplace(
        &mut self,
        vecs: &Self::EigVectors,
        vals: &Self::EigValues,
        rhs_and_dest: &mut Self::Vector,
    );

    fn array_mult_eigs(
        &mut self,
        stds: &Self::Vector,
        rhs: &Self::Vector,
        dest: &mut Self::Vector,
        vecs: &Self::EigVectors,
        vals: &Self::EigValues,
    );

    fn std_norm_flow(
        &mut self,
        pos: &Self::Vector,
        pos_out: &mut Self::Vector,
        vel: &mut Self::Vector,
        epsilon: f64,
    );
    fn std_norm_grad_flow(
        &mut self,
        pos: &Self::Vector,
        grad: &Self::Vector,
        vel: &Self::Vector,
        vel_out: &mut Self::Vector,
        epsilon: f64,
    );
    fn std_norm_grad_flow_inplace(
        &mut self,
        pos: &Self::Vector,
        grad: &Self::Vector,
        vel: &mut Self::Vector,
        epsilon: f64,
    );

    /// Normalise `v` to unit length in-place: `v := v / ‖v‖`.
    ///
    /// If `‖v‖ < 1e-300` the vector is left unchanged.
    fn array_normalize(&mut self, v: &mut Self::Vector);

    /// Perform one ESH (Extended Stochastic Hamiltonian) momentum half-step.
    ///
    /// Updates `mom` in-place so that it remains on the unit sphere, and
    /// returns the new cumulative kinetic-energy change `prev_delta_ke + ΔKE`.
    ///
    /// # Algorithm
    ///
    /// Given momentum `p` on the unit sphere, log-density gradient `g`,
    /// half-step size `step`, and dimension `n`:
    ///
    /// ```text
    /// ĝ      = g / ‖g‖
    /// α      = p · ĝ
    /// Δ      = step · ‖g‖ / (n − 1)
    /// ζ      = exp(−Δ)
    /// p_raw  = ĝ · (1 − ζ)(1 + ζ + α(1 − ζ))  +  2ζ p
    /// p'     = p_raw / ‖p_raw‖
    /// ΔKE    = (Δ − log 2 + log(1 + α + (1 − α)ζ²)) · (n − 1)
    /// ```
    ///
    /// Reference: Steeg & Gallagher, arXiv:2111.02434 (2021), ported from the
    /// [BlackJAX implementation](https://github.com/blackjax-devs/blackjax/blob/main/blackjax/mcmc/integrators.py#L314).
    fn esh_momentum_update(
        &mut self,
        grad: &Self::Vector,
        mom: &mut Self::Vector,
        step: f64,
    ) -> f64;

    fn array_vector_dot(&mut self, array1: &Self::Vector, array2: &Self::Vector) -> f64;
    fn array_gaussian<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Vector,
        stds: &Self::Vector,
    );
    fn array_gaussian_eigs<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Vector,
        scale: &Self::Vector,
        vals: &Self::EigValues,
        vecs: &Self::EigVectors,
    );
    fn array_update_variance(
        &mut self,
        mean: &mut Self::Vector,
        variance: &mut Self::Vector,
        value: &Self::Vector,
        diff_scale: f64,
    );
    fn array_update_var_inv_std_draw(
        &mut self,
        inv_std: &mut Self::Vector,
        std: &mut Self::Vector,
        draw_var: &Self::Vector,
        scale: f64,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    );
    fn array_update_var_inv_std_draw_grad(
        &mut self,
        inv_std: &mut Self::Vector,
        std: &mut Self::Vector,
        draw_var: &Self::Vector,
        grad_var: &Self::Vector,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    );

    fn array_update_var_inv_std_grad(
        &mut self,
        inv_std: &mut Self::Vector,
        std: &mut Self::Vector,
        gradient: &Self::Vector,
        fill_invalid: f64,
        clamp: (f64, f64),
    );

    fn inv_transform_normalize(
        &mut self,
        params: &Self::FlowParameters,
        untransformed_position: &Self::Vector,
        untransofrmed_gradient: &Self::Vector,
        transformed_position: &mut Self::Vector,
        transformed_gradient: &mut Self::Vector,
    ) -> Result<f64, Self::LogpErr>;

    fn init_from_untransformed_position(
        &mut self,
        params: &Self::FlowParameters,
        untransformed_position: &Self::Vector,
        untransformed_gradient: &mut Self::Vector,
        transformed_position: &mut Self::Vector,
        transformed_gradient: &mut Self::Vector,
    ) -> Result<(f64, f64), Self::LogpErr>;

    fn init_from_transformed_position(
        &mut self,
        params: &Self::FlowParameters,
        untransformed_position: &mut Self::Vector,
        untransformed_gradient: &mut Self::Vector,
        transformed_position: &Self::Vector,
        transformed_gradient: &mut Self::Vector,
    ) -> Result<(f64, f64), Self::LogpErr>;

    fn update_transformation<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        rng: &mut R,
        untransformed_positions: impl ExactSizeIterator<Item = &'a Self::Vector>,
        untransformed_gradients: impl ExactSizeIterator<Item = &'a Self::Vector>,
        untransformed_logps: impl ExactSizeIterator<Item = &'a f64>,
        params: &'a mut Self::FlowParameters,
    ) -> Result<(), Self::LogpErr>;

    fn new_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dim: usize,
        chain: u64,
    ) -> Result<Self::FlowParameters, Self::LogpErr>;

    fn init_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        untransformed_position: &Self::Vector,
        untransfogmed_gradient: &Self::Vector,
        chain: u64,
    ) -> Result<Self::FlowParameters, Self::LogpErr>;

    fn transformation_id(&self, params: &Self::FlowParameters) -> Result<i64, Self::LogpErr>;
}
