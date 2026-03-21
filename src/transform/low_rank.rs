//! Augment the diagonal transformation with a low-rank spectral correction for correlated posteriors.

use std::fmt::Debug;
use std::iter::repeat;

use faer::{Col, ColRef, Mat, MatRef};
use nuts_derive::Storable;
use serde::Serialize;

use crate::transform::Transformation;
use crate::{Math, sampler_stats::SamplerStats};

pub fn mat_all_finite(mat: &MatRef<f64>) -> bool {
    let mut ok = true;
    faer::zip!(mat).for_each(|faer::unzip!(val)| ok &= val.is_finite());
    ok
}

fn col_all_finite(mat: &ColRef<f64>) -> bool {
    let mut ok = true;
    faer::zip!(mat).for_each(|faer::unzip!(val)| ok &= val.is_finite());
    ok
}

/// The low-rank correction to the affine transformation.
///
/// Stores U (eigenvectors), λ^{1/2} (used for F and J_F), λ^{-1/2} (used for F⁻¹), and
/// the precomputed low-rank contribution to log|det J_{F⁻¹}|.
struct InnerMatrix<M: Math> {
    vecs: M::EigVectors,
    /// λ^{1/2} — used for the forward map F and its Jacobian J_F
    vals_sqrt: M::EigValues,
    /// λ^{-1/2} — used for the inverse position transform F⁻¹
    vals_sqrt_inv: M::EigValues,
    /// -½ Σ log(λᵢ) — low-rank contribution to log|det J_{F⁻¹}|, precomputed
    /// so we never need to pull eigenvalues back from a device (e.g. GPU).
    logdet_contribution: f64,
    num_eigenvalues: u64,
}

impl<M: Math> Debug for InnerMatrix<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InnerMatrix")
            .field("vecs", &"<eig vectors>")
            .field("vals_sqrt", &"<sqrt eig values>")
            .field("vals_sqrt_inv", &"<inv sqrt eig values>")
            .field("logdet_contribution", &self.logdet_contribution)
            .field("num_eigenvalues", &self.num_eigenvalues)
            .finish()
    }
}

impl<M: Math> InnerMatrix<M> {
    fn new(math: &mut M, mut vals: Col<f64>, vecs: Mat<f64>) -> Self {
        // Precompute -½ Σ log(λᵢ) while vals still holds the raw eigenvalues.
        let logdet_contribution: f64 = vals.iter().map(|&v| -0.5 * v.ln()).sum();
        let num_eigenvalues = vals.nrows() as u64;

        let vecs = math.new_eig_vectors(
            vecs.col_iter()
                .map(|col| col.try_as_col_major().unwrap().as_slice()),
        );

        // λ^{1/2} — needed for the forward map F and its Jacobian J_F
        vals.iter_mut().for_each(|x| *x = x.sqrt());
        let vals_sqrt = math.new_eig_values(vals.try_as_col_major().unwrap().as_slice());

        // λ^{-1/2} — needed for the inverse position transform F⁻¹
        vals.iter_mut().for_each(|x| *x = x.recip());
        let vals_sqrt_inv = math.new_eig_values(vals.try_as_col_major().unwrap().as_slice());

        Self {
            vecs,
            vals_sqrt,
            vals_sqrt_inv,
            logdet_contribution,
            num_eigenvalues,
        }
    }
}

/// Low-rank + diagonal affine transformation.
///
/// The full forward map (adapted → target) is
///
///   F(y) = σ ⊙ (I + U (diag(λ)^{1/2} − I) Uᵀ) y + μ
///
/// so the inverse (target → adapted) is
///
///   F⁻¹(x) = (I + U (diag(λ)^{-1/2} − I) Uᵀ) ((x − μ) ⊙ σ⁻¹)
///
/// The Jacobian of F is  J_F = diag(σ) (I + U (diag(λ)^{1/2} − I) Uᵀ),
/// so  log|det J_{F⁻¹}| = Σ log(σᵢ⁻¹) − ½ Σ log(λᵢ).
///
/// In the adapted space the mass matrix is the identity; leapfrog steps
/// operate entirely in that space.  When no eigenvectors are available
/// (early adaptation) the transform falls back to the pure diagonal case.
pub struct LowRankMassMatrix<M: Math> {
    /// μ* = x̄ + σ² ⊙ ᾱ — translation centre (target space)
    mean: M::Vector,
    variance: M::Vector,
    stds: M::Vector,
    inv_stds: M::Vector,
    inner: Option<InnerMatrix<M>>,
    settings: LowRankSettings,
    /// log|det J_{F⁻¹}| = Σ log(σᵢ⁻¹) − ½ Σ log(λᵢ), cached so leapfrog never calls write_to_slice.
    logdet: f64,
    /// Monotonically increasing id; bumped whenever the matrix changes.
    id: i64,
}

impl<M: Math> Debug for LowRankMassMatrix<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LowRankMassMatrix")
            .field("mean", &self.mean)
            .field("variance", &self.variance)
            .field("stds", &self.stds)
            .field("inv_stds", &self.inv_stds)
            .field("inner", &self.inner)
            .field("settings", &self.settings)
            .field("id", &self.id)
            .finish()
    }
}

impl<M: Math> LowRankMassMatrix<M> {
    pub fn new(math: &mut M, settings: LowRankSettings) -> Self {
        Self {
            mean: math.new_array(),
            variance: math.new_array(),
            inv_stds: math.new_array(),
            stds: math.new_array(),
            settings,
            logdet: 0f64,
            inner: None,
            id: -1,
        }
    }

    /// Initialise from the gradient at the first draw only (no mean / covariance information yet).
    pub fn update_from_grad(
        &mut self,
        math: &mut M,
        grad: &<M as Math>::Vector,
        fill_invalid: f64,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_grad(
            &mut self.variance,
            &mut self.inv_stds,
            grad,
            fill_invalid,
            clamp,
        );
        let mut inv_std_vals = vec![0f64; math.dim()];
        math.write_to_slice(&self.inv_stds, &mut inv_std_vals);
        self.logdet = inv_std_vals.iter().map(|v| v.ln()).sum();
        inv_std_vals.iter_mut().for_each(|x| *x = x.recip());
        math.read_from_slice(&mut self.stds, &inv_std_vals);
        // No mean information yet — keep at zero.
        math.fill_array(&mut self.mean, 0f64);
        self.id += 1;
    }

    /// Full update from a window of draws and scores.
    ///
    /// * `stds`  — diagonal scales σ
    /// * `mean`  — optimal translation μ* = x̄ + σ² ⊙ ᾱ (in target space)
    /// * `vals`  — filtered eigenvalues λ of the SPD geometric mean
    /// * `vecs`  — corresponding eigenvectors U (columns, back-projected to ℝᵈ)
    pub fn update(
        &mut self,
        math: &mut M,
        mut stds: Col<f64>,
        mean: Col<f64>,
        vals: Col<f64>,
        vecs: Mat<f64>,
    ) {
        math.read_from_slice(&mut self.stds, stds.try_as_col_major().unwrap().as_slice());
        math.read_from_slice(&mut self.mean, mean.try_as_col_major().unwrap().as_slice());

        stds.iter_mut().for_each(|x| *x = x.recip());
        math.read_from_slice(
            &mut self.inv_stds,
            stds.try_as_col_major().unwrap().as_slice(),
        );

        stds.iter_mut().for_each(|x| *x = x.recip() * x.recip());
        math.read_from_slice(
            &mut self.variance,
            stds.try_as_col_major().unwrap().as_slice(),
        );

        if col_all_finite(&vals.as_ref()) & mat_all_finite(&vecs.as_ref()) {
            self.inner = Some(InnerMatrix::new(math, vals, vecs));
        } else {
            self.inner = None;
        }
        // Cache the full logdet: diagonal part + low-rank correction.
        // write_to_slice here is fine — this runs only on adaptation updates, not leapfrog steps.
        let mut std_vals = vec![0f64; math.dim()];
        math.write_to_slice(&self.inv_stds, &mut std_vals);
        let diag_part: f64 = std_vals.iter().map(|v| v.ln()).sum();
        let lowrank_part = self
            .inner
            .as_ref()
            .map_or(0.0, |inner| inner.logdet_contribution);
        self.logdet = diag_part + lowrank_part;
        self.id += 1;
    }
}

#[derive(Clone, Debug, Copy, Serialize)]
pub struct LowRankSettings {
    pub store_mass_matrix: bool,
    pub gamma: f64,
    pub eigval_cutoff: f64,
}

impl Default for LowRankSettings {
    fn default() -> Self {
        Self {
            store_mass_matrix: false,
            gamma: 1e-5,
            eigval_cutoff: 2f64,
        }
    }
}

#[derive(Debug, Storable)]
pub struct MatrixStats {
    #[storable(dims("unconstrained_parameter"))]
    pub mass_matrix_eigvals: Option<Vec<f64>>,
    #[storable(dims("unconstrained_parameter"))]
    pub mass_matrix_stds: Option<Vec<f64>>,
    pub num_eigenvalues: u64,
}

impl<M: Math> SamplerStats<M> for LowRankMassMatrix<M> {
    type Stats = MatrixStats;
    type StatsOptions = ();

    fn extract_stats(&self, math: &mut M, _opt: Self::StatsOptions) -> Self::Stats {
        if self.settings.store_mass_matrix {
            let stds = Some(math.box_array(&self.stds));
            let eigvals = self
                .inner
                .as_ref()
                .map(|inner| math.eigs_as_array(&inner.vals_sqrt));
            let mut eigvals = eigvals.map(|x| x.into_vec());
            if let Some(ref mut eigvals) = eigvals {
                eigvals.extend(repeat(f64::NAN).take(stds.as_ref().unwrap().len() - eigvals.len()));
            }
            MatrixStats {
                mass_matrix_eigvals: eigvals,
                mass_matrix_stds: stds.map(|x| x.into_vec()),
                num_eigenvalues: self
                    .inner
                    .as_ref()
                    .map(|inner| inner.num_eigenvalues)
                    .unwrap_or(0),
            }
        } else {
            MatrixStats {
                mass_matrix_eigvals: None,
                mass_matrix_stds: None,
                num_eigenvalues: self
                    .inner
                    .as_ref()
                    .map(|inner| inner.num_eigenvalues)
                    .unwrap_or(0),
            }
        }
    }
}

impl<M: Math> Transformation<M> for LowRankMassMatrix<M> {
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
        Ok((logp, self.logdet(math)))
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
        Ok((logp, self.logdet(math)))
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
        Ok(self.logdet(math))
    }

    fn transformation_id(&self, _math: &mut M) -> i64 {
        self.id
    }
}

impl<M: Math> LowRankMassMatrix<M> {
    /// z = F⁻¹(x) = (I + U (diag(λ^{-1/2}) − I) Uᵀ) · ((x − μ) ⊙ σ⁻¹)
    fn compute_transformed_position(&self, math: &mut M, x: &M::Vector, z: &mut M::Vector) {
        // centered = x − μ
        let mut centered = math.new_array();
        math.copy_into(x, &mut centered);
        math.axpy(&self.mean, &mut centered, -1.0);

        // scaled = (x − μ) ⊙ inv_stds
        let mut scaled = math.new_array();
        math.array_mult(&centered, &self.inv_stds, &mut scaled);

        match &self.inner {
            None => math.copy_into(&scaled, z),
            Some(inner) => {
                // z = (I + U (diag(λ^{-1/2}) − I) Uᵀ) · scaled
                math.apply_lowrank_transform(&inner.vecs, &inner.vals_sqrt_inv, &scaled, z);
            }
        }
    }

    /// x = F(z) = σ ⊙ (I + U (diag(λ^{1/2}) − I) Uᵀ) · z + μ
    fn compute_untransformed_position(&self, math: &mut M, z: &M::Vector, x: &mut M::Vector) {
        match &self.inner {
            None => {
                math.array_mult(z, &self.stds, x);
            }
            Some(inner) => {
                // tmp = (I + U (diag(λ^{1/2}) − I) Uᵀ) · z
                let mut tmp = math.new_array();
                math.apply_lowrank_transform(&inner.vecs, &inner.vals_sqrt, z, &mut tmp);
                // x = σ ⊙ tmp
                math.array_mult(&tmp, &self.stds, x);
            }
        }
        // x += μ
        math.axpy(&self.mean, x, 1.0);
    }

    /// β = J_Fᵀ · α = (I + U (diag(λ^{1/2}) − I) Uᵀ) · (σ ⊙ α)
    fn compute_transformed_gradient(
        &self,
        math: &mut M,
        grad_x: &M::Vector,
        grad_z: &mut M::Vector,
    ) {
        // scaled = σ ⊙ grad_x
        let mut scaled = math.new_array();
        math.array_mult(grad_x, &self.stds, &mut scaled);

        match &self.inner {
            None => math.copy_into(&scaled, grad_z),
            Some(inner) => {
                // grad_z = (I + U (diag(λ^{1/2}) − I) Uᵀ) · scaled
                math.apply_lowrank_transform(&inner.vecs, &inner.vals_sqrt, &scaled, grad_z);
            }
        }
    }

    /// log|det J_{F⁻¹}| = Σ log(σᵢ⁻¹) − ½ Σ log(λᵢ)
    fn logdet(&self, _math: &mut M) -> f64 {
        self.logdet
    }
}
