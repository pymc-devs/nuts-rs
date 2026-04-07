//! Augment the diagonal transformation with a low-rank spectral correction for correlated posteriors.

use std::fmt::Debug;
use std::iter::repeat_n;

use faer::{Col, ColRef, Mat, MatRef};
use nuts_derive::Storable;
use serde::{Deserialize, Serialize};

use crate::transform::{DiagMassMatrix, Transformation};
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

    fn logdet(&self) -> f64 {
        self.logdet_contribution
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
    diag: DiagMassMatrix<M>,
    inner: Option<InnerMatrix<M>>,
    settings: LowRankSettings,
    logdet: f64,
    /// Monotonically increasing id; bumped whenever the matrix changes.
    id: i64,
}

impl<M: Math> Debug for LowRankMassMatrix<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LowRankMassMatrix")
            .field("diag", &self.diag)
            .field("inner", &self.inner)
            .field("settings", &self.settings)
            .field("id", &self.id)
            .finish()
    }
}

impl<M: Math> LowRankMassMatrix<M> {
    pub fn new(math: &mut M, settings: LowRankSettings) -> Self {
        Self {
            diag: DiagMassMatrix::new(math, settings.store_mass_matrix),
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
        pos: &M::Vector,
        grad: &M::Vector,
        fill_invalid: f64,
        clamp: (f64, f64),
    ) {
        self.inner = None;
        self.diag
            .update_diag_grad(math, pos, grad, fill_invalid, clamp);
        self.logdet = self.diag.logdet();
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
        stds: Col<f64>,
        mean: Col<f64>,
        vals: Col<f64>,
        vecs: Mat<f64>,
    ) {
        if (!col_all_finite(&stds.as_ref())) | (!col_all_finite(&mean.as_ref())) {
            return;
        }
        if (!col_all_finite(&vals.as_ref())) | (!mat_all_finite(&vecs.as_ref())) {
            return;
        }

        let mut stds_array = math.new_array();
        math.read_from_slice(&mut stds_array, stds.try_as_col_major().unwrap().as_slice());
        let mut mean_array = math.new_array();
        math.read_from_slice(&mut mean_array, mean.try_as_col_major().unwrap().as_slice());
        self.diag.set_transform(math, &stds_array, &mean_array);

        let inner = InnerMatrix::new(math, vals, vecs);
        self.logdet = inner.logdet() + self.diag.logdet();
        self.inner = Some(inner);
        self.id += 1;
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
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
    /// The transformation version counter at the time of this update.
    /// `Some` only on draws where the transformation changed.
    #[storable(event = "transformation_update")]
    pub transformation_update_id: Option<i64>,
    #[storable(event = "transformation_update", dims("unconstrained_parameter"))]
    pub mass_matrix_eigvals: Option<Vec<f64>>,
    #[storable(event = "transformation_update", dims("unconstrained_parameter"))]
    pub mass_matrix_stds: Option<Vec<f64>>,
    #[storable(event = "transformation_update")]
    pub num_eigenvalues: Option<u64>,
}

impl<M: Math> SamplerStats<M> for LowRankMassMatrix<M> {
    type Stats = MatrixStats;
    type StatsOptions = i64;

    fn extract_stats(&self, math: &mut M, last_id: Self::StatsOptions) -> Self::Stats {
        if self.id != last_id {
            let num_eigenvalues = Some(
                self.inner
                    .as_ref()
                    .map(|inner| inner.num_eigenvalues)
                    .unwrap_or(0),
            );
            if self.settings.store_mass_matrix {
                let stds = Some(math.box_array(self.diag.stds()));
                let eigvals = self
                    .inner
                    .as_ref()
                    .map(|inner| math.eigs_as_array(&inner.vals_sqrt));
                let mut eigvals = eigvals.map(|x| x.into_vec());
                if let Some(ref mut eigvals) = eigvals {
                    eigvals.extend(repeat_n(
                        f64::NAN,
                        stds.as_ref().unwrap().len() - eigvals.len(),
                    ));
                }
                MatrixStats {
                    transformation_update_id: Some(self.id),
                    mass_matrix_eigvals: eigvals,
                    mass_matrix_stds: stds.map(|x| x.into_vec()),
                    num_eigenvalues,
                }
            } else {
                MatrixStats {
                    transformation_update_id: Some(self.id),
                    mass_matrix_eigvals: None,
                    mass_matrix_stds: None,
                    num_eigenvalues,
                }
            }
        } else {
            MatrixStats {
                transformation_update_id: None,
                mass_matrix_eigvals: None,
                mass_matrix_stds: None,
                num_eigenvalues: None,
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

    fn next_stats_options(&self, _math: &mut M, _current: i64) -> i64 {
        self.id
    }
}

impl<M: Math> LowRankMassMatrix<M> {
    fn compute_transformed_position(
        &self,
        math: &mut M,
        untransformed_position: &M::Vector,
        transformed_position: &mut M::Vector,
    ) {
        math.axpy_out(
            &self.diag.mean(),
            &untransformed_position,
            -1.0,
            transformed_position,
        );
        math.array_mult_inplace(transformed_position, self.diag.inv_stds());

        if let Some(inner) = &self.inner {
            math.apply_lowrank_transform_inplace(
                &inner.vecs,
                &inner.vals_sqrt_inv,
                transformed_position,
            );
        }
    }

    fn compute_untransformed_position(
        &self,
        math: &mut M,
        transformed_position: &M::Vector,
        untransformed_position: &mut M::Vector,
    ) {
        match &self.inner {
            None => {
                math.array_mult(
                    transformed_position,
                    &self.diag.stds(),
                    untransformed_position,
                );
            }
            Some(inner) => {
                math.apply_lowrank_transform(
                    &inner.vecs,
                    &inner.vals_sqrt,
                    transformed_position,
                    untransformed_position,
                );
                math.array_mult_inplace(untransformed_position, &self.diag.stds());
            }
        }
        math.axpy(&self.diag.mean(), untransformed_position, 1.0);
    }

    fn compute_transformed_gradient(
        &self,
        math: &mut M,
        untransformed_gradient: &M::Vector,
        transformed_gradient: &mut M::Vector,
    ) {
        math.array_mult(
            untransformed_gradient,
            self.diag.stds(),
            transformed_gradient,
        );

        if let Some(inner) = &self.inner {
            math.apply_lowrank_transform_inplace(
                &inner.vecs,
                &inner.vals_sqrt,
                transformed_gradient,
            );
        }
    }

    /// log|det J_{F⁻¹}| = Σ log(σᵢ⁻¹) − ½ Σ log(λᵢ)
    fn logdet(&self, _math: &mut M) -> f64 {
        self.logdet
    }
}
