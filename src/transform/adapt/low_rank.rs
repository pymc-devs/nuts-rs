//! Online low-rank mass-matrix estimator that accumulates draw/gradient windows and fits a low-rank + diagonal approximation.

use std::collections::VecDeque;

use faer::{Col, ColRef, Mat, Scale};
use itertools::Itertools;

use crate::{
    LowRankSettings, Math, NutsError, SamplerStats,
    dynamics::Point,
    transform::{LowRankMassMatrix, MassMatrixAdaptStrategy, adapt::diagonal::DrawGradCollector},
};

#[derive(Debug)]
pub struct LowRankMassMatrixStrategy {
    draws: VecDeque<Vec<f64>>,
    grads: VecDeque<Vec<f64>>,
    ndim: usize,
    background_split: usize,
    settings: LowRankSettings,
}

impl LowRankMassMatrixStrategy {
    pub fn new(ndim: usize, settings: LowRankSettings) -> Self {
        let draws = VecDeque::with_capacity(100);
        let grads = VecDeque::with_capacity(100);

        Self {
            draws,
            grads,
            ndim,
            background_split: 0,
            settings,
        }
    }

    pub fn add_draw<M: Math>(&mut self, math: &mut M, point: &impl Point<M>) {
        assert!(math.dim() == self.ndim);
        let mut draw = vec![0f64; self.ndim];
        math.write_to_slice(point.position(), &mut draw);
        let mut grad = vec![0f64; self.ndim];
        math.write_to_slice(point.gradient(), &mut grad);

        self.draws.push_back(draw);
        self.grads.push_back(grad);
    }

    pub fn clear(&mut self) {
        self.draws.clear();
        self.grads.clear();
    }

    pub fn update<M: Math>(&self, math: &mut M, matrix: &mut LowRankMassMatrix<M>) {
        let ndraws = self.draws.len();
        assert!(self.grads.len() == ndraws);

        let mut draws: Mat<f64> = Mat::zeros(self.ndim, ndraws);
        let mut grads: Mat<f64> = Mat::zeros(self.ndim, ndraws);

        for (i, (draw, grad)) in self.draws.iter().zip(self.grads.iter()).enumerate() {
            draws.col_as_slice_mut(i).copy_from_slice(&draw[..]);
            grads.col_as_slice_mut(i).copy_from_slice(&grad[..]);
        }

        let Some((stds, mean, vals, vecs, mean_low_rank)) = self.compute_update(draws, grads)
        else {
            return;
        };

        matrix.update(math, stds, mean, vals, vecs, mean_low_rank);
    }

    fn compute_update(
        &self,
        mut draws: Mat<f64>,
        mut grads: Mat<f64>,
    ) -> Option<(Col<f64>, Col<f64>, Col<f64>, Mat<f64>, Col<f64>)> {
        let (stds, mean, draw_mean, grad_mean) = rescale_points(&mut draws, &mut grads);

        let svd_draws = draws.thin_svd().ok()?;
        let svd_grads = grads.thin_svd().ok()?;

        let subspace = faer::concat![[svd_draws.U(), svd_grads.U()]];

        let subspace_qr = subspace.col_piv_qr();
        let subspace_basis = subspace_qr.compute_thin_Q();

        let draws_proj = subspace_basis.transpose() * (&draws);
        let grads_proj = subspace_basis.transpose() * (&grads);

        let (vals, vecs) = estimate_mass_matrix(draws_proj, grads_proj, self.settings.gamma)?;

        let filtered = vals
            .iter()
            .zip(vecs.col_iter())
            .filter(|&(&val, _)| {
                (val > self.settings.eigval_cutoff) | (val < self.settings.eigval_cutoff.recip())
            })
            .collect_vec();

        let vals: Col<f64> =
            ColRef::from_slice(&filtered.iter().map(|x| *x.0).collect_vec()).to_owned();

        let vecs_vec: Vec<_> = filtered.into_iter().map(|x| x.1).collect();
        let mut vecs = Mat::zeros(subspace_basis.ncols(), vals.nrows());
        vecs.col_iter_mut()
            .zip(vecs_vec.iter())
            .for_each(|(mut col, src)| col.copy_from(src));

        let vecs = subspace_basis * vecs;

        let mut vals_m1 = vals.clone();
        vals_m1.iter_mut().for_each(|x| *x -= 1.0);
        let vals_m1 = vals_m1.into_diagonal();

        // TODO use low level api
        let b = vecs.transpose() * &grad_mean;
        let b = vals_m1 * b;
        let b = &vecs * b;
        //let mu = &vecs * (vecs.transpose() * draw_mean) + b;
        let mu = draw_mean + grad_mean + b;

        Some((stds, mean, vals, vecs, mu))
    }
}

/// Rescale draws and gradients in-place and return the parameters of that transformation.
///
/// **Step 1 — diagonal rescaling:**
/// Computes per-dimension scale `σ_i = sqrt(sqrt(var(x_i) / var(α_i)))` and translation
/// `μ*_i = x̄_i + σ_i² · ᾱ_i`, then applies the affine map
///   `x_i  ↦  (x_i − μ*_i) / σ_i`
///   `α_i  ↦  α_i · σ_i`
///
/// **Step 2 — mean computation:**
/// Computes `draw_mean` and `grad_mean`, the per-dimension means of the rescaled values.
///
/// **Step 3 — centering:**
/// Subtracts those means so that the output columns have zero mean.
///
/// Returns `(stds, mu, draw_mean, grad_mean)` where `stds` and `mu` are the scale and
/// translation from Step 1, and `draw_mean`/`grad_mean` are the pre-centering means of
/// the rescaled values from Step 2.
fn rescale_points(
    draws: &mut Mat<f64>,
    grads: &mut Mat<f64>,
) -> (Col<f64>, Col<f64>, Col<f64>, Col<f64>) {
    let (ndim, ndraws) = draws.shape();
    let n = ndraws as f64;

    let mut stds = Col::zeros(ndim);
    let mut mu = Col::zeros(ndim);
    let mut draw_mean_out = Col::zeros(ndim);
    let mut grad_mean_out = Col::zeros(ndim);

    for row in 0..ndim {
        let draw_mean = draws.row(row).sum() / n;
        let grad_mean = grads.row(row).sum() / n;

        let draw_var: f64 = draws
            .row(row)
            .iter()
            .map(|&v| (v - draw_mean) * (v - draw_mean))
            .sum::<f64>()
            / n;
        let grad_var: f64 = grads
            .row(row)
            .iter()
            .map(|&v| (v - grad_mean) * (v - grad_mean))
            .sum::<f64>()
            / n;

        let sigma = (draw_var / grad_var).sqrt().sqrt();

        // μ* = x̄ + σ² · ᾱ
        mu[row] = draw_mean + sigma * sigma * grad_mean;
        stds[row] = sigma;

        let draw_scale = sigma.recip();
        draws
            .row_mut(row)
            .iter_mut()
            .for_each(|v| *v = (*v - mu[row]) * draw_scale);

        let grad_scale = sigma;
        grads
            .row_mut(row)
            .iter_mut()
            .for_each(|v| *v = (*v) * grad_scale);

        // Compute means of the rescaled draws and grads before centering
        draw_mean_out[row] = draws.row(row).sum() / n;
        grad_mean_out[row] = grads.row(row).sum() / n;

        // Center the rescaled draws and grads
        let dm = draw_mean_out[row];
        draws.row_mut(row).iter_mut().for_each(|v| *v -= dm);
        let gm = grad_mean_out[row];
        grads.row_mut(row).iter_mut().for_each(|v| *v -= gm);
    }

    (stds, mu, draw_mean_out, grad_mean_out)
}

fn estimate_mass_matrix(
    draws: Mat<f64>,
    grads: Mat<f64>,
    gamma: f64,
) -> Option<(Col<f64>, Mat<f64>)> {
    let mut cov_draws = (&draws) * draws.transpose();
    let mut cov_grads = (&grads) * grads.transpose();

    cov_draws *= Scale(gamma.recip());
    cov_grads *= Scale(gamma.recip());

    cov_draws
        .diagonal_mut()
        .column_vector_mut()
        .iter_mut()
        .for_each(|x| *x += 1f64);

    cov_grads
        .diagonal_mut()
        .column_vector_mut()
        .iter_mut()
        .for_each(|x| *x += 1f64);

    let mean = spd_mean(cov_draws, cov_grads)?;
    let mean_eig = mean.self_adjoint_eigen(faer::Side::Lower).ok()?;

    Some((
        mean_eig.S().column_vector().to_owned(),
        mean_eig.U().to_owned(),
    ))
}

fn spd_mean(cov_draws: Mat<f64>, cov_grads: Mat<f64>) -> Option<Mat<f64>> {
    let eigs_grads = cov_grads.self_adjoint_eigen(faer::Side::Lower).ok()?;

    let u = eigs_grads.U();
    let eigs = eigs_grads.S().column_vector().to_owned();

    let mut eigs_sqrt = eigs.clone();
    eigs_sqrt.iter_mut().for_each(|v| *v = v.sqrt());
    let cov_grads_sqrt = u * eigs_sqrt.into_diagonal() * u.transpose();

    let m = (&cov_grads_sqrt) * cov_draws * cov_grads_sqrt;
    let m_eig = m.self_adjoint_eigen(faer::Side::Lower).ok()?;

    let m_u = m_eig.U();
    let mut m_s = m_eig.S().column_vector().to_owned();
    m_s.iter_mut().for_each(|v| *v = v.sqrt());
    let m_sqrt = m_u * m_s.into_diagonal() * m_u.transpose();

    let mut eigs_grads_inv = eigs;
    eigs_grads_inv
        .iter_mut()
        .for_each(|v| *v = v.sqrt().recip());
    let grads_inv_sqrt = u * eigs_grads_inv.into_diagonal() * u.transpose();

    Some((&grads_inv_sqrt) * m_sqrt * grads_inv_sqrt)
}

impl<M: Math> SamplerStats<M> for LowRankMassMatrixStrategy {
    type Stats = ();
    type StatsOptions = ();

    fn extract_stats(&self, _math: &mut M, _opt: Self::StatsOptions) -> Self::Stats {}
}

impl<M: Math> MassMatrixAdaptStrategy<M> for LowRankMassMatrixStrategy {
    type Transformation = LowRankMassMatrix<M>;
    type Collector = DrawGradCollector<M>;
    type Options = LowRankSettings;

    fn new(math: &mut M, options: Self::Options, _num_tune: u64, _chain: u64) -> Self {
        Self::new(math.dim(), options)
    }

    fn init<R: rand::Rng + ?Sized>(
        &mut self,
        math: &mut M,
        _options: &mut crate::nuts::NutsOptions,
        mass_matrix: &mut Self::Transformation,
        point: &impl Point<M>,
        _rng: &mut R,
    ) -> Result<(), NutsError> {
        self.add_draw(math, point);
        mass_matrix.update_from_grad(
            math,
            point.position(),
            point.gradient(),
            1f64,
            (1e-20, 1e20),
        );
        Ok(())
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        DrawGradCollector::new(math)
    }

    fn update_estimators(&mut self, math: &mut M, collector: &Self::Collector) {
        if collector.is_good {
            let mut draw = vec![0f64; self.ndim];
            math.write_to_slice(&collector.draw, &mut draw);
            self.draws.push_back(draw);

            let mut grad = vec![0f64; self.ndim];
            math.write_to_slice(&collector.grad, &mut grad);
            self.grads.push_back(grad);
        }
    }

    fn switch(&mut self, _math: &mut M) {
        for _ in 0..self.background_split {
            self.draws.pop_front().expect("Could not drop draw");
            self.grads.pop_front().expect("Could not drop gradient");
        }
        self.background_split = self.draws.len();
        assert!(self.draws.len() == self.grads.len());
    }

    fn current_count(&self) -> u64 {
        self.draws.len() as u64
    }

    fn background_count(&self) -> u64 {
        self.draws.len().checked_sub(self.background_split).unwrap() as u64
    }

    fn adapt(&self, math: &mut M, mass_matrix: &mut Self::Transformation) -> bool {
        if <LowRankMassMatrixStrategy as MassMatrixAdaptStrategy<M>>::current_count(self) < 3 {
            return false;
        }
        self.update(math, mass_matrix);
        true
    }
}

#[cfg(test)]
mod test {
    use std::ops::AddAssign;

    use equator::Cmp;
    use faer::{Col, Mat, utils::approx::ApproxEq};
    use rand::{RngExt, SeedableRng, rngs::SmallRng};
    use rand_distr::StandardNormal;

    use crate::transform::low_rank::mat_all_finite;

    use super::{estimate_mass_matrix, spd_mean};

    #[test]
    fn test_spd_mean() {
        let x_diag = faer::col![1., 4., 8.];
        let y_diag = faer::col![1., 1., 0.5];

        let mut x = faer::Mat::zeros(3, 3);
        let mut y = faer::Mat::zeros(3, 3);

        x.diagonal_mut().column_vector_mut().add_assign(x_diag);
        y.diagonal_mut().column_vector_mut().add_assign(y_diag);

        let out = spd_mean(x, y).expect("Failed to compute spd mean");
        let expected_diag = faer::col![1., 2., 4.];
        let mut expected = faer::Mat::zeros(3, 3);
        expected
            .diagonal_mut()
            .column_vector_mut()
            .add_assign(expected_diag);

        let comp = ApproxEq {
            abs_tol: 1e-10,
            rel_tol: 1e-10,
        };

        faer::zip!(&out, &expected).for_each(|faer::unzip!(out, expected)| {
            assert!(comp.test(out, expected));
        });
    }

    #[test]
    fn test_estimate_mass_matrix() {
        let distr = StandardNormal;

        let mut rng = SmallRng::seed_from_u64(1);

        let draws: Mat<f64> = Mat::from_fn(20, 3, |_, _| rng.sample(distr));
        let grads = -(&draws);

        let (vals, vecs) =
            estimate_mass_matrix(draws, grads, 0.0001).expect("Failed to compute mass matrix");
        assert!(vals.iter().cloned().all(|x| x > 0.));
        assert!(mat_all_finite(&vecs.as_ref()));

        let comp = ApproxEq {
            abs_tol: 1e-5,
            rel_tol: 1e-5,
        };

        let expected = Col::full(20, 1.);

        faer::zip!(&vals, &expected).for_each(|faer::unzip!(out, expected)| {
            assert!(comp.test(out, expected));
        });
    }
}
