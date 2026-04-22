//! Coordinate transformations that map the original parameter space to a whitened space for more efficient sampling.

mod adapt;
mod diagonal;
mod external;
mod low_rank;
mod transformation;

pub use adapt::DiagAdaptExpSettings;
pub(crate) use adapt::DiagAdaptStrategy;
pub use adapt::LowRankMassMatrixStrategy;
pub(crate) use adapt::MassMatrixAdaptStrategy;
pub(crate) use diagonal::DiagMassMatrix;
pub use external::ExternalTransformation;
pub(crate) use low_rank::LowRankMassMatrix;
pub use low_rank::LowRankSettings;
pub use transformation::Transformation;

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, error::Error, fmt::Display};

    use faer::{Col, Mat};
    use nuts_storable::{HasDims, Storable};

    use crate::{
        Math,
        math::{CpuLogpFunc, CpuMath, CpuMathError, LogpError},
        transform::{DiagMassMatrix, LowRankMassMatrix, LowRankSettings, Transformation},
    };

    // -----------------------------------------------------------------------
    // Minimal CpuLogpFunc for a multivariate normal N(0, Sigma)
    // -----------------------------------------------------------------------

    /// A 3-D multivariate normal with a fixed precision matrix P = Sigma^{-1}.
    /// logp(x) = -0.5 * x^T P x  (unnormalized; constant terms dropped)
    /// score(x) = -P x
    struct MvNormal {
        /// Precision matrix (inverse of covariance), stored row-major.
        precision: Mat<f64>,
        dim: usize,
    }

    impl MvNormal {
        fn new(precision: Mat<f64>) -> Self {
            let dim = precision.nrows();
            Self { precision, dim }
        }
    }

    #[derive(Debug)]
    struct NeverError;
    impl Display for NeverError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "never")
        }
    }
    impl Error for NeverError {}
    impl LogpError for NeverError {
        fn is_recoverable(&self) -> bool {
            false
        }
    }

    struct EmptyExpanded;
    impl Storable<MvNormal> for EmptyExpanded {
        fn names(_: &MvNormal) -> Vec<&str> {
            vec![]
        }
        fn item_type(_: &MvNormal, _: &str) -> nuts_storable::ItemType {
            unimplemented!()
        }
        fn dims<'a>(_: &'a MvNormal, _: &str) -> Vec<&'a str> {
            vec![]
        }
        fn get_all<'a>(
            &'a mut self,
            _: &'a MvNormal,
        ) -> Vec<(&'a str, Option<nuts_storable::Value>)> {
            vec![]
        }
    }

    impl HasDims for MvNormal {
        fn dim_sizes(&self) -> HashMap<String, u64> {
            let mut m = HashMap::new();
            m.insert("unconstrained_parameter".into(), self.dim as u64);
            m
        }
        fn coords(&self) -> HashMap<String, nuts_storable::Value> {
            HashMap::new()
        }
    }

    impl CpuLogpFunc for MvNormal {
        type LogpError = NeverError;
        type FlowParameters = ();
        type ExpandedVector = EmptyExpanded;

        fn dim(&self) -> usize {
            self.dim
        }

        fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, NeverError> {
            let x = Col::from_fn(self.dim, |i| position[i]);
            let px = &self.precision * &x;
            let logp = -0.5 * (&x).transpose() * &px;
            // score = -P x
            for i in 0..self.dim {
                gradient[i] = -px[i];
            }
            // Full normalisation: -0.5 * (d*log(2π) + log det Σ)
            //   log det Σ = -log det P = -Σ_i log(P_ii)  [diagonal P]
            let log_det_p: f64 = (0..self.dim).map(|i| self.precision[(i, i)].ln()).sum();
            let norm = -0.5 * (self.dim as f64 * std::f64::consts::TAU.ln() - log_det_p);
            Ok(logp + norm)
        }

        fn expand_vector<R: rand::Rng + ?Sized>(
            &mut self,
            _rng: &mut R,
            _array: &[f64],
        ) -> Result<EmptyExpanded, CpuMathError> {
            Ok(EmptyExpanded)
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_math(precision: Mat<f64>) -> CpuMath<MvNormal> {
        CpuMath::new(MvNormal::new(precision))
    }

    /// Read a Math vector into a plain Vec<f64>.
    fn read_vec(math: &mut CpuMath<MvNormal>, v: &Col<f64>) -> Vec<f64> {
        let mut out = vec![0f64; math.dim()];
        math.write_to_slice(v, &mut out);
        out
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (ai - bi).abs() <= tol,
                "index {i}: {ai} vs {bi} (tol {tol})"
            );
        }
    }

    /// Fully-normalised log-density of N(0, I) at z.
    fn standard_normal_logp(z: &[f64]) -> f64 {
        let d = z.len() as f64;
        -0.5 * (d * std::f64::consts::TAU.ln() + z.iter().map(|v| v * v).sum::<f64>())
    }

    // -----------------------------------------------------------------------
    // Diagonal tests
    // -----------------------------------------------------------------------

    /// Build a diagonal mass matrix whose scales exactly match the target
    /// standard deviations, so F^{-1}(x) should be a standard normal.
    ///
    /// Target: N(0, diag(sigma^2))
    /// Precision: diag(1/sigma^2)
    /// Optimal diagonal inv_stds: 1/sigma
    /// Optimal mean: 0 (zero-mean target, zero-mean gradient)
    ///
    /// We set the mass matrix directly via `update_diag_draw_grad` using the
    /// exact variances.
    #[test]
    fn test_diag_transform_position_and_gradient() {
        // Target: N(0, diag(1, 4, 9))  =>  sigma = [1, 2, 3], inv_stds = [1, 0.5, 1/3]
        let sigma2 = [1f64, 4., 9.];
        let mut precision = Mat::zeros(3, 3);
        for i in 0..3 {
            precision[(i, i)] = 1.0 / sigma2[i];
        }
        let mut math = make_math(precision);

        let mut mass = DiagMassMatrix::new(&mut math, false);

        // Supply exact draw variances and gradient variances so sigma^2* = sigma2.
        // draw_var = sigma2, grad_var = 1/sigma2  =>  sqrt(draw_var/grad_var) = sigma2
        let mut draw_var = math.new_array();
        let mut grad_var = math.new_array();
        let mut draw_mean = math.new_array();
        let mut grad_mean = math.new_array();
        math.read_from_slice(&mut draw_var, &sigma2);
        math.read_from_slice(&mut grad_var, &sigma2.map(|v| 1.0 / v));
        math.fill_array(&mut draw_mean, 0f64);
        math.fill_array(&mut grad_mean, 0f64);

        mass.update_diag_draw_grad(
            &mut math,
            &draw_mean,
            &grad_mean,
            &draw_var,
            &grad_var,
            None,
            (1e-20, 1e20),
        );

        // Test point x = [1, 2, 3]  (one sigma out in each dimension)
        let x = [1f64, 2., 3.];
        let mut untransformed_pos = math.new_array();
        let mut untransformed_grad = math.new_array();
        let mut transformed_pos = math.new_array();
        let mut transformed_grad = math.new_array();
        math.read_from_slice(&mut untransformed_pos, &x);

        let (logp, logdet) = mass
            .init_from_untransformed_position(
                &mut math,
                &untransformed_pos,
                &mut untransformed_grad,
                &mut transformed_pos,
                &mut transformed_grad,
            )
            .unwrap();

        // Adapted position: z = x / sigma = [1/1, 2/2, 3/3] = [1, 1, 1]
        let z = read_vec(&mut math, &transformed_pos);
        assert_close(&z, &[1.0, 1.0, 1.0], 1e-12);

        // Adapted gradient: beta = sigma * alpha = sigma * (-P x) = sigma * (-x/sigma^2) = -x/sigma = -[1,1,1]
        let beta = read_vec(&mut math, &transformed_grad);
        assert_close(&beta, &[-1.0, -1.0, -1.0], 1e-12);

        // logdet = sum(log(inv_stds)) = log(1) + log(0.5) + log(1/3)
        let expected_logdet: f64 = sigma2.iter().map(|&s| -(0.5 * s.ln())).sum();
        assert!(
            (logdet - expected_logdet).abs() < 1e-12,
            "logdet: {logdet} vs {expected_logdet}"
        );

        // The adapted log-density should equal a standard-normal log-density at z.
        // log p_adapted(z) = log p_x(F(z)) + log|det J_F|
        //                  = logp(x) + Σ log(σ_i)
        //                  = logp(x) - logdet          (logdet = Σ log(σ_i⁻¹) = -Σ log(σ_i))
        let logp_adapted = logp - logdet;
        let expected_logp_adapted = standard_normal_logp(&z);
        assert!(
            (logp_adapted - expected_logp_adapted).abs() < 1e-12,
            "adapted logp: {logp_adapted} vs {expected_logp_adapted}, {logp}, {logdet}"
        );
    }

    /// Round-trip: init_from_transformed_position recovers the original x and gradient.
    #[test]
    fn test_diag_round_trip() {
        let sigma2 = [2f64, 0.5, 3.];
        let mut precision = Mat::zeros(3, 3);
        for i in 0..3 {
            precision[(i, i)] = 1.0 / sigma2[i];
        }
        let mut math = make_math(precision);
        let mut mass = DiagMassMatrix::new(&mut math, false);

        let mut draw_var = math.new_array();
        let mut grad_var = math.new_array();
        let mut draw_mean = math.new_array();
        let mut grad_mean = math.new_array();
        math.read_from_slice(&mut draw_var, &sigma2);
        math.read_from_slice(&mut grad_var, &sigma2.map(|v| 1.0 / v));
        math.fill_array(&mut draw_mean, 0f64);
        math.fill_array(&mut grad_mean, 0f64);
        mass.update_diag_draw_grad(
            &mut math,
            &draw_mean,
            &grad_mean,
            &draw_var,
            &grad_var,
            None,
            (1e-20, 1e20),
        );

        // Forward pass
        let x_orig = [1.5f64, -0.3, 2.1];
        let mut untransformed_pos = math.new_array();
        let mut untransformed_grad = math.new_array();
        let mut transformed_pos = math.new_array();
        let mut transformed_grad = math.new_array();
        math.read_from_slice(&mut untransformed_pos, &x_orig);

        let (logp_fwd, logdet_fwd) = mass
            .init_from_untransformed_position(
                &mut math,
                &untransformed_pos,
                &mut untransformed_grad,
                &mut transformed_pos,
                &mut transformed_grad,
            )
            .unwrap();

        // Inverse pass: recover x from z
        let mut recovered_pos = math.new_array();
        let mut recovered_grad = math.new_array();
        let mut recovered_transformed_grad = math.new_array();

        let (logp_inv, logdet_inv) = mass
            .init_from_transformed_position(
                &mut math,
                &mut recovered_pos,
                &mut recovered_grad,
                &transformed_pos,
                &mut recovered_transformed_grad,
            )
            .unwrap();

        let x_recovered = read_vec(&mut math, &recovered_pos);
        assert_close(&x_recovered, &x_orig, 1e-12);
        assert!((logp_fwd - logp_inv).abs() < 1e-12, "logp mismatch");
        assert!((logdet_fwd - logdet_inv).abs() < 1e-12, "logdet mismatch");
    }

    /// With a non-zero mean the adapted position should be (x - mu) / sigma.
    #[test]
    fn test_diag_nonzero_mean() {
        let sigma2 = [4f64, 1., 9.];
        let mu = [3f64, -1., 2.];

        // Precision for N(mu, diag(sigma2)) — logp is the same up to constant
        let mut precision = Mat::zeros(3, 3);
        for i in 0..3 {
            precision[(i, i)] = 1.0 / sigma2[i];
        }
        let mut math = make_math(precision);
        let mut mass = DiagMassMatrix::new(&mut math, false);

        let mut draw_var = math.new_array();
        let mut grad_var = math.new_array();
        let mut draw_mean = math.new_array();
        let mut grad_mean = math.new_array();
        math.read_from_slice(&mut draw_var, &sigma2);
        math.read_from_slice(&mut grad_var, &sigma2.map(|v| 1.0 / v));
        // draw mean = mu, grad mean = 0 (scores have zero mean at the mode)
        math.read_from_slice(&mut draw_mean, &mu);
        math.fill_array(&mut grad_mean, 0f64);
        mass.update_diag_draw_grad(
            &mut math,
            &draw_mean,
            &grad_mean,
            &draw_var,
            &grad_var,
            None,
            (1e-20, 1e20),
        );

        // Evaluate at x = mu + sigma (one sigma above mean)
        let x: Vec<f64> = mu
            .iter()
            .zip(sigma2.iter())
            .map(|(&m, &s)| m + s.sqrt())
            .collect();
        let mut untransformed_pos = math.new_array();
        let mut transformed_pos = math.new_array();
        let mut untransformed_grad = math.new_array();
        let mut transformed_grad = math.new_array();
        math.read_from_slice(&mut untransformed_pos, &x);

        mass.init_from_untransformed_position(
            &mut math,
            &untransformed_pos,
            &mut untransformed_grad,
            &mut transformed_pos,
            &mut transformed_grad,
        )
        .unwrap();

        // z = (x - mu) / sigma = [1, 1, 1]
        let z = read_vec(&mut math, &transformed_pos);
        assert_close(&z, &[1.0, 1.0, 1.0], 1e-12);
    }

    // -----------------------------------------------------------------------
    // Low-rank tests
    // -----------------------------------------------------------------------

    /// Construct a LowRankMassMatrix that exactly represents a known covariance,
    /// then check that the transformation produces a standard normal.
    ///
    /// Target: N(0, Sigma) where Sigma has a non-trivial low-rank structure.
    /// We use Sigma = diag(sigma2) + v v^T for a single rank-1 perturbation v,
    /// but for simplicity we just use a diagonal Sigma and verify via
    /// the full adaptation path (update from exact draws/scores).
    #[test]
    fn test_lowrank_transform_position_and_gradient() {
        // Target: N(0, diag(1, 4, 9))
        let sigma2 = [1f64, 4., 9.];
        let mut precision = Mat::zeros(3, 3);
        for i in 0..3 {
            precision[(i, i)] = 1.0 / sigma2[i];
        }
        let mut math = make_math(precision);

        // Build the LowRankMassMatrix directly with known stds/vals/vecs.
        // For a purely diagonal target we expect the low-rank part to be trivial
        // (all eigenvalues ≈ 1 after diagonal rescaling), so we set the mass
        // matrix manually via the faer update path.
        let stds = Col::from_fn(3, |i| sigma2[i].sqrt());
        let mean = Col::zeros(3);
        // No low-rank correction: pass empty vals/vecs.
        let vals = Col::zeros(0);
        let vecs = Mat::zeros(3, 0);
        let mu = Col::zeros(3);

        let settings = LowRankSettings::default();
        let mut mass = LowRankMassMatrix::new(&mut math, settings);
        mass.update(&mut math, stds, mean, vals, vecs, mu);

        let x = [1f64, 2., 3.];
        let mut untransformed_pos = math.new_array();
        let mut untransformed_grad = math.new_array();
        let mut transformed_pos = math.new_array();
        let mut transformed_grad = math.new_array();
        math.read_from_slice(&mut untransformed_pos, &x);

        let (logp, logdet) = mass
            .init_from_untransformed_position(
                &mut math,
                &untransformed_pos,
                &mut untransformed_grad,
                &mut transformed_pos,
                &mut transformed_grad,
            )
            .unwrap();

        // Adapted position: z = x / sigma = [1, 1, 1]
        let z = read_vec(&mut math, &transformed_pos);
        assert_close(&z, &[1.0, 1.0, 1.0], 1e-12);

        // Adapted gradient: beta = sigma * alpha = sigma * (-x/sigma^2) = -x/sigma = [-1, -1, -1]
        let beta = read_vec(&mut math, &transformed_grad);
        assert_close(&beta, &[-1.0, -1.0, -1.0], 1e-12);

        // logdet = sum(log(1/sigma_i))
        let expected_logdet: f64 = sigma2.iter().map(|&s| -(0.5 * s.ln())).sum();
        assert!(
            (logdet - expected_logdet).abs() < 1e-12,
            "logdet: {logdet} vs {expected_logdet}"
        );

        // Adapted log-density = logp(x) - logdet should equal N(0,I) at z
        // (logdet = Σ log(σ_i⁻¹) = -log|det J_F|, so we subtract it to add log|det J_F|)
        let logp_adapted = logp - logdet;
        let expected_logp_adapted = standard_normal_logp(&z);
        assert!(
            (logp_adapted - expected_logp_adapted).abs() < 1e-12,
            "adapted logp: {logp_adapted} vs {expected_logp_adapted}"
        );
    }

    /// Low-rank round-trip: init_from_transformed_position recovers original x.
    #[test]
    fn test_lowrank_round_trip() {
        let sigma2 = [2f64, 0.5, 3.];
        let mut precision = Mat::zeros(3, 3);
        for i in 0..3 {
            precision[(i, i)] = 1.0 / sigma2[i];
        }
        let mut math = make_math(precision);

        let stds = Col::from_fn(3, |i| sigma2[i].sqrt());
        let mean = Col::zeros(3);
        let vals = Col::zeros(0);
        let vecs = Mat::zeros(3, 0);
        let mu = Col::zeros(3);
        let mut mass = LowRankMassMatrix::new(&mut math, LowRankSettings::default());
        mass.update(&mut math, stds, mean, vals, vecs, mu);

        let x_orig = [0.7f64, -1.2, 3.3];
        let mut untransformed_pos = math.new_array();
        let mut untransformed_grad = math.new_array();
        let mut transformed_pos = math.new_array();
        let mut transformed_grad = math.new_array();
        math.read_from_slice(&mut untransformed_pos, &x_orig);

        let (logp_fwd, logdet_fwd) = mass
            .init_from_untransformed_position(
                &mut math,
                &untransformed_pos,
                &mut untransformed_grad,
                &mut transformed_pos,
                &mut transformed_grad,
            )
            .unwrap();

        let mut recovered_pos = math.new_array();
        let mut recovered_grad = math.new_array();
        let mut recovered_transformed_grad = math.new_array();
        let (logp_inv, logdet_inv) = mass
            .init_from_transformed_position(
                &mut math,
                &mut recovered_pos,
                &mut recovered_grad,
                &transformed_pos,
                &mut recovered_transformed_grad,
            )
            .unwrap();

        let x_recovered = read_vec(&mut math, &recovered_pos);
        assert_close(&x_recovered, &x_orig, 1e-12);
        assert!((logp_fwd - logp_inv).abs() < 1e-12, "logp mismatch");
        assert!((logdet_fwd - logdet_inv).abs() < 1e-12, "logdet mismatch");
    }

    /// Low-rank with an actual non-trivial rank-1 correction.
    ///
    /// Target: N(0, Sigma) with Sigma diagonal but we supply a rank-1 eigenvector
    /// correction so that F = F2 ∘ F1 with:
    ///   F2: diagonal scaling by sigma_diag
    ///   F1: rank-1 scaling by lambda along u
    ///
    /// We pick u = e_1 (first basis vector), lambda = [4.0], sigma_diag = [1, 1, 1].
    /// Then F(y) = y + u*(sqrt(4)-1)*u^T*y = y + (sqrt(4)-1)*y[0]*e_1
    ///           = [2*y[0], y[1], y[2]]
    /// and F^{-1}(x) = [x[0]/2, x[1], x[2]].
    ///
    /// The mass matrix is M = diag(1,1,1) * (I + u*(1/4 - 1)*u^T) * diag(1,1,1)
    ///                      = I + u*(-3/4)*u^T
    /// and M^{-1} = diag(4, 1, 1).
    ///
    /// So this is actually a full preconditioner for N(0, diag(4, 1, 1)).
    #[test]
    fn test_lowrank_with_rank1_correction() {
        // Target precision: diag(1/4, 1, 1)  <=>  N(0, diag(4, 1, 1))
        let mut precision = Mat::zeros(3, 3);
        precision[(0, 0)] = 0.25;
        precision[(1, 1)] = 1.0;
        precision[(2, 2)] = 1.0;
        let mut math = make_math(precision);

        // sigma_diag = [1, 1, 1], lambda = [4.0], u = e_1
        let stds = Col::full(3, 1.0f64);
        let mean = Col::zeros(3);
        let vals = faer::col![4.0f64];
        let mut vecs = Mat::zeros(3, 1);
        let mu = Col::zeros(3);
        vecs[(0, 0)] = 1.0; // u = e_1

        let mut mass = LowRankMassMatrix::new(&mut math, LowRankSettings::default());
        mass.update(&mut math, stds, mean, vals, vecs, mu);

        // Test point x = [2, 1, 1]
        let x = [2f64, 1., 1.];
        let mut untransformed_pos = math.new_array();
        let mut untransformed_grad = math.new_array();
        let mut transformed_pos = math.new_array();
        let mut transformed_grad = math.new_array();
        math.read_from_slice(&mut untransformed_pos, &x);

        let (logp, logdet) = mass
            .init_from_untransformed_position(
                &mut math,
                &untransformed_pos,
                &mut untransformed_grad,
                &mut transformed_pos,
                &mut transformed_grad,
            )
            .unwrap();

        // F^{-1}(x) = (I + u*(lambda^{-1/2} - 1)*u^T) * (x * inv_stds)
        //   inv_stds = [1,1,1], so scaled = x = [2, 1, 1]
        //   lambda^{-1/2} = 1/sqrt(4) = 0.5
        //   u^T * scaled = 2  =>  correction = (0.5-1)*2 * e_1 = [-1, 0, 0]
        //   z = [2-1, 1, 1] = [1, 1, 1]
        let z = read_vec(&mut math, &transformed_pos);
        assert_close(&z, &[1.0, 1.0, 1.0], 1e-12);

        // J_F^T * alpha where alpha = score = -P*x = [-0.5, -1, -1]
        //   sigma * alpha = [1,1,1] * [-0.5, -1, -1] = [-0.5, -1, -1]
        //   (I + u*(sqrt(4)-1)*u^T) * [-0.5,-1,-1]
        //   u^T * [-0.5,-1,-1] = -0.5  =>  correction = (2-1)*(-0.5)*e_1 = [-0.5,0,0]
        //   beta = [-0.5-0.5, -1, -1] = [-1, -1, -1]
        let beta = read_vec(&mut math, &transformed_grad);
        assert_close(&beta, &[-1.0, -1.0, -1.0], 1e-12);

        // logdet = sum(log(inv_stds)) + (-0.5*log(lambda))
        //        = 0 + (-0.5*log(4)) = -log(2)
        let expected_logdet = -0.5f64 * 4f64.ln();
        assert!(
            (logdet - expected_logdet).abs() < 1e-12,
            "logdet: {logdet} vs {expected_logdet}"
        );

        // Adapted log-density should equal N(0,I) at z = [1,1,1]
        // log p_adapted(z) = log p_x(F(z)) + log|det J_F|
        //                  = logp(x) - logdet   (logdet = Σ log(σ_i⁻¹) + (-½ Σ log λ_i) = -log|det J_F|)
        let logp_adapted = logp - logdet;
        let expected_logp_adapted = standard_normal_logp(&z);
        assert!(
            (logp_adapted - expected_logp_adapted).abs() < 1e-12,
            "adapted logp: {logp_adapted} vs {expected_logp_adapted}"
        );

        // Round-trip
        let mut recovered_pos = math.new_array();
        let mut recovered_grad = math.new_array();
        let mut recovered_tgrad = math.new_array();
        mass.init_from_transformed_position(
            &mut math,
            &mut recovered_pos,
            &mut recovered_grad,
            &transformed_pos,
            &mut recovered_tgrad,
        )
        .unwrap();
        let x_rec = read_vec(&mut math, &recovered_pos);
        assert_close(&x_rec, &x, 1e-12);
    }

    /// Non-zero mean with low-rank: the translation is applied correctly.
    #[test]
    fn test_lowrank_nonzero_mean() {
        let sigma2 = [4f64, 1., 9.];
        let mu = [2f64, -1., 3.];
        let mut precision = Mat::zeros(3, 3);
        for i in 0..3 {
            precision[(i, i)] = 1.0 / sigma2[i];
        }
        let mut math = make_math(precision);

        let stds = Col::from_fn(3, |i| sigma2[i].sqrt());
        let mean = Col::from_fn(3, |i| mu[i]);
        let vals = Col::zeros(0);
        let vecs = Mat::zeros(3, 0);
        let mu_low_rank = Col::zeros(3);
        let mut mass = LowRankMassMatrix::new(&mut math, LowRankSettings::default());
        mass.update(&mut math, stds, mean, vals, vecs, mu_low_rank);

        // Evaluate at x = mu + sigma (one sigma above mean in each dim)
        let x: Vec<f64> = mu
            .iter()
            .zip(sigma2.iter())
            .map(|(&m, &s)| m + s.sqrt())
            .collect();
        let mut untransformed_pos = math.new_array();
        let mut transformed_pos = math.new_array();
        let mut untransformed_grad = math.new_array();
        let mut transformed_grad = math.new_array();
        math.read_from_slice(&mut untransformed_pos, &x);

        mass.init_from_untransformed_position(
            &mut math,
            &untransformed_pos,
            &mut untransformed_grad,
            &mut transformed_pos,
            &mut transformed_grad,
        )
        .unwrap();

        // z = (x - mu) / sigma = [1, 1, 1]
        let z = read_vec(&mut math, &transformed_pos);
        assert_close(&z, &[1.0, 1.0, 1.0], 1e-12);

        // Round-trip should recover x
        let mut recovered_pos = math.new_array();
        let mut recovered_grad = math.new_array();
        let mut recovered_tgrad = math.new_array();
        mass.init_from_transformed_position(
            &mut math,
            &mut recovered_pos,
            &mut recovered_grad,
            &transformed_pos,
            &mut recovered_tgrad,
        )
        .unwrap();
        let x_rec = read_vec(&mut math, &recovered_pos);
        assert_close(&x_rec, &x, 1e-12);
    }
}
