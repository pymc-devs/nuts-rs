use rand::{Rng, SeedableRng};

use crate::{
    adapt_strategy::{GradDiagOptions, GradDiagStrategy},
    mass_matrix::DiagMassMatrix,
    math_base::Math,
    nuts::{Chain, NutsChain, NutsError, NutsOptions, SampleStats},
    potential::EuclideanPotential,
};

/// Settings for the NUTS sampler
#[derive(Clone, Copy)]
pub struct SamplerArgs {
    /// The number of tuning steps, where we fit the step size and mass matrix.
    pub num_tune: u64,
    /// The number of draws after tuning
    pub num_draws: u64,
    /// The maximum tree depth during sampling. The number of leapfrog steps
    /// is smaller than 2 ^ maxdepth.
    pub maxdepth: u64,
    /// Store the gradient in the SampleStats
    pub store_gradient: bool,
    /// Store each unconstrained parameter vector in the sampler stats
    pub store_unconstrained: bool,
    /// If the energy error is larger than this threshold we treat the leapfrog
    /// step as a divergence.
    pub max_energy_error: f64,
    /// Store detailed information about each divergence in the sampler stats
    pub store_divergences: bool,
    /// Settings for mass matrix adaptation.
    pub mass_matrix_adapt: GradDiagOptions,
}

impl Default for SamplerArgs {
    fn default() -> Self {
        Self {
            num_tune: 300,
            num_draws: 1000,
            maxdepth: 10,
            max_energy_error: 1000f64,
            store_gradient: false,
            store_unconstrained: false,
            store_divergences: false,
            mass_matrix_adapt: GradDiagOptions::default(),
        }
    }
}

/// Propose new initial points for a sampler
///
/// This trait can be implemented by users to control how the different
/// chains should be initialized when using [`sample_parallel`].
pub trait InitPointFunc {
    fn new_init_point<R: Rng + ?Sized>(&mut self, rng: &mut R, out: &mut [f64]);
}

pub trait MathMaker: Send + Sync {
    type Math: Math;

    fn make_math(&self, id: usize) -> Result<Self::Math, <Self::Math as Math>::Err>;
    fn dim(&self) -> usize;
}

/// Create a new sampler
pub fn new_sampler<'math, M: Math + 'math, R: Rng + ?Sized>(
    mut math: M,
    settings: SamplerArgs,
    chain: u64,
    rng: &mut R,
) -> impl Chain<M> + 'math {
    use crate::nuts::AdaptStrategy;
    let num_tune = settings.num_tune;
    let strategy = GradDiagStrategy::new(&mut math, settings.mass_matrix_adapt, num_tune);
    let mass_matrix = DiagMassMatrix::new(&mut math);
    let max_energy_error = settings.max_energy_error;
    let potential = EuclideanPotential::new(mass_matrix, max_energy_error, 1f64);

    let options = NutsOptions {
        maxdepth: settings.maxdepth,
        store_gradient: settings.store_gradient,
        store_unconstrained: settings.store_unconstrained,
    };

    let rng = rand::rngs::SmallRng::from_rng(rng).expect("Could not seed rng");

    NutsChain::new(math, potential, strategy, options, rng, chain)
}

pub fn sample_sequentially<'math, M: Math + 'math, R: Rng + ?Sized>(
    math: M,
    settings: SamplerArgs,
    start: &[f64],
    draws: u64,
    chain: u64,
    rng: &mut R,
) -> Result<impl Iterator<Item = Result<(Box<[f64]>, impl SampleStats + 'math), NutsError>> + 'math, NutsError> {
    let mut sampler = new_sampler(math, settings, chain, rng);
    sampler.set_position(start)?;
    Ok((0..draws).map(move |_| sampler.draw()))
}

/// Initialize chains using uniform jitter around zero or some other provided value
pub struct JitterInitFunc {
    mu: Option<Box<[f64]>>,
}

impl JitterInitFunc {
    /// Initialize new chains with jitter in [-1, 1] around zero
    pub fn new() -> JitterInitFunc {
        JitterInitFunc { mu: None }
    }

    /// Initialize new chains with jitter in [mu - 1, mu + 1].
    pub fn new_with_mean(mu: Box<[f64]>) -> Self {
        Self { mu: Some(mu) }
    }
}

impl InitPointFunc for JitterInitFunc {
    fn new_init_point<R: Rng + ?Sized>(&mut self, rng: &mut R, out: &mut [f64]) {
        rng.fill(out);
        if self.mu.is_none() {
            out.iter_mut().for_each(|val| *val = 2. * *val - 1.);
        } else {
            let mu = self.mu.as_ref().unwrap();
            out.iter_mut()
                .zip(mu.iter().copied())
                .for_each(|(val, mu)| *val = 2. * *val - 1. + mu);
        }
    }
}

pub mod test_logps {
    use crate::{
        cpu_math::{CpuLogpFunc, CpuMath},
        math_base::Math,
        nuts::LogpError,
        MathMaker,
    };
    use multiversion::multiversion;
    use thiserror::Error;

    #[derive(Clone, Debug)]
    pub struct NormalLogp {
        dim: usize,
        mu: f64,
    }

    impl MathMaker for NormalLogp {
        type Math = CpuMath<NormalLogp>;

        fn make_math(
            &self,
            _chain: usize,
        ) -> Result<CpuMath<NormalLogp>, <Self::Math as Math>::Err> {
            Ok(CpuMath::new(self.clone()))
        }

        fn dim(&self) -> usize {
            self.dim
        }
    }

    impl NormalLogp {
        pub fn new(dim: usize, mu: f64) -> NormalLogp {
            NormalLogp { dim, mu }
        }
    }

    #[derive(Error, Debug)]
    pub enum NormalLogpError {}
    impl LogpError for NormalLogpError {
        fn is_recoverable(&self) -> bool {
            false
        }
    }

    impl CpuLogpFunc for NormalLogp {
        type LogpError = NormalLogpError;

        fn dim(&self) -> usize {
            self.dim
        }
        fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, NormalLogpError> {
            let n = position.len();
            assert!(gradient.len() == n);

            #[cfg(feature = "simd_support")]
            #[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
            fn logp_inner(mu: f64, position: &[f64], gradient: &mut [f64]) -> f64 {
                use std::simd::f64x4;
                use std::simd::SimdFloat;

                let n = position.len();
                assert!(gradient.len() == n);

                let head_length = n - n % 4;

                let (pos, pos_tail) = position.split_at(head_length);
                let (grad, grad_tail) = gradient.split_at_mut(head_length);

                let mu_splat = f64x4::splat(mu);

                let mut logp = f64x4::splat(0f64);

                for (p, g) in pos.chunks_exact(4).zip(grad.chunks_exact_mut(4)) {
                    let p = f64x4::from_slice(p);
                    let val = mu_splat - p;
                    logp = logp - val * val * f64x4::splat(0.5);
                    g.copy_from_slice(&val.to_array());
                }

                let mut logp_tail = 0f64;
                for (p, g) in pos_tail.iter().zip(grad_tail.iter_mut()).take(3) {
                    let val = mu - p;
                    logp_tail -= val * val / 2.;
                    *g = val;
                }

                logp.reduce_sum() + logp_tail
            }

            #[cfg(not(feature = "simd_support"))]
            #[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
            fn logp_inner(mu: f64, position: &[f64], gradient: &mut [f64]) -> f64 {
                let n = position.len();
                assert!(gradient.len() == n);

                let mut logp = 0f64;
                for (p, g) in position.iter().zip(gradient.iter_mut()) {
                    let val = mu - p;
                    logp -= val * val / 2.;
                    *g = val;
                }

                logp
            }

            let logp = logp_inner(self.mu, position, gradient);

            Ok(logp)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cpu_math::CpuMath, sample_sequentially, test_logps::NormalLogp, SampleStats, SamplerArgs,
    };

    use itertools::Itertools;
    use pretty_assertions::assert_eq;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn sample_seq() {
        let logp = NormalLogp::new(10, 0.1);
        let math = CpuMath::new(logp);
        let mut settings = SamplerArgs::default();
        settings.num_tune = 100;
        settings.num_draws = 100;
        let start = vec![0.2; 10];

        let mut rng = StdRng::seed_from_u64(42);

        let chain = sample_sequentially(math, settings, &start, 200, 1, &mut rng).unwrap();
        let mut draws = chain.collect_vec();
        assert_eq!(draws.len(), 200);

        let draw0 = draws.remove(100).unwrap();
        let (vals, stats) = draw0;
        assert_eq!(vals.len(), 10);
        assert_eq!(stats.chain(), 1);
        assert_eq!(stats.draw(), 100);
    }
}
