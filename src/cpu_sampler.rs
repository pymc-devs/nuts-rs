use rand::{prelude::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::thread::JoinHandle;
use thiserror::Error;

use crate::{
    adapt_strategy::{
        DualAverageSettings, GradDiagStrategy, GradDiagOptions
    },
    cpu_potential::EuclideanPotential,
    mass_matrix::DiagMassMatrix,
    nuts::{Chain, NutsChain, NutsError, NutsOptions, SampleStats},
    CpuLogpFunc,
};

/// Settings for the NUTS sampler
#[derive(Clone, Copy)]
pub struct SamplerArgs {
    /// The number of tuning steps, where we fit the step size and mass matrix.
    pub num_tune: u64,
    /// The maximum tree depth during sampling. The number of leapfrog steps
    /// is smaller than 2 ^ maxdepth.
    pub maxdepth: u64,
    /// Store the gradient in the SampleStats
    pub store_gradient: bool,
    /// If the energy error is larger than this threshold we treat the leapfrog
    /// step as a divergence.
    pub max_energy_error: f64,
    /// Settings for step size adaptation.
    pub step_size_adapt: DualAverageSettings,
    /// Settings for mass matrix adaptation.
    pub mass_matrix_adapt: GradDiagOptions,
}

impl Default for SamplerArgs {
    fn default() -> Self {
        Self {
            num_tune: 1000,
            maxdepth: 10,
            max_energy_error: 1000f64,
            store_gradient: false,
            step_size_adapt: DualAverageSettings::default(),
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

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum ParallelSamplingError {
    #[error("Could not send sample to controller thread")]
    ChannelClosed(),
    #[error("Nuts failed because of unrecoverable logp function error: {source}")]
    NutsError {
        #[from]
        source: NutsError,
    },
    #[error("Initialization of first point failed")]
    InitError { source: NutsError },
    #[error("Timeout occured while waiting for next sample")]
    Timeout,
    #[error("Drawing sample paniced")]
    Panic,
    #[error("Creating a logp function failed")]
    LogpFuncCreation {
        #[from]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

pub type ParallelChainResult = Result<(), ParallelSamplingError>;

pub trait CpuLogpFuncMaker: Send + Sync {
    type Func: CpuLogpFunc;

    fn make_logp_func(&self) -> Result<Self::Func, Box<dyn std::error::Error + Send + Sync>>;
    fn dim(&self) -> usize;
}

/// Sample several chains in parallel and return all of the samples live in a channel
pub fn sample_parallel<F: CpuLogpFuncMaker + 'static, I: InitPointFunc>(
    logp_func_maker: F,
    init_point_func: &mut I,
    settings: SamplerArgs,
    n_chains: u64,
    n_draws: u64,
    seed: u64,
    n_try_init: u64,
) -> Result<
    (
        JoinHandle<Vec<ParallelChainResult>>,
        crossbeam::channel::Receiver<(Box<[f64]>, Box<dyn SampleStats>)>,
    ),
    ParallelSamplingError,
> {
    let ndim = logp_func_maker.dim();
    let mut func = logp_func_maker.make_logp_func()?;
    assert!(ndim == func.dim());
    let draws = settings.num_tune + n_draws;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_sub(1));
    let mut points: Vec<Result<(Box<[f64]>, Box<[f64]>), <F::Func as CpuLogpFunc>::Err>> = (0
        ..n_chains)
        .map(|_| {
            let mut position = vec![0.; ndim];
            let mut grad = vec![0.; ndim];
            init_point_func.new_init_point(&mut rng, &mut position);

            let mut error = None;
            for _ in 0..n_try_init {
                match func.logp(&mut position, &mut grad) {
                    Err(e) => error = Some(e),
                    Ok(_) => {
                        error = None;
                        break;
                    }
                }
            }
            match error {
                Some(e) => Err(e),
                None => Ok((position.into(), grad.into())),
            }
        })
        .collect();

    let points: Result<Vec<(Box<[f64]>, Box<[f64]>)>, _> = points.drain(..).collect();
    let points = points.map_err(|e| ParallelSamplingError::InitError {
        source: NutsError::LogpFailure(Box::new(e)),
    })?;

    let (sender, receiver) = crossbeam::channel::bounded(128);

    let handle = std::thread::spawn(move || {
        let results: Vec<Result<(), ParallelSamplingError>> = points
            .into_par_iter()
            .with_max_len(1)
            .enumerate()
            .map_with(sender, |sender, (chain, point)| {
                let func = logp_func_maker.make_logp_func()?;
                let mut sampler = new_sampler(
                    func,
                    settings,
                    chain as u64,
                    seed.wrapping_add(chain as u64),
                );
                sampler.set_position(&point.0)?;
                for _ in 0..draws {
                    let (point2, info) = sampler.draw()?;
                    sender
                        .send((point2, Box::new(info) as Box<dyn SampleStats>))
                        .map_err(|_| ParallelSamplingError::ChannelClosed())?;
                }
                Ok(())
            })
            .collect();
        results
    });

    Ok((handle, receiver))
}

/// Create a new sampler
pub fn new_sampler<F: CpuLogpFunc>(
    logp: F,
    settings: SamplerArgs,
    chain: u64,
    seed: u64,
) -> impl Chain {
    use crate::nuts::AdaptStrategy;
    let num_tune = settings.num_tune;
    //let step_size_adapt = DualAverageStrategy::new(settings.step_size_adapt, num_tune, logp.dim());
    //let mass_matrix_adapt =
    //    ExpWindowDiagAdapt::new(settings.mass_matrix_adapt, num_tune, logp.dim());

    let strategy = GradDiagStrategy::new(settings.mass_matrix_adapt, num_tune, logp.dim());

    let mass_matrix = DiagMassMatrix::new(logp.dim());
    let max_energy_error = settings.max_energy_error;
    let potential = EuclideanPotential::new(logp, mass_matrix, max_energy_error, 1f64);

    let options = NutsOptions {
        maxdepth: settings.maxdepth,
        store_gradient: settings.store_gradient,
    };

    //let rng = { rand::rngs::StdRng::seed_from_u64(seed) };
    let rng = rand::rngs::SmallRng::seed_from_u64(seed);

    NutsChain::new(potential, strategy, options, rng, chain)
}

pub fn sample_sequentially<F: CpuLogpFunc>(
    logp: F,
    settings: SamplerArgs,
    start: &[f64],
    draws: u64,
    chain: u64,
    seed: u64,
) -> Result<impl Iterator<Item = Result<(Box<[f64]>, impl SampleStats), NutsError>>, NutsError> {
    let mut sampler = new_sampler(logp, settings, chain, seed);
    sampler.set_position(start)?;
    Ok((0..draws).into_iter().map(move |_| sampler.draw()))
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
    use crate::{cpu_potential::CpuLogpFunc, nuts::LogpError, CpuLogpFuncMaker};
    use multiversion::multiversion;
    use thiserror::Error;

    #[derive(Clone)]
    pub struct NormalLogp {
        dim: usize,
        mu: f64,
    }

    impl CpuLogpFuncMaker for NormalLogp {
        type Func = Self;

        fn make_logp_func(&self) -> Result<Self::Func, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.clone())
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
        type Err = NormalLogpError;

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
    use std::error::Error;

    use crate::{
        sample_parallel, sample_sequentially, test_logps::NormalLogp, CpuLogpFunc,
        CpuLogpFuncMaker, JitterInitFunc, SampleStats, SamplerArgs,
    };

    use itertools::Itertools;
    use pretty_assertions::assert_eq;

    #[test]
    fn sample_seq() {
        let logp = NormalLogp::new(10, 0.1);
        let mut settings = SamplerArgs::default();
        settings.num_tune = 100;
        let start = vec![0.2; 10];

        let chain = sample_sequentially(logp.clone(), settings, &start, 200, 1, 42).unwrap();
        let mut draws = chain.collect_vec();
        assert_eq!(draws.len(), 200);

        let draw0 = draws.remove(100).unwrap();
        let (vals, stats) = draw0;
        assert_eq!(vals.len(), 10);
        assert_eq!(stats.chain(), 1);
        assert_eq!(stats.draw(), 100);
        assert!(stats
            .to_vec()
            .iter()
            .any(|(key, _)| *key == "index_in_trajectory"));

        let maker = logp;

        let (handles, chains) =
            sample_parallel(maker, &mut JitterInitFunc::new(), settings, 4, 100, 42, 10).unwrap();
        let mut draws = chains.iter().collect_vec();
        assert_eq!(draws.len(), 800);
        assert!(handles.join().is_ok());

        let draw0 = draws.remove(100);
        let (vals, stats) = draw0;
        assert_eq!(vals.len(), 10);
        assert!(stats
            .to_vec()
            .iter()
            .any(|(key, _)| *key == "index_in_trajectory"));
    }
}
