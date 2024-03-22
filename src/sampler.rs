use anyhow::{Context, Result};
use arrow2::array::Array;
use rand::{rngs::SmallRng, thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::ScopeFifo;
use std::{
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc, Mutex,
    },
    time::Duration,
};

use crate::{
    adapt_strategy::{GradDiagOptions, GradDiagStrategy},
    mass_matrix::DiagMassMatrix,
    math_base::Math,
    nuts::{Chain, NutsChain, NutsOptions, SamplerStatTrace},
    potential::EuclideanPotential,
};

pub trait Settings: Clone + Copy + Default + Sync + Send + 'static {
    type Chain<M: Math>: Chain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        math: M,
        rng: &mut R,
    ) -> Self::Chain<M>;

    fn hint_num_tune(&self) -> usize;
    fn hint_num_draws(&self) -> usize;
}

/// Settings for the NUTS sampler
#[derive(Clone, Copy)]
pub struct DiagGradNutsSettings {
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
    pub check_turning: bool,
}

impl Default for DiagGradNutsSettings {
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
            check_turning: true,
        }
    }
}

impl Settings for DiagGradNutsSettings {
    type Chain<M: Math> =
        NutsChain<M, EuclideanPotential<M, DiagMassMatrix<M>>, SmallRng, GradDiagStrategy<M>>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        rng: &mut R,
    ) -> Self::Chain<M> {
        use crate::nuts::AdaptStrategy;
        let num_tune = self.num_tune;
        let strategy = GradDiagStrategy::new(&mut math, self.mass_matrix_adapt, num_tune);
        let mass_matrix = DiagMassMatrix::new(
            &mut math,
            self.mass_matrix_adapt.mass_matrix_options.store_mass_matrix,
        );
        let max_energy_error = self.max_energy_error;
        let potential = EuclideanPotential::new(mass_matrix, max_energy_error, 1f64);

        let options = NutsOptions {
            maxdepth: self.maxdepth,
            store_gradient: self.store_gradient,
            store_divergences: self.store_divergences,
            store_unconstrained: self.store_unconstrained,
            check_turning: self.check_turning,
        };

        let rng = rand::rngs::SmallRng::from_rng(rng).expect("Could not seed rng");

        NutsChain::new(math, potential, strategy, options, rng, chain)
    }

    fn hint_num_tune(&self) -> usize {
        self.num_tune as _
    }

    fn hint_num_draws(&self) -> usize {
        self.num_draws as _
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
/*
/// Create a new sampler
pub fn new_sampler<'math, M: Math + 'math, R: Rng + ?Sized>(
    math: M,
    settings: DiagGradNutsSettings,
    chain: u64,
    rng: &'math mut R,
) -> impl Chain<M> + 'math {
    settings.new_chain(chain, math, rng)
}

pub fn sample_sequentially<'math, M: Math + 'math, R: Rng + ?Sized>(
    math: M,
    settings: DiagGradNutsSettings,
    start: &[f64],
    draws: u64,
    chain: u64,
    rng: &mut R,
) -> Result<
    impl Iterator<Item = Result<(Box<[f64]>, impl SampleStats + 'math), NutsError>> + 'math,
    NutsError,
> {
    let mut sampler = new_sampler(math, settings, chain, rng);
    sampler.set_position(start)?;
    Ok((0..draws).map(move |_| sampler.draw()))
}
*/

/// Initialize chains using uniform jitter around zero or some other provided value
pub struct JitterInitFunc {
    mu: Option<Box<[f64]>>,
}

impl Default for JitterInitFunc {
    fn default() -> Self {
        Self::new()
    }
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

pub trait Trace<M: Math, S: Settings>: Send + Sync {
    fn append_value(
        &mut self,
        info: &<S::Chain<M> as SamplerStatTrace<M>>::Stats,
        point: &[f64],
    ) -> Result<()>;
    fn finalize(self) -> Result<Box<dyn Array>>;
    fn inspect(&mut self) -> Result<Box<dyn Array>>;
}

pub trait Model: Send + Sync {
    type Math<'model>: Math
    where
        Self: 'model;
    type Trace<'model, S: Settings>: Trace<Self::Math<'model>, S>
    where
        Self: 'model;

    fn new_trace<'model, S: Settings, R: Rng + ?Sized>(
        &'model self,
        rng: &mut R,
        chain_id: u64,
        settings: &'model S,
    ) -> Result<Self::Trace<'model, S>>;
    fn math(&self) -> Result<Self::Math<'_>>;
    fn init_position<R: Rng + ?Sized>(&self, rng: &mut R, position: &mut [f64]) -> Result<()>;
}

struct ChainProgress {
    finished_draws: usize,
    total_draws: usize,
    divergences: usize,
    total_time: Duration,
    running: bool,
}

struct ChainOutput {
    draws: Box<dyn Array>,
    stats: Box<dyn Array>,
}

enum ChainCommand {
    Pause,
    Resume,
}

struct ChainProcess<'model, M, S>
where
    M: Model + 'model,
    S: Settings,
{
    chain_idx: usize,
    results: Receiver<Result<()>>,
    trace: Arc<Mutex<M::Trace<'model, S>>>,
}

impl<'scope, M: Model + 'scope, S: Settings> ChainProcess<'scope, M, S> {
    fn start(
        model: &'scope M,
        chain_id: u64,
        seed: Option<u64>,
        settings: &'scope S,
        scope: ScopeFifo<'scope>,
        updates: Sender<u64>,
    ) -> Result<Self> {
        let (result_tx, result_rx) = channel();

        let mut rng = if let Some(seed) = seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_rng(thread_rng())?
        };
        rng.set_stream(chain_id);

        let trace = Arc::new(Mutex::new(
            model
                .new_trace(&mut rng, chain_id, settings)
                .context("Failed to create trace object")?,
        ));

        let trace_inner = trace.clone();

        scope.spawn_fifo(move |_| {
            let trace = trace_inner;
            let settings = settings;

            let mut sample = move || {
                let logp = model.math().context("Failed to create model density")?;
                let dim = logp.dim();

                let mut sampler = settings.new_chain(chain_id, logp, &mut rng);

                let mut initval = vec![0f64; dim];
                // TODO maxtries
                let mut error = None;
                for _ in 0..500 {
                    model
                        .init_position(&mut rng, &mut initval)
                        .context("Failed to generate a new initial position")?;
                    if let Err(err) = sampler.set_position(&initval) {
                        error = Some(err);
                        continue;
                    }
                    error = None;
                    break;
                }

                if let Some(error) = error {
                    let error: anyhow::Error = error.into();
                    return Err(error.context("All initialization points failed"));
                }

                // TODO let the sampler decide
                let draws = settings.hint_num_tune() + settings.hint_num_draws();
                for _ in 0..draws {
                    let (point, info) = sampler.draw().unwrap();
                    trace
                        .lock()
                        .expect("Could not unlock trace lock. Poisoned mutex")
                        .append_value(info, &point)?;
                    // We do not handle this error. If the draws can not be send, this
                    // could for instance be because the main thread was interrupted.
                    // In this case we just want to return the draws we have so far.
                    let result = updates.send(chain_id);
                    if result.is_err() {
                        break;
                    }
                }
                Ok(())
            };

            result_tx.send(sample());
        });

        Ok(Self {
            chain_idx: chain_id as _,
            trace,
            results: result_rx,
        })
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
        cpu_math::CpuMath, sample_sequentially, sampler::Settings, test_logps::NormalLogp, Chain,
        DiagGradNutsSettings,
    };

    use anyhow::Result;
    use itertools::Itertools;
    use pretty_assertions::assert_eq;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn sample_chain() -> Result<()> {
        let logp = NormalLogp::new(10, 0.1);
        let math = CpuMath::new(logp);
        let mut settings = DiagGradNutsSettings::default();
        settings.num_tune = 100;
        settings.num_draws = 100;
        let start = vec![0.2; 10];

        let mut rng = StdRng::seed_from_u64(42);

        let mut chain = settings.new_chain(0, math, &mut rng);

        let (draw, info) = chain.draw()?;
        info.logp();
        info.draw();

        let chain = sample_sequentially(math, settings, &start, 200, 1, &mut rng).unwrap();
        let mut draws = chain.collect_vec();
        assert_eq!(draws.len(), 200);

        let draw0 = draws.remove(100).unwrap();
        let (vals, stats) = draw0;
        assert_eq!(vals.len(), 10);
        assert_eq!(stats.chain(), 1);
        assert_eq!(stats.draw(), 100);
        Ok(())
    }

    #[test]
    fn sample_seq() {
        let logp = NormalLogp::new(10, 0.1);
        let math = CpuMath::new(logp);
        let mut settings = DiagGradNutsSettings::default();
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
