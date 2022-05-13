use itertools::Itertools;
use rand::{prelude::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::thread::{spawn, JoinHandle};
use thiserror::Error;

use crate::{
    adapt_strategy::{CombinedStrategy, DualAverageStrategy, ExpWindowDiagAdapt},
    cpu_potential::EuclideanPotential,
    mass_matrix::{DiagAdaptExpSettings, DiagMassMatrix},
    nuts::{NutsError, NutsOptions, NutsChain, SampleStats, Chain},
    stepsize::DualAverageSettings,
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
    /// If the energy error is larger than this threshold we treat the leapfrog
    /// step as a divergence.
    pub max_energy_error: f64,
    /// Settings for step size adaptation.
    pub step_size_adapt: DualAverageSettings,
    /// Settings for mass matrix adaptation.
    pub mass_matrix_adapt: DiagAdaptExpSettings,
}

impl Default for SamplerArgs {
    fn default() -> Self {
        Self {
            num_tune: 1000,
            maxdepth: 10,
            max_energy_error: 1000f64,
            step_size_adapt: DualAverageSettings::default(),
            mass_matrix_adapt: DiagAdaptExpSettings::default(),
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

#[derive(Error, Debug)]
pub enum ParallelSamplingError {
    #[error("Could not send sample to controller thread")]
    ChannelClosed(),
    #[error("Nuts failed because of unrecoverable logp function error")]
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
}

pub type ParallelChainResult = Result<(), ParallelSamplingError>;

/*
pub type Draw = (Box<[f64]>, Box<dyn SampleStats>);

pub trait Sampler {
    fn next_draw(&mut self) -> Result<Draw, ParallelSamplingError>;
    fn next_draw_timeout(&mut self, timeout: Duration) -> Result<Draw, ParallelSamplingError>;
}


struct ParallelSampler<F: CpuLogpFunc + Clone + Send + 'static> {

}
*/


/// Sample several chains in parallel and return all of the samples live in a channel
pub fn sample_parallel<F: CpuLogpFunc + Clone + Send + 'static, I: InitPointFunc>(
    func: F,
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
    let mut func = func;
    let draws = settings.num_tune + n_draws;
    let mut grad = vec![0.; func.dim()];
    let mut rng = StdRng::seed_from_u64(seed.wrapping_sub(1));
    let mut points: Vec<Result<Box<[f64]>, F::Err>> = (0..n_chains)
        .map(|_| {
            let mut position = vec![0.; func.dim()];
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
                None => Ok(position.into()),
            }
        })
        .collect();

    let points: Result<Vec<Box<[f64]>>, _> = points.drain(..).collect();
    let points = points.map_err(|e| NutsError::LogpFailure(Box::new(e)))?;

    let (sender, receiver) = crossbeam::channel::bounded(128);

    let funcs = (0..n_chains).map(|_| func.clone()).collect_vec();
    let handle = spawn(move || {
        let results: Vec<Result<(), ParallelSamplingError>> = points
            .into_par_iter()
            .zip(funcs)
            //.with_max_len(1)
            .enumerate()
            .map_with(sender, |sender, (chain, (point, func))| {
                let mut sampler = new_sampler(func, settings, chain as u64, seed);
                sampler.set_position(&point)?;
                //sampler
                //    .set_position(&point)
                //    .expect("Could not eval logp at initial positon, though we could previously.");
                for _ in 0..draws {
                    let (point, info) = sampler.draw()?;
                    sender
                        .send((point, Box::new(info) as Box<dyn SampleStats>))
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
    let step_size_adapt = DualAverageStrategy::new(settings.step_size_adapt, num_tune, logp.dim());
    let mass_matrix_adapt =
        ExpWindowDiagAdapt::new(settings.mass_matrix_adapt, num_tune, logp.dim());

    let strategy = CombinedStrategy::new(step_size_adapt, mass_matrix_adapt);

    let mass_matrix = DiagMassMatrix::new(vec![1f64; logp.dim()].into());
    let max_energy_error = settings.max_energy_error;
    let step_size = settings.step_size_adapt.initial_step;

    let potential = EuclideanPotential::new(logp, mass_matrix, max_energy_error, step_size);
    let options = NutsOptions {
        maxdepth: settings.maxdepth,
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
) -> Result<
    impl Iterator<
        Item = Result<(Box<[f64]>, impl SampleStats),
        NutsError>
    >,
    NutsError
>
{
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
    use crate::{cpu_potential::CpuLogpFunc, nuts::LogpError};
    use multiversion::multiversion;
    use thiserror::Error;

    #[derive(Clone)]
    pub struct NormalLogp {
        dim: usize,
        mu: f64,
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

            #[multiversion]
            #[clone(target = "[x64|x86_64]+avx+avx2+fma")]
            #[clone(target = "x86+sse")]
            fn logp_inner(mu: f64, position: &[f64], gradient: &mut [f64]) -> f64 {
                use std::simd::f64x4;

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

            let logp = logp_inner(self.mu, position, gradient);

            Ok(logp)
        }
    }
}

/*
#[cfg(test)]
mod tests {
    use crate::{
        cpu_sampler::SamplerArgs,
        mass_matrix::DiagAdaptExpSettings,
    };

    use super::test_logps::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn make_state() {
        let _ = State::new(10);
        let mut storage = Rc::new(StateStorage::with_capacity(0));
        let a = new_state(&mut storage, 10);
        assert!(storage.free_states.borrow_mut().len() == 0);
        drop(a);
        assert!(storage.free_states.borrow_mut().len() == 1);
        let a = new_state(&mut storage, 10);
        assert!(storage.free_states.borrow_mut().len() == 0);
        drop(a);
        assert!(storage.free_states.borrow_mut().len() == 1);
    }

    #[test]
    fn deterministic() {
        let dim = 3usize;
        let func = NormalLogp::new(dim, 3.);
        let init = vec![3.5; dim];

        let settings = DiagAdaptExpSettings::default();
        let mut sampler = EuclideanSampler::new(func, SamplerArgs::default(), settings, 10, 42);

        sampler.set_position(&init).unwrap();
        let (sample1, stats1) = sampler.draw().unwrap();

        let func = NormalLogp::new(dim, 3.);
        let mut sampler = EuclideanSampler::new(func, SamplerArgs::default(), settings, 10, 42);

        sampler.set_position(&init).unwrap();
        let (sample2, stats2) = sampler.draw().unwrap();

        dbg!(&sample1);
        dbg!(stats1);

        dbg!(&sample2);
        dbg!(stats2);

        assert_eq!(sample1, sample2);
    }
}
*/
