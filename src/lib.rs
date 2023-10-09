#![cfg_attr(feature = "simd_support", feature(stdsimd))]
#![cfg_attr(feature = "simd_support", feature(portable_simd))]
#![cfg_attr(feature = "simd_support", feature(slice_as_chunks))]
//! Sample from posterior distributions using the No U-turn Sampler (NUTS).
//! For details see the original [NUTS paper](https://arxiv.org/abs/1111.4246)
//! and the more recent [introduction](https://arxiv.org/abs/1701.02434).
//!
//! This crate was developed as a faster replacement of the sampler in PyMC,
//! to be used with the new numba backend of aesara. The python wrapper
//! for this sampler is [nutpie](https://github.com/pymc-devs/nutpie).
//!
//! ## Usage
//!
//! ```
//! use nuts_rs::{CpuLogpFunc, LogpError, new_sampler, SamplerArgs, Chain, SampleStats};
//! use thiserror::Error;
//! use rand::thread_rng;
//!
//! // Define a function that computes the unnormalized posterior density
//! // and its gradient.
//! struct PosteriorDensity {}
//!
//! // The density might fail in a recoverable or non-recoverable manner...
//! #[derive(Debug, Error)]
//! enum PosteriorLogpError {}
//! impl LogpError for PosteriorLogpError {
//!     fn is_recoverable(&self) -> bool { false }
//! }
//!
//! impl CpuLogpFunc for PosteriorDensity {
//!     type Err = PosteriorLogpError;
//!
//!     // We define a 10 dimensional normal distribution
//!     fn dim(&self) -> usize { 10 }
//!
//!     // The normal likelihood with mean 3 and its gradient.
//!     fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::Err> {
//!         let mu = 3f64;
//!         let logp = position
//!             .iter()
//!             .copied()
//!             .zip(grad.iter_mut())
//!             .map(|(x, grad)| {
//!                 let diff = x - mu;
//!                 *grad = -diff;
//!                 -diff * diff / 2f64
//!             })
//!             .sum();
//!         return Ok(logp)
//!     }
//! }
//!
//! // We get the default sampler arguments
//! let mut sampler_args = SamplerArgs::default();
//!
//! // and modify as we like
//! sampler_args.num_tune = 1000;
//! sampler_args.maxdepth = 3;  // small value just for testing...
//!
//! // We instanciate our posterior density function
//! let logp_func = PosteriorDensity {};
//!
//! let chain = 0;
//! let mut rng = thread_rng();
//! let mut sampler = new_sampler(logp_func, sampler_args, chain, &mut rng);
//!
//! // Set to some initial position and start drawing samples.
//! sampler.set_position(&vec![0f64; 10]).expect("Unrecoverable error during init");
//! let mut trace = vec![];  // Collection of all draws
//! for _ in 0..2000 {
//!     let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");
//!     trace.push(draw);
//!     // Or get more detailed information about divergences
//!     if let Some(div_info) = info.divergence_info() {
//!         println!("Divergence at position {:?}", div_info.start_location);
//!     }
//!     dbg!(&info);
//! }
//! ```
//!
//! Sampling several chains in parallel so that samples are accessable as they are generated
//! is implemented in [`sample_parallel`].
//!
//! ## Implementation details
//!
//! This crate mostly follows the implementation of NUTS in [Stan](https://mc-stan.org) and
//! [PyMC](https://docs.pymc.io/en/v3/), only tuning of mass matrix and step size differs
//! somewhat.

pub(crate) mod adapt_strategy;
pub(crate) mod cpu_potential;
pub(crate) mod cpu_sampler;
pub(crate) mod cpu_state;
pub(crate) mod mass_matrix;
pub mod math;
pub(crate) mod nuts;
pub(crate) mod stepsize;

pub use adapt_strategy::DualAverageSettings;
pub use cpu_potential::CpuLogpFunc;
pub use cpu_sampler::test_logps;
pub use cpu_sampler::{
    new_sampler, sample_parallel, sample_sequentially, CpuLogpFuncMaker, InitPointFunc,
    JitterInitFunc, ParallelChainResult, ParallelSamplingError, SamplerArgs,
};
#[cfg(feature = "arrow")]
pub use nuts::{ArrowBuilder, ArrowRow};
pub use nuts::{Chain, DivergenceInfo, LogpError, NutsError, SampleStats};
