![Workflow Status](https://github.com/pymc-devs/nuts-rs/actions/workflows/test.yml/badge.svg)
[![dependency status](https://deps.rs/repo/github/pymc-devs/nuts-rs/status.svg)](https://deps.rs/repo/github/pymc-devs/nuts-rs)

<!-- cargo-rdme start -->

Sample from posterior distributions using the No U-turn Sampler (NUTS).
For details see the original [NUTS paper](https://arxiv.org/abs/1111.4246)
and the more recent [introduction](https://arxiv.org/abs/1701.02434).

This crate was developed as a faster replacement of the sampler in PyMC,
to be used with the new numba backend of PyTensor. The python wrapper
for this sampler is [nutpie](https://github.com/pymc-devs/nutpie).

## Usage

```rust
use nuts_rs::{CpuLogpFunc, CpuMath, LogpError, DiagGradNutsSettings, Chain, SampleStats,
Settings};
use thiserror::Error;
use rand::thread_rng;

// Define a function that computes the unnormalized posterior density
// and its gradient.
#[derive(Debug)]
struct PosteriorDensity {}

// The density might fail in a recoverable or non-recoverable manner...
#[derive(Debug, Error)]
enum PosteriorLogpError {}
impl LogpError for PosteriorLogpError {
    fn is_recoverable(&self) -> bool { false }
}

impl CpuLogpFunc for PosteriorDensity {
    type LogpError = PosteriorLogpError;

    // Only used for transforming adaptation.
    type TransformParams = ();

    // We define a 10 dimensional normal distribution
    fn dim(&self) -> usize { 10 }

    // The normal likelihood with mean 3 and its gradient.
    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let mu = 3f64;
        let logp = position
            .iter()
            .copied()
            .zip(grad.iter_mut())
            .map(|(x, grad)| {
                let diff = x - mu;
                *grad = -diff;
                -diff * diff / 2f64
            })
            .sum();
        return Ok(logp)
    }
}

// We get the default sampler arguments
let mut settings = DiagGradNutsSettings::default();

// and modify as we like
settings.num_tune = 1000;
settings.maxdepth = 3;  // small value just for testing...

// We instanciate our posterior density function
let logp_func = PosteriorDensity {};
let math = CpuMath::new(logp_func);

let chain = 0;
let mut rng = thread_rng();
let mut sampler = settings.new_chain(0, math, &mut rng);

// Set to some initial position and start drawing samples.
sampler.set_position(&vec![0f64; 10]).expect("Unrecoverable error during init");
let mut trace = vec![];  // Collection of all draws
for _ in 0..2000 {
    let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");
    trace.push(draw);
}
```

Users can also implement the `Model` trait for more control and parallel sampling.

## Implementation details

This crate mostly follows the implementation of NUTS in [Stan](https://mc-stan.org) and
[PyMC](https://docs.pymc.io/en/v3/), only tuning of mass matrix and step size differs
somewhat.

<!-- cargo-rdme end -->
