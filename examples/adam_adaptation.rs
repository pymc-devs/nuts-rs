//! Example demonstrating the Adam optimizer for step size adaptation.
//!
//! This example shows how to use the Adam optimizer instead of dual averaging
//! for adapting the step size in NUTS.

use nuts_rs::{
    AdamOptions, Chain, CpuLogpFunc, CpuMath, DiagGradNutsSettings, LogpError, Settings,
    StepSizeAdaptMethod,
};
use nuts_storable::HasDims;
use thiserror::Error;

// Define a function that computes the unnormalized posterior density
// and its gradient.
#[derive(Debug)]
struct PosteriorDensity {}

// The density might fail in a recoverable or non-recoverable manner...
#[derive(Debug, Error)]
enum PosteriorLogpError {}
impl LogpError for PosteriorLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

impl HasDims for PosteriorDensity {
    fn dim_sizes(&self) -> std::collections::HashMap<String, u64> {
        vec![("unconstrained_parameter".to_string(), self.dim() as u64)]
            .into_iter()
            .collect()
    }
}

impl CpuLogpFunc for PosteriorDensity {
    type LogpError = PosteriorLogpError;
    type ExpandedVector = Vec<f64>;

    // Only used for transforming adaptation.
    type FlowParameters = ();

    // We define a 10 dimensional normal distribution
    fn dim(&self) -> usize {
        10
    }

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
        return Ok(logp);
    }

    fn expand_vector<R>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, nuts_rs::CpuMathError>
    where
        R: rand::Rng + ?Sized,
    {
        Ok(array.to_vec())
    }
}

fn main() {
    println!("Running NUTS with Adam step size adaptation...");

    // Create sampler settings with Adam optimizer
    let mut settings = DiagGradNutsSettings::default();

    // Configure for Adam adaptation
    settings
        .adapt_options
        .step_size_settings
        .adapt_options
        .method = StepSizeAdaptMethod::Adam;

    // Set Adam options
    let adam_options = AdamOptions {
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        learning_rate: 0.05,
    };

    settings.adapt_options.step_size_settings.adapt_options.adam = adam_options;

    // Standard MCMC settings
    settings.num_tune = 1000;
    settings.num_draws = 1000;
    settings.maxdepth = 10;

    // Create the posterior density function
    let logp_func = PosteriorDensity {};
    let math = CpuMath::new(logp_func);

    // Initialize the sampler
    let chain = 0;
    let mut rng = rand::rng();
    let mut sampler = settings.new_chain(chain, math, &mut rng);

    // Set initial position
    let initial_position = vec![0f64; 10];
    sampler
        .set_position(&initial_position)
        .expect("Unrecoverable error during init");

    // Collect samples
    let mut trace = vec![];
    let mut stats = vec![];

    // Sampling with progress reporting
    println!("Warmup phase:");
    for i in 0..settings.num_tune {
        if i % 100 == 0 {
            println!("\rWarmup: {}/{}", i, settings.num_tune);
        }

        let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");
        println!("{:?}", info.step_size);
        trace.push(draw);
        stats.push(info);
    }
    println!("\rWarmup: {}/{}", settings.num_tune, settings.num_tune);

    println!("\nSampling phase:");
    for i in 0..settings.num_draws {
        if i % 100 == 0 {
            print!("\rSampling: {}/{}", i, settings.num_draws);
        }

        let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");
        trace.push(draw);
        stats.push(info);
    }
    println!("\rSampling: {}/{}", settings.num_draws, settings.num_draws);

    // Calculate mean of samples (post-warmup)
    let warmup_samples = settings.num_tune as usize;
    let mut means = vec![0.0; 10];

    for i in warmup_samples..trace.len() {
        for (j, mean) in means.iter_mut().enumerate() {
            *mean += trace[i][j];
        }
    }

    for mean in &mut means {
        *mean /= settings.num_draws as f64;
    }

    // Print results
    println!("\nResults after {} samples:", settings.num_draws);
    println!("Target mean: 3.0 for all dimensions");
    println!("Estimated means:");
    for (i, mean) in means.iter().enumerate() {
        println!("Dimension {}: {:.4}", i, mean);
    }

    // Print adaptation statistics
    let last_stats = &stats[stats.len() - 1];
    println!("\nFinal adaptation statistics:");
    println!("Step size: {:.6}", last_stats.step_size);
    // Note: the full acceptance stats are in the Progress struct, but we don't have direct access to mean_tree_accept
    println!("Number of steps: {}", last_stats.num_steps);

    println!("\nSampling completed successfully!");
}
