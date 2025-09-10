//! CSV backend example for MCMC trace storage
//!
//! This example demonstrates how to use the nuts-rs library with CSV storage
//! for running MCMC sampling on a multivariate normal distribution. It shows:
//!
//! - Setting up a custom probability model
//! - Configuring CSV storage for results
//! - Running multiple parallel chains
//! - Monitoring progress during sampling
//! - Saving results in CmdStan-compatible CSV format

use std::{
    collections::HashMap,
    f64,
    time::{Duration, Instant},
};

use anyhow::Result;
use nuts_rs::{
    CpuLogpFunc, CpuMath, CpuMathError, CsvConfig, DiagGradNutsSettings, LogpError, Model, Sampler,
    SamplerWaitResult, Storable,
};
use nuts_storable::{HasDims, Value};
use rand::Rng;
use thiserror::Error;

/// A multivariate normal distribution model
///
/// This represents a probability distribution with mean μ and precision matrix P,
/// where the log probability is: logp(x) = -0.5 * (x - μ)^T * P * (x - μ)
#[derive(Clone, Debug)]
struct MultivariateNormal {
    mean: Vec<f64>,
    precision: Vec<Vec<f64>>, // Inverse of covariance matrix
}

impl MultivariateNormal {
    fn new(mean: Vec<f64>, precision: Vec<Vec<f64>>) -> Self {
        Self { mean, precision }
    }
}

/// Custom error type for log probability calculations
///
/// MCMC samplers need to distinguish between recoverable errors (like numerical
/// issues that can be handled by rejecting the proposal) and non-recoverable
/// errors (like programming bugs that should stop sampling).
#[allow(dead_code)]
#[derive(Debug, Error)]
enum MyLogpError {
    #[error("Recoverable error in logp calculation: {0}")]
    Recoverable(String),
    #[error("Non-recoverable error in logp calculation: {0}")]
    NonRecoverable(String),
}

impl LogpError for MyLogpError {
    fn is_recoverable(&self) -> bool {
        matches!(self, MyLogpError::Recoverable(_))
    }
}

/// Implementation of the log probability function for multivariate normal
///
/// This struct contains the model parameters and implements the mathematical
/// operations needed for MCMC sampling: computing log probability and gradients.
#[derive(Clone)]
struct MvnLogp {
    model: MultivariateNormal,
}

impl HasDims for MvnLogp {
    /// Define dimension names and sizes for storage
    ///
    /// This tells the storage system what array dimensions to expect.
    /// These dimensions will be used to structure the output data.
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            // Dimension for the parameter vector (for CSV storage)
            ("param".to_string(), self.model.mean.len() as u64),
        ])
    }

    fn coords(&self) -> HashMap<String, nuts_storable::Value> {
        HashMap::from([(
            "param".to_string(),
            Value::Strings(vec!["mu1".to_string(), "mu2".to_string()]),
        )])
    }
}

/// Additional quantities computed from each sample
///
/// The `Storable` derive macro automatically generates code to store this
/// struct in the trace. The `dims` attribute specifies which dimension
/// each field should use.
#[derive(Storable)]
struct ExpandedDraw {
    /// Store the parameter values as a vector with dimension "param"
    #[storable(dims("param"))]
    parameters: Vec<f64>,
}

impl CpuLogpFunc for MvnLogp {
    type LogpError = MyLogpError;
    type FlowParameters = (); // No parameter transformations needed
    type ExpandedVector = ExpandedDraw;

    /// Return the dimensionality of the parameter space
    fn dim(&self) -> usize {
        self.model.mean.len()
    }

    /// Compute log probability and gradient
    ///
    /// This is the core mathematical function that MCMC uses to explore
    /// the parameter space. It computes both the log probability density
    /// and its gradient for efficient sampling with Hamiltonian Monte Carlo.
    fn logp(&mut self, x: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let n = x.len();

        // Compute (x - mean)
        let mut diff = vec![0.0; n];
        for i in 0..n {
            diff[i] = x[i] - self.model.mean[i];
        }

        let mut quad = 0.0;

        // Compute quadratic form and gradient: logp = -0.5 * diff^T * P * diff
        for i in 0..n {
            // Compute i-th component of P * diff
            let mut pdot = 0.0;
            for j in 0..n {
                let pij = self.model.precision[i][j];
                pdot += pij * diff[j];
                quad += diff[i] * pij * diff[j];
            }
            // Gradient of logp w.r.t. x_i: derivative of -0.5 * diff^T P diff is -(P * diff)_i
            grad[i] = -pdot;
        }

        Ok(-0.5 * quad)
    }

    /// Compute additional quantities from each sample
    ///
    /// This function is called for each accepted sample to compute derived
    /// quantities that should be stored in the trace. These might be
    /// transformed parameters, predictions, or other quantities of interest.
    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        // Return the parameter vector for CSV storage
        Ok(ExpandedDraw {
            parameters: array.to_vec(),
        })
    }

    fn vector_coord(&self) -> Option<Value> {
        Some(Value::Strings(vec!["mu1".to_string(), "mu2".to_string()]))
    }
}

/// The complete MCMC model
///
/// This struct implements the Model trait, which is the main interface
/// that samplers use. It provides access to the mathematical operations
/// and handles initialization of the sampling chains.
struct MvnModel {
    math: CpuMath<MvnLogp>,
}

impl Model for MvnModel {
    type Math<'model>
        = CpuMath<MvnLogp>
    where
        Self: 'model;

    fn math(&self) -> Result<Self::Math<'_>> {
        Ok(self.math.clone())
    }

    /// Generate random initial positions for the chain
    ///
    /// Good initialization is important for MCMC efficiency. The starting
    /// points should be in a reasonable region of the parameter space
    /// where the log probability is finite.
    fn init_position<R: Rng + ?Sized>(&self, rng: &mut R, position: &mut [f64]) -> Result<()> {
        // Initialize each parameter randomly in the range [-2, 2]
        // For this simple example, this should put us in a reasonable
        // region around the mode of the distribution
        for p in position.iter_mut() {
            *p = rng.random_range(-2.0..2.0);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Multivariate Normal MCMC with CSV Storage ===\n");

    // Create a 2D multivariate normal distribution
    // This creates a distribution with mean [0, 0] and precision matrix
    // [[1.0, 0.5], [0.5, 1.0]], which corresponds to some correlation
    // between the two parameters
    let mean = vec![0.0, 0.0];
    let precision = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
    let mvn = MultivariateNormal::new(mean, precision);

    println!("Model: 2D Multivariate Normal");
    println!("Mean: {:?}", mvn.mean);
    println!("Precision matrix: {:?}\n", mvn.precision);

    // Configure output location
    let output_path = "csv_output";
    println!("Output will be saved to: {}/\n", output_path);

    // Sampling configuration
    let num_chains = 4; // Run 4 parallel chains for better convergence assessment
    let num_tune = 500; // Warmup samples to tune the sampler
    let num_draws = 500; // Post-warmup samples to keep

    println!("Sampling configuration:");
    println!("  Chains: {}", num_chains);
    println!("  Warmup samples: {}", num_tune);
    println!("  Sampling draws: {}", num_draws);

    // Configure MCMC settings
    // DiagGradNutsSettings provides sensible defaults for the NUTS sampler
    let mut settings = DiagGradNutsSettings::default();
    settings.num_chains = num_chains as _;
    settings.num_tune = num_tune;
    settings.num_draws = num_draws as _;
    settings.seed = 54; // For reproducible results

    // Set up CSV storage
    // This will create one CSV file per chain in the specified directory
    let csv_config = CsvConfig::new(output_path)
        .with_precision(6) // 6 decimal places for floating point values
        .store_warmup(true); // Include warmup samples with negative sample IDs

    // Create the model instance
    let model = MvnModel {
        math: CpuMath::new(MvnLogp { model: mvn }),
    };

    // Start sampling
    println!("\nStarting MCMC sampling...\n");
    let start = Instant::now();

    // Create sampler with 4 worker threads
    // The sampler runs asynchronously, so we can monitor progress
    let mut sampler = Some(Sampler::new(model, settings, csv_config, 4, None)?);

    let mut num_progress_updates = 0;

    // Main sampling loop with progress monitoring
    // This demonstrates how to monitor long-running sampling jobs
    while let Some(sampler_) = sampler.take() {
        match sampler_.wait_timeout(Duration::from_millis(50)) {
            // Sampling completed successfully
            SamplerWaitResult::Trace(_) => {
                println!("✓ Sampling completed in {:?}", start.elapsed());
                println!("✓ Traces written to CSV format in '{}'", output_path);

                // List the output files
                if let Ok(entries) = std::fs::read_dir(output_path) {
                    println!("\nOutput files:");
                    for entry in entries.flatten() {
                        if let Some(name) = entry.file_name().to_str() {
                            if name.ends_with(".csv") {
                                println!("  - {}", name);
                            }
                        }
                    }
                }

                // Provide instructions for analysis
                println!("\n=== Next Steps ===");
                println!("The CSV files are compatible with CmdStan format and can be read by:");
                println!("  - R: posterior package, bayesplot, etc.");
                println!("  - Python: arviz.from_cmdstanpy() or pandas.read_csv()");
                println!("  - Stan ecosystem tools");
                println!("\nExample usage in Python:");
                println!("  import pandas as pd");
                println!("  import arviz as az");
                println!("  # Read individual chains");
                println!("  chain0 = pd.read_csv('{}/chain_0.csv')", output_path);
                println!("  # Or use arviz to read all chains");
                println!("  # (Note: arviz.from_cmdstanpy might need adaptation)");
                println!("\nExample usage in R:");
                println!("  library(posterior)");
                println!("  draws <- read_cmdstan_csv(c(");
                for i in 0..num_chains {
                    let comma = if i == num_chains - 1 { "" } else { "," };
                    println!("    '{}/chain_{}.csv'{}", output_path, i, comma);
                }
                println!("  ))");
                println!("  summarise_draws(draws)");
                break;
            }

            // Timeout - sampler is still running, show progress
            SamplerWaitResult::Timeout(mut sampler_) => {
                num_progress_updates += 1;
                println!("Progress update {}:", num_progress_updates);

                // Get current progress from all chains
                let progress = sampler_.progress()?;
                for (i, chain) in progress.iter().enumerate() {
                    let phase = if chain.tuning { "warmup" } else { "sampling" };
                    println!(
                        "  Chain {}: {} samples ({} divergences), step size: {:.6} [{}]",
                        i, chain.finished_draws, chain.divergences, chain.step_size, phase
                    );
                }
                println!(); // Add blank line for readability

                sampler = Some(sampler_);
            }

            // An error occurred during sampling
            SamplerWaitResult::Err(err, _) => {
                eprintln!("✗ Sampling failed: {}", err);
                return Err(err);
            }
        }
    }

    Ok(())
}
