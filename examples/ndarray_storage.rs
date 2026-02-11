//! ndarray storage implementation example for MCMC traces
use std::{
    collections::HashMap,
    f64,
    time::{Duration, Instant},
};

use anyhow::Result;
use nuts_rs::{
    CpuLogpFunc, CpuMath, CpuMathError, DiagGradNutsSettings, LogpError, Model, NdarrayConfig,
    Sampler, SamplerWaitResult,
};
use nuts_storable::HasDims;
use rand::{Rng, RngExt};
use thiserror::Error;

// A simple multivariate normal distribution example
#[derive(Clone, Debug)]
struct MultivariateNormal {
    mean: Vec<f64>,
    precision: Vec<Vec<f64>>,
}

impl MultivariateNormal {
    fn new(mean: Vec<f64>, precision: Vec<Vec<f64>>) -> Self {
        Self { mean, precision }
    }
}

// Custom LogpError implementation
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

// Implementation of the model's logp function
#[derive(Clone)]
struct MvnLogp {
    model: MultivariateNormal,
}

impl HasDims for MvnLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            (
                "unconstrained_parameter".to_string(),
                self.model.mean.len() as u64,
            ),
            ("dim".to_string(), self.model.mean.len() as u64),
        ])
    }
}

impl CpuLogpFunc for MvnLogp {
    type LogpError = MyLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.model.mean.len()
    }

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
            // gradient of logp w.r.t. x_i: derivative of -0.5 * diff^T P diff is - (P * diff)_i
            grad[i] = -pdot;
        }

        Ok(-0.5 * quad)
    }

    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        // Simply return the parameter values
        Ok(array.to_vec())
    }
}

struct MvnModel {
    math: CpuMath<MvnLogp>,
}

/// Implementation of McmcModel for the ndarray backend
impl Model for MvnModel {
    type Math<'model>
        = CpuMath<MvnLogp>
    where
        Self: 'model;

    fn math<R: Rng + ?Sized>(&self, _rng: &mut R) -> Result<Self::Math<'_>> {
        Ok(self.math.clone())
    }

    /// Generate random initial positions for the chain
    fn init_position<R: Rng + ?Sized>(&self, rng: &mut R, position: &mut [f64]) -> Result<()> {
        // Initialize position randomly in [-2, 2]
        for p in position.iter_mut() {
            *p = rng.random_range(-2.0..2.0);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    // Create a 3D multivariate normal distribution for more interesting results
    let mean = vec![0.0, 1.0, -0.5];
    let precision = vec![
        vec![2.0, 0.3, 0.1],
        vec![0.3, 1.5, -0.2],
        vec![0.1, -0.2, 1.0],
    ];
    let mvn = MultivariateNormal::new(mean, precision);

    // Number of chains
    let num_chains = 3;

    // Configure number of draws
    let num_tune = 50;
    let num_draws = 100;

    // Configure MCMC settings
    let mut settings = DiagGradNutsSettings::default();
    settings.num_chains = num_chains as _;
    settings.num_tune = num_tune;
    settings.num_draws = num_draws as _;
    settings.seed = 123;

    let model = MvnModel {
        math: CpuMath::new(MvnLogp { model: mvn }),
    };

    // Create a new sampler with 3 threads
    let start = Instant::now();
    let trace_config = NdarrayConfig::new();
    let mut sampler = Some(Sampler::new(
        model,
        settings,
        trace_config,
        num_chains,
        None,
    )?);

    let mut num_progress_updates = 0;
    // Interleave progress updates with wait_timeout
    while let Some(sampler_) = sampler.take() {
        match sampler_.wait_timeout(Duration::from_millis(100)) {
            SamplerWaitResult::Trace(result) => {
                println!("Sampling completed in {:?}", start.elapsed());

                // Process the ndarray results
                println!("\nProcessing ndarray storage results:");

                // Print stats information
                println!("Sampler stats variables:");
                for (name, values) in &result.stats {
                    match values {
                        nuts_rs::NdarrayValue::F64(arr) => {
                            println!("  {}: shape {:?} (f64)", name, arr.shape());
                            if arr.len() > 0 {
                                // Print some sample values from the first chain
                                if arr.ndim() >= 2 {
                                    let chain_0_view = arr.slice(ndarray::s![0, ..5]);
                                    println!("    Chain 0, first 5 samples: {:?}", chain_0_view);
                                }
                            }
                        }
                        nuts_rs::NdarrayValue::Bool(arr) => {
                            println!("  {}: shape {:?} (bool)", name, arr.shape());
                            if arr.len() > 0 && arr.ndim() >= 2 {
                                let chain_0_view = arr.slice(ndarray::s![0, ..5]);
                                println!("    Chain 0, first 5 samples: {:?}", chain_0_view);
                            }
                        }
                        _ => println!("  {}: shape (other type)", name),
                    }
                }

                // Print draws information
                println!("\nDraw variables:");
                for (name, values) in &result.draws {
                    match values {
                        nuts_rs::NdarrayValue::F64(arr) => {
                            println!("  {}: shape {:?} (f64)", name, arr.shape());
                            if arr.len() > 0 {
                                // Print statistics for each parameter dimension
                                if arr.ndim() == 3 {
                                    // Shape is (chains, draws, parameters)
                                    let num_params = arr.shape()[2];
                                    for param_idx in 0..num_params {
                                        let param_slice = arr.slice(ndarray::s![.., .., param_idx]);
                                        let mean = param_slice.mean().unwrap_or(f64::NAN);
                                        let std = param_slice.std(0.0);
                                        println!(
                                            "    Parameter {}: mean={:.4}, std={:.4}",
                                            param_idx, mean, std
                                        );
                                    }

                                    // Print some sample values from each chain
                                    println!("    Sample values from each chain (first 3 draws):");
                                    for chain_idx in 0..(arr.shape()[0].min(3)) {
                                        let chain_samples =
                                            arr.slice(ndarray::s![chain_idx, ..3, ..]);
                                        println!("      Chain {}: {:?}", chain_idx, chain_samples);
                                    }
                                } else {
                                    // Just print overall mean if not the expected 3D shape
                                    let mean = arr.mean().unwrap_or(f64::NAN);
                                    println!("    Overall mean: {:.4}", mean);
                                }
                            }
                        }
                        _ => println!("  {}: (other type)", name),
                    }
                }

                // Demonstrate accessing individual samples
                if let Some(nuts_rs::NdarrayValue::F64(theta_arr)) = result.draws.get("theta") {
                    if theta_arr.ndim() == 3 && theta_arr.shape()[0] > 0 && theta_arr.shape()[1] > 0
                    {
                        println!("\nExample: Accessing specific samples:");

                        // Get the 10th sample from chain 0
                        if theta_arr.shape()[1] > 9 {
                            let sample = theta_arr.slice(ndarray::s![0, 9, ..]);
                            println!("  Chain 0, sample 10: {:?}", sample);
                        }

                        // Get all samples for parameter 0 from chain 1
                        if theta_arr.shape()[0] > 1 {
                            let param_0_chain_1 = theta_arr.slice(ndarray::s![1, .., 0]);
                            println!(
                                "  Chain 1, parameter 0, all samples: shape {:?}",
                                param_0_chain_1.shape()
                            );
                            println!(
                                "    First 5 values: {:?}",
                                param_0_chain_1.slice(ndarray::s![..5])
                            );
                        }
                    }
                }
                break;
            }
            SamplerWaitResult::Timeout(mut sampler_) => {
                // Request progress update
                if num_progress_updates < 5 {
                    // Limit progress updates
                    println!("Progress update {}", num_progress_updates + 1);
                    let progress = sampler_.progress()?;
                    for (i, chain) in progress.iter().enumerate() {
                        println!(
                            "Chain {}: {} samples ({} divergences), step size: {:.6}",
                            i, chain.finished_draws, chain.divergences, chain.step_size
                        );
                    }
                }
                sampler = Some(sampler_);
                num_progress_updates += 1;
            }
            SamplerWaitResult::Err(err, _) => {
                return Err(err);
            }
        }
    }

    println!("\nndarray storage example completed!");
    println!(
        "The results are stored as efficient ndarray structures with shape (chains, draws, parameters)."
    );
    println!(
        "This format is ideal for numerical analysis and can be easily converted to other formats."
    );

    Ok(())
}
