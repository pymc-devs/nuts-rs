//! HashMap storage implementation example for MCMC traces
use std::{
    f64,
    time::{Duration, Instant},
};

use anyhow::Result;
use nuts_rs::{
    CpuLogpFunc, CpuMath, CpuMathError, DiagGradNutsSettings, HashMapConfig, LogpError, Model,
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
    fn dim_sizes(&self) -> std::collections::HashMap<String, u64> {
        std::collections::HashMap::from([
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

/// Implementation of McmcModel for the HashMap backend
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
    // Create a 2D multivariate normal distribution
    let mean = vec![0.0, 0.0];
    let precision = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
    let mvn = MultivariateNormal::new(mean, precision);

    // Number of chains
    let num_chains = 4;

    // Configure number of draws
    let num_tune = 100;
    let num_draws = 200;

    // Configure MCMC settings
    let mut settings = DiagGradNutsSettings::default();
    settings.num_chains = num_chains as _;
    settings.num_tune = num_tune;
    settings.num_draws = num_draws as _;
    settings.seed = 42;

    let model = MvnModel {
        math: CpuMath::new(MvnLogp { model: mvn }),
    };

    // Create a new sampler with 4 threads
    let start = Instant::now();
    let trace_config = HashMapConfig::new();
    let mut sampler = Some(Sampler::new(model, settings, trace_config, 4, None)?);

    let mut num_progress_updates = 0;
    // Interleave progress updates with wait_timeout
    while let Some(sampler_) = sampler.take() {
        match sampler_.wait_timeout(Duration::from_millis(50)) {
            SamplerWaitResult::Trace(traces) => {
                println!("Sampling completed in {:?}", start.elapsed());

                // Process the HashMap results
                println!("\nProcessing HashMap storage results:");
                println!("Number of chains: {}", traces.len());

                for (chain_idx, chain_result) in traces.iter().enumerate() {
                    println!("\nChain {}:", chain_idx);

                    // Print stats information
                    println!("  Sampler stats variables:");
                    for (name, values) in &chain_result.stats {
                        match values {
                            nuts_rs::HashMapValue::F64(vec) => {
                                println!("    {}: {} samples (f64)", name, vec.len());
                                if !vec.is_empty() {
                                    println!("      First 5: {:?}", &vec[..vec.len().min(5)]);
                                }
                            }
                            nuts_rs::HashMapValue::Bool(vec) => {
                                println!("    {}: {} samples (bool)", name, vec.len());
                                if !vec.is_empty() {
                                    println!("      First 5: {:?}", &vec[..vec.len().min(5)]);
                                }
                            }
                            _ => println!("    {}: {} (other type)", name, "unknown length"),
                        }
                    }

                    // Print draws information
                    println!("  Draw variables:");
                    for (name, values) in &chain_result.draws {
                        match values {
                            nuts_rs::HashMapValue::F64(vec) => {
                                println!("    {}: {} scalar draws", name, vec.len());
                                if !vec.is_empty() {
                                    println!("      First 5: {:?}", &vec[..vec.len().min(5)]);
                                    if *name == "theta" && vec.len() >= 6 {
                                        // For multidimensional parameters stored as flattened arrays
                                        // Assume 2D parameter, so every 2 values form one draw
                                        println!("      Parameter structure (assuming 2D):");
                                        for i in (0..vec.len().min(10)).step_by(2) {
                                            if i + 1 < vec.len() {
                                                println!(
                                                    "        Draw {}: [{:.4}, {:.4}]",
                                                    i / 2,
                                                    vec[i],
                                                    vec[i + 1]
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            nuts_rs::HashMapValue::F32(vec) => {
                                println!("    {}: {} f32 draws", name, vec.len());
                            }
                            nuts_rs::HashMapValue::Bool(vec) => {
                                println!("    {}: {} bool draws", name, vec.len());
                            }
                            nuts_rs::HashMapValue::I64(vec) => {
                                println!("    {}: {} i64 draws", name, vec.len());
                            }
                            nuts_rs::HashMapValue::U64(vec) => {
                                println!("    {}: {} u64 draws", name, vec.len());
                            }
                            nuts_rs::HashMapValue::String(vec) => {
                                println!("    {}: {} string draws", name, vec.len());
                            }
                        }
                    }

                    // Calculate some basic statistics for theta parameter
                    if let Some(nuts_rs::HashMapValue::F64(param_samples)) =
                        chain_result.draws.get("theta")
                    {
                        if param_samples.len() >= 2 {
                            // Assuming 2D parameter stored as flattened array: [x0, y0, x1, y1, ...]
                            let x_values: Vec<f64> =
                                param_samples.iter().step_by(2).cloned().collect();
                            let y_values: Vec<f64> =
                                param_samples.iter().skip(1).step_by(2).cloned().collect();

                            if !x_values.is_empty() {
                                let mean_x = x_values.iter().sum::<f64>() / x_values.len() as f64;
                                println!("      theta[0] (x-component) mean: {:.4}", mean_x);
                            }
                            if !y_values.is_empty() {
                                let mean_y = y_values.iter().sum::<f64>() / y_values.len() as f64;
                                println!("      theta[1] (y-component) mean: {:.4}", mean_y);
                            }
                        }
                    }
                }
                break;
            }
            SamplerWaitResult::Timeout(mut sampler_) => {
                // Request progress update
                if num_progress_updates < 10 {
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

    println!("\nHashMap storage example completed!");
    println!("The results are stored in memory as HashMaps and can be easily processed in Rust.");

    Ok(())
}
