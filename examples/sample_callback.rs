//! Example demonstrating sample-level data access via ProgressCallback
//!
//! This example shows how to access per-sample data through the existing
//! ProgressCallback using the `latest_sample` field in ChainProgress.

use std::{
    f64,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::Result;
use nuts_rs::{
    CpuLogpFunc, CpuMath, CpuMathError, DiagGradNutsSettings, HashMapConfig, LogpError, Model,
    ProgressCallback, Sampler,
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

/// Implementation of Model for the HashMap backend
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
    println!("=== Sample-Level Data via ProgressCallback Example ===\n");
    println!("This example demonstrates accessing per-sample data through ProgressCallback.");
    println!("The callback fires periodically (rate-limited to 10ms) with chain progress,");
    println!("including the latest sample data for each chain.\n");

    // Create a 2D multivariate normal distribution
    let mean = vec![0.0, 0.0];
    let precision = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
    let mvn = MultivariateNormal::new(mean, precision);

    // Number of chains
    let num_chains = 2;

    // Configure number of draws
    let num_tune = 50;
    let num_draws = 100;

    // Configure MCMC settings
    let mut settings = DiagGradNutsSettings::default();
    settings.num_chains = num_chains as _;
    settings.num_tune = num_tune;
    settings.num_draws = num_draws as _;
    settings.seed = 42;

    let model = MvnModel {
        math: CpuMath::new(MvnLogp { model: mvn }),
    };

    // Track callback invocations for demonstration
    let callback_count = Arc::new(Mutex::new(0));
    let callback_count_clone = callback_count.clone();

    let divergence_count = Arc::new(Mutex::new(0));
    let divergence_count_clone = divergence_count.clone();

    // Create progress callback that accesses latest sample data
    let progress_callback = ProgressCallback {
        callback: Box::new(move |elapsed, chains| {
            let mut count = callback_count_clone.lock().unwrap();
            *count += 1;

            // Print progress information periodically
            if *count <= 10 {
                println!(
                    "Progress callback #{}: Elapsed: {:.1}s, {} chains",
                    count,
                    elapsed.as_secs_f64(),
                    chains.len()
                );

                for chain_progress in chains.iter() {
                    // Access the latest sample data if available
                    if let Some(sample_data) = &chain_progress.latest_sample {
                        // Demonstrate accessing optional fields with proper handling
                        let energy_str = sample_data
                            .draw_energy
                            .map(|e| format!("{:.3}", e))
                            .unwrap_or_else(|| "N/A".to_string());
                        let diverging_str = sample_data
                            .diverging
                            .map(|d| d.to_string())
                            .unwrap_or_else(|| "N/A".to_string());
                        let tree_depth_str = sample_data
                            .tree_depth
                            .map(|d| d.to_string())
                            .unwrap_or_else(|| "N/A".to_string());

                        println!(
                            "   Chain {}: Draw {}/{}, Energy: {}, Diverging: {}, Tree depth: {}",
                            sample_data.chain_id,
                            chain_progress.finished_draws,
                            chain_progress.total_draws,
                            energy_str,
                            diverging_str,
                            tree_depth_str
                        );

                        if let Some(step_size) = sample_data.step_size {
                            println!(
                                "   Step size: {:.6}, Tuning: {}",
                                step_size, sample_data.is_tuning
                            );
                        }

                        if let Some(max_depth) = sample_data.reached_max_treedepth {
                            if max_depth {
                                println!("   ⚠ Maximum tree depth reached!");
                            }
                        }

                        // Track divergences
                        if sample_data.diverging.unwrap_or(false) {
                            let mut div_count = divergence_count_clone.lock().unwrap();
                            *div_count += 1;
                        }
                    }
                }
                println!();
            } else if *count == 11 {
                println!("   ... (suppressing further callback output) ...\n");
            }
        }),
        rate: Duration::from_millis(10), // Rate limit: at most one callback per 10ms
    };

    // Create a new sampler with the progress callback
    let trace_config = HashMapConfig::new();
    let mut sampler = Sampler::new(
        model,
        settings,
        trace_config,
        4,                       // num_cores
        Some(progress_callback), // progress callback with sample data access
    )?;

    println!("Starting sampling with progress callback...\n");

    // Wait for sampling to complete
    let traces = loop {
        match sampler.wait_timeout(std::time::Duration::from_millis(100)) {
            nuts_rs::SamplerWaitResult::Trace(traces) => break traces,
            nuts_rs::SamplerWaitResult::Timeout(s) => sampler = s,
            nuts_rs::SamplerWaitResult::Err(e, _) => return Err(e),
        }
    };

    println!("\n=== Sampling Complete ===");
    println!(
        "Total callback invocations: {}",
        *callback_count.lock().unwrap()
    );
    println!(
        "Divergences detected via callback: {}",
        *divergence_count.lock().unwrap()
    );
    println!("Number of chains: {}", traces.len());

    // Show some basic statistics from the traces
    for (chain_idx, chain_result) in traces.iter().enumerate() {
        println!("\nChain {}:", chain_idx);

        // Count divergences from stats
        if let Some(nuts_rs::HashMapValue::Bool(divergences)) = chain_result.stats.get("diverging")
        {
            let div_count = divergences.iter().filter(|&&d| d).count();
            println!("  Divergences in trace: {}", div_count);
        }

        // Calculate mean position
        if let Some(nuts_rs::HashMapValue::F64(positions)) = chain_result.draws.get("theta") {
            if positions.len() >= 2 {
                let x_mean: f64 =
                    positions.iter().step_by(2).sum::<f64>() / (positions.len() / 2) as f64;
                let y_mean: f64 =
                    positions.iter().skip(1).step_by(2).sum::<f64>() / (positions.len() / 2) as f64;
                println!("  Mean position: [{:.4}, {:.4}]", x_mean, y_mean);
            }
        }
    }

    println!("\n✓ Example completed successfully!");
    println!("\nKey features demonstrated:");
    println!("  - ProgressCallback provides both chain progress and latest sample data");
    println!("  - Time-based rate limiting (10ms) prevents excessive overhead");
    println!(
        "  - latest_sample includes rich optional data (energy, divergence, tree depth, etc.)"
    );
    println!("  - All sampler-specific stats are Option<T> for compatibility with other samplers");
    println!("  - Works seamlessly with multi-chain sampling");
    println!("  - Single callback mechanism for all monitoring needs");

    Ok(())
}
