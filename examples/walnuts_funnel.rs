//! Sample Neal's funnel with plain NUTS and with WALNUTS and compare them.
//!
//! The funnel has a log-scale parameter `v ~ N(0, 3)` and parameters
//! `x_i ~ N(0, exp(v / 2))`. The curvature in the neck of the funnel (small
//! `v`) is much larger than in the mouth, so a single step size does not fit
//! everywhere: plain NUTS diverges in the neck and underexplores it. WALNUTS
//! splits steps where needed.
use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::{Result, bail};
use nuts_rs::{
    CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HashMapConfig, HashMapResult,
    HashMapValue, InitPositionError, LogpError, Model, Sampler, SamplerWaitResult, WalnutsOptions,
};
use nuts_storable::HasDims;
use rand::{Rng, RngExt};
use thiserror::Error;

/// Total number of parameters: `v` and four `x_i`.
const DIM: usize = 5;

#[derive(Debug, Error)]
enum FunnelError {}

impl LogpError for FunnelError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

#[derive(Clone)]
struct FunnelLogp;

impl HasDims for FunnelLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            ("unconstrained_parameter".to_string(), DIM as u64),
            ("dim".to_string(), DIM as u64),
        ])
    }
}

impl CpuLogpFunc for FunnelLogp {
    type LogpError = FunnelError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        DIM
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, FunnelError> {
        let v = position[0];
        let xs = &position[1..];
        let n = xs.len() as f64;
        let inv_var = (-v).exp();

        // log N(v | 0, 3) + sum_i log N(x_i | 0, exp(v / 2)), up to constants
        let sum_sq: f64 = xs.iter().map(|x| x * x).sum();
        let logp = -v * v / 18.0 - 0.5 * sum_sq * inv_var - 0.5 * n * v;

        grad[0] = -v / 9.0 + 0.5 * sum_sq * inv_var - 0.5 * n;
        for (g, x) in grad[1..].iter_mut().zip(xs) {
            *g = -x * inv_var;
        }
        Ok(logp)
    }

    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        Ok(array.to_vec())
    }
}

struct FunnelModel;

impl Model for FunnelModel {
    type Math = CpuMath<FunnelLogp>;

    fn math<R: Rng + ?Sized>(self: Arc<Self>, _rng: &mut R) -> Result<Self::Math> {
        Ok(CpuMath::new(FunnelLogp))
    }

    fn init_position<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        _chain_id: u64,
        position: &mut [f64],
    ) -> Result<(), InitPositionError> {
        for p in position.iter_mut() {
            *p = rng.random_range(-1.0..1.0);
        }
        Ok(())
    }
}

fn sample(settings: DiagNutsSettings) -> Result<Vec<HashMapResult>> {
    let mut sampler = Sampler::new(
        Arc::new(FunnelModel),
        settings,
        HashMapConfig::new(),
        4,
        None,
    )?;
    loop {
        match sampler.wait_timeout(Duration::from_millis(100)) {
            SamplerWaitResult::Trace(traces) => return Ok(traces),
            SamplerWaitResult::Timeout(next) => sampler = next,
            SamplerWaitResult::Err(err, _) => return Err(err),
        }
    }
}

fn get<'a>(trace: &'a HashMapResult, name: &str) -> Result<&'a HashMapValue> {
    match trace.stats.get(name).or_else(|| trace.draws.get(name)) {
        Some(value) => Ok(value),
        None => bail!("Missing value {name} in trace"),
    }
}

/// Summary of the draws after tuning, over all chains.
struct Summary {
    draws: usize,
    divergences: usize,
    irreversible: usize,
    gradients: u64,
    v_mean: f64,
    v_sd: f64,
    v_below_minus_3: f64,
}

fn summarize(traces: &[HashMapResult]) -> Result<Summary> {
    let mut vs = vec![];
    let mut divergences = 0;
    let mut irreversible = 0;
    let mut gradients = 0;

    for trace in traces {
        let HashMapValue::Bool(tuning) = get(trace, "tuning")? else {
            bail!("tuning has unexpected type");
        };
        let HashMapValue::Bool(diverging) = get(trace, "diverging")? else {
            bail!("diverging has unexpected type");
        };
        let HashMapValue::U64(n_gradients) = get(trace, "n_gradients")? else {
            bail!("n_gradients has unexpected type");
        };
        let HashMapValue::F64(values) = get(trace, "value")? else {
            bail!("value has unexpected type");
        };
        // Only present if WALNUTS is enabled
        let irreversible_stops = match trace.stats.get("irreversible") {
            Some(HashMapValue::Bool(values)) => values.clone(),
            _ => vec![false; tuning.len()],
        };

        for (i, draw) in values.as_chunks::<DIM>().0.iter().enumerate() {
            if tuning[i] {
                continue;
            }
            vs.push(draw[0]);
            divergences += diverging[i] as usize;
            irreversible += irreversible_stops[i] as usize;
            gradients += n_gradients[i];
        }
    }

    let n = vs.len() as f64;
    let v_mean = vs.iter().sum::<f64>() / n;
    let v_sd = (vs.iter().map(|v| (v - v_mean).powi(2)).sum::<f64>() / n).sqrt();
    let v_below_minus_3 = vs.iter().filter(|&&v| v < -3.0).count() as f64 / n;

    Ok(Summary {
        draws: vs.len(),
        divergences,
        irreversible,
        gradients,
        v_mean,
        v_sd,
        v_below_minus_3,
    })
}

fn main() -> Result<()> {
    let nuts_settings = DiagNutsSettings {
        num_chains: 4,
        num_tune: 1000,
        num_draws: 2000,
        seed: 42,
        ..Default::default()
    };
    let walnuts_settings = DiagNutsSettings {
        walnuts: Some(WalnutsOptions::default()),
        ..nuts_settings
    };

    println!("Neal's funnel in {DIM} dimensions, 4 chains with 2000 draws each\n");
    println!(
        "{:<8} {:>11} {:>12} {:>14} {:>7} {:>6} {:>10}",
        "sampler", "divergences", "irreversible", "grads per draw", "mean v", "sd v", "P(v < -3)"
    );

    for (name, settings) in [("NUTS", nuts_settings), ("WALNUTS", walnuts_settings)] {
        let summary = summarize(&sample(settings)?)?;
        println!(
            "{:<8} {:>11} {:>12} {:>14.1} {:>7.3} {:>6.3} {:>10.3}",
            name,
            summary.divergences,
            summary.irreversible,
            summary.gradients as f64 / summary.draws as f64,
            summary.v_mean,
            summary.v_sd,
            summary.v_below_minus_3,
        );
    }
    // P(v < -3) for v ~ N(0, 3) is Phi(-1)
    println!(
        "{:<8} {:>11} {:>12} {:>14} {:>7.3} {:>6.3} {:>10.3}",
        "exact", "", "", "", 0.0, 3.0, 0.1587
    );

    Ok(())
}
