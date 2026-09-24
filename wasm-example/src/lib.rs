//! Using nuts-rs from JavaScript through `wasm32-unknown-unknown`.
//!
//! Without the `parallel` feature, `Sampler` runs every chain on the calling thread,
//! a draw at a time, while the caller is inside `wait_timeout`. Here the draws are stored
//! with the Arrow backend.
//!
//! The tests run in node:
//!
//! ```text
//! wasm-pack test --node wasm-example
//! ```
use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::{Context, Result, bail};
use arrow::array::{Array, Float64Array, LargeListArray};
use nuts_rs::{
    ArrowConfig, ArrowTrace, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, LogpError,
    Model, Sampler, SamplerWaitResult,
};
use nuts_storable::HasDims;
use rand::Rng;
use thiserror::Error;
use wasm_bindgen::prelude::*;

/// The mean of the normal distribution that `posterior_mean` samples.
pub const MEAN: f64 = 1.5;
const DIM: usize = 3;

#[derive(Debug, Error)]
#[error("never happens")]
struct NoError;

impl LogpError for NoError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

/// A standard normal around `MEAN` in `DIM` dimensions.
#[derive(Clone)]
struct NormalLogp;

impl HasDims for NormalLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            ("unconstrained_parameter".to_string(), DIM as u64),
            ("dim".to_string(), DIM as u64),
        ])
    }
}

impl CpuLogpFunc for NormalLogp {
    type LogpError = NoError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        DIM
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, NoError> {
        let mut logp = 0.0;
        for (x, grad) in position.iter().zip(gradient.iter_mut()) {
            let diff = x - MEAN;
            logp -= diff * diff / 2.0;
            *grad = -diff;
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

struct NormalModel;

impl Model for NormalModel {
    type Math = CpuMath<NormalLogp>;

    fn math<R: Rng + ?Sized>(self: Arc<Self>, _rng: &mut R) -> Result<Self::Math> {
        Ok(CpuMath::new(NormalLogp))
    }

    fn init_position<R: Rng + ?Sized>(&self, _rng: &mut R, position: &mut [f64]) -> Result<()> {
        position.fill(0.0);
        Ok(())
    }
}

/// Sample two chains, returning to the caller every 10ms the way a web page would to
/// keep its event loop running.
pub fn sample(seed: u64) -> Result<Vec<ArrowTrace>> {
    let settings = DiagNutsSettings {
        num_tune: 200,
        num_draws: 500,
        num_chains: 2,
        seed,
        ..Default::default()
    };
    // Only keep the draws after tuning.
    let config = ArrowConfig::new().store_warmup(false);
    let mut sampler = Sampler::new(Arc::new(NormalModel), settings, config, 1, None)?;
    loop {
        match sampler.wait_timeout(Duration::from_millis(10)) {
            SamplerWaitResult::Trace(chains) => return Ok(chains),
            SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
            SamplerWaitResult::Err(err, _) => return Err(err),
        }
    }
}

/// The mean of all posterior draws of all chains.
pub fn mean_of_draws(chains: &[ArrowTrace]) -> Result<f64> {
    let mut sum = 0.0;
    let mut count = 0;
    for chain in chains {
        let column = chain
            .posterior
            .column_by_name("value")
            .context("No `value` column in the posterior")?;
        let Some(draws) = column.as_any().downcast_ref::<LargeListArray>() else {
            bail!("`value` should hold one list per draw");
        };
        let Some(values) = draws.values().as_any().downcast_ref::<Float64Array>() else {
            bail!("`value` should hold f64 values");
        };
        sum += values.values().iter().sum::<f64>();
        count += values.len();
    }
    Ok(sum / count as f64)
}

/// Sample and return the posterior mean, which should be close to `MEAN`.
#[wasm_bindgen]
pub fn posterior_mean(seed: u32) -> Result<f64, JsError> {
    sample(seed.into())
        .and_then(|chains| mean_of_draws(&chains))
        .map_err(|err| JsError::new(&format!("{err:#}")))
}
