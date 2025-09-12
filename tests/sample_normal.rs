use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Context;
use nuts_rs::{
    CpuLogpFunc, CpuMath, DiagAdaptExpSettings, DiagGradNutsSettings, EuclideanAdaptOptions,
    LogpError, LowRankNutsSettings, Model, Sampler, SamplerWaitResult, ZarrConfig,
};
use nuts_storable::HasDims;
use rand::prelude::Rng;
use rand_distr::{Distribution, StandardNormal};
use thiserror::Error;
use zarrs::{
    array::Array,
    array_subset::ArraySubset,
    storage::{ReadableListableStorageTraits, store::MemoryStore},
};

struct NormalLogp<'a> {
    dim: usize,
    mu: &'a [f64],
}

#[derive(Error, Debug)]
enum NormalLogpError {}

impl LogpError for NormalLogpError {
    fn is_recoverable(&self) -> bool {
        true
    }
}

impl HasDims for NormalLogp<'_> {
    fn dim_sizes(&self) -> std::collections::HashMap<String, u64> {
        std::collections::HashMap::from([
            ("unconstrained_parameter".to_string(), self.dim as u64),
            ("dim".to_string(), self.dim as u64),
        ])
    }
}

impl<'a> CpuLogpFunc for NormalLogp<'a> {
    type LogpError = NormalLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let n = position.len();
        assert!(grad.len() == n);
        let mut logp = 0f64;

        position
            .iter()
            .zip(self.mu.iter())
            .zip(grad.iter_mut())
            .for_each(|((&p, &mu), grad)| {
                let diff = p - mu;
                logp -= diff * diff / 2.;
                *grad = -diff;
            });
        Ok(logp)
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

struct NormalModel {
    mu: Box<[f64]>,
}

impl NormalModel {
    fn new(mu: Box<[f64]>) -> Self {
        NormalModel { mu }
    }
}

impl Model for NormalModel {
    type Math<'model>
        = CpuMath<NormalLogp<'model>>
    where
        Self: 'model;

    fn math<R: Rng + ?Sized>(&self, _rng: &mut R) -> anyhow::Result<Self::Math<'_>> {
        Ok(CpuMath::new(NormalLogp {
            dim: self.mu.len(),
            mu: &self.mu,
        }))
    }

    fn init_position<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> anyhow::Result<()> {
        let normal = StandardNormal;
        position.iter_mut().for_each(|x| *x = normal.sample(rng));
        Ok(())
    }
}

fn sample() -> anyhow::Result<Arc<MemoryStore>> {
    let mu = vec![0.5; 100];
    let model = NormalModel::new(mu.into());
    let settings = DiagGradNutsSettings {
        seed: 42,
        num_chains: 6,
        ..Default::default()
    };

    let store = Arc::new(MemoryStore::new());
    let trace_config = ZarrConfig::new(store.clone());
    let mut sampler = Sampler::new(model, settings, trace_config, 6, None)?;

    let _ = loop {
        match sampler.wait_timeout(Duration::from_secs(1)) {
            SamplerWaitResult::Trace(trace) => break trace,
            SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
            SamplerWaitResult::Err(err, _trace) => return Err(err),
        };
    };
    Ok(store)
}

fn sample_debug_stats() -> anyhow::Result<Arc<dyn ReadableListableStorageTraits>> {
    let mu = vec![0.5; 100];
    let model = NormalModel::new(mu.into());
    let settings = DiagGradNutsSettings {
        seed: 42,
        num_chains: 6,
        store_gradient: true,
        store_divergences: true,
        store_unconstrained: true,
        adapt_options: EuclideanAdaptOptions {
            mass_matrix_options: DiagAdaptExpSettings {
                store_mass_matrix: true,
                use_grad_based_estimate: true,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let store = Arc::new(MemoryStore::new());
    let trace_config = ZarrConfig::new(store.clone());
    let mut sampler = Sampler::new(model, settings, trace_config, 6, None)?;

    let _ = loop {
        match sampler.wait_timeout(Duration::from_secs(1)) {
            SamplerWaitResult::Trace(trace) => break trace,
            SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
            SamplerWaitResult::Err(err, _trace) => return Err(err),
        };
    };

    let store_dyn: Arc<dyn ReadableListableStorageTraits> = store.clone();

    let diverging = Array::open(store_dyn.clone(), "/sample_stats/diverging")
        .context("Could not read diverging array")?;
    assert!(
        diverging.dimension_names().as_ref().unwrap()
            == &[Some("chain".to_string()), Some("draw".to_string())]
    );

    let _: Vec<bool> = diverging
        .retrieve_array_subset_elements(&ArraySubset::new_with_shape(diverging.shape().to_vec()))?;

    let logp = Array::open(store_dyn.clone(), "/sample_stats/logp")
        .context("Could not read logp array")?;
    assert!(
        logp.dimension_names().as_ref().unwrap()
            == &[Some("chain".to_string()), Some("draw".to_string())]
    );
    let _: Vec<f64> =
        logp.retrieve_array_subset_elements(&ArraySubset::new_with_shape(logp.shape().to_vec()))?;

    Ok(store)
}

fn sample_eigs_debug_stats() -> anyhow::Result<Arc<MemoryStore>> {
    let mu = vec![0.5; 10];
    let model = NormalModel::new(mu.into());
    let settings = LowRankNutsSettings {
        seed: 42,
        num_chains: 1,
        num_tune: 200,
        num_draws: 10,
        store_gradient: true,
        store_divergences: true,
        store_unconstrained: true,
        adapt_options: EuclideanAdaptOptions {
            mass_matrix_options: nuts_rs::LowRankSettings {
                store_mass_matrix: false,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let store = Arc::new(MemoryStore::new());
    let trace_config = ZarrConfig::new(store.clone());
    let mut sampler = Sampler::new(model, settings, trace_config, 1, None)?;

    let _trace = loop {
        match sampler.wait_timeout(Duration::from_secs(1)) {
            SamplerWaitResult::Trace(trace) => break trace,
            SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
            SamplerWaitResult::Err(err, _trace) => return Err(err),
        };
    };

    Ok(store)
}

#[test]
fn run() -> anyhow::Result<()> {
    let start = Instant::now();
    let _ = sample()?;
    dbg!(start.elapsed());
    Ok(())
}

#[test]
fn run_debug_stats() -> anyhow::Result<()> {
    let start = Instant::now();
    let _ = sample_debug_stats()?;
    dbg!(start.elapsed());
    Ok(())
}

#[test]
fn run_debug_stats_eigs() -> anyhow::Result<()> {
    let start = Instant::now();
    let _ = sample_eigs_debug_stats()?;
    dbg!(start.elapsed());
    Ok(())
}
