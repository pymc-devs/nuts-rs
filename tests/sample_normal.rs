use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Context;
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, DiagAdaptExpSettings, DiagNutsSettings, EuclideanAdaptOptions,
    LogpError, LowRankNutsSettings, Model, Sampler, SamplerWaitResult, Settings, ZarrConfig,
};
use nuts_storable::HasDims;
use rand::SeedableRng;
use rand::prelude::Rng;
use rand_distr::{Distribution, StandardNormal};
use thiserror::Error;
use zarrs::{
    array::{Array, ArraySubset},
    storage::{ReadableListableStorageTraits, store::MemoryStore},
};

/// A correlated multivariate normal with covariance Σ = I + rank1_scale * ones * ones^T.
///
/// By the Woodbury identity the precision matrix is
///   Σ⁻¹ = I - c * ones * ones^T,  where c = rank1_scale / (1 + rank1_scale * dim).
///
/// This gives a non-trivial rank-1 correlation structure that the low-rank mass-matrix
/// adaptation should be able to recover exactly (up to numerical precision) once the
/// estimation window contains more draws than the dimensionality.
struct CorrelatedNormalLogp {
    dim: usize,
    mu: Vec<f64>,
    /// Coefficient `c` in the precision matrix I - c * ones * ones^T.
    prec_rank1_coeff: f64,
}

impl CorrelatedNormalLogp {
    fn new(dim: usize, rank1_scale: f64) -> Self {
        let prec_rank1_coeff = rank1_scale / (1.0 + rank1_scale * dim as f64);
        Self {
            dim,
            mu: vec![0.0; dim],
            prec_rank1_coeff,
        }
    }
}

#[derive(Error, Debug)]
enum CorrelatedNormalLogpError {}

impl LogpError for CorrelatedNormalLogpError {
    fn is_recoverable(&self) -> bool {
        true
    }
}

impl HasDims for CorrelatedNormalLogp {
    fn dim_sizes(&self) -> std::collections::HashMap<String, u64> {
        std::collections::HashMap::from([
            ("unconstrained_parameter".to_string(), self.dim as u64),
            ("dim".to_string(), self.dim as u64),
        ])
    }
}

impl CpuLogpFunc for CorrelatedNormalLogp {
    type LogpError = CorrelatedNormalLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let n = position.len();
        assert_eq!(grad.len(), n);

        // diff = position - mu; sum_diff = ones^T diff
        let sum_diff: f64 = position
            .iter()
            .zip(self.mu.iter())
            .map(|(p, m)| p - m)
            .sum();
        let rank1_term = self.prec_rank1_coeff * sum_diff;

        // gradient[i] = -(Λ diff)[i] = -(diff[i] - c * sum_diff)
        // logp        = -0.5 * diff^T Λ diff
        let mut logp = 0f64;
        for i in 0..n {
            let diff = position[i] - self.mu[i];
            let prec_times_diff = diff - rank1_term;
            grad[i] = -prec_times_diff;
            logp -= 0.5 * diff * prec_times_diff;
        }
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
    let settings = DiagNutsSettings {
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
    let settings = DiagNutsSettings {
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
        .retrieve_array_subset(&ArraySubset::new_with_shape(diverging.shape().to_vec()))?;

    let logp = Array::open(store_dyn.clone(), "/sample_stats/logp")
        .context("Could not read logp array")?;
    assert!(
        logp.dimension_names().as_ref().unwrap()
            == &[Some("chain".to_string()), Some("draw".to_string())]
    );
    let _: Vec<f64> =
        logp.retrieve_array_subset(&ArraySubset::new_with_shape(logp.shape().to_vec()))?;

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

/// Check that the Fisher divergence is (numerically) zero for all post-warmup draws when
/// sampling a correlated Gaussian with the low-rank mass-matrix adaptation.
///
/// For any Gaussian p(x) = N(μ, Σ), once the transformation F perfectly maps the posterior
/// to a standard normal, the gradient in transformed space satisfies g_y = -y pointwise.
/// Therefore fisher_distance = ‖y + g_y‖² = 0 for every single draw, not just on average.
/// The low-rank estimator can achieve this exactly when the estimation window contains
/// more draws than the dimensionality of the posterior.
fn check_low_rank_fisher_divergence() -> anyhow::Result<()> {
    let dim = 10;
    // Covariance: Σ = I + 0.5 * ones * ones^T  (rank-1 off-diagonal structure)
    let logp = CorrelatedNormalLogp::new(dim, 0.5);
    let math = CpuMath::new(logp);

    // num_tune is much larger than dim so the estimation window is overdetermined.
    let settings = LowRankNutsSettings {
        num_tune: 500,
        num_draws: 100,
        num_chains: 1,
        seed: 42,
        ..Default::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut chain = settings.new_chain(0, math, &mut rng);
    chain.set_position(&vec![1.0f64; dim])?;

    for _ in 0..(settings.num_tune + settings.num_draws) {
        let (_, _, stats, progress) = chain.expanded_draw()?;
        if !progress.tuning {
            assert!(
                stats.point.fisher_distance < 1e-10,
                "fisher_distance = {} should be ~0 after low-rank adaptation converges",
                stats.point.fisher_distance,
            );
        }
    }
    Ok(())
}

#[test]
fn low_rank_exact_gaussian() -> anyhow::Result<()> {
    check_low_rank_fisher_divergence()
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
