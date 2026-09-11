use anyhow::Result;
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, HashMapConfig, LogpError,
    Model, SequentialOptions, SequentialSampler, Settings,
};
use rand::{Rng, SeedableRng, rngs::ChaCha8Rng};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread::ThreadId,
};

struct Normal {
    thread: ThreadId,
    fail: AtomicBool,
    initializations: AtomicUsize,
    bad_starts: usize,
}
impl Normal {
    fn new() -> Self {
        Self {
            thread: std::thread::current().id(),
            fail: AtomicBool::new(false),
            initializations: AtomicUsize::new(0),
            bad_starts: 0,
        }
    }
}
struct Density<'a>(&'a Normal);
#[derive(Debug, thiserror::Error)]
#[error("test density failure")]
struct Failure;
impl LogpError for Failure {
    fn is_recoverable(&self) -> bool {
        false
    }
}
impl HasDims for Density<'_> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("unconstrained_parameter".into(), 1), ("dim".into(), 1)])
    }
}
impl CpuLogpFunc for Density<'_> {
    type LogpError = Failure;
    type ExpandedVector = Vec<f64>;
    type FlowParameters = ();
    fn dim(&self) -> usize {
        1
    }
    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Failure> {
        assert_eq!(self.0.thread, std::thread::current().id());
        if self.0.fail.load(Ordering::Relaxed) || !position[0].is_finite() {
            return Err(Failure);
        }
        gradient[0] = -position[0];
        Ok(-0.5 * position[0].powi(2))
    }
    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        _: &mut R,
        position: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        Ok(position.to_vec())
    }
}
impl Model for Normal {
    type Math<'a> = CpuMath<Density<'a>>;
    fn math<R: Rng + ?Sized>(&self, _: &mut R) -> Result<Self::Math<'_>> {
        assert_eq!(self.thread, std::thread::current().id());
        Ok(CpuMath::new(Density(self)))
    }
    fn init_position<R: Rng + ?Sized>(&self, _: &mut R, position: &mut [f64]) -> Result<()> {
        assert_eq!(self.thread, std::thread::current().id());
        let attempt = self.initializations.fetch_add(1, Ordering::Relaxed);
        position[0] = if attempt < self.bad_starts {
            f64::INFINITY
        } else {
            0.5
        };
        Ok(())
    }
}
fn settings() -> DiagNutsSettings {
    DiagNutsSettings {
        num_tune: 20,
        num_draws: 5,
        maxdepth: 3,
        ..Default::default()
    }
}
fn rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(42)
}

#[test]
fn matches_direct_chain_and_stops_at_draw_limit() {
    let model = Normal::new();
    let settings = settings();
    let mut sampler = SequentialSampler::new(
        &model,
        settings,
        Some(HashMapConfig::default()),
        SequentialOptions::default(),
        &mut rng(),
        &mut rng(),
    )
    .unwrap();
    let mut direct = settings.new_chain(0, CpuMath::new(Density(&model)), &mut rng());
    direct.set_position(sampler.initial_position()).unwrap();
    for _ in 0..25 {
        let result = sampler.step().unwrap().unwrap();
        let (point, _, _, info) = direct.expanded_draw().unwrap();
        assert_eq!(result.point, point);
        assert_eq!(result.progress.tuning, info.tuning);
        assert!(!result.values.is_empty());
    }
    assert!(sampler.step().unwrap().is_none());
    assert!(sampler.step().unwrap().is_none());
    assert!(sampler.finalize().unwrap().is_some());
}

#[test]
fn streaming_and_skipped_warmup() {
    let model = Normal::new();
    let mut sampler = SequentialSampler::new(
        &model,
        settings(),
        None::<HashMapConfig>,
        SequentialOptions {
            expand_warmup: false,
            ..Default::default()
        },
        &mut rng(),
        &mut rng(),
    )
    .unwrap();
    for i in 0..25 {
        let draw = sampler.step().unwrap().unwrap();
        assert_eq!(draw.values.is_empty(), i < 20);
    }
    assert!(sampler.finalize().unwrap().is_none());
}

#[test]
fn initialization_retries_are_bounded() {
    let model = Normal {
        bad_starts: 2,
        ..Normal::new()
    };
    let sampler = SequentialSampler::new(
        &model,
        settings(),
        Some(HashMapConfig::default()),
        SequentialOptions {
            max_init_attempts: 3,
            ..Default::default()
        },
        &mut rng(),
        &mut rng(),
    )
    .unwrap();
    assert_eq!(model.initializations.load(Ordering::Relaxed), 3);
    assert_eq!(sampler.initial_position(), &[0.5]);
    let model = Normal {
        bad_starts: 3,
        ..Normal::new()
    };
    assert!(
        SequentialSampler::new(
            &model,
            settings(),
            Some(HashMapConfig::default()),
            SequentialOptions {
                max_init_attempts: 2,
                ..Default::default()
            },
            &mut rng(),
            &mut rng()
        )
        .is_err()
    );
    assert_eq!(model.initializations.load(Ordering::Relaxed), 2);
}

#[test]
fn sampling_error_prevents_further_steps() {
    let model = Normal::new();
    let mut sampler = SequentialSampler::new(
        &model,
        settings(),
        Some(HashMapConfig::default()),
        SequentialOptions::default(),
        &mut rng(),
        &mut rng(),
    )
    .unwrap();
    sampler.step().unwrap();
    model.fail.store(true, Ordering::Relaxed);
    assert!(sampler.step().is_err());
    model.fail.store(false, Ordering::Relaxed);
    assert!(sampler.step().is_err());
    sampler.finalize().unwrap();
}

#[cfg(feature = "arrow")]
#[test]
fn arrow_partial_and_empty_traces() {
    use nuts_rs::ArrowConfig;
    let model = Normal::new();
    let mut config = ArrowConfig::default();
    config.store_warmup = false;
    let mut sampler = SequentialSampler::new(
        &model,
        settings(),
        Some(config),
        SequentialOptions {
            chain_id: 7,
            ..Default::default()
        },
        &mut rng(),
        &mut rng(),
    )
    .unwrap();
    for _ in 0..23 {
        assert_eq!(sampler.step().unwrap().unwrap().progress.chain, 7);
    }
    let traces = sampler.finalize().unwrap().unwrap();
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].posterior.num_rows(), 3);
    assert_eq!(traces[0].sample_stats.num_rows(), 3);
    // Finalizing before the first step is a valid cancellation point.
    let sampler = SequentialSampler::new(
        &model,
        settings(),
        Some(ArrowConfig::default()),
        SequentialOptions::default(),
        &mut rng(),
        &mut rng(),
    )
    .unwrap();
    assert_eq!(
        sampler.finalize().unwrap().unwrap()[0].posterior.num_rows(),
        0
    );
}

#[test]
fn invalid_runner_options_return_errors() {
    let model = Normal::new();
    let empty = DiagNutsSettings {
        num_tune: 0,
        num_draws: 0,
        ..settings()
    };
    assert!(
        SequentialSampler::new(
            &model,
            empty,
            None::<HashMapConfig>,
            SequentialOptions::default(),
            &mut rng(),
            &mut rng()
        )
        .is_err()
    );
    assert!(
        SequentialSampler::new(
            &model,
            settings(),
            None::<HashMapConfig>,
            SequentialOptions {
                max_init_attempts: 0,
                ..Default::default()
            },
            &mut rng(),
            &mut rng()
        )
        .is_err()
    );
}
