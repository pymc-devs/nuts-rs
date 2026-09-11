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
    thread: Option<ThreadId>,
    fail: AtomicBool,
    initializations: AtomicUsize,
    bad_starts: usize,
    fail_expansion: bool,
}
impl Normal {
    fn new() -> Self {
        Self {
            thread: Some(std::thread::current().id()),
            fail: AtomicBool::new(false),
            initializations: AtomicUsize::new(0),
            bad_starts: 0,
            fail_expansion: false,
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
        if let Some(thread) = self.0.thread {
            assert_eq!(thread, std::thread::current().id());
        }
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
        if self.0.fail_expansion {
            return Err(CpuMathError::ExpandError("test expansion failure".into()));
        }
        Ok(position.to_vec())
    }
}
impl Model for Normal {
    type Math<'a> = CpuMath<Density<'a>>;
    fn math<R: Rng + ?Sized>(&self, _: &mut R) -> Result<Self::Math<'_>> {
        if let Some(thread) = self.thread {
            assert_eq!(thread, std::thread::current().id());
        }
        Ok(CpuMath::new(Density(self)))
    }
    fn init_position<R: Rng + ?Sized>(&self, _: &mut R, position: &mut [f64]) -> Result<()> {
        if let Some(thread) = self.thread {
            assert_eq!(thread, std::thread::current().id());
        }
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
    let mut sampler = SequentialSampler::with_rngs(
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
    let mut sampler = SequentialSampler::with_rngs(
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
fn skipped_warmup_matches_nuts_and_mclmc_progress() {
    use nuts_rs::{
        DiagMclmcSettings, LowRankMclmcSettings, LowRankNutsSettings, MclmcTrajectoryKind,
    };

    fn check(settings: impl Settings) {
        let model = Normal::new();
        let mut sampler = SequentialSampler::with_options(
            &model,
            settings,
            None::<HashMapConfig>,
            SequentialOptions {
                expand_warmup: false,
                ..Default::default()
            },
        )
        .unwrap();
        for i in 0..25 {
            let draw = sampler.step().unwrap().unwrap();
            assert_eq!(draw.progress.tuning, i < 20, "draw {i}");
            assert_eq!(draw.values.is_empty(), draw.progress.tuning, "draw {i}");
        }
        assert!(sampler.step().unwrap().is_none());
    }

    check(settings());
    check(LowRankNutsSettings {
        num_tune: 20,
        num_draws: 5,
        maxdepth: 3,
        ..Default::default()
    });
    check(DiagMclmcSettings {
        num_tune: 20,
        num_draws: 5,
        trajectory_kind: MclmcTrajectoryKind::Euclidean,
        ..Default::default()
    });
    check(LowRankMclmcSettings {
        num_tune: 20,
        num_draws: 5,
        trajectory_kind: MclmcTrajectoryKind::Euclidean,
        ..Default::default()
    });
}

#[cfg(feature = "ndarray")]
#[test]
fn ndarray_storage_allocates_only_one_chain() {
    use nuts_rs::{NdarrayConfig, NdarrayValue};
    let model = Normal::new();
    let sampler = SequentialSampler::with_options(
        &model,
        settings(),
        Some(NdarrayConfig::default()),
        SequentialOptions {
            chain_id: 7,
            ..Default::default()
        },
    )
    .unwrap();
    let trace = sampler.finalize().unwrap().unwrap();
    let NdarrayValue::U64(depth) = &trace.stats["depth"] else {
        panic!("unexpected depth type")
    };
    assert_eq!(depth.shape()[0], 1);
}

#[cfg(feature = "zarr")]
#[test]
fn zarr_storage_uses_one_slot_for_a_logical_chain_id() {
    use nuts_rs::ZarrConfig;
    use std::sync::Arc;
    use zarrs::{
        array::{Array, ArraySubset},
        storage::store::MemoryStore,
    };

    for num_chains in [0, 6] {
        let model = Normal::new();
        let store = Arc::new(MemoryStore::new());
        let mut sampler = SequentialSampler::with_options(
            &model,
            DiagNutsSettings {
                num_chains,
                ..settings()
            },
            Some(ZarrConfig::new(store.clone()).store_warmup(false)),
            SequentialOptions {
                chain_id: 7,
                ..Default::default()
            },
        )
        .unwrap();
        let mut posterior = vec![];
        while let Some(draw) = sampler.step().unwrap() {
            assert_eq!(draw.progress.chain, 7);
            if !draw.progress.tuning {
                posterior.extend_from_slice(&draw.point);
            }
        }
        sampler.finalize().unwrap();
        let array = Array::open(store, "/posterior/value").unwrap();
        assert_eq!(array.shape(), &[1, 5, 1]);
        let stored: Vec<f64> = array
            .retrieve_array_subset(&ArraySubset::new_with_shape(array.shape().to_vec()))
            .unwrap();
        assert_eq!(stored, posterior);
    }
}

#[test]
fn csv_preserves_logical_chain_filenames() {
    use nuts_rs::CsvConfig;
    let model = Normal::new();
    let directory = tempfile::tempdir().unwrap();
    for chain_id in [7, 8] {
        let mut sampler = SequentialSampler::with_options(
            &model,
            settings(),
            Some(CsvConfig::new(directory.path())),
            SequentialOptions {
                chain_id,
                ..Default::default()
            },
        )
        .unwrap();
        sampler.step().unwrap();
        sampler.finalize().unwrap();
        assert!(
            directory
                .path()
                .join(format!("chain_{chain_id}.csv"))
                .exists()
        );
    }
    assert!(!directory.path().join("chain_0.csv").exists());
}

#[test]
fn initialization_retries_are_bounded() {
    let model = Normal {
        bad_starts: 2,
        ..Normal::new()
    };
    let sampler = SequentialSampler::with_rngs(
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
        SequentialSampler::with_rngs(
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
    let mut sampler = SequentialSampler::with_rngs(
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
    let mut sampler = SequentialSampler::with_rngs(
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
    let sampler = SequentialSampler::with_rngs(
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
        SequentialSampler::with_rngs(
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
        SequentialSampler::with_rngs(
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

#[test]
fn default_constructor_is_reproducible() {
    let model = Normal::new();
    let mut first = SequentialSampler::new(&model, settings(), HashMapConfig::default()).unwrap();
    let mut second = SequentialSampler::new(&model, settings(), HashMapConfig::default()).unwrap();
    for _ in 0..25 {
        assert_eq!(
            first.step().unwrap().unwrap().point,
            second.step().unwrap().unwrap().point
        );
    }
}

#[cfg(all(feature = "parallel", feature = "arrow"))]
#[test]
fn parallel_and_sequential_record_identical_traces() {
    use nuts_rs::{ArrowConfig, Sampler, SamplerWaitResult};
    let settings = DiagNutsSettings {
        num_chains: 1,
        seed: 42,
        ..settings()
    };
    let parallel = Sampler::new(
        Normal {
            thread: None,
            ..Normal::new()
        },
        settings,
        ArrowConfig::default(),
        1,
        None,
    )
    .unwrap();
    let parallel = match parallel.wait_timeout(std::time::Duration::from_secs(10)) {
        SamplerWaitResult::Trace(trace) => trace,
        SamplerWaitResult::Err(error, _) => panic!("{error:#}"),
        SamplerWaitResult::Timeout(sampler) => {
            sampler.abort().unwrap();
            panic!("sampling timed out")
        }
    };
    let model = Normal::new();
    let mut chain_rng = ChaCha8Rng::seed_from_u64(settings.seed);
    chain_rng.set_stream(1); // Parallel chain zero's stream.
    let mut sequential = SequentialSampler::with_rngs(
        &model,
        settings,
        Some(ArrowConfig::default()),
        SequentialOptions::default(),
        &mut chain_rng,
        &mut rng(),
    )
    .unwrap();
    while sequential.step().unwrap().is_some() {}
    let sequential = sequential.finalize().unwrap().unwrap();
    assert_eq!(sequential[0].posterior, parallel[0].posterior);
    assert_eq!(sequential[0].sample_stats, parallel[0].sample_stats);
}

#[cfg(all(feature = "parallel", feature = "arrow"))]
#[test]
fn parallel_expansion_errors_are_returned() {
    use nuts_rs::{ArrowConfig, Sampler, SamplerWaitResult};
    let model = Normal {
        thread: None,
        fail_expansion: true,
        ..Normal::new()
    };
    let settings = DiagNutsSettings {
        num_chains: 1,
        ..settings()
    };
    let sampler = Sampler::new(model, settings, ArrowConfig::default(), 1, None).unwrap();
    match sampler.wait_timeout(std::time::Duration::from_secs(10)) {
        SamplerWaitResult::Err(error, _) => {
            assert!(format!("{error:#}").contains("test expansion failure"))
        }
        SamplerWaitResult::Trace(_) => panic!("expected expansion failure"),
        SamplerWaitResult::Timeout(sampler) => {
            sampler.abort().unwrap();
            panic!("sampling timed out")
        }
    }
}
