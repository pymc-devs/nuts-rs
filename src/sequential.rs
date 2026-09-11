//! Caller-driven sampling through the model interface, without worker threads.

use crate::sampler_stats::StatsDims;
use crate::storage::{ChainStorage, StorageConfig, TraceStorage};
use crate::{Chain, Math, Model, Progress, Settings, Storable, Value};
use anyhow::{Context, Result, ensure};
use rand::Rng;

/// Options for a single caller-driven chain.
#[derive(Clone, Copy, Debug)]
pub struct SequentialOptions {
    /// Identifier included in progress and trace metadata.
    pub chain_id: u64,
    /// Number of model-generated starting points to try. Must be positive.
    pub max_init_attempts: usize,
    /// Expand and record warmup draws. Disable only when warmup output is unwanted.
    /// Skipping stochastic expansion can change subsequent sampling randomness.
    pub expand_warmup: bool,
}

impl Default for SequentialOptions {
    fn default() -> Self {
        Self {
            chain_id: 0,
            max_init_attempts: 500,
            expand_warmup: true,
        }
    }
}

/// One completed draw. Warmup values are empty when expansion is disabled.
pub struct SequentialDraw {
    pub point: Box<[f64]>,
    pub values: Vec<(String, Option<Value>)>,
    pub progress: Progress,
}

/// A single chain that runs entirely on the calling thread.
///
/// Initialization, expansion and trace recording stay inside nuts-rs. Call
/// `step` at the desired pace; stop calling it to pause, or call `finalize` to
/// return a partial trace. No threads, channels, or background work are created.
/// The model's existing thread-safety bounds are unchanged.
///
/// ```
/// use nuts_rs::{Model, DiagNutsSettings, HashMapConfig, SequentialOptions, SequentialSampler};
/// use rand::{SeedableRng, rngs::ChaCha8Rng};
///
/// fn sample(model: &impl Model) -> anyhow::Result<()> {
///     let mut rng = ChaCha8Rng::seed_from_u64(42);
///     let mut init_rng = ChaCha8Rng::seed_from_u64(43);
///     let mut sampler = SequentialSampler::new(
///         model, DiagNutsSettings::default(), Some(HashMapConfig::default()),
///         SequentialOptions::default(), &mut rng, &mut init_rng,
///     )?;
///     while let Some(draw) = sampler.step()? {
///         println!("Draw {}", draw.progress.draw);
///         // Returning control between calls allows progress and cancellation.
///     }
///     let trace = sampler.finalize()?.unwrap();
///     Ok(())
/// }
/// ```
pub struct SequentialSampler<'model, M: Model, S: Settings, C: StorageConfig> {
    chain: S::Chain<M::Math<'model>>,
    settings: S,
    storage: Option<C::Storage>,
    trace: Option<<C::Storage as TraceStorage>::ChainStorage>,
    initial_position: Vec<f64>,
    completed: usize,
    total: usize,
    expand_warmup: bool,
    failed: bool,
}

impl<'model, M: Model, S: Settings, C: StorageConfig> SequentialSampler<'model, M, S, C> {
    /// Initialize a chain using the model interface and separate sampling and
    /// initialization RNGs. `None` storage enables streaming without retaining a trace.
    /// The chain ID is supplied independently of `settings.num_chains()`; create
    /// one runner for each desired chain.
    pub fn new<R: Rng + ?Sized, I: Rng + ?Sized>(
        model: &'model M,
        settings: S,
        config: Option<C>,
        options: SequentialOptions,
        rng: &mut R,
        init_rng: &mut I,
    ) -> Result<Self> {
        ensure!(
            options.max_init_attempts > 0,
            "max_init_attempts must be positive"
        );
        let total = settings
            .hint_num_tune()
            .checked_add(settings.hint_num_draws())
            .context("Draw count overflow")?;
        ensure!(
            total > 0,
            "At least one tuning or sampling draw is required"
        );
        let math = model.math(rng).context("Failed to create model density")?;
        let storage = config
            .map(|config| config.new_trace(&settings, &math))
            .transpose()?;
        let trace = storage
            .as_ref()
            .map(|storage| storage.initialize_trace_for_chain(options.chain_id))
            .transpose()?;
        let mut position = vec![0.; math.dim()];
        let mut chain = settings.new_chain(options.chain_id, math, rng);
        for attempt in 0..options.max_init_attempts {
            model
                .init_position(init_rng, &mut position)
                .context("Failed to generate a new initial position")?;
            match chain.set_position(&position) {
                Ok(()) => break,
                Err(error) if attempt + 1 == options.max_init_attempts => {
                    return Err(error.context("All initialization points failed"));
                }
                Err(_) => {}
            }
        }
        Ok(Self {
            chain,
            settings,
            storage,
            trace,
            initial_position: position,
            completed: 0,
            total,
            expand_warmup: options.expand_warmup,
            failed: false,
        })
    }

    /// The accepted starting position, before any draws.
    pub fn initial_position(&self) -> &[f64] {
        &self.initial_position
    }

    /// Advance one draw, or return `None` once the configured count is reached.
    /// After an error the runner cannot advance again, but can still be finalized.
    pub fn step(&mut self) -> Result<Option<SequentialDraw>> {
        ensure!(!self.failed, "Cannot advance a failed sequential sampler");
        if self.completed == self.total {
            return Ok(None);
        }
        let result = self.step_inner();
        match result {
            Ok(draw) => {
                self.completed += 1;
                Ok(Some(draw))
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    fn step_inner(&mut self) -> Result<SequentialDraw> {
        if !self.expand_warmup && self.completed < self.settings.hint_num_tune() {
            let (point, progress) = self.chain.draw()?;
            return Ok(SequentialDraw {
                point,
                values: vec![],
                progress,
            });
        }
        let (point, mut expanded, mut stats, progress) = self.chain.expanded_draw()?;
        let math = self.chain.math();
        let values = expanded.get_all(&*math);
        // Values are returned for streaming, while storage consumes its own copy.
        if let Some(trace) = self.trace.as_mut() {
            trace.record_sample(
                &self.settings,
                stats.get_all(&StatsDims::from(&*math)),
                values.clone(),
                &progress,
            )?;
        }
        Ok(SequentialDraw {
            point,
            values: values
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value))
                .collect(),
            progress,
        })
    }

    /// Finalize all completed draws, including when sampling stopped early.
    /// Returns `None` when constructed without storage. Backend errors propagate.
    pub fn finalize(self) -> Result<Option<<C::Storage as TraceStorage>::Finalized>> {
        match (self.storage, self.trace) {
            (Some(storage), Some(trace)) => {
                let (error, result) = storage.finalize(vec![trace.finalize()])?;
                if let Some(error) = error {
                    return Err(error);
                }
                Ok(Some(result))
            }
            _ => Ok(None),
        }
    }
}
