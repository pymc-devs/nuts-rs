use anyhow::{Context, Result, bail};
use itertools::Itertools;
use nuts_storable::{HasDims, Storable, Value};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_chacha::ChaCha8Rng;
use rayon::{ScopeFifo, ThreadPoolBuilder};
use serde::Serialize;
use std::{
    collections::HashMap,
    fmt::Debug,
    ops::Deref,
    sync::{
        Arc, Mutex,
        mpsc::{
            Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError, channel, sync_channel,
        },
    },
    thread::{JoinHandle, spawn},
    time::{Duration, Instant},
};

use crate::{
    DiagAdaptExpSettings,
    adapt_strategy::{EuclideanAdaptOptions, GlobalStrategy, GlobalStrategyStatsOptions},
    chain::{AdaptStrategy, Chain, NutsChain, StatOptions},
    euclidean_hamiltonian::EuclideanHamiltonian,
    low_rank_mass_matrix::{LowRankMassMatrix, LowRankMassMatrixStrategy, LowRankSettings},
    mass_matrix::DiagMassMatrix,
    mass_matrix_adapt::Strategy as DiagMassMatrixStrategy,
    math_base::Math,
    model::Model,
    nuts::NutsOptions,
    sampler_stats::{SamplerStats, StatsDims},
    storage::{ChainStorage, StorageConfig, TraceStorage},
    transform_adapt_strategy::{TransformAdaptation, TransformedSettings},
    transformed_hamiltonian::{TransformedHamiltonian, TransformedPointStatsOptions},
};

/// All sampler configurations implement this trait
pub trait Settings:
    private::Sealed + Clone + Copy + Default + Sync + Send + Serialize + 'static
{
    type Chain<M: Math>: Chain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        math: M,
        rng: &mut R,
    ) -> Self::Chain<M>;

    fn hint_num_tune(&self) -> usize;
    fn hint_num_draws(&self) -> usize;
    fn num_chains(&self) -> usize;
    fn seed(&self) -> u64;
    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions;

    fn stat_names<M: Math>(&self, math: &M) -> Vec<String> {
        let dims = StatsDims::from(math);
        <<Self::Chain<M> as SamplerStats<M>>::Stats as Storable<_>>::names(&dims)
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn data_names<M: Math>(&self, math: &M) -> Vec<String> {
        <M::ExpandedVector as Storable<_>>::names(math)
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn stat_types<M: Math>(&self, math: &M) -> Vec<(String, nuts_storable::ItemType)> {
        self.stat_names(math)
            .into_iter()
            .map(|name| (name.clone(), self.stat_type::<M>(math, &name)))
            .collect()
    }

    fn stat_type<M: Math>(&self, math: &M, name: &str) -> nuts_storable::ItemType {
        let dims = StatsDims::from(math);
        <<Self::Chain<M> as SamplerStats<M>>::Stats as Storable<_>>::item_type(&dims, name)
    }

    fn data_types<M: Math>(&self, math: &M) -> Vec<(String, nuts_storable::ItemType)> {
        self.data_names(math)
            .into_iter()
            .map(|name| (name.clone(), self.data_type(math, &name)))
            .collect()
    }
    fn data_type<M: Math>(&self, math: &M, name: &str) -> nuts_storable::ItemType {
        <M::ExpandedVector as Storable<_>>::item_type(math, name)
    }

    fn stat_dims_all<M: Math>(&self, math: &M) -> Vec<(String, Vec<String>)> {
        self.stat_names(math)
            .into_iter()
            .map(|name| (name.clone(), self.stat_dims::<M>(math, &name)))
            .collect()
    }

    fn stat_dims<M: Math>(&self, math: &M, name: &str) -> Vec<String> {
        let dims = StatsDims::from(math);
        <<Self::Chain<M> as SamplerStats<M>>::Stats as Storable<_>>::dims(&dims, name)
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn stat_dim_sizes<M: Math>(&self, math: &M) -> HashMap<String, u64> {
        let dims = StatsDims::from(math);
        dims.dim_sizes()
    }

    fn data_dims_all<M: Math>(&self, math: &M) -> Vec<(String, Vec<String>)> {
        self.data_names(math)
            .into_iter()
            .map(|name| (name.clone(), self.data_dims(math, &name)))
            .collect()
    }

    fn data_dims<M: Math>(&self, math: &M, name: &str) -> Vec<String> {
        <M::ExpandedVector as Storable<_>>::dims(math, name)
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn stat_coords<M: Math>(&self, math: &M) -> HashMap<String, Value> {
        let dims = StatsDims::from(math);
        dims.coords()
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Progress {
    pub draw: u64,
    pub chain: u64,
    pub diverging: bool,
    pub tuning: bool,
    pub step_size: f64,
    pub num_steps: u64,
}

mod private {
    use crate::DiagGradNutsSettings;

    use super::{LowRankNutsSettings, TransformedNutsSettings};

    pub trait Sealed {}

    impl Sealed for DiagGradNutsSettings {}

    impl Sealed for LowRankNutsSettings {}

    impl Sealed for TransformedNutsSettings {}
}

/// Settings for the NUTS sampler
#[derive(Debug, Clone, Copy, Serialize)]
pub struct NutsSettings<A: Debug + Copy + Default + Serialize> {
    /// The number of tuning steps, where we fit the step size and mass matrix.
    pub num_tune: u64,
    /// The number of draws after tuning
    pub num_draws: u64,
    /// The maximum tree depth during sampling. The number of leapfrog steps
    /// is smaller than 2 ^ maxdepth.
    pub maxdepth: u64,
    /// The minimum tree depth during sampling. The number of leapfrog steps
    /// is larger than 2 ^ mindepth.
    pub mindepth: u64,
    /// Store the gradient in the SampleStats
    pub store_gradient: bool,
    /// Store each unconstrained parameter vector in the sampler stats
    pub store_unconstrained: bool,
    /// If the energy error is larger than this threshold we treat the leapfrog
    /// step as a divergence.
    pub max_energy_error: f64,
    /// Store detailed information about each divergence in the sampler stats
    pub store_divergences: bool,
    /// Settings for mass matrix adaptation.
    pub adapt_options: A,
    pub check_turning: bool,

    pub num_chains: usize,
    pub seed: u64,
}

pub type DiagGradNutsSettings = NutsSettings<EuclideanAdaptOptions<DiagAdaptExpSettings>>;
pub type LowRankNutsSettings = NutsSettings<EuclideanAdaptOptions<LowRankSettings>>;
pub type TransformedNutsSettings = NutsSettings<TransformedSettings>;

impl Default for DiagGradNutsSettings {
    fn default() -> Self {
        Self {
            num_tune: 400,
            num_draws: 1000,
            maxdepth: 10,
            mindepth: 0,
            max_energy_error: 1000f64,
            store_gradient: false,
            store_unconstrained: false,
            store_divergences: false,
            adapt_options: EuclideanAdaptOptions::default(),
            check_turning: true,
            seed: 0,
            num_chains: 6,
        }
    }
}

impl Default for LowRankNutsSettings {
    fn default() -> Self {
        let mut vals = Self {
            num_tune: 800,
            num_draws: 1000,
            maxdepth: 10,
            mindepth: 0,
            max_energy_error: 1000f64,
            store_gradient: false,
            store_unconstrained: false,
            store_divergences: false,
            adapt_options: EuclideanAdaptOptions::default(),
            check_turning: true,
            seed: 0,
            num_chains: 6,
        };
        vals.adapt_options.mass_matrix_update_freq = 10;
        vals
    }
}

impl Default for TransformedNutsSettings {
    fn default() -> Self {
        Self {
            num_tune: 1500,
            num_draws: 1000,
            maxdepth: 10,
            mindepth: 0,
            max_energy_error: 20f64,
            store_gradient: false,
            store_unconstrained: false,
            store_divergences: false,
            adapt_options: Default::default(),
            check_turning: true,
            seed: 0,
            num_chains: 1,
        }
    }
}

type DiagGradNutsChain<M> = NutsChain<M, SmallRng, GlobalStrategy<M, DiagMassMatrixStrategy<M>>>;

type LowRankNutsChain<M> = NutsChain<M, SmallRng, GlobalStrategy<M, LowRankMassMatrixStrategy>>;

type TransformingNutsChain<M> = NutsChain<M, SmallRng, TransformAdaptation>;

impl Settings for LowRankNutsSettings {
    type Chain<M: Math> = LowRankNutsChain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        mut rng: &mut R,
    ) -> Self::Chain<M> {
        let num_tune = self.num_tune;
        let strategy = GlobalStrategy::new(&mut math, self.adapt_options, num_tune, chain);
        let mass_matrix = LowRankMassMatrix::new(&mut math, self.adapt_options.mass_matrix_options);
        let max_energy_error = self.max_energy_error;
        let potential = EuclideanHamiltonian::new(&mut math, mass_matrix, max_energy_error, 1f64);

        let options = NutsOptions {
            maxdepth: self.maxdepth,
            mindepth: self.mindepth,
            store_gradient: self.store_gradient,
            store_divergences: self.store_divergences,
            store_unconstrained: self.store_unconstrained,
            check_turning: self.check_turning,
        };

        let rng = rand::rngs::SmallRng::try_from_rng(&mut rng).expect("Could not seed rng");

        NutsChain::new(
            math,
            potential,
            strategy,
            options,
            rng,
            chain,
            self.stats_options(),
        )
    }

    fn hint_num_tune(&self) -> usize {
        self.num_tune as _
    }

    fn hint_num_draws(&self) -> usize {
        self.num_draws as _
    }

    fn num_chains(&self) -> usize {
        self.num_chains
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions {
        StatOptions {
            adapt: GlobalStrategyStatsOptions {
                mass_matrix: (),
                step_size: (),
            },
            hamiltonian: (),
            point: (),
        }
    }
}

impl Settings for DiagGradNutsSettings {
    type Chain<M: Math> = DiagGradNutsChain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        mut rng: &mut R,
    ) -> Self::Chain<M> {
        let num_tune = self.num_tune;
        let strategy = GlobalStrategy::new(&mut math, self.adapt_options, num_tune, chain);
        let mass_matrix = DiagMassMatrix::new(
            &mut math,
            self.adapt_options.mass_matrix_options.store_mass_matrix,
        );
        let max_energy_error = self.max_energy_error;
        let potential = EuclideanHamiltonian::new(&mut math, mass_matrix, max_energy_error, 1f64);

        let options = NutsOptions {
            maxdepth: self.maxdepth,
            mindepth: self.mindepth,
            store_gradient: self.store_gradient,
            store_divergences: self.store_divergences,
            store_unconstrained: self.store_unconstrained,
            check_turning: self.check_turning,
        };

        let rng = rand::rngs::SmallRng::try_from_rng(&mut rng).expect("Could not seed rng");

        NutsChain::new(
            math,
            potential,
            strategy,
            options,
            rng,
            chain,
            self.stats_options(),
        )
    }

    fn hint_num_tune(&self) -> usize {
        self.num_tune as _
    }

    fn hint_num_draws(&self) -> usize {
        self.num_draws as _
    }

    fn num_chains(&self) -> usize {
        self.num_chains
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions {
        StatOptions {
            adapt: GlobalStrategyStatsOptions {
                mass_matrix: (),
                step_size: (),
            },
            hamiltonian: (),
            point: (),
        }
    }
}

impl Settings for TransformedNutsSettings {
    type Chain<M: Math> = TransformingNutsChain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        mut rng: &mut R,
    ) -> Self::Chain<M> {
        let num_tune = self.num_tune;
        let max_energy_error = self.max_energy_error;

        let strategy = TransformAdaptation::new(&mut math, self.adapt_options, num_tune, chain);
        let hamiltonian = TransformedHamiltonian::new(&mut math, max_energy_error);

        let options = NutsOptions {
            maxdepth: self.maxdepth,
            mindepth: self.mindepth,
            store_gradient: self.store_gradient,
            store_divergences: self.store_divergences,
            store_unconstrained: self.store_unconstrained,
            check_turning: self.check_turning,
        };

        let rng = rand::rngs::SmallRng::try_from_rng(&mut rng).expect("Could not seed rng");
        NutsChain::new(
            math,
            hamiltonian,
            strategy,
            options,
            rng,
            chain,
            self.stats_options(),
        )
    }

    fn hint_num_tune(&self) -> usize {
        self.num_tune as _
    }

    fn hint_num_draws(&self) -> usize {
        self.num_draws as _
    }

    fn num_chains(&self) -> usize {
        self.num_chains
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions {
        // TODO make extra config
        let point = TransformedPointStatsOptions {
            store_transformed: self.store_unconstrained,
        };
        StatOptions {
            adapt: (),
            hamiltonian: (),
            point,
        }
    }
}

pub fn sample_sequentially<'math, M: Math + 'math, R: Rng + ?Sized>(
    math: M,
    settings: DiagGradNutsSettings,
    start: &[f64],
    draws: u64,
    chain: u64,
    rng: &mut R,
) -> Result<impl Iterator<Item = Result<(Box<[f64]>, Progress)>> + 'math> {
    let mut sampler = settings.new_chain(chain, math, rng);
    sampler.set_position(start)?;
    Ok((0..draws).map(move |_| sampler.draw()))
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ChainProgress {
    pub finished_draws: usize,
    pub total_draws: usize,
    pub divergences: usize,
    pub tuning: bool,
    pub started: bool,
    pub latest_num_steps: usize,
    pub total_num_steps: usize,
    pub step_size: f64,
    pub runtime: Duration,
    pub divergent_draws: Vec<usize>,
}

impl ChainProgress {
    fn new(total: usize) -> Self {
        Self {
            finished_draws: 0,
            total_draws: total,
            divergences: 0,
            tuning: true,
            started: false,
            latest_num_steps: 0,
            step_size: 0f64,
            total_num_steps: 0,
            runtime: Duration::ZERO,
            divergent_draws: Vec::new(),
        }
    }

    fn update(&mut self, stats: &Progress, draw_duration: Duration) {
        if stats.diverging & !stats.tuning {
            self.divergences += 1;
            self.divergent_draws.push(self.finished_draws);
        }
        self.finished_draws += 1;
        self.tuning = stats.tuning;

        self.latest_num_steps = stats.num_steps as usize;
        self.total_num_steps += stats.num_steps as usize;
        self.step_size = stats.step_size;
        self.runtime += draw_duration;
    }
}

enum ChainCommand {
    Resume,
    Pause,
}

struct ChainProcess<T>
where
    T: TraceStorage,
{
    stop_marker: Sender<ChainCommand>,
    trace: Arc<Mutex<Option<T::ChainStorage>>>,
    progress: Arc<Mutex<ChainProgress>>,
}

impl<T: TraceStorage> ChainProcess<T> {
    fn finalize_many(trace: T, chains: Vec<Self>) -> Result<(Option<anyhow::Error>, T::Finalized)> {
        let finalized_chain_traces = chains
            .into_iter()
            .filter_map(|chain| chain.trace.lock().expect("Poisoned lock").take())
            .map(|chain| chain.finalize())
            .collect_vec();
        trace.finalize(finalized_chain_traces)
    }

    fn progress(&self) -> ChainProgress {
        self.progress.lock().expect("Poisoned lock").clone()
    }

    fn resume(&self) -> Result<()> {
        self.stop_marker.send(ChainCommand::Resume)?;
        Ok(())
    }

    fn pause(&self) -> Result<()> {
        self.stop_marker.send(ChainCommand::Pause)?;
        Ok(())
    }

    fn start<'model, M: Model, S: Settings>(
        model: &'model M,
        chain_trace: T::ChainStorage,
        chain_id: u64,
        seed: u64,
        settings: &'model S,
        scope: &ScopeFifo<'model>,
        results: Sender<Result<()>>,
    ) -> Result<Self> {
        let (stop_marker_tx, stop_marker_rx) = channel();

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(chain_id + 1);

        let chain_trace = Arc::new(Mutex::new(Some(chain_trace)));
        let progress = Arc::new(Mutex::new(ChainProgress::new(
            settings.hint_num_draws() + settings.hint_num_tune(),
        )));

        let trace_inner = chain_trace.clone();
        let progress_inner = progress.clone();

        scope.spawn_fifo(move |_| {
            let chain_trace = trace_inner;
            let progress = progress_inner;

            let mut sample = move || {
                let logp = model
                    .math(&mut rng)
                    .context("Failed to create model density")?;
                let dim = logp.dim();

                let mut sampler = settings.new_chain(chain_id, logp, &mut rng);

                progress.lock().expect("Poisoned mutex").started = true;

                let mut initval = vec![0f64; dim];
                // TODO maxtries
                let mut error = None;
                for _ in 0..500 {
                    model
                        .init_position(&mut rng, &mut initval)
                        .context("Failed to generate a new initial position")?;
                    if let Err(err) = sampler.set_position(&initval) {
                        error = Some(err);
                        continue;
                    }
                    error = None;
                    break;
                }

                if let Some(error) = error {
                    return Err(error.context("All initialization points failed"));
                }

                let draws = settings.hint_num_tune() + settings.hint_num_draws();

                let mut msg = stop_marker_rx.try_recv();
                let mut draw = 0;
                loop {
                    match msg {
                        // The remote end is dead
                        Err(TryRecvError::Disconnected) => {
                            break;
                        }
                        Err(TryRecvError::Empty) => {}
                        Ok(ChainCommand::Pause) => {
                            msg = stop_marker_rx.recv().map_err(|e| e.into());
                            continue;
                        }
                        Ok(ChainCommand::Resume) => {}
                    }

                    let now = Instant::now();
                    //let (point, info) = sampler.draw().unwrap();
                    let (_point, draw_data, stats, info) = sampler.expanded_draw().unwrap();

                    let mut guard = chain_trace
                        .lock()
                        .expect("Could not unlock trace lock. Poisoned mutex");

                    let Some(trace_val) = guard.as_mut() else {
                        // The trace was removed by controller thread. We can stop sampling
                        break;
                    };
                    progress
                        .lock()
                        .expect("Poisoned mutex")
                        .update(&info, now.elapsed());

                    let math = sampler.math();
                    let dims = StatsDims::from(math.deref());
                    trace_val.record_sample(
                        settings,
                        stats.get_all(&dims),
                        draw_data.get_all(math.deref()),
                        &info,
                    )?;

                    draw += 1;
                    if draw == draws {
                        break;
                    }

                    msg = stop_marker_rx.try_recv();
                }
                Ok(())
            };

            let result = sample();

            // We intentionally ignore errors here, because this means some other
            // chain already failed, and should have reported the error.
            let _ = results.send(result);
            drop(results);
        });

        Ok(Self {
            trace: chain_trace,
            stop_marker: stop_marker_tx,
            progress,
        })
    }

    fn flush(&self) -> Result<()> {
        self.trace
            .lock()
            .map_err(|_| anyhow::anyhow!("Could not lock trace mutex"))
            .context("Could not flush trace")?
            .as_mut().map(|v| v.flush())
            .transpose()?;
        Ok(())
    }
}

#[derive(Debug)]
enum SamplerCommand {
    Pause,
    Continue,
    Progress,
    Flush,
}

enum SamplerResponse {
    Ok(),
    Progress(Box<[ChainProgress]>),
}

pub enum SamplerWaitResult<F: Send + 'static> {
    Trace(F),
    Timeout(Sampler<F>),
    Err(anyhow::Error, Option<F>),
}

pub struct Sampler<F: Send + 'static> {
    main_thread: JoinHandle<Result<(Option<anyhow::Error>, F)>>,
    commands: SyncSender<SamplerCommand>,
    responses: Receiver<SamplerResponse>,
    results: Receiver<Result<()>>,
}

pub struct ProgressCallback {
    pub callback: Box<dyn FnMut(Duration, Box<[ChainProgress]>) + Send>,
    pub rate: Duration,
}

impl<F: Send + 'static> Sampler<F> {
    pub fn new<M, S, C, T>(
        model: M,
        settings: S,
        trace_config: C,
        num_cores: usize,
        callback: Option<ProgressCallback>,
    ) -> Result<Self>
    where
        S: Settings,
        C: StorageConfig<Storage = T>,
        M: Model,
        T: TraceStorage<Finalized = F>,
    {
        let (commands_tx, commands_rx) = sync_channel(0);
        let (responses_tx, responses_rx) = sync_channel(0);
        let (results_tx, results_rx) = channel();

        let main_thread = spawn(move || {
            let pool = ThreadPoolBuilder::new()
                .num_threads(num_cores + 1) // One more thread because the controller also uses one
                .thread_name(|i| format!("nutpie-worker-{i}"))
                .build()
                .context("Could not start thread pool")?;

            let settings_ref = &settings;
            let model_ref = &model;
            let mut callback = callback;

            pool.scope_fifo(move |scope| {
                let results = results_tx;
                let mut chains = Vec::with_capacity(settings.num_chains());

                let mut rng = ChaCha8Rng::seed_from_u64(settings.seed());
                rng.set_stream(0);

                let math = model_ref
                    .math(&mut rng)
                    .context("Could not create model density")?;
                let trace = trace_config
                    .new_trace(settings_ref, &math)
                    .context("Could not create trace object")?;
                drop(math);

                for chain_id in 0..settings.num_chains() {
                    let chain_trace_val = trace
                        .initialize_trace_for_chain(chain_id as u64)
                        .context("Failed to create trace object")?;
                    let chain = ChainProcess::start(
                        model_ref,
                        chain_trace_val,
                        chain_id as u64,
                        settings.seed(),
                        settings_ref,
                        scope,
                        results.clone(),
                    );
                    chains.push(chain);
                }
                drop(results);

                let (chains, errors): (Vec<_>, Vec<_>) = chains.into_iter().partition_result();
                if let Some(error) = errors.into_iter().next() {
                    let _ = ChainProcess::finalize_many(trace, chains);
                    return Err(error).context("Could not start chains");
                }

                let mut main_loop = || {
                    let start_time = Instant::now();
                    let mut pause_start = Instant::now();
                    let mut pause_time = Duration::ZERO;

                    let mut progress_rate = Duration::MAX;
                    if let Some(ProgressCallback { callback, rate }) = &mut callback {
                        let progress = chains.iter().map(|chain| chain.progress()).collect_vec();
                        callback(start_time.elapsed(), progress.into());
                        progress_rate = *rate;
                    }
                    let mut last_progress = Instant::now();
                    let mut is_paused = false;

                    loop {
                        let timeout = progress_rate.checked_sub(last_progress.elapsed());
                        let timeout = timeout.unwrap_or_else(|| {
                            if let Some(ProgressCallback { callback, .. }) = &mut callback {
                                let progress =
                                    chains.iter().map(|chain| chain.progress()).collect_vec();
                                let mut elapsed = start_time.elapsed().saturating_sub(pause_time);
                                if is_paused {
                                    elapsed = elapsed.saturating_sub(pause_start.elapsed());
                                }
                                callback(elapsed, progress.into());
                            }
                            last_progress = Instant::now();
                            progress_rate
                        });

                        // TODO return when all chains are done
                        match commands_rx.recv_timeout(timeout) {
                            Ok(SamplerCommand::Pause) => {
                                for chain in chains.iter() {
                                    // This failes if the thread is done.
                                    // We just want to ignore those threads.
                                    let _ = chain.pause();
                                }
                                if !is_paused {
                                    pause_start = Instant::now();
                                }
                                is_paused = true;
                                responses_tx.send(SamplerResponse::Ok())?;
                            }
                            Ok(SamplerCommand::Continue) => {
                                for chain in chains.iter() {
                                    // This failes if the thread is done.
                                    // We just want to ignore those threads.
                                    let _ = chain.resume();
                                }
                                pause_time += pause_start.elapsed();
                                is_paused = false;
                                responses_tx.send(SamplerResponse::Ok())?;
                            }
                            Ok(SamplerCommand::Progress) => {
                                let progress =
                                    chains.iter().map(|chain| chain.progress()).collect_vec();
                                responses_tx.send(SamplerResponse::Progress(progress.into()))?;
                            }
                            Ok(SamplerCommand::Flush) => {
                                for chain in chains.iter() {
                                    chain.flush()?;
                                }
                                responses_tx.send(SamplerResponse::Ok())?;
                            }
                            Err(RecvTimeoutError::Timeout) => {}
                            Err(RecvTimeoutError::Disconnected) => {
                                if let Some(ProgressCallback { callback, .. }) = &mut callback {
                                    let progress =
                                        chains.iter().map(|chain| chain.progress()).collect_vec();
                                    let mut elapsed =
                                        start_time.elapsed().saturating_sub(pause_time);
                                    if is_paused {
                                        elapsed = elapsed.saturating_sub(pause_start.elapsed());
                                    }
                                    callback(elapsed, progress.into());
                                }
                                return Ok(());
                            }
                        };
                    }
                };
                let result: Result<()> = main_loop();
                // Run finalization even if something failed
                let output = ChainProcess::finalize_many(trace, chains)?;

                result?;
                Ok(output)
            })
        });

        Ok(Self {
            main_thread,
            commands: commands_tx,
            responses: responses_rx,
            results: results_rx,
        })
    }

    pub fn pause(&mut self) -> Result<()> {
        self.commands
            .send(SamplerCommand::Pause)
            .context("Could not send pause command to controller thread")?;
        let response = self
            .responses
            .recv()
            .context("Could not recieve pause response from controller thread")?;
        let SamplerResponse::Ok() = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(())
    }

    pub fn resume(&mut self) -> Result<()> {
        self.commands.send(SamplerCommand::Continue)?;
        let response = self.responses.recv()?;
        let SamplerResponse::Ok() = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.commands.send(SamplerCommand::Flush)?;
        let response = self
            .responses
            .recv()
            .context("Could not recieve flush response from controller thread")?;
        let SamplerResponse::Ok() = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(())
    }

    pub fn abort(self) -> Result<(Option<anyhow::Error>, F)> {
        drop(self.commands);
        let result = self.main_thread.join();
        match result {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(Ok(val)) => Ok(val),
            Ok(Err(err)) => Err(err),
        }
    }

    pub fn wait_timeout(self, timeout: Duration) -> SamplerWaitResult<F> {
        let start = Instant::now();
        let mut remaining = Some(timeout);
        while remaining.is_some() {
            match self.results.recv_timeout(timeout) {
                Ok(Ok(_)) => remaining = timeout.checked_sub(start.elapsed()),
                Ok(Err(e)) => return SamplerWaitResult::Err(e, None),
                Err(RecvTimeoutError::Disconnected) => match self.abort() {
                    Ok((Some(err), trace)) => return SamplerWaitResult::Err(err, Some(trace)),
                    Ok((None, trace)) => return SamplerWaitResult::Trace(trace),
                    Err(err) => return SamplerWaitResult::Err(err, None),
                },
                Err(RecvTimeoutError::Timeout) => break,
            }
        }
        SamplerWaitResult::Timeout(self)
    }

    pub fn progress(&mut self) -> Result<Box<[ChainProgress]>> {
        self.commands.send(SamplerCommand::Progress)?;
        let response = self.responses.recv()?;
        let SamplerResponse::Progress(progress) = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(progress)
    }
}

#[cfg(test)]
pub mod test_logps {

    use std::collections::HashMap;

    use crate::{
        Model,
        cpu_math::{CpuLogpFunc, CpuMath},
        math_base::LogpError,
    };
    use anyhow::Result;
    use nuts_storable::HasDims;
    use rand::Rng;
    use thiserror::Error;

    #[derive(Clone, Debug)]
    pub struct NormalLogp {
        pub dim: usize,
        pub mu: f64,
    }

    #[derive(Error, Debug)]
    pub enum NormalLogpError {}

    impl LogpError for NormalLogpError {
        fn is_recoverable(&self) -> bool {
            false
        }
    }

    impl HasDims for &NormalLogp {
        fn dim_sizes(&self) -> HashMap<String, u64> {
            vec![
                ("unconstrained_parameter".to_string(), self.dim as u64),
                ("dim".to_string(), self.dim as u64),
            ]
            .into_iter()
            .collect()
        }
    }

    impl CpuLogpFunc for &NormalLogp {
        type LogpError = NormalLogpError;
        type FlowParameters = ();
        type ExpandedVector = Vec<f64>;

        fn dim(&self) -> usize {
            self.dim
        }

        fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, NormalLogpError> {
            let n = position.len();
            assert!(gradient.len() == n);

            let mut logp = 0f64;
            for (p, g) in position.iter().zip(gradient.iter_mut()) {
                let val = self.mu - p;
                logp -= val * val / 2.;
                *g = val;
            }

            Ok(logp)
        }

        fn expand_vector<R>(
            &mut self,
            _rng: &mut R,
            array: &[f64],
        ) -> std::result::Result<Self::ExpandedVector, crate::cpu_math::CpuMathError>
        where
            R: rand::Rng + ?Sized,
        {
            Ok(array.to_vec())
        }

        fn inv_transform_normalize(
            &mut self,
            _params: &Self::FlowParameters,
            _untransformed_position: &[f64],
            _untransofrmed_gradient: &[f64],
            _transformed_position: &mut [f64],
            _transformed_gradient: &mut [f64],
        ) -> std::result::Result<f64, Self::LogpError> {
            unimplemented!()
        }

        fn init_from_untransformed_position(
            &mut self,
            _params: &Self::FlowParameters,
            _untransformed_position: &[f64],
            _untransformed_gradient: &mut [f64],
            _transformed_position: &mut [f64],
            _transformed_gradient: &mut [f64],
        ) -> std::result::Result<(f64, f64), Self::LogpError> {
            unimplemented!()
        }

        fn init_from_transformed_position(
            &mut self,
            _params: &Self::FlowParameters,
            _untransformed_position: &mut [f64],
            _untransformed_gradient: &mut [f64],
            _transformed_position: &[f64],
            _transformed_gradient: &mut [f64],
        ) -> std::result::Result<(f64, f64), Self::LogpError> {
            unimplemented!()
        }

        fn update_transformation<'b, R: rand::Rng + ?Sized>(
            &'b mut self,
            _rng: &mut R,
            _untransformed_positions: impl Iterator<Item = &'b [f64]>,
            _untransformed_gradients: impl Iterator<Item = &'b [f64]>,
            _untransformed_logp: impl Iterator<Item = &'b f64>,
            _params: &'b mut Self::FlowParameters,
        ) -> std::result::Result<(), Self::LogpError> {
            unimplemented!()
        }

        fn new_transformation<R: rand::Rng + ?Sized>(
            &mut self,
            _rng: &mut R,
            _untransformed_position: &[f64],
            _untransfogmed_gradient: &[f64],
            _chain: u64,
        ) -> std::result::Result<Self::FlowParameters, Self::LogpError> {
            unimplemented!()
        }

        fn transformation_id(
            &self,
            _params: &Self::FlowParameters,
        ) -> std::result::Result<i64, Self::LogpError> {
            unimplemented!()
        }
    }

    pub struct CpuModel<F> {
        logp: F,
    }

    impl<F> CpuModel<F> {
        pub fn new(logp: F) -> Self {
            Self { logp }
        }
    }

    impl<F> Model for CpuModel<F>
    where
        F: Send + Sync + 'static,
        for<'a> &'a F: CpuLogpFunc,
    {
        type Math<'model> = CpuMath<&'model F>;

        fn math<R: Rng + ?Sized>(&self, _rng: &mut R) -> Result<Self::Math<'_>> {
            Ok(CpuMath::new(&self.logp))
        }

        fn init_position<R: rand::prelude::Rng + ?Sized>(
            &self,
            _rng: &mut R,
            position: &mut [f64],
        ) -> Result<()> {
            position.iter_mut().for_each(|x| *x = 0.);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use super::test_logps::NormalLogp;
    use crate::{
        Chain, DiagGradNutsSettings, Sampler, ZarrConfig,
        cpu_math::CpuMath,
        sample_sequentially,
        sampler::{Settings, test_logps::CpuModel},
    };

    use anyhow::Result;
    use itertools::Itertools;
    use pretty_assertions::assert_eq;
    use rand::{SeedableRng, rngs::StdRng};
    use zarrs::storage::store::MemoryStore;

    #[test]
    fn sample_chain() -> Result<()> {
        let logp = NormalLogp { dim: 10, mu: 0.1 };
        let math = CpuMath::new(&logp);
        let settings = DiagGradNutsSettings {
            num_tune: 100,
            num_draws: 100,
            ..Default::default()
        };
        let start = vec![0.2; 10];

        let mut rng = StdRng::seed_from_u64(42);

        let mut chain = settings.new_chain(0, math, &mut rng);

        let (_draw, info) = chain.draw()?;
        assert!(info.tuning);
        assert_eq!(info.draw, 0);

        let math = CpuMath::new(&logp);
        let chain = sample_sequentially(math, settings, &start, 200, 1, &mut rng).unwrap();
        let mut draws = chain.collect_vec();
        assert_eq!(draws.len(), 200);

        let draw0 = draws.remove(100).unwrap();
        let (vals, stats) = draw0;
        assert_eq!(vals.len(), 10);
        assert_eq!(stats.chain, 1);
        assert_eq!(stats.draw, 100);
        Ok(())
    }

    #[test]
    fn sample_parallel() -> Result<()> {
        let logp = NormalLogp { dim: 100, mu: 0.1 };
        let settings = DiagGradNutsSettings {
            num_tune: 100,
            num_draws: 100,
            seed: 10,
            ..Default::default()
        };

        let model = CpuModel::new(logp.clone());
        let store = MemoryStore::new();

        let zarr_config = ZarrConfig::new(Arc::new(store));
        let mut sampler = Sampler::new(model, settings, zarr_config, 4, None)?;
        sampler.pause()?;
        sampler.pause()?;
        // TODO flush trace
        sampler.resume()?;
        let (ok, _) = sampler.abort()?;
        if let Some(err) = ok {
            Err(err)?;
        }

        let store = MemoryStore::new();
        let zarr_config = ZarrConfig::new(Arc::new(store));
        let model = CpuModel::new(logp.clone());
        let mut sampler = Sampler::new(model, settings, zarr_config, 4, None)?;
        sampler.pause()?;
        if let (Some(err), _) = sampler.abort()? {
            Err(err)?;
        }

        let store = MemoryStore::new();
        let zarr_config = ZarrConfig::new(Arc::new(store));
        let model = CpuModel::new(logp.clone());
        let start = Instant::now();
        let sampler = Sampler::new(model, settings, zarr_config, 4, None)?;

        let mut sampler = match sampler.wait_timeout(Duration::from_nanos(100)) {
            super::SamplerWaitResult::Trace(_) => {
                dbg!(start.elapsed());
                panic!("finished");
            }
            super::SamplerWaitResult::Timeout(sampler) => sampler,
            super::SamplerWaitResult::Err(_, _) => {
                panic!("error")
            }
        };

        for _ in 0..30 {
            sampler.progress()?;
        }

        match sampler.wait_timeout(Duration::from_secs(1)) {
            super::SamplerWaitResult::Trace(_) => {
                dbg!(start.elapsed());
            }
            super::SamplerWaitResult::Timeout(_) => {
                panic!("timeout")
            }
            super::SamplerWaitResult::Err(err, _) => Err(err)?,
        };

        Ok(())
    }

    #[test]
    fn sample_seq() {
        let logp = NormalLogp { dim: 10, mu: 0.1 };
        let math = CpuMath::new(&logp);
        let settings = DiagGradNutsSettings {
            num_tune: 100,
            num_draws: 100,
            ..Default::default()
        };
        let start = vec![0.2; 10];

        let mut rng = StdRng::seed_from_u64(42);

        let chain = sample_sequentially(math, settings, &start, 200, 1, &mut rng).unwrap();
        let mut draws = chain.collect_vec();
        assert_eq!(draws.len(), 200);

        let draw0 = draws.remove(100).unwrap();
        let (vals, stats) = draw0;
        assert_eq!(vals.len(), 10);
        assert_eq!(stats.chain, 1);
        assert_eq!(stats.draw, 100);
    }
}
