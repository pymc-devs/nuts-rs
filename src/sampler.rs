use anyhow::{bail, Context, Result};
use arrow::array::Array;
use itertools::Itertools;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::{ScopeFifo, ThreadPoolBuilder};
use std::{
    fmt::Debug,
    sync::{
        mpsc::{
            channel, sync_channel, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError,
        },
        Arc, Mutex,
    },
    thread::{spawn, JoinHandle},
    time::{Duration, Instant},
};

use crate::{
    adapt_strategy::{EuclideanAdaptOptions, GlobalStrategy},
    chain::{AdaptStrategy, Chain, NutsChain, StatOptions},
    euclidean_hamiltonian::EuclideanHamiltonian,
    low_rank_mass_matrix::{LowRankMassMatrix, LowRankMassMatrixStrategy, LowRankSettings},
    mass_matrix::DiagMassMatrix,
    mass_matrix_adapt::Strategy as DiagMassMatrixStrategy,
    math_base::Math,
    nuts::NutsOptions,
    sampler_stats::{SamplerStats, StatTraceBuilder},
    transform_adapt_strategy::{TransformAdaptation, TransformedSettings},
    transformed_hamiltonian::{TransformedHamiltonian, TransformedPointStatsOptions},
    DiagAdaptExpSettings,
};

/// All sampler configurations implement this trait
pub trait Settings: private::Sealed + Clone + Copy + Default + Sync + Send + 'static {
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
    fn stats_options<M: Math>(
        &self,
        chain: &Self::Chain<M>,
    ) -> <Self::Chain<M> as SamplerStats<M>>::StatOptions;
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
#[derive(Debug, Clone, Copy)]
pub struct NutsSettings<A: Debug + Copy + Default> {
    /// The number of tuning steps, where we fit the step size and mass matrix.
    pub num_tune: u64,
    /// The number of draws after tuning
    pub num_draws: u64,
    /// The maximum tree depth during sampling. The number of leapfrog steps
    /// is smaller than 2 ^ maxdepth.
    pub maxdepth: u64,
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
            num_tune: 4000,
            num_draws: 1000,
            maxdepth: 8,
            max_energy_error: 1000f64,
            store_gradient: false,
            store_unconstrained: false,
            store_divergences: false,
            adapt_options: Default::default(),
            check_turning: true,
            seed: 0,
            num_chains: 6,
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
        rng: &mut R,
    ) -> Self::Chain<M> {
        let num_tune = self.num_tune;
        let strategy = GlobalStrategy::new(&mut math, self.adapt_options, num_tune, chain);
        let mass_matrix = LowRankMassMatrix::new(&mut math, self.adapt_options.mass_matrix_options);
        let max_energy_error = self.max_energy_error;
        let potential = EuclideanHamiltonian::new(&mut math, mass_matrix, max_energy_error, 1f64);

        let options = NutsOptions {
            maxdepth: self.maxdepth,
            store_gradient: self.store_gradient,
            store_divergences: self.store_divergences,
            store_unconstrained: self.store_unconstrained,
            check_turning: self.check_turning,
        };

        let rng = rand::rngs::SmallRng::from_rng(rng).expect("Could not seed rng");

        NutsChain::new(math, potential, strategy, options, rng, chain)
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

    fn stats_options<M: Math>(
        &self,
        _chain: &Self::Chain<M>,
    ) -> <Self::Chain<M> as SamplerStats<M>>::StatOptions {
        StatOptions {
            adapt: (),
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
        rng: &mut R,
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
            store_gradient: self.store_gradient,
            store_divergences: self.store_divergences,
            store_unconstrained: self.store_unconstrained,
            check_turning: self.check_turning,
        };

        let rng = rand::rngs::SmallRng::from_rng(rng).expect("Could not seed rng");

        NutsChain::new(math, potential, strategy, options, rng, chain)
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

    fn stats_options<M: Math>(
        &self,
        _chain: &Self::Chain<M>,
    ) -> <Self::Chain<M> as SamplerStats<M>>::StatOptions {
        StatOptions {
            adapt: (),
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
        rng: &mut R,
    ) -> Self::Chain<M> {
        let num_tune = self.num_tune;
        let max_energy_error = self.max_energy_error;

        let strategy = TransformAdaptation::new(&mut math, self.adapt_options, num_tune, chain);
        let hamiltonian = TransformedHamiltonian::new(&mut math, max_energy_error);

        let options = NutsOptions {
            maxdepth: self.maxdepth,
            store_gradient: self.store_gradient,
            store_divergences: self.store_divergences,
            store_unconstrained: self.store_unconstrained,
            check_turning: self.check_turning,
        };

        let rng = rand::rngs::SmallRng::from_rng(rng).expect("Could not seed rng");
        NutsChain::new(math, hamiltonian, strategy, options, rng, chain)
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

    fn stats_options<M: Math>(
        &self,
        _chain: &Self::Chain<M>,
    ) -> <Self::Chain<M> as SamplerStats<M>>::StatOptions {
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

pub trait DrawStorage: Send {
    fn append_value(&mut self, point: &[f64]) -> Result<()>;
    fn finalize(self) -> Result<Arc<dyn Array>>;
    fn inspect(&self) -> Result<Arc<dyn Array>>;
}

pub trait Model: Send + Sync + 'static {
    type Math<'model>: Math
    where
        Self: 'model;
    type DrawStorage<'model, S: Settings>: DrawStorage
    where
        Self: 'model;

    fn new_trace<'model, S: Settings, R: Rng + ?Sized>(
        &'model self,
        rng: &mut R,
        chain_id: u64,
        settings: &'model S,
    ) -> Result<Self::DrawStorage<'model, S>>;
    fn math(&self) -> Result<Self::Math<'_>>;
    fn init_position<R: Rng + ?Sized>(&self, rng: &mut R, position: &mut [f64]) -> Result<()>;
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

pub struct ChainOutput {
    pub draws: Arc<dyn Array>,
    pub stats: Arc<dyn Array>,
    pub chain_id: u64,
}

enum ChainCommand {
    Resume,
    Pause,
}

struct ChainTrace<'model, M: Model + 'model, S: Settings> {
    draws_builder: M::DrawStorage<'model, S>,
    stats_builder: <S::Chain<M::Math<'model>> as SamplerStats<M::Math<'model>>>::Builder,
    chain_id: u64,
}

impl<'model, M: Model + 'model, S: Settings> ChainTrace<'model, M, S> {
    fn inspect(&self) -> Result<ChainOutput> {
        let stats = self.stats_builder.inspect().expect("No sample stats");
        let draws = self.draws_builder.inspect()?;
        Ok(ChainOutput {
            chain_id: self.chain_id,
            draws,
            stats: Arc::new(stats) as Arc<_>,
        })
    }

    fn finalize(self) -> Result<ChainOutput> {
        let draws = self.draws_builder.finalize()?;
        let stats = self.stats_builder.finalize().expect("No sample stats");
        Ok(ChainOutput {
            chain_id: self.chain_id,
            draws,
            stats: Arc::new(stats) as Arc<dyn Array>,
        })
    }
}

struct ChainProcess<'model, M, S>
where
    M: Model + 'model,
    S: Settings,
{
    stop_marker: Sender<ChainCommand>,
    trace: Arc<Mutex<Option<ChainTrace<'model, M, S>>>>,
    progress: Arc<Mutex<ChainProgress>>,
}

impl<'scope, M: Model, S: Settings> ChainProcess<'scope, M, S> {
    fn finalize_many(chains: Vec<Self>) -> Vec<Result<Option<ChainOutput>>> {
        chains
            .into_iter()
            .map(|chain| chain.finalize())
            .collect_vec()
    }

    fn progress(&self) -> ChainProgress {
        self.progress.lock().expect("Poisoned lock").clone()
    }

    fn current_trace(&self) -> Result<Option<ChainOutput>> {
        self.trace
            .lock()
            .expect("Poisoned lock")
            .as_ref()
            .map(|trace| trace.inspect())
            .transpose()
    }

    fn resume(&self) -> Result<()> {
        self.stop_marker.send(ChainCommand::Resume)?;
        Ok(())
    }

    fn pause(&self) -> Result<()> {
        self.stop_marker.send(ChainCommand::Pause)?;
        Ok(())
    }

    fn finalize(self) -> Result<Option<ChainOutput>> {
        drop(self.stop_marker);
        self.trace
            .lock()
            .expect("Poisoned lock")
            .take()
            .map(|trace| trace.finalize())
            .transpose()
    }

    fn start<'model>(
        model: &'model M,
        chain_id: u64,
        seed: u64,
        settings: &'model S,
        scope: &ScopeFifo<'scope>,
        results: Sender<Result<()>>,
    ) -> Result<Self>
    where
        'model: 'scope,
    {
        let (stop_marker_tx, stop_marker_rx) = channel();

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(chain_id);

        let trace = Arc::new(Mutex::new(None));
        let progress = Arc::new(Mutex::new(ChainProgress::new(
            settings.hint_num_draws() + settings.hint_num_tune(),
        )));

        let trace_inner = trace.clone();
        let progress_inner = progress.clone();

        scope.spawn_fifo(move |_| {
            let trace = trace_inner;
            let progress = progress_inner;

            let mut sample = move || {
                let logp = model.math().context("Failed to create model density")?;
                let dim = logp.dim();

                let mut sampler = settings.new_chain(chain_id, logp, &mut rng);

                let draw_trace = model
                    .new_trace(&mut rng, chain_id, settings)
                    .context("Failed to create trace object")?;
                let stat_opts = settings.stats_options(&sampler);
                let stats_trace = sampler.new_builder(stat_opts, settings, dim);

                let new_trace = ChainTrace {
                    draws_builder: draw_trace,
                    stats_builder: stats_trace,
                    chain_id,
                };
                *trace.lock().expect("Poisoned mutex") = Some(new_trace);
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
                    let (point, info) = sampler.draw().unwrap();
                    let mut guard = trace
                        .lock()
                        .expect("Could not unlock trace lock. Poisoned mutex");

                    let Some(val) = guard.as_mut() else {
                        // The trace was removed by controller thread. We can stop sampling
                        break;
                    };
                    progress
                        .lock()
                        .expect("Poisoned mutex")
                        .update(&info, now.elapsed());
                    DrawStorage::append_value(&mut val.draws_builder, &point)?;
                    StatTraceBuilder::append_value(&mut val.stats_builder, None, &sampler);
                    draw += 1;
                    if draw == draws {
                        break;
                    }

                    msg = stop_marker_rx.try_recv();
                }
                Ok(())
            };

            let result = sample();

            results
                .send(result)
                .expect("Could not send sampling results to main thread.");
            drop(results);
        });

        Ok(Self {
            trace,
            stop_marker: stop_marker_tx,
            progress,
        })
    }
}

#[derive(Debug)]
enum SamplerCommand {
    Pause,
    Continue,
    InspectTrace,
    Progress,
}

enum SamplerResponse {
    Ok(),
    IntermediateTrace(Trace),
    Progress(Box<[ChainProgress]>),
}

pub enum SamplerWaitResult {
    Trace(Trace),
    Timeout(Sampler),
    Err(anyhow::Error, Option<Trace>),
}

pub struct Sampler {
    main_thread: JoinHandle<Result<Vec<Result<Option<ChainOutput>>>>>,
    commands: SyncSender<SamplerCommand>,
    responses: Receiver<SamplerResponse>,
    results: Receiver<Result<()>>,
}

pub struct Trace {
    pub chains: Vec<ChainOutput>,
}

impl<I: Iterator<Item = ChainOutput>> From<I> for Trace {
    fn from(value: I) -> Self {
        let mut chains = value.into_iter().collect_vec();
        chains.sort_unstable_by_key(|x| x.chain_id);
        Trace { chains }
    }
}

pub struct ProgressCallback {
    pub callback: Box<dyn FnMut(Duration, Box<[ChainProgress]>) + Send>,
    pub rate: Duration,
}

impl Sampler {
    pub fn new<S: Settings, M: Model>(
        model: M,
        settings: S,
        num_cores: usize,
        callback: Option<ProgressCallback>,
    ) -> Result<Self> {
        let (commands_tx, commands_rx) = sync_channel(0);
        let (responses_tx, responses_rx) = sync_channel(0);
        let (results_tx, results_rx) = channel();

        let main_thread = spawn(move || {
            let pool = ThreadPoolBuilder::new()
                .num_threads(num_cores + 1) // One more thread because the controller also uses one
                .thread_name(|i| format!("nutpie-worker-{}", i))
                .build()
                .context("Could not start thread pool")?;

            let settings_ref = &settings;
            let model_ref = &model;
            let mut callback = callback;

            pool.scope_fifo(move |scope| {
                let results = results_tx;
                let mut chains = Vec::with_capacity(settings.num_chains());

                for chain_id in 0..settings.num_chains() {
                    let chain = ChainProcess::start(
                        model_ref,
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
                    let _ = ChainProcess::finalize_many(chains);
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
                            Ok(SamplerCommand::InspectTrace) => {
                                let traces: Result<Vec<_>> =
                                    chains.iter().map(|chain| chain.current_trace()).collect();
                                responses_tx.send(SamplerResponse::IntermediateTrace(
                                    traces?.into_iter().flatten().into(),
                                ))?;
                            }
                            Ok(SamplerCommand::Progress) => {
                                let progress =
                                    chains.iter().map(|chain| chain.progress()).collect_vec();
                                responses_tx.send(SamplerResponse::Progress(progress.into()))?;
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
                let output = Ok(ChainProcess::finalize_many(chains));

                result?;
                output
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

    pub fn abort(self) -> (Result<()>, Option<Trace>) {
        drop(self.commands);
        let result = self.main_thread.join();
        match result {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(Ok(traces)) => {
                let (traces, errors): (Vec<_>, Vec<_>) = traces.into_iter().partition_result();
                let trace: Trace = traces.into_iter().flatten().into();
                match errors.into_iter().next() {
                    Some(err) => (Err(err), Some(trace)),
                    None => (Ok(()), Some(trace)),
                }
            }
            Ok(Err(err)) => (Err(err), None),
        }
    }

    pub fn inspect_trace(&mut self) -> Result<Trace> {
        self.commands.send(SamplerCommand::InspectTrace)?;
        let response = self.responses.recv()?;
        let SamplerResponse::IntermediateTrace(trace) = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(trace)
    }

    pub fn wait_timeout(self, timeout: Duration) -> SamplerWaitResult {
        let start = Instant::now();
        let mut remaining = Some(timeout);
        while remaining.is_some() {
            match self.results.recv_timeout(timeout) {
                Ok(Ok(_)) => remaining = timeout.checked_sub(start.elapsed()),
                Ok(Err(e)) => return SamplerWaitResult::Err(e, None),
                Err(RecvTimeoutError::Disconnected) => {
                    let (res, trace) = self.abort();
                    if let Err(err) = res {
                        return SamplerWaitResult::Err(err, trace);
                    }
                    return SamplerWaitResult::Trace(trace.expect("No chains available"));
                }
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

    use std::sync::Arc;

    use crate::{
        cpu_math::{CpuLogpFunc, CpuMath},
        math_base::LogpError,
        Settings,
    };
    use anyhow::Result;
    use arrow::{
        array::{Array, ArrayBuilder, FixedSizeListBuilder, PrimitiveBuilder},
        datatypes::Float64Type,
    };
    use multiversion::multiversion;
    use thiserror::Error;

    use super::{DrawStorage, Model};

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

    impl CpuLogpFunc for &NormalLogp {
        type LogpError = NormalLogpError;
        type TransformParams = ();

        fn dim(&self) -> usize {
            self.dim
        }
        fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, NormalLogpError> {
            let n = position.len();
            assert!(gradient.len() == n);

            #[cfg(feature = "simd_support")]
            #[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
            fn logp_inner(mu: f64, position: &[f64], gradient: &mut [f64]) -> f64 {
                use std::simd::f64x4;
                use std::simd::num::SimdFloat;

                let n = position.len();
                assert!(gradient.len() == n);

                let head_length = n - n % 4;

                let (pos, pos_tail) = position.split_at(head_length);
                let (grad, grad_tail) = gradient.split_at_mut(head_length);

                let mu_splat = f64x4::splat(mu);

                let mut logp = f64x4::splat(0f64);

                for (p, g) in pos.chunks_exact(4).zip(grad.chunks_exact_mut(4)) {
                    let p = f64x4::from_slice(p);
                    let val = mu_splat - p;
                    logp = val * val * f64x4::splat(0.5);
                    g.copy_from_slice(&val.to_array());
                }

                let mut logp_tail = 0f64;
                for (p, g) in pos_tail.iter().zip(grad_tail.iter_mut()).take(3) {
                    let val = mu - p;
                    logp_tail -= val * val / 2.;
                    *g = val;
                }

                logp.reduce_sum() + logp_tail
            }

            #[cfg(not(feature = "simd_support"))]
            #[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
            fn logp_inner(mu: f64, position: &[f64], gradient: &mut [f64]) -> f64 {
                let n = position.len();
                assert!(gradient.len() == n);

                let mut logp = 0f64;
                for (p, g) in position.iter().zip(gradient.iter_mut()) {
                    let val = mu - p;
                    logp -= val * val / 2.;
                    *g = val;
                }

                logp
            }

            let logp = logp_inner(self.mu, position, gradient);

            Ok(logp)
        }

        fn inv_transform_normalize(
            &mut self,
            _params: &Self::TransformParams,
            _untransformed_position: &[f64],
            _untransofrmed_gradient: &[f64],
            _transformed_position: &mut [f64],
            _transformed_gradient: &mut [f64],
        ) -> std::result::Result<f64, Self::LogpError> {
            unimplemented!()
        }

        fn init_from_untransformed_position(
            &mut self,
            _params: &Self::TransformParams,
            _untransformed_position: &[f64],
            _untransformed_gradient: &mut [f64],
            _transformed_position: &mut [f64],
            _transformed_gradient: &mut [f64],
        ) -> std::result::Result<(f64, f64), Self::LogpError> {
            unimplemented!()
        }

        fn init_from_transformed_position(
            &mut self,
            _params: &Self::TransformParams,
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
            _params: &'b mut Self::TransformParams,
        ) -> std::result::Result<(), Self::LogpError> {
            unimplemented!()
        }

        fn new_transformation<R: rand::Rng + ?Sized>(
            &mut self,
            _rng: &mut R,
            _untransformed_position: &[f64],
            _untransfogmed_gradient: &[f64],
            _chain: u64,
        ) -> std::result::Result<Self::TransformParams, Self::LogpError> {
            unimplemented!()
        }

        fn transformation_id(
            &self,
            _params: &Self::TransformParams,
        ) -> std::result::Result<i64, Self::LogpError> {
            unimplemented!()
        }
    }

    pub struct SimpleDrawStorage {
        draws: FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>,
    }

    impl SimpleDrawStorage {
        pub fn new(size: usize) -> Self {
            let items = PrimitiveBuilder::new();
            let draws = FixedSizeListBuilder::new(items, size as _);
            Self { draws }
        }
    }

    impl DrawStorage for SimpleDrawStorage {
        fn append_value(&mut self, point: &[f64]) -> Result<()> {
            self.draws.values().append_slice(point);
            self.draws.append(true);
            Ok(())
        }

        fn finalize(mut self) -> Result<Arc<dyn Array>> {
            Ok(ArrayBuilder::finish(&mut self.draws))
        }

        fn inspect(&self) -> Result<Arc<dyn Array>> {
            Ok(ArrayBuilder::finish_cloned(&self.draws))
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

    impl<F: Send + Sync + 'static> Model for CpuModel<F>
    where
        for<'a> &'a F: CpuLogpFunc,
    {
        type Math<'model> = CpuMath<&'model F>;

        type DrawStorage<'model, S: Settings> = SimpleDrawStorage;

        fn new_trace<'model, S: Settings, R: rand::prelude::Rng + ?Sized>(
            &'model self,
            _rng: &mut R,
            _chain_id: u64,
            _settings: &'model S,
        ) -> Result<Self::DrawStorage<'model, S>> {
            Ok(SimpleDrawStorage::new((&self.logp).dim()))
        }

        fn math(&self) -> Result<Self::Math<'_>> {
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
    use std::time::{Duration, Instant};

    use super::test_logps::NormalLogp;
    use crate::{
        cpu_math::CpuMath,
        sample_sequentially,
        sampler::{test_logps::CpuModel, Settings},
        Chain, DiagGradNutsSettings, Sampler,
    };

    use anyhow::Result;
    use itertools::Itertools;
    use pretty_assertions::assert_eq;
    use rand::{rngs::StdRng, SeedableRng};

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
        let mut sampler = Sampler::new(model, settings, 4, None)?;
        sampler.pause()?;
        sampler.pause()?;
        let _trace = sampler.inspect_trace()?;
        sampler.resume()?;
        let (ok, trace) = sampler.abort();
        ok?;
        assert!(trace.expect("No trace").chains.len() <= settings.num_chains);

        let model = CpuModel::new(logp.clone());
        let mut sampler = Sampler::new(model, settings, 4, None)?;
        sampler.pause()?;
        sampler.abort().0?;

        let model = CpuModel::new(logp.clone());
        let start = Instant::now();
        let sampler = Sampler::new(model, settings, 4, None)?;

        let mut sampler = match sampler.wait_timeout(Duration::from_nanos(100)) {
            super::SamplerWaitResult::Trace(trace) => {
                dbg!(start.elapsed());
                assert!(trace.chains.len() == settings.num_chains);
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
            super::SamplerWaitResult::Trace(trace) => {
                dbg!(start.elapsed());
                assert!(trace.chains.len() == settings.num_chains);
                trace.chains.iter().for_each(|chain| {
                    assert!(chain.draws.len() as u64 == settings.num_tune + settings.num_draws);
                });
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
