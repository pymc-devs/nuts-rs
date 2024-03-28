use anyhow::{bail, Context, Result};
use arrow2::array::Array;
use itertools::Itertools;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::{ScopeFifo, ThreadPoolBuilder};
use std::{
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
    adapt_strategy::{GradDiagOptions, GradDiagStrategy},
    mass_matrix::DiagMassMatrix,
    math_base::Math,
    nuts::{Chain, NutsChain, NutsOptions, SampleStats, SamplerStats, StatTraceBuilder},
    potential::EuclideanPotential,
};

pub trait Settings: private::Sealed + Clone + Copy + Default + Sync + Send + 'static {
    type Chain<M: Math>: Chain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        math: M,
        rng: &mut R,
    ) -> Self::Chain<M>;

    fn sample_stats<M: Math>(
        &self,
        stats: &<Self::Chain<M> as SamplerStats<M>>::Stats,
    ) -> SampleStats;
    fn hint_num_tune(&self) -> usize;
    fn hint_num_draws(&self) -> usize;
    fn num_chains(&self) -> usize;
    fn seed(&self) -> u64;
}

mod private {
    use crate::DiagGradNutsSettings;

    pub trait Sealed {}

    impl Sealed for DiagGradNutsSettings {}
}

/// Settings for the NUTS sampler
#[derive(Clone, Copy)]
pub struct DiagGradNutsSettings {
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
    pub mass_matrix_adapt: GradDiagOptions,
    pub check_turning: bool,

    pub num_chains: usize,
    pub seed: u64,
}

impl Default for DiagGradNutsSettings {
    fn default() -> Self {
        Self {
            num_tune: 300,
            num_draws: 1000,
            maxdepth: 10,
            max_energy_error: 1000f64,
            store_gradient: false,
            store_unconstrained: false,
            store_divergences: false,
            mass_matrix_adapt: GradDiagOptions::default(),
            check_turning: true,
            seed: 0,
            num_chains: 6,
        }
    }
}

type DiagGradNutsChain<M> =
    NutsChain<M, EuclideanPotential<M, DiagMassMatrix<M>>, SmallRng, GradDiagStrategy<M>>;

impl Settings for DiagGradNutsSettings {
    type Chain<M: Math> = DiagGradNutsChain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        rng: &mut R,
    ) -> Self::Chain<M> {
        use crate::nuts::AdaptStrategy;
        let num_tune = self.num_tune;
        let strategy = GradDiagStrategy::new(&mut math, self.mass_matrix_adapt, num_tune);
        let mass_matrix = DiagMassMatrix::new(
            &mut math,
            self.mass_matrix_adapt.mass_matrix_options.store_mass_matrix,
        );
        let max_energy_error = self.max_energy_error;
        let potential = EuclideanPotential::new(mass_matrix, max_energy_error, 1f64);

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

    fn sample_stats<M: Math>(
        &self,
        stats: &<Self::Chain<M> as SamplerStats<M>>::Stats,
    ) -> SampleStats {
        SampleStats {
            chain: stats.chain,
            draw: stats.draw,
            diverging: stats.divergence_info.is_some(),
            tuning: stats.tuning,
        }
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
}

pub fn sample_sequentially<'math, M: Math + 'math, R: Rng + ?Sized>(
    math: M,
    settings: DiagGradNutsSettings,
    start: &[f64],
    draws: u64,
    chain: u64,
    rng: &mut R,
) -> Result<impl Iterator<Item = Result<(Box<[f64]>, SampleStats)>> + 'math> {
    let mut sampler = settings.new_chain(chain, math, rng);
    sampler.set_position(start)?;
    Ok((0..draws).map(move |_| {
        sampler
            .draw()
            .map(|(point, info)| (point, settings.sample_stats::<M>(&info)))
            .map_err(|e| e.into())
    }))
}

pub trait DrawStorage: Send + Clone {
    fn append_value(&mut self, point: &[f64]) -> Result<()>;
    fn finalize(self) -> Result<Box<dyn Array>>;
    fn inspect(&mut self) -> Result<Box<dyn Array>>;
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
#[derive(Clone)]
pub struct ChainProgress {
    pub finished_draws: usize,
    pub total_draws: usize,
    pub divergences: usize,
    pub tuning: bool,
}

impl ChainProgress {
    fn new(total: usize) -> Self {
        Self {
            finished_draws: 0,
            total_draws: total,
            divergences: 0,
            tuning: true,
        }
    }

    fn update(&mut self, stats: &SampleStats) {
        self.finished_draws += 1;
        if stats.diverging {
            self.divergences += 1;
        }
        self.tuning = stats.tuning;
    }
}

struct ChainOutput {
    draws: Box<dyn Array>,
    stats: Option<Box<dyn Array>>,
    chain_id: u64,
}

enum ChainCommand {
    Resume,
    Pause,
}

type Builder<'model, M, S> = <<S as Settings>::Chain<<M as Model>::Math<'model>> as SamplerStats<
    <M as Model>::Math<'model>,
>>::Builder;

struct ChainTrace<'model, M: Model + 'model, S: Settings>
//where S::Chain<M>: SamplerStats<M::Math<'model>>
{
    draws_builder: M::DrawStorage<'model, S>,
    stats_builder: Builder<'model, M, S>,
    progress: ChainProgress,
    chain_id: u64,
}

impl<'model, M: Model + 'model, S: Settings> Clone for ChainTrace<'model, M, S> {
    fn clone(&self) -> Self {
        Self {
            draws_builder: self.draws_builder.clone(),
            stats_builder: self.stats_builder.clone(),
            progress: self.progress.clone(),
            chain_id: self.chain_id,
        }
    }
}

impl<'model, M: Model + 'model, S: Settings> ChainTrace<'model, M, S>
//where S::Chain<M>: SamplerStats<M::Math<'model>>
{
    fn inspect(&self) -> Result<ChainOutput> {
        self.clone().finalize()
    }

    fn finalize(self) -> Result<ChainOutput> {
        let draws = self.draws_builder.finalize()?;
        let stats = self.stats_builder.finalize();
        Ok(ChainOutput {
            chain_id: self.chain_id,
            draws,
            stats: stats.map(|x| x.boxed()),
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
}

impl<'scope, M: Model + 'scope, S: Settings> ChainProcess<'scope, M, S> {
    fn abort_many(chains: Vec<Self>) {
        for chain in chains.into_iter() {
            drop(chain.stop_marker);
        }
    }

    fn progress(&self) -> Option<ChainProgress> {
        self.trace
            .lock()
            .expect("Poisoned lock")
            .as_ref()
            .map(|val| val.progress.clone())
    }

    fn current_trace(&self) -> Option<Result<ChainOutput>> {
        self.trace
            .lock()
            .expect("Poisoned lock")
            .as_ref()
            .map(|val| val.inspect())
    }

    fn resume(&self) -> Result<()> {
        self.stop_marker.send(ChainCommand::Resume)?;
        Ok(())
    }

    fn pause(&self) -> Result<()> {
        self.stop_marker.send(ChainCommand::Pause)?;
        Ok(())
    }

    fn start<'model>(
        model: &'model M,
        chain_id: u64,
        seed: u64,
        settings: &'model S,
        scope: &ScopeFifo<'scope>,
        results: Sender<Result<ChainOutput>>,
    ) -> Result<Self>
    where
        'model: 'scope,
    {
        let (stop_marker_tx, stop_marker_rx) = channel();

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(chain_id);

        let trace = Arc::new(Mutex::new(None));

        let trace_inner = trace.clone();

        scope.spawn_fifo(move |_| {
            let trace = trace_inner;

            let mut sample = move || {
                let logp = model.math().context("Failed to create model density")?;
                let dim = logp.dim();

                let mut sampler = settings.new_chain(chain_id, logp, &mut rng);

                let draw_trace = model
                    .new_trace(&mut rng, chain_id, settings)
                    .context("Failed to create trace object")?;
                let progress = ChainProgress::new(settings.hint_num_draws());
                let stats_trace = sampler.new_builder(settings, settings.hint_num_draws());

                let new_trace = ChainTrace {
                    draws_builder: draw_trace,
                    stats_builder: stats_trace,
                    progress,
                    chain_id,
                };
                *trace.lock().expect("Poisoned mutex") = Some(new_trace);

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
                    let error: anyhow::Error = error.into();
                    return Err(error.context("All initialization points failed"));
                }

                // TODO let the sampler decide
                let draws = settings.hint_num_tune() + settings.hint_num_draws();

                let mut msg = stop_marker_rx.try_recv();
                let mut draw = 0;
                loop {
                    match msg {
                        Err(TryRecvError::Disconnected) => {
                            return trace
                                .lock()
                                .expect("Could not unlock trace lock. Poisoned mutex")
                                .take()
                                .expect("Trace was empty")
                                .finalize()
                        }
                        Ok(ChainCommand::Pause) => {
                            msg = stop_marker_rx.recv().map_err(|e| e.into());
                            continue;
                        }
                        Ok(ChainCommand::Resume) => {}
                        // The remote end is dead
                        Err(TryRecvError::Empty) => break,
                    }

                    let (point, info) = sampler.draw().unwrap();
                    let mut guard = trace
                        .lock()
                        .expect("Could not unlock trace lock. Poisoned mutex");

                    let Some(val) = guard.as_mut() else {
                        panic!("Trace was empty")
                    };
                    val.progress.update(&settings.sample_stats(&info));
                    DrawStorage::append_value(&mut val.draws_builder, &point)?;
                    StatTraceBuilder::append_value(&mut val.stats_builder, info);
                    draw += 1;
                    if draw == draws {
                        break;
                    }

                    msg = stop_marker_rx.try_recv();
                }

                trace
                    .lock()
                    .expect("Could not unlock trace lock. Poisoned mutex")
                    .take()
                    .expect("Trace was empty")
                    .finalize()
            };

            results
                .send(sample())
                .expect("Could not send sampling results to main thread.");
        });

        Ok(Self {
            trace,
            stop_marker: stop_marker_tx,
        })
    }
}

enum SamplerCommand {
    Pause,
    Continue,
    InspectTrace,
    Progress,
}

enum SamplerResponse {
    Ok(),
    IntermediateTrace(Trace),
    Progress(Box<[Option<ChainProgress>]>),
}

pub enum SamplerWaitResult {
    Trace(Trace),
    Timeout(Sampler),
    Err((anyhow::Error, Sampler)),
}

pub struct Sampler {
    main_thread: JoinHandle<Result<()>>,
    commands: SyncSender<SamplerCommand>,
    responses: Receiver<SamplerResponse>,
    results: Receiver<Result<ChainOutput>>,
    finished: Vec<ChainOutput>,
}

pub struct Trace {
    pub chains: Vec<Option<(u64, Box<dyn Array>, Option<Box<dyn Array>>)>>,
}

impl<I: Iterator<Item = Option<ChainOutput>>> From<I> for Trace {
    fn from(value: I) -> Self {
        Trace {
            chains: value
                .map(|out| out.map(|out| (out.chain_id, out.draws, out.stats)))
                .collect_vec(),
        }
    }
}

impl Sampler {
    pub fn new<S: Settings, M: Model>(model: M, settings: S, num_cores: usize) -> Result<Self> {
        let (commands_tx, commands_rx) = sync_channel(0);
        let (responses_tx, responses_rx) = sync_channel(0);
        let (results_tx, results_rx) = channel();

        let main_thread = spawn(move || {
            let pool = ThreadPoolBuilder::new()
                .num_threads(num_cores)
                .thread_name(|i| format!("nutpie-worker-{}", i))
                .build()?;

            let settings_ref = &settings;
            let model_ref = &model;

            pool.scope_fifo(move |scope| {
                let results = results_tx.clone();
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

                let (chains, errors): (Vec<_>, Vec<_>) = chains.into_iter().partition_result();
                if let Some(error) = errors.into_iter().next() {
                    ChainProcess::abort_many(chains);
                    return Err(error);
                }

                let main_loop = || {
                    let mut command = SamplerCommand::Continue;
                    loop {
                        match command {
                            SamplerCommand::Pause => {
                                for chain in chains.iter() {
                                    chain.pause()?;
                                }
                                responses_tx.send(SamplerResponse::Ok())?;
                            }
                            SamplerCommand::Continue => {
                                for chain in chains.iter() {
                                    chain.resume()?;
                                }
                                responses_tx.send(SamplerResponse::Ok())?;
                            }
                            SamplerCommand::InspectTrace => {
                                let traces: Result<Vec<_>> = chains
                                    .iter()
                                    .map(|chain| chain.current_trace().transpose())
                                    .collect();
                                responses_tx.send(SamplerResponse::IntermediateTrace(
                                    traces?.into_iter().into(),
                                ))?;
                            }
                            SamplerCommand::Progress => {
                                let progress =
                                    chains.iter().map(|chain| chain.progress()).collect_vec();
                                responses_tx.send(SamplerResponse::Progress(progress.into()))?;
                            }
                        };

                        let Ok(next_command) = commands_rx.recv() else {
                            return Ok(());
                        };

                        command = next_command;
                    }
                };
                let result: Result<()> = main_loop();
                ChainProcess::abort_many(chains);
                result?;
                Ok(())
            })
        });

        Ok(Self {
            main_thread,
            commands: commands_tx,
            responses: responses_rx,
            results: results_rx,
            finished: Vec::new(),
        })
    }

    pub fn pause(&mut self) -> Result<()> {
        self.commands.send(SamplerCommand::Pause)?;
        Ok(())
    }

    pub fn resume(&mut self) -> Result<()> {
        self.commands.send(SamplerCommand::Continue)?;
        Ok(())
    }

    pub fn abort(self) -> Result<()> {
        drop(self.commands);
        let result = self.main_thread.join();
        if let Err(payload) = result {
            std::panic::resume_unwind(payload)
        }
        Ok(())
    }

    pub fn inspect_trace(&mut self) -> Result<Trace> {
        self.commands.send(SamplerCommand::InspectTrace)?;
        let response = self.responses.recv()?;
        let SamplerResponse::IntermediateTrace(trace) = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(trace)
    }

    pub fn wait_timeout(mut self, timeout: Duration) -> SamplerWaitResult {
        let start = Instant::now();
        let mut remaining = Some(timeout);
        while remaining.is_some() {
            match self.results.recv_timeout(timeout) {
                Ok(Ok(trace)) => {
                    self.finished.push(trace);
                    remaining = timeout.checked_sub(start.elapsed())
                }
                Ok(Err(e)) => return SamplerWaitResult::Err((e, self)),
                Err(RecvTimeoutError::Disconnected) => {
                    let res = self.main_thread.join();
                    if let Err(payload) = res {
                        std::panic::resume_unwind(payload)
                    };
                    return SamplerWaitResult::Trace(
                        self.finished.into_iter().map(Some).into(),
                    );
                }
                Err(RecvTimeoutError::Timeout) => break,
            }
        }
        SamplerWaitResult::Timeout(self)
    }

    pub fn progress(&mut self) -> Result<Box<[Option<ChainProgress>]>> {
        self.commands.send(SamplerCommand::Progress)?;
        let response = self.responses.recv()?;
        let SamplerResponse::Progress(progress) = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(progress)
    }
}

pub mod test_logps {
    use crate::{
        cpu_math::{CpuLogpFunc, CpuMath},
        nuts::LogpError,
    };
    use multiversion::multiversion;
    use thiserror::Error;

    use super::{DrawStorage, Model};

    #[derive(Clone, Debug)]
    pub struct NormalLogp {
        pub dim: usize,
        pub mu: f64,
    }

    #[derive(Clone)]
    pub struct Storage {}

    impl DrawStorage for Storage {
        fn append_value(&mut self, _point: &[f64]) -> anyhow::Result<()> {
            Ok(())
        }

        fn finalize(self) -> anyhow::Result<Box<dyn arrow2::array::Array>> {
            todo!()
        }

        fn inspect(&mut self) -> anyhow::Result<Box<dyn arrow2::array::Array>> {
            self.clone().finalize()
        }
    }

    impl Model for NormalLogp {
        type Math<'math> = CpuMath<NormalLogp>;

        type DrawStorage<'model, S: super::Settings> = Storage;

        fn new_trace<'model, S: super::Settings, R: rand::prelude::Rng + ?Sized>(
            &'model self,
            _rng: &mut R,
            _chain_id: u64,
            _settings: &'model S,
        ) -> anyhow::Result<Self::DrawStorage<'model, S>> {
            Ok(Storage {})
        }

        fn math(&self) -> anyhow::Result<Self::Math<'_>> {
            Ok(CpuMath::new(self.clone()))
        }

        fn init_position<R: rand::prelude::Rng + ?Sized>(
            &self,
            _rng: &mut R,
            position: &mut [f64],
        ) -> anyhow::Result<()> {
            position.iter_mut().for_each(|x| *x = 0.);
            Ok(())
        }
    }

    #[derive(Error, Debug)]
    pub enum NormalLogpError {}
    impl LogpError for NormalLogpError {
        fn is_recoverable(&self) -> bool {
            false
        }
    }

    impl CpuLogpFunc for NormalLogp {
        type LogpError = NormalLogpError;

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
                use std::simd::SimdFloat;

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
                    logp = logp - val * val * f64x4::splat(0.5);
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
    }
}

#[cfg(test)]
mod tests {
    use super::test_logps::NormalLogp;
    use crate::{
        cpu_math::CpuMath, sample_sequentially, sampler::Settings, Chain, DiagGradNutsSettings,
    };

    use anyhow::Result;
    use itertools::Itertools;
    use pretty_assertions::assert_eq;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn sample_chain() -> Result<()> {
        let logp = NormalLogp { dim: 10, mu: 0.1 };
        let math = CpuMath::new(logp);
        let mut settings = DiagGradNutsSettings::default();
        settings.num_tune = 100;
        settings.num_draws = 100;
        let start = vec![0.2; 10];

        let mut rng = StdRng::seed_from_u64(42);

        let mut chain = settings.new_chain(0, math, &mut rng);

        let (_draw, info) = chain.draw()?;
        assert!(settings.sample_stats::<CpuMath<NormalLogp>>(&info).tuning);
        assert_eq!(info.draw, 0);

        let logp = NormalLogp { dim: 10, mu: 0.1 };
        let math = CpuMath::new(logp);
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
    fn sample_seq() {
        let logp = NormalLogp { dim: 10, mu: 0.1 };
        let math = CpuMath::new(logp);
        let mut settings = DiagGradNutsSettings::default();
        settings.num_tune = 100;
        settings.num_draws = 100;
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
