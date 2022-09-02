use thiserror::Error;

use std::{fmt::Debug, marker::PhantomData};

use crate::math::logaddexp;

#[derive(Error, Debug)]
pub enum NutsError {
    #[error("Logp function returned error: {0}")]
    LogpFailure(Box<dyn std::error::Error + Send>),
}

pub type Result<T> = std::result::Result<T, NutsError>;

// TODO This should be an Box<enum> instead of a trait object?
/// Details about a divergence that might have occured during sampling
///
/// There are two reasons why we might observe a divergence:
/// - The integration error of the Hamiltonian is larger than
///   a cutoff value or nan.
/// - The logp function caused a recoverable error (eg if an ODE solver
///   failed)
pub trait DivergenceInfo: AsSampleStatVec + std::fmt::Debug + Send {
    /// The position in parameter space where the diverging leapfrog started
    fn start_location(&self) -> Option<&[f64]>;

    /// The position in parameter space where the diverging leapfrog ended
    fn end_location(&self) -> Option<&[f64]>;

    /// The difference between the energy at the initial location of the trajectory and
    /// the energy at the end of the diverging leapfrog step.
    ///
    /// This is not available if the divergence was caused by a logp function error
    fn energy_error(&self) -> Option<f64>;

    /// The index of the end location of the diverging leapfrog.
    fn end_idx_in_trajectory(&self) -> Option<i64>;

    /// The index of the start location of the diverging leapfrog.
    fn start_idx_in_trajectory(&self) -> Option<i64>;

    /// Return the logp function error that caused the divergence if there was any
    ///
    /// This is not available if the divergence was cause because of a large energy
    /// difference.
    fn logp_function_error(&self) -> Option<&dyn std::error::Error>;
}

#[derive(Debug, Copy, Clone)]
pub enum Direction {
    Forward,
    Backward,
}

impl rand::distributions::Distribution<Direction> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Direction {
        if rng.gen::<bool>() {
            Direction::Forward
        } else {
            Direction::Backward
        }
    }
}

/// Callbacks for various events during a Nuts sampling step.
///
/// Collectors can compute statistics like the mean acceptance rate
/// or collect data for mass matrix adaptation.
pub trait Collector {
    type State: State;

    fn register_leapfrog(
        &mut self,
        _start: &Self::State,
        _end: &Self::State,
        _divergence_info: Option<&dyn DivergenceInfo>,
    ) {
    }
    fn register_draw(&mut self, _state: &Self::State, _info: &SampleInfo) {}
    fn register_init(&mut self, _state: &Self::State, _options: &NutsOptions) {}
}

/// Errors that happen when we evaluate the logp and gradient function
pub trait LogpError: std::error::Error {
    /// Unrecoverable errors during logp computation stop sampling,
    /// recoverable errors are seen as divergences.
    fn is_recoverable(&self) -> bool;
}

/// The hamiltonian defined by the potential energy and the kinetic energy
pub trait Hamiltonian {
    /// The type that stores a point in phase space
    type State: State;
    /// More detailed information about divergences
    type DivergenceInfo: DivergenceInfo + 'static;
    /// Errors that happen during logp evaluation
    type LogpError: LogpError + Send;
    /// Statistics that should be exported to the trace as part of the sampler stats
    type Stats: Send + AsSampleStatVec;

    /// Perform one leapfrog step.
    ///
    /// Return either an unrecoverable error, a new state or a divergence.
    fn leapfrog<C: Collector<State = Self::State>>(
        &mut self,
        pool: &mut <Self::State as State>::Pool,
        start: &Self::State,
        dir: Direction,
        initial_energy: f64,
        collector: &mut C,
    ) -> Result<std::result::Result<Self::State, Self::DivergenceInfo>>;

    /// Initialize a state at a new location.
    ///
    /// The momentum should be initialized to some arbitrary invalid number,
    /// it will later be set using Self::randomize_momentum.
    fn init_state(
        &mut self,
        pool: &mut <Self::State as State>::Pool,
        init: &[f64],
    ) -> Result<Self::State>;

    /// Randomize the momentum part of a state
    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut Self::State, rng: &mut R);

    /// Return sampler statistics defined in Self::Stats
    fn current_stats(&self) -> Self::Stats;

    fn new_empty_state(&mut self, pool: &mut <Self::State as State>::Pool) -> Self::State;

    /// Crate a new state pool that can be used to crate new states.
    fn new_pool(&mut self, capacity: usize) -> <Self::State as State>::Pool;

    /// The dimension of the hamiltonian (position only).
    fn dim(&self) -> usize;
}

/// A point in phase space
///
/// This also needs to store the sum of momentum terms
/// from the initial point of the trajectory to this point,
/// so that it can compute the termination criterion in
/// `is_turming`.
pub trait State: Clone + Debug {
    /// The state pool can be used to crate new states
    type Pool;

    /// Write the position stored in the state to a different location
    fn write_position(&self, out: &mut [f64]);

    /// Write the gradient stored in the state to a different location
    fn write_gradient(&self, out: &mut [f64]);

    /// Compute the termination criterion for NUTS
    fn is_turning(&self, other: &Self) -> bool;

    /// The total energy (potential + kinetic)
    fn energy(&self) -> f64;
    fn potential_energy(&self) -> f64;
    fn index_in_trajectory(&self) -> i64;

    /// Initialize the point to be the first in the trajectory.
    ///
    /// Set index_in_trajectory to 0 and reinitialize the sum of
    /// the momentum terms.
    fn make_init_point(&mut self);

    fn log_acceptance_probability(&self, initial_energy: f64) -> f64 {
        (initial_energy - self.energy()).min(0.)
    }
}

/// Information about a draw, exported as part of the sampler stats
#[derive(Debug)]
pub struct SampleInfo {
    /// The depth of the trajectory that this point was sampled from
    pub depth: u64,

    /// More detailed information about a divergence that might have
    /// occured in the trajectory.
    pub divergence_info: Option<Box<dyn DivergenceInfo>>,

    /// Whether the trajectory was terminated because it reached
    /// the maximum tree depth.
    pub reached_maxdepth: bool,
}

/// A part of the trajectory tree during NUTS sampling.
struct NutsTree<P: Hamiltonian, C: Collector<State = P::State>> {
    /// The left position of the tree.
    ///
    /// The left side always has the smaller index_in_trajectory.
    /// Leapfrogs in backward direction will replace the left.
    left: P::State,
    right: P::State,

    /// A draw from the trajectory between left and right using
    /// multinomial sampling.
    draw: P::State,
    log_size: f64,
    depth: u64,
    initial_energy: f64,

    /// A tree is the main tree if it contains the initial point
    /// of the trajectory.
    is_main: bool,
    collector: PhantomData<C>,
}

enum ExtendResult<P: Hamiltonian, C: Collector<State = P::State>> {
    /// The tree extension succeeded properly, and the termination
    /// criterion was not reached.
    Ok(NutsTree<P, C>),
    /// An unrecoverable error happend during a leapfrog step
    Err(NutsError),
    /// Tree extension succeeded and the termination criterion
    /// was reached.
    Turning(NutsTree<P, C>),
    /// A divergence happend during tree extension.
    Diverging(NutsTree<P, C>, P::DivergenceInfo),
}

impl<P: Hamiltonian, C: Collector<State = P::State>> NutsTree<P, C> {
    fn new(state: P::State) -> NutsTree<P, C> {
        let initial_energy = state.energy();
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: 0.,
            initial_energy,
            is_main: true,
            collector: PhantomData,
        }
    }

    #[inline]
    fn extend<R>(
        mut self,
        pool: &mut <P::State as State>::Pool,
        rng: &mut R,
        potential: &mut P,
        direction: Direction,
        options: &NutsOptions,
        collector: &mut C,
    ) -> ExtendResult<P, C>
    where
        P: Hamiltonian,
        R: rand::Rng + ?Sized,
    {
        let mut other = match self.single_step(pool, potential, direction, collector) {
            Ok(Ok(tree)) => tree,
            Ok(Err(info)) => return ExtendResult::Diverging(self, info),
            Err(err) => return ExtendResult::Err(err),
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(pool, rng, potential, direction, options, collector) {
                Ok(tree) => tree,
                Turning(_) => {
                    return Turning(self);
                }
                Diverging(_, info) => {
                    return Diverging(self, info);
                }
                Err(error) => {
                    return Err(error);
                }
            };
        }

        let (first, last) = match direction {
            Direction::Forward => (&self.left, &other.right),
            Direction::Backward => (&other.left, &self.right),
        };

        let mut turning = first.is_turning(last);
        if self.depth > 0 {
            if !turning {
                turning = self.right.is_turning(&other.right);
            }
            if !turning {
                turning = self.left.is_turning(&other.left);
            }
        }

        self.merge_into(other, rng, direction);

        if turning {
            ExtendResult::Turning(self)
        } else {
            ExtendResult::Ok(self)
        }
    }

    fn merge_into<R: rand::Rng + ?Sized>(
        &mut self,
        other: NutsTree<P, C>,
        rng: &mut R,
        direction: Direction,
    ) {
        assert!(self.depth == other.depth);
        assert!(self.left.index_in_trajectory() <= self.right.index_in_trajectory());
        match direction {
            Direction::Forward => {
                self.right = other.right;
            }
            Direction::Backward => {
                self.left = other.left;
            }
        }
        let log_size = logaddexp(self.log_size, other.log_size);

        let self_log_size = if self.is_main {
            assert!(self.left.index_in_trajectory() <= 0);
            assert!(self.right.index_in_trajectory() >= 0);
            self.log_size
        } else {
            log_size
        };

        if other.log_size >= self_log_size {
            self.draw = other.draw;
        } else if rng.gen_bool((other.log_size - self_log_size).exp()) {
            self.draw = other.draw;
        }

        self.depth += 1;
        self.log_size = log_size;
    }

    fn single_step(
        &self,
        pool: &mut <P::State as State>::Pool,
        potential: &mut P,
        direction: Direction,
        collector: &mut C,
    ) -> Result<std::result::Result<NutsTree<P, C>, P::DivergenceInfo>> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let end = match potential.leapfrog(pool, start, direction, self.initial_energy, collector) {
            Ok(Ok(end)) => end,
            Ok(Err(info)) => return Ok(Err(info)),
            Err(error) => return Err(error),
        };

        let log_size = self.initial_energy - end.energy();
        Ok(Ok(NutsTree {
            right: end.clone(),
            left: end.clone(),
            draw: end,
            depth: 0,
            log_size,
            initial_energy: self.initial_energy,
            is_main: false,
            collector: PhantomData,
        }))
    }

    fn info(&self, maxdepth: bool, divergence_info: Option<P::DivergenceInfo>) -> SampleInfo {
        let info: Option<Box<dyn DivergenceInfo>> = match divergence_info {
            Some(info) => Some(Box::new(info)),
            None => None,
        };
        SampleInfo {
            depth: self.depth,
            divergence_info: info,
            reached_maxdepth: maxdepth,
        }
    }
}

pub struct NutsOptions {
    pub maxdepth: u64,
    pub store_gradient: bool,
}

pub(crate) fn draw<P, R, C>(
    pool: &mut <P::State as State>::Pool,
    init: &mut P::State,
    rng: &mut R,
    potential: &mut P,
    options: &NutsOptions,
    collector: &mut C,
) -> Result<(P::State, SampleInfo)>
where
    P: Hamiltonian,
    R: rand::Rng + ?Sized,
    C: Collector<State = P::State>,
{
    potential.randomize_momentum(init, rng);
    init.make_init_point();
    collector.register_init(init, options);

    let mut tree = NutsTree::new(init.clone());
    while tree.depth < options.maxdepth {
        let direction: Direction = rng.gen();
        tree = match tree.extend(pool, rng, potential, direction, options, collector) {
            ExtendResult::Ok(tree) => tree,
            ExtendResult::Turning(tree) => {
                let info = tree.info(false, None);
                collector.register_draw(&tree.draw, &info);
                return Ok((tree.draw, info));
            }
            ExtendResult::Diverging(tree, info) => {
                let info = tree.info(false, Some(info));
                collector.register_draw(&tree.draw, &info);
                return Ok((tree.draw, info));
            }
            ExtendResult::Err(error) => {
                return Err(error);
            }
        };
    }
    let info = tree.info(true, None);
    Ok((tree.draw, info))
}

#[derive(Debug)]
pub(crate) struct NutsSampleStats<HStats: Send + Debug, AdaptStats: Send + Debug> {
    pub depth: u64,
    pub maxdepth_reached: bool,
    pub idx_in_trajectory: i64,
    pub logp: f64,
    pub energy: f64,
    pub divergence_info: Option<Box<dyn DivergenceInfo>>,
    pub chain: u64,
    pub draw: u64,
    pub gradient: Option<Box<[f64]>>,
    pub potential_stats: HStats,
    pub strategy_stats: AdaptStats,
}

#[derive(Debug, Clone)]
pub enum SampleStatValue {
    Array(Box<[f64]>),
    OptionArray(Option<Box<[f64]>>),
    U64(u64),
    I64(i64),
    OptionI64(Option<i64>),
    F64(f64),
    OptionF64(Option<f64>),
    Bool(bool),
    String(String),
}

impl From<Box<[f64]>> for SampleStatValue {
    fn from(val: Box<[f64]>) -> Self {
        SampleStatValue::Array(val)
    }
}

impl From<Option<Box<[f64]>>> for SampleStatValue {
    fn from(val: Option<Box<[f64]>>) -> Self {
        SampleStatValue::OptionArray(val)
    }
}

impl From<u64> for SampleStatValue {
    fn from(val: u64) -> Self {
        SampleStatValue::U64(val)
    }
}

impl From<i64> for SampleStatValue {
    fn from(val: i64) -> Self {
        SampleStatValue::I64(val)
    }
}

impl From<Option<i64>> for SampleStatValue {
    fn from(val: Option<i64>) -> Self {
        SampleStatValue::OptionI64(val)
    }
}

impl From<f64> for SampleStatValue {
    fn from(val: f64) -> Self {
        SampleStatValue::F64(val)
    }
}

impl From<Option<f64>> for SampleStatValue {
    fn from(val: Option<f64>) -> Self {
        SampleStatValue::OptionF64(val)
    }
}

impl From<bool> for SampleStatValue {
    fn from(val: bool) -> Self {
        SampleStatValue::Bool(val)
    }
}

impl From<String> for SampleStatValue {
    fn from(val: String) -> Self {
        SampleStatValue::String(val)
    }
}

pub trait AsSampleStatVec: Debug {
    fn add_to_vec(&self, vec: &mut Vec<SampleStatItem>);
}

pub type SampleStatItem = (&'static str, SampleStatValue);

/// Diagnostic information about draws and the state of the sampler for each draw
pub trait SampleStats: Send + Debug {
    /// The depth of the NUTS tree that the draw was sampled from
    fn depth(&self) -> u64;
    /// Whether the trajectory was stopped because the maximum size
    /// was reached.
    fn maxdepth_reached(&self) -> bool;
    /// The index of the accepted sample in the trajectory
    fn index_in_trajectory(&self) -> i64;
    /// The unnormalized posterior density at the draw
    fn logp(&self) -> f64;
    /// The value of the hamiltonian of the draw
    fn energy(&self) -> f64;
    /// More detailed information if the draw came from a diverging trajectory.
    fn divergence_info(&self) -> Option<&dyn DivergenceInfo>;
    /// An ID for the chain that the sample produce the draw.
    fn chain(&self) -> u64;
    /// The draw number
    fn draw(&self) -> u64;
    /// The logp gradient at the location of the draw. This is only stored
    /// if NutsOptions.store_gradient is `true`.
    fn gradient(&self) -> Option<&[f64]>;
    /// Export the sample statisitcs to a vector. This might include some additional
    /// diagnostics coming from the step size and matrix adaptation strategies.
    fn to_vec(&self) -> Vec<SampleStatItem>;
}

impl<H, A> SampleStats for NutsSampleStats<H, A>
where
    H: Send + Debug + AsSampleStatVec,
    A: Send + Debug + AsSampleStatVec,
{
    fn depth(&self) -> u64 {
        self.depth
    }
    fn maxdepth_reached(&self) -> bool {
        self.maxdepth_reached
    }
    fn index_in_trajectory(&self) -> i64 {
        self.idx_in_trajectory
    }
    fn logp(&self) -> f64 {
        self.logp
    }
    fn energy(&self) -> f64 {
        self.energy
    }
    fn divergence_info(&self) -> Option<&dyn DivergenceInfo> {
        self.divergence_info.as_ref().map(|x| x.as_ref())
    }
    fn chain(&self) -> u64 {
        self.chain
    }
    fn draw(&self) -> u64 {
        self.draw
    }
    fn gradient(&self) -> Option<&[f64]> {
        self.gradient.as_ref().map(|x| &x[..])
    }
    fn to_vec(&self) -> Vec<SampleStatItem> {
        let mut vec = Vec::with_capacity(20);
        vec.push(("depth", self.depth.into()));
        vec.push(("maxdepth_reached", self.maxdepth_reached.into()));
        vec.push(("index_in_trajectory", self.idx_in_trajectory.into()));
        vec.push(("logp", self.logp.into()));
        vec.push(("energy", self.energy.into()));
        vec.push(("diverging", self.divergence_info.is_some().into()));
        self.potential_stats.add_to_vec(&mut vec);
        self.strategy_stats.add_to_vec(&mut vec);
        if let Some(info) = self.divergence_info() {
            info.add_to_vec(&mut vec);
        }
        if let Some(grad) = self.gradient() {
            vec.push(("gradient", grad.to_vec().into_boxed_slice().into()));
        } else {
            vec.push(("gradient", SampleStatValue::OptionArray(None)));
        }
        vec
    }
}

/// Draw samples from the posterior distribution using Hamiltonian MCMC.
pub trait Chain {
    type Hamiltonian: Hamiltonian;
    type AdaptStrategy: AdaptStrategy;
    type Stats: SampleStats;

    /// Initialize the sampler to a position. This should be called
    /// before calling draw.
    ///
    /// This fails if the logp function returns an error.
    fn set_position(&mut self, position: &[f64]) -> Result<()>;

    /// Draw a new sample and return the position and some diagnosic information.
    fn draw(&mut self) -> Result<(Box<[f64]>, Self::Stats)>;

    /// The dimensionality of the posterior.
    fn dim(&self) -> usize;
}

pub(crate) struct NutsChain<P, R, S>
where
    P: Hamiltonian,
    R: rand::Rng,
    S: AdaptStrategy<Potential = P>,
{
    pool: <P::State as State>::Pool,
    potential: P,
    collector: S::Collector,
    options: NutsOptions,
    rng: R,
    init: P::State,
    chain: u64,
    draw_count: u64,
    strategy: S,
}

impl<P, R, S> NutsChain<P, R, S>
where
    P: Hamiltonian,
    R: rand::Rng,
    S: AdaptStrategy<Potential = P>,
{
    pub fn new(mut potential: P, strategy: S, options: NutsOptions, rng: R, chain: u64) -> Self {
        let pool_size: usize = options.maxdepth.checked_mul(2).unwrap().try_into().unwrap();
        let mut pool = potential.new_pool(pool_size);
        let init = potential.new_empty_state(&mut pool);
        let collector = strategy.new_collector();
        NutsChain {
            pool,
            potential,
            collector,
            options,
            rng,
            init,
            chain,
            draw_count: 0,
            strategy,
        }
    }
}

pub trait AdaptStrategy {
    type Potential: Hamiltonian;
    type Collector: Collector<State = <Self::Potential as Hamiltonian>::State>;
    type Stats: Send + AsSampleStatVec;
    type Options: Copy + Send + Default;

    fn new(options: Self::Options, num_tune: u64, dim: usize) -> Self;

    fn init(
        &mut self,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &<Self::Potential as Hamiltonian>::State,
    );

    fn adapt(
        &mut self,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: u64,
        collector: &Self::Collector,
    );

    fn new_collector(&self) -> Self::Collector;

    fn current_stats(
        &self,
        options: &NutsOptions,
        potential: &Self::Potential,
        collector: &Self::Collector,
    ) -> Self::Stats;
}

impl<H, R, S> Chain for NutsChain<H, R, S>
where
    H: Hamiltonian,
    R: rand::Rng,
    S: AdaptStrategy<Potential = H>,
{
    type Hamiltonian = H;
    type AdaptStrategy = S;
    type Stats = NutsSampleStats<H::Stats, S::Stats>;

    fn set_position(&mut self, position: &[f64]) -> Result<()> {
        let state = self.potential.init_state(&mut self.pool, position)?;
        self.init = state;
        self.strategy
            .init(&mut self.options, &mut self.potential, &self.init);
        Ok(())
    }

    fn draw(&mut self) -> Result<(Box<[f64]>, Self::Stats)> {
        let (state, info) = draw(
            &mut self.pool,
            &mut self.init,
            &mut self.rng,
            &mut self.potential,
            &self.options,
            &mut self.collector,
        )?;
        let mut position: Box<[f64]> = vec![0f64; self.potential.dim()].into();
        state.write_position(&mut position);
        let stats = NutsSampleStats {
            depth: info.depth,
            maxdepth_reached: info.reached_maxdepth,
            idx_in_trajectory: state.index_in_trajectory(),
            logp: -state.potential_energy(),
            energy: state.energy(),
            divergence_info: info.divergence_info,
            chain: self.chain,
            draw: self.draw_count,
            potential_stats: self.potential.current_stats(),
            strategy_stats: self.strategy.current_stats(
                &self.options,
                &self.potential,
                &self.collector,
            ),
            gradient: if self.options.store_gradient {
                let mut gradient: Box<[f64]> = vec![0f64; self.potential.dim()].into();
                state.write_gradient(&mut gradient);
                Some(gradient)
            } else {
                None
            },
        };
        self.strategy.adapt(
            &mut self.options,
            &mut self.potential,
            self.draw_count,
            &self.collector,
        );
        self.init = state;
        self.draw_count += 1;
        Ok((position, stats))
    }

    fn dim(&self) -> usize {
        self.potential.dim()
    }
}
