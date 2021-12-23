use thiserror::Error;

use std::{collections::HashMap, marker::PhantomData};

use crate::math::logaddexp;

#[derive(Error, Debug)]
pub enum NutsError {
    #[error("Logp function returned unrecoverable error")]
    LogpFailure(Box<dyn std::error::Error + Send>),
}

pub type Result<T> = std::result::Result<T, NutsError>;

pub trait DivergenceInfo: std::fmt::Debug + Send {
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
    type Stats: Copy + Send + AsSampleStatMap;

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
pub trait State: Clone {
    /// The state pool can be used to crate new states
    type Pool;

    /// Write the position stored in the state to a different location
    fn write_position(&self, out: &mut [f64]);

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

    #[inline]
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

    #[inline]
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
pub(crate) struct NutsSampleStats<HStats: Send, AdaptStats: Send> {
    pub depth: u64,
    pub maxdepth_reached: bool,
    pub idx_in_trajectory: i64,
    pub logp: f64,
    pub energy: f64,
    pub divergence_info: Option<Box<dyn DivergenceInfo>>,
    pub chain: u64,
    pub draw: u64,
    pub potential_stats: HStats,
    pub strategy_stats: AdaptStats,
}

pub enum SampleStatValue {
    Array(Box<[f64]>),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
}

impl From<Box<[f64]>> for SampleStatValue {
    fn from(val: Box<[f64]>) -> Self {
        SampleStatValue::Array(val)
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

impl From<f64> for SampleStatValue {
    fn from(val: f64) -> Self {
        SampleStatValue::F64(val)
    }
}

impl From<bool> for SampleStatValue {
    fn from(val: bool) -> Self {
        SampleStatValue::Bool(val)
    }
}

pub trait AsSampleStatMap {
    fn as_map(&self) -> HashMap<&'static str, SampleStatValue>;
}

pub trait SampleStats: Send + AsSampleStatMap {
    fn depth(&self) -> u64;
    fn maxdepth_reached(&self) -> bool;
    fn index_in_trajectory(&self) -> i64;
    fn logp(&self) -> f64;
    fn energy(&self) -> f64;
    fn divergence_info(&self) -> Option<&dyn DivergenceInfo>;
    fn chain(&self) -> u64;
    fn draw(&self) -> u64;
}

impl<HStats, AdaptStats> AsSampleStatMap for NutsSampleStats<HStats, AdaptStats>
where
    HStats: Send + AsSampleStatMap,
    AdaptStats: Send + AsSampleStatMap,
{
    fn as_map(&self) -> HashMap<&'static str, SampleStatValue> {
        let mut map: HashMap<_, SampleStatValue> = HashMap::with_capacity(20);
        map.insert("depth", self.depth.into());
        map.insert("maxdepth_reached", self.maxdepth_reached.into());
        map.insert("index_in_trajectory", self.idx_in_trajectory.into());
        map.insert("logp", self.logp.into());
        map.insert("energy", self.energy.into());
        map.insert("diverging", self.divergence_info.is_some().into());
        map.extend(self.potential_stats.as_map());
        map.extend(self.strategy_stats.as_map());
        map
    }
}

impl<HStats, AdaptStats> SampleStats for NutsSampleStats<HStats, AdaptStats>
where
    HStats: Send + AsSampleStatMap,
    AdaptStats: Send + AsSampleStatMap,
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
}

pub trait Sampler {
    type Hamiltonian: Hamiltonian;
    type AdaptStrategy: AdaptStrategy;
    type Stats: SampleStats;

    fn set_position(&mut self, position: &[f64]) -> Result<()>;
    fn draw(&mut self) -> Result<(Box<[f64]>, Self::Stats)>;
    fn dim(&self) -> usize;
}

pub(crate) struct NutsSampler<P, R, S>
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

impl<P, R, S> NutsSampler<P, R, S>
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
        NutsSampler {
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
    type Stats: Copy + Send + AsSampleStatMap;
    type Options: Copy + Send + Default;

    fn new(options: Self::Options, num_tune: u64, dim: usize) -> Self;

    fn adapt(&mut self, potential: &mut Self::Potential, draw: u64, collector: &Self::Collector);

    fn new_collector(&self) -> Self::Collector;

    fn current_stats(&self, collector: &Self::Collector) -> Self::Stats;
}

impl<H, R, S> Sampler for NutsSampler<H, R, S>
where
    H: Hamiltonian,
    R: rand::Rng,
    S: AdaptStrategy<Potential = H>,
{
    type Hamiltonian = H;
    type AdaptStrategy = S;
    type Stats = NutsSampleStats<H::Stats, S::Stats>;

    fn set_position(&mut self, position: &[f64]) -> Result<()> {
        self.potential
            .init_state(&mut self.pool, position)
            .map(|_| ())
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
        let position: Box<[f64]> = vec![0f64; self.potential.dim()].into();
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
            strategy_stats: self.strategy.current_stats(&self.collector),
            /*  TODO
            step_size: self.options.step_size,
            step_size_bar: self.options.step_size,
            mean_acceptance_rate: self.collector.stats().mean_acceptance_rate,
            tree_size: self.collector.acceptance_rate.mean.count,
            first_diag_mass_matrix: self.potential.mass_matrix_mut().current.variance[0],
            */
        };
        self.strategy
            .adapt(&mut self.potential, self.draw_count, &self.collector);
        self.init = state;
        self.draw_count += 1;
        Ok((position, stats))
    }

    fn dim(&self) -> usize {
        self.potential.dim()
    }
}
