use arrow2::array::{MutableFixedSizeListArray, MutableUtf8Array, TryPush};
use arrow2::{
    array::{MutableArray, MutableBooleanArray, MutablePrimitiveArray, StructArray},
    datatypes::{DataType, Field},
};
use thiserror::Error;

use std::sync::Arc;
use std::{fmt::Debug, marker::PhantomData};

use crate::math::logaddexp;
use crate::sampler::Settings;
use crate::state::{State, StatePool};

use crate::math_base::Math;

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum NutsError {
    #[error("Logp function returned error: {0}")]
    LogpFailure(Box<dyn std::error::Error + Send + Sync>),

    #[error("Could not serialize sample stats")]
    SerializeFailure(),

    #[error("Could not initialize state because of bad initial gradient.")]
    BadInitGrad(),
}

pub type Result<T> = std::result::Result<T, NutsError>;

/// Details about a divergence that might have occured during sampling
///
/// There are two reasons why we might observe a divergence:
/// - The integration error of the Hamiltonian is larger than
///   a cutoff value or nan.
/// - The logp function caused a recoverable error (eg if an ODE solver
///   failed)
#[derive(Debug, Clone)]
pub struct DivergenceInfo {
    pub start_momentum: Option<Box<[f64]>>,
    pub start_location: Option<Box<[f64]>>,
    pub start_gradient: Option<Box<[f64]>>,
    pub end_location: Option<Box<[f64]>>,
    pub energy_error: Option<f64>,
    pub end_idx_in_trajectory: Option<i64>,
    pub start_idx_in_trajectory: Option<i64>,
    pub logp_function_error: Option<Arc<dyn std::error::Error + Send + Sync>>,
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
pub(crate) trait Collector<M: Math> {
    fn register_leapfrog(
        &mut self,
        _math: &mut M,
        _start: &State<M>,
        _end: &State<M>,
        _divergence_info: Option<&DivergenceInfo>,
    ) {
    }
    fn register_draw(&mut self, _math: &mut M, _state: &State<M>, _info: &SampleInfo) {}
    fn register_init(&mut self, _math: &mut M, _state: &State<M>, _options: &NutsOptions) {}
}

/// Errors that happen when we evaluate the logp and gradient function
pub trait LogpError: std::error::Error {
    /// Unrecoverable errors during logp computation stop sampling,
    /// recoverable errors are seen as divergences.
    fn is_recoverable(&self) -> bool;
}

/// The hamiltonian defined by the potential energy and the kinetic energy
pub(crate) trait Hamiltonian<M>: SamplerStats<M>
where
    M: Math,
{
    /// The type that stores a point in phase space
    //type State: State;
    /// Errors that happen during logp evaluation
    type LogpError: LogpError + Send;

    /// Perform one leapfrog step.
    ///
    /// Return either an unrecoverable error, a new state or a divergence.
    fn leapfrog<C: Collector<M>>(
        &mut self,
        math: &mut M,
        pool: &mut StatePool<M>,
        start: &State<M>,
        dir: Direction,
        initial_energy: f64,
        collector: &mut C,
    ) -> Result<std::result::Result<State<M>, DivergenceInfo>>;

    /// Initialize a state at a new location.
    ///
    /// The momentum should be initialized to some arbitrary invalid number,
    /// it will later be set using Self::randomize_momentum.
    fn init_state(
        &mut self,
        math: &mut M,
        pool: &mut StatePool<M>,
        init: &[f64],
    ) -> Result<State<M>>;

    /// Randomize the momentum part of a state
    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut State<M>,
        rng: &mut R,
    );

    fn new_empty_state(&mut self, math: &mut M, pool: &mut StatePool<M>) -> State<M>;

    /// Crate a new state pool that can be used to crate new states.
    fn new_pool(&mut self, math: &mut M, capacity: usize) -> StatePool<M>;
}

/// Information about a draw, exported as part of the sampler stats
#[derive(Debug)]
pub struct SampleInfo {
    /// The depth of the trajectory that this point was sampled from
    pub depth: u64,

    /// More detailed information about a divergence that might have
    /// occured in the trajectory.
    pub divergence_info: Option<DivergenceInfo>,

    /// Whether the trajectory was terminated because it reached
    /// the maximum tree depth.
    pub reached_maxdepth: bool,

    pub initial_energy: f64,
    pub draw_energy: f64,
}

/// A part of the trajectory tree during NUTS sampling.
struct NutsTree<M: Math, H: Hamiltonian<M>, C: Collector<M>> {
    /// The left position of the tree.
    ///
    /// The left side always has the smaller index_in_trajectory.
    /// Leapfrogs in backward direction will replace the left.
    left: State<M>,
    right: State<M>,

    /// A draw from the trajectory between left and right using
    /// multinomial sampling.
    draw: State<M>,
    log_size: f64,
    depth: u64,
    initial_energy: f64,

    /// A tree is the main tree if it contains the initial point
    /// of the trajectory.
    is_main: bool,
    _phantom: PhantomData<H>,
    _phantom2: PhantomData<C>,
}

enum ExtendResult<M: Math, H: Hamiltonian<M>, C: Collector<M>> {
    /// The tree extension succeeded properly, and the termination
    /// criterion was not reached.
    Ok(NutsTree<M, H, C>),
    /// An unrecoverable error happend during a leapfrog step
    Err(NutsError),
    /// Tree extension succeeded and the termination criterion
    /// was reached.
    Turning(NutsTree<M, H, C>),
    /// A divergence happend during tree extension.
    Diverging(NutsTree<M, H, C>, DivergenceInfo),
}

impl<M: Math, H: Hamiltonian<M>, C: Collector<M>> NutsTree<M, H, C> {
    fn new(state: State<M>) -> NutsTree<M, H, C> {
        let initial_energy = state.energy();
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: 0.,
            initial_energy,
            is_main: true,
            _phantom: PhantomData,
            _phantom2: PhantomData,
        }
    }

    #[inline]
    fn extend<R>(
        mut self,
        math: &mut M,
        pool: &mut StatePool<M>,
        rng: &mut R,
        potential: &mut H,
        direction: Direction,
        collector: &mut C,
        options: &NutsOptions,
    ) -> ExtendResult<M, H, C>
    where
        H: Hamiltonian<M>,
        R: rand::Rng + ?Sized,
    {
        let mut other = match self.single_step(math, pool, potential, direction, collector) {
            Ok(Ok(tree)) => tree,
            Ok(Err(info)) => return ExtendResult::Diverging(self, info),
            Err(err) => return ExtendResult::Err(err),
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(math, pool, rng, potential, direction, collector, options) {
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

        let turning = if options.check_turning {
            let mut turning = first.is_turning(math, last);
            if self.depth > 0 {
                if !turning {
                    turning = self.right.is_turning(math, &other.right);
                }
                if !turning {
                    turning = self.left.is_turning(math, &other.left);
                }
            }
            turning
        } else {
            false
        };

        self.merge_into(math, other, rng, direction);

        if turning {
            ExtendResult::Turning(self)
        } else {
            ExtendResult::Ok(self)
        }
    }

    fn merge_into<R: rand::Rng + ?Sized>(
        &mut self,
        _math: &mut M,
        other: NutsTree<M, H, C>,
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
        math: &mut M,
        pool: &mut StatePool<M>,
        potential: &mut H,
        direction: Direction,
        collector: &mut C,
    ) -> Result<std::result::Result<NutsTree<M, H, C>, DivergenceInfo>> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let end = match potential.leapfrog(
            math,
            pool,
            start,
            direction,
            self.initial_energy,
            collector,
        ) {
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
            _phantom: PhantomData,
            _phantom2: PhantomData,
        }))
    }

    fn info(&self, maxdepth: bool, divergence_info: Option<DivergenceInfo>) -> SampleInfo {
        SampleInfo {
            depth: self.depth,
            divergence_info,
            reached_maxdepth: maxdepth,
            initial_energy: self.initial_energy,
            draw_energy: self.draw.energy(),
        }
    }
}

pub struct NutsOptions {
    pub maxdepth: u64,
    pub store_gradient: bool,
    pub store_unconstrained: bool,
    pub check_turning: bool,
    pub store_divergences: bool,
}

pub(crate) fn draw<M, P, R, C>(
    math: &mut M,
    pool: &mut StatePool<M>,
    init: &mut State<M>,
    rng: &mut R,
    potential: &mut P,
    options: &NutsOptions,
    collector: &mut C,
) -> Result<(State<M>, SampleInfo)>
where
    M: Math,
    P: Hamiltonian<M>,
    R: rand::Rng + ?Sized,
    C: Collector<M>,
{
    potential.randomize_momentum(math, init, rng);
    init.make_init_point(math);
    collector.register_init(math, init, options);

    let mut tree = NutsTree::new(init.clone());
    while tree.depth < options.maxdepth {
        let direction: Direction = rng.gen();
        tree = match tree.extend(math, pool, rng, potential, direction, collector, options) {
            ExtendResult::Ok(tree) => tree,
            ExtendResult::Turning(tree) => {
                let info = tree.info(false, None);
                collector.register_draw(math, &tree.draw, &info);
                return Ok((tree.draw, info));
            }
            ExtendResult::Diverging(tree, info) => {
                let info = tree.info(false, Some(info));
                collector.register_draw(math, &tree.draw, &info);
                return Ok((tree.draw, info));
            }
            ExtendResult::Err(error) => {
                return Err(error);
            }
        };
    }
    let info = tree.info(true, None);
    collector.register_draw(math, &tree.draw, &info);
    Ok((tree.draw, info))
}

pub(crate) trait SamplerStats<M: Math> {
    type Stats: Send + Debug + Clone;
    type Builder: StatTraceBuilder<Self::Stats>;

    fn new_builder(&self, settings: &impl Settings, dim: usize) -> Self::Builder;
    fn current_stats(&self, math: &mut M) -> Self::Stats;
}

impl StatTraceBuilder<()> for () {
    fn append_value(&mut self, _value: ()) {}

    fn finalize(self) -> Option<StructArray> {
        None
    }
}

pub trait StatTraceBuilder<T: ?Sized>: Clone + Send {
    fn append_value(&mut self, value: T);
    fn finalize(self) -> Option<StructArray>;
}

#[derive(Debug, Clone)]
pub(crate) struct NutsSampleStats<HStats: Send + Debug + Clone, AdaptStats: Send + Debug + Clone> {
    depth: u64,
    maxdepth_reached: bool,
    idx_in_trajectory: i64,
    logp: f64,
    energy: f64,
    energy_error: f64,
    pub(crate) divergence_info: Option<DivergenceInfo>,
    pub(crate) chain: u64,
    pub(crate) draw: u64,
    gradient: Option<Box<[f64]>>,
    unconstrained: Option<Box<[f64]>>,
    pub(crate) potential_stats: HStats,
    pub(crate) strategy_stats: AdaptStats,
    pub(crate) tuning: bool,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SampleStats {
    pub draw: u64,
    pub chain: u64,
    pub diverging: bool,
    pub tuning: bool,
}

#[derive(Clone)]
pub(crate) struct NutsStatsBuilder<H, A> {
    depth: MutablePrimitiveArray<u64>,
    maxdepth_reached: MutableBooleanArray,
    index_in_trajectory: MutablePrimitiveArray<i64>,
    logp: MutablePrimitiveArray<f64>,
    energy: MutablePrimitiveArray<f64>,
    chain: MutablePrimitiveArray<u64>,
    draw: MutablePrimitiveArray<u64>,
    energy_error: MutablePrimitiveArray<f64>,
    unconstrained: Option<MutableFixedSizeListArray<MutablePrimitiveArray<f64>>>,
    gradient: Option<MutableFixedSizeListArray<MutablePrimitiveArray<f64>>>,
    hamiltonian: H,
    adapt: A,
    diverging: MutableBooleanArray,
    divergence_start: Option<MutableFixedSizeListArray<MutablePrimitiveArray<f64>>>,
    divergence_start_grad: Option<MutableFixedSizeListArray<MutablePrimitiveArray<f64>>>,
    divergence_end: Option<MutableFixedSizeListArray<MutablePrimitiveArray<f64>>>,
    divergence_momentum: Option<MutableFixedSizeListArray<MutablePrimitiveArray<f64>>>,
    divergence_msg: Option<MutableUtf8Array<i64>>,
}

impl<HB, AB> NutsStatsBuilder<HB, AB> {
    fn new_with_capacity<
        M: Math,
        H: Hamiltonian<M, Builder = HB>,
        A: AdaptStrategy<M, Builder = AB>,
    >(
        settings: &impl Settings,
        hamiltonian: &H,
        adapt: &A,
        dim: usize,
        options: &NutsOptions,
    ) -> Self {
        let capacity = settings.hint_num_tune() + settings.hint_num_draws();

        let gradient = if options.store_gradient {
            let items = MutablePrimitiveArray::new();
            Some(MutableFixedSizeListArray::new_with_field(
                items, "item", false, dim,
            ))
        } else {
            None
        };

        let unconstrained = if options.store_unconstrained {
            let items = MutablePrimitiveArray::new();
            Some(MutableFixedSizeListArray::new_with_field(
                items, "item", false, dim,
            ))
        } else {
            None
        };

        let (div_start, div_start_grad, div_end, div_mom, div_msg) = if options.store_divergences {
            let start_location_prim = MutablePrimitiveArray::new();
            let start_location_list =
                MutableFixedSizeListArray::new_with_field(start_location_prim, "item", false, dim);

            let start_grad_prim = MutablePrimitiveArray::new();
            let start_grad_list =
                MutableFixedSizeListArray::new_with_field(start_grad_prim, "item", false, dim);

            let end_location_prim = MutablePrimitiveArray::new();
            let end_location_list =
                MutableFixedSizeListArray::new_with_field(end_location_prim, "item", false, dim);

            let momentum_location_prim = MutablePrimitiveArray::new();
            let momentum_location_list = MutableFixedSizeListArray::new_with_field(
                momentum_location_prim,
                "item",
                false,
                dim,
            );

            let msg_list = MutableUtf8Array::new();

            (
                Some(start_location_list),
                Some(start_grad_list),
                Some(end_location_list),
                Some(momentum_location_list),
                Some(msg_list),
            )
        } else {
            (None, None, None, None, None)
        };

        Self {
            depth: MutablePrimitiveArray::with_capacity(capacity),
            maxdepth_reached: MutableBooleanArray::with_capacity(capacity),
            index_in_trajectory: MutablePrimitiveArray::with_capacity(capacity),
            logp: MutablePrimitiveArray::with_capacity(capacity),
            energy: MutablePrimitiveArray::with_capacity(capacity),
            chain: MutablePrimitiveArray::with_capacity(capacity),
            draw: MutablePrimitiveArray::with_capacity(capacity),
            energy_error: MutablePrimitiveArray::with_capacity(capacity),
            gradient,
            unconstrained,
            hamiltonian: hamiltonian.new_builder(settings, dim),
            adapt: adapt.new_builder(settings, dim),
            diverging: MutableBooleanArray::with_capacity(capacity),
            divergence_start: div_start,
            divergence_start_grad: div_start_grad,
            divergence_end: div_end,
            divergence_momentum: div_mom,
            divergence_msg: div_msg,
        }
    }
}

impl<HS, AS, HB, AB> StatTraceBuilder<NutsSampleStats<HS, AS>> for NutsStatsBuilder<HB, AB>
where
    HB: StatTraceBuilder<HS>,
    AB: StatTraceBuilder<AS>,
    HS: Clone + Send + Debug,
    AS: Clone + Send + Debug,
{
    fn append_value(&mut self, value: NutsSampleStats<HS, AS>) {
        self.depth.push(Some(value.depth));
        self.maxdepth_reached.push(Some(value.maxdepth_reached));
        self.index_in_trajectory.push(Some(value.idx_in_trajectory));
        self.logp.push(Some(value.logp));
        self.energy.push(Some(value.energy));
        self.chain.push(Some(value.chain));
        self.draw.push(Some(value.draw));
        self.diverging.push(Some(value.divergence_info.is_some()));
        self.energy_error.push(Some(value.energy_error));

        if let Some(store) = self.gradient.as_mut() {
            store
                .try_push(
                    value
                        .gradient
                        .as_ref()
                        .map(|vals| vals.iter().map(|&x| Some(x))),
                )
                .unwrap();
        }

        if let Some(store) = self.unconstrained.as_mut() {
            store
                .try_push(
                    value
                        .unconstrained
                        .as_ref()
                        .map(|vals| vals.iter().map(|&x| Some(x))),
                )
                .unwrap();
        }

        let info_option = value.divergence_info.as_ref();
        if let Some(div_start) = self.divergence_start.as_mut() {
            div_start
                .try_push(info_option.and_then(|info| {
                    info.start_location
                        .as_ref()
                        .map(|vals| vals.iter().map(|&x| Some(x)))
                }))
                .unwrap();
        }

        let info_option = value.divergence_info.as_ref();
        if let Some(div_grad) = self.divergence_start_grad.as_mut() {
            div_grad
                .try_push(info_option.and_then(|info| {
                    info.start_gradient
                        .as_ref()
                        .map(|vals| vals.iter().map(|&x| Some(x)))
                }))
                .unwrap();
        }

        if let Some(div_end) = self.divergence_end.as_mut() {
            div_end
                .try_push(info_option.and_then(|info| {
                    info.end_location
                        .as_ref()
                        .map(|vals| vals.iter().map(|&x| Some(x)))
                }))
                .unwrap();
        }

        if let Some(div_mom) = self.divergence_momentum.as_mut() {
            div_mom
                .try_push(info_option.and_then(|info| {
                    info.start_momentum
                        .as_ref()
                        .map(|vals| vals.iter().map(|&x| Some(x)))
                }))
                .unwrap();
        }

        if let Some(div_msg) = self.divergence_msg.as_mut() {
            div_msg
                .try_push(info_option.and_then(|info| {
                    info.logp_function_error
                        .as_ref()
                        .map(|err| format!("{}", err))
                }))
                .unwrap();
        }

        self.hamiltonian.append_value(value.potential_stats);
        self.adapt.append_value(value.strategy_stats);
    }

    fn finalize(mut self) -> Option<StructArray> {
        let mut fields = vec![
            Field::new("depth", DataType::UInt64, false),
            Field::new("maxdepth_reached", DataType::Boolean, false),
            Field::new("index_in_trajectory", DataType::Int64, false),
            Field::new("logp", DataType::Float64, false),
            Field::new("energy", DataType::Float64, false),
            Field::new("chain", DataType::UInt64, false),
            Field::new("draw", DataType::UInt64, false),
            Field::new("diverging", DataType::Boolean, false),
            Field::new("energy_error", DataType::Float64, false),
        ];

        let mut arrays = vec![
            self.depth.as_box(),
            self.maxdepth_reached.as_box(),
            self.index_in_trajectory.as_box(),
            self.logp.as_box(),
            self.energy.as_box(),
            self.chain.as_box(),
            self.draw.as_box(),
            self.diverging.as_box(),
            self.energy_error.as_box(),
        ];

        if let Some(hamiltonian) = self.hamiltonian.finalize() {
            let hamiltonian = hamiltonian.into_data();
            assert!(hamiltonian.2.is_none());
            fields.extend(hamiltonian.0);
            arrays.extend(hamiltonian.1);
        }
        if let Some(adapt) = self.adapt.finalize() {
            let adapt = adapt.into_data();
            assert!(adapt.2.is_none());
            fields.extend(adapt.0);
            arrays.extend(adapt.1);
        }

        if let Some(mut gradient) = self.gradient.take() {
            fields.push(Field::new("gradient", gradient.data_type().clone(), true));
            arrays.push(gradient.as_box());
        }

        if let Some(mut unconstrained) = self.unconstrained.take() {
            fields.push(Field::new(
                "unconstrained_draw",
                unconstrained.data_type().clone(),
                true,
            ));
            arrays.push(unconstrained.as_box());
        }

        if let Some(mut div_start) = self.divergence_start.take() {
            fields.push(Field::new(
                "divergence_start",
                div_start.data_type().clone(),
                true,
            ));
            arrays.push(div_start.as_box());
        }

        if let Some(mut div_start_grad) = self.divergence_start_grad.take() {
            fields.push(Field::new(
                "divergence_start_gradient",
                div_start_grad.data_type().clone(),
                true,
            ));
            arrays.push(div_start_grad.as_box());
        }

        if let Some(mut div_end) = self.divergence_end.take() {
            fields.push(Field::new(
                "divergence_end",
                div_end.data_type().clone(),
                true,
            ));
            arrays.push(div_end.as_box());
        }

        if let Some(mut div_mom) = self.divergence_momentum.take() {
            fields.push(Field::new(
                "divergence_momentum",
                div_mom.data_type().clone(),
                true,
            ));
            arrays.push(div_mom.as_box());
        }

        if let Some(mut div_msg) = self.divergence_msg.take() {
            fields.push(Field::new(
                "divergence_message",
                div_msg.data_type().clone(),
                true,
            ));
            arrays.push(div_msg.as_box());
        }

        Some(StructArray::new(DataType::Struct(fields), arrays, None))
    }
}

impl<H, A> From<NutsSampleStats<H, A>> for SampleStats
where
    H: Clone + Debug + Send,
    A: Clone + Debug + Send,
{
    fn from(value: NutsSampleStats<H, A>) -> Self {
        todo!()
    }
}

/// Draw samples from the posterior distribution using Hamiltonian MCMC.
pub(crate) trait Chain<M: Math>: SamplerStats<M> {
    type Hamiltonian; //: Hamiltonian<M>;
    type AdaptStrategy; //: AdaptStrategy<M>;

    /// Initialize the sampler to a position. This should be called
    /// before calling draw.
    ///
    /// This fails if the logp function returns an error.
    fn set_position(&mut self, position: &[f64]) -> Result<()>;

    /// Draw a new sample and return the position and some diagnosic information.
    fn draw(&mut self) -> Result<(Box<[f64]>, Self::Stats)>;

    // Extract a summary of the sample stats
    fn stats_summary(stats: &Self::Stats) -> SampleStats;

    /// The dimensionality of the posterior.
    fn dim(&self) -> usize;
}

pub struct NutsChain<M, P, R, A>
where
    M: Math,
    P: Hamiltonian<M>,
    R: rand::Rng,
    A: AdaptStrategy<M, Potential = P>,
{
    pool: StatePool<M>,
    potential: P,
    collector: A::Collector,
    options: NutsOptions,
    rng: R,
    init: State<M>,
    chain: u64,
    draw_count: u64,
    strategy: A,
    math: M,
    stats: Option<NutsSampleStats<P::Stats, A::Stats>>,
}

impl<M, P, R, A> NutsChain<M, P, R, A>
where
    M: Math,
    P: Hamiltonian<M>,
    R: rand::Rng,
    A: AdaptStrategy<M, Potential = P>,
{
    pub fn new(
        mut math: M,
        mut potential: P,
        strategy: A,
        options: NutsOptions,
        rng: R,
        chain: u64,
    ) -> Self {
        let pool_size: usize = options.maxdepth.checked_mul(2).unwrap().try_into().unwrap();
        let mut pool = potential.new_pool(&mut math, pool_size);
        let init = potential.new_empty_state(&mut math, &mut pool);
        let collector = strategy.new_collector(&mut math);
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
            math,
            stats: None,
        }
    }
}

pub(crate) trait AdaptStrategy<M: Math>: SamplerStats<M> {
    type Potential: Hamiltonian<M>;
    type Collector: Collector<M>;
    type Options: Copy + Send + Default;

    fn new(math: &mut M, options: Self::Options, num_tune: u64) -> Self;

    fn init(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &State<M>,
    );

    fn adapt(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: u64,
        collector: &Self::Collector,
    );

    fn new_collector(&self, math: &mut M) -> Self::Collector;
    fn is_tuning(&self) -> bool;
}

impl<M, H, R, A> SamplerStats<M> for NutsChain<M, H, R, A>
where
    M: Math,
    H: Hamiltonian<M> + SamplerStats<M>,
    R: rand::Rng,
    A: AdaptStrategy<M, Potential = H>,
{
    type Builder = NutsStatsBuilder<H::Builder, A::Builder>;
    type Stats = NutsSampleStats<H::Stats, A::Stats>;

    fn new_builder(&self, settings: &impl Settings, dim: usize) -> Self::Builder {
        NutsStatsBuilder::new_with_capacity(
            settings,
            &self.potential,
            &self.strategy,
            dim,
            &self.options,
        )
    }

    fn current_stats(&self, _math: &mut M) -> Self::Stats {
        self.stats.as_ref().expect("No stats available").clone()
    }
}

impl<M, H, R, A> Chain<M> for NutsChain<M, H, R, A>
where
    M: Math,
    H: Hamiltonian<M>,
    R: rand::Rng,
    A: AdaptStrategy<M, Potential = H>,
{
    type Hamiltonian = H;
    type AdaptStrategy = A;

    fn set_position(&mut self, position: &[f64]) -> Result<()> {
        let state = self
            .potential
            .init_state(&mut self.math, &mut self.pool, position)?;
        self.init = state;
        self.strategy.init(
            &mut self.math,
            &mut self.options,
            &mut self.potential,
            &self.init,
        );
        Ok(())
    }

    fn draw(&mut self) -> Result<(Box<[f64]>, Self::Stats)> {
        let (state, info) = draw(
            &mut self.math,
            &mut self.pool,
            &mut self.init,
            &mut self.rng,
            &mut self.potential,
            &self.options,
            &mut self.collector,
        )?;
        let mut position: Box<[f64]> = vec![0f64; self.math.dim()].into();
        state.write_position(&mut self.math, &mut position);

        self.strategy.adapt(
            &mut self.math,
            &mut self.options,
            &mut self.potential,
            self.draw_count,
            &self.collector,
        );

        self.draw_count += 1;

        let stats = NutsSampleStats {
            depth: info.depth,
            maxdepth_reached: info.reached_maxdepth,
            idx_in_trajectory: state.index_in_trajectory(),
            logp: -state.potential_energy(),
            energy: state.energy(),
            energy_error: info.draw_energy - info.initial_energy,
            divergence_info: info.divergence_info,
            chain: self.chain,
            draw: self.draw_count,
            potential_stats: self.potential.current_stats(&mut self.math),
            strategy_stats: self.strategy.current_stats(&mut self.math),
            gradient: if self.options.store_gradient {
                let mut gradient: Box<[f64]> = vec![0f64; self.math.dim()].into();
                state.write_gradient(&mut self.math, &mut gradient);
                Some(gradient)
            } else {
                None
            },
            unconstrained: if self.options.store_unconstrained {
                let mut unconstrained: Box<[f64]> = vec![0f64; self.math.dim()].into();
                state.write_position(&mut self.math, &mut unconstrained);
                Some(unconstrained)
            } else {
                None
            },
            tuning: self.strategy.is_tuning(),
        };

        self.init = state;
        Ok((position, stats))
    }

    fn dim(&self) -> usize {
        self.math.dim()
    }

    fn stats_summary(stats: &Self::Stats) -> SampleStats {
        SampleStats {
            draw: stats.draw,
            chain: stats.chain,
            diverging: stats.divergence_info.is_some(),
            tuning: stats.tuning,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{
        adapt_strategy::test_logps::NormalLogp, cpu_math::CpuMath, sampler::DiagGradNutsSettings,
        Chain, Settings,
    };

    use super::StatTraceBuilder;

    #[test]
    fn to_arrow() {
        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.);
        let math = CpuMath::new(func);

        let settings = DiagGradNutsSettings::default();
        let mut rng = thread_rng();

        let mut chain = settings.new_chain(0, math, &mut rng);

        let mut builder = chain.new_builder(ndim, &settings);

        for _ in 0..10 {
            let (_, stats) = chain.draw().unwrap();
            builder.append_value(&stats);
        }

        let stats = builder.finalize();
        dbg!(stats);
    }
}
