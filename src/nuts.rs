use arrow::array::{
    Array, ArrayBuilder, BooleanBuilder, FixedSizeListBuilder, PrimitiveBuilder, StringBuilder,
    StructArray,
};
use arrow::datatypes::{DataType, Field, Fields, Float64Type, Int64Type, UInt64Type};
use rand::Rng;
use thiserror::Error;

use std::ops::Deref;
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
pub trait Collector<M: Math> {
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

pub trait HamiltonianStats<M: Math>: SamplerStats<M> {
    fn stat_step_size(stats: &Self::Stats) -> f64;
}

/// The hamiltonian defined by the potential energy and the kinetic energy
pub trait Hamiltonian<M>: HamiltonianStats<M>
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

    fn copy_state(&mut self, math: &mut M, pool: &mut StatePool<M>, state: &State<M>) -> State<M>;

    fn stepsize_mut(&mut self) -> &mut f64;
    fn stepsize(&self) -> f64;
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

    #[allow(clippy::too_many_arguments)]
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

        if (other.log_size >= self_log_size)
            || (rng.gen_bool((other.log_size - self_log_size).exp()))
        {
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

pub trait SamplerStats<M: Math> {
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

    fn inspect(&self) -> Option<StructArray> {
        None
    }
}

pub trait StatTraceBuilder<T: ?Sized>: Send {
    fn append_value(&mut self, value: T);
    fn finalize(self) -> Option<StructArray>;
    fn inspect(&self) -> Option<StructArray>;
}

#[derive(Debug, Clone)]
pub struct NutsSampleStats<HStats: Send + Debug + Clone, AdaptStats: Send + Debug + Clone> {
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
    pub num_steps: usize,
    pub step_size: f64,
}

pub struct NutsStatsBuilder<H, A> {
    depth: PrimitiveBuilder<UInt64Type>,
    maxdepth_reached: BooleanBuilder,
    index_in_trajectory: PrimitiveBuilder<Int64Type>,
    logp: PrimitiveBuilder<Float64Type>,
    energy: PrimitiveBuilder<Float64Type>,
    chain: PrimitiveBuilder<UInt64Type>,
    draw: PrimitiveBuilder<UInt64Type>,
    energy_error: PrimitiveBuilder<Float64Type>,
    unconstrained: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    gradient: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    hamiltonian: H,
    adapt: A,
    diverging: BooleanBuilder,
    divergence_start: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    divergence_start_grad: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    divergence_end: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    divergence_momentum: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    divergence_msg: Option<StringBuilder>,
    n_dim: usize,
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
            let items = PrimitiveBuilder::with_capacity(capacity);
            Some(FixedSizeListBuilder::new(items, dim as i32))
        } else {
            None
        };

        let unconstrained = if options.store_unconstrained {
            let items = PrimitiveBuilder::with_capacity(capacity);
            Some(FixedSizeListBuilder::with_capacity(
                items, dim as i32, capacity,
            ))
        } else {
            None
        };

        let (div_start, div_start_grad, div_end, div_mom, div_msg) = if options.store_divergences {
            let start_location_prim = PrimitiveBuilder::new();
            let start_location_list = FixedSizeListBuilder::new(start_location_prim, dim as i32);

            let start_grad_prim = PrimitiveBuilder::new();
            let start_grad_list = FixedSizeListBuilder::new(start_grad_prim, dim as i32);

            let end_location_prim = PrimitiveBuilder::new();
            let end_location_list = FixedSizeListBuilder::new(end_location_prim, dim as i32);

            let momentum_location_prim = PrimitiveBuilder::new();
            let momentum_location_list =
                FixedSizeListBuilder::new(momentum_location_prim, dim as i32);

            let msg_list = StringBuilder::new();

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
            depth: PrimitiveBuilder::with_capacity(capacity),
            maxdepth_reached: BooleanBuilder::with_capacity(capacity),
            index_in_trajectory: PrimitiveBuilder::with_capacity(capacity),
            logp: PrimitiveBuilder::with_capacity(capacity),
            energy: PrimitiveBuilder::with_capacity(capacity),
            chain: PrimitiveBuilder::with_capacity(capacity),
            draw: PrimitiveBuilder::with_capacity(capacity),
            energy_error: PrimitiveBuilder::with_capacity(capacity),
            gradient,
            unconstrained,
            hamiltonian: hamiltonian.new_builder(settings, dim),
            adapt: adapt.new_builder(settings, dim),
            diverging: BooleanBuilder::with_capacity(capacity),
            divergence_start: div_start,
            divergence_start_grad: div_start_grad,
            divergence_end: div_end,
            divergence_momentum: div_mom,
            divergence_msg: div_msg,
            n_dim: dim,
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
        let NutsSampleStats {
            depth,
            maxdepth_reached,
            idx_in_trajectory,
            logp,
            energy,
            energy_error,
            divergence_info,
            chain,
            draw,
            gradient,
            unconstrained,
            potential_stats,
            strategy_stats,
            tuning,
        } = value;

        // We don't need to store tuning explicity
        let _ = tuning;

        self.depth.append_value(depth);
        self.maxdepth_reached.append_value(maxdepth_reached);
        self.index_in_trajectory.append_value(idx_in_trajectory);
        self.logp.append_value(logp);
        self.energy.append_value(energy);
        self.chain.append_value(chain);
        self.draw.append_value(draw);
        self.diverging.append_value(divergence_info.is_some());
        self.energy_error.append_value(energy_error);

        fn add_slice<V: AsRef<[f64]>>(
            store: &mut Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
            values: Option<V>,
            n_dim: usize,
        ) {
            let Some(store) = store.as_mut() else {
                return;
            };

            if let Some(values) = values.as_ref() {
                store.values().append_slice(values.as_ref());
                store.append(true);
            } else {
                store.values().append_nulls(n_dim);
                store.append(false);
            }
        }

        add_slice(&mut self.gradient, gradient.as_ref(), self.n_dim);
        add_slice(&mut self.unconstrained, unconstrained.as_ref(), self.n_dim);

        let div_info = divergence_info.as_ref();
        add_slice(
            &mut self.divergence_start,
            div_info.and_then(|info| info.start_location.as_ref()),
            self.n_dim,
        );
        add_slice(
            &mut self.divergence_start_grad,
            div_info.and_then(|info| info.start_gradient.as_ref()),
            self.n_dim,
        );
        add_slice(
            &mut self.divergence_end,
            div_info.and_then(|info| info.end_location.as_ref()),
            self.n_dim,
        );
        add_slice(
            &mut self.divergence_momentum,
            div_info.and_then(|info| info.start_momentum.as_ref()),
            self.n_dim,
        );

        if let Some(div_msg) = self.divergence_msg.as_mut() {
            if let Some(err) = div_info.and_then(|info| info.logp_function_error.as_ref()) {
                div_msg.append_value(format!("{}", err));
            } else {
                div_msg.append_null();
            }
        }

        self.hamiltonian.append_value(potential_stats);
        self.adapt.append_value(strategy_stats);
    }

    fn finalize(self) -> Option<StructArray> {
        let Self {
            mut depth,
            mut maxdepth_reached,
            mut index_in_trajectory,
            mut logp,
            mut energy,
            mut chain,
            mut draw,
            mut energy_error,
            unconstrained,
            gradient,
            hamiltonian,
            adapt,
            mut diverging,
            divergence_start,
            divergence_start_grad,
            divergence_end,
            divergence_momentum,
            divergence_msg,
            n_dim,
        } = self;

        let _ = n_dim;

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

        let mut arrays: Vec<Arc<_>> = vec![
            ArrayBuilder::finish(&mut depth),
            ArrayBuilder::finish(&mut maxdepth_reached),
            ArrayBuilder::finish(&mut index_in_trajectory),
            ArrayBuilder::finish(&mut logp),
            ArrayBuilder::finish(&mut energy),
            ArrayBuilder::finish(&mut chain),
            ArrayBuilder::finish(&mut draw),
            ArrayBuilder::finish(&mut diverging),
            ArrayBuilder::finish(&mut energy_error),
        ];

        fn merge_into<T: ?Sized, B: StatTraceBuilder<T>>(
            builder: B,
            arrays: &mut Vec<Arc<dyn Array>>,
            fields: &mut Vec<Field>,
        ) {
            let Some(struct_array) = builder.finalize() else {
                return;
            };

            let (struct_fields, struct_arrays, bitmap) = struct_array.into_parts();
            assert!(bitmap.is_none());
            arrays.extend(struct_arrays);
            fields.extend(struct_fields.into_iter().map(|x| x.deref().clone()));
        }

        fn add_field<B: ArrayBuilder>(
            mut builder: Option<B>,
            name: &str,
            arrays: &mut Vec<Arc<dyn Array>>,
            fields: &mut Vec<Field>,
        ) {
            let Some(mut builder) = builder.take() else {
                return;
            };

            let array = ArrayBuilder::finish(&mut builder);
            fields.push(Field::new(name, array.data_type().clone(), true));
            arrays.push(array);
        }

        merge_into(hamiltonian, &mut arrays, &mut fields);
        merge_into(adapt, &mut arrays, &mut fields);

        add_field(gradient, "gradient", &mut arrays, &mut fields);
        add_field(
            unconstrained,
            "unconstrained_draw",
            &mut arrays,
            &mut fields,
        );
        add_field(
            divergence_start,
            "divergence_start",
            &mut arrays,
            &mut fields,
        );
        add_field(
            divergence_start_grad,
            "divergence_start_gradient",
            &mut arrays,
            &mut fields,
        );
        add_field(divergence_end, "divergence_end", &mut arrays, &mut fields);
        add_field(
            divergence_momentum,
            "divergence_momentum",
            &mut arrays,
            &mut fields,
        );
        add_field(
            divergence_msg,
            "divergence_messagem",
            &mut arrays,
            &mut fields,
        );

        let fields = Fields::from(fields);
        Some(StructArray::new(fields, arrays, None))
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self {
            depth,
            maxdepth_reached,
            index_in_trajectory,
            logp,
            energy,
            chain,
            draw,
            energy_error,
            unconstrained,
            gradient,
            hamiltonian,
            adapt,
            diverging,
            divergence_start,
            divergence_start_grad,
            divergence_end,
            divergence_momentum,
            divergence_msg,
            n_dim,
        } = self;

        let _ = n_dim;

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

        let mut arrays: Vec<Arc<_>> = vec![
            ArrayBuilder::finish_cloned(depth),
            ArrayBuilder::finish_cloned(maxdepth_reached),
            ArrayBuilder::finish_cloned(index_in_trajectory),
            ArrayBuilder::finish_cloned(logp),
            ArrayBuilder::finish_cloned(energy),
            ArrayBuilder::finish_cloned(chain),
            ArrayBuilder::finish_cloned(draw),
            ArrayBuilder::finish_cloned(diverging),
            ArrayBuilder::finish_cloned(energy_error),
        ];

        fn merge_into<T: ?Sized, B: StatTraceBuilder<T>>(
            builder: &B,
            arrays: &mut Vec<Arc<dyn Array>>,
            fields: &mut Vec<Field>,
        ) {
            let Some(struct_array) = builder.inspect() else {
                return;
            };

            let (struct_fields, struct_arrays, bitmap) = struct_array.into_parts();
            assert!(bitmap.is_none());
            arrays.extend(struct_arrays);
            fields.extend(struct_fields.into_iter().map(|x| x.deref().clone()));
        }

        fn add_field<B: ArrayBuilder>(
            builder: &Option<B>,
            name: &str,
            arrays: &mut Vec<Arc<dyn Array>>,
            fields: &mut Vec<Field>,
        ) {
            let Some(builder) = builder.as_ref() else {
                return;
            };

            let array = ArrayBuilder::finish_cloned(builder);
            fields.push(Field::new(name, array.data_type().clone(), true));
            arrays.push(array);
        }

        merge_into(hamiltonian, &mut arrays, &mut fields);
        merge_into(adapt, &mut arrays, &mut fields);

        add_field(gradient, "gradient", &mut arrays, &mut fields);
        add_field(
            unconstrained,
            "unconstrained_draw",
            &mut arrays,
            &mut fields,
        );
        add_field(
            divergence_start,
            "divergence_start",
            &mut arrays,
            &mut fields,
        );
        add_field(
            divergence_start_grad,
            "divergence_start_gradient",
            &mut arrays,
            &mut fields,
        );
        add_field(divergence_end, "divergence_end", &mut arrays, &mut fields);
        add_field(
            divergence_momentum,
            "divergence_momentum",
            &mut arrays,
            &mut fields,
        );
        add_field(
            divergence_msg,
            "divergence_messagem",
            &mut arrays,
            &mut fields,
        );

        let fields = Fields::from(fields);
        Some(StructArray::new(fields, arrays, None))
    }
}

/// Draw samples from the posterior distribution using Hamiltonian MCMC.
pub trait Chain<M: Math>: SamplerStats<M> {
    type Hamiltonian: Hamiltonian<M>;
    type AdaptStrategy: AdaptStrategy<M>;

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

pub trait AdaptStats<M: Math>: SamplerStats<M> {
    fn num_grad_evals(stats: &Self::Stats) -> usize;
}

pub trait AdaptStrategy<M: Math>: AdaptStats<M> {
    type Potential: Hamiltonian<M>;
    type Collector: Collector<M>;
    type Options: Copy + Send + Debug + Default;

    fn new(math: &mut M, options: Self::Options, num_tune: u64) -> Self;

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &State<M>,
        rng: &mut R,
    );

    #[allow(clippy::too_many_arguments)]
    fn adapt<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: u64,
        collector: &Self::Collector,
        state: &State<M>,
        rng: &mut R,
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
            &mut self.rng,
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

        self.strategy.adapt(
            &mut self.math,
            &mut self.options,
            &mut self.potential,
            self.draw_count,
            &self.collector,
            &state,
            &mut self.rng,
        );

        self.draw_count += 1;

        self.init = state;
        Ok((position, stats))
    }

    fn dim(&self) -> usize {
        self.math.dim()
    }

    fn stats_summary(stats: &Self::Stats) -> SampleStats {
        let pot_stats = &stats.potential_stats;
        let step_size = H::stat_step_size(pot_stats);
        let adapt_stats = &stats.strategy_stats;
        let num_steps = A::num_grad_evals(adapt_stats);
        SampleStats {
            draw: stats.draw,
            chain: stats.chain,
            diverging: stats.divergence_info.is_some(),
            tuning: stats.tuning,
            num_steps,
            step_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{
        adapt_strategy::test_logps::NormalLogp,
        cpu_math::CpuMath,
        nuts::{Chain, SamplerStats},
        sampler::DiagGradNutsSettings,
        Settings,
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

        let mut builder = chain.new_builder(&settings, ndim);

        for _ in 0..10 {
            let (_, stats) = chain.draw().unwrap();
            builder.append_value(stats);
        }

        builder.finalize();
    }
}
