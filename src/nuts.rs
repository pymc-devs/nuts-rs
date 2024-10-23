use arrow::array::{
    Array, ArrayBuilder, BooleanBuilder, FixedSizeListBuilder, PrimitiveBuilder, StringBuilder,
    StructArray,
};
use arrow::datatypes::{DataType, Field, Fields, Float64Type, Int64Type, UInt64Type};
use thiserror::Error;

use std::ops::Deref;
use std::sync::Arc;
use std::{fmt::Debug, marker::PhantomData};

use crate::chain::AdaptStrategy;
use crate::hamiltonian::{Direction, DivergenceInfo, Hamiltonian, LeapfrogResult, Point};
use crate::math::logaddexp;
use crate::sampler::Settings;
use crate::sampler_stats::StatTraceBuilder;
use crate::state::State;

use crate::math_base::Math;

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum NutsError {
    #[error("Logp function returned error: {0:?}")]
    LogpFailure(Box<dyn std::error::Error + Send + Sync>),

    #[error("Could not serialize sample stats")]
    SerializeFailure(),

    #[error("Could not initialize state because of bad initial gradient: {0:?}")]
    BadInitGrad(Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, NutsError>;

/// Callbacks for various events during a Nuts sampling step.
///
/// Collectors can compute statistics like the mean acceptance rate
/// or collect data for mass matrix adaptation.
pub trait Collector<M: Math, P: Point<M>> {
    fn register_leapfrog(
        &mut self,
        _math: &mut M,
        _start: &State<M, P>,
        _end: &State<M, P>,
        _divergence_info: Option<&DivergenceInfo>,
    ) {
    }
    fn register_draw(&mut self, _math: &mut M, _state: &State<M, P>, _info: &SampleInfo) {}
    fn register_init(&mut self, _math: &mut M, _state: &State<M, P>, _options: &NutsOptions) {}
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
struct NutsTree<M: Math, H: Hamiltonian<M>, C: Collector<M, H::Point>> {
    /// The left position of the tree.
    ///
    /// The left side always has the smaller index_in_trajectory.
    /// Leapfrogs in backward direction will replace the left.
    left: State<M, H::Point>,
    right: State<M, H::Point>,

    /// A draw from the trajectory between left and right using
    /// multinomial sampling.
    draw: State<M, H::Point>,
    log_size: f64,
    depth: u64,

    /// A tree is the main tree if it contains the initial point
    /// of the trajectory.
    is_main: bool,
    _phantom2: PhantomData<C>,
}

enum ExtendResult<M: Math, H: Hamiltonian<M>, C: Collector<M, H::Point>> {
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

impl<M: Math, H: Hamiltonian<M>, C: Collector<M, H::Point>> NutsTree<M, H, C> {
    fn new(state: State<M, H::Point>) -> NutsTree<M, H, C> {
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: 0.,
            is_main: true,
            _phantom2: PhantomData,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn extend<R>(
        mut self,
        math: &mut M,
        rng: &mut R,
        hamiltonian: &mut H,
        direction: Direction,
        collector: &mut C,
        options: &NutsOptions,
    ) -> ExtendResult<M, H, C>
    where
        H: Hamiltonian<M>,
        R: rand::Rng + ?Sized,
    {
        let mut other = match self.single_step(math, hamiltonian, direction, collector) {
            Ok(Ok(tree)) => tree,
            Ok(Err(info)) => return ExtendResult::Diverging(self, info),
            Err(err) => return ExtendResult::Err(err),
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(math, rng, hamiltonian, direction, collector, options) {
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
            let mut turning = hamiltonian.is_turning(math, first, last);
            if self.depth > 0 {
                if !turning {
                    turning = hamiltonian.is_turning(math, &self.right, &other.right);
                }
                if !turning {
                    turning = hamiltonian.is_turning(math, &self.left, &other.left);
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
        hamiltonian: &mut H,
        direction: Direction,
        collector: &mut C,
    ) -> Result<std::result::Result<NutsTree<M, H, C>, DivergenceInfo>> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let end = match hamiltonian.leapfrog(math, start, direction, collector) {
            LeapfrogResult::Divergence(info) => return Ok(Err(info)),
            LeapfrogResult::Err(err) => return Err(NutsError::LogpFailure(err.into())),
            LeapfrogResult::Ok(end) => end,
        };

        let log_size = -end.point().energy_error();
        Ok(Ok(NutsTree {
            right: end.clone(),
            left: end.clone(),
            draw: end,
            depth: 0,
            log_size,
            is_main: false,
            _phantom2: PhantomData,
        }))
    }

    fn info(&self, maxdepth: bool, divergence_info: Option<DivergenceInfo>) -> SampleInfo {
        SampleInfo {
            depth: self.depth,
            divergence_info,
            reached_maxdepth: maxdepth,
            initial_energy: self.draw.point().initial_energy(),
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

pub(crate) fn draw<M, H, R, C>(
    math: &mut M,
    init: &mut State<M, H::Point>,
    rng: &mut R,
    hamiltonian: &mut H,
    options: &NutsOptions,
    collector: &mut C,
) -> Result<(State<M, H::Point>, SampleInfo)>
where
    M: Math,
    H: Hamiltonian<M>,
    R: rand::Rng + ?Sized,
    C: Collector<M, H::Point>,
{
    hamiltonian.initialize_trajectory(math, init, rng)?;
    collector.register_init(math, init, options);

    let mut tree = NutsTree::new(init.clone());
    while tree.depth < options.maxdepth {
        let direction: Direction = rng.gen();
        tree = match tree.extend(math, rng, hamiltonian, direction, collector, options) {
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

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NutsSampleStats<HStats: Send + Debug + Clone, AdaptStats: Send + Debug + Clone> {
    pub depth: u64,
    pub maxdepth_reached: bool,
    pub idx_in_trajectory: i64,
    pub logp: f64,
    pub energy: f64,
    pub energy_error: f64,
    pub divergence_info: Option<DivergenceInfo>,
    pub chain: u64,
    pub draw: u64,
    pub gradient: Option<Box<[f64]>>,
    pub unconstrained: Option<Box<[f64]>>,
    pub potential_stats: HStats,
    pub strategy_stats: AdaptStats,
    pub tuning: bool,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SampleStats {
    pub draw: u64,
    pub chain: u64,
    pub diverging: bool,
    pub tuning: bool,
    pub step_size: f64,
    pub num_steps: u64,
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
    pub fn new_with_capacity<
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

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{
        adapt_strategy::test_logps::NormalLogp,
        cpu_math::CpuMath,
        sampler::DiagGradNutsSettings,
        sampler_stats::{SamplerStats, StatTraceBuilder},
        Chain, Settings,
    };

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
