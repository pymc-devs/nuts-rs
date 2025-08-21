use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use arrow::{
    array::{
        Array, ArrayBuilder, BooleanBuilder, FixedSizeListBuilder, PrimitiveBuilder, StringBuilder,
        StructArray,
    },
    datatypes::{DataType, Field, Fields, Float64Type, Int64Type, UInt64Type},
};
use rand::Rng;

use crate::{
    hamiltonian::{Hamiltonian, Point},
    nuts::{draw, Collector, NutsOptions, SampleInfo},
    sampler::Progress,
    sampler_stats::{SamplerStats, StatTraceBuilder},
    state::State,
    Math, NutsError, Settings,
};

use anyhow::Result;

/// Draw samples from the posterior distribution using Hamiltonian MCMC.
pub trait Chain<M: Math>: SamplerStats<M> {
    type AdaptStrategy: AdaptStrategy<M>;

    /// Initialize the sampler to a position. This should be called
    /// before calling draw.
    ///
    /// This fails if the logp function returns an error.
    fn set_position(&mut self, position: &[f64]) -> Result<()>;

    /// Draw a new sample and return the position and some diagnosic information.
    fn draw(&mut self) -> Result<(Box<[f64]>, Progress)>;

    /// The dimensionality of the posterior.
    fn dim(&self) -> usize;
}

pub struct NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    hamiltonian: A::Hamiltonian,
    collector: A::Collector,
    options: NutsOptions,
    rng: R,
    state: State<M, <A::Hamiltonian as Hamiltonian<M>>::Point>,
    last_info: Option<SampleInfo>,
    chain: u64,
    draw_count: u64,
    strategy: A,
    math: RefCell<M>,
}

impl<M, R, A> NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    pub fn new(
        mut math: M,
        mut hamiltonian: A::Hamiltonian,
        strategy: A,
        options: NutsOptions,
        rng: R,
        chain: u64,
    ) -> Self {
        let init = hamiltonian.pool().new_state(&mut math);
        let collector = strategy.new_collector(&mut math);
        NutsChain {
            hamiltonian,
            collector,
            options,
            rng,
            state: init,
            last_info: None,
            chain,
            draw_count: 0,
            strategy,
            math: math.into(),
        }
    }
}

pub trait AdaptStrategy<M: Math>: SamplerStats<M> {
    type Hamiltonian: Hamiltonian<M>;
    type Collector: Collector<M, <Self::Hamiltonian as Hamiltonian<M>>::Point>;
    type Options: Copy + Send + Debug + Default;

    fn new(math: &mut M, options: Self::Options, num_tune: u64, chain: u64) -> Self;

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        position: &[f64],
        rng: &mut R,
    ) -> Result<(), NutsError>;

    #[allow(clippy::too_many_arguments)]
    fn adapt<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        draw: u64,
        collector: &Self::Collector,
        state: &State<M, <Self::Hamiltonian as Hamiltonian<M>>::Point>,
        rng: &mut R,
    ) -> Result<(), NutsError>;

    fn new_collector(&self, math: &mut M) -> Self::Collector;
    fn is_tuning(&self) -> bool;
    fn last_num_steps(&self) -> u64;
}

impl<M, R, A> SamplerStats<M> for NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    type Builder = NutsStatsBuilder<M, A>;
    type StatOptions = StatOptions<M, A>;

    fn new_builder(
        &self,
        options: StatOptions<M, A>,
        settings: &impl Settings,
        dim: usize,
    ) -> Self::Builder {
        NutsStatsBuilder::new_with_capacity(
            options,
            settings,
            &self.hamiltonian,
            &self.strategy,
            self.state.point(),
            dim,
            &self.options,
        )
    }
}

impl<M, R, A> Chain<M> for NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    type AdaptStrategy = A;

    fn set_position(&mut self, position: &[f64]) -> Result<()> {
        let mut math_ = self.math.borrow_mut();
        let math = math_.deref_mut();
        self.strategy.init(
            math,
            &mut self.options,
            &mut self.hamiltonian,
            position,
            &mut self.rng,
        )?;
        self.state = self.hamiltonian.init_state(math, position)?;
        Ok(())
    }

    fn draw(&mut self) -> Result<(Box<[f64]>, Progress)> {
        let mut math_ = self.math.borrow_mut();
        let math = math_.deref_mut();
        let (state, info) = draw(
            math,
            &mut self.state,
            &mut self.rng,
            &mut self.hamiltonian,
            &self.options,
            &mut self.collector,
        )?;
        let mut position: Box<[f64]> = vec![0f64; math.dim()].into();
        state.write_position(math, &mut position);

        let progress = Progress {
            draw: self.draw_count,
            chain: self.chain,
            diverging: info.divergence_info.is_some(),
            tuning: self.strategy.is_tuning(),
            step_size: self.hamiltonian.step_size(),
            num_steps: self.strategy.last_num_steps(),
            log_p: state.log_p(),
        };

        self.strategy.adapt(
            math,
            &mut self.options,
            &mut self.hamiltonian,
            self.draw_count,
            &self.collector,
            &state,
            &mut self.rng,
        )?;

        self.draw_count += 1;

        self.state = state;
        self.last_info = Some(info);
        Ok((position, progress))
    }

    fn dim(&self) -> usize {
        self.math.borrow().dim()
    }
}

pub struct NutsStatsBuilder<M: Math, A: AdaptStrategy<M>> {
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
    hamiltonian: <A::Hamiltonian as SamplerStats<M>>::Builder,
    adapt: A::Builder,
    point: <<A::Hamiltonian as Hamiltonian<M>>::Point as SamplerStats<M>>::Builder,
    diverging: BooleanBuilder,
    divergence_start: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    divergence_start_grad: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    divergence_end: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    divergence_momentum: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    divergence_msg: Option<StringBuilder>,
}

pub struct StatOptions<M: Math, A: AdaptStrategy<M>> {
    pub adapt: A::StatOptions,
    pub hamiltonian: <A::Hamiltonian as SamplerStats<M>>::StatOptions,
    pub point: <<A::Hamiltonian as Hamiltonian<M>>::Point as SamplerStats<M>>::StatOptions,
}

impl<M: Math, A: AdaptStrategy<M>> NutsStatsBuilder<M, A> {
    pub fn new_with_capacity(
        stat_options: StatOptions<M, A>,
        settings: &impl Settings,
        hamiltonian: &A::Hamiltonian,
        adapt: &A,
        point: &<A::Hamiltonian as Hamiltonian<M>>::Point,
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
            hamiltonian: hamiltonian.new_builder(stat_options.hamiltonian, settings, dim),
            adapt: adapt.new_builder(stat_options.adapt, settings, dim),
            point: point.new_builder(stat_options.point, settings, dim),
            diverging: BooleanBuilder::with_capacity(capacity),
            divergence_start: div_start,
            divergence_start_grad: div_start_grad,
            divergence_end: div_end,
            divergence_momentum: div_mom,
            divergence_msg: div_msg,
        }
    }
}

impl<M: Math, R: rand::Rng, A: AdaptStrategy<M>> StatTraceBuilder<M, NutsChain<M, R, A>>
    for NutsStatsBuilder<M, A>
{
    fn append_value(&mut self, _math: Option<&mut M>, value: &NutsChain<M, R, A>) {
        let mut math_ = value.math.borrow_mut();
        let math = math_.deref_mut();
        let Self {
            ref mut depth,
            ref mut maxdepth_reached,
            ref mut index_in_trajectory,
            logp,
            energy,
            chain,
            draw,
            energy_error,
            ref mut unconstrained,
            ref mut gradient,
            hamiltonian,
            adapt,
            point,
            diverging,
            ref mut divergence_start,
            divergence_start_grad,
            divergence_end,
            divergence_momentum,
            divergence_msg,
        } = self;

        let info = value.last_info.as_ref().expect("Sampler has not started");
        let draw_point = value.state.point();

        depth.append_value(info.depth);
        maxdepth_reached.append_value(info.reached_maxdepth);
        index_in_trajectory.append_value(draw_point.index_in_trajectory());
        logp.append_value(draw_point.logp());
        energy.append_value(draw_point.energy());
        chain.append_value(value.chain);
        draw.append_value(value.draw_count);
        diverging.append_value(info.divergence_info.is_some());
        energy_error.append_value(draw_point.energy_error());

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

        let n_dim = math.dim();
        add_slice(gradient, Some(math.box_array(draw_point.gradient())), n_dim);
        add_slice(
            unconstrained,
            Some(math.box_array(draw_point.position())),
            n_dim,
        );

        let div_info = info.divergence_info.as_ref();
        add_slice(
            divergence_start,
            div_info.and_then(|info| info.start_location.as_ref()),
            n_dim,
        );
        add_slice(
            divergence_start_grad,
            div_info.and_then(|info| info.start_gradient.as_ref()),
            n_dim,
        );
        add_slice(
            divergence_end,
            div_info.and_then(|info| info.end_location.as_ref()),
            n_dim,
        );
        add_slice(
            divergence_momentum,
            div_info.and_then(|info| info.start_momentum.as_ref()),
            n_dim,
        );

        if let Some(div_msg) = divergence_msg.as_mut() {
            if let Some(err) = div_info.and_then(|info| info.logp_function_error.as_ref()) {
                div_msg.append_value(format!("{err}"));
            } else {
                div_msg.append_null();
            }
        }

        hamiltonian.append_value(Some(math), &value.hamiltonian);
        adapt.append_value(Some(math), &value.strategy);
        point.append_value(Some(math), draw_point);
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
            point,
            mut diverging,
            divergence_start,
            divergence_start_grad,
            divergence_end,
            divergence_momentum,
            divergence_msg,
        } = self;

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

        fn merge_into<M: Math, T: ?Sized, B: StatTraceBuilder<M, T>>(
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
        merge_into(point, &mut arrays, &mut fields);

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
            point,
            diverging,
            divergence_start,
            divergence_start_grad,
            divergence_end,
            divergence_momentum,
            divergence_msg,
        } = self;

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

        fn merge_into<M: Math, T: ?Sized, B: StatTraceBuilder<M, T>>(
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
        merge_into(point, &mut arrays, &mut fields);

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
