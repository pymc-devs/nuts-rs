use arrow::{
    array::{ArrayBuilder, FixedSizeListBuilder, PrimitiveBuilder, StructArray},
    datatypes::{Field, Float64Type},
};

use crate::{
    euclidean_hamiltonian::EuclideanPoint,
    hamiltonian::Point,
    math_base::Math,
    nuts::Collector,
    sampler::Settings,
    sampler_stats::{SamplerStats, StatTraceBuilder},
    state::State,
};

pub trait MassMatrix<M: Math>: SamplerStats<M> {
    fn update_velocity(&self, math: &mut M, state: &mut EuclideanPoint<M>);
    fn update_kinetic_energy(&self, math: &mut M, state: &mut EuclideanPoint<M>);
    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        point: &mut EuclideanPoint<M>,
        rng: &mut R,
    );
}

pub struct NullCollector {}

impl<M: Math, P: Point<M>> Collector<M, P> for NullCollector {}

#[derive(Debug)]
pub struct DiagMassMatrix<M: Math> {
    inv_stds: M::Vector,
    pub(crate) variance: M::Vector,
    store_mass_matrix: bool,
}

pub struct DiagMassMatrixStatsBuilder {
    mass_matrix_inv: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
}

impl<M: Math> StatTraceBuilder<M, DiagMassMatrix<M>> for DiagMassMatrixStatsBuilder {
    fn append_value(&mut self, math: Option<&mut M>, value: &DiagMassMatrix<M>) {
        let math = math.expect("Need reference to math for stats");
        let Self { mass_matrix_inv } = self;

        if let Some(store) = mass_matrix_inv {
            let values = math.box_array(&value.variance);
            store.values().append_slice(&values);
            store.append(true);
        }
    }

    fn finalize(self) -> Option<StructArray> {
        let Self { mass_matrix_inv } = self;

        let array = ArrayBuilder::finish(&mut mass_matrix_inv?);

        let fields = vec![Field::new(
            "mass_matrix_inv",
            array.data_type().clone(),
            true,
        )];
        let arrays = vec![array];
        Some(StructArray::new(fields.into(), arrays, None))
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self { mass_matrix_inv } = self;

        let array = ArrayBuilder::finish_cloned(mass_matrix_inv.as_ref()?);
        let fields = vec![Field::new(
            "mass_matrix_inv",
            array.data_type().clone(),
            true,
        )];
        let arrays = vec![array];
        Some(StructArray::new(fields.into(), arrays, None))
    }
}

impl<M: Math> SamplerStats<M> for DiagMassMatrix<M> {
    type Builder = DiagMassMatrixStatsBuilder;
    type StatOptions = ();

    fn new_builder(
        &self,
        _stat_options: Self::StatOptions,
        _settings: &impl Settings,
        dim: usize,
    ) -> Self::Builder {
        if self.store_mass_matrix {
            let items = PrimitiveBuilder::new();
            let values = FixedSizeListBuilder::new(items, dim as _);
            Self::Builder {
                mass_matrix_inv: Some(values),
            }
        } else {
            Self::Builder {
                mass_matrix_inv: None,
            }
        }
    }
}

impl<M: Math> DiagMassMatrix<M> {
    pub(crate) fn new(math: &mut M, store_mass_matrix: bool) -> Self {
        Self {
            inv_stds: math.new_array(),
            variance: math.new_array(),
            store_mass_matrix,
        }
    }

    pub(crate) fn update_diag_draw(
        &mut self,
        math: &mut M,
        draw_var: &M::Vector,
        scale: f64,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_draw(
            &mut self.variance,
            &mut self.inv_stds,
            draw_var,
            scale,
            fill_invalid,
            clamp,
        );
    }

    pub(crate) fn update_diag_draw_grad(
        &mut self,
        math: &mut M,
        draw_var: &M::Vector,
        grad_var: &M::Vector,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_draw_grad(
            &mut self.variance,
            &mut self.inv_stds,
            draw_var,
            grad_var,
            fill_invalid,
            clamp,
        );
    }

    pub(crate) fn update_diag_grad(
        &mut self,
        math: &mut M,
        gradient: &M::Vector,
        fill_invalid: f64,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_grad(
            &mut self.variance,
            &mut self.inv_stds,
            gradient,
            fill_invalid,
            clamp,
        );
    }
}

impl<M: Math> MassMatrix<M> for DiagMassMatrix<M> {
    fn update_velocity(&self, math: &mut M, point: &mut EuclideanPoint<M>) {
        math.array_mult(&self.variance, &point.momentum, &mut point.velocity);
    }

    fn update_kinetic_energy(&self, math: &mut M, point: &mut EuclideanPoint<M>) {
        point.kinetic_energy = 0.5 * math.array_vector_dot(&point.momentum, &point.velocity);
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        point: &mut EuclideanPoint<M>,
        rng: &mut R,
    ) {
        math.array_gaussian(rng, &mut point.momentum, &self.inv_stds);
    }
}

#[derive(Debug)]
pub struct RunningVariance<M: Math> {
    mean: M::Vector,
    variance: M::Vector,
    count: u64,
}

impl<M: Math> RunningVariance<M> {
    pub(crate) fn new(math: &mut M) -> Self {
        Self {
            mean: math.new_array(),
            variance: math.new_array(),
            count: 0,
        }
    }

    pub(crate) fn add_sample(&mut self, math: &mut M, value: &M::Vector) {
        self.count += 1;
        if self.count == 1 {
            math.copy_into(value, &mut self.mean);
        } else {
            math.array_update_variance(
                &mut self.mean,
                &mut self.variance,
                value,
                (self.count as f64).recip(),
            );
        }
    }

    /// Return current variance and scaling factor
    pub(crate) fn current(&self) -> (&M::Vector, f64) {
        assert!(self.count > 1);
        (&self.variance, ((self.count - 1) as f64).recip())
    }

    pub(crate) fn count(&self) -> u64 {
        self.count
    }
}

pub struct DrawGradCollector<M: Math> {
    pub(crate) draw: M::Vector,
    pub(crate) grad: M::Vector,
    pub(crate) is_good: bool,
}

impl<M: Math> DrawGradCollector<M> {
    pub(crate) fn new(math: &mut M) -> Self {
        DrawGradCollector {
            draw: math.new_array(),
            grad: math.new_array(),
            is_good: true,
        }
    }
}

impl<M: Math> Collector<M, EuclideanPoint<M>> for DrawGradCollector<M> {
    fn register_draw(
        &mut self,
        math: &mut M,
        state: &State<M, EuclideanPoint<M>>,
        info: &crate::nuts::SampleInfo,
    ) {
        math.copy_into(state.point().position(), &mut self.draw);
        math.copy_into(state.point().gradient(), &mut self.grad);
        let idx = state.index_in_trajectory();
        if info.divergence_info.is_some() {
            self.is_good = idx.abs() > 4;
        } else {
            self.is_good = idx != 0;
        }
    }
}
