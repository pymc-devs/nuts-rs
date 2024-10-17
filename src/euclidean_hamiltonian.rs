use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{ArrayBuilder, Float64Builder, StructArray};
use arrow::datatypes::{DataType, Field};

use crate::hamiltonian::{Direction, DivergenceInfo, Hamiltonian, LeapfrogResult, Point};
use crate::mass_matrix::MassMatrix;
use crate::math_base::Math;
use crate::nuts::{Collector, NutsError};
use crate::sampler::Settings;
use crate::sampler_stats::{SamplerStats, StatTraceBuilder};
use crate::state::{State, StatePool};
use crate::LogpError;

pub struct EuclideanHamiltonian<M: Math, Mass: MassMatrix<M>> {
    pub(crate) mass_matrix: Mass,
    max_energy_error: f64,
    step_size: f64,
    pool: StatePool<M, EuclideanPoint<M>>,
    _phantom: PhantomData<M>,
}

impl<M: Math, Mass: MassMatrix<M>> EuclideanHamiltonian<M, Mass> {
    pub(crate) fn new(
        math: &mut M,
        mass_matrix: Mass,
        max_energy_error: f64,
        step_size: f64,
    ) -> Self {
        let pool = StatePool::new(math, 10);
        EuclideanHamiltonian {
            mass_matrix,
            max_energy_error,
            step_size,
            pool,
            _phantom: PhantomData,
        }
    }
}

pub struct EuclideanPoint<M: Math> {
    pub position: M::Vector,
    pub velocity: M::Vector,
    pub gradient: M::Vector,
    pub momentum: M::Vector,
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    pub index_in_trajectory: i64,
    pub p_sum: M::Vector,
    pub initial_energy: f64,
}

impl<M: Math> EuclideanPoint<M> {
    fn is_turning(&self, math: &mut M, other: &Self) -> bool {
        let (start, end) = if self.index_in_trajectory() < other.index_in_trajectory() {
            (self, other)
        } else {
            (other, self)
        };

        let a = start.index_in_trajectory();
        let b = end.index_in_trajectory();

        assert!(a < b);
        let (turn1, turn2) = if (a >= 0) & (b >= 0) {
            math.scalar_prods3(
                &end.p_sum,
                &start.p_sum,
                &start.momentum,
                &end.velocity,
                &start.velocity,
            )
        } else if (b >= 0) & (a < 0) {
            math.scalar_prods2(&end.p_sum, &start.p_sum, &end.velocity, &start.velocity)
        } else {
            assert!((a < 0) & (b < 0));
            math.scalar_prods3(
                &start.p_sum,
                &end.p_sum,
                &end.momentum,
                &end.velocity,
                &start.velocity,
            )
        };

        (turn1 < 0.) | (turn2 < 0.)
    }

    fn first_momentum_halfstep(&self, math: &mut M, out: &mut Self, epsilon: f64) {
        math.axpy_out(
            &self.gradient,
            &self.momentum,
            epsilon / 2.,
            &mut out.momentum,
        );
    }

    fn position_step(&self, math: &mut M, out: &mut Self, epsilon: f64) {
        math.axpy_out(&out.velocity, &self.position, epsilon, &mut out.position);
    }

    fn second_momentum_halfstep(&mut self, math: &mut M, epsilon: f64) {
        math.axpy(&self.gradient, &mut self.momentum, epsilon / 2.);
    }

    fn set_psum(&self, math: &mut M, out: &mut Self, _dir: Direction) {
        assert!(out.index_in_trajectory != 0);

        if out.index_in_trajectory == -1 {
            math.copy_into(&out.momentum, &mut out.p_sum);
        } else {
            math.axpy_out(&out.momentum, &self.p_sum, 1., &mut out.p_sum);
        }
    }

    fn update_potential_gradient(&mut self, math: &mut M) -> Result<(), M::LogpErr> {
        let logp = { math.logp_array(&self.position, &mut self.gradient) }?;
        self.potential_energy = -logp;
        Ok(())
    }
}

impl<M: Math> Point<M> for EuclideanPoint<M> {
    fn position(&self) -> &<M as Math>::Vector {
        &self.position
    }

    fn gradient(&self) -> &<M as Math>::Vector {
        &self.gradient
    }

    fn energy(&self) -> f64 {
        self.potential_energy + self.kinetic_energy
    }

    fn initial_energy(&self) -> f64 {
        self.initial_energy
    }

    fn new(math: &mut M) -> Self {
        Self {
            position: math.new_array(),
            velocity: math.new_array(),
            gradient: math.new_array(),
            momentum: math.new_array(),
            kinetic_energy: 0f64,
            potential_energy: 0f64,
            index_in_trajectory: 0,
            p_sum: math.new_array(),
            initial_energy: 0f64,
        }
    }

    fn index_in_trajectory(&self) -> i64 {
        self.index_in_trajectory
    }

    fn logp(&self) -> f64 {
        -self.potential_energy
    }

    fn copy_into(&self, math: &mut M, other: &mut Self) {
        let Self {
            position,
            velocity,
            gradient,
            momentum,
            kinetic_energy,
            potential_energy,
            index_in_trajectory,
            p_sum,
            initial_energy,
        } = self;
        math.copy_into(position, &mut other.position);
        math.copy_into(velocity, &mut other.velocity);
        math.copy_into(gradient, &mut other.gradient);
        math.copy_into(momentum, &mut other.momentum);
        math.copy_into(p_sum, &mut other.p_sum);
        other.kinetic_energy = *kinetic_energy;
        other.potential_energy = *potential_energy;
        other.initial_energy = *initial_energy;
        other.index_in_trajectory = *index_in_trajectory;
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PotentialStats<S: Clone + Debug> {
    mass_matrix_stats: S,
    pub step_size: f64,
}

pub struct PotentialStatsBuilder<B> {
    mass_matrix: B,
    step_size: Float64Builder,
}

impl<S: Clone + Debug, B: StatTraceBuilder<S>> StatTraceBuilder<PotentialStats<S>>
    for PotentialStatsBuilder<B>
{
    fn append_value(&mut self, value: PotentialStats<S>) {
        let PotentialStats {
            mass_matrix_stats,
            step_size,
        } = value;

        self.mass_matrix.append_value(mass_matrix_stats);
        self.step_size.append_value(step_size);
    }

    fn finalize(self) -> Option<StructArray> {
        let Self {
            mass_matrix,
            mut step_size,
        } = self;

        let mut fields = vec![Field::new("step_size", DataType::Float64, false)];
        let mut arrays = vec![ArrayBuilder::finish(&mut step_size)];

        if let Some(mass_matrix) = mass_matrix.finalize() {
            let (m_fields, m_data, m_bitmap) = mass_matrix.into_parts();
            assert!(m_bitmap.is_none());
            fields.extend(
                m_fields
                    .into_iter()
                    .map(|v| Arc::unwrap_or_clone(v.to_owned())),
            );
            arrays.extend(m_data);
        }

        Some(StructArray::new(fields.into(), arrays, None))
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self {
            mass_matrix,
            step_size,
        } = self;

        let mut fields = vec![Field::new("step_size", DataType::Float64, false)];
        let mut arrays = vec![ArrayBuilder::finish_cloned(step_size)];

        if let Some(mass_matrix) = mass_matrix.inspect() {
            let (m_fields, m_data, m_bitmap) = mass_matrix.into_parts();
            assert!(m_bitmap.is_none());
            fields.extend(
                m_fields
                    .into_iter()
                    .map(|v| Arc::unwrap_or_clone(v.to_owned())),
            );
            arrays.extend(m_data);
        }

        Some(StructArray::new(fields.into(), arrays, None))
    }
}

impl<M: Math, Mass: MassMatrix<M>> SamplerStats<M> for EuclideanHamiltonian<M, Mass> {
    type Builder = PotentialStatsBuilder<Mass::Builder>;
    type Stats = PotentialStats<Mass::Stats>;

    fn new_builder(&self, settings: &impl Settings, dim: usize) -> Self::Builder {
        Self::Builder {
            mass_matrix: self.mass_matrix.new_builder(settings, dim),
            step_size: Float64Builder::with_capacity(
                settings.hint_num_draws() + settings.hint_num_tune(),
            ),
        }
    }

    fn current_stats(&self, math: &mut M) -> Self::Stats {
        PotentialStats {
            mass_matrix_stats: self.mass_matrix.current_stats(math),
            step_size: self.step_size,
        }
    }
}

impl<M: Math, Mass: MassMatrix<M>> Hamiltonian<M> for EuclideanHamiltonian<M, Mass> {
    type Point = EuclideanPoint<M>;

    fn leapfrog<C: Collector<M, Self::Point>>(
        &mut self,
        math: &mut M,
        start: &State<M, Self::Point>,
        dir: Direction,
        collector: &mut C,
    ) -> LeapfrogResult<M, Self::Point> {
        let mut out = self.pool().new_state(math);
        let out_point = out.try_point_mut().expect("New point has other references");

        out_point.initial_energy = start.point().initial_energy();

        let sign = match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };

        let epsilon = (sign as f64) * self.step_size;

        start
            .point()
            .first_momentum_halfstep(math, out_point, epsilon);
        self.mass_matrix.update_velocity(math, out_point);

        start.point().position_step(math, out_point, epsilon);
        if let Err(logp_error) = out_point.update_potential_gradient(math) {
            if !logp_error.is_recoverable() {
                return LeapfrogResult::Err(logp_error);
            }
            let div_info = DivergenceInfo {
                logp_function_error: Some(Arc::new(Box::new(logp_error))),
                start_location: Some(math.box_array(start.point().position())),
                start_gradient: Some(math.box_array(&start.point().gradient)),
                start_momentum: Some(math.box_array(&start.point().momentum)),
                end_location: None,
                start_idx_in_trajectory: Some(start.point().index_in_trajectory()),
                end_idx_in_trajectory: None,
                energy_error: None,
            };
            collector.register_leapfrog(math, start, &out, Some(&div_info));
            return LeapfrogResult::Divergence(div_info);
        }

        out_point.second_momentum_halfstep(math, epsilon);

        self.mass_matrix.update_velocity(math, out_point);
        self.mass_matrix.update_kinetic_energy(math, out_point);

        out_point.index_in_trajectory = start.index_in_trajectory() + sign;

        start.point().set_psum(math, out_point, dir);

        let energy_error = out_point.energy_error();
        if (energy_error > self.max_energy_error) | !energy_error.is_finite() {
            let divergence_info = DivergenceInfo {
                logp_function_error: None,
                start_location: Some(math.box_array(start.point().position())),
                start_gradient: Some(math.box_array(start.point().gradient())),
                end_location: Some(math.box_array(&out_point.position)),
                start_momentum: Some(math.box_array(&out_point.momentum)),
                start_idx_in_trajectory: Some(start.index_in_trajectory()),
                end_idx_in_trajectory: Some(out.index_in_trajectory()),
                energy_error: Some(energy_error),
            };
            collector.register_leapfrog(math, start, &out, Some(&divergence_info));
            return LeapfrogResult::Divergence(divergence_info);
        }

        collector.register_leapfrog(math, start, &out, None);

        LeapfrogResult::Ok(out)
    }

    fn init_state(
        &mut self,
        math: &mut M,
        init: &[f64],
    ) -> Result<State<M, Self::Point>, NutsError> {
        let mut state = self.pool().new_state(math);
        let point = state.try_point_mut().expect("State already in use");
        math.read_from_slice(&mut point.position, init);
        math.fill_array(&mut point.p_sum, 0.);

        point
            .update_potential_gradient(math)
            .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;
        if !math.array_all_finite_and_nonzero(&point.gradient) {
            Err(NutsError::BadInitGrad())
        } else {
            Ok(state)
        }
    }

    fn initialize_trajectory<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut State<M, Self::Point>,
        rng: &mut R,
    ) -> Result<(), NutsError> {
        let inner = state.try_point_mut().expect("State has other references");
        self.mass_matrix.randomize_momentum(math, inner, rng);
        self.mass_matrix.update_velocity(math, inner);
        self.mass_matrix.update_kinetic_energy(math, inner);
        inner.index_in_trajectory = 0;
        inner.initial_energy = inner.energy();
        math.copy_into(&inner.momentum, &mut inner.p_sum);
        Ok(())
    }

    fn is_turning(
        &self,
        math: &mut M,
        state1: &State<M, Self::Point>,
        state2: &State<M, Self::Point>,
    ) -> bool {
        state1.point().is_turning(math, state2.point())
    }

    fn copy_state(&mut self, math: &mut M, state: &State<M, Self::Point>) -> State<M, Self::Point> {
        let mut new_state = self.pool().new_state(math);
        state.point().copy_into(
            math,
            new_state
                .try_point_mut()
                .expect("New point should not have other references"),
        );
        new_state
    }

    fn pool(&mut self) -> &mut StatePool<M, Self::Point> {
        &mut self.pool
    }

    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn step_size_mut(&mut self) -> &mut f64 {
        &mut self.step_size
    }
}
