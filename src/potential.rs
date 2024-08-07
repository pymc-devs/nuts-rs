use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{ArrayBuilder, PrimitiveBuilder, StructArray};
use arrow::datatypes::{DataType, Field, Float64Type};

use crate::mass_matrix::MassMatrix;
use crate::math_base::Math;
use crate::nuts::{
    Collector, Direction, DivergenceInfo, Hamiltonian, HamiltonianStats, LogpError, NutsError,
};
use crate::nuts::{SamplerStats, StatTraceBuilder};
use crate::sampler::Settings;
use crate::state::{State, StatePool};

pub struct EuclideanPotential<M: Math, Mass: MassMatrix<M>> {
    pub(crate) mass_matrix: Mass,
    max_energy_error: f64,
    pub(crate) step_size: f64,
    _phantom: PhantomData<M>,
}

impl<M: Math, Mass: MassMatrix<M>> EuclideanPotential<M, Mass> {
    pub(crate) fn new(mass_matrix: Mass, max_energy_error: f64, step_size: f64) -> Self {
        EuclideanPotential {
            mass_matrix,
            max_energy_error,
            step_size,
            _phantom: PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PotentialStats<S: Clone + Debug> {
    step_size: f64,
    mass_matrix_stats: S,
}

pub struct PotentialStatsBuilder<B> {
    step_size: PrimitiveBuilder<Float64Type>,
    mass_matrix: B,
}

impl<S: Clone + Debug, B: StatTraceBuilder<S>> StatTraceBuilder<PotentialStats<S>>
    for PotentialStatsBuilder<B>
{
    fn append_value(&mut self, value: PotentialStats<S>) {
        let PotentialStats {
            step_size,
            mass_matrix_stats,
        } = value;

        self.step_size.append_value(step_size);
        self.mass_matrix.append_value(mass_matrix_stats)
    }

    fn finalize(self) -> Option<StructArray> {
        let Self {
            mut step_size,
            mass_matrix,
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
            step_size,
            mass_matrix,
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

impl<M: Math, Mass: MassMatrix<M>> SamplerStats<M> for EuclideanPotential<M, Mass> {
    type Builder = PotentialStatsBuilder<Mass::Builder>;
    type Stats = PotentialStats<Mass::Stats>;

    fn new_builder(&self, settings: &impl Settings, dim: usize) -> Self::Builder {
        Self::Builder {
            step_size: PrimitiveBuilder::new(),
            mass_matrix: self.mass_matrix.new_builder(settings, dim),
        }
    }

    fn current_stats(&self, math: &mut M) -> Self::Stats {
        PotentialStats {
            step_size: self.step_size,
            mass_matrix_stats: self.mass_matrix.current_stats(math),
        }
    }
}

impl<M: Math, Mass: MassMatrix<M>> HamiltonianStats<M> for EuclideanPotential<M, Mass> {
    fn stat_step_size(stats: &Self::Stats) -> f64 {
        stats.step_size
    }
}

impl<M: Math, Mass: MassMatrix<M>> Hamiltonian<M> for EuclideanPotential<M, Mass> {
    type LogpError = M::LogpErr;

    fn leapfrog<C: Collector<M>>(
        &mut self,
        math: &mut M,
        pool: &mut StatePool<M>,
        start: &State<M>,
        dir: Direction,
        initial_energy: f64,
        collector: &mut C,
    ) -> Result<Result<State<M>, DivergenceInfo>, NutsError> {
        let mut out = pool.new_state(math);

        let sign = match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };

        let epsilon = (sign as f64) * self.step_size;

        start.first_momentum_halfstep(math, &mut out, epsilon);
        self.update_velocity(math, &mut out);

        start.position_step(math, &mut out, epsilon);
        if let Err(logp_error) = self.update_potential_gradient(math, &mut out) {
            if !logp_error.is_recoverable() {
                return Err(NutsError::LogpFailure(Box::new(logp_error)));
            }
            let div_info = DivergenceInfo {
                logp_function_error: Some(Arc::new(Box::new(logp_error))),
                start_location: Some(math.box_array(&start.q)),
                start_gradient: Some(math.box_array(&start.grad)),
                start_momentum: Some(math.box_array(&start.p)),
                end_location: None,
                start_idx_in_trajectory: Some(start.idx_in_trajectory),
                end_idx_in_trajectory: None,
                energy_error: None,
            };
            collector.register_leapfrog(math, start, &out, Some(&div_info));
            return Ok(Err(div_info));
        }

        out.second_momentum_halfstep(math, epsilon);

        self.update_velocity(math, &mut out);
        self.update_kinetic_energy(math, &mut out);

        *out.index_in_trajectory_mut() = start.index_in_trajectory() + sign;

        start.set_psum(math, &mut out, dir);

        let energy_error = { out.energy() - initial_energy };
        if (energy_error > self.max_energy_error) | !energy_error.is_finite() {
            let divergence_info = DivergenceInfo {
                logp_function_error: None,
                start_location: Some(math.box_array(&start.q)),
                start_gradient: Some(math.box_array(&start.grad)),
                end_location: Some(math.box_array(&out.q)),
                start_momentum: Some(math.box_array(&out.p)),
                start_idx_in_trajectory: Some(start.index_in_trajectory()),
                end_idx_in_trajectory: Some(out.index_in_trajectory()),
                energy_error: Some(energy_error),
            };
            collector.register_leapfrog(math, start, &out, Some(&divergence_info));
            return Ok(Err(divergence_info));
        }

        collector.register_leapfrog(math, start, &out, None);

        Ok(Ok(out))
    }

    fn init_state(
        &mut self,
        math: &mut M,
        pool: &mut StatePool<M>,
        init: &[f64],
    ) -> Result<State<M>, NutsError> {
        let mut state = pool.new_state(math);
        {
            let inner = state.try_mut_inner().expect("State already in use");
            math.read_from_slice(&mut inner.q, init);
            math.fill_array(&mut inner.p_sum, 0.);
        }
        self.update_potential_gradient(math, &mut state)
            .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;
        if !math.array_all_finite_and_nonzero(&state.grad) {
            Err(NutsError::BadInitGrad())
        } else {
            Ok(state)
        }
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut State<M>,
        rng: &mut R,
    ) {
        let inner = state.try_mut_inner().unwrap();
        self.mass_matrix.randomize_momentum(math, inner, rng);
        self.mass_matrix.update_velocity(math, inner);
        self.mass_matrix.update_kinetic_energy(math, inner);
    }

    fn new_empty_state(&mut self, math: &mut M, pool: &mut StatePool<M>) -> State<M> {
        pool.new_state(math)
    }

    fn new_pool(&mut self, math: &mut M, capacity: usize) -> StatePool<M> {
        StatePool::new(math, capacity)
    }

    fn copy_state(&mut self, math: &mut M, pool: &mut StatePool<M>, state: &State<M>) -> State<M> {
        pool.copy_state(math, state)
    }

    fn stepsize_mut(&mut self) -> &mut f64 {
        &mut self.step_size
    }

    fn stepsize(&self) -> f64 {
        self.step_size
    }
}

impl<M: Math, Mass: MassMatrix<M>> EuclideanPotential<M, Mass> {
    fn update_potential_gradient(
        &mut self,
        math: &mut M,
        state: &mut State<M>,
    ) -> Result<(), M::LogpErr> {
        let logp = {
            let inner = state.try_mut_inner().unwrap();
            math.logp_array(&inner.q, &mut inner.grad)
        }?;

        let inner = state.try_mut_inner().unwrap();
        inner.potential_energy = -logp;
        Ok(())
    }

    fn update_velocity(&mut self, math: &mut M, state: &mut State<M>) {
        self.mass_matrix
            .update_velocity(math, state.try_mut_inner().expect("State already in us"))
    }

    fn update_kinetic_energy(&mut self, math: &mut M, state: &mut State<M>) {
        self.mass_matrix
            .update_kinetic_energy(math, state.try_mut_inner().expect("State already in us"))
    }
}
