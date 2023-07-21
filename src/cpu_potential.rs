use std::fmt::Debug;

#[cfg(feature = "arrow")]
use arrow2::array::{MutableArray, MutablePrimitiveArray, StructArray};
#[cfg(feature = "arrow")]
use arrow2::datatypes::{DataType, Field};

use crate::cpu_state::{State, StatePool};
use crate::mass_matrix::MassMatrix;
use crate::nuts::{Collector, Direction, DivergenceInfo, Hamiltonian, LogpError, NutsError};
#[cfg(feature = "arrow")]
use crate::SamplerArgs;

#[cfg(feature = "arrow")]
use crate::nuts::{ArrowBuilder, ArrowRow};

/// Compute the unnormalized log probability density of the posterior
///
/// This needs to be implemnted by users of the library to define
/// what distribution the users wants to sample from.
///
/// Errors during that computation can be recoverable or non-recoverable.
/// If a non-recoverable error occurs during sampling, the sampler will
/// stop and return an error.
pub trait CpuLogpFunc {
    type Err: Debug + Send + Sync + LogpError + 'static;

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::Err>;
    fn dim(&self) -> usize;
}

pub(crate) struct EuclideanPotential<F: CpuLogpFunc, M: MassMatrix> {
    logp: F,
    pub(crate) mass_matrix: M,
    max_energy_error: f64,
    pub(crate) step_size: f64,
}

impl<F: CpuLogpFunc, M: MassMatrix> EuclideanPotential<F, M> {
    pub(crate) fn new(logp: F, mass_matrix: M, max_energy_error: f64, step_size: f64) -> Self {
        EuclideanPotential {
            logp,
            mass_matrix,
            max_energy_error,
            step_size,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct PotentialStats {
    step_size: f64,
}

#[cfg(feature = "arrow")]
pub(crate) struct PotentialStatsBuilder {
    step_size: MutablePrimitiveArray<f64>,
}

#[cfg(feature = "arrow")]
impl ArrowBuilder<PotentialStats> for PotentialStatsBuilder {
    fn append_value(&mut self, value: &PotentialStats) {
        self.step_size.push(Some(value.step_size));
    }

    fn finalize(mut self) -> Option<StructArray> {
        let fields = vec![Field::new("step_size", DataType::Float64, false)];

        let arrays = vec![self.step_size.as_box()];

        Some(StructArray::new(DataType::Struct(fields), arrays, None))
    }
}

#[cfg(feature = "arrow")]
impl ArrowRow for PotentialStats {
    type Builder = PotentialStatsBuilder;

    fn new_builder(_dim: usize, _settings: &SamplerArgs) -> Self::Builder {
        Self::Builder {
            step_size: MutablePrimitiveArray::new(),
        }
    }
}

impl<F: CpuLogpFunc, M: MassMatrix> Hamiltonian for EuclideanPotential<F, M> {
    type State = State;
    type LogpError = F::Err;
    type Stats = PotentialStats;

    fn leapfrog<C: Collector<State = Self::State>>(
        &mut self,
        pool: &mut StatePool,
        start: &Self::State,
        dir: Direction,
        initial_energy: f64,
        collector: &mut C,
    ) -> Result<Result<Self::State, DivergenceInfo>, NutsError> {
        let mut out = pool.new_state();

        let sign = match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };

        let epsilon = (sign as f64) * self.step_size;

        start.first_momentum_halfstep(&mut out, epsilon);
        self.update_velocity(&mut out);

        start.position_step(&mut out, epsilon);
        if let Err(logp_error) = self.update_potential_gradient(&mut out) {
            if !logp_error.is_recoverable() {
                return Err(NutsError::LogpFailure(Box::new(logp_error)));
            }
            let div_info = DivergenceInfo {
                logp_function_error: Some(Box::new(logp_error)),
                start_location: Some(start.q.clone()),
                end_location: None,
                start_idx_in_trajectory: Some(start.idx_in_trajectory),
                end_idx_in_trajectory: None,
                energy_error: None,
            };
            collector.register_leapfrog(start, &out, Some(&div_info));
            return Ok(Err(div_info));
        }

        out.second_momentum_halfstep(epsilon);

        self.update_velocity(&mut out);
        self.update_kinetic_energy(&mut out);

        *out.index_in_trajectory_mut() = start.index_in_trajectory() + sign;

        start.set_psum(&mut out, dir);

        let energy_error = {
            use crate::nuts::State;
            out.energy() - initial_energy
        };
        if (energy_error > self.max_energy_error) | !energy_error.is_finite() {
            let divergence_info = DivergenceInfo {
                logp_function_error: None,
                start_location: Some(start.q.clone()),
                end_location: Some(out.q.clone()),
                start_idx_in_trajectory: Some(start.index_in_trajectory()),
                end_idx_in_trajectory: Some(out.index_in_trajectory()),
                energy_error: Some(energy_error),
            };
            collector.register_leapfrog(start, &out, Some(&divergence_info));
            return Ok(Err(divergence_info));
        }

        collector.register_leapfrog(start, &out, None);

        Ok(Ok(out))
    }

    fn init_state(&mut self, pool: &mut StatePool, init: &[f64]) -> Result<Self::State, NutsError> {
        let mut state = pool.new_state();
        {
            let inner = state.try_mut_inner().expect("State already in use");
            inner.q.copy_from_slice(init);
            inner.p_sum.fill(0.);
        }
        self.update_potential_gradient(&mut state)
            .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;
        if state
            .grad
            .iter()
            .cloned()
            .any(|val| (val == 0f64) | !val.is_finite())
        {
            Err(NutsError::BadInitGrad())
        } else {
            Ok(state)
        }
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut Self::State, rng: &mut R) {
        let inner = state.try_mut_inner().unwrap();
        self.mass_matrix.randomize_momentum(inner, rng);
        self.mass_matrix.update_velocity(inner);
        self.mass_matrix.update_kinetic_energy(inner);
    }

    fn current_stats(&self) -> Self::Stats {
        PotentialStats {
            step_size: self.step_size,
        }
    }

    fn new_empty_state(&mut self, pool: &mut StatePool) -> Self::State {
        pool.new_state()
    }

    fn new_pool(&mut self, _capacity: usize) -> StatePool {
        StatePool::new(self.dim())
    }

    fn dim(&self) -> usize {
        self.logp.dim()
    }
}

impl<F: CpuLogpFunc, M: MassMatrix> EuclideanPotential<F, M> {
    fn update_potential_gradient(&mut self, state: &mut State) -> Result<(), F::Err> {
        let logp = {
            let inner = state.try_mut_inner().unwrap();
            self.logp.logp(&inner.q, &mut inner.grad)
        }?;

        let inner = state.try_mut_inner().unwrap();
        inner.potential_energy = -logp;
        Ok(())
    }

    fn update_velocity(&mut self, state: &mut State) {
        self.mass_matrix
            .update_velocity(state.try_mut_inner().expect("State already in us"))
    }

    fn update_kinetic_energy(&mut self, state: &mut State) {
        self.mass_matrix
            .update_kinetic_energy(state.try_mut_inner().expect("State already in us"))
    }
}
