use std::fmt::Debug;

use crate::cpu_state::{InnerState, State, StatePool};
use crate::mass_matrix::MassMatrix;
use crate::nuts::{
    AsSampleStatVec, Collector, Direction, DivergenceInfo, Hamiltonian, LogpError, NutsError,
};

/// Compute the unnormalized log probability density of the posterior
///
/// This needs to be implemnted by users of the library to define
/// what distribution the users wants to sample from.
///
/// Errors during that computation can be recoverable or non-recoverable.
/// If a non-recoverable error occurs during sampling, the sampler will
/// stop and return an error.
pub trait CpuLogpFunc {
    type Err: Debug + Send + LogpError + 'static;

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::Err>;
    fn dim(&self) -> usize;
}

#[derive(Debug)]
pub(crate) struct DivergenceInfoImpl<E: Send + std::error::Error> {
    logp_function_error: Option<E>,
    start: Option<InnerState>,
    end: Option<InnerState>,
    energy_error: Option<f64>,
}

impl<E: Debug + Send + std::error::Error> DivergenceInfo for DivergenceInfoImpl<E> {
    fn start_location(&self) -> Option<&[f64]> {
        Some(&self.start.as_ref()?.q)
    }

    fn end_location(&self) -> Option<&[f64]> {
        Some(&self.end.as_ref()?.q)
    }

    fn energy_error(&self) -> Option<f64> {
        self.energy_error
    }

    fn end_idx_in_trajectory(&self) -> Option<i64> {
        Some(self.end.as_ref()?.idx_in_trajectory)
    }

    fn start_idx_in_trajectory(&self) -> Option<i64> {
        Some(self.end.as_ref()?.idx_in_trajectory)
    }

    fn logp_function_error(&self) -> Option<&dyn std::error::Error> {
        self.logp_function_error
            .as_ref()
            .map(|x| x as &dyn std::error::Error)
    }
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

impl AsSampleStatVec for PotentialStats {
    fn add_to_vec(&self, vec: &mut Vec<crate::nuts::SampleStatItem>) {
        vec.push(("step_size", self.step_size.into()));
    }
}

impl<F: CpuLogpFunc, M: MassMatrix> Hamiltonian for EuclideanPotential<F, M> {
    type State = State;
    type DivergenceInfo = DivergenceInfoImpl<F::Err>;
    type LogpError = F::Err;
    type Stats = PotentialStats;

    fn leapfrog<C: Collector<State = Self::State>>(
        &mut self,
        pool: &mut StatePool,
        start: &Self::State,
        dir: Direction,
        initial_energy: f64,
        collector: &mut C,
    ) -> Result<Result<Self::State, Self::DivergenceInfo>, NutsError> {
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
            let div_info = DivergenceInfoImpl {
                logp_function_error: Some(logp_error),
                start: Some(start.clone_inner()),
                end: None,
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
            let divergence_info = DivergenceInfoImpl {
                logp_function_error: None,
                start: Some(start.clone_inner()),
                end: Some(out.clone_inner()),
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
        Ok(state)
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
