use std::fmt::Debug;

use crate::cpu_state::{InnerState, State, StatePool};
use crate::mass_matrix::MassMatrix;
use crate::nuts::{Collector, Direction, DivergenceInfo, LogpError, NutsError, Potential};

pub trait CpuLogpFunc {
    type Err: Debug + Send + LogpError + 'static;

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::Err>;
    fn dim(&self) -> usize;
}

#[derive(Debug)]
pub struct DivergenceInfoImpl<E: Send> {
    pub logp_function_error: Option<E>,
    start: Option<InnerState>,
    right: Option<InnerState>,
    energy_error: Option<f64>,
}

impl<E: Debug + Send> DivergenceInfo for DivergenceInfoImpl<E> {
    fn start_location(&self) -> Option<&[f64]> {
        Some(&self.start.as_ref()?.q)
    }

    fn end_location(&self) -> Option<&[f64]> {
        Some(&self.right.as_ref()?.q)
    }

    fn energy_error(&self) -> Option<f64> {
        Some(self.energy_error?)
    }

    fn end_idx_in_trajectory(&self) -> Option<i64> {
        Some(self.right.as_ref()?.idx_in_trajectory)
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

    pub(crate) fn mass_matrix_mut(&mut self) -> &mut M {
        &mut self.mass_matrix
    }
}

#[derive(Copy, Clone)]
pub(crate) struct PotentialStats {}

impl<F: CpuLogpFunc, M: MassMatrix> Potential for EuclideanPotential<F, M> {
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
                right: None,
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
        if (energy_error.abs() > self.max_energy_error) | !energy_error.is_finite() {
            let divergence_info =
                self.new_divergence_info(start.clone(), out.clone(), energy_error);
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
        inner.idx_in_trajectory = 0;
        inner.p_sum.copy_from_slice(&inner.p);
    }

    fn current_stats(&self) -> Self::Stats {
        PotentialStats {}
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

    fn new_divergence_info(
        &mut self,
        left: State,
        end: State,
        energy_error: f64,
    ) -> DivergenceInfoImpl<F::Err> {
        DivergenceInfoImpl {
            logp_function_error: None,
            start: Some(left.clone_inner()),
            right: Some(end.clone_inner()),
            energy_error: Some(energy_error),
        }
    }
}
