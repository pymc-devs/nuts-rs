use std::{marker::PhantomData, sync::Arc};

use arrow::array::StructArray;

use crate::{
    hamiltonian::{Direction, Hamiltonian, LeapfrogResult, Point},
    sampler_stats::{SamplerStats, StatTraceBuilder},
    state::{State, StatePool},
    DivergenceInfo, LogpError, Math, NutsError, Settings,
};

pub struct TransformedPoint<M: Math> {
    untransformed_position: M::Vector,
    untransformed_gradient: M::Vector,
    transformed_position: M::Vector,
    transformed_gradient: M::Vector,
    velocity: M::Vector,
    index_in_trajectory: i64,
    logp: f64,
    logdet: f64,
    kinetic_energy: f64,
    initial_energy: f64,
    transform_id: i64,
}

impl<M: Math> TransformedPoint<M> {
    fn first_velocity_halfstep(&self, math: &mut M, out: &mut Self, epsilon: f64) {
        math.axpy_out(
            &self.transformed_gradient,
            &self.velocity,
            epsilon / 2.,
            &mut out.velocity,
        );
    }

    fn position_step(&self, math: &mut M, out: &mut Self, epsilon: f64) {
        math.axpy_out(
            &out.velocity,
            &self.transformed_position,
            epsilon,
            &mut out.transformed_position,
        );
    }

    fn second_velocity_halfstep(&mut self, math: &mut M, epsilon: f64) {
        math.axpy(&self.transformed_gradient, &mut self.velocity, epsilon / 2.);
    }

    fn update_kinetic_energy(&mut self, math: &mut M) {
        self.kinetic_energy = math.array_vector_dot(&self.velocity, &self.velocity);
    }

    fn update_gradient(
        &mut self,
        hamiltonian: &TransformedHamiltonian<M>,
        math: &mut M,
    ) -> Result<(), M::LogpErr> {
        let (logp, logdet) = {
            math.transformed_logp(
                hamiltonian.params.as_ref().expect("No transformation set"),
                &self.untransformed_position,
                &mut self.untransformed_gradient,
                &mut self.transformed_position,
                &mut self.transformed_gradient,
            )
        }?;
        self.logp = logp;
        self.logdet = logdet;
        Ok(())
    }

    fn is_valid(&self, math: &mut M) -> bool {
        if !math.array_all_finite(&self.transformed_position) {
            return false;
        }
        if !math.array_all_finite_and_nonzero(&self.transformed_gradient) {
            return false;
        }
        if !math.array_all_finite(&self.untransformed_gradient) {
            return false;
        }
        if !math.array_all_finite(&self.untransformed_position) {
            return false;
        }

        true
    }
}

impl<M: Math> Point<M> for TransformedPoint<M> {
    fn position(&self) -> &<M as Math>::Vector {
        &self.untransformed_position
    }

    fn gradient(&self) -> &<M as Math>::Vector {
        &self.untransformed_gradient
    }

    fn index_in_trajectory(&self) -> i64 {
        self.index_in_trajectory
    }

    fn energy(&self) -> f64 {
        self.kinetic_energy - self.logp - self.logdet
    }

    fn energy_error(&self) -> f64 {
        self.energy() - self.initial_energy
    }

    fn logp(&self) -> f64 {
        self.logp
    }

    fn new(math: &mut M) -> Self {
        Self {
            untransformed_position: math.new_array(),
            untransformed_gradient: math.new_array(),
            transformed_position: math.new_array(),
            transformed_gradient: math.new_array(),
            velocity: math.new_array(),
            index_in_trajectory: 0,
            logp: 0f64,
            logdet: 0f64,
            kinetic_energy: 0f64,
            transform_id: -1,
            initial_energy: 0f64,
        }
    }

    fn copy_into(&self, math: &mut M, other: &mut Self) {
        let Self {
            untransformed_position,
            untransformed_gradient,
            transformed_position,
            transformed_gradient,
            velocity,
            index_in_trajectory,
            logp,
            logdet,
            kinetic_energy,
            transform_id,
            initial_energy,
        } = self;

        other.index_in_trajectory = *index_in_trajectory;
        other.logp = *logp;
        other.logdet = *logdet;
        other.kinetic_energy = *kinetic_energy;
        other.transform_id = *transform_id;
        other.initial_energy = *initial_energy;
        math.copy_into(untransformed_position, &mut other.untransformed_position);
        math.copy_into(untransformed_gradient, &mut other.untransformed_gradient);
        math.copy_into(transformed_position, &mut other.transformed_position);
        math.copy_into(transformed_gradient, &mut other.transformed_gradient);
        math.copy_into(velocity, &mut other.velocity);
    }
}

pub struct TransformedHamiltonian<M: Math> {
    ones: M::Vector,
    step_size: f64,
    params: Option<M::TransformParams>,
    max_energy_error: f64,
    _phantom: PhantomData<M>,
}

impl<M: Math> TransformedHamiltonian<M> {
    pub fn new(math: &mut M, max_energy_error: f64) -> Self {
        let mut ones = math.new_array();
        math.fill_array(&mut ones, 1f64);
        Self {
            step_size: 0f64,
            ones,
            params: None,
            max_energy_error,
            _phantom: Default::default(),
        }
    }

    pub fn init_transformation(
        &mut self,
        math: &mut M,
        state: &TransformedPoint<M>,
    ) -> Result<(), NutsError> {
        let params = math
            .new_transformation(state.position(), state.gradient())
            .map_err(|_| NutsError::BadInitGrad())?;
        self.params = Some(params);
        Ok(())
    }

    pub fn update_params<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        math: &'a mut M,
        rng: &mut R,
        draws: impl Iterator<Item = &'a M::Vector>,
        grads: impl Iterator<Item = &'a M::Vector>,
    ) -> Result<(), NutsError> {
        math.update_transformation(
            rng,
            draws,
            grads,
            self.params.as_mut().expect("Transformation was empty"),
        )
        .map_err(|_| NutsError::BadInitGrad())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct Stats {}

pub struct Builder {}

impl StatTraceBuilder<Stats> for Builder {
    fn append_value(&mut self, value: Stats) {
        let Stats {} = value;
    }

    fn finalize(self) -> Option<StructArray> {
        let Self {} = self;
        None
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self {} = self;
        None
    }
}

impl<M: Math> SamplerStats<M> for TransformedHamiltonian<M> {
    type Stats = Stats;

    type Builder = Builder;

    fn new_builder(&self, _settings: &impl Settings, _dim: usize) -> Self::Builder {
        Builder {}
    }

    fn current_stats(&self, _math: &mut M) -> Self::Stats {
        Stats {}
    }
}

impl<M: Math> Hamiltonian<M> for TransformedHamiltonian<M> {
    type Point = TransformedPoint<M>;

    fn leapfrog<C: crate::nuts::Collector<M, Self::Point>>(
        &mut self,
        math: &mut M,
        pool: &mut StatePool<M, Self::Point>,
        start: &State<M, Self::Point>,
        dir: Direction,
        collector: &mut C,
    ) -> LeapfrogResult<M, Self::Point> {
        let mut out = pool.new_state(math);
        let out_point = out.try_point_mut().expect("New point has other references");

        let sign = match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };

        let epsilon = (sign as f64) * self.step_size;

        start
            .point()
            .first_velocity_halfstep(math, out_point, epsilon);

        start.point().position_step(math, out_point, epsilon);
        if let Err(logp_error) = out_point.update_gradient(self, math) {
            if !logp_error.is_recoverable() {
                return LeapfrogResult::Err(logp_error);
            }
            let div_info = DivergenceInfo {
                logp_function_error: Some(Arc::new(Box::new(logp_error))),
                start_location: Some(math.box_array(start.point().position())),
                start_gradient: Some(math.box_array(start.point().gradient())),
                start_momentum: None,
                end_location: None,
                start_idx_in_trajectory: Some(start.point().index_in_trajectory()),
                end_idx_in_trajectory: None,
                energy_error: None,
            };
            collector.register_leapfrog(math, start, &out, Some(&div_info));
            return LeapfrogResult::Divergence(div_info);
        }

        out_point.second_velocity_halfstep(math, epsilon);

        out_point.update_kinetic_energy(math);
        out_point.index_in_trajectory = start.index_in_trajectory() + sign;

        let energy_error = { out_point.energy() - start.point().initial_energy };
        if (energy_error > self.max_energy_error) | !energy_error.is_finite() {
            let divergence_info = DivergenceInfo {
                logp_function_error: None,
                start_location: Some(math.box_array(start.point().position())),
                start_gradient: Some(math.box_array(start.point().gradient())),
                end_location: Some(math.box_array(out_point.position())),
                start_momentum: None,
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

    fn is_turning(
        &self,
        math: &mut M,
        state1: &State<M, Self::Point>,
        state2: &State<M, Self::Point>,
    ) -> bool {
        let (start, end) = if state1.index_in_trajectory() < state2.index_in_trajectory() {
            (state1, state2)
        } else {
            (state2, state1)
        };

        let a = start.index_in_trajectory();
        let b = end.index_in_trajectory();

        assert!(a < b);
        // TODO double check
        let (turn1, turn2) = if (a >= 0) & (b >= 0) {
            math.scalar_prods3(
                &end.point().transformed_position,
                &start.point().transformed_position,
                &start.point().velocity,
                &end.point().velocity,
                &start.point().velocity,
            )
        } else if (b >= 0) & (a < 0) {
            math.scalar_prods2(
                &end.point().transformed_position,
                &start.point().transformed_position,
                &end.point().velocity,
                &start.point().velocity,
            )
        } else {
            assert!((a < 0) & (b < 0));
            math.scalar_prods3(
                &start.point().transformed_position,
                &end.point().transformed_position,
                &end.point().velocity,
                &end.point().velocity,
                &start.point().velocity,
            )
        };

        (turn1 < 0.) | (turn2 < 0.)
    }

    fn init_state(
        &mut self,
        math: &mut M,
        pool: &mut StatePool<M, Self::Point>,
        init: &[f64],
    ) -> Result<State<M, Self::Point>, NutsError> {
        let mut state = pool.new_state(math);
        let point = state.try_point_mut().expect("State already in use");
        math.read_from_slice(&mut point.untransformed_position, init);

        point
            .update_gradient(self, math)
            .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;

        if !point.is_valid(math) {
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
        let point = state.try_point_mut().expect("State has other references");
        math.array_gaussian(rng, &mut point.velocity, &self.ones);
        if math.transformation_id(self.params.as_ref().expect("No transformation set"))
            != point.transform_id
        {
            let logdet = math
                .inv_transform_normalize(
                    self.params.as_ref().expect("No transformation set"),
                    &point.untransformed_position,
                    &point.untransformed_gradient,
                    &mut point.transformed_position,
                    &mut point.transformed_gradient,
                )
                .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;
            point.logdet = logdet;
        }
        point.update_kinetic_energy(math);
        point.index_in_trajectory = 0;
        Ok(())
    }

    fn copy_state(
        &self,
        math: &mut M,
        pool: &mut StatePool<M, Self::Point>,
        state: &State<M, Self::Point>,
    ) -> State<M, Self::Point> {
        let mut new_state = pool.new_state(math);
        state.point().copy_into(
            math,
            new_state
                .try_point_mut()
                .expect("New point should not have other references"),
        );
        new_state
    }

    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn step_size_mut(&mut self) -> &mut f64 {
        &mut self.step_size
    }
}
