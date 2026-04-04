//! Concrete Hamiltonian that runs leapfrog in a whitened space to improve sampling geometry.
//!
//! Three trajectory kinds are supported via [`KineticEnergyKind`]:
//! - [`KineticEnergyKind::Euclidean`]: standard leapfrog with Euclidean kinetic energy.
//! - [`KineticEnergyKind::ExactNormal`]: geodesic leapfrog that is exact for a standard-normal
//!   potential (position and velocity rotate together in each 2-D plane).
//! - [`KineticEnergyKind::Microcanonical`]: isokinetic ESH-dynamics leapfrog (microcanonical
//!   HMC). The momentum is constrained to the unit sphere and the ESH update keeps it there
//!   while tracking the accumulated kinetic-energy change along the trajectory.

use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use nuts_derive::Storable;
use nuts_storable::HasDims;
use serde::{Deserialize, Serialize};

use crate::{
    DivergenceInfo, LogpError, Math, NutsError,
    dynamics::{Direction, Hamiltonian, LeapfrogResult, Point},
    dynamics::{State, StatePool},
    sampler_stats::{SamplerStats, StatsDims},
    transform::{ExternalTransformation, Transformation},
};

/// Selects the kinetic-energy form (and thus the integrator) used by
/// [`TransformedHamiltonian`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum KineticEnergyKind {
    /// Standard Euclidean kinetic energy `K = ½ ‖v‖²`.
    /// Uses the ordinary leapfrog integrator (velocity Verlet).
    #[default]
    Euclidean,

    /// Geodesic leapfrog that is *exact* for a standard-normal potential.
    /// Position and velocity are rotated together in each 2-D plane `(q_i, v_i)`.
    ExactNormal,

    /// Microcanonical / isokinetic HMC using ESH dynamics
    /// ([Steeg & Gallagher 2021](https://arxiv.org/abs/2111.02434),
    /// ported from the [BlackJAX implementation](https://github.com/blackjax-devs/blackjax/blob/main/blackjax/mcmc/integrators.py#L314)).
    ///
    /// The momentum is constrained to the unit sphere (`‖v‖ = 1`).
    /// The momentum update uses the ESH formula which preserves `‖v‖ = 1` exactly
    /// while tracking the cumulative kinetic-energy change needed for the
    /// Metropolis accept/reject decision.
    ///
    /// No partial momentum refreshment is performed — this variant only implements
    /// the deterministic ESH trajectory.
    Microcanonical,
}

// ---------------------------------------------------------------------------
// ESH (Extended Stochastic Hamiltonian) momentum update
// ---------------------------------------------------------------------------

pub struct TransformedPoint<M: Math> {
    pub(crate) untransformed_position: M::Vector,
    pub(crate) untransformed_gradient: M::Vector,
    pub(crate) transformed_position: M::Vector,
    pub(crate) transformed_gradient: M::Vector,
    pub(crate) velocity: M::Vector,
    index_in_trajectory: i64,
    logp: f64,
    logdet: f64,
    /// For Euclidean / ExactNormal: `½ ‖v‖²`.
    /// For Microcanonical: the accumulated kinetic-energy change ΔKE along the
    /// current leapfrog step (carried through `kinetic_energy` between the two
    /// half-steps, then fixed after the second half-step).
    kinetic_energy: f64,
    initial_energy: f64,
    transform_id: i64,
    /// The step size factor used by the leapfrog step that produced this point.
    /// For NUTS and static MCLMC this is always `1.0`; for MCLMC with dynamic
    /// step size reduction it may be `< 1.0`.  Used to compute importance
    /// weights: `log_weight = log(step_size_factor) - energy_error`.
    pub(crate) step_size_factor: f64,
}

impl<M: Math> Debug for TransformedPoint<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformedPoint")
            .field("untransformed_position", &self.untransformed_position)
            .field("untransformed_gradient", &self.untransformed_gradient)
            .field("transformed_position", &self.transformed_position)
            .field("transformed_gradient", &self.transformed_gradient)
            .field("velocity", &self.velocity)
            .field("index_in_trajectory", &self.index_in_trajectory)
            .field("logp", &self.logp)
            .field("logdet", &self.logdet)
            .field("kinetic_energy", &self.kinetic_energy)
            .field("transform_id", &self.transform_id)
            .finish()
    }
}

#[derive(Debug, Storable)]
pub struct PointStats {
    pub index_in_trajectory: i64,
    pub logp: f64,
    pub energy: f64,
    pub energy_error: f64,
    #[storable(dims("unconstrained_parameter"))]
    pub unconstrained_draw: Option<Vec<f64>>,
    #[storable(dims("unconstrained_parameter"))]
    pub gradient: Option<Vec<f64>>,
    pub fisher_distance: f64,
    #[storable(dims("unconstrained_parameter"))]
    pub transformed_position: Option<Vec<f64>>,
    #[storable(dims("unconstrained_parameter"))]
    pub transformed_gradient: Option<Vec<f64>>,
    pub transformation_index: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct TransformedPointStatsOptions {
    pub store_gradient: bool,
    pub store_unconstrained: bool,
    pub store_transformed: bool,
}

impl<M: Math> SamplerStats<M> for TransformedPoint<M> {
    type Stats = PointStats;
    type StatsOptions = TransformedPointStatsOptions;

    fn extract_stats(&self, math: &mut M, opt: Self::StatsOptions) -> Self::Stats {
        let unconstrained_draw = if opt.store_unconstrained {
            Some(math.box_array(&self.untransformed_position).into_vec())
        } else {
            None
        };
        let gradient = if opt.store_gradient {
            Some(math.box_array(&self.untransformed_gradient).into_vec())
        } else {
            None
        };
        let mut transformed_position = None;
        let mut transformed_gradient = None;
        if opt.store_transformed {
            transformed_position = Some(math.box_array(&self.transformed_position));
            transformed_gradient = Some(math.box_array(&self.transformed_gradient));
        }
        let fisher_distance =
            math.sq_norm_sum(&self.transformed_position, &self.transformed_gradient);
        PointStats {
            index_in_trajectory: self.index_in_trajectory,
            logp: self.logp,
            energy: self.energy(),
            energy_error: self.energy_error(),
            unconstrained_draw,
            gradient,
            fisher_distance,
            transformation_index: self.transform_id,
            transformed_gradient: transformed_gradient.map(|x| x.into_vec()),
            transformed_position: transformed_position.map(|x| x.into_vec()),
        }
    }
}

impl<M: Math> TransformedPoint<M> {
    /// First velocity half-step.
    fn first_velocity_halfstep(
        &self,
        math: &mut M,
        out: &mut Self,
        epsilon: f64,
        kind: KineticEnergyKind,
    ) {
        match kind {
            KineticEnergyKind::ExactNormal => {
                math.std_norm_grad_flow(
                    &self.transformed_position,
                    &self.transformed_gradient,
                    &self.velocity,
                    &mut out.velocity,
                    epsilon / 2.,
                );
            }
            KineticEnergyKind::Euclidean => {
                math.axpy_out(
                    &self.transformed_gradient,
                    &self.velocity,
                    epsilon / 2.,
                    &mut out.velocity,
                );
            }
            KineticEnergyKind::Microcanonical => {
                // TODO this is an extra copy we could get rid of
                math.copy_into(&self.velocity, &mut out.velocity);
                let ndim = math.dim();
                out.kinetic_energy = self.kinetic_energy
                    + math.esh_momentum_update(
                        &self.transformed_gradient,
                        &mut out.velocity,
                        // Make the step sizes comparable
                        (ndim as f64).sqrt() * epsilon / 2.,
                    );
            }
        }
    }

    /// Position (and, for geodesic integrators, simultaneous velocity) step.
    fn position_step(&self, math: &mut M, out: &mut Self, epsilon: f64, kind: KineticEnergyKind) {
        match kind {
            //   q' =  q cos ε + v sin ε
            //   v' = −q sin ε + v cos ε
            KineticEnergyKind::ExactNormal => {
                math.std_norm_flow(
                    &self.transformed_position,
                    &mut out.transformed_position,
                    &mut out.velocity,
                    epsilon,
                );
            }
            KineticEnergyKind::Euclidean | KineticEnergyKind::Microcanonical => {
                let epsilon = if matches!(kind, KineticEnergyKind::Microcanonical) {
                    epsilon * (math.dim() as f64).sqrt()
                } else {
                    epsilon
                };
                math.axpy_out(
                    &out.velocity,
                    &self.transformed_position,
                    epsilon,
                    &mut out.transformed_position,
                );
            }
        }
    }

    /// Second velocity half-step.
    ///
    /// `accumulated_delta_ke` is the ΔKE from the first half-step (only used for
    /// Microcanonical; ignored for other variants).  After this call `self.kinetic_energy`
    /// holds the final value appropriate for `energy()`.
    fn second_velocity_halfstep(&mut self, math: &mut M, epsilon: f64, kind: KineticEnergyKind) {
        match kind {
            KineticEnergyKind::ExactNormal => {
                math.std_norm_grad_flow_inplace(
                    &self.transformed_position,
                    &self.transformed_gradient,
                    &mut self.velocity,
                    epsilon / 2.,
                );
            }
            KineticEnergyKind::Euclidean => {
                math.axpy(&self.transformed_gradient, &mut self.velocity, epsilon / 2.);
            }
            KineticEnergyKind::Microcanonical => {
                let ndim = math.dim();
                self.kinetic_energy = self.kinetic_energy
                    + math.esh_momentum_update(
                        &self.transformed_gradient,
                        &mut self.velocity,
                        (ndim as f64).sqrt() * epsilon / 2.,
                    );
            }
        }
    }

    fn update_kinetic_energy(&mut self, math: &mut M) {
        self.kinetic_energy = 0.5 * math.array_vector_dot(&self.velocity, &self.velocity);
    }

    /// Reset the trajectory-tracking fields so that `energy_error()` is measured
    /// relative to the current state (e.g. after a partial momentum refresh).
    ///
    /// For [`KineticEnergyKind::Microcanonical`]: sets `kinetic_energy = 0` (the
    /// accumulated ΔKE accumulator is zeroed for the new trajectory).
    /// For [`KineticEnergyKind::Euclidean`] / [`KineticEnergyKind::ExactNormal`]:
    /// recomputes `kinetic_energy = ½‖v‖²` from the current (post-refresh) velocity.
    ///
    /// In both cases sets `index_in_trajectory = 0`, `initial_energy = energy()`,
    /// and `step_size_factor = 1.0`.
    pub(crate) fn reset_trajectory_energy(&mut self, math: &mut M, kind: KineticEnergyKind) {
        match kind {
            KineticEnergyKind::Microcanonical => {
                self.kinetic_energy = 0.0;
            }
            KineticEnergyKind::Euclidean | KineticEnergyKind::ExactNormal => {
                self.update_kinetic_energy(math);
            }
        }
        self.index_in_trajectory = 0;
        self.initial_energy = self.energy();
        self.step_size_factor = 1.0;
    }

    fn init_from_untransformed_position<T: Transformation<M>>(
        &mut self,
        transformation: &T,
        math: &mut M,
    ) -> Result<(), M::LogpErr> {
        let (logp, logdet) = transformation.init_from_untransformed_position(
            math,
            &self.untransformed_position,
            &mut self.untransformed_gradient,
            &mut self.transformed_position,
            &mut self.transformed_gradient,
        )?;
        self.logp = logp;
        self.logdet = logdet;
        self.transform_id = transformation.transformation_id(math);
        Ok(())
    }

    fn init_from_transformed_position<T: Transformation<M>>(
        &mut self,
        transformation: &T,
        math: &mut M,
    ) -> Result<(), M::LogpErr> {
        let (logp, logdet) = transformation.init_from_transformed_position(
            math,
            &mut self.untransformed_position,
            &mut self.untransformed_gradient,
            &self.transformed_position,
            &mut self.transformed_gradient,
        )?;
        self.logp = logp;
        self.logdet = logdet;
        self.transform_id = transformation.transformation_id(math);
        Ok(())
    }

    fn check_untransformed(&self, math: &mut M) -> bool {
        if !math.array_all_finite(&self.untransformed_gradient) {
            return false;
        }
        if !math.array_all_finite(&self.untransformed_position) {
            return false;
        }
        true
    }

    fn check_all(&self, math: &mut M) -> bool {
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

    /// The Hamiltonian energy at this point.
    ///
    /// For Euclidean / ExactNormal:  `E = ½‖v‖² − (logp + logdet)`
    /// For Microcanonical:           `E = ΔKE_accum − (logp + logdet)`
    ///
    /// In both cases `energy_error = energy − initial_energy` is used for
    /// divergence detection.  The constant offset `−(n−1) log 2` present in
    /// the ESH kinetic energy cancels in the difference and is therefore
    /// omitted.
    fn energy(&self) -> f64 {
        self.kinetic_energy - (self.logp + self.logdet)
    }

    fn initial_energy(&self) -> f64 {
        self.initial_energy
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
            step_size_factor: 1.0,
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
            step_size_factor,
        } = self;

        other.index_in_trajectory = *index_in_trajectory;
        other.logp = *logp;
        other.logdet = *logdet;
        other.kinetic_energy = *kinetic_energy;
        other.transform_id = *transform_id;
        other.initial_energy = *initial_energy;
        other.step_size_factor = *step_size_factor;
        math.copy_into(untransformed_position, &mut other.untransformed_position);
        math.copy_into(untransformed_gradient, &mut other.untransformed_gradient);
        math.copy_into(transformed_position, &mut other.transformed_position);
        math.copy_into(transformed_gradient, &mut other.transformed_gradient);
        math.copy_into(velocity, &mut other.velocity);
    }
}

pub struct TransformedHamiltonian<M: Math, T: Transformation<M>> {
    ones: M::Vector,
    zeros: M::Vector,
    step_size: f64,
    /// Momentum decoherence length `L` for the isokinetic Langevin refresh.
    /// `None` disables the refresh (used by NUTS); `Some(L)` enables it (MCLMC).
    momentum_decoherence_length: Option<f64>,
    transformation: T,
    max_energy_error: f64,
    pub kinetic_energy_kind: KineticEnergyKind,
    pool: StatePool<M, TransformedPoint<M>>,
}

impl<M: Math, T: Transformation<M>> TransformedHamiltonian<M, T> {
    pub fn new(
        math: &mut M,
        max_energy_error: f64,
        transformation: T,
        kinetic_energy_kind: KineticEnergyKind,
    ) -> Self {
        let mut ones = math.new_array();
        math.fill_array(&mut ones, 1f64);
        let mut zeros = math.new_array();
        math.fill_array(&mut zeros, 0f64);
        let pool = StatePool::new(math, 10);
        Self {
            step_size: 0f64,
            momentum_decoherence_length: None,
            ones,
            zeros,
            transformation,
            max_energy_error,
            kinetic_energy_kind,
            pool,
        }
    }

    pub fn transformation(&self) -> &T {
        &self.transformation
    }

    pub fn transformation_mut(&mut self) -> &mut T {
        &mut self.transformation
    }

    pub fn set_momentum_decoherence_length(&mut self, l: Option<f64>) {
        self.momentum_decoherence_length = l;
    }

    /// Change the kinetic-energy kind (and thus the leapfrog integrator and
    /// momentum distribution) used by this Hamiltonian.
    ///
    /// When switching from [`KineticEnergyKind::Euclidean`] to
    /// [`KineticEnergyKind::Microcanonical`] the caller is responsible for
    /// reinitializing the state.
    pub fn set_kinetic_energy_kind(&mut self, kind: KineticEnergyKind) {
        self.kinetic_energy_kind = kind;
    }
}

impl<M: Math> TransformedHamiltonian<M, ExternalTransformation<M>> {
    pub fn init_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        math: &mut M,
        position: &[f64],
        chain: u64,
    ) -> Result<(), NutsError> {
        let mut gradient_array = math.new_array();
        let mut position_array = math.new_array();
        math.read_from_slice(&mut position_array, position);
        let _ = math
            .logp_array(&position_array, &mut gradient_array)
            .map_err(|e| NutsError::BadInitGrad(Box::new(e)))?;
        let mut params = math
            .init_transformation(rng, &position_array, &gradient_array, chain)
            .map_err(|e| NutsError::BadInitGrad(Box::new(e)))?;
        std::mem::swap(self.transformation_mut().params_mut(), &mut params);
        Ok(())
    }

    pub fn update_params<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        math: &'a mut M,
        rng: &mut R,
        draws: impl ExactSizeIterator<Item = &'a M::Vector>,
        grads: impl ExactSizeIterator<Item = &'a M::Vector>,
        logps: impl ExactSizeIterator<Item = &'a f64>,
    ) -> Result<(), NutsError> {
        let t = self.transformation_mut();
        math.update_transformation(rng, draws, grads, logps, t.params_mut())
            .map_err(|e| NutsError::BadInitGrad(Box::new(e)))?;
        Ok(())
    }
}

#[derive(Debug, Storable)]
pub struct HamiltonianStats<P: HasDims, S: nuts_storable::Storable<P>> {
    pub step_size: f64,
    #[storable(flatten)]
    pub transformation: S,
    #[storable(ignore)]
    _phantom: PhantomData<fn() -> P>,
}

impl<M: Math, T: Transformation<M>> SamplerStats<M> for TransformedHamiltonian<M, T> {
    type Stats = HamiltonianStats<StatsDims, T::Stats>;
    type StatsOptions = T::StatsOptions;

    fn extract_stats(&self, math: &mut M, opt: Self::StatsOptions) -> Self::Stats {
        let transformation_stats = self.transformation.extract_stats(math, opt);
        HamiltonianStats {
            step_size: self.step_size,
            transformation: transformation_stats,
            _phantom: PhantomData,
        }
    }
}

impl<M: Math, T: Transformation<M>> Hamiltonian<M> for TransformedHamiltonian<M, T> {
    type Point = TransformedPoint<M>;

    fn leapfrog<C: crate::nuts::Collector<M, Self::Point>>(
        &mut self,
        math: &mut M,
        start: &State<M, Self::Point>,
        dir: Direction,
        step_size_factor: f64,
        energy_baseline: f64,
        collector: &mut C,
    ) -> LeapfrogResult<M, Self::Point> {
        let mut out = self.pool().new_state(math);
        let out_point = out.try_point_mut().expect("New point has other references");

        out_point.initial_energy = start.point().initial_energy();
        out_point.transform_id = start.point().transform_id;

        let sign = match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };

        let epsilon = (sign as f64) * self.step_size * step_size_factor;
        out_point.step_size_factor = step_size_factor;
        let kind = self.kinetic_energy_kind;

        // --- First velocity half-step ---
        // For Microcanonical: out_point.kinetic_energy receives the running ΔKE
        // after this call; for other kinds it is left at whatever value it had
        // (it will be overwritten by update_kinetic_energy below).
        start
            .point()
            .first_velocity_halfstep(math, out_point, epsilon, kind);

        // --- Position step ---
        start.point().position_step(math, out_point, epsilon, kind);

        // --- Evaluate log-density at new position ---
        let transformation = self.transformation();
        if let Err(logp_error) = out_point.init_from_transformed_position(transformation, math) {
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

        out_point.second_velocity_halfstep(math, epsilon, kind);

        // For Microcanonical, kinetic_energy already holds the total accumulated ΔKE
        // (set by second_velocity_halfstep). For other kinds we recompute from ½‖v‖².
        if kind != KineticEnergyKind::Microcanonical {
            out_point.update_kinetic_energy(math);
        }

        out_point.index_in_trajectory = start.index_in_trajectory() + sign;

        let energy_error = out_point.energy() - energy_baseline;
        let bad_energy = match self.kinetic_energy_kind {
            KineticEnergyKind::Euclidean | KineticEnergyKind::ExactNormal => {
                energy_error > self.max_energy_error
            }
            KineticEnergyKind::Microcanonical => energy_error.abs() >= self.max_energy_error,
        };
        if bad_energy | !energy_error.is_finite() {
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

        let (turn1, turn2) = math.scalar_prods3(
            &end.point().transformed_position,
            &start.point().transformed_position,
            &self.zeros,
            &start.point().velocity,
            &end.point().velocity,
        );

        (turn1 < 0f64) | (turn2 < 0f64)
    }

    fn init_state(
        &mut self,
        math: &mut M,
        init: &[f64],
    ) -> Result<State<M, Self::Point>, NutsError> {
        let mut state = self.pool().new_state(math);
        let point = state.try_point_mut().expect("State already in use");
        math.read_from_slice(&mut point.untransformed_position, init);

        let transformation = self.transformation();
        point
            .init_from_untransformed_position(transformation, math)
            .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;

        if !point.check_all(math) {
            Err(NutsError::BadInitGrad(
                anyhow::anyhow!("Invalid initial point").into(),
            ))
        } else {
            Ok(state)
        }
    }

    fn init_state_untransformed(
        &mut self,
        math: &mut M,
        untransformed_position: &[f64],
    ) -> Result<State<M, Self::Point>, NutsError> {
        let mut state = self.pool().new_state(math);
        let point = state.try_point_mut().expect("State already in use");
        math.read_from_slice(&mut point.untransformed_position, untransformed_position);
        math.logp_array(
            &point.untransformed_position,
            &mut point.untransformed_gradient,
        )
        .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;
        // Force recomputation of transformed coordinates on first leapfrog step
        point.transform_id = -1;
        if !point.check_untransformed(math) {
            Err(NutsError::BadInitGrad(
                anyhow::anyhow!("Invalid initial point").into(),
            ))
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

        // Sample raw isotropic Gaussian momentum.
        math.array_gaussian(rng, &mut point.velocity, &self.ones);

        // For Microcanonical HMC the momentum must lie on the unit sphere.
        if self.kinetic_energy_kind == KineticEnergyKind::Microcanonical {
            math.array_normalize(&mut point.velocity);
        }

        let current_transform_id = self.transformation().transformation_id(math);
        if current_transform_id != point.transform_id {
            let logdet = self
                .transformation()
                .inv_transform_normalize(
                    math,
                    &point.untransformed_position,
                    &point.untransformed_gradient,
                    &mut point.transformed_position,
                    &mut point.transformed_gradient,
                )
                .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;
            point.logdet = logdet;
            point.transform_id = current_transform_id;
        }

        match self.kinetic_energy_kind {
            KineticEnergyKind::Microcanonical => {
                // Initial accumulated ΔKE is 0 (no steps taken yet).
                // energy() = 0 − (logp + logdet) = −(logp + logdet).
                point.kinetic_energy = 0.0;
            }
            _ => {
                point.update_kinetic_energy(math);
            }
        }

        point.index_in_trajectory = 0;
        point.initial_energy = point.energy();
        Ok(())
    }

    fn pool(&mut self) -> &mut StatePool<M, Self::Point> {
        &mut self.pool
    }

    fn copy_state(&mut self, math: &mut M, state: &State<M, Self::Point>) -> State<M, Self::Point> {
        let mut new_state = self.pool.new_state(math);
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

    fn update_stats_options(
        &mut self,
        math: &mut M,
        current: <Self as SamplerStats<M>>::StatsOptions,
    ) -> <Self as SamplerStats<M>>::StatsOptions {
        self.transformation.next_stats_options(math, current)
    }

    fn step_size_mut(&mut self) -> &mut f64 {
        &mut self.step_size
    }

    fn momentum_decoherence_length(&self) -> Option<f64> {
        self.momentum_decoherence_length
    }

    fn momentum_decoherence_length_mut(&mut self) -> Option<&mut f64> {
        self.momentum_decoherence_length.as_mut()
    }

    fn partial_momentum_refresh<R: rand::Rng + ?Sized>(
        &mut self,
        math: &mut M,
        state: &mut State<M, Self::Point>,
        rng: &mut R,
        factor: f64,
    ) -> Result<(), NutsError> {
        let Some(momentum_decoherence_length) = self.momentum_decoherence_length else {
            return Ok(());
        };

        let half_step = self.step_size * factor / 2.0;

        // TODO: Avoid array allocation
        let mut noise = math.new_array();
        math.array_gaussian(rng, &mut noise, &self.ones);

        let point = state.try_point_mut().map_err(|_| {
            NutsError::BadInitGrad(anyhow::anyhow!("State in use during momentum refresh").into())
        })?;

        match self.kinetic_energy_kind {
            KineticEnergyKind::Microcanonical => {
                // Isokinetic Langevin (OU on the unit sphere):
                // ν = sqrt((exp(2·half_step/L) − 1) / n),  n = dim
                // p ← (p + ν·z) / ‖p + ν·z‖,  z ~ N(0, I)
                let n = math.dim() as f64;
                let nu = ((2.0 * half_step / momentum_decoherence_length).exp_m1() / n).sqrt();
                math.axpy(&noise, &mut point.velocity, nu);
                math.array_normalize(&mut point.velocity);
            }
            KineticEnergyKind::Euclidean | KineticEnergyKind::ExactNormal => {
                // Ornstein–Uhlenbeck for Gaussian momentum p ~ N(0, I):
                //   α = exp(−half_step / L)
                //   β = sqrt(1 − α²)
                //   p_new = α · p + β · z,  z ~ N(0, I)
                //
                // `axpy_out(x, y, a, out)` computes `out = y + a·x`.
                // So `axpy_out(&velocity, &zeros, alpha, &mut new_velocity)`
                // gives `new_velocity = zeros + alpha·velocity = alpha·velocity`.
                let alpha = (-half_step / momentum_decoherence_length).exp();
                let beta = (1.0 - alpha * alpha).sqrt();
                let mut new_velocity = math.new_array();
                math.axpy_out(&point.velocity, &self.zeros, alpha, &mut new_velocity);
                math.axpy(&noise, &mut new_velocity, beta);
                math.copy_into(&new_velocity, &mut point.velocity);
                // Keep kinetic_energy consistent with the updated velocity.
                point.update_kinetic_energy(math);
            }
        }

        Ok(())
    }
}
