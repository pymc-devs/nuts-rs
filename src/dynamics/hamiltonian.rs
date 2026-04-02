//! Define the abstract interface for a Hamiltonian system (leapfrog, U-turn test, divergence detection).

use std::{fmt::Debug, sync::Arc};

use nuts_derive::Storable;
use rand::{
    Rng, RngExt,
    distr::{Distribution, StandardUniform},
};

use crate::{
    Math, NutsError,
    dynamics::{State, StatePool},
    nuts::Collector,
    sampler_stats::SamplerStats,
};

/// Details about a divergence that might have occured during sampling
///
/// There are two reasons why we might observe a divergence:
/// - The integration error of the Hamiltonian is larger than
///   a cutoff value or nan.
/// - The logp function caused a recoverable error (eg if an ODE solver
///   failed)
#[derive(Debug, Clone)]
pub struct DivergenceInfo {
    pub start_momentum: Option<Box<[f64]>>,
    pub start_location: Option<Box<[f64]>>,
    pub start_gradient: Option<Box<[f64]>>,
    pub end_location: Option<Box<[f64]>>,
    pub energy_error: Option<f64>,
    pub end_idx_in_trajectory: Option<i64>,
    pub start_idx_in_trajectory: Option<i64>,
    pub logp_function_error: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

/// Per-draw divergence statistics, suitable for storage.
#[derive(Debug, Storable)]
pub struct DivergenceStats {
    pub diverging: bool,
    pub divergence_message: Option<String>,
    #[storable(dims("unconstrained_parameter"))]
    pub divergence_start: Option<Vec<f64>>,
    #[storable(dims("unconstrained_parameter"))]
    pub divergence_start_gradient: Option<Vec<f64>>,
    #[storable(dims("unconstrained_parameter"))]
    pub divergence_end: Option<Vec<f64>>,
    #[storable(dims("unconstrained_parameter"))]
    pub divergence_momentum: Option<Vec<f64>>,
    pub divergence_energy_error: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct DivergenceStatsOptions {
    pub store_divergences: bool,
}

impl From<(Option<&DivergenceInfo>, DivergenceStatsOptions)> for DivergenceStats {
    fn from((info, options): (Option<&DivergenceInfo>, DivergenceStatsOptions)) -> Self {
        DivergenceStats {
            diverging: info.is_some(),
            divergence_start: if options.store_divergences {
                info.and_then(|d| d.start_location.as_ref().map(|v| v.as_ref().to_vec()))
            } else {
                None
            },
            divergence_start_gradient: if options.store_divergences {
                info.and_then(|d| d.start_gradient.as_ref().map(|v| v.as_ref().to_vec()))
            } else {
                None
            },
            divergence_end: if options.store_divergences {
                info.and_then(|d| d.end_location.as_ref().map(|v| v.as_ref().to_vec()))
            } else {
                None
            },
            divergence_momentum: if options.store_divergences {
                info.and_then(|d| d.start_momentum.as_ref().map(|v| v.as_ref().to_vec()))
            } else {
                None
            },
            divergence_message: info
                .and_then(|d| d.logp_function_error.as_ref().map(|err| err.to_string())),
            divergence_energy_error: info.and_then(|d| d.energy_error),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Direction {
    Forward,
    Backward,
}

impl Distribution<Direction> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Direction {
        if rng.random::<bool>() {
            Direction::Forward
        } else {
            Direction::Backward
        }
    }
}

pub enum LeapfrogResult<M: Math, P: Point<M>> {
    Ok(State<M, P>),
    Divergence(DivergenceInfo),
    Err(M::LogpErr),
}

pub trait Point<M: Math>: Sized + SamplerStats<M> + Debug {
    fn position(&self) -> &M::Vector;
    fn gradient(&self) -> &M::Vector;
    fn index_in_trajectory(&self) -> i64;
    fn energy(&self) -> f64;
    fn logp(&self) -> f64;

    fn energy_error(&self) -> f64 {
        self.energy() - self.initial_energy()
    }

    fn initial_energy(&self) -> f64;

    fn new(math: &mut M) -> Self;
    fn copy_into(&self, math: &mut M, other: &mut Self);
}

/// The hamiltonian defined by the potential energy and the kinetic energy
pub trait Hamiltonian<M: Math>: SamplerStats<M> + Sized {
    /// The type that stores a point in phase space, together
    /// with some information about the location inside the
    /// integration trajectory.
    type Point: Point<M>;

    /// Perform one leapfrog step.
    ///
    /// `step_size_factor` scales the hamiltonian's base step size for this
    /// step only: `actual_ε = hamiltonian.step_size() * step_size_factor`.
    /// The actual step size used is stored on the output point as
    /// `point.step_size`, so callers can compute importance weights as
    /// `log(point.step_size) - energy_error`.
    ///
    /// Return either an unrecoverable error, a new state or a divergence.
    fn leapfrog<C: Collector<M, Self::Point>>(
        &mut self,
        math: &mut M,
        start: &State<M, Self::Point>,
        dir: Direction,
        step_size_factor: f64,
        collector: &mut C,
    ) -> LeapfrogResult<M, Self::Point>;

    fn is_turning(
        &self,
        math: &mut M,
        state1: &State<M, Self::Point>,
        state2: &State<M, Self::Point>,
    ) -> bool;

    /// Initialize a state at a new location.
    ///
    /// The momentum should be initialized to some arbitrary invalid number,
    /// it will later be set using Self::randomize_momentum.
    fn init_state(
        &mut self,
        math: &mut M,
        init: &[f64],
    ) -> Result<State<M, Self::Point>, NutsError>;

    /// Initialize a state at a new location, without applying a transformation.
    fn init_state_untransformed(
        &mut self,
        math: &mut M,
        init: &[f64],
    ) -> Result<State<M, Self::Point>, NutsError>;

    /// Randomize the momentum part of a state
    fn initialize_trajectory<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut State<M, Self::Point>,
        rng: &mut R,
    ) -> Result<(), NutsError>;

    fn pool(&mut self) -> &mut StatePool<M, Self::Point>;

    fn copy_state(&mut self, math: &mut M, state: &State<M, Self::Point>) -> State<M, Self::Point>;

    fn step_size(&self) -> f64;
    fn step_size_mut(&mut self) -> &mut f64;

    /// The momentum decoherence length `L` used for the isokinetic Langevin
    /// (partial momentum refresh) step.
    ///
    /// - `None` means no refresh is performed (default, used by NUTS).
    /// - `Some(L)` enables a half-step Ornstein–Uhlenbeck refresh with
    ///   `ν = sqrt((exp(2·ε/L) − 1) / n)` around each trajectory.
    fn momentum_decoherence_length(&self) -> Option<f64> {
        None
    }

    fn momentum_decoherence_length_mut(&mut self) -> Option<&mut f64> {
        None
    }

    /// Apply one isokinetic Langevin partial momentum refresh to `state`.
    ///
    /// `factor` scales the base step size: the half-step used internally is
    /// `hamiltonian.step_size() * factor / 2`.  When
    /// [`Self::momentum_decoherence_length`] returns `None` this must be a
    /// no-op.  Implementations that support the refresh should override this
    /// method.
    fn partial_momentum_refresh<R: rand::Rng + ?Sized>(
        &mut self,
        math: &mut M,
        state: &mut State<M, Self::Point>,
        rng: &mut R,
        factor: f64,
    ) -> Result<(), NutsError> {
        let _ = (math, state, rng, factor);
        Ok(())
    }
}
