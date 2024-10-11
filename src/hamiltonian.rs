use std::sync::Arc;

use crate::{
    nuts::Collector,
    sampler_stats::SamplerStats,
    state::{State, StatePool},
    Math, NutsError,
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

#[derive(Debug, Copy, Clone)]
pub enum Direction {
    Forward,
    Backward,
}

impl rand::distributions::Distribution<Direction> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Direction {
        if rng.gen::<bool>() {
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

pub trait Point<M: Math>: Sized {
    fn position(&self) -> &M::Vector;
    fn gradient(&self) -> &M::Vector;
    fn index_in_trajectory(&self) -> i64;
    fn energy(&self) -> f64;
    fn logp(&self) -> f64;
    fn energy_error(&self) -> f64;

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
    /// Return either an unrecoverable error, a new state or a divergence.
    fn leapfrog<C: Collector<M, Self::Point>>(
        &mut self,
        math: &mut M,
        pool: &mut StatePool<M, Self::Point>,
        start: &State<M, Self::Point>,
        dir: Direction,
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
        pool: &mut StatePool<M, Self::Point>,
        init: &[f64],
    ) -> Result<State<M, Self::Point>, NutsError>;

    /// Randomize the momentum part of a state
    fn initialize_trajectory<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut State<M, Self::Point>,
        rng: &mut R,
    ) -> Result<(), NutsError>;

    fn new_pool(&self, math: &mut M, pool_size: usize) -> StatePool<M, Self::Point> {
        StatePool::new(math, pool_size)
    }

    fn copy_state(
        &self,
        math: &mut M,
        pool: &mut StatePool<M, Self::Point>,
        state: &State<M, Self::Point>,
    ) -> State<M, Self::Point>;

    fn step_size(&self) -> f64;
    fn step_size_mut(&mut self) -> &mut f64;
}
