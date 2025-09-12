//! Core abstractions for MCMC models.
//!
//! Provides the `Model` trait which defines the interface for MCMC models,
//! including the math backend and initialization methods needed for sampling.

use anyhow::Result;
use rand::Rng;

use crate::math_base::Math;

/// Trait for MCMC models with associated math backend and initialization.
///
/// Defines the interface for models that can be used with MCMC sampling algorithms.
/// Provides access to mathematical operations needed for sampling and methods for
/// initializing the sampling position.
///
/// The trait is thread-safe to enable parallel sampling scenarios.
pub trait Model: Send + Sync + 'static {
    /// The math backend used by this MCMC model.
    ///
    /// Specifies which math implementation will be used for computing log probability
    /// densities, gradients, and other operations required during sampling.
    ///
    /// The lifetime parameter allows the math backend to borrow from the model instance.
    type Math<'model>: Math
    where
        Self: 'model;

    /// Returns the math backend for this model.
    fn math<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Self::Math<'_>>;

    /// Initializes the starting position for MCMC sampling.
    ///
    /// Sets initial values for the parameter vector. The starting position should
    /// be in a reasonable region where the log probability density is finite.
    fn init_position<R: Rng + ?Sized>(&self, rng: &mut R, position: &mut [f64]) -> Result<()>;
}
