//! Core abstractions for MCMC models.
//!
//! Provides the `Model` trait which defines the interface for MCMC models,
//! including the math backend and initialization methods needed for sampling.

use std::sync::Arc;

use anyhow::Result;
use rand::Rng;

use crate::Math;

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
    /// The math backend owns whatever it needs from the model, usually by holding
    /// on to the `Arc` passed to [`Model::math`], so it can outlive any particular
    /// borrow of the model.
    type Math: Math;

    /// Returns the math backend for this model.
    ///
    /// Called once per chain, and once more to set up the trace. Call it as
    /// `Arc::clone(&model).math(rng)` to keep your own handle.
    fn math<R: Rng + ?Sized>(self: Arc<Self>, rng: &mut R) -> Result<Self::Math>;

    /// Initializes the starting position for MCMC sampling.
    ///
    /// Sets initial values for the parameter vector. The starting position should
    /// be in a reasonable region where the log probability density is finite.
    fn init_position<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        chain_id: u64,
        position: &mut [f64],
    ) -> Result<()>;
}
