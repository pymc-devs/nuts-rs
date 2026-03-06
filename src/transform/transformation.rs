//! Define the interface for a differentiable bijection between original and whitened parameter spaces.

use std::fmt::Debug;

use crate::{Math, SamplerStats};

/// A transformation that maps from the untransformed (original) parameter space
/// to a transformed space in which sampling is performed with a unit mass matrix.
///
/// For the mass-matrix case this is an affine (diagonal or low-rank) scaling;
/// for the flow case this is a learned normalizing flow.
pub trait Transformation<M: Math>: SamplerStats<M> + Debug {
    /// Map from untransformed → transformed space, computing logp and logdet.
    ///
    /// * Fills `untransformed_gradient` with ∂logp/∂x.
    /// * Fills `transformed_position` with T(x).
    /// * Fills `transformed_gradient` with ∂logp/∂z  (= ∂logp/∂x · ∂x/∂z).
    /// * Returns `(logp, logdet)` where `logdet = log |∂z/∂x|`.
    fn init_from_untransformed_position(
        &self,
        math: &mut M,
        untransformed_position: &M::Vector,
        untransformed_gradient: &mut M::Vector,
        transformed_position: &mut M::Vector,
        transformed_gradient: &mut M::Vector,
    ) -> Result<(f64, f64), M::LogpErr>;

    /// Map from transformed → untransformed space, computing logp and logdet.
    ///
    /// * Fills `untransformed_position` with T⁻¹(z).
    /// * Fills `untransformed_gradient` with ∂logp/∂x.
    /// * Fills `transformed_gradient` with ∂logp/∂z.
    /// * Returns `(logp, logdet)`.
    fn init_from_transformed_position(
        &self,
        math: &mut M,
        untransformed_position: &mut M::Vector,
        untransformed_gradient: &mut M::Vector,
        transformed_position: &M::Vector,
        transformed_gradient: &mut M::Vector,
    ) -> Result<(f64, f64), M::LogpErr>;

    /// Recompute only the transformed coordinates from an already-evaluated
    /// untransformed point (logp and gradient already known).
    ///
    /// Returns the new `logdet`.
    fn inv_transform_normalize(
        &self,
        math: &mut M,
        untransformed_position: &M::Vector,
        untransformed_gradient: &M::Vector,
        transformed_position: &mut M::Vector,
        transformed_gradient: &mut M::Vector,
    ) -> Result<f64, M::LogpErr>;

    /// A monotonically increasing version counter.  When this changes the
    /// transformed coordinates of a cached point must be recomputed.
    fn transformation_id(&self, math: &mut M) -> i64;
}
