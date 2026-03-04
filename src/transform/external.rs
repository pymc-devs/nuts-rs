use std::fmt::Debug;

use nuts_derive::Storable;

use crate::{Math, SamplerStats, transform::Transformation};

/// Wraps a user-provided normalizing flow stored as `M::FlowParameters`.
pub struct ExternalTransformation<M: Math> {
    params: M::FlowParameters,
}

impl<M: Math> Debug for ExternalTransformation<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalTransformation")
            .field("params", &"<flow parameters>")
            .finish()
    }
}

impl<M: Math> ExternalTransformation<M> {
    pub fn new(params: M::FlowParameters) -> Self {
        Self { params }
    }

    pub fn params(&self) -> &M::FlowParameters {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut M::FlowParameters {
        &mut self.params
    }
}

#[derive(Debug, Storable)]
pub struct ExternalTransformationStats {}

impl<M: Math> SamplerStats<M> for ExternalTransformation<M> {
    type Stats = ExternalTransformationStats;
    type StatsOptions = ();

    fn extract_stats(&self, _math: &mut M, _opt: ()) -> ExternalTransformationStats {
        ExternalTransformationStats {}
    }
}

impl<M: Math> Transformation<M> for ExternalTransformation<M> {
    fn init_from_untransformed_position(
        &self,
        math: &mut M,
        untransformed_position: &M::Vector,
        untransformed_gradient: &mut M::Vector,
        transformed_position: &mut M::Vector,
        transformed_gradient: &mut M::Vector,
    ) -> Result<(f64, f64), M::LogpErr> {
        math.init_from_untransformed_position(
            &self.params,
            untransformed_position,
            untransformed_gradient,
            transformed_position,
            transformed_gradient,
        )
    }

    fn init_from_transformed_position(
        &self,
        math: &mut M,
        untransformed_position: &mut M::Vector,
        untransformed_gradient: &mut M::Vector,
        transformed_position: &M::Vector,
        transformed_gradient: &mut M::Vector,
    ) -> Result<(f64, f64), M::LogpErr> {
        math.init_from_transformed_position(
            &self.params,
            untransformed_position,
            untransformed_gradient,
            transformed_position,
            transformed_gradient,
        )
    }

    fn inv_transform_normalize(
        &self,
        math: &mut M,
        untransformed_position: &M::Vector,
        untransformed_gradient: &M::Vector,
        transformed_position: &mut M::Vector,
        transformed_gradient: &mut M::Vector,
    ) -> Result<f64, M::LogpErr> {
        math.inv_transform_normalize(
            &self.params,
            untransformed_position,
            untransformed_gradient,
            transformed_position,
            transformed_gradient,
        )
    }

    fn transformation_id(&self, math: &mut M) -> i64 {
        math.transformation_id(&self.params)
            .expect("Transformation ID should be retrievable")
    }
}
