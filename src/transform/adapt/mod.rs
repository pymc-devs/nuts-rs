//! Online estimators that update the mass-matrix transformation from samples collected during warmup.

mod diagonal;
mod low_rank;
mod strategy;

pub use diagonal::DiagAdaptExpSettings;
pub use diagonal::Strategy as DiagAdaptStrategy;
pub use low_rank::LowRankMassMatrixStrategy;
pub use strategy::MassMatrixAdaptStrategy;
