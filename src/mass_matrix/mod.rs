mod adapt;
mod diagonal;
mod low_rank;

pub use adapt::DiagAdaptExpSettings;
pub(crate) use adapt::MassMatrixAdaptStrategy;
pub(crate) use adapt::Strategy;
pub(crate) use diagonal::{DiagMassMatrix, MassMatrix};
pub use low_rank::LowRankSettings;
pub(crate) use low_rank::{LowRankMassMatrix, LowRankMassMatrixStrategy};
