//! Math backends: the abstract `Math` trait and its CPU implementation backed by a user-supplied logp function.

mod cpu_math;
mod math;
mod util;

pub use cpu_math::{CpuLogpFunc, CpuMath, CpuMathError};
pub use math::{LogpError, Math};
pub(crate) use util::logaddexp;
