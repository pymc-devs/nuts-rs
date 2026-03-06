mod cpu_math;
mod math;
mod util;

pub use cpu_math::{CpuLogpFunc, CpuMath, CpuMathError};
pub use math::{LogpError, Math};
pub(crate) use util::logaddexp;
