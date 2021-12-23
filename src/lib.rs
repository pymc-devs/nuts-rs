#![feature(portable_simd)]

pub(crate) mod math;
pub(crate) mod nuts;
pub(crate) mod cpu_potential;
pub(crate) mod cpu_sampler;
pub(crate) mod adapt_strategy;
pub(crate) mod cpu_state;
pub(crate) mod mass_matrix;
pub(crate) mod stepsize;

pub use nuts::{Sampler, NutsOptions, LogpError, SampleStats, AsSampleStatMap, SampleStatValue};
pub use cpu_potential::CpuLogpFunc;
pub use cpu_sampler::{InitPointFunc, JitterInitFunc, sample_parallel, SamplerArgs, new_sampler};
pub use stepsize::DualAverageSettings;
pub use mass_matrix::DiagAdaptExpSettings;
