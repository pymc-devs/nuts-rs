//! Trace storage backends for recording per-draw statistics and samples to various output formats.

#[cfg(feature = "arrow")]
mod arrow;
mod core;
mod csv;
mod hashmap;
#[cfg(feature = "ndarray")]
mod ndarray;
#[cfg(feature = "zarr")]
mod zarr;

#[cfg(feature = "arrow")]
pub use arrow::{ArrowConfig, ArrowTrace, ArrowTraceStorage};
#[cfg(feature = "zarr")]
pub use zarr::{ZarrAsyncConfig, ZarrAsyncTraceStorage, ZarrConfig, ZarrTraceStorage};

pub use csv::{CsvConfig, CsvTraceStorage};
#[cfg(all(test, not(feature = "parallel")))]
pub(crate) use hashmap::HashMapResult;
pub use hashmap::{HashMapConfig, HashMapValue};
#[cfg(feature = "ndarray")]
pub use ndarray::{NdarrayConfig, NdarrayTrace, NdarrayValue};

pub use core::{ChainStorage, StorageConfig, TraceStorage};

/// Name of the variant of a `Value`, for error messages about values that do not
/// match their declared type.
pub(crate) fn value_type_name(value: &nuts_storable::Value) -> &'static str {
    use nuts_storable::Value;
    match value {
        Value::U64(_) => "U64",
        Value::I64(_) => "I64",
        Value::F64(_) => "F64",
        Value::F32(_) => "F32",
        Value::Bool(_) => "Bool",
        Value::ScalarString(_) => "ScalarString",
        Value::DateTime64(_, _) => "DateTime64",
        Value::TimeDelta64(_, _) => "TimeDelta64",
        Value::ScalarU64(_) => "ScalarU64",
        Value::ScalarI64(_) => "ScalarI64",
        Value::ScalarF64(_) => "ScalarF64",
        Value::ScalarF32(_) => "ScalarF32",
        Value::ScalarBool(_) => "ScalarBool",
        Value::Strings(_) => "Strings",
    }
}

/// How many values a variable holds in one draw.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ExpectedLen {
    Exact(usize),
    /// Stats with an event dim hold whole events of this size.
    PerEvent(usize),
}

impl ExpectedLen {
    /// The expected length of each variable from its dims. `has_event_dim` tells which
    /// variables are event stats.
    pub(crate) fn for_variables(
        dims: &[(String, Vec<String>)],
        dim_sizes: &std::collections::HashMap<String, u64>,
        has_event_dim: impl Fn(&str) -> bool,
    ) -> anyhow::Result<std::collections::HashMap<String, ExpectedLen>> {
        use anyhow::Context;
        dims.iter()
            .map(|(name, dims)| {
                let size = dims
                    .iter()
                    .map(|dim| {
                        let size = dim_sizes
                            .get(dim)
                            .with_context(|| format!("Unknown dimension {dim} of {name}"))?;
                        usize::try_from(*size)
                            .with_context(|| format!("Dimension {dim} of {name} is too large"))
                    })
                    .product::<anyhow::Result<usize>>()?;
                let expected = if has_event_dim(name) {
                    ExpectedLen::PerEvent(size)
                } else {
                    ExpectedLen::Exact(size)
                };
                Ok((name.clone(), expected))
            })
            .collect()
    }

    pub(crate) fn check(self, value: &nuts_storable::Value) -> anyhow::Result<()> {
        let len = value_len(value);
        match self {
            ExpectedLen::Exact(expected) if len != expected => anyhow::bail!(
                "Got {len} values, but the declared shape holds {expected}. \
                 Does the model return values of the declared shape?"
            ),
            ExpectedLen::PerEvent(size)
                if (size == 0 && len != 0) || (size > 0 && len % size != 0) =>
            {
                anyhow::bail!(
                    "Got {len} values, which is not a whole number of events of size {size}"
                )
            }
            _ => Ok(()),
        }
    }
}

fn value_len(value: &nuts_storable::Value) -> usize {
    use nuts_storable::Value;
    match value {
        Value::U64(v) => v.len(),
        Value::I64(v) | Value::DateTime64(_, v) | Value::TimeDelta64(_, v) => v.len(),
        Value::F64(v) => v.len(),
        Value::F32(v) => v.len(),
        Value::Bool(v) => v.len(),
        Value::Strings(v) => v.len(),
        Value::ScalarString(_)
        | Value::ScalarU64(_)
        | Value::ScalarI64(_)
        | Value::ScalarF64(_)
        | Value::ScalarF32(_)
        | Value::ScalarBool(_) => 1,
    }
}
