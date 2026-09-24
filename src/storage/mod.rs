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
