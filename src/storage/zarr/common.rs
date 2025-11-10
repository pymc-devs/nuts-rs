use std::mem::replace;
use std::sync::Arc;
use std::{collections::HashMap, num::NonZero};

use anyhow::Result;
use nuts_storable::{ItemType, Value};
use zarrs::array::{Array, ArrayBuilder, DataType, FillValue};
use zarrs::metadata_ext::data_type::NumpyTimeUnit;

/// Container for different types of sample values
#[derive(Clone, Debug)]
pub enum SampleBufferValue {
    F64(Vec<f64>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    I64(Vec<i64>),
    U64(Vec<u64>),
}

/// Buffer for collecting samples before writing to storage
pub struct SampleBuffer {
    pub items: SampleBufferValue,
    pub len: usize,
    pub full_at: usize,
    pub current_chunk: usize,
}

/// A chunk of samples ready to be written to storage
#[derive(Debug)]
pub struct Chunk {
    pub chunk_idx: usize,
    pub len: usize,
    pub full_at: usize,
    pub values: SampleBufferValue,
}

impl Chunk {
    /// Check if the chunk has reached its capacity
    pub fn is_full(&self) -> bool {
        self.full_at == self.len
    }
}

impl SampleBuffer {
    /// Create a new sample buffer with specified type and chunk size
    pub fn new(item_type: ItemType, chunk_size: u64) -> Self {
        let chunk_size = chunk_size.try_into().expect("Chunk size too large");
        let inner = match item_type {
            ItemType::F64 => SampleBufferValue::F64(Vec::with_capacity(chunk_size)),
            ItemType::F32 => SampleBufferValue::F32(Vec::with_capacity(chunk_size)),
            ItemType::U64 => SampleBufferValue::U64(Vec::with_capacity(chunk_size)),
            ItemType::Bool => SampleBufferValue::Bool(Vec::with_capacity(chunk_size)),
            ItemType::I64 => SampleBufferValue::I64(Vec::with_capacity(chunk_size)),
            ItemType::String => panic!("String type not supported in SampleBuffer"),
            ItemType::DateTime64(_) => panic!("DateTime64 type not supported in SampleBuffer"),
            ItemType::TimeDelta64(_) => panic!("TimeDelta64 type not supported in SampleBuffer"),
        };
        Self {
            items: inner,
            len: 0,
            full_at: chunk_size,
            current_chunk: 0,
        }
    }

    /// Reset the buffer and return any accumulated data as a chunk
    pub fn reset(&mut self) -> Option<Chunk> {
        if self.len == 0 {
            self.current_chunk = 0;
            return None;
        }
        let out = self.finish_chunk();
        self.current_chunk = 0;
        Some(out)
    }

    /// Finalize the current chunk and prepare for a new one
    pub fn finish_chunk(&mut self) -> Chunk {
        let values = match &mut self.items {
            SampleBufferValue::F64(vec) => {
                SampleBufferValue::F64(replace(vec, Vec::with_capacity(vec.len())))
            }
            SampleBufferValue::F32(vec) => {
                SampleBufferValue::F32(replace(vec, Vec::with_capacity(vec.len())))
            }
            SampleBufferValue::U64(vec) => {
                SampleBufferValue::U64(replace(vec, Vec::with_capacity(vec.len())))
            }
            SampleBufferValue::Bool(vec) => {
                SampleBufferValue::Bool(replace(vec, Vec::with_capacity(vec.len())))
            }
            SampleBufferValue::I64(vec) => {
                SampleBufferValue::I64(replace(vec, Vec::with_capacity(vec.len())))
            }
        };

        let output = Chunk {
            chunk_idx: self.current_chunk,
            len: self.len,
            values,
            full_at: self.full_at,
        };

        self.current_chunk += 1;
        self.len = 0;
        output
    }

    /// Creates a temporary chunk containing a copy of the current buffer's data
    pub fn copy_as_chunk(&self) -> Option<Chunk> {
        if self.len == 0 {
            return None;
        }

        let values = match &self.items {
            SampleBufferValue::F64(vec) => SampleBufferValue::F64(vec.clone()),
            SampleBufferValue::F32(vec) => SampleBufferValue::F32(vec.clone()),
            SampleBufferValue::U64(vec) => SampleBufferValue::U64(vec.clone()),
            SampleBufferValue::Bool(vec) => SampleBufferValue::Bool(vec.clone()),
            SampleBufferValue::I64(vec) => SampleBufferValue::I64(vec.clone()),
        };

        Some(Chunk {
            chunk_idx: self.current_chunk,
            len: self.len,
            values,
            full_at: self.full_at,
        })
    }

    /// Add an item to the buffer, returning a chunk if buffer becomes full
    pub fn push(&mut self, item: Value) -> Option<Chunk> {
        assert!(self.len < self.full_at);
        match (&mut self.items, item) {
            (SampleBufferValue::F64(vec), Value::ScalarF64(v)) => vec.push(v),
            (SampleBufferValue::F32(vec), Value::ScalarF32(v)) => vec.push(v),
            (SampleBufferValue::U64(vec), Value::ScalarU64(v)) => vec.push(v),
            (SampleBufferValue::Bool(vec), Value::ScalarBool(v)) => vec.push(v),
            (SampleBufferValue::I64(vec), Value::ScalarI64(v)) => vec.push(v),
            (SampleBufferValue::F64(vec), Value::F64(v)) => vec.extend(v),
            (SampleBufferValue::F32(vec), Value::F32(v)) => vec.extend(v),
            (SampleBufferValue::U64(vec), Value::U64(v)) => vec.extend(v),
            (SampleBufferValue::Bool(vec), Value::Bool(v)) => vec.extend(v),
            (SampleBufferValue::I64(vec), Value::I64(v)) => vec.extend(v),
            _ => panic!("Mismatched item type"),
        }
        self.len += 1;

        if self.len == self.full_at {
            Some(self.finish_chunk())
        } else {
            None
        }
    }
}

/// Create Zarr arrays for storing MCMC trace data
pub fn create_arrays<TStorage: ?Sized>(
    store: Arc<TStorage>,
    group_path: &str,
    item_types: &Vec<(String, ItemType)>,
    item_dims: &Vec<(String, Vec<String>)>,
    n_chains: u64,
    n_draws: u64,
    dim_sizes: &HashMap<String, u64>,
    draw_chunk_size: u64,
) -> Result<HashMap<String, Array<TStorage>>> {
    let mut arrays = HashMap::new();
    for ((name1, item_type), (name2, extra_dims)) in item_types.iter().zip(item_dims.iter()) {
        assert!(name1 == name2);
        let name = name1;
        if ["draw", "chain"].contains(&name.as_str()) {
            continue;
        }
        let dims = std::iter::once("chain".to_string())
            .chain(std::iter::once("draw".to_string()))
            .chain(extra_dims.iter().cloned());
        let extra_shape: Result<Vec<u64>> = extra_dims
            .iter()
            .map(|dim| {
                dim_sizes
                    .get(dim)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Unknown dimension size for dimension {}", dim)
                            .context(format!("Could not write {}/{}", group_path, name))
                    })
                    .copied()
            })
            .collect();
        let extra_shape = extra_shape?;
        let shape: Vec<u64> = std::iter::once(n_chains)
            .chain(std::iter::once(n_draws))
            .chain(extra_shape.clone())
            .collect();
        let zarr_type = match item_type {
            ItemType::F64 => DataType::Float64,
            ItemType::F32 => DataType::Float32,
            ItemType::U64 => DataType::UInt64,
            ItemType::I64 => DataType::Int64,
            ItemType::Bool => DataType::Bool,
            ItemType::String => DataType::String,
            ItemType::DateTime64(unit) => DataType::NumpyDateTime64 {
                unit: match unit {
                    nuts_storable::DateTimeUnit::Seconds => NumpyTimeUnit::Second,
                    nuts_storable::DateTimeUnit::Milliseconds => NumpyTimeUnit::Millisecond,
                    nuts_storable::DateTimeUnit::Microseconds => NumpyTimeUnit::Microsecond,
                    nuts_storable::DateTimeUnit::Nanoseconds => NumpyTimeUnit::Nanosecond,
                },
                scale_factor: NonZero::new(1).unwrap(),
            },
            ItemType::TimeDelta64(unit) => DataType::NumpyTimeDelta64 {
                unit: match unit {
                    nuts_storable::DateTimeUnit::Seconds => NumpyTimeUnit::Second,
                    nuts_storable::DateTimeUnit::Milliseconds => NumpyTimeUnit::Millisecond,
                    nuts_storable::DateTimeUnit::Microseconds => NumpyTimeUnit::Microsecond,
                    nuts_storable::DateTimeUnit::Nanoseconds => NumpyTimeUnit::Nanosecond,
                },
                scale_factor: NonZero::new(1).unwrap(),
            },
        };
        let fill_value = match item_type {
            ItemType::F64 => FillValue::from(f64::NAN),
            ItemType::F32 => FillValue::from(f32::NAN),
            ItemType::U64 => FillValue::from(0u64),
            ItemType::I64 => FillValue::from(0i64),
            ItemType::Bool => FillValue::from(false),
            ItemType::String => FillValue::from(""),
            ItemType::DateTime64(_) => FillValue::new_null(),
            ItemType::TimeDelta64(_) => FillValue::new_null(),
        };
        let grid: Vec<u64> = std::iter::once(1)
            .chain(std::iter::once(draw_chunk_size))
            .chain(extra_shape)
            .collect();
        let array = ArrayBuilder::new(shape, grid, zarr_type, fill_value)
            .dimension_names(Some(dims))
            .build(store.clone(), &format!("{}/{}", group_path, name))?;
        arrays.insert(name.to_string(), array);
    }
    Ok(arrays)
}
