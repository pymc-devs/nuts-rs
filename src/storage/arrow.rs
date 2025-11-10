use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{
    ArrayBuilder, ArrayRef, BooleanBuilder, Float32Builder, Float64Builder, Int64Builder,
    LargeListBuilder, RecordBatch, RecordBatchOptions, StringBuilder, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use nuts_storable::{ItemType, Value};

use crate::storage::{ChainStorage, StorageConfig, TraceStorage};
use crate::{Math, Progress, Settings};

/// Container for different types of Arrow array builders
enum ArrowBuilder {
    Tensor(LargeListBuilder<Box<dyn ArrayBuilder>>),
    Scalar(Box<dyn ArrayBuilder>),
}

impl ArrowBuilder {
    fn new(item_type: ItemType, capacity: usize, shape: Vec<usize>) -> Result<Self> {
        let list_size = shape.iter().product::<usize>();
        let capacity = capacity
            .checked_mul(list_size)
            .ok_or_else(|| anyhow::anyhow!("Capacity overflow when creating ArrowBuilder"))?;

        let value_builder: Box<dyn ArrayBuilder> = match item_type {
            ItemType::F64 => Box::new(Float64Builder::with_capacity(capacity)),
            ItemType::F32 => Box::new(Float32Builder::with_capacity(capacity)),
            ItemType::Bool => Box::new(BooleanBuilder::with_capacity(capacity)),
            ItemType::I64 => Box::new(Int64Builder::with_capacity(capacity)),
            ItemType::U64 => Box::new(UInt64Builder::with_capacity(capacity)),
            ItemType::String => Box::new(StringBuilder::with_capacity(capacity, capacity)),
            ItemType::DateTime64(_) => {
                panic!("DateTime values not supported as values in arrow storage")
            }
            ItemType::TimeDelta64(_) => {
                panic!("TimeDelta values not supported as values in arrow storage")
            }
        };

        if shape.is_empty() {
            Ok(ArrowBuilder::Scalar(value_builder))
        } else {
            let data_type = item_type_to_arrow_type(item_type);
            let list_builder = LargeListBuilder::new(value_builder);
            let list_builder = list_builder.with_field(Field::new("item", data_type, false));
            Ok(ArrowBuilder::Tensor(list_builder))
        }
    }

    fn append_value(&mut self, value: Value) -> Result<()> {
        macro_rules! downcast_builder {
            ($builder:expr, $ty:ty, $variant:ident) => {
                $builder
                    .as_any_mut()
                    .downcast_mut::<$ty>()
                    .ok_or_else(|| anyhow::anyhow!(concat!("Expected ", stringify!($ty))))
            };
        }
        match self {
            ArrowBuilder::Scalar(builder) => match value {
                Value::ScalarF64(v) => {
                    downcast_builder!(builder, Float64Builder, ScalarF64)?.append_value(v);
                }
                Value::ScalarF32(v) => {
                    downcast_builder!(builder, Float32Builder, ScalarF32)?.append_value(v);
                }
                Value::ScalarBool(v) => {
                    downcast_builder!(builder, BooleanBuilder, ScalarBool)?.append_value(v);
                }
                Value::ScalarI64(v) => {
                    downcast_builder!(builder, Int64Builder, ScalarI64)?.append_value(v);
                }
                Value::ScalarU64(v) => {
                    downcast_builder!(builder, UInt64Builder, ScalarU64)?.append_value(v);
                }
                Value::ScalarString(v) => {
                    downcast_builder!(builder, StringBuilder, ScalarString)?.append_value(&v);
                }
                Value::U64(items) => {
                    assert!(items.len() == 1);
                    downcast_builder!(builder, UInt64Builder, U64)?.append_slice(items.as_slice());
                }
                Value::I64(items) => {
                    assert!(items.len() == 1);
                    downcast_builder!(builder, Int64Builder, I64)?.append_slice(items.as_slice());
                }
                Value::F64(items) => {
                    assert!(items.len() == 1);
                    downcast_builder!(builder, Float64Builder, F64)?.append_slice(items.as_slice());
                }
                Value::F32(items) => {
                    assert!(items.len() == 1);
                    downcast_builder!(builder, Float32Builder, F32)?.append_slice(items.as_slice());
                }
                Value::Bool(items) => {
                    assert!(items.len() == 1);
                    downcast_builder!(builder, BooleanBuilder, Bool)?
                        .append_slice(items.as_slice());
                }
                Value::Strings(items) => {
                    let string_builder = downcast_builder!(builder, StringBuilder, Strings)?;
                    for item in items {
                        string_builder.append_value(&item);
                    }
                }
                Value::DateTime64(_, _) => {
                    panic!("DateTime64 scalar values not supported in arrow storage")
                }
                Value::TimeDelta64(_, _) => {
                    panic!("TimeDelta64 scalar values not supported in arrow storage")
                }
            },
            ArrowBuilder::Tensor(list_builder) => {
                match value {
                    Value::F64(v) => {
                        downcast_builder!(list_builder.values(), Float64Builder, F64)?
                            .append_slice(v.as_slice());
                    }
                    Value::F32(v) => {
                        downcast_builder!(list_builder.values(), Float32Builder, F32)?
                            .append_slice(v.as_slice());
                    }
                    Value::I64(v) => {
                        downcast_builder!(list_builder.values(), Int64Builder, I64)?
                            .append_slice(v.as_slice());
                    }
                    Value::U64(v) => {
                        downcast_builder!(list_builder.values(), UInt64Builder, U64)?
                            .append_slice(v.as_slice());
                    }
                    Value::Bool(v) => {
                        downcast_builder!(list_builder.values(), BooleanBuilder, Bool)?
                            .append_slice(v.as_slice());
                    }
                    Value::Strings(items) => {
                        let string_builder =
                            downcast_builder!(list_builder.values(), StringBuilder, Strings)?;
                        for item in items {
                            string_builder.append_value(&item);
                        }
                    }
                    Value::ScalarString(val) => {
                        downcast_builder!(list_builder.values(), StringBuilder, ScalarString)?
                            .append_value(val);
                    }
                    Value::ScalarU64(val) => {
                        downcast_builder!(list_builder.values(), UInt64Builder, ScalarU64)?
                            .append_value(val);
                    }
                    Value::ScalarI64(val) => {
                        downcast_builder!(list_builder.values(), Int64Builder, ScalarI64)?
                            .append_value(val);
                    }
                    Value::ScalarF64(val) => {
                        downcast_builder!(list_builder.values(), Float64Builder, ScalarF64)?
                            .append_value(val);
                    }
                    Value::ScalarF32(val) => {
                        downcast_builder!(list_builder.values(), Float32Builder, ScalarF32)?
                            .append_value(val);
                    }
                    Value::ScalarBool(val) => {
                        downcast_builder!(list_builder.values(), BooleanBuilder, ScalarBool)?
                            .append_value(val);
                    }
                    Value::DateTime64(_, _) => {
                        panic!("DateTime64 scalar values not supported in arrow storage")
                    }
                    Value::TimeDelta64(_, _) => {
                        panic!("TimeDelta64 scalar values not supported in arrow storage")
                    }
                }
                list_builder.append(true);
            }
        }
        Ok(())
    }

    fn append_null(&mut self) -> Result<()> {
        match self {
            ArrowBuilder::Scalar(builder) => {
                if let Some(builder) = builder.as_any_mut().downcast_mut::<Float64Builder>() {
                    builder.append_null();
                } else if let Some(builder) = builder.as_any_mut().downcast_mut::<Float32Builder>()
                {
                    builder.append_null();
                } else if let Some(builder) = builder.as_any_mut().downcast_mut::<Int64Builder>() {
                    builder.append_null();
                } else if let Some(builder) = builder.as_any_mut().downcast_mut::<UInt64Builder>() {
                    builder.append_null();
                } else if let Some(builder) = builder.as_any_mut().downcast_mut::<BooleanBuilder>()
                {
                    builder.append_null();
                } else if let Some(builder) = builder.as_any_mut().downcast_mut::<StringBuilder>() {
                    builder.append_null();
                } else {
                    return Err(anyhow::anyhow!("Unknown builder type for null"));
                }
            }
            ArrowBuilder::Tensor(builder) => builder.append(false),
        }
        Ok(())
    }

    fn finish(&mut self) -> ArrayRef {
        match self {
            ArrowBuilder::Scalar(builder) => Arc::new(builder.finish()),
            ArrowBuilder::Tensor(builder) => Arc::new(builder.finish()),
        }
    }

    fn finish_cloned(&self) -> ArrayRef {
        match self {
            ArrowBuilder::Scalar(builder) => Arc::new(builder.finish_cloned()),
            ArrowBuilder::Tensor(builder) => Arc::new(builder.finish_cloned()),
        }
    }
}

/// Convert ItemType to Arrow DataType
fn item_type_to_arrow_type(item_type: ItemType) -> DataType {
    match item_type {
        ItemType::F64 => DataType::Float64,
        ItemType::F32 => DataType::Float32,
        ItemType::U64 => DataType::UInt64,
        ItemType::I64 => DataType::Int64,
        ItemType::Bool => DataType::Boolean,
        ItemType::String => DataType::Utf8,
        ItemType::DateTime64(_) => {
            panic!("DateTime64 scalar values not supported in arrow storage")
        }
        ItemType::TimeDelta64(_) => {
            panic!("TimeDelta64 scalar values not supported in arrow storage")
        }
    }
}

/// Create a field with tensor extension type if shape is provided
fn create_field_with_shape(
    name: &str,
    item_type: ItemType,
    dims: &Vec<String>,
    dim_sizes: &HashMap<String, u64>,
) -> Result<Field> {
    let arrow_type = item_type_to_arrow_type(item_type);

    if !dims.is_empty() {
        // Multi-dimensional tensor
        let metadata = HashMap::from([
            (
                "dims".to_string(),
                dims.iter().cloned().collect::<Vec<_>>().join(","),
            ),
            (
                "shape".to_string(),
                dims.iter()
                    .map(|dim| {
                        dim_sizes
                            .get(dim)
                            .copied()
                            .map(|size| size.to_string())
                            .expect("Dimension size not found")
                    })
                    .collect::<Vec<_>>()
                    .join(","),
            ),
        ]);

        let inner_field = Field::new("item", arrow_type, false);
        let field = Field::new_large_list(name, inner_field, true);
        let field = field.with_metadata(metadata);
        Ok(field)
    } else {
        Ok(Field::new(name, arrow_type, true))
    }
}
/// Main storage for Arrow MCMC traces
pub struct ArrowTraceStorage {
    stat_types: Vec<(String, ItemType)>,
    draw_types: Vec<(String, ItemType)>,
    stat_dims: Vec<(String, Vec<String>)>,
    draw_dims: Vec<(String, Vec<String>)>,
    stat_dim_sizes: HashMap<String, u64>,
    draw_dim_sizes: HashMap<String, u64>,
    expected_draws: usize,
}

/// Per-chain storage for Arrow MCMC traces
pub struct ArrowChainStorage {
    draw_builders: Vec<(String, ArrowBuilder)>,
    stats_builders: Vec<(String, ArrowBuilder)>,
    stat_types: Vec<(String, ItemType)>,
    draw_types: Vec<(String, ItemType)>,
    stats_dims: Vec<(String, Vec<String>)>,
    draw_dims: Vec<(String, Vec<String>)>,
    stat_dim_sizes: HashMap<String, u64>,
    draw_dim_sizes: HashMap<String, u64>,
    draw_count: usize,
}

/// Final result containing Arrow record batches
#[derive(Clone, Debug)]
pub struct ArrowTrace {
    pub posterior: RecordBatch,
    pub sample_stats: RecordBatch,
}

impl ArrowChainStorage {
    fn new(
        expected_draws: usize,
        stat_types: &[(String, ItemType)],
        draw_types: &[(String, ItemType)],
        stat_dims: &[(String, Vec<String>)],
        draw_dims: &[(String, Vec<String>)],
        stat_dim_sizes: &HashMap<String, u64>,
        draw_dim_sizes: &HashMap<String, u64>,
    ) -> Result<Self> {
        let draw_builders = draw_types
            .iter()
            .zip(draw_dims.iter())
            .map(|((name, item_type), (name2, dims))| {
                assert_eq!(
                    name, name2,
                    "Draw types and dims must have matching names and order"
                );
                let shape = dims
                    .iter()
                    .map(|dim| {
                        draw_dim_sizes
                            .get(dim)
                            .copied()
                            .map(|x| x as usize)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Unknown dimension size for dimension {}", dim)
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((
                    name.clone(),
                    ArrowBuilder::new(*item_type, expected_draws, shape)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let stats_builders = stat_types
            .iter()
            .zip(stat_dims.iter())
            .map(|((name, item_type), (name2, dims))| {
                assert_eq!(
                    name, name2,
                    "Draw types and dims must have matching names and order"
                );
                let shape = dims
                    .iter()
                    .map(|dim| {
                        stat_dim_sizes
                            .get(dim)
                            .copied()
                            .map(|x| x as usize)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Unknown dimension size for dimension {}", dim)
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((
                    name.clone(),
                    ArrowBuilder::new(*item_type, expected_draws, shape)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            draw_builders,
            stats_builders,
            stat_types: stat_types.to_vec(),
            draw_types: draw_types.to_vec(),
            stats_dims: stat_dims.to_vec(),
            draw_dims: draw_dims.to_vec(),
            stat_dim_sizes: stat_dim_sizes.clone(),
            draw_dim_sizes: draw_dim_sizes.clone(),
            draw_count: 0,
        })
    }

    fn finalize_builders(mut self) -> Result<ArrowTrace> {
        // Create posterior schema and arrays

        let posterior_fields = self
            .draw_types
            .iter()
            .zip(self.draw_dims.iter())
            .map(|((name, item_type), (_, dims))| {
                create_field_with_shape(name, *item_type, dims, &self.draw_dim_sizes)
            })
            .collect::<Result<Vec<Field>>>()?;

        let posterior_arrays: Vec<ArrayRef> = self
            .draw_builders
            .iter_mut()
            .map(|(_, builder)| builder.finish())
            .collect();

        let posterior_schema = Schema::new(posterior_fields);
        let posterior_options = RecordBatchOptions::new().with_row_count(Some(self.draw_count));
        let posterior = RecordBatch::try_new_with_options(
            Arc::new(posterior_schema),
            posterior_arrays,
            &posterior_options,
        )
        .context("Could not convert posterior to RecordBatch")?;

        // Create stats schema and arrays
        let stats_fields = self
            .stat_types
            .iter()
            .zip(self.stats_dims.iter())
            .map(|((name, item_type), (_, dims))| {
                create_field_with_shape(name, *item_type, dims, &self.stat_dim_sizes)
            })
            .collect::<Result<Vec<Field>>>()?;

        let stats_arrays: Vec<ArrayRef> = self
            .stats_builders
            .iter_mut()
            .map(|(_, builder)| builder.finish())
            .collect();

        let stats_schema = Schema::new(stats_fields);
        let stats_options = RecordBatchOptions::new().with_row_count(Some(self.draw_count));
        let sample_stats =
            RecordBatch::try_new_with_options(Arc::new(stats_schema), stats_arrays, &stats_options)
                .context("Could not convert sample stats to RecordBatch")?;

        Ok(ArrowTrace {
            posterior,
            sample_stats,
        })
    }
}

impl ChainStorage for ArrowChainStorage {
    type Finalized = ArrowTrace;

    fn record_sample(
        &mut self,
        _settings: &impl Settings,
        stats: Vec<(&str, Option<Value>)>,
        draws: Vec<(&str, Option<Value>)>,
        _info: &Progress,
    ) -> Result<()> {
        stats
            .into_iter()
            .zip(self.stats_builders.iter_mut())
            .try_for_each(|((name, value), (expected_name, builder))| {
                if name != expected_name {
                    panic!(
                        "Draw name mismatch: expected {}, got {}",
                        expected_name, name
                    );
                }

                if let Some(value) = value {
                    builder.append_value(value)?;
                } else {
                    builder.append_null()?;
                }
                Ok::<_, anyhow::Error>(())
            })?;

        draws
            .into_iter()
            .zip(self.draw_builders.iter_mut())
            .try_for_each(|((name, value), (expected_name, builder))| {
                if name != expected_name {
                    panic!(
                        "Draw name mismatch: expected {}, got {}",
                        expected_name, name
                    );
                }

                if let Some(value) = value {
                    builder.append_value(value)?;
                } else {
                    builder.append_null()?;
                }
                Ok::<_, anyhow::Error>(())
            })?;

        self.draw_count += 1;

        Ok(())
    }

    fn finalize(self) -> Result<Self::Finalized> {
        self.finalize_builders()
    }

    fn flush(&self) -> Result<()> {
        // No-op for in-memory storage
        Ok(())
    }

    fn inspect(&self) -> Result<Option<Self::Finalized>> {
        let posterior_fields = self
            .draw_types
            .iter()
            .zip(self.draw_dims.iter())
            .map(|((name, item_type), (_, dims))| {
                create_field_with_shape(name, *item_type, dims, &self.draw_dim_sizes)
            })
            .collect::<Result<Vec<Field>>>()?;

        let posterior_arrays: Vec<ArrayRef> = self
            .draw_builders
            .iter()
            .map(|(_, builder)| builder.finish_cloned())
            .collect();

        let posterior_schema = Schema::new(posterior_fields);
        let posterior_options = RecordBatchOptions::new().with_row_count(Some(self.draw_count));
        let posterior = RecordBatch::try_new_with_options(
            Arc::new(posterior_schema),
            posterior_arrays,
            &posterior_options,
        )
        .context("Could not convert posterior to RecordBatch")?;

        // Create stats schema and arrays
        let stats_fields = self
            .stat_types
            .iter()
            .zip(self.stats_dims.iter())
            .map(|((name, item_type), (_, dims))| {
                create_field_with_shape(name, *item_type, dims, &self.stat_dim_sizes)
            })
            .collect::<Result<Vec<Field>>>()?;

        let stats_arrays: Vec<ArrayRef> = self
            .stats_builders
            .iter()
            .map(|(_, builder)| builder.finish_cloned())
            .collect();

        let stats_schema = Schema::new(stats_fields);
        let stats_options = RecordBatchOptions::new().with_row_count(Some(self.draw_count));
        let sample_stats =
            RecordBatch::try_new_with_options(Arc::new(stats_schema), stats_arrays, &stats_options)
                .context("Could not convert sample stats to RecordBatch")?;

        Ok(Some(ArrowTrace {
            posterior,
            sample_stats,
        }))
    }
}

/// Configuration for Arrow-based MCMC storage.
///
/// This storage backend keeps all data in memory using Arrow's columnar format.
/// It's efficient for moderate-sized datasets and provides interoperability
/// with other Arrow-based tools. Multi-dimensional parameters
/// are stored as Arrow LargeList arrays with custom metadata containing
/// dimension names.
pub struct ArrowConfig {}

impl ArrowConfig {
    /// Create a new Arrow configuration.
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ArrowConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageConfig for ArrowConfig {
    type Storage = ArrowTraceStorage;

    fn new_trace<M: Math>(self, settings: &impl Settings, math: &M) -> Result<Self::Storage> {
        let stat_types = settings.stat_types(math);
        let draw_types = settings.data_types(math);
        let stat_dims = settings.stat_dims_all(math);
        let draw_dims = settings.data_dims_all(math);
        let stat_dim_sizes = settings.stat_dim_sizes(math);
        let draw_dim_sizes = math.dim_sizes();

        // Calculate expected total draws (warmup + sampling)
        let expected_draws = (settings.hint_num_tune() + settings.hint_num_draws()) as usize;

        Ok(ArrowTraceStorage {
            stat_types,
            draw_types,
            stat_dims,
            draw_dims,
            stat_dim_sizes,
            draw_dim_sizes,
            expected_draws,
        })
    }
}

impl TraceStorage for ArrowTraceStorage {
    type ChainStorage = ArrowChainStorage;
    type Finalized = Vec<ArrowTrace>;

    fn initialize_trace_for_chain(&self, _chain_id: u64) -> Result<Self::ChainStorage> {
        ArrowChainStorage::new(
            self.expected_draws,
            &self.stat_types,
            &self.draw_types,
            &self.stat_dims,
            &self.draw_dims,
            &self.stat_dim_sizes,
            &self.draw_dim_sizes,
        )
    }

    fn finalize(
        self,
        traces: Vec<Result<<Self::ChainStorage as ChainStorage>::Finalized>>,
    ) -> Result<(Option<anyhow::Error>, Self::Finalized)> {
        let mut results = Vec::new();
        let mut first_error = None;

        for trace in traces {
            match trace {
                Ok(trace) => results.push(trace),
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
        Ok((first_error, results))
    }

    fn inspect(
        &self,
        traces: Vec<Result<Option<<Self::ChainStorage as ChainStorage>::Finalized>>>,
    ) -> Result<(Option<anyhow::Error>, Self::Finalized)> {
        let mut results = Vec::new();
        let mut first_error = None;

        for trace in traces {
            match trace {
                Ok(Some(trace)) => results.push(trace),
                Ok(None) => {}
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
        Ok((first_error, results))
    }
}
