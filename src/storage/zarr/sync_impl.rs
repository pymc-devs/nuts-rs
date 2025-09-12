use std::collections::HashMap;
use std::iter::once;
use std::sync::Arc;

use anyhow::{Context, Result};
use nuts_storable::{ItemType, Value};
use zarrs::array::{ArrayBuilder, DataType, FillValue};
use zarrs::array_subset::ArraySubset;
use zarrs::group::GroupBuilder;
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

use super::common::{Chunk, SampleBuffer, SampleBufferValue};
use super::create_arrays;
use crate::storage::{ChainStorage, StorageConfig, TraceStorage};
use crate::{Math, Progress, Settings};

pub type Array = zarrs::array::Array<dyn ReadableWritableListableStorageTraits>;

struct ArrayCollection {
    pub warmup_param_arrays: HashMap<String, Array>,
    pub sample_param_arrays: HashMap<String, Array>,
    pub warmup_draw_arrays: HashMap<String, Array>,
    pub sample_draw_arrays: HashMap<String, Array>,
}

/// Store coordinates in zarr arrays
pub fn store_coords(
    store: ReadableWritableListableStorage,
    group: String,
    coords: &HashMap<String, Value>,
) -> Result<()> {
    for (name, coord) in coords {
        let (data_type, len, fill_value) = match coord {
            &Value::F64(ref v) => (DataType::Float64, v.len(), FillValue::from(f64::NAN)),
            &Value::F32(ref v) => (DataType::Float32, v.len(), FillValue::from(f32::NAN)),
            &Value::U64(ref v) => (DataType::UInt64, v.len(), FillValue::from(0u64)),
            &Value::I64(ref v) => (DataType::Int64, v.len(), FillValue::from(0i64)),
            &Value::Bool(ref v) => (DataType::Bool, v.len(), FillValue::from(false)),
            &Value::Strings(ref v) => (DataType::String, v.len(), FillValue::from("")),
            _ => panic!("Unsupported coordinate type for {}", name),
        };
        let name: &String = name;
        let coord_array = ArrayBuilder::new(
            vec![len as u64],
            data_type,
            vec![len as u64].try_into().expect("Invalid chunk size"),
            fill_value,
        )
        .dimension_names(Some(vec![name.to_string()]))
        .build(store.clone(), &format!("{}/{}", group, name))?;
        let subset = vec![0];
        match coord {
            &Value::F64(ref v) => coord_array.store_chunk_elements::<f64>(&subset, v)?,
            &Value::F32(ref v) => coord_array.store_chunk_elements::<f32>(&subset, v)?,
            &Value::U64(ref v) => coord_array.store_chunk_elements::<u64>(&subset, v)?,
            &Value::I64(ref v) => coord_array.store_chunk_elements::<i64>(&subset, v)?,
            &Value::Bool(ref v) => coord_array.store_chunk_elements::<bool>(&subset, v)?,
            &Value::Strings(ref v) => coord_array.store_chunk_elements::<String>(&subset, v)?,
            _ => unreachable!(),
        }
        coord_array.store_metadata()?;
    }
    Ok(())
}

/// Main storage for Zarr MCMC traces
pub struct ZarrTraceStorage {
    arrays: Arc<ArrayCollection>,
    draw_chunk_size: u64,
    param_types: Vec<(String, ItemType)>,
    draw_types: Vec<(String, ItemType)>,
}

/// Per-chain storage for Zarr MCMC traces
pub struct ZarrChainStorage {
    draw_buffers: HashMap<String, SampleBuffer>,
    stats_buffers: HashMap<String, SampleBuffer>,
    arrays: Arc<ArrayCollection>,
    chain: u64,
    last_sample_was_warmup: bool,
}

/// Write a chunk of data to a Zarr array
fn store_zarr_chunk(array: &Array, data: Chunk, chain_chunk_index: u64) -> Result<()> {
    let rank = array.chunk_grid().dimensionality();
    assert!(rank >= 2);
    // append one value per rank
    let chunk_vec: Vec<_> = once(chain_chunk_index as u64)
        .chain(once(data.chunk_idx as u64))
        .chain(once(0).cycle().take(rank - 2))
        .collect();
    let chunk = &chunk_vec[..];

    let result = if data.is_full() {
        match data.values {
            SampleBufferValue::F64(v) => array.store_chunk_elements::<f64>(&chunk, &v),
            SampleBufferValue::F32(v) => array.store_chunk_elements::<f32>(&chunk, &v),
            SampleBufferValue::U64(v) => array.store_chunk_elements::<u64>(&chunk, &v),
            SampleBufferValue::I64(v) => array.store_chunk_elements::<i64>(&chunk, &v),
            SampleBufferValue::Bool(v) => array.store_chunk_elements::<bool>(&chunk, &v),
        }
    } else {
        let mut shape: Vec<_> = array.shape().iter().cloned().collect();
        assert!(shape.len() >= 2);
        shape[0] = 1;
        shape[1] = data.len as u64;
        let chunk_subset = ArraySubset::new_with_shape(shape);
        match data.values {
            SampleBufferValue::F64(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array.store_chunk_subset_elements(&chunk, &chunk_subset, &v)
            }
            SampleBufferValue::F32(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array.store_chunk_subset_elements(&chunk, &chunk_subset, &v)
            }
            SampleBufferValue::U64(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array.store_chunk_subset_elements(&chunk, &chunk_subset, &v)
            }
            SampleBufferValue::I64(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array.store_chunk_subset_elements(&chunk, &chunk_subset, &v)
            }
            SampleBufferValue::Bool(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array.store_chunk_subset_elements(&chunk, &chunk_subset, &v)
            }
        }
    };

    result.context(format!(
        "Failed to store chunk for variable {} at chunk {} with length {}",
        array.path(),
        data.chunk_idx,
        data.len
    ))?;
    Ok(())
}

impl ZarrChainStorage {
    /// Create a new chain storage with buffers for parameters and samples
    fn new(
        arrays: Arc<ArrayCollection>,
        param_types: &Vec<(String, ItemType)>,
        draw_types: &Vec<(String, ItemType)>,
        buffer_size: u64,
        chain: u64,
    ) -> Self {
        let draw_buffers = draw_types
            .iter()
            .map(|(name, item_type)| (name.clone(), SampleBuffer::new(*item_type, buffer_size)))
            .collect();

        let stats_buffers = param_types
            .iter()
            .map(|(name, item_type)| (name.clone(), SampleBuffer::new(*item_type, buffer_size)))
            .collect();
        Self {
            draw_buffers,
            stats_buffers,
            arrays,
            chain,
            last_sample_was_warmup: true,
        }
    }

    /// Store a parameter value, writing to Zarr when buffer is full
    fn push_param(&mut self, name: &str, value: Value, is_warmup: bool) -> Result<()> {
        if ["draw", "chain"].contains(&name) {
            return Ok(());
        }
        let Some(buffer) = self.stats_buffers.get_mut(name) else {
            panic!("Unknown param name: {}", name);
        };
        if let Some(chunk) = buffer.push(value) {
            let array = if is_warmup {
                &self.arrays.warmup_param_arrays[name]
            } else {
                &self.arrays.sample_param_arrays[name]
            };
            store_zarr_chunk(array, chunk, self.chain)?;
        }
        Ok(())
    }

    /// Store a draw value, writing to Zarr when buffer is full
    fn push_draw(&mut self, name: &str, value: Value, is_warmup: bool) -> Result<()> {
        if ["draw", "chain"].contains(&name) {
            return Ok(());
        }
        let Some(buffer) = self.draw_buffers.get_mut(name) else {
            panic!("Unknown posterior variable name: {}", name);
        };
        if let Some(chunk) = buffer.push(value) {
            let array = if is_warmup {
                &self.arrays.warmup_draw_arrays[name]
            } else {
                &self.arrays.sample_draw_arrays[name]
            };
            store_zarr_chunk(array, chunk, self.chain)?;
        }
        Ok(())
    }
}

impl ChainStorage for ZarrChainStorage {
    type Finalized = ();

    fn record_sample(
        &mut self,
        _settings: &impl Settings,
        stats: Vec<(&str, Option<Value>)>,
        draws: Vec<(&str, Option<Value>)>,
        info: &Progress,
    ) -> Result<()> {
        let is_first_draw = self.last_sample_was_warmup && !info.tuning;
        if is_first_draw {
            for (key, buffer) in self.draw_buffers.iter_mut() {
                if let Some(chunk) = buffer.reset() {
                    store_zarr_chunk(&self.arrays.warmup_draw_arrays[key], chunk, self.chain)?;
                }
            }
            for (key, buffer) in self.stats_buffers.iter_mut() {
                if let Some(chunk) = buffer.reset() {
                    store_zarr_chunk(&self.arrays.warmup_param_arrays[key], chunk, self.chain)?;
                }
            }
            self.last_sample_was_warmup = false;
        }

        for (name, value) in stats {
            if let Some(value) = value {
                self.push_param(name, value, info.tuning)?;
            }
        }
        for (name, value) in draws {
            if let Some(value) = value {
                self.push_draw(name, value, info.tuning)?;
            } else {
                panic!("Missing draw value for {}", name);
            }
        }
        Ok(())
    }

    /// Flush remaining samples and finalize storage
    fn finalize(self) -> Result<Self::Finalized> {
        for (key, mut buffer) in self.draw_buffers.into_iter() {
            if let Some(chunk) = buffer.reset() {
                let array = if self.last_sample_was_warmup {
                    &self.arrays.warmup_draw_arrays[&key]
                } else {
                    &self.arrays.sample_draw_arrays[&key]
                };
                store_zarr_chunk(array, chunk, self.chain)?;
            }
        }
        for (key, mut buffer) in self.stats_buffers.into_iter() {
            if let Some(chunk) = buffer.reset() {
                let array = if self.last_sample_was_warmup {
                    &self.arrays.warmup_param_arrays[&key]
                } else {
                    &self.arrays.sample_param_arrays[&key]
                };
                store_zarr_chunk(array, chunk, self.chain)?;
            }
        }
        Ok(())
    }

    /// Write current buffer contents to storage without modifying the buffers
    fn flush(&self) -> Result<()> {
        // Flush all draw buffers that have data
        for (key, buffer) in &self.draw_buffers {
            if let Some(temp_chunk) = buffer.copy_as_chunk() {
                // Store the temporary chunk
                let array = if self.last_sample_was_warmup {
                    &self.arrays.warmup_draw_arrays[key]
                } else {
                    &self.arrays.sample_draw_arrays[key]
                };
                store_zarr_chunk(array, temp_chunk, self.chain)?;
            }
        }

        // Flush all stats buffers that have data
        for (key, buffer) in &self.stats_buffers {
            if let Some(temp_chunk) = buffer.copy_as_chunk() {
                // Store the temporary chunk
                let array = if self.last_sample_was_warmup {
                    &self.arrays.warmup_param_arrays[key]
                } else {
                    &self.arrays.sample_param_arrays[key]
                };
                store_zarr_chunk(array, temp_chunk, self.chain)?;
            }
        }

        Ok(())
    }
}

/// Configuration for Zarr-based MCMC storage.
///
/// This is the main interface for configuring Zarr storage for MCMC sampling.
/// Zarr provides efficient, chunked storage for large datasets with good
/// compression and parallel I/O support.
///
/// The storage organizes data into groups:
/// - `posterior/` - posterior samples
/// - `sample_stats/` - sampling statistics
/// - `warmup_posterior/` - warmup samples (optional)
/// - `warmup_sample_stats/` - warmup statistics (optional)
pub struct ZarrConfig {
    store: ReadableWritableListableStorage,
    group_path: Option<String>,
    draw_chunk_size: u64,
    store_warmup: bool,
}

impl ZarrConfig {
    /// Create a new Zarr configuration with default settings.
    ///
    /// Default settings:
    /// - `draw_chunk_size`: 100 samples per chunk
    /// - `store_warmup`: true (warmup samples are stored)
    /// - `group_path`: root of the store
    pub fn new(store: ReadableWritableListableStorage) -> Self {
        Self {
            store,
            group_path: None,
            draw_chunk_size: 100,
            store_warmup: true,
        }
    }

    /// Set the number of samples per chunk.
    ///
    /// Larger chunks use more memory but may provide better I/O performance.
    /// Smaller chunks provide more frequent flushing and lower memory usage.
    pub fn with_chunk_size(mut self, chunk_size: u64) -> Self {
        self.draw_chunk_size = chunk_size;
        self
    }

    /// Set the group path within the Zarr store.
    ///
    /// If not set, data is stored at the root of the store.
    pub fn with_group_path<S: Into<String>>(mut self, path: S) -> Self {
        self.group_path = Some(path.into());
        self
    }

    /// Configure whether to store warmup samples.
    ///
    /// When true, warmup samples are stored in separate groups.
    /// When false, only post-warmup samples are stored.
    pub fn store_warmup(mut self, store: bool) -> Self {
        self.store_warmup = store;
        self
    }
}

impl StorageConfig for ZarrConfig {
    type Storage = ZarrTraceStorage;

    fn new_trace<M: Math>(self, settings: &impl Settings, math: &M) -> Result<Self::Storage> {
        let n_chains = settings.num_chains() as u64;
        let n_tune = settings.hint_num_tune() as u64;
        let n_draws = settings.hint_num_draws() as u64;

        let param_types = settings.stat_types(math);
        let draw_types = settings.data_types(math);

        let param_dims = settings.stat_dims_all(math);
        let draw_dims = settings.data_dims_all(math);

        let draw_dim_sizes = math.dim_sizes();
        let stat_dim_sizes = settings.stat_dim_sizes(math);

        let mut group_path = self.group_path.unwrap_or_else(|| "".to_string());
        if !group_path.ends_with('/') {
            group_path.push('/');
        }
        let store = self.store;
        let draw_chunk_size = self.draw_chunk_size;

        let mut root = GroupBuilder::new().build(store.clone(), &group_path)?;

        let attrs = root.attributes_mut();
        attrs.insert(
            "sampler".to_string(),
            serde_json::Value::String(env!("CARGO_PKG_NAME").to_string()),
        );
        attrs.insert(
            "sampler_version".to_string(),
            serde_json::Value::String(env!("CARGO_PKG_VERSION").to_string()),
        );
        attrs.insert(
            "sampler_settings".to_string(),
            serde_json::to_value(settings).context("Could not serialize sampler settings")?,
        );
        root.store_metadata()?;

        GroupBuilder::new()
            .build(store.clone(), &format!("{}warmup_posterior", group_path))?
            .store_metadata()?;
        GroupBuilder::new()
            .build(store.clone(), &format!("{}warmup_sample_stats", group_path))?
            .store_metadata()?;
        GroupBuilder::new()
            .build(store.clone(), &format!("{}posterior", group_path))?
            .store_metadata()?;
        GroupBuilder::new()
            .build(store.clone(), &format!("{}sample_stats", group_path))?
            .store_metadata()?;

        let warmup_param_arrays = create_arrays(
            store.clone(),
            &format!("{}warmup_sample_stats", group_path),
            &param_types,
            &param_dims,
            n_chains,
            n_tune,
            &stat_dim_sizes,
            self.draw_chunk_size,
        )?;
        for array in warmup_param_arrays.values() {
            array.store_metadata()?;
        }
        let sample_param_arrays = create_arrays(
            store.clone(),
            &format!("{}sample_stats", group_path),
            &param_types,
            &param_dims,
            n_chains,
            n_draws,
            &stat_dim_sizes,
            self.draw_chunk_size,
        )?;
        for array in sample_param_arrays.values() {
            array.store_metadata()?;
        }
        let warmup_draw_arrays = create_arrays(
            store.clone(),
            &format!("{}warmup_posterior", group_path),
            &draw_types,
            &draw_dims,
            n_chains,
            n_tune,
            &draw_dim_sizes,
            self.draw_chunk_size,
        )?;
        for array in warmup_draw_arrays.values() {
            array.store_metadata()?;
        }
        let sample_draw_arrays = create_arrays(
            store.clone(),
            &format!("{}posterior", group_path),
            &draw_types,
            &draw_dims,
            n_chains,
            n_draws,
            &draw_dim_sizes,
            self.draw_chunk_size,
        )?;
        for array in sample_draw_arrays.values() {
            array.store_metadata()?;
        }
        let trace_storage = ArrayCollection {
            warmup_param_arrays,
            sample_param_arrays,
            warmup_draw_arrays,
            sample_draw_arrays,
        };

        let draw_coords = math.coords();
        let stat_coords = settings.stat_coords(math);

        store_coords(
            store.clone(),
            format!("{}posterior", &group_path),
            &draw_coords,
        )?;
        store_coords(
            store.clone(),
            format!("{}warmup_posterior", &group_path),
            &draw_coords,
        )?;
        store_coords(
            store.clone(),
            format!("{}sample_stats", &group_path),
            &stat_coords,
        )?;
        store_coords(
            store.clone(),
            format!("{}warmup_sample_stats", &group_path),
            &stat_coords,
        )?;

        Ok(ZarrTraceStorage {
            arrays: Arc::new(trace_storage),
            param_types,
            draw_types,
            draw_chunk_size,
        })
    }
}

impl TraceStorage for ZarrTraceStorage {
    type ChainStorage = ZarrChainStorage;

    type Finalized = ();

    fn initialize_trace_for_chain(&self, chain_id: u64) -> Result<Self::ChainStorage> {
        Ok(ZarrChainStorage::new(
            self.arrays.clone(),
            &self.param_types,
            &self.draw_types,
            self.draw_chunk_size,
            chain_id as _,
        ))
    }

    fn finalize(
        self,
        traces: Vec<Result<<Self::ChainStorage as ChainStorage>::Finalized>>,
    ) -> Result<(Option<anyhow::Error>, Self::Finalized)> {
        for trace in traces {
            if let Err(err) = trace {
                return Ok((Some(err), ()));
            }
        }
        Ok((None, ()))
    }
}
