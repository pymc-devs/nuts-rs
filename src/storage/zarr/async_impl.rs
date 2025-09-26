use std::collections::HashMap;
use std::iter::once;
use std::sync::Arc;
use tokio::task::JoinHandle;

use anyhow::{Context, Result};
use nuts_storable::{ItemType, Value};
use zarrs::array::{ArrayBuilder, DataType, FillValue};
use zarrs::array_subset::ArraySubset;
use zarrs::group::GroupBuilder;
use zarrs::storage::{
    AsyncReadableWritableListableStorage, AsyncReadableWritableListableStorageTraits,
};

use super::common::{Chunk, SampleBuffer, SampleBufferValue, create_arrays};
use crate::storage::{ChainStorage, StorageConfig, TraceStorage};
use crate::{Math, Progress, Settings};

pub type Array = Arc<zarrs::array::Array<dyn AsyncReadableWritableListableStorageTraits + 'static>>;

struct ArrayCollection {
    pub warmup_param_arrays: HashMap<String, Array>,
    pub sample_param_arrays: HashMap<String, Array>,
    pub warmup_draw_arrays: HashMap<String, Array>,
    pub sample_draw_arrays: HashMap<String, Array>,
}

/// Main storage for async Zarr MCMC traces
pub struct ZarrAsyncTraceStorage {
    arrays: Arc<ArrayCollection>,
    draw_chunk_size: u64,
    param_types: Vec<(String, ItemType)>,
    draw_types: Vec<(String, ItemType)>,
    rt_handle: tokio::runtime::Handle,
}

/// Per-chain storage for async Zarr MCMC traces
pub struct ZarrAsyncChainStorage {
    draw_buffers: HashMap<String, SampleBuffer>,
    stats_buffers: HashMap<String, SampleBuffer>,
    arrays: Arc<ArrayCollection>,
    chain: u64,
    last_sample_was_warmup: bool,
    pending_writes: Vec<JoinHandle<Result<()>>>,
    rt_handle: tokio::runtime::Handle,
}

/// Write a chunk of data to a Zarr array asynchronously
async fn store_zarr_chunk_async(array: Array, data: Chunk, chain_chunk_index: u64) -> Result<()> {
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
            SampleBufferValue::F64(v) => array.async_store_chunk_elements::<f64>(&chunk, &v).await,
            SampleBufferValue::F32(v) => array.async_store_chunk_elements::<f32>(&chunk, &v).await,
            SampleBufferValue::U64(v) => array.async_store_chunk_elements::<u64>(&chunk, &v).await,
            SampleBufferValue::I64(v) => array.async_store_chunk_elements::<i64>(&chunk, &v).await,
            SampleBufferValue::Bool(v) => {
                array.async_store_chunk_elements::<bool>(&chunk, &v).await
            }
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
                array
                    .async_store_chunk_subset_elements(&chunk, &chunk_subset, &v)
                    .await
            }
            SampleBufferValue::F32(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array
                    .async_store_chunk_subset_elements(&chunk, &chunk_subset, &v)
                    .await
            }
            SampleBufferValue::U64(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array
                    .async_store_chunk_subset_elements(&chunk, &chunk_subset, &v)
                    .await
            }
            SampleBufferValue::I64(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array
                    .async_store_chunk_subset_elements(&chunk, &chunk_subset, &v)
                    .await
            }
            SampleBufferValue::Bool(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array
                    .async_store_chunk_subset_elements(&chunk, &chunk_subset, &v)
                    .await
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

/// Store a chunk synchronously by blocking on the async operation
fn store_zarr_chunk_sync(
    handle: &tokio::runtime::Handle,
    array: &Array,
    data: Chunk,
    chain_chunk_index: u64,
) -> Result<()> {
    let array = array.clone();
    handle.block_on(async move { store_zarr_chunk_async(array, data, chain_chunk_index).await })
}

/// Store coordinates in zarr arrays
async fn store_coords(
    store: AsyncReadableWritableListableStorage,
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
        let coord_array =
            ArrayBuilder::new(vec![len as u64], vec![len as u64], data_type, fill_value)
                .dimension_names(Some(vec![name.to_string()]))
                .build(store.clone(), &format!("{}/{}", group, name))?;
        let subset = vec![0];
        match coord {
            &Value::F64(ref v) => {
                coord_array
                    .async_store_chunk_elements::<f64>(&subset, v)
                    .await?
            }
            &Value::F32(ref v) => {
                coord_array
                    .async_store_chunk_elements::<f32>(&subset, v)
                    .await?
            }
            &Value::U64(ref v) => {
                coord_array
                    .async_store_chunk_elements::<u64>(&subset, v)
                    .await?
            }
            &Value::I64(ref v) => {
                coord_array
                    .async_store_chunk_elements::<i64>(&subset, v)
                    .await?
            }
            &Value::Bool(ref v) => {
                coord_array
                    .async_store_chunk_elements::<bool>(&subset, v)
                    .await?
            }
            &Value::Strings(ref v) => {
                coord_array
                    .async_store_chunk_elements::<String>(&subset, v)
                    .await?
            }
            _ => unreachable!(),
        }
        coord_array.async_store_metadata().await?;
    }
    Ok(())
}

impl ZarrAsyncChainStorage {
    /// Create a new chain storage with buffers for parameters and samples
    fn new(
        arrays: Arc<ArrayCollection>,
        param_types: &Vec<(String, ItemType)>,
        draw_types: &Vec<(String, ItemType)>,
        buffer_size: u64,
        chain: u64,
        rt_handle: tokio::runtime::Handle,
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
            pending_writes: Vec::new(),
            rt_handle,
        }
    }

    /// Store a parameter value, spawning async write when buffer is full
    fn push_param(&mut self, name: &str, value: Value, is_warmup: bool) -> Result<()> {
        if ["draw", "chain"].contains(&name) {
            return Ok(());
        }
        let Some(buffer) = self.stats_buffers.get_mut(name) else {
            panic!("Unknown param name: {}", name);
        };
        if let Some(chunk) = buffer.push(value) {
            let array = if is_warmup {
                self.arrays.warmup_param_arrays[name].clone()
            } else {
                self.arrays.sample_param_arrays[name].clone()
            };
            let chain = self.chain;
            let handle = self
                .rt_handle
                .spawn(async move { store_zarr_chunk_async(array, chunk, chain).await });
            self.pending_writes.push(handle);
        }
        Ok(())
    }

    /// Store a draw value, spawning async write when buffer is full
    fn push_draw(&mut self, name: &str, value: Value, is_warmup: bool) -> Result<()> {
        if ["draw", "chain"].contains(&name) {
            return Ok(());
        }
        let Some(buffer) = self.draw_buffers.get_mut(name) else {
            panic!("Unknown posterior variable name: {}", name);
        };
        if let Some(chunk) = buffer.push(value) {
            let array = if is_warmup {
                self.arrays.warmup_draw_arrays[name].clone()
            } else {
                self.arrays.sample_draw_arrays[name].clone()
            };
            let chain = self.chain;
            let handle = self
                .rt_handle
                .spawn(async move { store_zarr_chunk_async(array, chunk, chain).await });
            self.pending_writes.push(handle);
        }
        Ok(())
    }
}

impl ChainStorage for ZarrAsyncChainStorage {
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
                    let array = self.arrays.warmup_draw_arrays[key].clone();
                    let chain = self.chain;
                    let handle = self
                        .rt_handle
                        .spawn(async move { store_zarr_chunk_async(array, chunk, chain).await });
                    self.pending_writes.push(handle);
                }
            }
            for (key, buffer) in self.stats_buffers.iter_mut() {
                if let Some(chunk) = buffer.reset() {
                    let array = self.arrays.warmup_param_arrays[key].clone();
                    let chain = self.chain;
                    let handle = self
                        .rt_handle
                        .spawn(async move { store_zarr_chunk_async(array, chunk, chain).await });
                    self.pending_writes.push(handle);
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

    /// Flush remaining samples and finalize storage, joining all pending writes
    fn finalize(self) -> Result<Self::Finalized> {
        // Handle remaining buffers synchronously
        for (key, mut buffer) in self.draw_buffers.into_iter() {
            if let Some(chunk) = buffer.reset() {
                let array = if self.last_sample_was_warmup {
                    &self.arrays.warmup_draw_arrays[&key]
                } else {
                    &self.arrays.sample_draw_arrays[&key]
                };
                store_zarr_chunk_sync(&self.rt_handle, array, chunk, self.chain)?;
            }
        }
        for (key, mut buffer) in self.stats_buffers.into_iter() {
            if let Some(chunk) = buffer.reset() {
                let array = if self.last_sample_was_warmup {
                    &self.arrays.warmup_param_arrays[&key]
                } else {
                    &self.arrays.sample_param_arrays[&key]
                };
                store_zarr_chunk_sync(&self.rt_handle, array, chunk, self.chain)?;
            }
        }

        // Join all pending writes
        self.rt_handle.block_on(async move {
            for join_handle in self.pending_writes {
                let _ = join_handle
                    .await
                    .context("Failed to await async chunk write operation")?;
            }
            Ok::<(), anyhow::Error>(())
        })?;

        Ok(())
    }

    /// Write current buffer contents to storage without modifying the buffers
    fn flush(&self) -> Result<()> {
        // Flush all draw buffers that have data (synchronously)
        for (key, buffer) in &self.draw_buffers {
            if let Some(temp_chunk) = buffer.copy_as_chunk() {
                let array = if self.last_sample_was_warmup {
                    &self.arrays.warmup_draw_arrays[key]
                } else {
                    &self.arrays.sample_draw_arrays[key]
                };
                store_zarr_chunk_sync(&self.rt_handle, array, temp_chunk, self.chain)?;
            }
        }

        // Flush all stats buffers that have data (synchronously)
        for (key, buffer) in &self.stats_buffers {
            if let Some(temp_chunk) = buffer.copy_as_chunk() {
                let array = if self.last_sample_was_warmup {
                    &self.arrays.warmup_param_arrays[key]
                } else {
                    &self.arrays.sample_param_arrays[key]
                };
                store_zarr_chunk_sync(&self.rt_handle, array, temp_chunk, self.chain)?;
            }
        }

        Ok(())
    }
}

/// Configuration for async Zarr-based MCMC storage.
///
/// This is the async version of ZarrConfig that uses tokio for async I/O operations.
/// It provides the same interface but spawns tasks for write operations to avoid
/// blocking the sampling process.
///
/// The storage organizes data into groups:
/// - `posterior/` - posterior samples
/// - `sample_stats/` - sampling statistics
/// - `warmup_posterior/` - warmup samples (optional)
/// - `warmup_sample_stats/` - warmup statistics (optional)
pub struct ZarrAsyncConfig {
    store: AsyncReadableWritableListableStorage,
    group_path: Option<String>,
    draw_chunk_size: u64,
    store_warmup: bool,
    rt_handle: tokio::runtime::Handle,
}

impl ZarrAsyncConfig {
    /// Create a new async Zarr configuration with default settings.
    ///
    /// Default settings:
    /// - `draw_chunk_size`: 100 samples per chunk
    /// - `store_warmup`: true (warmup samples are stored)
    /// - `group_path`: root of the store
    pub fn new(
        rt_handle: tokio::runtime::Handle,
        store: AsyncReadableWritableListableStorage,
    ) -> Self {
        Self {
            store,
            group_path: None,
            draw_chunk_size: 100,
            store_warmup: true,
            rt_handle,
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

impl StorageConfig for ZarrAsyncConfig {
    type Storage = ZarrAsyncTraceStorage;

    fn new_trace<M: Math>(self, settings: &impl Settings, math: &M) -> Result<Self::Storage> {
        let handle = self.rt_handle.clone();
        let rt_handle = handle.clone();
        handle.block_on(async move {
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
            root.async_store_metadata().await?;

            GroupBuilder::new()
                .build(store.clone(), &format!("{}warmup_posterior", group_path))?
                .async_store_metadata()
                .await?;
            GroupBuilder::new()
                .build(store.clone(), &format!("{}warmup_sample_stats", group_path))?
                .async_store_metadata()
                .await?;
            GroupBuilder::new()
                .build(store.clone(), &format!("{}posterior", group_path))?
                .async_store_metadata()
                .await?;
            GroupBuilder::new()
                .build(store.clone(), &format!("{}sample_stats", group_path))?
                .async_store_metadata()
                .await?;

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
            // add arc around each value
            let warmup_param_arrays: HashMap<_, _> = warmup_param_arrays
                .into_iter()
                .map(|(k, v)| (k, Arc::new(v) as Array))
                .collect();
            let sample_param_arrays: HashMap<_, _> = sample_param_arrays
                .into_iter()
                .map(|(k, v)| (k, Arc::new(v) as Array))
                .collect();
            let warmup_draw_arrays: HashMap<_, _> = warmup_draw_arrays
                .into_iter()
                .map(|(k, v)| (k, Arc::new(v) as Array))
                .collect();
            let sample_draw_arrays: HashMap<_, _> = sample_draw_arrays
                .into_iter()
                .map(|(k, v)| (k, Arc::new(v) as Array))
                .collect();
            for array in warmup_param_arrays
                .values()
                .chain(sample_param_arrays.values())
                .chain(warmup_draw_arrays.values())
                .chain(sample_draw_arrays.values())
            {
                array.async_store_metadata().await?;
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
            )
            .await?;
            store_coords(
                store.clone(),
                format!("{}warmup_posterior", &group_path),
                &draw_coords,
            )
            .await?;
            store_coords(
                store.clone(),
                format!("{}sample_stats", &group_path),
                &stat_coords,
            )
            .await?;
            store_coords(
                store.clone(),
                format!("{}warmup_sample_stats", &group_path),
                &stat_coords,
            )
            .await?;
            Ok(ZarrAsyncTraceStorage {
                arrays: Arc::new(trace_storage),
                param_types,
                draw_types,
                draw_chunk_size,
                rt_handle,
            })
        })
    }
}

impl TraceStorage for ZarrAsyncTraceStorage {
    type ChainStorage = ZarrAsyncChainStorage;

    type Finalized = ();

    fn initialize_trace_for_chain(&self, chain_id: u64) -> Result<Self::ChainStorage> {
        Ok(ZarrAsyncChainStorage::new(
            self.arrays.clone(),
            &self.param_types,
            &self.draw_types,
            self.draw_chunk_size,
            chain_id as _,
            self.rt_handle.clone(),
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

    fn inspect(
        &self,
        traces: Vec<Result<Option<<Self::ChainStorage as ChainStorage>::Finalized>>>,
    ) -> Result<(Option<anyhow::Error>, Self::Finalized)> {
        for trace in traces {
            if let Err(err) = trace {
                return Ok((Some(err), ()));
            };
        }
        Ok((None, ()))
    }
}
