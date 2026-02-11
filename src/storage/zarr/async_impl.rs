use std::collections::HashMap;
use std::iter::once;

use std::sync::Arc;
use tokio::runtime::Handle;
use tokio::task::JoinSet;

use anyhow::{Context, Result};
use nuts_storable::{ItemType, Value};
use zarrs::array::{ArrayBuilder, ArraySubset};
use zarrs::group::GroupBuilder;

use zarrs::storage::{
    AsyncReadableWritableListableStorage, AsyncReadableWritableListableStorageTraits,
};

use super::common::{
    Chunk, SampleBuffer, SampleBufferValue, create_arrays, value_to_zarr_coord_params,
};
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
    pending_writes: Arc<tokio::sync::Mutex<JoinSet<Result<()>>>>,
    rt_handle: tokio::runtime::Handle,
    max_queued_writes: usize,
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

    if data.values.len() == 0 {
        return Ok(());
    }

    let result = if data.is_full() {
        match data.values {
            SampleBufferValue::F64(v) => array.async_store_chunk(&chunk, &v).await,
            SampleBufferValue::F32(v) => array.async_store_chunk(&chunk, &v).await,
            SampleBufferValue::U64(v) => array.async_store_chunk(&chunk, &v).await,
            SampleBufferValue::I64(v) => array.async_store_chunk(&chunk, &v).await,
            SampleBufferValue::Bool(v) => array.async_store_chunk(&chunk, &v).await,
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
                    .async_store_chunk_subset(&chunk, &chunk_subset, &v)
                    .await
            }
            SampleBufferValue::F32(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array
                    .async_store_chunk_subset(&chunk, &chunk_subset, &v)
                    .await
            }
            SampleBufferValue::U64(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array
                    .async_store_chunk_subset(&chunk, &chunk_subset, &v)
                    .await
            }
            SampleBufferValue::I64(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array
                    .async_store_chunk_subset(&chunk, &chunk_subset, &v)
                    .await
            }
            SampleBufferValue::Bool(v) => {
                assert!(v.len() == chunk_subset.num_elements_usize());
                array
                    .async_store_chunk_subset(&chunk, &chunk_subset, &v)
                    .await
            }
        }
    };

    result.with_context(|| {
        format!(
            "Failed to store chunk for variable {} at chunk {} with length {}",
            array.path(),
            data.chunk_idx,
            data.len
        )
    })?;
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
        let (data_type, len, fill_value) = value_to_zarr_coord_params(coord);
        let name: &String = name;
        let coord_array = ArrayBuilder::new(
            vec![len as u64],
            vec![(len as u64).max(1)],
            data_type,
            fill_value,
        )
        .dimension_names(Some(vec![name.to_string()]))
        .build(store.clone(), &format!("{}/{}", group, name))
        .with_context(|| {
            format!(
                "Failed to create coordinate array for {} in group {}",
                name, group
            )
        })?;

        if len > 0 {
            let subset = vec![0];
            match coord {
                &Value::F64(ref v) => coord_array
                    .async_store_chunk(&subset, v)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to store chunk for float64 coordinate {} in group {}",
                            name, group
                        )
                    })?,
                &Value::F32(ref v) => coord_array
                    .async_store_chunk(&subset, v)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to store chunk for float32 coordinate {} in group {}",
                            name, group
                        )
                    })?,
                &Value::U64(ref v) => coord_array
                    .async_store_chunk(&subset, v)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to store chunk for uint64 coordinate {} in group {}",
                            name, group
                        )
                    })?,
                &Value::I64(ref v) => coord_array
                    .async_store_chunk(&subset, v)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to store chunk for int64 coordinate {} in group {}",
                            name, group
                        )
                    })?,
                &Value::Bool(ref v) => coord_array
                    .async_store_chunk(&subset, v)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to store chunk for bool coordinate {} in group {}",
                            name, group
                        )
                    })?,
                &Value::Strings(ref v) => coord_array
                    .async_store_chunk(&subset, v)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to store chunk for string coordinate {} in group {}",
                            name, group
                        )
                    })?,
                &Value::DateTime64(_, ref data) => coord_array
                    .async_store_chunk(&subset, data)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to store chunk for datetime coordinate {} in group {}",
                            name, group
                        )
                    })?,
                &Value::TimeDelta64(_, ref data) => coord_array
                    .async_store_chunk(&subset, data)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to store chunk for time delta coordinate {} in group {}",
                            name, group
                        )
                    })?,
                _ => unreachable!(),
            }
        }
        coord_array.async_store_metadata().await.with_context(|| {
            format!(
                "Failed to write metadata for coordinate {} in group {}",
                name, group
            )
        })?;
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
        let draw_buffers: HashMap<String, SampleBuffer> = draw_types
            .iter()
            .map(|(name, item_type)| (name.clone(), SampleBuffer::new(*item_type, buffer_size)))
            .collect();

        let stats_buffers: HashMap<String, SampleBuffer> = param_types
            .iter()
            .map(|(name, item_type)| (name.clone(), SampleBuffer::new(*item_type, buffer_size)))
            .collect();

        let num_arrays = draw_buffers.len() + stats_buffers.len();

        Self {
            draw_buffers,
            stats_buffers,
            arrays,
            chain,
            last_sample_was_warmup: true,
            pending_writes: Arc::new(tokio::sync::Mutex::new(JoinSet::new())),
            // We allow up to the number of arrays in pending writes, so
            // that we queue one write per draw.
            max_queued_writes: num_arrays.max(1),
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

            queue_write(
                &self.rt_handle,
                self.pending_writes.clone(),
                self.max_queued_writes,
                array,
                chunk,
                chain,
            )?;
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

            queue_write(
                &self.rt_handle,
                self.pending_writes.clone(),
                self.max_queued_writes,
                array,
                chunk,
                chain,
            )?;
        }
        Ok(())
    }
}

fn queue_write(
    handle: &Handle,
    queue: Arc<tokio::sync::Mutex<JoinSet<Result<()>>>>,
    max_queued_writes: usize,
    array: Array,
    chunk: Chunk,
    chain: u64,
) -> Result<()> {
    let rt_handle = handle.clone();
    // We need an async task to interface with the async storage
    // and JoinSet API.
    let spawn_write_task = handle.spawn(async move {
        // This should never actually block, because this lock
        // is only held in tasks that are spawned and immediately blocked_on
        // from the sampling thread.
        let mut writes_guard = queue.lock().await;

        while writes_guard.len() >= max_queued_writes {
            let out = writes_guard.join_next().await;
            if let Some(out) = out {
                out.context("Failed to await previous trace write operation")?
                    .context("Chunk write operation failed")?;
            } else {
                break;
            }
        }
        writes_guard.spawn_on(
            async move { store_zarr_chunk_async(array, chunk, chain).await },
            &rt_handle,
        );
        Ok(())
    });
    let res: Result<()> = handle.block_on(spawn_write_task)?;
    res?;
    Ok(())
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

                    queue_write(
                        &self.rt_handle,
                        self.pending_writes.clone(),
                        self.max_queued_writes,
                        array,
                        chunk,
                        chain,
                    )?;
                }
            }
            for (key, buffer) in self.stats_buffers.iter_mut() {
                if let Some(chunk) = buffer.reset() {
                    let array = self.arrays.warmup_param_arrays[key].clone();
                    let chain = self.chain;

                    queue_write(
                        &self.rt_handle,
                        self.pending_writes.clone(),
                        self.max_queued_writes,
                        array,
                        chunk,
                        chain,
                    )?;
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
        // All tasks that hold a reference to the queue are blocked_on
        // right away, so we hold the only refercne to `self.pending_writes`.
        let pending_writes = Arc::into_inner(self.pending_writes)
            .expect("Could not take ownership of pending writes queue")
            .into_inner();
        self.rt_handle.block_on(async move {
            for join_handle in pending_writes.join_all().await {
                let _ = join_handle.context("Failed to await async chunk write operation")?;
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

        // Join all pending writes
        let pending_writes = self.pending_writes.clone();
        self.rt_handle.block_on(async move {
            let mut pending_writes = pending_writes.lock().await;
            loop {
                let Some(join_handle) = pending_writes.join_next().await else {
                    break;
                };
                join_handle
                    .context("Failed to await async chunk write operation")?
                    .context("Chunk write operation failed")?;
            }
            Ok::<(), anyhow::Error>(())
        })?;

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
                .build(store.clone(), &format!("{}warmup_posterior", group_path))
                .context("Failed to create warmup_posterior group")?
                .async_store_metadata()
                .await?;
            GroupBuilder::new()
                .build(store.clone(), &format!("{}warmup_sample_stats", group_path))
                .context("Failed to create warmup_sample_stats group")?
                .async_store_metadata()
                .await?;
            GroupBuilder::new()
                .build(store.clone(), &format!("{}posterior", group_path))
                .context("Failed to create posterior group")?
                .async_store_metadata()
                .await?;
            GroupBuilder::new()
                .build(store.clone(), &format!("{}sample_stats", group_path))
                .context("Failed to create sample_stats group")?
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
            )
            .context("Failed to create warmup_param_arrays")?;
            let sample_param_arrays = create_arrays(
                store.clone(),
                &format!("{}sample_stats", group_path),
                &param_types,
                &param_dims,
                n_chains,
                n_draws,
                &stat_dim_sizes,
                self.draw_chunk_size,
            )
            .context("Failed to create sample_param_arrays")?;
            let warmup_draw_arrays = create_arrays(
                store.clone(),
                &format!("{}warmup_posterior", group_path),
                &draw_types,
                &draw_dims,
                n_chains,
                n_tune,
                &draw_dim_sizes,
                self.draw_chunk_size,
            )
            .context("Failed to create warmup_draw_arrays")?;
            let sample_draw_arrays = create_arrays(
                store.clone(),
                &format!("{}posterior", group_path),
                &draw_types,
                &draw_dims,
                n_chains,
                n_draws,
                &draw_dim_sizes,
                self.draw_chunk_size,
            )
            .context("Failed to create sample_draw_arrays")?;
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
            .await
            .context("Failed to store posterior coordinates")?;
            store_coords(
                store.clone(),
                format!("{}warmup_posterior", &group_path),
                &draw_coords,
            )
            .await
            .context("Failed to store warmup_posterior coordinates")?;
            store_coords(
                store.clone(),
                format!("{}sample_stats", &group_path),
                &stat_coords,
            )
            .await
            .context("Failed to store sample_stats coordinates")?;
            store_coords(
                store.clone(),
                format!("{}warmup_sample_stats", &group_path),
                &stat_coords,
            )
            .await
            .context("Failed to store warmup_sample_stats coordinates")?;
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
