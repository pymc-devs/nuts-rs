use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};
use nuts_storable::{ItemType, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::storage::{ChainStorage, StorageConfig, TraceStorage};
use crate::{Math, Progress, Settings};

/// Container for different types of ndarray values
#[derive(Debug, Clone)]
pub enum NdarrayValue {
    F64(ArrayD<f64>),
    F32(ArrayD<f32>),
    Bool(ArrayD<bool>),
    I64(ArrayD<i64>),
    U64(ArrayD<u64>),
    String(ArrayD<String>),
}

impl NdarrayValue {
    /// Create a new ndarray with the specified type and shape
    fn new(item_type: ItemType, shape: &[usize]) -> Self {
        match item_type {
            ItemType::F64 => NdarrayValue::F64(ArrayD::zeros(IxDyn(shape))),
            ItemType::F32 => NdarrayValue::F32(ArrayD::zeros(IxDyn(shape))),
            ItemType::Bool => NdarrayValue::Bool(ArrayD::from_elem(IxDyn(shape), false)),
            ItemType::I64 => NdarrayValue::I64(ArrayD::zeros(IxDyn(shape))),
            ItemType::U64 => NdarrayValue::U64(ArrayD::zeros(IxDyn(shape))),
            ItemType::String => {
                NdarrayValue::String(ArrayD::from_elem(IxDyn(shape), String::new()))
            }
        }
    }

    /// Set values at the specified indices
    fn set_value(&mut self, indices: &[usize], value: Value) -> Result<()> {
        match (self, value) {
            (NdarrayValue::F64(arr), Value::ScalarF64(v)) => {
                arr[IxDyn(indices)] = v;
            }
            (NdarrayValue::F32(arr), Value::ScalarF32(v)) => {
                arr[IxDyn(indices)] = v;
            }
            (NdarrayValue::Bool(arr), Value::ScalarBool(v)) => {
                arr[IxDyn(indices)] = v;
            }
            (NdarrayValue::I64(arr), Value::ScalarI64(v)) => {
                arr[IxDyn(indices)] = v;
            }
            (NdarrayValue::U64(arr), Value::ScalarU64(v)) => {
                arr[IxDyn(indices)] = v;
            }
            (NdarrayValue::F64(arr), Value::F64(v)) => {
                // For vector values, we need to handle the extra dimensions
                if indices.len() == 2 {
                    // Simple case: just set the slice
                    let mut view = arr.slice_mut(ndarray::s![indices[0], indices[1], ..]);
                    for (i, val) in v.iter().enumerate() {
                        view[i] = *val;
                    }
                } else {
                    return Err(anyhow::anyhow!(
                        "Vector assignment with complex indices not implemented"
                    ));
                }
            }
            (NdarrayValue::F32(arr), Value::F32(v)) => {
                if indices.len() == 2 {
                    let mut view = arr.slice_mut(ndarray::s![indices[0], indices[1], ..]);
                    for (i, val) in v.iter().enumerate() {
                        view[i] = *val;
                    }
                } else {
                    return Err(anyhow::anyhow!(
                        "Vector assignment with complex indices not implemented"
                    ));
                }
            }
            (NdarrayValue::Bool(arr), Value::Bool(v)) => {
                if indices.len() == 2 {
                    let mut view = arr.slice_mut(ndarray::s![indices[0], indices[1], ..]);
                    for (i, val) in v.iter().enumerate() {
                        view[i] = *val;
                    }
                } else {
                    return Err(anyhow::anyhow!(
                        "Vector assignment with complex indices not implemented"
                    ));
                }
            }
            (NdarrayValue::I64(arr), Value::I64(v)) => {
                if indices.len() == 2 {
                    let mut view = arr.slice_mut(ndarray::s![indices[0], indices[1], ..]);
                    for (i, val) in v.iter().enumerate() {
                        view[i] = *val;
                    }
                } else {
                    return Err(anyhow::anyhow!(
                        "Vector assignment with complex indices not implemented"
                    ));
                }
            }
            (NdarrayValue::U64(arr), Value::U64(v)) => {
                if indices.len() == 2 {
                    let mut view = arr.slice_mut(ndarray::s![indices[0], indices[1], ..]);
                    for (i, val) in v.iter().enumerate() {
                        view[i] = *val;
                    }
                } else {
                    return Err(anyhow::anyhow!(
                        "Vector assignment with complex indices not implemented"
                    ));
                }
            }
            _ => return Err(anyhow::anyhow!("Mismatched item type")),
        }
        Ok(())
    }
}

/// Final result containing the collected samples as ndarrays
#[derive(Debug, Clone)]
pub struct NdarrayTrace {
    /// HashMap containing sampler stats as ndarrays with shape (n_chains, n_draws, *extra_dims)
    pub stats: HashMap<String, NdarrayValue>,
    /// HashMap containing draws as ndarrays with shape (n_chains, n_draws, *extra_dims)
    pub draws: HashMap<String, NdarrayValue>,
}

/// Shared storage container with interior mutability
#[derive(Clone)]
struct SharedArrays {
    stats_arrays: HashMap<String, NdarrayValue>,
    draws_arrays: HashMap<String, NdarrayValue>,
}

/// Main storage for ndarray MCMC traces
#[derive(Clone)]
pub struct NdarrayTraceStorage {
    shared_arrays: Arc<Mutex<SharedArrays>>,
}

/// Per-chain storage for ndarray MCMC traces
pub struct NdarrayChainStorage {
    shared_arrays: Arc<Mutex<SharedArrays>>,
    chain: usize,
    current_draw: usize,
}

impl NdarrayChainStorage {
    /// Create a new chain storage
    fn new(trace_storage: &NdarrayTraceStorage, chain: usize) -> Self {
        Self {
            shared_arrays: trace_storage.shared_arrays.clone(),
            chain,
            current_draw: 0,
        }
    }

    /// Store a parameter value in the ndarray
    fn push_param(&mut self, name: &str, value: Value) -> Result<()> {
        if ["draw", "chain"].contains(&name) {
            return Ok(());
        }

        let mut shared = self.shared_arrays.lock().unwrap();
        if let Some(array) = shared.stats_arrays.get_mut(name) {
            let indices = vec![self.chain, self.current_draw];
            array.set_value(&indices, value)?;
        } else {
            return Err(anyhow::anyhow!("Unknown param name: {}", name));
        }
        Ok(())
    }

    /// Store a draw value in the ndarray
    fn push_draw(&mut self, name: &str, value: Value) -> Result<()> {
        if ["draw", "chain"].contains(&name) {
            return Ok(());
        }

        let mut shared = self.shared_arrays.lock().unwrap();
        if let Some(array) = shared.draws_arrays.get_mut(name) {
            let indices = vec![self.chain, self.current_draw];
            array.set_value(&indices, value)?;
        } else {
            return Err(anyhow::anyhow!("Unknown posterior variable name: {}", name));
        }
        Ok(())
    }
}

pub struct NdarrayConfig {}

impl Default for NdarrayConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl NdarrayConfig {
    pub fn new() -> Self {
        Self {}
    }
}

impl StorageConfig for NdarrayConfig {
    type Storage = NdarrayTraceStorage;

    fn new_trace<M: Math>(self, settings: &impl Settings, math: &M) -> Result<Self::Storage> {
        let n_chains = settings.num_chains();
        let n_tune = settings.hint_num_tune();
        let n_draws = settings.hint_num_draws();
        let total_draws = n_tune + n_draws;

        let mut stats_arrays = HashMap::new();
        let mut draws_arrays = HashMap::new();

        let dim_sizes = math.dim_sizes();

        // Create arrays for stats
        for ((name, extra_dims), (name2, item_type)) in settings
            .stat_dims_all(math)
            .into_iter()
            .zip(settings.stat_types(math).into_iter())
        {
            assert!(name == name2);
            if ["draw", "chain"].contains(&name.as_str()) {
                continue;
            }

            // Build shape: [n_chains, total_draws, ...extra_dims]
            let mut shape = vec![n_chains, total_draws];
            for dim in extra_dims {
                let dim_size = *dim_sizes
                    .get(&dim.to_string())
                    .context(format!("Unknown dimension: {}", dim))?
                    as usize;
                shape.push(dim_size);
            }

            let array = NdarrayValue::new(item_type, &shape);
            stats_arrays.insert(name, array);
        }

        for ((name, extra_dims), (name2, item_type)) in settings
            .stat_dims_all(math)
            .into_iter()
            .zip(settings.stat_types(math).into_iter())
        {
            assert!(name == name2);
            if ["draw", "chain"].contains(&name.as_str()) {
                continue;
            }
            // Build shape: [n_chains, total_draws, ...extra_dims]
            let mut shape = vec![n_chains, total_draws];
            for dim in extra_dims {
                let dim_size = *dim_sizes
                    .get(&dim.to_string())
                    .context(format!("Unknown dimension: {}", dim))?
                    as usize;
                shape.push(dim_size);
            }

            let array = NdarrayValue::new(item_type, &shape);
            draws_arrays.insert(name, array);
        }

        let shared_arrays = Arc::new(Mutex::new(SharedArrays {
            stats_arrays,
            draws_arrays,
        }));

        Ok(NdarrayTraceStorage { shared_arrays })
    }
}

impl ChainStorage for NdarrayChainStorage {
    type Finalized = ();

    fn record_sample(
        &mut self,
        _settings: &impl Settings,
        stats: Vec<(&str, Option<Value>)>,
        draws: Vec<(&str, Option<Value>)>,
        _info: &Progress,
    ) -> Result<()> {
        for (name, value) in stats {
            if let Some(value) = value {
                self.push_param(name, value)?;
            }
        }
        for (name, value) in draws {
            if let Some(value) = value {
                self.push_draw(name, value)?;
            } else {
                return Err(anyhow::anyhow!("Missing draw value for {}", name));
            }
        }
        self.current_draw += 1;
        Ok(())
    }

    /// Finalize storage - nothing to do for ndarray storage
    fn finalize(self) -> Result<Self::Finalized> {
        Ok(())
    }

    /// Flush - no-op for ndarray storage since everything is in shared arrays
    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

impl TraceStorage for NdarrayTraceStorage {
    type ChainStorage = NdarrayChainStorage;

    type Finalized = NdarrayTrace;

    fn initialize_trace_for_chain(&self, chain_id: u64) -> Result<Self::ChainStorage> {
        Ok(NdarrayChainStorage::new(self, chain_id as usize))
    }

    fn finalize(
        self,
        traces: Vec<Result<<Self::ChainStorage as ChainStorage>::Finalized>>,
    ) -> Result<(Option<anyhow::Error>, Self::Finalized)> {
        let mut first_error = None;

        for trace in traces {
            if let Err(err) = trace
                && first_error.is_none()
            {
                first_error = Some(err);
            }
        }

        // Clone the arrays from the shared container since we can't move out of &self
        let shared_arrays = self.shared_arrays.lock().unwrap();
        let stats_arrays = shared_arrays.stats_arrays.clone();
        let draws_arrays = shared_arrays.draws_arrays.clone();
        drop(shared_arrays);

        let result = NdarrayTrace {
            stats: stats_arrays,
            draws: draws_arrays,
        };

        Ok((first_error, result))
    }

    fn inspect(
        &self,
        traces: Vec<Result<Option<<Self::ChainStorage as ChainStorage>::Finalized>>>,
    ) -> Result<(Option<anyhow::Error>, Self::Finalized)> {
        self.clone().finalize(
            traces
                .into_iter()
                .map(|res| match res {
                    Ok(Some(_)) => Ok(()),
                    Ok(None) => Ok(()),
                    Err(err) => Err(err),
                })
                .collect(),
        )
    }
}
