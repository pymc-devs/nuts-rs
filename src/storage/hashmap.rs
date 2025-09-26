use anyhow::Result;
use nuts_storable::{ItemType, Value};
use std::collections::HashMap;

use crate::storage::{ChainStorage, StorageConfig, TraceStorage};
use crate::{Progress, Settings};

/// Container for different types of sample values in HashMaps
#[derive(Clone, Debug)]
pub enum HashMapValue {
    F64(Vec<f64>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    String(Vec<String>),
}

impl HashMapValue {
    /// Create a new empty HashMapValue of the specified type
    fn new(item_type: ItemType) -> Self {
        match item_type {
            ItemType::F64 => HashMapValue::F64(Vec::new()),
            ItemType::F32 => HashMapValue::F32(Vec::new()),
            ItemType::Bool => HashMapValue::Bool(Vec::new()),
            ItemType::I64 => HashMapValue::I64(Vec::new()),
            ItemType::U64 => HashMapValue::U64(Vec::new()),
            ItemType::String => HashMapValue::String(Vec::new()),
        }
    }

    /// Push a value to the internal vector
    fn push(&mut self, value: Value) {
        match (self, value) {
            // Scalar values - store as single element vectors for array types
            (HashMapValue::F64(vec), Value::ScalarF64(v)) => vec.push(v),
            (HashMapValue::F32(vec), Value::ScalarF32(v)) => vec.push(v),
            (HashMapValue::U64(vec), Value::ScalarU64(v)) => vec.push(v),
            (HashMapValue::Bool(vec), Value::ScalarBool(v)) => vec.push(v),
            (HashMapValue::I64(vec), Value::ScalarI64(v)) => vec.push(v),

            (HashMapValue::F64(vec), Value::F64(v)) => vec.extend(v),
            (HashMapValue::F32(vec), Value::F32(v)) => vec.extend(v),
            (HashMapValue::U64(vec), Value::U64(v)) => vec.extend(v),
            (HashMapValue::Bool(vec), Value::Bool(v)) => vec.extend(v),
            (HashMapValue::I64(vec), Value::I64(v)) => vec.extend(v),

            _ => panic!("Mismatched item type"),
        }
    }
}

/// Main storage for HashMap MCMC traces
#[derive(Clone)]
pub struct HashMapTraceStorage {
    draw_types: Vec<(String, ItemType)>,
    param_types: Vec<(String, ItemType)>,
}

/// Per-chain storage for HashMap MCMC traces
#[derive(Clone)]
pub struct HashMapChainStorage {
    warmup_stats: HashMap<String, HashMapValue>,
    sample_stats: HashMap<String, HashMapValue>,
    warmup_draws: HashMap<String, HashMapValue>,
    sample_draws: HashMap<String, HashMapValue>,
    last_sample_was_warmup: bool,
}

/// Final result containing the collected samples
#[derive(Debug, Clone)]
pub struct HashMapResult {
    /// HashMap containing sampler stats including warmup samples
    pub stats: HashMap<String, HashMapValue>,
    /// HashMap containing draws including warmup samples
    pub draws: HashMap<String, HashMapValue>,
}

impl HashMapChainStorage {
    /// Create a new chain storage with HashMaps for parameters and samples
    fn new(param_types: &Vec<(String, ItemType)>, draw_types: &Vec<(String, ItemType)>) -> Self {
        let warmup_stats = param_types
            .iter()
            .cloned()
            .map(|(name, item_type)| (name, HashMapValue::new(item_type)))
            .collect();

        let sample_stats = param_types
            .iter()
            .cloned()
            .map(|(name, item_type)| (name, HashMapValue::new(item_type)))
            .collect();

        let warmup_draws = draw_types
            .iter()
            .cloned()
            .map(|(name, item_type)| (name, HashMapValue::new(item_type)))
            .collect();

        let sample_draws = draw_types
            .iter()
            .cloned()
            .map(|(name, item_type)| (name, HashMapValue::new(item_type)))
            .collect();

        Self {
            warmup_stats,
            sample_stats,
            warmup_draws,
            sample_draws,
            last_sample_was_warmup: true,
        }
    }

    /// Store a parameter value
    fn push_param(&mut self, name: &str, value: Value, is_warmup: bool) -> Result<()> {
        if ["draw", "chain"].contains(&name) {
            return Ok(());
        }

        let target_map = if is_warmup {
            &mut self.warmup_stats
        } else {
            &mut self.sample_stats
        };

        if let Some(hash_value) = target_map.get_mut(name) {
            hash_value.push(value);
        } else {
            panic!("Unknown param name: {}", name);
        }
        Ok(())
    }

    /// Store a draw value
    fn push_draw(&mut self, name: &str, value: Value, is_warmup: bool) -> Result<()> {
        if ["draw", "chain"].contains(&name) {
            return Ok(());
        }

        let target_map = if is_warmup {
            &mut self.warmup_draws
        } else {
            &mut self.sample_draws
        };

        if let Some(hash_value) = target_map.get_mut(name) {
            hash_value.push(value);
        } else {
            panic!("Unknown posterior variable name: {}", name);
        }
        Ok(())
    }
}

impl ChainStorage for HashMapChainStorage {
    type Finalized = HashMapResult;

    fn record_sample(
        &mut self,
        _settings: &impl Settings,
        stats: Vec<(&str, Option<Value>)>,
        draws: Vec<(&str, Option<Value>)>,
        info: &Progress,
    ) -> Result<()> {
        let is_first_draw = self.last_sample_was_warmup && !info.tuning;
        if is_first_draw {
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

    /// Finalize storage and return the collected samples
    fn finalize(self) -> Result<Self::Finalized> {
        // Combine warmup and sample data
        let mut combined_stats = HashMap::new();
        let mut combined_draws = HashMap::new();

        // Combine stats
        for (key, warmup_values) in self.warmup_stats {
            let sample_values = &self.sample_stats[&key];
            let mut combined = warmup_values.clone();

            match (&mut combined, sample_values) {
                (HashMapValue::F64(combined_vec), HashMapValue::F64(sample_vec)) => {
                    combined_vec.extend(sample_vec.iter().cloned());
                }
                (HashMapValue::F32(combined_vec), HashMapValue::F32(sample_vec)) => {
                    combined_vec.extend(sample_vec.iter().cloned());
                }
                (HashMapValue::Bool(combined_vec), HashMapValue::Bool(sample_vec)) => {
                    combined_vec.extend(sample_vec.iter().cloned());
                }
                (HashMapValue::I64(combined_vec), HashMapValue::I64(sample_vec)) => {
                    combined_vec.extend(sample_vec.iter().cloned());
                }
                (HashMapValue::U64(combined_vec), HashMapValue::U64(sample_vec)) => {
                    combined_vec.extend(sample_vec.iter().cloned());
                }
                _ => panic!("Type mismatch when combining stats for {}", key),
            }

            combined_stats.insert(key, combined);
        }

        // Combine draws
        for (key, warmup_values) in self.warmup_draws {
            let sample_values = &self.sample_draws[&key];
            let mut combined = warmup_values.clone();

            match (&mut combined, sample_values) {
                (HashMapValue::F64(combined_vec), HashMapValue::F64(sample_vec)) => {
                    combined_vec.extend(sample_vec.iter().cloned());
                }
                (HashMapValue::F32(combined_vec), HashMapValue::F32(sample_vec)) => {
                    combined_vec.extend(sample_vec.iter().cloned());
                }
                (HashMapValue::Bool(combined_vec), HashMapValue::Bool(sample_vec)) => {
                    combined_vec.extend(sample_vec.iter().cloned());
                }
                (HashMapValue::I64(combined_vec), HashMapValue::I64(sample_vec)) => {
                    combined_vec.extend(sample_vec.iter().cloned());
                }
                (HashMapValue::U64(combined_vec), HashMapValue::U64(sample_vec)) => {
                    combined_vec.extend(sample_vec.iter().cloned());
                }
                _ => panic!("Type mismatch when combining draws for {}", key),
            }

            combined_draws.insert(key, combined);
        }

        Ok(HashMapResult {
            stats: combined_stats,
            draws: combined_draws,
        })
    }

    /// Flush - no-op for HashMap storage since everything is in memory
    fn flush(&self) -> Result<()> {
        Ok(())
    }

    fn inspect(&self) -> Result<Option<Self::Finalized>> {
        self.clone().finalize().map(Some)
    }
}

pub struct HashMapConfig {}

impl Default for HashMapConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl HashMapConfig {
    pub fn new() -> Self {
        Self {}
    }
}

impl StorageConfig for HashMapConfig {
    type Storage = HashMapTraceStorage;

    fn new_trace<M: crate::Math>(
        self,
        settings: &impl Settings,
        math: &M,
    ) -> Result<Self::Storage> {
        Ok(HashMapTraceStorage {
            param_types: settings.stat_types(math),
            draw_types: settings.data_types(math),
        })
    }
}

impl TraceStorage for HashMapTraceStorage {
    type ChainStorage = HashMapChainStorage;

    type Finalized = Vec<HashMapResult>;

    fn initialize_trace_for_chain(&self, _chain_id: u64) -> Result<Self::ChainStorage> {
        Ok(HashMapChainStorage::new(
            &self.param_types,
            &self.draw_types,
        ))
    }

    fn finalize(
        self,
        traces: Vec<Result<<Self::ChainStorage as ChainStorage>::Finalized>>,
    ) -> Result<(Option<anyhow::Error>, Self::Finalized)> {
        let mut results = Vec::new();
        let mut first_error = None;

        for trace in traces {
            match trace {
                Ok(result) => results.push(result),
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
        self.clone()
            .finalize(traces.into_iter().map(|r| r.map(|o| o.unwrap())).collect())
    }
}
