use anyhow::Result;
use nuts_storable::Value;

use crate::{Math, Progress, Settings};

/// Trait for storing MCMC sampling results from a single chain.
///
/// Handles progressive accumulation of statistics and draws during sampling,
/// with methods to record samples and finalize results.
pub trait ChainStorage: Send {
    /// The type returned when the chain storage is finalized.
    type Finalized: Send + Sync + 'static;

    /// Appends a new sample to the storage.
    fn record_sample(
        &mut self,
        settings: &impl Settings,
        stats: Vec<(&str, Option<Value>)>,
        draws: Vec<(&str, Option<Value>)>,
        info: &Progress,
    ) -> Result<()>;

    /// Finalizes the storage and returns processed results.
    fn finalize(self) -> Result<Self::Finalized>;

    fn inspect(&self) -> Result<Option<Self::Finalized>> {
        Ok(None)
    }

    /// Flush any buffered data to ensure all samples are stored.
    fn flush(&self) -> Result<()>;
}

/// Configuration trait for creating MCMC storage backends.
///
/// This is the main user-facing trait for configuring storage. Users choose
/// a storage backend by providing an implementation of this trait to the
/// sampling functions.
pub trait StorageConfig: Send + 'static {
    /// The storage backend type this config creates.
    type Storage: TraceStorage;

    /// Creates a new storage backend instance.
    fn new_trace<M: Math>(self, settings: &impl Settings, math: &M) -> Result<Self::Storage>;
}

/// Trait for managing storage across multiple MCMC chains.
///
/// Defines the interface for initializing chain storage and combining results
/// from multiple chains into a final result.
pub trait TraceStorage: Send + Sync + Sized + 'static {
    /// The storage type for individual chains.
    type ChainStorage: ChainStorage;

    /// The final result type combining all chains.
    type Finalized: Send + Sync + 'static;

    /// Create storage for a single chain.
    fn initialize_trace_for_chain(&self, chain_id: u64) -> Result<Self::ChainStorage>;

    /// Combine results from all chains into final output.
    ///
    /// # Arguments
    ///
    /// * `traces` - Finalized results from all chains
    fn finalize(
        self,
        traces: Vec<Result<<Self::ChainStorage as ChainStorage>::Finalized>>,
    ) -> Result<(Option<anyhow::Error>, Self::Finalized)>;

    fn inspect(
        &self,
        traces: Vec<Result<Option<<Self::ChainStorage as ChainStorage>::Finalized>>>,
    ) -> Result<(Option<anyhow::Error>, Self::Finalized)>;
}
