//! A storage config that picks one of the built-in backends at runtime.
//!
//! `Sampler::new` is generic over the storage config, so choosing between backends
//! with the concrete config types compiles the sampler driver once per backend.
//! `AnyStorageConfig` compiles it once for all of them.

use anyhow::Result;
use nuts_storable::Value;

use super::{ChainStorage, CsvConfig, HashMapConfig, StorageConfig, TraceStorage};
use crate::{Math, Progress, Settings};

#[cfg(feature = "arrow")]
use super::ArrowConfig;
#[cfg(feature = "ndarray")]
use super::NdarrayConfig;
#[cfg(feature = "zarr")]
use super::{ZarrAsyncConfig, ZarrConfig};

fn backend_mismatch() -> anyhow::Error {
    anyhow::anyhow!("Chain trace was produced by a different storage backend")
}

macro_rules! any_storage {
    ($($(#[$attr:meta])* $variant:ident($config:ty),)*) => {
        /// Configuration for any of the built-in storage backends that are enabled.
        ///
        /// Variants depend on the enabled cargo features, so the enum is
        /// non-exhaustive. Convert a concrete config with `into()`.
        #[non_exhaustive]
        pub enum AnyStorageConfig {
            $($(#[$attr])* $variant($config),)*
        }

        /// The finalized trace of an [`AnyStorageConfig`], with the same variant as
        /// the config that produced it.
        #[non_exhaustive]
        pub enum AnyTrace {
            $($(#[$attr])* $variant(<<$config as StorageConfig>::Storage as TraceStorage>::Finalized),)*
        }

        pub enum AnyTraceStorage {
            $($(#[$attr])* $variant(<$config as StorageConfig>::Storage),)*
        }

        pub enum AnyChainStorage {
            $($(#[$attr])* $variant(<<$config as StorageConfig>::Storage as TraceStorage>::ChainStorage),)*
        }

        pub enum AnyChainTrace {
            $($(#[$attr])* $variant(
                <<<$config as StorageConfig>::Storage as TraceStorage>::ChainStorage as ChainStorage>::Finalized
            ),)*
        }

        $(
            $(#[$attr])*
            impl From<$config> for AnyStorageConfig {
                fn from(config: $config) -> Self {
                    Self::$variant(config)
                }
            }
        )*

        impl StorageConfig for AnyStorageConfig {
            type Storage = AnyTraceStorage;

            fn new_trace<M: Math>(self, settings: &impl Settings, math: &M) -> Result<Self::Storage> {
                match self {
                    $($(#[$attr])* Self::$variant(config) => {
                        Ok(AnyTraceStorage::$variant(config.new_trace(settings, math)?))
                    })*
                }
            }
        }

        impl TraceStorage for AnyTraceStorage {
            type ChainStorage = AnyChainStorage;
            type Finalized = AnyTrace;

            fn initialize_trace_for_chain(&self, chain_id: u64) -> Result<Self::ChainStorage> {
                match self {
                    $($(#[$attr])* Self::$variant(trace) => {
                        Ok(AnyChainStorage::$variant(trace.initialize_trace_for_chain(chain_id)?))
                    })*
                }
            }

            fn finalize(
                self,
                traces: Vec<Result<AnyChainTrace>>,
            ) -> Result<(Option<anyhow::Error>, AnyTrace)> {
                match self {
                    $($(#[$attr])* Self::$variant(trace) => {
                        let traces = traces
                            .into_iter()
                            .map(|chain| match chain? {
                                AnyChainTrace::$variant(chain) => Ok(chain),
                                #[allow(unreachable_patterns)]
                                _ => Err(backend_mismatch()),
                            })
                            .collect();
                        let (err, trace) = trace.finalize(traces)?;
                        Ok((err, AnyTrace::$variant(trace)))
                    })*
                }
            }

            fn inspect(
                &self,
                traces: Vec<Result<Option<AnyChainTrace>>>,
            ) -> Result<(Option<anyhow::Error>, AnyTrace)> {
                match self {
                    $($(#[$attr])* Self::$variant(trace) => {
                        let traces = traces
                            .into_iter()
                            .map(|chain| match chain? {
                                None => Ok(None),
                                Some(AnyChainTrace::$variant(chain)) => Ok(Some(chain)),
                                #[allow(unreachable_patterns)]
                                Some(_) => Err(backend_mismatch()),
                            })
                            .collect();
                        let (err, trace) = trace.inspect(traces)?;
                        Ok((err, AnyTrace::$variant(trace)))
                    })*
                }
            }
        }

        impl ChainStorage for AnyChainStorage {
            type Finalized = AnyChainTrace;

            fn record_sample(
                &mut self,
                settings: &impl Settings,
                stats: Vec<(&str, Option<Value>)>,
                draws: Vec<(&str, Option<Value>)>,
                info: &Progress,
            ) -> Result<()> {
                match self {
                    $($(#[$attr])* Self::$variant(chain) => {
                        chain.record_sample(settings, stats, draws, info)
                    })*
                }
            }

            fn finalize(self) -> Result<AnyChainTrace> {
                match self {
                    $($(#[$attr])* Self::$variant(chain) => {
                        Ok(AnyChainTrace::$variant(chain.finalize()?))
                    })*
                }
            }

            fn inspect(&self) -> Result<Option<AnyChainTrace>> {
                match self {
                    $($(#[$attr])* Self::$variant(chain) => {
                        Ok(chain.inspect()?.map(AnyChainTrace::$variant))
                    })*
                }
            }

            fn flush(&self) -> Result<()> {
                match self {
                    $($(#[$attr])* Self::$variant(chain) => chain.flush(),)*
                }
            }
        }
    };
}

any_storage! {
    HashMap(HashMapConfig),
    Csv(CsvConfig),
    #[cfg(feature = "ndarray")]
    Ndarray(NdarrayConfig),
    #[cfg(feature = "arrow")]
    Arrow(ArrowConfig),
    #[cfg(feature = "zarr")]
    Zarr(ZarrConfig),
    #[cfg(feature = "zarr")]
    ZarrAsync(ZarrAsyncConfig),
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use super::{AnyStorageConfig, AnyTrace};
    use crate::{
        DiagNutsSettings, HashMapConfig, Sampler, SamplerWaitResult, math::test_logps::NormalLogp,
        sampler::test_logps::CpuModel,
    };

    #[test]
    fn hashmap_through_any() {
        let settings = DiagNutsSettings {
            num_tune: 20,
            num_draws: 30,
            num_chains: 2,
            seed: 1,
            ..Default::default()
        };
        let model = Arc::new(CpuModel::new(NormalLogp { dim: 3, mu: 0.1 }));
        let config: AnyStorageConfig = HashMapConfig::new().into();
        let mut sampler =
            Sampler::new(model, settings, config, 2, None).expect("Sampler should start");
        let trace = loop {
            match sampler.wait_timeout(Duration::from_secs(10)) {
                SamplerWaitResult::Trace(trace) => break trace,
                SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
                SamplerWaitResult::Err(err, _) => panic!("sampling failed: {err:#}"),
            }
        };
        let AnyTrace::HashMap(chains) = trace else {
            panic!("expected a HashMap trace");
        };
        assert_eq!(chains.len(), 2);
        assert!(chains.iter().all(|chain| !chain.stats.is_empty()));
    }
}
