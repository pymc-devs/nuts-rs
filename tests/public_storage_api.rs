//! External-crate regression test: custom sequential runners need these exports.
use nuts_rs::{ChainStorage, HasDims, StatsDims, StorageConfig, TraceStorage};

fn assert_chain<T: ChainStorage>() {}
fn assert_storage<T: TraceStorage>() {
    assert_chain::<T::ChainStorage>();
}
fn assert_config<T: StorageConfig>() {
    assert_storage::<T::Storage>();
}
fn assert_dims<T: HasDims>() {}

#[test]
fn storage_traits_are_public() {
    assert_config::<nuts_rs::HashMapConfig>();
    assert_dims::<StatsDims>();
    #[cfg(feature = "arrow")]
    assert_config::<nuts_rs::ArrowConfig>();
}
