use std::collections::HashMap;

use nuts_storable::{HasDims, Storable, Value};

use crate::Math;

#[derive(Clone)]
pub struct StatsDims {
    n_dim: u64,
    coord: Option<Value>,
}

impl HasDims for StatsDims {
    fn dim_sizes(&self) -> std::collections::HashMap<String, u64> {
        std::collections::HashMap::from([("unconstrained_parameter".to_string(), self.n_dim)])
    }

    fn coords(&self) -> HashMap<String, Value> {
        if let Some(coord) = &self.coord {
            return HashMap::from([("unconstrained_parameter".to_string(), coord.clone())]);
        }
        HashMap::new()
    }
}

impl<M: Math> From<&M> for StatsDims {
    fn from(math: &M) -> Self {
        StatsDims {
            n_dim: math.dim() as u64,
            coord: math.vector_coord(),
        }
    }
}

pub trait SamplerStats<M: Math> {
    type Stats: Storable<StatsDims>;
    type StatsOptions: Copy + Send + Sync;

    fn extract_stats(&self, math: &mut M, opt: Self::StatsOptions) -> Self::Stats;
}
