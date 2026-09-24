use std::collections::HashMap;

use nuts_storable::{HasDims, ItemType, Storable, Value};
use thiserror::Error;

use super::{CpuLogpFunc, CpuMathError, LogpError};

#[derive(Clone, Debug)]
pub struct NormalLogp {
    pub dim: usize,
    pub mu: f64,
}

impl NormalLogp {
    pub(crate) fn new(dim: usize, mu: f64) -> NormalLogp {
        NormalLogp { dim, mu }
    }
}

#[derive(Error, Debug)]
pub enum NormalLogpError {}

impl LogpError for NormalLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

impl HasDims for NormalLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        vec![
            ("unconstrained_parameter".to_string(), self.dim as u64),
            ("dim".to_string(), self.dim as u64),
        ]
        .into_iter()
        .collect()
    }
}

impl CpuLogpFunc for NormalLogp {
    type LogpError = NormalLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, NormalLogpError> {
        assert!(gradient.len() == position.len());
        let mut logp = 0f64;
        for (p, g) in position.iter().zip(gradient.iter_mut()) {
            let val = *p - self.mu;
            logp -= val * val / 2.;
            *g = -val;
        }
        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        Ok(array.to_vec())
    }
}

impl HasDims for &NormalLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        (*self).dim_sizes()
    }
}

impl CpuLogpFunc for &NormalLogp {
    type LogpError = NormalLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, NormalLogpError> {
        assert!(gradient.len() == position.len());
        let mut logp = 0f64;
        for (p, g) in position.iter().zip(gradient.iter_mut()) {
            let val = *p - self.mu;
            logp -= val * val / 2.;
            *g = -val;
        }
        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        Ok(array.to_vec())
    }
}

/// Expands into two variables but declares only one, the way a model restricted to a subset
/// of its variables does (nutpie's `var_names`).
pub struct PartialExpandLogp {
    pub inner: NormalLogp,
}

pub struct PartialExpanded(Vec<f64>);

impl<P: HasDims> Storable<P> for PartialExpanded {
    fn names(_parent: &P) -> Vec<&str> {
        vec!["kept"]
    }

    fn item_type(_parent: &P, _item: &str) -> ItemType {
        ItemType::F64
    }

    fn dims<'a>(_parent: &'a P, _item: &str) -> Vec<&'a str> {
        vec!["dim"]
    }

    fn get_all<'a>(&'a mut self, _parent: &'a P) -> Vec<(&'a str, Option<Value>)> {
        // undeclared first: a backend zipping positionally then shifts unless it is dropped
        vec![
            ("undeclared", Some(Value::F64(self.0.clone()))),
            ("kept", Some(Value::F64(self.0.clone()))),
        ]
    }
}

impl HasDims for &PartialExpandLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        self.inner.dim_sizes()
    }
}

impl CpuLogpFunc for &PartialExpandLogp {
    type LogpError = NormalLogpError;
    type FlowParameters = ();
    type ExpandedVector = PartialExpanded;

    fn dim(&self) -> usize {
        self.inner.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, NormalLogpError> {
        (&mut &self.inner).logp(position, gradient)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        Ok(PartialExpanded(array.to_vec()))
    }
}
