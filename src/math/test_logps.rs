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
#[derive(Clone, Debug)]
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

impl HasDims for PartialExpandLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        self.inner.dim_sizes()
    }
}

impl CpuLogpFunc for PartialExpandLogp {
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

/// How `MismatchedExpandLogp` gets its expanded draw wrong.
#[derive(Clone, Copy, Debug)]
pub enum ExpandMismatch {
    /// `F32` values for a variable declared as `F64`.
    WrongType,
    /// One value more than the declared shape holds.
    WrongLength,
    /// No value at all.
    Missing,
}

/// Declares one `F64` variable `x` of shape `dim`, but returns values that do not match
/// that declaration, the way a buggy user model might.
#[derive(Clone, Debug)]
pub struct MismatchedExpandLogp {
    pub inner: NormalLogp,
    pub mismatch: ExpandMismatch,
}

pub struct MismatchedExpanded(Vec<f64>, ExpandMismatch);

impl<P: HasDims> Storable<P> for MismatchedExpanded {
    fn names(_parent: &P) -> Vec<&str> {
        vec!["x"]
    }

    fn item_type(_parent: &P, _item: &str) -> ItemType {
        ItemType::F64
    }

    fn dims<'a>(_parent: &'a P, _item: &str) -> Vec<&'a str> {
        vec!["dim"]
    }

    fn get_all<'a>(&'a mut self, _parent: &'a P) -> Vec<(&'a str, Option<Value>)> {
        let value = match self.1 {
            ExpandMismatch::WrongType => {
                Some(Value::F32(self.0.iter().map(|&x| x as f32).collect()))
            }
            ExpandMismatch::WrongLength => {
                Some(Value::F64(self.0.iter().copied().chain([0.0]).collect()))
            }
            ExpandMismatch::Missing => None,
        };
        vec![("x", value)]
    }
}

impl HasDims for MismatchedExpandLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        self.inner.dim_sizes()
    }
}

impl CpuLogpFunc for MismatchedExpandLogp {
    type LogpError = NormalLogpError;
    type FlowParameters = ();
    type ExpandedVector = MismatchedExpanded;

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
        Ok(MismatchedExpanded(array.to_vec(), self.mismatch))
    }
}

#[derive(Error, Debug)]
#[error("logp failed on purpose")]
pub struct FailOnPurpose;

impl LogpError for FailOnPurpose {
    fn is_recoverable(&self) -> bool {
        false
    }
}

/// Fails with a non-recoverable error once it has been evaluated `fail_after` times, like
/// a model that hits a bug partway through sampling. Each clone counts on its own.
#[derive(Clone, Debug)]
pub struct FailingLogp {
    pub inner: NormalLogp,
    pub fail_after: usize,
    pub calls: usize,
}

impl HasDims for FailingLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        self.inner.dim_sizes()
    }
}

impl CpuLogpFunc for FailingLogp {
    type LogpError = FailOnPurpose;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.inner.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, FailOnPurpose> {
        self.calls += 1;
        if self.calls > self.fail_after {
            return Err(FailOnPurpose);
        }
        self.inner
            .logp(position, gradient)
            .map_err(|never| match never {})
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        Ok(array.to_vec())
    }
}
