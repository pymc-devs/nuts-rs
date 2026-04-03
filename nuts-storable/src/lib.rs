//! Minimal serialisation abstractions that let the sampler emit typed values without depending on any specific serialisation format.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DateTimeUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemType {
    U64,
    I64,
    F64,
    F32,
    Bool,
    String,
    DateTime64(DateTimeUnit),
    TimeDelta64(DateTimeUnit),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    ScalarString(String),
    DateTime64(DateTimeUnit, Vec<i64>),
    TimeDelta64(DateTimeUnit, Vec<i64>),
    ScalarU64(u64),
    ScalarI64(i64),
    ScalarF64(f64),
    ScalarF32(f32),
    ScalarBool(bool),
    Strings(Vec<String>),
}

impl From<Vec<u64>> for Value {
    fn from(value: Vec<u64>) -> Self {
        Value::U64(value)
    }
}
impl From<Vec<i64>> for Value {
    fn from(value: Vec<i64>) -> Self {
        Value::I64(value)
    }
}
impl From<Vec<f64>> for Value {
    fn from(value: Vec<f64>) -> Self {
        Value::F64(value)
    }
}
impl From<Vec<f32>> for Value {
    fn from(value: Vec<f32>) -> Self {
        Value::F32(value)
    }
}
impl From<Vec<bool>> for Value {
    fn from(value: Vec<bool>) -> Self {
        Value::Bool(value)
    }
}
impl From<u64> for Value {
    fn from(value: u64) -> Self {
        Value::ScalarU64(value)
    }
}
impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::ScalarI64(value)
    }
}
impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::ScalarF64(value)
    }
}
impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::ScalarF32(value)
    }
}
impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::ScalarBool(value)
    }
}

pub trait HasDims {
    fn dim_sizes(&self) -> HashMap<String, u64>;
    fn coords(&self) -> HashMap<String, Value> {
        HashMap::new()
    }
}

/// Trait for types whose fields can be progressively written to a trace backend.
///
/// Each field in a `Storable` struct has a *primary dimension* — the dimension
/// along which one entry is appended per event. For most fields this is the draw
/// dimension: one value is recorded per MCMC draw. Fields annotated with
/// `#[storable(event = "name")]` use a different primary dimension, meaning they
/// only receive a value when that particular event occurs.
///
/// For example, divergence statistics use `event = "divergence"`: fields like
/// `divergence_draw` or `divergence_message` only have a value on draws where a
/// divergence actually occurred. Storage backends collect these values into a
/// separate array whose second axis is named after the event (e.g.
/// `"divergence"`) rather than `"draw"`, and whose length equals the number of
/// events that occurred rather than the total number of draws.
///
/// The struct itself is responsible for ensuring that all fields sharing the
/// same event dimension produce a value on exactly the same set of draws — the
/// storage layer does not enforce this.
pub trait Storable<P: HasDims + ?Sized>: Send + Sync {
    fn names(parent: &P) -> Vec<&str>;
    fn item_type(parent: &P, item: &str) -> ItemType;
    fn dims<'a>(parent: &'a P, item: &str) -> Vec<&'a str>;

    /// Return the name of the primary dimension for the given field, or `None`
    /// if the field uses the default draw dimension.
    fn event_dim(_parent: &P, _item: &str) -> Option<&'static str> {
        None
    }

    fn get_all<'a>(&'a mut self, parent: &'a P) -> Vec<(&'a str, Option<Value>)>;
}

impl<P: HasDims> Storable<P> for Vec<f64> {
    fn names(_parent: &P) -> Vec<&str> {
        vec!["value"]
    }

    fn item_type(_parent: &P, _item: &str) -> ItemType {
        ItemType::F64
    }

    fn dims<'a>(_parent: &'a P, _item: &str) -> Vec<&'a str> {
        vec!["dim"]
    }

    fn event_dim(_parent: &P, _item: &str) -> Option<&'static str> {
        None
    }

    fn get_all<'a>(&'a mut self, _parent: &'a P) -> Vec<(&'a str, Option<Value>)> {
        vec![("value", Some(Value::F64(self.clone())))]
    }
}

impl<P: HasDims> Storable<P> for () {
    fn names(_parent: &P) -> Vec<&str> {
        vec![]
    }

    fn item_type(_parent: &P, _item: &str) -> ItemType {
        panic!("No items in unit type")
    }

    fn dims<'a>(_parent: &'a P, _item: &str) -> Vec<&'a str> {
        panic!("No items in unit type")
    }

    fn event_dim(_parent: &P, _item: &str) -> Option<&'static str> {
        None
    }

    fn get_all(&mut self, _parent: &P) -> Vec<(&str, Option<Value>)> {
        vec![]
    }
}
