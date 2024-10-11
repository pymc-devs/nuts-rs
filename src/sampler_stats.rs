use std::fmt::Debug;

use arrow::array::StructArray;

use crate::{Math, Settings};

pub trait SamplerStats<M: Math> {
    type Stats: Send + Debug + Clone;
    type Builder: StatTraceBuilder<Self::Stats>;

    fn new_builder(&self, settings: &impl Settings, dim: usize) -> Self::Builder;
    fn current_stats(&self, math: &mut M) -> Self::Stats;
}

impl StatTraceBuilder<()> for () {
    fn append_value(&mut self, _value: ()) {}

    fn finalize(self) -> Option<StructArray> {
        None
    }

    fn inspect(&self) -> Option<StructArray> {
        None
    }
}

pub trait StatTraceBuilder<T: ?Sized>: Send {
    fn append_value(&mut self, value: T);
    fn finalize(self) -> Option<StructArray>;
    fn inspect(&self) -> Option<StructArray>;
}
