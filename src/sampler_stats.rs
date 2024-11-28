use arrow::array::StructArray;

use crate::{Math, Settings};

pub trait SamplerStats<M: Math> {
    type Builder: StatTraceBuilder<M, Self>;
    type StatOptions;

    fn new_builder(
        &self,
        options: Self::StatOptions,
        settings: &impl Settings,
        dim: usize,
    ) -> Self::Builder;
}

pub trait StatTraceBuilder<M: Math, T: ?Sized>: Send {
    fn append_value(&mut self, math: Option<&mut M>, value: &T);
    fn finalize(self) -> Option<StructArray>;
    fn inspect(&self) -> Option<StructArray>;
}

impl<M: Math, T> StatTraceBuilder<M, T> for () {
    fn append_value(&mut self, _math: Option<&mut M>, _value: &T) {}

    fn finalize(self) -> Option<StructArray> {
        None
    }

    fn inspect(&self) -> Option<StructArray> {
        None
    }
}
