use rand::Rng;

use crate::{
    Math, NutsError, SamplerStats,
    hamiltonian::Point,
    nuts::{Collector, NutsOptions},
    transform::Transformation,
    transformed_hamiltonian::TransformedPoint,
};

pub trait MassMatrixAdaptStrategy<M: Math>: SamplerStats<M> {
    type Transformation: Transformation<M>;
    type Collector: Collector<M, TransformedPoint<M>>;
    type Options: std::fmt::Debug + Default + Clone + Send + Sync + Copy;

    fn update_estimators(&mut self, math: &mut M, collector: &Self::Collector);

    fn switch(&mut self, math: &mut M);

    fn current_count(&self) -> u64;

    fn background_count(&self) -> u64;

    /// Give the opportunity to update the potential and return if it was changed
    fn adapt(&self, math: &mut M, mass_matrix: &mut Self::Transformation) -> bool;

    fn new(math: &mut M, options: Self::Options, _num_tune: u64, _chain: u64) -> Self;

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        _options: &mut NutsOptions,
        mass_matrix: &mut Self::Transformation,
        point: &impl Point<M>,
        _rng: &mut R,
    ) -> Result<(), NutsError>;

    fn new_collector(&self, math: &mut M) -> Self::Collector;
}
