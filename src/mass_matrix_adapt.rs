use std::marker::PhantomData;

use nuts_derive::Storable;
use rand::Rng;
use serde::Serialize;

use crate::{
    Math, NutsError,
    euclidean_hamiltonian::EuclideanPoint,
    hamiltonian::Point,
    mass_matrix::{DiagMassMatrix, DrawGradCollector, MassMatrix, RunningVariance},
    nuts::{Collector, NutsOptions},
    sampler_stats::SamplerStats,
};
const LOWER_LIMIT: f64 = 1e-20f64;
const UPPER_LIMIT: f64 = 1e20f64;

const INIT_LOWER_LIMIT: f64 = 1e-20f64;
const INIT_UPPER_LIMIT: f64 = 1e20f64;

/// Settings for mass matrix adaptation
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DiagAdaptExpSettings {
    pub store_mass_matrix: bool,
    pub use_grad_based_estimate: bool,
}

impl Default for DiagAdaptExpSettings {
    fn default() -> Self {
        Self {
            store_mass_matrix: false,
            use_grad_based_estimate: true,
        }
    }
}

pub struct Strategy<M: Math> {
    exp_variance_draw: RunningVariance<M>,
    exp_variance_grad: RunningVariance<M>,
    exp_variance_grad_bg: RunningVariance<M>,
    exp_variance_draw_bg: RunningVariance<M>,
    _settings: DiagAdaptExpSettings,
    _phantom: PhantomData<M>,
}

#[derive(Debug, Storable)]
pub struct Stats {}

impl<M: Math> SamplerStats<M> for Strategy<M> {
    type Stats = Stats;
    type StatsOptions = ();

    fn extract_stats(&self, _math: &mut M, _opt: Self::StatsOptions) -> Self::Stats {
        Stats {}
    }
}

pub trait MassMatrixAdaptStrategy<M: Math>: SamplerStats<M> {
    type MassMatrix: MassMatrix<M>;
    type Collector: Collector<M, EuclideanPoint<M>>;
    type Options: std::fmt::Debug + Default + Clone + Send + Sync + Copy;

    fn update_estimators(&mut self, math: &mut M, collector: &Self::Collector);

    fn switch(&mut self, math: &mut M);

    fn current_count(&self) -> u64;

    fn background_count(&self) -> u64;

    /// Give the opportunity to update the potential and return if it was changed
    fn adapt(&self, math: &mut M, mass_matrix: &mut Self::MassMatrix) -> bool;

    fn new(math: &mut M, options: Self::Options, _num_tune: u64, _chain: u64) -> Self;

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        _options: &mut NutsOptions,
        mass_matrix: &mut Self::MassMatrix,
        point: &impl Point<M>,
        _rng: &mut R,
    ) -> Result<(), NutsError>;

    fn new_collector(&self, math: &mut M) -> Self::Collector;
}

impl<M: Math> MassMatrixAdaptStrategy<M> for Strategy<M> {
    type MassMatrix = DiagMassMatrix<M>;
    type Collector = DrawGradCollector<M>;
    type Options = DiagAdaptExpSettings;

    fn update_estimators(&mut self, math: &mut M, collector: &DrawGradCollector<M>) {
        if collector.is_good {
            self.exp_variance_draw.add_sample(math, &collector.draw);
            self.exp_variance_grad.add_sample(math, &collector.grad);
            self.exp_variance_draw_bg.add_sample(math, &collector.draw);
            self.exp_variance_grad_bg.add_sample(math, &collector.grad);
        }
    }

    fn switch(&mut self, math: &mut M) {
        self.exp_variance_draw =
            std::mem::replace(&mut self.exp_variance_draw_bg, RunningVariance::new(math));
        self.exp_variance_grad =
            std::mem::replace(&mut self.exp_variance_grad_bg, RunningVariance::new(math));
    }

    fn current_count(&self) -> u64 {
        assert!(self.exp_variance_draw.count() == self.exp_variance_grad.count());
        self.exp_variance_draw.count()
    }

    fn background_count(&self) -> u64 {
        assert!(self.exp_variance_draw_bg.count() == self.exp_variance_grad_bg.count());
        self.exp_variance_draw_bg.count()
    }

    /// Give the opportunity to update the potential and return if it was changed
    fn adapt(&self, math: &mut M, mass_matrix: &mut DiagMassMatrix<M>) -> bool {
        if self.current_count() < 3 {
            return false;
        }

        let (draw_var, draw_scale) = self.exp_variance_draw.current();
        let (grad_var, grad_scale) = self.exp_variance_grad.current();
        assert!(draw_scale == grad_scale);

        if self._settings.use_grad_based_estimate {
            mass_matrix.update_diag_draw_grad(
                math,
                draw_var,
                grad_var,
                None,
                (LOWER_LIMIT, UPPER_LIMIT),
            );
        } else {
            let scale = (self.exp_variance_draw.count() as f64).recip();
            mass_matrix.update_diag_draw(math, draw_var, scale, None, (LOWER_LIMIT, UPPER_LIMIT));
        }

        true
    }

    fn new(math: &mut M, options: Self::Options, _num_tune: u64, _chain: u64) -> Self {
        Self {
            exp_variance_draw: RunningVariance::new(math),
            exp_variance_grad: RunningVariance::new(math),
            exp_variance_draw_bg: RunningVariance::new(math),
            exp_variance_grad_bg: RunningVariance::new(math),
            _settings: options,
            _phantom: PhantomData,
        }
    }

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        _options: &mut NutsOptions,
        mass_matrix: &mut Self::MassMatrix,
        point: &impl Point<M>,
        _rng: &mut R,
    ) -> Result<(), NutsError> {
        self.exp_variance_draw.add_sample(math, point.position());
        self.exp_variance_draw_bg.add_sample(math, point.position());
        self.exp_variance_grad.add_sample(math, point.gradient());
        self.exp_variance_grad_bg.add_sample(math, point.gradient());

        mass_matrix.update_diag_grad(
            math,
            point.gradient(),
            1f64,
            (INIT_LOWER_LIMIT, INIT_UPPER_LIMIT),
        );
        Ok(())
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        DrawGradCollector::new(math)
    }
}
