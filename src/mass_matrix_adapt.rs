use std::marker::PhantomData;

use rand::Rng;

use crate::{
    chain::AdaptStrategy,
    euclidean_hamiltonian::{EuclideanHamiltonian, EuclideanPoint},
    hamiltonian::{Hamiltonian, Point},
    mass_matrix::{DiagMassMatrix, DrawGradCollector, MassMatrix, RunningVariance},
    nuts::NutsOptions,
    sampler_stats::SamplerStats,
    state::State,
    Math, NutsError, Settings,
};
const LOWER_LIMIT: f64 = 1e-20f64;
const UPPER_LIMIT: f64 = 1e20f64;

const INIT_LOWER_LIMIT: f64 = 1e-20f64;
const INIT_UPPER_LIMIT: f64 = 1e20f64;

/// Settings for mass matrix adaptation
#[derive(Clone, Copy, Debug)]
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

pub trait MassMatrixAdaptStrategy<M: Math>: AdaptStrategy<M> {
    type MassMatrix: MassMatrix<M>;

    fn update_estimators(&mut self, math: &mut M, collector: &Self::Collector);

    fn switch(&mut self, math: &mut M);

    fn current_count(&self) -> u64;

    fn background_count(&self) -> u64;

    /// Give the opportunity to update the potential and return if it was changed
    fn update_potential(&self, math: &mut M, potential: &mut Self::Hamiltonian) -> bool;
}

impl<M: Math> MassMatrixAdaptStrategy<M> for Strategy<M> {
    type MassMatrix = DiagMassMatrix<M>;

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
    fn update_potential(
        &self,
        math: &mut M,
        potential: &mut EuclideanHamiltonian<M, Self::MassMatrix>,
    ) -> bool {
        if self.current_count() < 3 {
            return false;
        }

        let (draw_var, draw_scale) = self.exp_variance_draw.current();
        let (grad_var, grad_scale) = self.exp_variance_grad.current();
        assert!(draw_scale == grad_scale);

        if self._settings.use_grad_based_estimate {
            potential.mass_matrix.update_diag_draw_grad(
                math,
                draw_var,
                grad_var,
                None,
                (LOWER_LIMIT, UPPER_LIMIT),
            );
        } else {
            let scale = (self.exp_variance_draw.count() as f64).recip();
            potential.mass_matrix.update_diag_draw(
                math,
                draw_var,
                scale,
                None,
                (LOWER_LIMIT, UPPER_LIMIT),
            );
        }

        true
    }
}

pub type Stats = ();
pub type StatsBuilder = ();

impl<M: Math> SamplerStats<M> for Strategy<M> {
    type Builder = Stats;
    type Stats = StatsBuilder;

    fn new_builder(&self, _settings: &impl Settings, _dim: usize) -> Self::Builder {}

    fn current_stats(&self, _math: &mut M) -> Self::Stats {}
}

impl<M: Math> AdaptStrategy<M> for Strategy<M> {
    type Hamiltonian = EuclideanHamiltonian<M, DiagMassMatrix<M>>;
    type Collector = DrawGradCollector<M>;
    type Options = DiagAdaptExpSettings;

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
        hamiltonian: &mut Self::Hamiltonian,
        position: &[f64],
        _rng: &mut R,
    ) -> Result<(), NutsError> {
        let state = hamiltonian.init_state(math, position)?;

        self.exp_variance_draw
            .add_sample(math, state.point().position());
        self.exp_variance_draw_bg
            .add_sample(math, state.point().position());
        self.exp_variance_grad
            .add_sample(math, state.point().gradient());
        self.exp_variance_grad_bg
            .add_sample(math, state.point().gradient());

        hamiltonian.mass_matrix.update_diag_grad(
            math,
            state.point().gradient(),
            1f64,
            (INIT_LOWER_LIMIT, INIT_UPPER_LIMIT),
        );
        Ok(())
    }

    fn adapt<R: Rng + ?Sized>(
        &mut self,
        _math: &mut M,
        _options: &mut NutsOptions,
        _potential: &mut Self::Hamiltonian,
        _draw: u64,
        _collector: &Self::Collector,
        _state: &State<M, EuclideanPoint<M>>,
        _rng: &mut R,
    ) -> Result<(), NutsError> {
        // Must be controlled from a different meta strategy
        Ok(())
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        DrawGradCollector::new(math)
    }

    fn is_tuning(&self) -> bool {
        unreachable!()
    }
}
