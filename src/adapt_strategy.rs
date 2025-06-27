//! Orchestrate the tuning schedule that jointly adapts step size and mass matrix during warmup.

use std::{fmt::Debug, marker::PhantomData};

use nuts_derive::Storable;
use nuts_storable::{HasDims, Storable};
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::stepsize::AcceptanceRateCollector;
use super::stepsize::{StepSizeSettings, Strategy as StepSizeStrategy};
use crate::dynamics::{
    DivergenceInfo, Hamiltonian, Point, State, TransformedHamiltonian, TransformedPoint,
};
use crate::transform::MassMatrixAdaptStrategy;
use crate::{
    NutsError,
    chain::AdaptStrategy,
    math::Math,
    nuts::{Collector, NutsOptions},
    sampler_stats::{SamplerStats, StatsDims},
};

pub struct GlobalStrategy<M: Math, A: MassMatrixAdaptStrategy<M>> {
    step_size: StepSizeStrategy,
    mass_matrix_adapt: A,
    options: EuclideanAdaptOptions<A::Options>,
    num_tune: u64,
    // The number of draws in the early window
    early_end: u64,

    // The first draw number for the final step size adaptation window
    final_step_size_window: u64,
    tuning: bool,
    has_initial_mass_matrix: bool,
    last_update: u64,
    // Current target window size for the main (non-early) phase; grows after each switch.
    current_window_size: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EuclideanAdaptOptions<S: Debug + Default> {
    pub step_size_settings: StepSizeSettings,
    pub mass_matrix_options: S,
    pub early_window: f64,
    pub step_size_window: f64,
    /// Initial window size for the main (non-early) mass-matrix adaptation phase.
    pub mass_matrix_switch_freq: u64,
    pub early_mass_matrix_switch_freq: u64,
    pub mass_matrix_update_freq: u64,
    /// Multiplicative growth factor applied to the window size after each switch in the
    /// main phase. 1.0 means constant windows (old behaviour). Must be >= 1.0.
    pub mass_matrix_window_growth: f64,
}

impl<S: Debug + Default> Default for EuclideanAdaptOptions<S> {
    fn default() -> Self {
        Self {
            step_size_settings: StepSizeSettings::default(),
            mass_matrix_options: S::default(),
            early_window: 0.3,
            step_size_window: 0.15,
            mass_matrix_switch_freq: 80,
            early_mass_matrix_switch_freq: 10,
            mass_matrix_update_freq: 1,
            mass_matrix_window_growth: 1.5,
        }
    }
}

impl<M: Math, A: MassMatrixAdaptStrategy<M>> AdaptStrategy<M> for GlobalStrategy<M, A> {
    type Hamiltonian = TransformedHamiltonian<M, A::Transformation>;
    type Collector =
        CombinedCollector<M, TransformedPoint<M>, AcceptanceRateCollector, A::Collector>;
    type Options = EuclideanAdaptOptions<A::Options>;

    fn new(math: &mut M, options: Self::Options, num_tune: u64, chain: u64) -> Self {
        let num_tune_f = num_tune as f64;
        let step_size_window = (options.step_size_window * num_tune_f) as u64;
        let early_end = (options.early_window * num_tune_f) as u64;
        let final_second_step_size = num_tune.saturating_sub(step_size_window);

        assert!(early_end < num_tune);
        assert!(options.mass_matrix_window_growth >= 1.0);

        Self {
            step_size: StepSizeStrategy::new(options.step_size_settings),
            mass_matrix_adapt: A::new(math, options.mass_matrix_options, num_tune, chain),
            options,
            num_tune,
            early_end,
            final_step_size_window: final_second_step_size,
            tuning: true,
            has_initial_mass_matrix: true,
            last_update: 0,
            current_window_size: options.mass_matrix_switch_freq,
        }
    }

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        position: &[f64],
        rng: &mut R,
    ) -> Result<(), NutsError> {
        let state = hamiltonian.init_state_untransformed(math, position)?;
        self.mass_matrix_adapt.init(
            math,
            options,
            hamiltonian.transformation_mut(),
            state.point(),
            rng,
        )?;
        self.step_size
            .init(math, options, hamiltonian, position, rng)?;
        Ok(())
    }

    fn adapt<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        draw: u64,
        collector: &Self::Collector,
        state: &State<M, TransformedPoint<M>>,
        rng: &mut R,
    ) -> Result<(), NutsError> {
        self.step_size.update(&collector.collector1);

        if draw >= self.num_tune {
            // Needed for step size jitter
            self.step_size.update_stepsize(rng, hamiltonian, true);
            self.tuning = false;
            return Ok(());
        }

        if draw < self.final_step_size_window {
            let is_early = draw < self.early_end;

            // At the transition from early to main phase, seed current_window_size as the
            // maximum of the configured initial size and the background count already
            // accumulated, so we never shrink the window.
            if !is_early && draw == self.early_end {
                self.current_window_size = self
                    .current_window_size
                    .max(self.mass_matrix_adapt.background_count());
            }

            let switch_freq = if is_early {
                self.options.early_mass_matrix_switch_freq
            } else {
                self.current_window_size
            };

            self.mass_matrix_adapt
                .update_estimators(math, &collector.collector2);
            // We only switch if we have switch_freq draws in the background estimate,
            // and if the number of remaining mass matrix steps is larger than
            // the switch frequency.
            let could_switch = self.mass_matrix_adapt.background_count() >= switch_freq;
            // For the main phase: after switching, the *next* window will be larger, so
            // is_late must look ahead using that next size to decide whether there is
            // still room for another full window before the step-size window.
            let next_window_size = if is_early {
                self.options.early_mass_matrix_switch_freq
            } else {
                (self.current_window_size + 1).max(
                    (self.current_window_size as f64 * self.options.mass_matrix_window_growth)
                        .round() as u64,
                )
            };
            let is_late = next_window_size + draw > self.final_step_size_window;

            let mut force_update = false;
            if could_switch && (!is_late) {
                self.mass_matrix_adapt.switch(math);
                force_update = true;
                // Grow the window for the next main-phase switch.
                if !is_early {
                    self.current_window_size = next_window_size;
                }
            }

            let did_change = if force_update
                | (draw - self.last_update >= self.options.mass_matrix_update_freq)
            {
                self.mass_matrix_adapt
                    .adapt(math, hamiltonian.transformation_mut())
            } else {
                false
            };

            if did_change {
                self.last_update = draw;
            }

            if is_late {
                self.step_size.update_estimator_late();
            } else {
                self.step_size.update_estimator_early();
            }

            // First time we change the mass matrix
            if did_change & self.has_initial_mass_matrix {
                self.has_initial_mass_matrix = false;
                let position = math.box_array(state.point().position());
                self.step_size
                    .init(math, options, hamiltonian, &position, rng)?;
            } else {
                self.step_size.update_stepsize(rng, hamiltonian, false)
            }
            return Ok(());
        }

        self.step_size.update_estimator_late();
        let is_last = draw == self.num_tune - 1;
        self.step_size.update_stepsize(rng, hamiltonian, is_last);
        Ok(())
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        Self::Collector::new(
            self.step_size.new_collector(),
            self.mass_matrix_adapt.new_collector(math),
        )
    }

    fn is_tuning(&self) -> bool {
        self.tuning
    }

    fn last_num_steps(&self) -> u64 {
        self.step_size.last_n_steps
    }
}

#[derive(Debug, Storable)]
pub struct GlobalStrategyStats<P: HasDims, S: Storable<P>, M: Storable<P>> {
    #[storable(flatten)]
    pub step_size: S,
    #[storable(flatten)]
    pub mass_matrix: M,
    pub tuning: bool,
    #[storable(ignore)]
    _phantom: std::marker::PhantomData<fn() -> P>,
}

#[derive(Debug)]
pub struct GlobalStrategyStatsOptions<M: Math, A: MassMatrixAdaptStrategy<M>> {
    pub step_size: (),
    pub mass_matrix: A::StatsOptions,
}

impl<M: Math, A: MassMatrixAdaptStrategy<M>> Clone for GlobalStrategyStatsOptions<M, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M: Math, A: MassMatrixAdaptStrategy<M>> Copy for GlobalStrategyStatsOptions<M, A> {}

impl<M: Math, A> SamplerStats<M> for GlobalStrategy<M, A>
where
    A: MassMatrixAdaptStrategy<M>,
{
    type Stats =
        GlobalStrategyStats<StatsDims, <StepSizeStrategy as SamplerStats<M>>::Stats, A::Stats>;
    type StatsOptions = GlobalStrategyStatsOptions<M, A>;

    fn extract_stats(&self, math: &mut M, opt: Self::StatsOptions) -> Self::Stats {
        GlobalStrategyStats {
            step_size: {
                let _: () = opt.step_size;
                self.step_size.extract_stats(math, ())
            },
            mass_matrix: self.mass_matrix_adapt.extract_stats(math, opt.mass_matrix),
            tuning: self.tuning,
            _phantom: PhantomData,
        }
    }
}

pub struct CombinedCollector<M, P, C1, C2>
where
    M: Math,
    P: Point<M>,
    C1: Collector<M, P>,
    C2: Collector<M, P>,
{
    pub collector1: C1,
    pub collector2: C2,
    _phantom: PhantomData<M>,
    _phantom2: PhantomData<P>,
}

impl<M, P, C1, C2> CombinedCollector<M, P, C1, C2>
where
    M: Math,
    P: Point<M>,
    C1: Collector<M, P>,
    C2: Collector<M, P>,
{
    pub fn new(collector1: C1, collector2: C2) -> Self {
        CombinedCollector {
            collector1,
            collector2,
            _phantom: PhantomData,
            _phantom2: PhantomData,
        }
    }
}

impl<M, P, C1, C2> Collector<M, P> for CombinedCollector<M, P, C1, C2>
where
    M: Math,
    P: Point<M>,
    C1: Collector<M, P>,
    C2: Collector<M, P>,
{
    fn register_leapfrog(
        &mut self,
        math: &mut M,
        start: &State<M, P>,
        end: &State<M, P>,
        divergence_info: Option<&DivergenceInfo>,
        num_substeps: u64,
    ) {
        self.collector1
            .register_leapfrog(math, start, end, divergence_info, num_substeps);
        self.collector2
            .register_leapfrog(math, start, end, divergence_info, num_substeps);
    }

    fn register_draw(&mut self, math: &mut M, state: &State<M, P>, info: &crate::nuts::SampleInfo) {
        self.collector1.register_draw(math, state, info);
        self.collector2.register_draw(math, state, info);
    }

    fn register_init(
        &mut self,
        math: &mut M,
        state: &State<M, P>,
        options: &crate::nuts::NutsOptions,
    ) {
        self.collector1.register_init(math, state, options);
        self.collector2.register_init(math, state, options);
    }
}

#[cfg(test)]
pub mod test_logps {
    use std::collections::HashMap;

    use crate::math::{CpuLogpFunc, LogpError};
    use nuts_storable::HasDims;
    use thiserror::Error;

    #[derive(Clone, Debug)]
    pub struct NormalLogp {
        dim: usize,
        mu: f64,
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
            vec![("unconstrained_parameter".to_string(), self.dim as u64)]
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
            let n = position.len();
            assert!(gradient.len() == n);

            let mut logp = 0f64;
            for (p, g) in position.iter().zip(gradient.iter_mut()) {
                let val = *p - self.mu;
                logp -= val * val / 2.;
                *g = -val;
            }
            Ok(logp)
        }

        fn expand_vector<R>(
            &mut self,
            _rng: &mut R,
            array: &[f64],
        ) -> Result<Self::ExpandedVector, crate::math::CpuMathError>
        where
            R: rand::Rng + ?Sized,
        {
            Ok(array.to_vec())
        }
    }
}

#[cfg(test)]
mod test {
    use super::test_logps::NormalLogp;
    use super::*;
    use crate::{
        Chain, DiagAdaptExpSettings,
        chain::{NutsChain, StatOptions},
        dynamics::{
            DivergenceStatsOptions, KineticEnergyKind, TransformedHamiltonian,
            TransformedPointStatsOptions,
        },
        math::CpuMath,
        transform::{DiagAdaptStrategy, DiagMassMatrix},
    };

    #[test]
    fn instanciate_adaptive_sampler() {
        let ndim = 10;
        let func = NormalLogp::new(ndim, 30.);
        let mut math = CpuMath::new(func);
        let num_tune = 100;
        let options = EuclideanAdaptOptions::<DiagAdaptExpSettings>::default();
        let strategy =
            GlobalStrategy::<_, DiagAdaptStrategy<_>>::new(&mut math, options, num_tune, 0u64);

        let mass_matrix = DiagMassMatrix::new(&mut math, true);

        let hamiltonian: TransformedHamiltonian<_, DiagMassMatrix<CpuMath<NormalLogp>>> =
            TransformedHamiltonian::new(&mut math, mass_matrix, KineticEnergyKind::Euclidean);

        let options = NutsOptions {
            maxdepth: 10u64,
            mindepth: 0,
            check_turning: true,
            store_divergences: false,
            target_integration_time: None,
            extra_doublings: 0,
            max_energy_error: 1000.0,
            walnuts_options: None,
        };

        let rng = {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(42)
        };
        let chain = 0u64;

        let stats_options = StatOptions {
            adapt: GlobalStrategyStatsOptions {
                step_size: (),
                mass_matrix: (),
            },
            hamiltonian: -1i64,
            point: TransformedPointStatsOptions {
                store_gradient: true,
                store_unconstrained: true,
                store_transformed: false,
            },
            divergence: DivergenceStatsOptions {
                store_divergences: true,
            },
        };

        let mut sampler = NutsChain::new(
            math,
            hamiltonian,
            strategy,
            options,
            rng,
            chain,
            stats_options,
        );
        sampler.set_position(&vec![1.5f64; ndim]).unwrap();
        for _ in 0..200 {
            sampler.draw().unwrap();
        }

        // Check that we arrive at 3
        let (last_position, _, _, prog) = sampler.expanded_draw().unwrap();
        dbg!(&last_position);
        for p in last_position {
            assert!((p - 30.).abs() < 5.0);
        }
        assert!(!prog.diverging);
    }
}
