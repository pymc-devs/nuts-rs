use std::{fmt::Debug, marker::PhantomData, ops::Deref};

use arrow::array::StructArray;
use itertools::Itertools;
use rand::Rng;

use crate::{
    chain::AdaptStrategy,
    hamiltonian::{DivergenceInfo, Hamiltonian, Point},
    mass_matrix_adapt::MassMatrixAdaptStrategy,
    math_base::Math,
    nuts::{Collector, NutsOptions},
    sampler::Settings,
    sampler_stats::{SamplerStats, StatTraceBuilder},
    state::State,
    stepsize::AcceptanceRateCollector,
    stepsize_adapt::{
        DualAverageSettings, Stats as StepSizeStats, StatsBuilder as StepSizeStatsBuilder,
        Strategy as StepSizeStrategy,
    },
    NutsError,
};

pub struct GlobalStrategy<M: Math, A: MassMatrixAdaptStrategy<M>> {
    step_size: StepSizeStrategy,
    mass_matrix: A,
    options: EuclideanAdaptOptions<A::Options>,
    num_tune: u64,
    // The number of draws in the the early window
    early_end: u64,

    // The first draw number for the final step size adaptation window
    final_step_size_window: u64,
    tuning: bool,
    has_initial_mass_matrix: bool,
    last_update: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct EuclideanAdaptOptions<S: Debug + Default> {
    pub dual_average_options: DualAverageSettings,
    pub mass_matrix_options: S,
    pub early_window: f64,
    pub step_size_window: f64,
    pub mass_matrix_switch_freq: u64,
    pub early_mass_matrix_switch_freq: u64,
    pub mass_matrix_update_freq: u64,
}

impl<S: Debug + Default> Default for EuclideanAdaptOptions<S> {
    fn default() -> Self {
        Self {
            dual_average_options: DualAverageSettings::default(),
            mass_matrix_options: S::default(),
            early_window: 0.3,
            step_size_window: 0.15,
            mass_matrix_switch_freq: 80,
            early_mass_matrix_switch_freq: 10,
            mass_matrix_update_freq: 1,
        }
    }
}

impl<M: Math, A: MassMatrixAdaptStrategy<M>> SamplerStats<M> for GlobalStrategy<M, A> {
    type Stats = CombinedStats<StepSizeStats, A::Stats>;
    type Builder = CombinedStatsBuilder<StepSizeStatsBuilder, A::Builder>;

    fn current_stats(&self, math: &mut M) -> Self::Stats {
        CombinedStats {
            stats1: self.step_size.current_stats(math),
            stats2: self.mass_matrix.current_stats(math),
        }
    }

    fn new_builder(&self, settings: &impl Settings, dim: usize) -> Self::Builder {
        CombinedStatsBuilder {
            stats1: SamplerStats::<M>::new_builder(&self.step_size, settings, dim),
            stats2: self.mass_matrix.new_builder(settings, dim),
        }
    }
}

impl<M: Math, A: MassMatrixAdaptStrategy<M>> AdaptStrategy<M> for GlobalStrategy<M, A> {
    type Hamiltonian = A::Hamiltonian;
    type Collector = CombinedCollector<
        M,
        <Self::Hamiltonian as Hamiltonian<M>>::Point,
        AcceptanceRateCollector,
        A::Collector,
    >;
    type Options = EuclideanAdaptOptions<A::Options>;

    fn new(math: &mut M, options: Self::Options, num_tune: u64, chain: u64) -> Self {
        let num_tune_f = num_tune as f64;
        let step_size_window = (options.step_size_window * num_tune_f) as u64;
        let early_end = (options.early_window * num_tune_f) as u64;
        let final_second_step_size = num_tune.saturating_sub(step_size_window);

        assert!(early_end < num_tune);

        Self {
            step_size: StepSizeStrategy::new(options.dual_average_options),
            mass_matrix: A::new(math, options.mass_matrix_options, num_tune, chain),
            options,
            num_tune,
            early_end,
            final_step_size_window: final_second_step_size,
            tuning: true,
            has_initial_mass_matrix: true,
            last_update: 0,
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
        self.mass_matrix
            .init(math, options, hamiltonian, position, rng)?;
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
        state: &State<M, <Self::Hamiltonian as Hamiltonian<M>>::Point>,
        rng: &mut R,
    ) -> Result<(), NutsError> {
        self.step_size.update(&collector.collector1);

        if draw >= self.num_tune {
            self.tuning = false;
            return Ok(());
        }

        if draw < self.final_step_size_window {
            let is_early = draw < self.early_end;
            let switch_freq = if is_early {
                self.options.early_mass_matrix_switch_freq
            } else {
                self.options.mass_matrix_switch_freq
            };

            self.mass_matrix
                .update_estimators(math, &collector.collector2);
            // We only switch if we have switch_freq draws in the background estimate,
            // and if the number of remaining mass matrix steps is larger than
            // the switch frequency.
            let could_switch = self.mass_matrix.background_count() >= switch_freq;
            let is_late = switch_freq + draw > self.final_step_size_window;

            let mut force_update = false;
            if could_switch && (!is_late) {
                self.mass_matrix.switch(math);
                force_update = true;
            }

            let did_change = if force_update
                | (draw - self.last_update >= self.options.mass_matrix_update_freq)
            {
                self.mass_matrix.update_potential(math, hamiltonian)
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
                self.step_size.update_stepsize(hamiltonian, false)
            }
            return Ok(());
        }

        self.step_size.update_estimator_late();
        let is_last = draw == self.num_tune - 1;
        self.step_size.update_stepsize(hamiltonian, is_last);
        Ok(())
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        Self::Collector::new(
            self.step_size.new_collector(),
            self.mass_matrix.new_collector(math),
        )
    }

    fn is_tuning(&self) -> bool {
        self.tuning
    }
}

#[derive(Debug, Clone)]
pub struct CombinedStats<D1, D2> {
    pub stats1: D1,
    pub stats2: D2,
}

#[derive(Clone)]
pub struct CombinedStatsBuilder<B1, B2> {
    stats1: B1,
    stats2: B2,
}

impl<S1, S2, B1, B2> StatTraceBuilder<CombinedStats<S1, S2>> for CombinedStatsBuilder<B1, B2>
where
    B1: StatTraceBuilder<S1>,
    B2: StatTraceBuilder<S2>,
{
    fn append_value(&mut self, value: CombinedStats<S1, S2>) {
        self.stats1.append_value(value.stats1);
        self.stats2.append_value(value.stats2);
    }

    fn finalize(self) -> Option<StructArray> {
        let Self { stats1, stats2 } = self;
        match (stats1.finalize(), stats2.finalize()) {
            (None, None) => None,
            (Some(stats1), None) => Some(stats1),
            (None, Some(stats2)) => Some(stats2),
            (Some(stats1), Some(stats2)) => {
                let mut data1 = stats1.into_parts();
                let data2 = stats2.into_parts();

                assert!(data1.2.is_none());
                assert!(data2.2.is_none());

                let mut fields = data1.0.into_iter().map(|x| x.deref().clone()).collect_vec();

                fields.extend(data2.0.into_iter().map(|x| x.deref().clone()));
                data1.1.extend(data2.1);

                Some(StructArray::new(data1.0, data1.1, None))
            }
        }
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self { stats1, stats2 } = self;
        match (stats1.inspect(), stats2.inspect()) {
            (None, None) => None,
            (Some(stats1), None) => Some(stats1),
            (None, Some(stats2)) => Some(stats2),
            (Some(stats1), Some(stats2)) => {
                let mut data1 = stats1.into_parts();
                let data2 = stats2.into_parts();

                assert!(data1.2.is_none());
                assert!(data2.2.is_none());

                let mut fields = data1.0.into_iter().map(|x| x.deref().clone()).collect_vec();

                fields.extend(data2.0.into_iter().map(|x| x.deref().clone()));
                data1.1.extend(data2.1);

                Some(StructArray::new(data1.0, data1.1, None))
            }
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
    ) {
        self.collector1
            .register_leapfrog(math, start, end, divergence_info);
        self.collector2
            .register_leapfrog(math, start, end, divergence_info);
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
    use crate::{cpu_math::CpuLogpFunc, math_base::LogpError};
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

    impl CpuLogpFunc for NormalLogp {
        type LogpError = NormalLogpError;
        type TransformParams = ();

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

        fn inv_transform_normalize(
            &mut self,
            _params: &Self::TransformParams,
            _untransformed_position: &[f64],
            _untransofrmed_gradient: &[f64],
            _transformed_position: &mut [f64],
            _transformed_gradient: &mut [f64],
        ) -> Result<f64, Self::LogpError> {
            unimplemented!()
        }

        fn init_from_transformed_position(
            &mut self,
            _params: &Self::TransformParams,
            _untransformed_position: &mut [f64],
            _untransformed_gradient: &mut [f64],
            _transformed_position: &[f64],
            _transformed_gradient: &mut [f64],
        ) -> Result<(f64, f64), Self::LogpError> {
            unimplemented!()
        }

        fn init_from_untransformed_position(
            &mut self,
            _params: &Self::TransformParams,
            _untransformed_position: &[f64],
            _untransformed_gradient: &mut [f64],
            _transformed_position: &mut [f64],
            _transformed_gradient: &mut [f64],
        ) -> Result<(f64, f64), Self::LogpError> {
            unimplemented!()
        }

        fn update_transformation<'a, R: rand::Rng + ?Sized>(
            &'a mut self,
            _rng: &mut R,
            _untransformed_positions: impl Iterator<Item = &'a [f64]>,
            _untransformed_gradients: impl Iterator<Item = &'a [f64]>,
            _params: &'a mut Self::TransformParams,
        ) -> Result<(), Self::LogpError> {
            unimplemented!()
        }

        fn new_transformation<R: rand::Rng + ?Sized>(
            &mut self,
            _rng: &mut R,
            _untransformed_position: &[f64],
            _untransfogmed_gradient: &[f64],
        ) -> Result<Self::TransformParams, Self::LogpError> {
            unimplemented!()
        }

        fn transformation_id(
            &self,
            _params: &Self::TransformParams,
        ) -> Result<i64, Self::LogpError> {
            unimplemented!()
        }
    }
}

#[cfg(test)]
mod test {
    use super::test_logps::NormalLogp;
    use super::*;
    use crate::{
        chain::NutsChain, cpu_math::CpuMath, euclidean_hamiltonian::EuclideanHamiltonian,
        mass_matrix::DiagMassMatrix, Chain, DiagAdaptExpSettings,
    };

    #[test]
    fn instanciate_adaptive_sampler() {
        use crate::mass_matrix_adapt::Strategy;

        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.);
        let mut math = CpuMath::new(func);
        let num_tune = 100;
        let options = EuclideanAdaptOptions::<DiagAdaptExpSettings>::default();
        let strategy = GlobalStrategy::<_, Strategy<_>>::new(&mut math, options, num_tune);

        let mass_matrix = DiagMassMatrix::new(&mut math, true);
        let max_energy_error = 1000f64;
        let step_size = 0.1f64;

        let hamiltonian =
            EuclideanHamiltonian::new(&mut math, mass_matrix, max_energy_error, step_size);
        let options = NutsOptions {
            maxdepth: 10u64,
            store_gradient: true,
            store_unconstrained: true,
            check_turning: true,
            store_divergences: false,
        };

        let rng = {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(42)
        };
        let chain = 0u64;

        let mut sampler = NutsChain::new(math, hamiltonian, strategy, options, rng, chain);
        sampler.set_position(&vec![1.5f64; ndim]).unwrap();
        for _ in 0..200 {
            sampler.draw().unwrap();
        }
    }
}
