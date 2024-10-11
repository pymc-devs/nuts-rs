use std::fmt::Debug;

use rand::Rng;

use crate::{
    hamiltonian::{Hamiltonian, Point},
    nuts::{draw, Collector, NutsOptions, NutsSampleStats, NutsStatsBuilder},
    sampler_stats::SamplerStats,
    state::{State, StatePool},
    Math, NutsError, Settings,
};

use anyhow::Result;

/// Draw samples from the posterior distribution using Hamiltonian MCMC.
pub trait Chain<M: Math>: SamplerStats<M> {
    type AdaptStrategy: AdaptStrategy<M>;

    /// Initialize the sampler to a position. This should be called
    /// before calling draw.
    ///
    /// This fails if the logp function returns an error.
    fn set_position(&mut self, position: &[f64]) -> Result<()>;

    /// Draw a new sample and return the position and some diagnosic information.
    fn draw(&mut self) -> Result<(Box<[f64]>, Self::Stats)>;

    /// The dimensionality of the posterior.
    fn dim(&self) -> usize;
}

pub struct NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    pool: StatePool<M, <A::Hamiltonian as Hamiltonian<M>>::Point>,
    hamiltonian: A::Hamiltonian,
    collector: A::Collector,
    options: NutsOptions,
    rng: R,
    init: State<M, <A::Hamiltonian as Hamiltonian<M>>::Point>,
    chain: u64,
    draw_count: u64,
    strategy: A,
    math: M,
    stats: Option<NutsSampleStats<<A::Hamiltonian as SamplerStats<M>>::Stats, A::Stats>>,
}

impl<M, R, A> NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    pub fn new(
        mut math: M,
        hamiltonian: A::Hamiltonian,
        strategy: A,
        options: NutsOptions,
        rng: R,
        chain: u64,
    ) -> Self {
        let pool_size: usize = options.maxdepth.checked_mul(2).unwrap().try_into().unwrap();
        let pool = hamiltonian.new_pool(&mut math, pool_size);
        let init = pool.new_state(&mut math);
        let collector = strategy.new_collector(&mut math);
        NutsChain {
            pool,
            hamiltonian,
            collector,
            options,
            rng,
            init,
            chain,
            draw_count: 0,
            strategy,
            math,
            stats: None,
        }
    }
}

pub trait AdaptStrategy<M: Math>: SamplerStats<M> {
    type Hamiltonian: Hamiltonian<M>;
    type Collector: Collector<M, <Self::Hamiltonian as Hamiltonian<M>>::Point>;
    type Options: Copy + Send + Debug + Default;

    fn new(math: &mut M, options: Self::Options, num_tune: u64) -> Self;

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        state: &State<M, <Self::Hamiltonian as Hamiltonian<M>>::Point>,
        rng: &mut R,
    ) -> Result<(), NutsError>;

    #[allow(clippy::too_many_arguments)]
    fn adapt<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        draw: u64,
        collector: &Self::Collector,
        state: &State<M, <Self::Hamiltonian as Hamiltonian<M>>::Point>,
        rng: &mut R,
    ) -> Result<(), NutsError>;

    fn new_collector(&self, math: &mut M) -> Self::Collector;
    fn is_tuning(&self) -> bool;
}

impl<M, R, A> SamplerStats<M> for NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    type Builder = NutsStatsBuilder<
        <A::Hamiltonian as SamplerStats<M>>::Builder,
        <A as SamplerStats<M>>::Builder,
    >;
    type Stats =
        NutsSampleStats<<A::Hamiltonian as SamplerStats<M>>::Stats, <A as SamplerStats<M>>::Stats>;

    fn new_builder(&self, settings: &impl Settings, dim: usize) -> Self::Builder {
        NutsStatsBuilder::new_with_capacity(
            settings,
            &self.hamiltonian,
            &self.strategy,
            dim,
            &self.options,
        )
    }

    fn current_stats(&self, _math: &mut M) -> Self::Stats {
        self.stats.as_ref().expect("No stats available").clone()
    }
}

impl<M, R, A> Chain<M> for NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    type AdaptStrategy = A;

    fn set_position(&mut self, position: &[f64]) -> Result<()> {
        let state = self
            .hamiltonian
            .init_state(&mut self.math, &mut self.pool, position)?;
        self.init = state;
        self.strategy.init(
            &mut self.math,
            &mut self.options,
            &mut self.hamiltonian,
            &self.init,
            &mut self.rng,
        )?;
        Ok(())
    }

    fn draw(&mut self) -> Result<(Box<[f64]>, Self::Stats)> {
        let (state, info) = draw(
            &mut self.math,
            &mut self.pool,
            &mut self.init,
            &mut self.rng,
            &mut self.hamiltonian,
            &self.options,
            &mut self.collector,
        )?;
        let mut position: Box<[f64]> = vec![0f64; self.math.dim()].into();
        state.write_position(&mut self.math, &mut position);

        let stats = NutsSampleStats {
            depth: info.depth,
            maxdepth_reached: info.reached_maxdepth,
            idx_in_trajectory: state.index_in_trajectory(),
            logp: state.point().logp(),
            energy: state.point().energy(),
            energy_error: info.draw_energy - info.initial_energy,
            divergence_info: info.divergence_info,
            chain: self.chain,
            draw: self.draw_count,
            potential_stats: self.hamiltonian.current_stats(&mut self.math),
            strategy_stats: self.strategy.current_stats(&mut self.math),
            gradient: if self.options.store_gradient {
                let mut gradient: Box<[f64]> = vec![0f64; self.math.dim()].into();
                state.write_gradient(&mut self.math, &mut gradient);
                Some(gradient)
            } else {
                None
            },
            unconstrained: if self.options.store_unconstrained {
                let mut unconstrained: Box<[f64]> = vec![0f64; self.math.dim()].into();
                state.write_position(&mut self.math, &mut unconstrained);
                Some(unconstrained)
            } else {
                None
            },
            tuning: self.strategy.is_tuning(),
        };

        self.strategy.adapt(
            &mut self.math,
            &mut self.options,
            &mut self.hamiltonian,
            self.draw_count,
            &self.collector,
            &state,
            &mut self.rng,
        )?;

        self.draw_count += 1;

        self.init = state;
        Ok((position, stats))
    }

    fn dim(&self) -> usize {
        self.math.dim()
    }
}
