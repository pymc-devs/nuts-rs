//! Drive a single sampling chain by wiring together the Hamiltonian, adaptation, and per-draw bookkeeping.

use std::{
    cell::{Ref, RefCell},
    fmt::Debug,
    marker::PhantomData,
    ops::DerefMut,
};

use nuts_storable::{HasDims, Storable};
use rand::Rng;

use crate::{
    Math, NutsError,
    dynamics::{DivergenceStats, Hamiltonian, Point, State},
    nuts::{Collector, NutsOptions, SampleInfo, draw},
    sampler::Progress,
    sampler_stats::{SamplerStats, StatsDims},
    warmup::WarmupPhase,
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
    fn draw(&mut self) -> Result<(Box<[f64]>, Progress)>;

    /// The dimensionality of the posterior.
    fn dim(&self) -> usize;

    fn expanded_draw(&mut self) -> Result<(Box<[f64]>, M::ExpandedVector, Self::Stats, Progress)>;

    fn math(&self) -> Ref<'_, M>;
}

/// The last draw of a chain, used for the sampler stats.
pub(crate) struct LastDraw<I> {
    /// 0-based index of the draw.
    pub(crate) draw: u64,
    pub(crate) info: I,
}

pub struct NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    hamiltonian: A::Hamiltonian,
    collector: A::Collector,
    options: NutsOptions,
    rng: R,
    state: State<M, <A::Hamiltonian as Hamiltonian<M>>::Point>,
    last_draw: Option<LastDraw<SampleInfo>>,
    chain: u64,
    /// Index of the next draw.
    draw_count: u64,
    strategy: A,
    math: RefCell<M>,
    stats_options: StatOptions<M, A>,
}

impl<M, R, A> NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    pub fn new(
        mut math: M,
        mut hamiltonian: A::Hamiltonian,
        strategy: A,
        options: NutsOptions,
        rng: R,
        chain: u64,
        stats_options: StatOptions<M, A>,
    ) -> Self {
        let init = hamiltonian.pool().new_state(&mut math);
        let collector = strategy.new_collector(&mut math);
        NutsChain {
            hamiltonian,
            collector,
            options,
            rng,
            state: init,
            last_draw: None,
            chain,
            draw_count: 0,
            strategy,
            math: math.into(),
            stats_options,
        }
    }
}

pub trait AdaptStrategy<M: Math>: SamplerStats<M> {
    type Hamiltonian: Hamiltonian<M>;
    type Collector: Collector<M, <Self::Hamiltonian as Hamiltonian<M>>::Point>;
    type Options: Copy + Send + Debug + Default;

    fn new(math: &mut M, options: Self::Options, num_tune: u64, chain: u64) -> Self;

    fn init<R: Rng + ?Sized>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        position: &[f64],
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
    /// The warmup phase of the draw with index `draw`.
    fn phase(&self, draw: u64) -> WarmupPhase;
    fn last_num_steps(&self) -> u64;
    fn last_num_gradients(&self) -> u64;
}

impl<M, R, A> Chain<M> for NutsChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
{
    type AdaptStrategy = A;

    fn set_position(&mut self, position: &[f64]) -> Result<()> {
        if position.len() != self.dim() {
            anyhow::bail!(
                "Initial position has length {}, but the model has {} dimensions",
                position.len(),
                self.dim()
            );
        }
        let mut math_ = self.math.borrow_mut();
        let math = math_.deref_mut();
        // The initial step size search should already use the settings of
        // the first draw.
        let phase = self.strategy.phase(self.draw_count);
        self.options.set_phase(&mut self.hamiltonian, phase);
        self.strategy.init(
            math,
            &mut self.options,
            &mut self.hamiltonian,
            position,
            &mut self.rng,
        )?;
        self.state = self.hamiltonian.init_state(math, position)?;
        Ok(())
    }

    fn draw(&mut self) -> Result<(Box<[f64]>, Progress)> {
        let mut math_ = self.math.borrow_mut();
        let math = math_.deref_mut();
        let phase = self.strategy.phase(self.draw_count);
        self.options.set_phase(&mut self.hamiltonian, phase);
        let (state, info) = draw(
            math,
            &mut self.state,
            &mut self.rng,
            &mut self.hamiltonian,
            &self.options,
            &mut self.collector,
        )?;
        let mut position: Box<[f64]> = vec![0f64; math.dim()].into();
        state.write_position(math, &mut position);

        self.strategy.adapt(
            math,
            &mut self.options,
            &mut self.hamiltonian,
            self.draw_count,
            &self.collector,
            &state,
            &mut self.rng,
        )?;
        let progress = Progress {
            draw: self.draw_count,
            chain: self.chain,
            diverging: info.divergence_info.is_some(),
            tuning: self.strategy.is_tuning(),
            step_size: self.hamiltonian.step_size(),
            num_steps: self.strategy.last_num_steps(),
            num_gradients: self.strategy.last_num_gradients(),
            fisher_distance: state.point().fisher_distance(math),
        };

        self.last_draw = Some(LastDraw {
            draw: self.draw_count,
            info,
        });
        self.draw_count += 1;

        self.state = state;
        Ok((position, progress))
    }

    fn expanded_draw(&mut self) -> Result<(Box<[f64]>, M::ExpandedVector, Self::Stats, Progress)> {
        let (position, progress) = self.draw()?;
        let mut math_ = self.math.borrow_mut();
        let math = math_.deref_mut();

        let stats = self.extract_stats(&mut *math, self.stats_options);
        // Update the stats_options of the hamiltonian. This is used to
        // store only changes in the transformation.
        self.stats_options.hamiltonian = self
            .hamiltonian
            .update_stats_options(&mut *math, self.stats_options.hamiltonian);
        let expanded = math.expand_vector(&mut self.rng, self.state.point().position())?;

        Ok((position, expanded, stats, progress))
    }

    fn dim(&self) -> usize {
        self.math.borrow().dim()
    }

    fn math(&self) -> Ref<'_, M> {
        self.math.borrow()
    }
}

#[derive(Debug, nuts_derive::Storable)]
pub struct NutsStats<P: HasDims, H: Storable<P>, A: Storable<P>, D: Storable<P>> {
    pub depth: u64,
    pub maxdepth_reached: bool,
    /// Whether the trajectory was stopped because extending it would have
    /// needed more than `max_leapfrog_steps` leapfrog steps.
    pub max_leapfrog_steps_reached: bool,
    /// Whether the trajectory was stopped because a WALNUTS macro step was
    /// not reversible. `None` without WALNUTS.
    pub irreversible: Option<bool>,
    /// The number of micro steps of the first attempt of each WALNUTS macro
    /// step. Changes during tuning if `target_macro_steps` is set. `None`
    /// without WALNUTS.
    pub min_micro_steps: Option<u64>,
    pub chain: u64,
    pub draw: u64,
    /// The warmup phase of the draw, see [`WarmupPhase::as_str`].
    pub warmup_phase: String,
    #[storable(flatten)]
    pub hamiltonian: H,
    #[storable(flatten)]
    pub adapt: A,
    #[storable(flatten)]
    pub point: D,
    #[storable(flatten)]
    pub divergence: DivergenceStats,
    #[storable(ignore)]
    _phantom: PhantomData<fn() -> P>,
}

pub struct StatOptions<M: Math, A: AdaptStrategy<M>> {
    pub adapt: A::StatsOptions,
    pub hamiltonian: <A::Hamiltonian as SamplerStats<M>>::StatsOptions,
    pub point: <<A::Hamiltonian as Hamiltonian<M>>::Point as SamplerStats<M>>::StatsOptions,
    pub divergence: crate::dynamics::DivergenceStatsOptions,
}

impl<M, A> Clone for StatOptions<M, A>
where
    M: Math,
    A: AdaptStrategy<M>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<M, A> Copy for StatOptions<M, A>
where
    M: Math,
    A: AdaptStrategy<M>,
{
}

impl<M: Math, R: rand::Rng, A: AdaptStrategy<M>> SamplerStats<M> for NutsChain<M, R, A> {
    type Stats = NutsStats<
        StatsDims,
        <A::Hamiltonian as SamplerStats<M>>::Stats,
        A::Stats,
        <<A::Hamiltonian as Hamiltonian<M>>::Point as SamplerStats<M>>::Stats,
    >;
    type StatsOptions = StatOptions<M, A>;

    fn extract_stats(&self, math: &mut M, options: Self::StatsOptions) -> Self::Stats {
        let hamiltonian_stats = self.hamiltonian.extract_stats(math, options.hamiltonian);
        let adapt_stats = self.strategy.extract_stats(math, options.adapt);
        let point_stats = self.state.point().extract_stats(math, options.point);
        let last = self.last_draw.as_ref().expect("Sampler has not started");
        let info = &last.info;
        let div_info = info.divergence_info.as_ref();

        NutsStats {
            depth: info.depth,
            maxdepth_reached: info.reached_maxdepth,
            max_leapfrog_steps_reached: info.reached_max_leapfrog_steps,
            irreversible: self.options.walnuts.map(|_| info.irreversible),
            min_micro_steps: self.options.walnuts.map(|w| w.min_micro_steps),
            chain: self.chain,
            draw: last.draw,
            warmup_phase: self.options.phase.as_str().to_string(),
            hamiltonian: hamiltonian_stats,
            adapt: adapt_stats,
            point: point_stats,
            divergence: (div_info, options.divergence, last.draw).into(),
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::ChaCha8Rng};

    use crate::{
        Chain, DiagMclmcSettings, DiagNutsSettings, Settings, StepSizeAdaptMethod, math::CpuMath,
        math::test_logps::NormalLogp,
    };

    /// The stats of a draw contain its 0-based index, like the progress.
    #[test]
    fn draw_index_in_stats() {
        let mut nuts = DiagNutsSettings {
            num_tune: 20,
            ..Default::default()
        };
        // A huge step size, so that the draws diverge.
        nuts.adapt_options.step_size_settings.adapt_options.method =
            StepSizeAdaptMethod::Fixed(100.0);
        nuts.adapt_options.step_size_settings.jitter = None;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut chain = nuts
            .new_chain(0, CpuMath::new(NormalLogp::new(10, 3.0)), &mut rng)
            .unwrap();
        chain.set_position(&[0.1; 10]).unwrap();
        let mut num_divergences = 0;
        for draw in 0..30 {
            let (_, _, stats, progress) = chain.expanded_draw().unwrap();
            assert_eq!(progress.draw, draw);
            assert_eq!(stats.draw, draw);
            if stats.divergence.diverging {
                num_divergences += 1;
                assert_eq!(stats.divergence.divergence_draw, Some(draw));
            }
        }
        assert!(num_divergences > 0);

        let mclmc = DiagMclmcSettings {
            num_tune: 20,
            ..Default::default()
        };
        let mut chain = mclmc
            .new_chain(0, CpuMath::new(NormalLogp::new(10, 3.0)), &mut rng)
            .unwrap();
        chain.set_position(&[0.1; 10]).unwrap();
        for draw in 0..30 {
            let (_, _, stats, progress) = chain.expanded_draw().unwrap();
            assert_eq!(progress.draw, draw);
            assert_eq!(stats.draw, draw);
            if stats.divergence.diverging {
                assert_eq!(stats.divergence.divergence_draw, Some(draw));
            }
        }
    }
}
