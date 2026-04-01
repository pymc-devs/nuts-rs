//! Unadjusted Microcanonical Langevin Monte Carlo (MCLMC) sampler.
//!
//! This module implements the **unadjusted** version of the MCLMC algorithm
//! (Robnik, De Luca, Silverstein & Seljak 2023), using the isokinetic Langevin
//! (Ornstein–Uhlenbeck) momentum refresh from the
//! [BlackJAX](https://github.com/blackjax-devs/blackjax) implementation.
//!
//! ## Algorithm (one draw)
//!
//! Let `L` be the momentum decoherence length and `ε` the step size.
//! The number of leapfrog steps per draw is `num_steps = round(L / ε)`.
//!
//! 1. **Partial momentum refresh** (half Langevin step):
//!    - `ν = sqrt((exp(2·ε/L) − 1) / n)`,  `n = dim`
//!    - `p ← (p + ν·z) / ‖p + ν·z‖`,  `z ~ N(0, I)`
//! 2. **`num_steps` ESH leapfrog steps** using
//!    [`KineticEnergyKind::Microcanonical`] inside [`TransformedHamiltonian`].
//! 3. **Partial momentum refresh** again (second half, same ν).
//!
//! No Metropolis accept/reject is performed — the draw is always accepted.
//!
//! ## Adaptation
//!
//! [`MclmcChain`] is generic over any [`AdaptStrategy`] whose `Hamiltonian`
//! associated type is a [`TransformedHamiltonian`].  [`MclmcSettings`] wires
//! this up to [`GlobalStrategy`] configured with
//! [`StepSizeAdaptMethod::Fixed`] and `step_size_window = 0.0`, so the full
//! warmup budget is spent on mass-matrix adaptation and the step size is left
//! untouched.

use std::{
    cell::{Ref, RefCell},
    fmt::Debug,
    marker::PhantomData,
    ops::DerefMut,
};

use anyhow::Result;
use nuts_derive::Storable;
use nuts_storable::{HasDims, Storable};

use crate::{
    Math, NutsError,
    chain::{AdaptStrategy, Chain, StatOptions},
    dynamics::{DivergenceInfo, DivergenceStats, Hamiltonian, Point, State, TransformedPoint},
    nuts::{Collector, NutsOptions},
    sampler::Progress,
    sampler_stats::{SamplerStats, StatsDims},
};

// ── Per-draw diagnostic info ──────────────────────────────────────────────────

/// Diagnostic information returned for a single MCLMC draw.
#[derive(Debug, Clone)]
pub struct MclmcInfo {
    /// Accumulated energy change `ΔKE − Δlogp` over the trajectory.
    pub energy_change: f64,
    /// Whether the trajectory hit a divergence.
    pub diverging: bool,
    /// Full divergence details, if any.
    pub divergence_info: Option<DivergenceInfo>,
    /// Number of leapfrog steps actually taken (may be less than `num_steps`
    /// if a divergence was encountered).
    pub num_steps: u64,
}

// ── A null leapfrog collector ─────────────────────────────────────────────────

/// Passed to each individual leapfrog call inside the MCLMC loop.
/// Does nothing — the adaptation collector is driven at the draw level.
struct LeapfrogNullCollector;

impl<M: Math, P: Point<M>> Collector<M, P> for LeapfrogNullCollector {}

// ── Stats structs ─────────────────────────────────────────────────────────────

/// Per-draw sampler statistics for [`MclmcChain`].
#[derive(Debug, Storable)]
pub struct MclmcStats<P: HasDims, H: Storable<P>, A: Storable<P>, Pt: Storable<P>> {
    pub chain: u64,
    pub draw: u64,
    pub num_steps: u64,
    pub energy_change: f64,
    /// Unnormalized log importance weight for this draw: `−energy_change`.
    ///
    /// Unadjusted MCLMC draws are not exactly from the target distribution;
    /// reweighting by `exp(log_weight)` (normalized across draws) corrects for
    /// the bias.  See Robnik & Seljak (2023), arXiv:2212.08549.
    pub log_weight: f64,
    pub tuning: bool,
    #[storable(flatten)]
    pub hamiltonian: H,
    #[storable(flatten)]
    pub adapt: A,
    #[storable(flatten)]
    pub point: Pt,
    #[storable(flatten)]
    pub divergence: DivergenceStats,
    #[storable(ignore)]
    _phantom: PhantomData<fn() -> P>,
}

// ── MclmcChain ────────────────────────────────────────────────────────────────

/// Single-chain MCLMC sampler.
///
/// Generic over any [`AdaptStrategy`] `A` whose `Hamiltonian` is a
/// [`TransformedHamiltonian`].  Drives the ESH leapfrog manually (rather than
/// via the NUTS tree), applies the isokinetic Langevin momentum refresh around
/// each trajectory, and calls `A::adapt` after every draw so that the
/// mass-matrix transformation is kept up to date during warmup.
pub struct MclmcChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
    A::Hamiltonian: Hamiltonian<M, Point = TransformedPoint<M>>,
{
    hamiltonian: A::Hamiltonian,
    collector: A::Collector,
    adapt: A,
    state: State<M, TransformedPoint<M>>,
    rng: R,
    chain: u64,
    draw_count: u64,
    /// NutsOptions kept only to satisfy `AdaptStrategy::adapt`'s signature;
    /// MCLMC does not use tree-related options.
    nuts_options: NutsOptions,
    math: RefCell<M>,
    stats_options: StatOptions<M, A>,
    last_info: Option<MclmcInfo>,
}

impl<M, R, A> MclmcChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
    A::Hamiltonian: Hamiltonian<M, Point = TransformedPoint<M>>,
{
    pub fn new(
        mut math: M,
        mut hamiltonian: A::Hamiltonian,
        adapt: A,
        rng: R,
        chain: u64,
        stats_options: StatOptions<M, A>,
    ) -> Self {
        let state = hamiltonian.pool().new_state(&mut math);
        let collector = adapt.new_collector(&mut math);
        Self {
            hamiltonian,
            collector,
            adapt,
            state,
            rng,
            chain,
            draw_count: 0,
            nuts_options: NutsOptions::default(),
            math: math.into(),
            stats_options,
            last_info: None,
        }
    }

    fn mclmc_kernel(&mut self) -> Result<(State<M, TransformedPoint<M>>, MclmcInfo)> {
        let math = self.math.get_mut();

        let step_size = self.hamiltonian.step_size();
        let momentum_decoherence_length = self
            .hamiltonian
            .momentum_decoherence_length()
            .unwrap_or(f64::INFINITY);
        let num_steps = (momentum_decoherence_length / step_size).round().max(1.0) as u64;
        let half_eps = step_size * 0.5;

        // ── First partial momentum refresh ───────────────────────────────────
        self.hamiltonian
            .refresh_momentum(math, &mut self.state, &mut self.rng, half_eps)?;

        // Reset the energy baseline to the post-refresh state so that
        // energy_error() measures drift over this trajectory only.
        {
            let point = self
                .state
                .try_point_mut()
                .expect("State has no other references at start of draw");
            point.reset_trajectory_energy();
        }

        // ── num_steps ESH leapfrog steps ─────────────────────────────────────
        let mut lf_collector = LeapfrogNullCollector;
        let mut current = self.state.clone();
        let mut divergence_info: Option<DivergenceInfo> = None;
        let mut steps_taken = 0u64;

        for _ in 0..num_steps {
            use crate::dynamics::LeapfrogResult;
            match self.hamiltonian.leapfrog(
                math,
                &current,
                crate::dynamics::Direction::Forward,
                &mut lf_collector,
            ) {
                LeapfrogResult::Ok(next) => {
                    current = next;
                    steps_taken += 1;
                }
                LeapfrogResult::Divergence(info) => {
                    divergence_info = Some(info);
                    break;
                }
                LeapfrogResult::Err(e) => {
                    return Err(NutsError::LogpFailure(e.into()).into());
                }
            }
        }

        let energy_change = current.point().energy_error();
        let diverging = divergence_info.is_some();

        // ── Second partial momentum refresh ──────────────────────────────────
        // Ensure exclusive access before mutating the momentum.
        let mut next_state = if current.try_point_mut().is_err() {
            self.hamiltonian.copy_state(math, &current)
        } else {
            current
        };

        self.hamiltonian
            .refresh_momentum(math, &mut next_state, &mut self.rng, half_eps)?;

        let info = MclmcInfo {
            energy_change,
            diverging,
            divergence_info,
            num_steps: steps_taken,
        };

        Ok((next_state, info))
    }
}

// ── SamplerStats ──────────────────────────────────────────────────────────────

impl<M, R, A> SamplerStats<M> for MclmcChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
    A::Hamiltonian: Hamiltonian<M, Point = TransformedPoint<M>>,
{
    type Stats = MclmcStats<
        StatsDims,
        <A::Hamiltonian as SamplerStats<M>>::Stats,
        A::Stats,
        <TransformedPoint<M> as SamplerStats<M>>::Stats,
    >;
    type StatsOptions = StatOptions<M, A>;

    fn extract_stats(&self, math: &mut M, options: Self::StatsOptions) -> Self::Stats {
        let info = self
            .last_info
            .as_ref()
            .expect("Sampler has not started yet");
        let hamiltonian_stats = self.hamiltonian.extract_stats(math, options.hamiltonian);
        let adapt_stats = self.adapt.extract_stats(math, options.adapt);
        let point_stats = self.state.point().extract_stats(math, options.point);
        MclmcStats {
            chain: self.chain,
            draw: self.draw_count,
            num_steps: info.num_steps,
            energy_change: info.energy_change,
            log_weight: -info.energy_change,
            tuning: self.adapt.is_tuning(),
            hamiltonian: hamiltonian_stats,
            adapt: adapt_stats,
            point: point_stats,
            divergence: info.divergence_info.as_ref().into(),
            _phantom: PhantomData,
        }
    }
}

// ── Chain impl ────────────────────────────────────────────────────────────────

impl<M, R, A> Chain<M> for MclmcChain<M, R, A>
where
    M: Math,
    R: rand::Rng,
    A: AdaptStrategy<M>,
    A::Hamiltonian: Hamiltonian<M, Point = TransformedPoint<M>>,
{
    type AdaptStrategy = A;

    fn set_position(&mut self, position: &[f64]) -> Result<()> {
        let mut math_ = self.math.borrow_mut();
        let math = math_.deref_mut();
        self.adapt.init(
            math,
            &mut self.nuts_options,
            &mut self.hamiltonian,
            position,
            &mut self.rng,
        )?;
        self.state = self.hamiltonian.init_state(math, position)?;
        // Initialise momentum to a random unit vector.
        self.hamiltonian
            .initialize_trajectory(math, &mut self.state, &mut self.rng)?;
        Ok(())
    }

    fn draw(&mut self) -> Result<(Box<[f64]>, Progress)> {
        let (state, info) = self.mclmc_kernel()?;

        let position: Box<[f64]> = {
            let mut math_ = self.math.borrow_mut();
            let math = math_.deref_mut();
            let mut pos = vec![0f64; math.dim()];
            state.write_position(math, &mut pos);
            pos.into()
        };

        let progress = Progress {
            draw: self.draw_count,
            chain: self.chain,
            diverging: info.diverging,
            tuning: self.adapt.is_tuning(),
            step_size: self.hamiltonian.step_size(),
            num_steps: info.num_steps,
        };

        // Drive adaptation (mass-matrix update) from the end state of the trajectory.
        // We synthesise a minimal SampleInfo so the adapt collector's register_draw
        // picks up the position/gradient correctly.
        {
            let sample_info = crate::nuts::SampleInfo {
                depth: info.num_steps,
                divergence_info: info.divergence_info.clone(),
                reached_maxdepth: false,
            };
            self.collector
                .register_draw(self.math.get_mut(), &state, &sample_info);
        }

        {
            let mut math_ = self.math.borrow_mut();
            let math = math_.deref_mut();
            self.adapt.adapt(
                math,
                &mut self.nuts_options,
                &mut self.hamiltonian,
                self.draw_count,
                &self.collector,
                &state,
                &mut self.rng,
            )?;
            // Refresh the collector for the next draw.
            self.collector = self.adapt.new_collector(math);
        }

        self.draw_count += 1;
        self.state = state;
        self.last_info = Some(info);
        Ok((position, progress))
    }

    fn dim(&self) -> usize {
        self.math.borrow().dim()
    }

    fn expanded_draw(&mut self) -> Result<(Box<[f64]>, M::ExpandedVector, Self::Stats, Progress)> {
        let (position, progress) = self.draw()?;
        let mut math_ = self.math.borrow_mut();
        let math = math_.deref_mut();
        let stats = self.extract_stats(math, self.stats_options);
        let expanded = math.expand_vector(&mut self.rng, self.state.point().position())?;
        Ok((position, expanded, stats, progress))
    }

    fn math(&self) -> Ref<'_, M> {
        self.math.borrow()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use rand::rng;

    use crate::{Chain, adapt_strategy::test_logps::NormalLogp, math::CpuMath, sampler::Settings};

    #[test]
    fn mclmc_draws_normal() {
        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.0);
        let math = CpuMath::new(func);

        let settings = crate::MclmcSettings {
            step_size: 0.5,
            momentum_decoherence_length: 3.0,
            num_tune: 200,
            num_draws: 500,
            ..crate::MclmcSettings::default()
        };

        let mut rng = rng();
        let mut chain = settings.new_chain(0, math, &mut rng);
        chain.set_position(&vec![0.0f64; ndim]).unwrap();

        let mut last_pos = vec![0.0f64; ndim];
        for _ in 0..500 {
            let (draw, progress) = chain.draw().unwrap();
            assert!(!progress.diverging, "unexpected divergence");
            last_pos.copy_from_slice(&draw);
        }

        // After 500 draws the chain should be exploring near the mean (3.0).
        let mean: f64 = last_pos.iter().sum::<f64>() / ndim as f64;
        assert!(
            (mean - 3.0).abs() < 3.0,
            "mean {mean} too far from expected 3.0"
        );
    }
}
