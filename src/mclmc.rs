//! Unadjusted Microcanonical Langevin Monte Carlo (MCLMC) sampler.
//!
//! This module implements the **unadjusted** version of the MCLMC algorithm
//! (Robnik, De Luca, Silverstein & Seljak 2023), using the isokinetic Langevin
//! (Ornstein–Uhlenbeck) momentum refresh from the
//! [BlackJAX](https://github.com/blackjax-devs/blackjax) implementation.
//!
//! ## Algorithm (one draw)
//!
//! Let `L` be the momentum decoherence length, `ε` the step size, and `f` the
//! `subsample_frequency`.  Each call to [`MclmcChain::draw`] runs exactly
//! `num_steps = round(f · L / ε).max(1)` leapfrog steps:
//!
//! 1. **Partial momentum refresh** (half Langevin step with `half_step = ε/2`):
//!    - `ν = sqrt((exp(2·half_step/L) − 1) / n)`,  `n = dim`
//!    - `p ← (p + ν·z) / ‖p + ν·z‖`,  `z ~ N(0, I)`
//! 2. **`num_steps` ESH leapfrog steps** — the end state is the draw.
//! 3. **Partial momentum refresh** again (second half, same ν).
//!
//! No Metropolis accept/reject is performed — the draw is always accepted.
//! With `f = 1` each draw spans one full decoherence length; smaller `f`
//! produces more frequent (cheaper) draws.
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
    /// Accumulated energy change `ΔKE − Δlogp` over the draw's leapfrog steps.
    pub energy_change: f64,
    /// Whether the trajectory hit a divergence.
    pub diverging: bool,
    /// Full divergence details, if any.
    pub divergence_info: Option<DivergenceInfo>,
    /// Number of leapfrog steps actually taken.
    pub num_steps: u64,
}

// ── Stats structs ─────────────────────────────────────────────────────────────

/// Per-draw sampler statistics for [`MclmcChain`].
#[derive(Debug, Storable)]
pub struct MclmcStats<P: HasDims, H: Storable<P>, A: Storable<P>, Pt: Storable<P>> {
    pub chain: u64,
    pub draw: u64,
    pub num_steps: u64,
    pub energy_change: f64,
    /// Unnormalized log importance weight for this draw:
    /// `log_weight = log(step_size_factor) − energy_change`.
    ///
    /// `step_size_factor` is the factor applied to the base step size for the
    /// final leapfrog step (`1.0` when no dynamic retry occurred).
    /// `log(factor)` corrects for varying sampling density under dynamic step
    /// size reduction; it is `0.0` (no correction) when the step is taken at
    /// full size. The `−energy_change` term corrects for integration error.
    /// See Robnik & Seljak (2023), arXiv:2212.08549.
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
    /// Number of leapfrog steps per draw, expressed as a fraction of `L / ε`.
    ///
    /// `num_steps = round(subsample_frequency * L / ε).max(1)`
    ///
    /// - `1.0` → one full decoherence length per draw (default).
    /// - Smaller values → cheaper draws, more frequent adaptation updates.
    /// Scales naturally when `L` or `ε` is changed by adaptation.
    subsample_frequency: f64,
    /// When `true`, use the tree-structured step size retry: on divergence,
    /// halve the step size factor and try to make 2 steps before doubling
    /// back. `log_weight` always includes `log(step_size)` to correct for
    /// the varying sampling density. When `false`, `MAX_HALVINGS = 0` so
    /// divergences are recorded immediately without any retry.
    dynamic_step_size: bool,
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
        subsample_frequency: f64,
        dynamic_step_size: bool,
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
            subsample_frequency,
            dynamic_step_size,
            nuts_options: NutsOptions::default(),
            math: math.into(),
            stats_options,
            last_info: None,
        }
    }

    fn mclmc_kernel(&mut self) -> Result<(State<M, TransformedPoint<M>>, MclmcInfo)> {
        let math = self.math.get_mut();

        let base_step_size = self.hamiltonian.step_size();
        let momentum_decoherence_length = self
            .hamiltonian
            .momentum_decoherence_length()
            .unwrap_or(f64::INFINITY);
        // Each draw runs exactly this many leapfrog steps at the base step size.
        let num_steps = (self.subsample_frequency * momentum_decoherence_length / base_step_size)
            .round()
            .max(1.0) as u64;

        // ── First partial momentum refresh ───────────────────────────────────
        self.hamiltonian
            .partial_momentum_refresh(math, &mut self.state, &mut self.rng, 1.0)?;

        // Reset the energy baseline to the post-refresh state so that
        // energy_error() measures drift over this draw only.
        {
            let point = self
                .state
                .try_point_mut()
                .expect("State has no other references at start of draw");
            point.reset_trajectory_energy();
        }

        // ── num_steps ESH leapfrog steps with tree-structured step size retry ─
        //
        // When `dynamic_step_size` is enabled and a leapfrog step diverges, we
        // halve the step size factor and try to make 2 steps at the smaller
        // size. If those succeed we double back and continue; if one fails we
        // halve again. We cap the depth to MAX_HALVINGS; beyond that we record
        // a real divergence.  When `dynamic_step_size` is false MAX_HALVINGS=0
        // so any divergence is recorded immediately.
        let max_halvings: u32 = if self.dynamic_step_size { 10 } else { 0 };

        use crate::dynamics::LeapfrogResult;

        let mut current = self.state.clone();
        let mut divergence_info: Option<DivergenceInfo> = None;
        let mut steps_taken = 0u64;

        // `factor` is the current step size multiplier (always a power of 2).
        // `debt` is how many steps we still owe at the current factor before
        // we're allowed to double back up.
        let mut factor = 1.0_f64;
        let mut halvings: u32 = 0;
        let mut debt: u32 = 0; // steps remaining at the current halved level

        let mut remaining = num_steps;
        while remaining > 0 {
            let next = match self.hamiltonian.leapfrog(
                math,
                &current,
                crate::dynamics::Direction::Forward,
                factor,
                &mut self.collector,
            ) {
                LeapfrogResult::Ok(next) => next,
                LeapfrogResult::Divergence(info) => {
                    if halvings >= max_halvings {
                        // Genuinely diverged — give up.
                        divergence_info = Some(info);
                        break;
                    }
                    // Halve the step size and require 2 successful steps before
                    // we're allowed to double back.
                    factor *= 0.5;
                    halvings += 1;
                    debt = 2;
                    continue;
                }
                LeapfrogResult::Err(e) => {
                    return Err(NutsError::LogpFailure(e.into()).into());
                }
            };

            current = next;
            steps_taken += 1;
            remaining -= 1;

            if debt > 0 {
                debt -= 1;
                // Paid off our debt at this level — double back up.
                if debt == 0 && halvings > 0 {
                    factor *= 2.0;
                    halvings -= 1;
                    // If we're back at a halved level, we need to pay off that
                    // level's debt too (it had one step already paid, one remaining).
                    if halvings > 0 {
                        debt = 1;
                    }
                }
            }
        }

        let diverging = divergence_info.is_some();

        if diverging {
            // On divergence: stay at the pre-trajectory position, but fully
            // resample the momentum so the next draw does not reuse the
            // already-refreshed momentum from this failed trajectory.
            let mut next_state = self.hamiltonian.copy_state(math, &self.state);
            self.hamiltonian
                .initialize_trajectory(math, &mut next_state, &mut self.rng)?;
            let energy_change = current.point().energy_error();
            let info = MclmcInfo {
                energy_change,
                diverging: true,
                divergence_info,
                num_steps: steps_taken,
            };
            return Ok((next_state, info));
        }

        // Register the end state as the draw for the adaptation collector.
        let sample_info = crate::nuts::SampleInfo {
            depth: steps_taken,
            divergence_info: None,
            reached_maxdepth: false,
        };
        self.collector.register_draw(math, &current, &sample_info);

        let energy_change = current.point().energy_error();

        // ── Second partial momentum refresh ──────────────────────────────────
        let mut next_state = if current.try_point_mut().is_err() {
            self.hamiltonian.copy_state(math, &current)
        } else {
            current
        };

        self.hamiltonian
            .partial_momentum_refresh(math, &mut next_state, &mut self.rng, 1.0)?;

        let info = MclmcInfo {
            energy_change,
            diverging: false,
            divergence_info: None,
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
            log_weight: self.state.point().step_size_factor.ln() - info.energy_change,
            tuning: self.adapt.is_tuning(),
            hamiltonian: hamiltonian_stats,
            adapt: adapt_stats,
            point: point_stats,
            divergence: (
                info.divergence_info.as_ref(),
                options.divergence,
                self.draw_count,
            )
                .into(),
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

        // The collector was already fed during mclmc_kernel via register_leapfrog
        // on the sampled steps. Now call adapt with whatever was accumulated.
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

        let settings = crate::DiagMclmcSettings {
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
