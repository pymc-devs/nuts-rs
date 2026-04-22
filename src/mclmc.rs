//! Unadjusted Microcanonical Langevin Monte Carlo (MCLMC) sampler.
//!
//! > ⚠️ **Experimental — use with caution**: The MCLMC sampler and all of its
//! > variants are highly experimental. They have not been thoroughly validated
//! > and may **not return correct posteriors**. The API, defaults, and
//! > adaptation behaviour are all subject to breaking changes at any time.
//! > Do not use these samplers in production or for results you rely on.
//!
//! This module implements the **unadjusted** version of the MCLMC algorithm
//! (Robnik, De Luca, Silverstein & Seljak 2023), using the isokinetic Langevin
//! (Ornstein–Uhlenbeck) momentum refresh from the
//! [BlackJAX](https://github.com/blackjax-devs/blackjax) implementation.

use std::{
    cell::{Ref, RefCell},
    fmt::Debug,
    marker::PhantomData,
    ops::DerefMut,
};

use anyhow::{Result, bail};
use nuts_derive::Storable;
use nuts_storable::{HasDims, Storable};
use rand_distr::num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::{
    Math, NutsError,
    chain::{AdaptStrategy, Chain, StatOptions},
    dynamics::{
        Direction, DivergenceInfo, DivergenceStats, Hamiltonian, KineticEnergyKind, Point, State,
        TransformedHamiltonian, TransformedPoint,
    },
    nuts::{Collector, NutsOptions},
    sampler::Progress,
    sampler_stats::{SamplerStats, StatsDims},
    transform::Transformation,
};

// ── Trajectory kind ───────────────────────────────────────────────────────────

/// Selects which leapfrog integrator and partial momentum refresh are used by
/// [`MclmcChain`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MclmcTrajectoryKind {
    /// Microcanonical (ESH) trajectory with isokinetic Langevin momentum
    /// refresh — original MCLMC (default).
    ///
    /// Momentum lives on the unit sphere; the partial refresh is the
    /// isokinetic Ornstein–Uhlenbeck step that keeps `‖p‖ = 1`.
    #[default]
    Microcanonical,

    /// Standard Euclidean HMC trajectory with Ornstein–Uhlenbeck partial
    /// momentum refresh.
    ///
    /// Momentum is Gaussian `N(0, I)`.  The partial refresh is
    /// `p_new = α·p + sqrt(1−α²)·z`, `z ~ N(0, I)`,
    /// `α = exp(−ε/(2L))`.
    Euclidean,

    /// Use the Euclidean trajectory during the early tuning window (the first
    /// `trajectory_switch_fraction · num_tune` draws), then switch permanently
    /// to the Microcanonical trajectory for the remainder of warmup and all
    /// post-warmup draws.
    ///
    /// This can ease mass-matrix estimation in the early phase while still
    /// benefiting from the efficiency of the microcanonical trajectory later.
    EuclideanEarlyThenMicrocanonical,
}

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
    pub average_step_size: f64,
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
    pub average_step_size: f64,
    #[storable(flatten)]
    pub divergence: DivergenceStats,
    #[storable(ignore)]
    _phantom: PhantomData<fn() -> P>,
}

// ── MclmcChain ────────────────────────────────────────────────────────────────

/// Single-chain MCLMC sampler.
pub struct MclmcChain<M, R, A, T>
where
    M: Math,
    R: rand::Rng,
    T: Transformation<M>,
    A: AdaptStrategy<M, Hamiltonian = TransformedHamiltonian<M, T>>,
{
    hamiltonian: TransformedHamiltonian<M, T>,
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
    /// the varying sampling density. When `false`, divergences are recorded
    /// immediately without any retry and `log_weight = -energy_change`.
    dynamic_step_size: bool,
    /// Which trajectory kind is currently requested.
    trajectory_kind: MclmcTrajectoryKind,
    /// For [`MclmcTrajectoryKind::EuclideanEarlyThenMicrocanonical`]: the
    /// draw index at which to permanently switch from Euclidean to
    /// Microcanonical.  Ignored for other trajectory kinds.
    switch_draw: u64,
    max_energy_error: f64,
    /// NutsOptions kept only to satisfy `AdaptStrategy::adapt`'s signature;
    /// MCLMC does not use tree-related options.
    nuts_options: NutsOptions,
    math: RefCell<M>,
    stats_options: StatOptions<M, A>,
    last_info: Option<MclmcInfo>,
    tmp_velocity: M::Vector,
}

impl<M, R, A, T> MclmcChain<M, R, A, T>
where
    M: Math,
    R: rand::Rng,
    T: Transformation<M>,
    A: AdaptStrategy<M, Hamiltonian = TransformedHamiltonian<M, T>>,
{
    pub fn new(
        mut math: M,
        mut hamiltonian: TransformedHamiltonian<M, T>,
        adapt: A,
        rng: R,
        chain: u64,
        subsample_frequency: f64,
        dynamic_step_size: bool,
        trajectory_kind: MclmcTrajectoryKind,
        switch_draw: u64,
        max_energy_error: f64,
        stats_options: StatOptions<M, A>,
    ) -> Self {
        let state = hamiltonian.pool().new_state(&mut math);
        let collector = adapt.new_collector(&mut math);
        let tmp_velocity = math.new_array();
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
            trajectory_kind,
            switch_draw,
            nuts_options: NutsOptions::default(),
            math: math.into(),
            stats_options,
            last_info: None,
            tmp_velocity,
            max_energy_error,
        }
    }

    fn mclmc_kernel(
        &mut self,
        resample_velocity: bool,
    ) -> Result<(State<M, TransformedPoint<M>>, MclmcInfo)> {
        let math = self.math.get_mut();

        let base_step_size = self.hamiltonian.step_size();
        let num_base_steps: u64 = self
            .hamiltonian
            .momentum_decoherence_length()
            .map(|length| {
                let num_steps = (self.subsample_frequency * length / base_step_size)
                    .round()
                    .max(1.0)
                    .min(1e6);
                if !num_steps.is_finite() {
                    bail!("Invalid number of integration steps");
                }
                Ok(num_steps as u64)
            })
            .unwrap_or(Ok(1))?;

        // ── num_steps leapfrog steps with tree-structured step size retry ────
        //
        // When `dynamic_step_size` is enabled and a leapfrog step diverges, we
        // halve the step size factor and try to make 2 steps at the smaller
        // size. If those succeed we double back and continue; if one fails we
        // halve again. We cap the depth to MAX_HALVINGS; beyond that we record
        // a real divergence.  When `dynamic_step_size` is false MAX_HALVINGS=0
        // so any divergence is recorded immediately.
        let max_halvings: u64 = if self.dynamic_step_size { 10 } else { 0 };

        use crate::dynamics::LeapfrogResult;

        // TODO make this copy conditional
        let mut current = self.hamiltonian.copy_state(math, &self.state);

        self.hamiltonian.initialize_trajectory(
            math,
            &mut current,
            resample_velocity,
            &mut self.rng,
        )?;

        let ones = {
            let mut ones = math.new_array();
            math.fill_array(&mut ones, 1.0);
            ones
        };
        let mut momentum_noise = math.new_array();
        math.array_gaussian(&mut self.rng, &mut momentum_noise, &ones);

        // Capture the draw-start energy once; used at the end to compute
        // MclmcInfo.energy_change independently of the per-step baselines.
        let draw_start_energy = current.point().energy();

        let mut divergence_info: Option<DivergenceInfo> = None;
        let mut steps_taken = 0u64;

        // `factor` is the current step size multiplier (always a power of 2).
        let mut factor = 1.0_f64;

        let mut remaining_stack: Vec<u64> = Vec::with_capacity(max_halvings.try_into().unwrap());

        let mut remaining = num_base_steps;
        let mut time = 0.0;

        while remaining > 0 {
            // Store the current momentum in case we need to try again
            // with smaller step size
            math.copy_into(&current.point().velocity, &mut self.tmp_velocity);

            self.hamiltonian.partial_momentum_refresh(
                math,
                &mut current,
                &momentum_noise,
                &mut self.rng,
                factor,
            )?;

            // Use the post-refresh energy as the divergence baseline so that
            // the leapfrog's energy_error measures only this single step's
            // integration error (O(ε²)), not the cumulative drift from the
            // draw start.  Without this, many small steps with a tight
            // max_energy_error threshold can exhaust all halvings because the
            // accumulated baseline already sits near the threshold.
            let step_baseline = current.point().energy();
            // We normalize the max energy error by the length of the step
            match self.hamiltonian.leapfrog(
                math,
                &current,
                Direction::Forward,
                factor,
                step_baseline,
                // TODO: Should not include subsample_freq?
                self.max_energy_error * factor / num_base_steps.to_f64().unwrap(),
                &mut self.collector,
            ) {
                LeapfrogResult::Ok(mut next) => {
                    math.array_gaussian(&mut self.rng, &mut momentum_noise, &ones);
                    self.hamiltonian.partial_momentum_refresh(
                        math,
                        &mut next,
                        &momentum_noise,
                        &mut self.rng,
                        factor,
                    )?;
                    // Sample for next momentum refresh
                    math.array_gaussian(&mut self.rng, &mut momentum_noise, &ones);
                    current = next;
                    steps_taken += 1;
                    remaining -= 1;
                    time += factor * base_step_size;

                    while remaining == 0 {
                        if let Some(prev_remaining) = remaining_stack.pop() {
                            remaining = prev_remaining - 1;
                            factor *= 2.0;
                        } else {
                            break;
                        }
                    }
                }
                LeapfrogResult::Divergence(info) => {
                    if remaining_stack.len() >= max_halvings.try_into().unwrap() {
                        // Genuinely diverged — give up.
                        divergence_info = Some(info);
                        break;
                    }
                    // Halve the step size and require 2 successful steps before
                    // we're allowed to double back.
                    factor *= 0.5;
                    remaining_stack.push(remaining);
                    remaining = 2;

                    // Restore the previous momentum for retry
                    math.copy_into(
                        &self.tmp_velocity,
                        &mut current.try_point_mut().unwrap().velocity,
                    );
                    // We don't refresh the momentum noise, so that the old
                    // noise is reused in the retry.
                }
                LeapfrogResult::Err(e) => {
                    return Err(NutsError::LogpFailure(e.into()).into());
                }
            }
        }

        if divergence_info.is_some() {
            // On divergence: stay at the pre-trajectory position, but fully
            // resample the momentum so the next draw does not reuse
            // momentum from this failed trajectory.
            let mut next_state = self.hamiltonian.copy_state(math, &self.state);
            self.hamiltonian
                .initialize_trajectory(math, &mut next_state, true, &mut self.rng)?;
            let energy_change = current.point().energy() - draw_start_energy;
            let info = MclmcInfo {
                energy_change,
                diverging: true,
                // TODO: This clone is annoing, get rid of it.
                divergence_info: divergence_info.clone(),
                num_steps: steps_taken,
                average_step_size: time / steps_taken.to_f64().unwrap(),
            };
            let sample_info = crate::nuts::SampleInfo {
                depth: steps_taken,
                divergence_info,
                reached_maxdepth: false,
            };
            self.collector.register_draw(math, &current, &sample_info);
            return Ok((next_state, info));
        }

        assert!(steps_taken >= num_base_steps);

        // Register the end state as the draw for the adaptation collector.
        let sample_info = crate::nuts::SampleInfo {
            depth: steps_taken,
            divergence_info: None,
            reached_maxdepth: false,
        };
        self.collector.register_draw(math, &current, &sample_info);

        // TODO: In the euclidian case this includes the change in kinetic
        // energy due to the momentum refresh, but it really shouldn't.
        let energy_change = current.point().energy_error();

        let info = MclmcInfo {
            energy_change,
            diverging: false,
            divergence_info: None,
            num_steps: steps_taken,
            average_step_size: time / steps_taken.to_f64().unwrap(),
        };

        Ok((current, info))
    }
}

// ── SamplerStats ──────────────────────────────────────────────────────────────

impl<M, R, A, T> SamplerStats<M> for MclmcChain<M, R, A, T>
where
    M: Math,
    R: rand::Rng,
    T: Transformation<M>,
    A: AdaptStrategy<M, Hamiltonian = TransformedHamiltonian<M, T>>,
{
    type Stats = MclmcStats<
        StatsDims,
        <TransformedHamiltonian<M, T> as SamplerStats<M>>::Stats,
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
            log_weight: info.energy_change,
            tuning: self.adapt.is_tuning(),
            hamiltonian: hamiltonian_stats,
            adapt: adapt_stats,
            point: point_stats,
            average_step_size: info.average_step_size,
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

impl<M, R, A, T> Chain<M> for MclmcChain<M, R, A, T>
where
    M: Math,
    R: rand::Rng,
    T: Transformation<M>,
    A: AdaptStrategy<M, Hamiltonian = TransformedHamiltonian<M, T>>,
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
        // Initialise momentum according to the current trajectory kind.
        self.hamiltonian
            .initialize_trajectory(math, &mut self.state, true, &mut self.rng)?;
        Ok(())
    }

    fn draw(&mut self) -> Result<(Box<[f64]>, Progress)> {
        // ── Euclidean → Microcanonical switch ─────────────────────────────────
        // At the configured draw boundary, switch the Hamiltonian to use the
        // Microcanonical leapfrog and fully resample the momentum from the
        // correct distribution (unit-sphere), rather than trying to recycle
        // the Euclidean-Gaussian momentum by projecting it.
        let resample_velocity = if self.trajectory_kind
            == MclmcTrajectoryKind::EuclideanEarlyThenMicrocanonical
            && self.draw_count == self.switch_draw
            && self.hamiltonian.kinetic_energy_kind != KineticEnergyKind::Microcanonical
        {
            self.hamiltonian
                .set_kinetic_energy_kind(KineticEnergyKind::Microcanonical);
            true
        } else {
            false
        };

        let (state, info) = self.mclmc_kernel(resample_velocity)?;

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
        // Update the stats_options of the hamiltonian. This is used to
        // store only changes in the transformation.
        self.stats_options.hamiltonian = self
            .hamiltonian
            .update_stats_options(math, self.stats_options.hamiltonian);
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

    use crate::{
        Chain, DiagMclmcSettings, MclmcSettings, math::test_logps::NormalLogp,
        math::CpuMath, sampler::Settings,
    };

    #[test]
    fn mclmc_draws_normal() {
        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.0);
        let math = CpuMath::new(func);

        let settings = DiagMclmcSettings {
            step_size: 0.5,
            momentum_decoherence_length: 3.0,
            num_tune: 200,
            num_draws: 500,
            ..MclmcSettings::default()
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

    #[test]
    fn mclmc_euclidean_trajectory() {
        use crate::mclmc::MclmcTrajectoryKind;

        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.0);
        let math = CpuMath::new(func);

        let settings = DiagMclmcSettings {
            step_size: 0.3,
            momentum_decoherence_length: 3.0,
            num_tune: 200,
            num_draws: 500,
            trajectory_kind: MclmcTrajectoryKind::Euclidean,
            ..MclmcSettings::default()
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

        let mean: f64 = last_pos.iter().sum::<f64>() / ndim as f64;
        assert!(
            (mean - 3.0).abs() < 3.0,
            "mean {mean} too far from expected 3.0"
        );
    }

    #[test]
    fn mclmc_euclidean_early_then_microcanonical() {
        use crate::mclmc::MclmcTrajectoryKind;

        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.0);
        let math = CpuMath::new(func);

        let settings = DiagMclmcSettings {
            step_size: 0.5,
            momentum_decoherence_length: 3.0,
            num_tune: 200,
            num_draws: 500,
            trajectory_kind: MclmcTrajectoryKind::EuclideanEarlyThenMicrocanonical,
            trajectory_switch_fraction: 0.3,
            ..MclmcSettings::default()
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

        let mean: f64 = last_pos.iter().sum::<f64>() / ndim as f64;
        assert!(
            (mean - 3.0).abs() < 3.0,
            "mean {mean} too far from expected 3.0"
        );
    }
}
