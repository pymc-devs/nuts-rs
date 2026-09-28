//! WALNUTS: within-orbit adaptive step sizes for the leaves of the NUTS tree.
//!
//! Each leaf of the NUTS tree ("macro step") is integrated with
//! `min_micro_steps * 2^k` leapfrog steps ("micro steps") of size
//! `step_size / (min_micro_steps * 2^k)`, where `k` is the smallest value for
//! which the Hamiltonian stays within `max_error`. To keep the transition
//! reversible, a macro step that needed `k > 0` is only accepted if none of
//! the coarser step counts would have been chosen when integrating backwards
//! from its end point. Otherwise the trajectory is stopped, just like after a
//! divergence.
//!
//! Step size adaptation uses the energy error of the first (coarsest) attempt
//! of each macro step, because the accepted macro steps are within tolerance
//! by construction.
//!
//! See Bou-Rabee, Carpenter, Kleppe, Liu: "The within-orbit adaptive leapfrog
//! no-U-turn sampler" and the reference implementation in walnutpie.

use serde::{Deserialize, Serialize};

use crate::{
    Math,
    dynamics::{Direction, DivergenceInfo, Hamiltonian, LeapfrogResult, Point, State},
    nuts::{BaseStepOutcome, Collector, StepInfo},
};

/// How the energy error of a macro step is compared against `max_error`.
///
/// Both criteria are symmetric under time reversal, so both give a valid
/// sampler.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalnutsEnergyCriterion {
    /// `max H - min H` over all micro steps of the macro step, including the
    /// start. Stricter, and lets us stop failing attempts early.
    #[default]
    MaxMin,
    /// `|H(end) - H(start)|` of the macro step, as in walnutpie.
    Endpoint,
}

/// Settings for WALNUTS.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WalnutsOptions {
    /// Maximum number of times the micro step size is halved after the first
    /// attempt. If the macro step is still not within tolerance after that,
    /// it is treated as a divergence.
    pub max_step_halvings: u64,
    /// Maximum error of the Hamiltonian within a macro step.
    pub max_error: f64,
    /// Number of micro steps of the first attempt of each macro step.
    pub min_micro_steps: u64,
    /// How the energy error of a macro step is measured.
    pub energy_criterion: WalnutsEnergyCriterion,
    /// If set, `min_micro_steps` is adapted during tuning as in walnutpie,
    /// see `MinMicroStepsAdaptation`. `min_micro_steps` is then the lower
    /// bound. walnutpie uses 15.
    #[serde(default)]
    pub target_macro_steps: Option<f64>,
    /// Upper bound for the adapted `min_micro_steps`.
    #[serde(default = "default_max_min_micro_steps")]
    pub max_min_micro_steps: u64,
}

fn default_max_min_micro_steps() -> u64 {
    1024
}

impl Default for WalnutsOptions {
    fn default() -> Self {
        WalnutsOptions {
            max_step_halvings: 5,
            max_error: 0.5,
            min_micro_steps: 1,
            energy_criterion: WalnutsEnergyCriterion::MaxMin,
            target_macro_steps: None,
            max_min_micro_steps: default_max_min_micro_steps(),
        }
    }
}

impl WalnutsOptions {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if !(self.max_error.is_finite() && self.max_error > 0.0) {
            anyhow::bail!(
                "walnuts max_error must be positive and finite, got {}",
                self.max_error
            );
        }
        if self.min_micro_steps == 0 {
            anyhow::bail!("walnuts min_micro_steps must be positive");
        }
        // With adaptation, `min_micro_steps` can go up to `max_min_micro_steps`.
        let max_micro_steps = match self.target_macro_steps {
            None => self.min_micro_steps,
            Some(target) => {
                if !(target.is_finite() && target > 0.0) {
                    anyhow::bail!(
                        "walnuts target_macro_steps must be positive and finite, got {target}"
                    );
                }
                if self.max_min_micro_steps < self.min_micro_steps {
                    anyhow::bail!(
                        "walnuts max_min_micro_steps ({}) must be at least min_micro_steps ({})",
                        self.max_min_micro_steps,
                        self.min_micro_steps
                    );
                }
                self.max_min_micro_steps
            }
        };
        if max_micro_steps
            .checked_shl(self.max_step_halvings as u32)
            .is_none_or(|n| n >> self.max_step_halvings != max_micro_steps)
        {
            anyhow::bail!(
                "walnuts min_micro_steps * 2^max_step_halvings is too large ({} * 2^{})",
                max_micro_steps,
                self.max_step_halvings
            );
        }
        Ok(())
    }
}

pub(crate) enum MacroStep<M: Math, P: Point<M>> {
    Ok(State<M, P>),
    Divergence(DivergenceInfo),
    Irreversible,
}

struct NoopCollector;

impl<M: Math, P: Point<M>> Collector<M, P> for NoopCollector {}

enum MicroResult<M: Math, P: Point<M>> {
    /// The micro steps stayed within tolerance.
    Accepted { end: State<M, P>, energy_error: f64 },
    /// The micro steps left the tolerance. `last` is the last state that was
    /// computed.
    Rejected {
        last: State<M, P>,
        energy_error: f64,
        divergence: Option<DivergenceInfo>,
    },
}

/// Take `num_steps` leapfrog steps with step size factor `1 / num_steps`
/// and check the energy criterion.
///
/// If `stop_early` is set, integration stops as soon as the outcome is known
/// to be a rejection. The returned energy error is then not the one of the
/// full macro step.
#[allow(clippy::too_many_arguments)]
fn micro_integrate<M: Math, H: Hamiltonian<M>>(
    math: &mut M,
    hamiltonian: &mut H,
    start: &State<M, H::Point>,
    dir: Direction,
    num_steps: u64,
    options: &WalnutsOptions,
    stop_early: bool,
    num_gradients: &mut u64,
) -> Result<MicroResult<M, H::Point>, M::LogpErr> {
    let factor = 1.0 / (num_steps as f64);
    let start_energy = start.energy();
    let mut min_energy = start_energy;
    let mut max_energy = start_energy;
    let mut exceeded = false;
    let mut current = start.clone();

    for _ in 0..num_steps {
        *num_gradients += 1;
        // The walnuts criterion replaces the usual divergence check, so the
        // leapfrog only reports non-finite energies and logp errors.
        current = match hamiltonian.leapfrog(
            math,
            &current,
            dir,
            factor,
            start_energy,
            f64::INFINITY,
            &mut NoopCollector,
        ) {
            LeapfrogResult::Ok(state) => state,
            LeapfrogResult::Divergence(divergence) => {
                return Ok(MicroResult::Rejected {
                    last: current,
                    energy_error: f64::NAN,
                    divergence: Some(divergence),
                });
            }
            LeapfrogResult::Err(err) => return Err(err),
        };
        let energy = current.energy();
        min_energy = min_energy.min(energy);
        max_energy = max_energy.max(energy);
        if options.energy_criterion == WalnutsEnergyCriterion::MaxMin
            && max_energy - min_energy > options.max_error
        {
            exceeded = true;
            if stop_early {
                break;
            }
        }
    }

    let energy_error = current.energy() - start_energy;
    if exceeded || energy_error.abs() > options.max_error {
        Ok(MicroResult::Rejected {
            last: current,
            energy_error,
            divergence: None,
        })
    } else {
        Ok(MicroResult::Accepted {
            end: current,
            energy_error,
        })
    }
}

/// Check that integrating backwards from `end` would not have picked a
/// coarser step count than `num_steps`.
fn is_reversible<M: Math, H: Hamiltonian<M>>(
    math: &mut M,
    hamiltonian: &mut H,
    end: &State<M, H::Point>,
    dir: Direction,
    mut num_steps: u64,
    options: &WalnutsOptions,
    num_gradients: &mut u64,
) -> Result<bool, M::LogpErr> {
    let back = dir.reverse();
    while num_steps >= 2 * options.min_micro_steps {
        num_steps /= 2;
        match micro_integrate(
            math,
            hamiltonian,
            end,
            back,
            num_steps,
            options,
            true,
            num_gradients,
        )? {
            MicroResult::Accepted { .. } => return Ok(false),
            MicroResult::Rejected { .. } => {}
        }
    }
    Ok(true)
}

/// Compute one leaf of the NUTS tree.
///
/// The collector sees one `register_leapfrog` call per macro step, which
/// includes the outcome of the first attempt at the base step size and the
/// number of gradient evaluations. Individual micro steps are not reported.
///
/// An irreversible macro step is registered like an accepted one, because
/// its end point was computed within tolerance, but it ends the trajectory.
pub(crate) fn macro_step<M, H, C>(
    math: &mut M,
    hamiltonian: &mut H,
    start: &State<M, H::Point>,
    dir: Direction,
    options: &WalnutsOptions,
    collector: &mut C,
) -> Result<MacroStep<M, H::Point>, M::LogpErr>
where
    M: Math,
    H: Hamiltonian<M>,
    C: Collector<M, H::Point>,
{
    let mut num_gradients = 0;
    let mut base_step = BaseStepOutcome::Diverged;
    let mut rejected = None;
    for halvings in 0..=options.max_step_halvings {
        let num_steps = options.min_micro_steps << halvings;
        let first = halvings == 0;
        // The first attempt is always integrated fully, so that step size
        // adaptation sees the energy error of the whole macro step.
        let result = micro_integrate(
            math,
            hamiltonian,
            start,
            dir,
            num_steps,
            options,
            !first,
            &mut num_gradients,
        )?;

        match result {
            MicroResult::Accepted {
                mut end,
                energy_error,
            } => {
                let reversible = if first {
                    base_step = BaseStepOutcome::EnergyError(energy_error);
                    true
                } else {
                    is_reversible(
                        math,
                        hamiltonian,
                        &end,
                        dir,
                        num_steps,
                        options,
                        &mut num_gradients,
                    )?
                };

                let sign = match dir {
                    Direction::Forward => 1,
                    Direction::Backward => -1,
                };
                end.try_point_mut()
                    .expect("Macro step end state should not have other references")
                    .set_index_in_trajectory(start.index_in_trajectory() + sign);

                let step = StepInfo {
                    num_gradients,
                    base_step,
                };
                collector.register_leapfrog(math, start, &end, None, &step);
                if reversible {
                    return Ok(MacroStep::Ok(end));
                } else {
                    return Ok(MacroStep::Irreversible);
                }
            }
            MicroResult::Rejected {
                last,
                energy_error,
                divergence,
            } => {
                if first && divergence.is_none() {
                    base_step = BaseStepOutcome::EnergyError(energy_error);
                }
                rejected = Some((last, energy_error, divergence));
            }
        }
    }

    let (last, energy_error, divergence) = rejected.expect("At least one attempt was made");
    let divergence = divergence.unwrap_or_else(|| DivergenceInfo {
        start_momentum: None,
        start_location: Some(math.box_array(start.point().position())),
        start_location_transformed: None,
        start_gradient: Some(math.box_array(start.point().gradient())),
        end_location: Some(math.box_array(last.point().position())),
        end_location_transformed: None,
        energy_error: Some(energy_error),
        end_idx_in_trajectory: None,
        start_idx_in_trajectory: Some(start.index_in_trajectory()),
        logp_function_error: None,
    });
    let step = StepInfo {
        num_gradients,
        base_step,
    };
    collector.register_leapfrog(math, start, &last, Some(&divergence), &step);
    Ok(MacroStep::Divergence(divergence))
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::ChaCha8Rng};

    use super::*;
    use crate::{
        Chain, DiagNutsSettings, EuclideanAdaptOptions, KineticEnergyKind, SamplerStats, Settings,
        StepSizeAdaptMethod, StepSizeAdaptOptions, StepSizeSettings, math::CpuMath,
        math::test_logps::NormalLogp,
    };

    fn fixed_step_settings(
        step_size: f64,
        kind: KineticEnergyKind,
        walnuts: Option<WalnutsOptions>,
    ) -> DiagNutsSettings {
        DiagNutsSettings {
            num_tune: 0,
            trajectory_kind: kind,
            walnuts,
            adapt_options: EuclideanAdaptOptions {
                step_size_settings: StepSizeSettings {
                    jitter: None,
                    adapt_options: StepSizeAdaptOptions {
                        method: StepSizeAdaptMethod::Fixed(step_size),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Run a chain and return the draws, the total number of gradient
    /// evaluations and leapfrog steps, and the number of irreversible stops.
    fn run(
        settings: DiagNutsSettings,
        logp: NormalLogp,
        num_draws: usize,
        seed: u64,
    ) -> (Vec<Box<[f64]>>, u64, u64, u64) {
        let dim = logp.dim;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut chain = settings.new_chain(0, CpuMath::new(logp), &mut rng).unwrap();
        chain.set_position(&vec![0.1; dim]).unwrap();
        let options = settings.stats_options();

        let mut draws = vec![];
        let mut num_gradients = 0;
        let mut num_steps = 0;
        let mut num_irreversible = 0;
        for _ in 0..num_draws {
            let (draw, _) = chain.draw().unwrap();
            let stats = chain.extract_stats(&mut chain.math().clone(), options);
            num_gradients += stats.adapt.step_size.n_gradients;
            num_steps += stats.adapt.step_size.n_steps;
            if stats.irreversible == Some(true) {
                num_irreversible += 1;
            }
            assert_eq!(stats.irreversible.is_some(), settings.walnuts.is_some());
            assert_eq!(
                stats.min_micro_steps,
                settings.walnuts.map(|w| w.min_micro_steps)
            );
            draws.push(draw);
        }
        (draws, num_gradients, num_steps, num_irreversible)
    }

    #[test]
    fn validate_options() {
        assert!(WalnutsOptions::default().validate().is_ok());
        let bad = WalnutsOptions {
            min_micro_steps: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        let bad = WalnutsOptions {
            max_error: f64::NAN,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        let bad = WalnutsOptions {
            min_micro_steps: 3,
            max_step_halvings: 63,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        for target in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let bad = WalnutsOptions {
                target_macro_steps: Some(target),
                ..Default::default()
            };
            assert!(bad.validate().is_err());
        }
        let bad = WalnutsOptions {
            target_macro_steps: Some(3.0),
            min_micro_steps: 4,
            max_min_micro_steps: 2,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        // Only checked with adaptation.
        assert!(
            WalnutsOptions {
                target_macro_steps: None,
                ..bad
            }
            .validate()
            .is_ok()
        );
        let bad = WalnutsOptions {
            target_macro_steps: Some(3.0),
            max_min_micro_steps: u64::MAX,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    struct Adapted {
        min_micro_steps: u64,
        macro_steps_mean: f64,
        /// Mean of `2^depth` after tuning.
        sampling_macro_steps: f64,
    }

    /// Run tuning plus `num_draws` draws with `min_micro_steps` adaptation.
    fn run_adapted(settings: DiagNutsSettings, num_draws: usize) -> Adapted {
        let walnuts = settings.walnuts.unwrap();
        let target = walnuts.target_macro_steps.unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut chain = settings
            .new_chain(0, CpuMath::new(NormalLogp::new(100, 3.0)), &mut rng)
            .unwrap();
        chain.set_position(&[0.1; 100]).unwrap();
        let options = settings.stats_options();
        let mut adapted: Option<Adapted> = None;
        let mut sampling_macro_steps = 0.0;
        for i in 0..(settings.num_tune as usize + num_draws) {
            chain.draw().unwrap();
            let stats = chain.extract_stats(&mut chain.math().clone(), options);
            let min_micro_steps = stats.min_micro_steps.unwrap();
            let macro_steps_mean = stats
                .adapt
                .step_size
                .min_micro_steps
                .macro_steps_mean
                .unwrap();
            let unrounded = stats
                .adapt
                .step_size
                .min_micro_steps
                .min_micro_steps_unrounded
                .unwrap();
            assert_eq!(unrounded, macro_steps_mean / target);
            assert_eq!(
                min_micro_steps,
                (unrounded.round() as u64)
                    .clamp(walnuts.min_micro_steps, walnuts.max_min_micro_steps)
            );
            if i < settings.num_tune as usize {
                continue;
            }
            // Everything is frozen after tuning.
            let current = adapted.get_or_insert(Adapted {
                min_micro_steps,
                macro_steps_mean,
                sampling_macro_steps: 0.0,
            });
            assert_eq!(current.min_micro_steps, min_micro_steps);
            assert_eq!(current.macro_steps_mean, macro_steps_mean);
            sampling_macro_steps += 2f64.powi(stats.depth as i32);
        }
        let mut adapted = adapted.unwrap();
        adapted.sampling_macro_steps = sampling_macro_steps / num_draws as f64;
        adapted
    }

    #[test]
    fn adapt_min_micro_steps() {
        let walnuts = WalnutsOptions {
            target_macro_steps: Some(1.0),
            ..Default::default()
        };
        let settings = DiagNutsSettings {
            num_tune: 1000,
            walnuts: Some(walnuts),
            ..Default::default()
        };
        let adapted = run_adapted(settings, 500);
        assert!(adapted.min_micro_steps > 1);
        // As in walnutpie, the trajectories end up longer than the target.
        assert!(adapted.sampling_macro_steps > 1.5);

        // The configured value is a lower bound.
        let settings = DiagNutsSettings {
            walnuts: Some(WalnutsOptions {
                target_macro_steps: Some(1000.0),
                min_micro_steps: 2,
                ..walnuts
            }),
            ..settings
        };
        assert_eq!(run_adapted(settings, 10).min_micro_steps, 2);

        // And `max_min_micro_steps` the upper bound.
        let settings = DiagNutsSettings {
            walnuts: Some(WalnutsOptions {
                target_macro_steps: Some(0.1),
                max_min_micro_steps: 4,
                ..walnuts
            }),
            ..settings
        };
        assert_eq!(run_adapted(settings, 10).min_micro_steps, 4);

        // Also adapted with a fixed step size.
        let mut settings = fixed_step_settings(0.5, KineticEnergyKind::Euclidean, Some(walnuts));
        settings.num_tune = 100;
        assert!(run_adapted(settings, 10).min_micro_steps > 1);
    }

    #[test]
    fn microcanonical_not_supported() {
        let settings = fixed_step_settings(
            0.5,
            KineticEnergyKind::Microcanonical,
            Some(WalnutsOptions::default()),
        );
        assert!(settings.validate().is_err());
    }

    /// If no step is ever split, WALNUTS takes exactly the same steps and
    /// uses the same random numbers as NUTS.
    #[test]
    fn same_as_nuts_without_splitting() {
        for kind in [KineticEnergyKind::Euclidean, KineticEnergyKind::ExactNormal] {
            let walnuts = WalnutsOptions {
                max_error: 1e10,
                ..Default::default()
            };
            let logp = NormalLogp::new(10, 3.0);
            let (nuts_draws, nuts_grads, _, _) =
                run(fixed_step_settings(0.5, kind, None), logp.clone(), 100, 42);
            let (walnuts_draws, walnuts_grads, walnuts_steps, irreversible) =
                run(fixed_step_settings(0.5, kind, Some(walnuts)), logp, 100, 42);
            assert_eq!(nuts_draws, walnuts_draws);
            assert_eq!(nuts_grads, walnuts_grads);
            assert_eq!(walnuts_grads, walnuts_steps);
            assert_eq!(irreversible, 0);
        }
    }

    /// With a step size that is too large for plain leapfrog, WALNUTS has to
    /// split steps, and the draws should still have the right moments.
    #[test]
    fn moments_with_splitting() {
        let dim = 10;
        let mu = 3.0;
        let num_draws = 4000;
        for kind in [KineticEnergyKind::Euclidean, KineticEnergyKind::ExactNormal] {
            for energy_criterion in [
                WalnutsEnergyCriterion::MaxMin,
                WalnutsEnergyCriterion::Endpoint,
            ] {
                let walnuts = WalnutsOptions {
                    energy_criterion,
                    ..Default::default()
                };
                let settings = fixed_step_settings(1.5, kind, Some(walnuts));
                let (draws, num_gradients, num_steps, _) =
                    run(settings, NormalLogp::new(dim, mu), num_draws, 7);
                assert!(
                    num_gradients > num_steps,
                    "{kind:?} {energy_criterion:?}: expected split steps, \
                     got {num_gradients} gradients for {num_steps} steps"
                );

                for i in 0..dim {
                    let mean = draws.iter().map(|d| d[i]).sum::<f64>() / num_draws as f64;
                    let var =
                        draws.iter().map(|d| (d[i] - mean).powi(2)).sum::<f64>() / num_draws as f64;
                    assert!(
                        (mean - mu).abs() < 0.1,
                        "{kind:?} {energy_criterion:?}: mean {mean} in dim {i}"
                    );
                    assert!(
                        (var - 1.0).abs() < 0.15,
                        "{kind:?} {energy_criterion:?}: variance {var} in dim {i}"
                    );
                }
            }
        }
    }

    /// Dual averaging uses the energy error at the base step size, so the
    /// step size must not run away even though split steps always succeed.
    #[test]
    fn step_size_adaptation_stays_bounded() {
        let settings = DiagNutsSettings {
            num_tune: 500,
            walnuts: Some(WalnutsOptions::default()),
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut chain = settings
            .new_chain(0, CpuMath::new(NormalLogp::new(10, 3.0)), &mut rng)
            .unwrap();
        chain.set_position(&[0.1; 10]).unwrap();
        let mut step_size = 0.0;
        for _ in 0..600 {
            let (_, progress) = chain.draw().unwrap();
            step_size = progress.step_size;
        }
        // Plain leapfrog on a standard normal is unstable above 2.
        assert!(step_size > 0.1 && step_size < 2.0, "step size {step_size}");
    }
}
