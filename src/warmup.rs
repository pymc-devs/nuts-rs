//! Phases of the warmup schedule, shared by the adaptation strategies and
//! the settings that change during warmup.

use crate::{
    Math,
    dynamics::{Hamiltonian, KineticEnergyKind},
    nuts::NutsOptions,
    walnuts::WalnutsOptions,
};

/// The phase of the warmup schedule that a draw belongs to.
///
/// Adaptation strategies use it to decide what to adapt, and to set the
/// settings that change between phases, like the `early_*` settings of the
/// adaptation options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupPhase {
    /// The first draws of warmup, far from the typical set. The mass matrix
    /// or transformation is updated frequently.
    Early,
    /// Joint adaptation of the step size and the mass matrix or
    /// transformation.
    Main,
    /// Final window of warmup, where only the step size is adapted.
    FinalStepSize,
    /// After warmup.
    Sampling,
}

impl WarmupPhase {
    /// The name of the phase in the sampler stats.
    pub fn as_str(&self) -> &'static str {
        match self {
            WarmupPhase::Early => "early",
            WarmupPhase::Main => "main",
            WarmupPhase::FinalStepSize => "final_step_size",
            WarmupPhase::Sampling => "sampling",
        }
    }
}

/// The settings of the dynamics that can change between warmup phases.
///
/// They are part of the adaptation options, because the adaptation sets
/// them for each phase, see [`set_phase_settings`].
pub(crate) trait DynamicsOptions {
    fn trajectory_kind(&self) -> KineticEnergyKind;
    fn early_trajectory_kind(&self) -> Option<KineticEnergyKind>;
    fn gradient_clipping(&self) -> Option<f64>;
    fn early_gradient_clipping(&self) -> Option<f64>;
    fn walnuts(&self) -> Option<WalnutsOptions>;

    /// Whether the dynamics are microcanonical in any warmup phase.
    fn uses_microcanonical(&self) -> bool {
        std::iter::once(self.trajectory_kind())
            .chain(self.early_trajectory_kind())
            .any(|kind| kind == KineticEnergyKind::Microcanonical)
    }

    fn validate_dynamics(&self) -> anyhow::Result<()> {
        if let Some(clip) = self.early_gradient_clipping()
            && (clip.is_nan() || clip <= 0.0)
        {
            anyhow::bail!("early_gradient_clipping must be positive, got {clip}");
        }
        if let Some(walnuts) = self.walnuts() {
            walnuts.validate()?;
            if self.uses_microcanonical() {
                anyhow::bail!("WALNUTS does not support microcanonical dynamics");
            }
        }
        Ok(())
    }
}

/// Set the settings that change between warmup phases to their values in
/// `phase`. Adaptation strategies call this when a new phase starts.
///
/// The trajectory kind is only set if there is an `early_trajectory_kind`,
/// because MCLMC switches its trajectory kind itself.
pub(crate) fn set_phase_settings<M: Math, H: Hamiltonian<M>>(
    phase: WarmupPhase,
    dynamics: &impl DynamicsOptions,
    options: &mut NutsOptions,
    hamiltonian: &mut H,
) {
    let early = phase == WarmupPhase::Early;
    if let Some(early_kind) = dynamics.early_trajectory_kind() {
        hamiltonian.set_kinetic_energy_kind(if early {
            early_kind
        } else {
            dynamics.trajectory_kind()
        });
    }
    hamiltonian.set_gradient_clipping(match dynamics.early_gradient_clipping() {
        Some(clip) if early => Some(clip),
        _ => dynamics.gradient_clipping(),
    });
    if let (Some(step), Some(walnuts)) = (options.walnuts.as_mut(), dynamics.walnuts()) {
        step.max_energy_error = walnuts.max_energy_error(phase);
        step.check_reversibility = walnuts.reversibility_check.enabled(phase);
    }
}

/// Draw indices at which the warmup phases start.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WarmupSchedule {
    /// The first draw after the early window.
    pub(crate) early_end: u64,
    /// The first draw of the final step size window.
    pub(crate) final_start: u64,
    /// The first draw after warmup.
    pub(crate) num_tune: u64,
}

impl WarmupSchedule {
    /// The phase of `draw`. If the early window overlaps with the final step
    /// size window, the final window takes precedence.
    pub(crate) fn phase(&self, draw: u64) -> WarmupPhase {
        if draw >= self.num_tune {
            WarmupPhase::Sampling
        } else if draw >= self.final_start {
            WarmupPhase::FinalStepSize
        } else if draw < self.early_end {
            WarmupPhase::Early
        } else {
            WarmupPhase::Main
        }
    }

    /// Whether `draw` is the first draw of its phase.
    pub(crate) fn is_phase_start(&self, draw: u64) -> bool {
        draw == 0 || self.phase(draw) != self.phase(draw - 1)
    }

    /// Whether `draw` is the last draw of its phase.
    pub(crate) fn is_phase_end(&self, draw: u64) -> bool {
        self.phase(draw) != self.phase(draw + 1)
    }

    /// Whether `draw` is the last tuning draw, whatever phase it is in.
    pub(crate) fn is_last_tuning_draw(&self, draw: u64) -> bool {
        self.is_phase_end(draw) && self.phase(draw + 1) == WarmupPhase::Sampling
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::ChaCha8Rng};

    use super::*;
    use crate::{
        Chain, DiagMclmcSettings, DiagNutsSettings, KineticEnergyKind, Settings, math::CpuMath,
        math::test_logps::NormalLogp,
    };

    fn run(
        trajectory_kind: KineticEnergyKind,
        early_trajectory_kind: Option<KineticEnergyKind>,
    ) -> Vec<Box<[f64]>> {
        let mut settings = DiagNutsSettings {
            num_tune: 100,
            ..Default::default()
        };
        settings.adapt_options.trajectory_kind = trajectory_kind;
        settings.adapt_options.early_trajectory_kind = early_trajectory_kind;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let logp = NormalLogp::new(10, 3.0);
        let mut chain = settings.new_chain(0, CpuMath::new(logp), &mut rng).unwrap();
        chain.set_position(&[0.1; 10]).unwrap();
        (0..60).map(|_| chain.draw().unwrap().0).collect()
    }

    /// `early_trajectory_kind` is used exactly in the early phase, which is
    /// the first 30 draws with the default settings.
    #[test]
    fn early_trajectory_kind() {
        let euclidean = run(KineticEnergyKind::Euclidean, None);
        let exact = run(KineticEnergyKind::ExactNormal, None);
        let switched = run(
            KineticEnergyKind::ExactNormal,
            Some(KineticEnergyKind::Euclidean),
        );
        assert_ne!(euclidean[..30], exact[..30]);
        assert_eq!(euclidean[..30], switched[..30]);
        assert_ne!(euclidean[30..], switched[30..]);

        // MCLMC switches the trajectory kind itself, and has no WALNUTS.
        assert!(DiagMclmcSettings::default().validate().is_ok());
        let mut mclmc = DiagMclmcSettings::default();
        mclmc.adapt_options.early_trajectory_kind = Some(KineticEnergyKind::Euclidean);
        assert!(mclmc.validate().is_err());
        let mut mclmc = DiagMclmcSettings::default();
        mclmc.adapt_options.trajectory_kind = KineticEnergyKind::ExactNormal;
        assert!(mclmc.validate().is_err());
        let mut mclmc = DiagMclmcSettings::default();
        mclmc.adapt_options.walnuts = Some(crate::WalnutsOptions::default());
        assert!(mclmc.validate().is_err());
    }

    /// Without a final step size window, the last tuning draw still sets the
    /// final step size, so the first draw after tuning uses it.
    #[test]
    fn final_step_size_without_final_window() {
        let mut settings = DiagNutsSettings {
            num_tune: 100,
            ..Default::default()
        };
        settings.adapt_options.step_size_window = 0.0;
        settings.adapt_options.step_size_settings.jitter = None;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let logp = NormalLogp::new(10, 3.0);
        let mut chain = settings.new_chain(0, CpuMath::new(logp), &mut rng).unwrap();
        chain.set_position(&[0.1; 10]).unwrap();
        // The step size in the progress is the one for the next draw.
        let step_sizes: Vec<_> = (0..103)
            .map(|_| chain.draw().unwrap().1.step_size)
            .collect();
        assert_ne!(step_sizes[98], step_sizes[99]);
        assert_eq!(step_sizes[99], step_sizes[100]);
        assert_eq!(step_sizes[100], step_sizes[102]);
    }

    /// The stats record the phase of each draw.
    #[test]
    fn warmup_phase_stat() {
        // 30% early window and 15% final step size window of 100 draws.
        let schedule = WarmupSchedule {
            early_end: 30,
            final_start: 85,
            num_tune: 100,
        };
        let expected: Vec<_> = (0..110).map(|d| schedule.phase(d).as_str()).collect();

        let nuts = DiagNutsSettings {
            num_tune: 100,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let logp = NormalLogp::new(10, 3.0);
        let mut chain = nuts.new_chain(0, CpuMath::new(logp), &mut rng).unwrap();
        chain.set_position(&[0.1; 10]).unwrap();
        let phases: Vec<_> = (0..110)
            .map(|_| chain.expanded_draw().unwrap().2.warmup_phase)
            .collect();
        assert_eq!(phases, expected);

        let mclmc = DiagMclmcSettings {
            num_tune: 100,
            ..Default::default()
        };
        let logp = NormalLogp::new(10, 3.0);
        let mut chain = mclmc.new_chain(0, CpuMath::new(logp), &mut rng).unwrap();
        chain.set_position(&[0.1; 10]).unwrap();
        let phases: Vec<_> = (0..110)
            .map(|_| chain.expanded_draw().unwrap().2.warmup_phase)
            .collect();
        assert_eq!(phases, expected);
    }

    #[test]
    fn phases() {
        let schedule = WarmupSchedule {
            early_end: 3,
            final_start: 8,
            num_tune: 10,
        };
        let phases: Vec<_> = (0..12).map(|draw| schedule.phase(draw)).collect();
        use WarmupPhase::*;
        assert_eq!(
            phases,
            [
                Early,
                Early,
                Early,
                Main,
                Main,
                Main,
                Main,
                Main,
                FinalStepSize,
                FinalStepSize,
                Sampling,
                Sampling
            ]
        );

        let overlapping = WarmupSchedule {
            early_end: 9,
            final_start: 8,
            num_tune: 10,
        };
        assert_eq!(overlapping.phase(7), Early);
        assert_eq!(overlapping.phase(8), FinalStepSize);
    }

    #[test]
    fn changepoints() {
        let schedule = WarmupSchedule {
            early_end: 3,
            final_start: 8,
            num_tune: 10,
        };
        let starts: Vec<_> = (0..12).filter(|&d| schedule.is_phase_start(d)).collect();
        assert_eq!(starts, [0, 3, 8, 10]);
        let ends: Vec<_> = (0..12).filter(|&d| schedule.is_phase_end(d)).collect();
        assert_eq!(ends, [2, 7, 9]);
        let last: Vec<_> = (0..12)
            .filter(|&d| schedule.is_last_tuning_draw(d))
            .collect();
        assert_eq!(last, [9]);

        // Without a final step size window, tuning ends in the main phase.
        let no_final = WarmupSchedule {
            early_end: 3,
            final_start: 10,
            num_tune: 10,
        };
        assert_eq!(no_final.phase(9), WarmupPhase::Main);
        assert!(no_final.is_last_tuning_draw(9));

        // Without tuning, there is no last tuning draw.
        let no_tune = WarmupSchedule {
            early_end: 0,
            final_start: 0,
            num_tune: 0,
        };
        assert!((0..3).all(|d| !no_tune.is_last_tuning_draw(d)));
    }
}
