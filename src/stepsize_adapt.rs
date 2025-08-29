use itertools::Either;
use nuts_derive::Storable;
use rand::Rng;
use rand_distr::Uniform;
use serde::Serialize;

use crate::{
    Math, NutsError,
    hamiltonian::{Direction, Hamiltonian, LeapfrogResult, Point},
    nuts::{Collector, NutsOptions},
    sampler_stats::SamplerStats,
    stepsize_adam::{Adam, AdamOptions},
    stepsize_dual_avg::{AcceptanceRateCollector, DualAverage, DualAverageOptions},
};
use std::fmt::Debug;

/// Method used for step size adaptation
#[derive(Debug, Clone, Copy, Serialize)]
pub enum StepSizeAdaptMethod {
    /// Use dual averaging for step size adaptation (default)
    DualAverage,
    /// Use Adam optimizer for step size adaptation
    Adam,
    Fixed(f64),
}

impl Default for StepSizeAdaptMethod {
    fn default() -> Self {
        StepSizeAdaptMethod::Adam
    }
}

/// Options for step size adaptation
#[derive(Debug, Clone, Copy, Serialize)]
pub struct StepSizeAdaptOptions {
    pub method: StepSizeAdaptMethod,
    /// Dual averaging adaptation options
    pub dual_average: DualAverageOptions,
    /// Adam optimizer adaptation options
    pub adam: AdamOptions,
}

impl Default for StepSizeAdaptOptions {
    fn default() -> Self {
        Self {
            method: StepSizeAdaptMethod::DualAverage,
            dual_average: DualAverageOptions::default(),
            adam: AdamOptions::default(),
        }
    }
}

/// Step size adaptation strategy
pub struct Strategy {
    /// The step size adaptation method being used
    adaptation: Option<Either<DualAverage, Adam>>,
    /// Settings for step size adaptation
    options: StepSizeSettings,
    /// Last mean tree accept rate
    pub last_mean_tree_accept: f64,
    /// Last symmetric mean tree accept rate
    pub last_sym_mean_tree_accept: f64,
    /// Last number of steps
    pub last_n_steps: u64,
}

impl Strategy {
    pub fn new(options: StepSizeSettings) -> Self {
        let adaptation = match options.adapt_options.method {
            StepSizeAdaptMethod::DualAverage => Some(Either::Left(DualAverage::new(
                options.adapt_options.dual_average,
                options.initial_step,
            ))),
            StepSizeAdaptMethod::Adam => Some(Either::Right(Adam::new(
                options.adapt_options.adam,
                options.initial_step,
            ))),
            StepSizeAdaptMethod::Fixed(_) => None,
        };

        Self {
            adaptation,
            options,
            last_n_steps: 0,
            last_sym_mean_tree_accept: 0.0,
            last_mean_tree_accept: 0.0,
        }
    }

    pub fn init<M: Math, R: Rng + ?Sized, P: Point<M>>(
        &mut self,
        math: &mut M,
        options: &mut NutsOptions,
        hamiltonian: &mut impl Hamiltonian<M, Point = P>,
        position: &[f64],
        rng: &mut R,
    ) -> Result<(), NutsError> {
        if let StepSizeAdaptMethod::Fixed(step_size) = self.options.adapt_options.method {
            *hamiltonian.step_size_mut() = step_size;
            return Ok(());
        };
        let mut state = hamiltonian.init_state(math, position)?;
        hamiltonian.initialize_trajectory(math, &mut state, rng)?;

        let mut collector = AcceptanceRateCollector::new();

        collector.register_init(math, &state, options);

        *hamiltonian.step_size_mut() = self.options.initial_step;

        let state_next = hamiltonian.leapfrog(math, &state, Direction::Forward, &mut collector);

        let LeapfrogResult::Ok(_) = state_next else {
            return Ok(());
        };

        let accept_stat = collector.mean.current();
        let dir = if accept_stat > self.options.target_accept {
            Direction::Forward
        } else {
            Direction::Backward
        };

        for _ in 0..100 {
            let mut collector = AcceptanceRateCollector::new();
            collector.register_init(math, &state, options);
            let state_next = hamiltonian.leapfrog(math, &state, dir, &mut collector);
            let LeapfrogResult::Ok(_) = state_next else {
                *hamiltonian.step_size_mut() = self.options.initial_step;
                return Ok(());
            };
            let accept_stat = collector.mean.current();
            match dir {
                Direction::Forward => {
                    if (accept_stat <= self.options.target_accept) | (hamiltonian.step_size() > 1e5)
                    {
                        match self.adaptation.as_mut().expect("Adaptation must be set") {
                            Either::Left(adapt) => {
                                *adapt = DualAverage::new(
                                    self.options.adapt_options.dual_average,
                                    hamiltonian.step_size(),
                                );
                            }
                            Either::Right(adapt) => {
                                *adapt = Adam::new(
                                    self.options.adapt_options.adam,
                                    hamiltonian.step_size(),
                                );
                            }
                        }
                        return Ok(());
                    }
                    *hamiltonian.step_size_mut() *= 2.;
                }
                Direction::Backward => {
                    if (accept_stat >= self.options.target_accept)
                        | (hamiltonian.step_size() < 1e-10)
                    {
                        match self.adaptation.as_mut().expect("Adaptation must be set") {
                            Either::Left(adapt) => {
                                *adapt = DualAverage::new(
                                    self.options.adapt_options.dual_average,
                                    hamiltonian.step_size(),
                                );
                            }
                            Either::Right(adapt) => {
                                *adapt = Adam::new(
                                    self.options.adapt_options.adam,
                                    hamiltonian.step_size(),
                                );
                            }
                        }
                        return Ok(());
                    }
                    *hamiltonian.step_size_mut() /= 2.;
                }
            }
        }
        // If we don't find something better, use the specified initial value
        *hamiltonian.step_size_mut() = self.options.initial_step;
        Ok(())
    }

    pub fn update(&mut self, collector: &AcceptanceRateCollector) {
        let mean_sym = collector.mean_sym.current();
        let mean = collector.mean.current();
        let n_steps = collector.mean.count();
        self.last_mean_tree_accept = mean;
        self.last_sym_mean_tree_accept = mean_sym;
        self.last_n_steps = n_steps;
    }

    pub fn update_estimator_early(&mut self) {
        match self.adaptation.as_mut() {
            None => {}
            Some(Either::Left(adapt)) => {
                adapt.advance(self.last_mean_tree_accept, self.options.target_accept);
            }
            Some(Either::Right(adapt)) => {
                adapt.advance(self.last_mean_tree_accept, self.options.target_accept);
            }
        }
    }

    pub fn update_estimator_late(&mut self) {
        match self.adaptation.as_mut() {
            None => {}
            Some(Either::Left(adapt)) => {
                adapt.advance(self.last_sym_mean_tree_accept, self.options.target_accept);
            }
            Some(Either::Right(adapt)) => {
                adapt.advance(self.last_sym_mean_tree_accept, self.options.target_accept);
            }
        }
    }

    pub fn update_stepsize<M: Math, R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        hamiltonian: &mut impl Hamiltonian<M>,
        use_best_guess: bool,
    ) {
        let step_size = match self.adaptation {
            None => {
                if let StepSizeAdaptMethod::Fixed(val) = self.options.adapt_options.method {
                    val
                } else {
                    panic!("Adaptation method must be Fixed if adaptation is None")
                }
            }
            Some(Either::Left(ref adapt)) => {
                if use_best_guess {
                    adapt.current_step_size_adapted()
                } else {
                    adapt.current_step_size()
                }
            }
            Some(Either::Right(ref adapt)) => adapt.current_step_size(),
        };

        if let Some(jitter) = self.options.jitter {
            let jitter =
                rng.sample(Uniform::new(1.0 - jitter, 1.0 + jitter).expect("Invalid jitter"));
            let jittered_step_size = step_size * jitter;
            *hamiltonian.step_size_mut() = jittered_step_size;
        } else {
            *hamiltonian.step_size_mut() = step_size;
        }
    }

    pub fn new_collector(&self) -> AcceptanceRateCollector {
        AcceptanceRateCollector::new()
    }
}

#[derive(Debug, Storable)]
pub struct Stats {
    pub step_size_bar: f64,
    pub mean_tree_accept: f64,
    pub mean_tree_accept_sym: f64,
    pub n_steps: u64,
}

impl<M: Math> SamplerStats<M> for Strategy {
    type Stats = Stats;
    type StatsOptions = ();

    fn extract_stats(&self, _math: &mut M, _opt: Self::StatsOptions) -> Self::Stats {
        Stats {
            step_size_bar: match self.adaptation {
                None => {
                    if let StepSizeAdaptMethod::Fixed(val) = self.options.adapt_options.method {
                        val
                    } else {
                        panic!("Adaptation method must be Fixed if adaptation is None")
                    }
                }
                Some(Either::Left(ref adapt)) => adapt.current_step_size_adapted(),
                Some(Either::Right(ref adapt)) => adapt.current_step_size(),
            },
            mean_tree_accept: self.last_mean_tree_accept,
            mean_tree_accept_sym: self.last_sym_mean_tree_accept,
            n_steps: self.last_n_steps,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct StepSizeSettings {
    /// Target acceptance rate
    pub target_accept: f64,
    /// Initial step size
    pub initial_step: f64,
    /// Optional jitter to add to step size (randomization)
    pub jitter: Option<f64>,
    /// Adaptation options specific to the chosen method
    pub adapt_options: StepSizeAdaptOptions,
}

impl Default for StepSizeSettings {
    fn default() -> Self {
        Self {
            target_accept: 0.8,
            initial_step: 0.1,
            jitter: Some(0.1),
            adapt_options: StepSizeAdaptOptions::default(),
        }
    }
}
