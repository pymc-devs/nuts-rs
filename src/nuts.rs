//! Implement the recursive doubling tree expansion that is the heart of the NUTS algorithm.

use rand::RngExt;
use rand_distr::num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use std::{fmt::Debug, marker::PhantomData};

use crate::dynamics::{Direction, DivergenceInfo, Hamiltonian, LeapfrogResult, Point, State};
use crate::math::{Math, logaddexp};

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum NutsError {
    #[error("Logp function returned error: {0:?}")]
    LogpFailure(Box<dyn std::error::Error + Send + Sync>),

    #[error("Could not serialize sample stats")]
    SerializeFailure(),

    #[error("Could not initialize state because of bad initial gradient: {0:?}")]
    BadInitGrad(Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, NutsError>;

/// Callbacks for various events during a Nuts sampling step.
///
/// Collectors can compute statistics like the mean acceptance rate
/// or collect data for mass matrix adaptation.
pub trait Collector<M: Math, P: Point<M>> {
    fn register_leapfrog(
        &mut self,
        _math: &mut M,
        _start: &State<M, P>,
        _end: &State<M, P>,
        _divergence_info: Option<&DivergenceInfo>,
        _num_substeps: u64,
    ) {
    }
    fn register_draw(&mut self, _math: &mut M, _state: &State<M, P>, _info: &SampleInfo) {}
    fn register_init(&mut self, _math: &mut M, _state: &State<M, P>, _options: &NutsOptions) {}
}

/// Information about a draw, exported as part of the sampler stats
#[derive(Debug)]
pub struct SampleInfo {
    /// The depth of the trajectory that this point was sampled from
    pub depth: u64,

    /// More detailed information about a divergence that might have
    /// occured in the trajectory.
    pub divergence_info: Option<DivergenceInfo>,

    /// Whether the trajectory was terminated because it reached
    /// the maximum tree depth.
    pub reached_maxdepth: bool,
}

/// A part of the trajectory tree during NUTS sampling.
///
/// Corresponds to SpanW in walnuts C++ code
struct NutsTree<M: Math, H: Hamiltonian<M>, C: Collector<M, H::Point>> {
    /// The left position of the tree.
    ///
    /// The left side always has the smaller index_in_trajectory.
    /// Leapfrogs in backward direction will replace the left.
    ///
    /// theta_bk_, rho_bk_, grad_theta_bk_, logp_bk_ in C++ code
    left: State<M, H::Point>,

    /// The right position of the tree.
    ///
    /// theta_fw_, rho_fw_, grad_theta_fw_, logp_fw_ in C++ code
    right: State<M, H::Point>,

    /// A draw from the trajectory between left and right using
    /// multinomial sampling.
    ///
    /// theta_select_ in C++ code
    draw: State<M, H::Point>,

    /// Constant for acceptance probability
    ///
    /// logp_ in C++ code
    log_size: f64,

    /// The depth of the tree
    depth: u64,

    /// A tree is the main tree if it contains the initial point
    /// of the trajectory.
    ///
    /// This is used to determine whether to use Metropolis
    /// accptance or Barker
    is_main: bool,
    _phantom2: PhantomData<C>,
}

enum ExtendResult<M: Math, H: Hamiltonian<M>, C: Collector<M, H::Point>> {
    /// The tree extension succeeded properly, and the termination
    /// criterion was not reached.
    Ok(NutsTree<M, H, C>),
    /// An unrecoverable error happend during a leapfrog step
    Err(NutsError),
    /// Tree extension succeeded and the termination criterion
    /// was reached.
    Turning(NutsTree<M, H, C>),
    /// A divergence happend during tree extension.
    Diverging(NutsTree<M, H, C>, DivergenceInfo),
}

impl<M: Math, H: Hamiltonian<M>, C: Collector<M, H::Point>> NutsTree<M, H, C> {
    fn new(state: State<M, H::Point>) -> NutsTree<M, H, C> {
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: 0.,
            is_main: true,
            _phantom2: PhantomData,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn extend<R>(
        mut self,
        math: &mut M,
        rng: &mut R,
        hamiltonian: &mut H,
        direction: Direction,
        collector: &mut C,
        options: &NutsOptions,
        early: bool,
    ) -> ExtendResult<M, H, C>
    where
        H: Hamiltonian<M>,
        R: rand::Rng + ?Sized,
    {
        let mut other =
            match self.single_step(math, hamiltonian, direction, options, collector, early) {
                Ok(Ok(tree)) => tree,
                Ok(Err(info)) => return ExtendResult::Diverging(self, info),
                Err(err) => return ExtendResult::Err(err),
            };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(math, rng, hamiltonian, direction, collector, options, early)
            {
                Ok(tree) => tree,
                Turning(_) => {
                    return Turning(self);
                }
                Diverging(_, info) => {
                    return Diverging(self, info);
                }
                Err(error) => {
                    return Err(error);
                }
            };
        }

        let (first, last) = match direction {
            Direction::Forward => (&self.left, &other.right),
            Direction::Backward => (&other.left, &self.right),
        };

        let turning = if options.check_turning {
            let mut turning = hamiltonian.is_turning(math, first, last);
            if self.depth > 0 {
                if !turning {
                    turning = hamiltonian.is_turning(math, &self.right, &other.right);
                }
                if !turning {
                    turning = hamiltonian.is_turning(math, &self.left, &other.left);
                }
            }
            turning
        } else {
            false
        };

        self.merge_into(math, other, rng, direction);

        if turning {
            ExtendResult::Turning(self)
        } else {
            ExtendResult::Ok(self)
        }
    }

    // `combine` in C++ code
    fn merge_into<R: rand::Rng + ?Sized>(
        &mut self,
        _math: &mut M,
        other: NutsTree<M, H, C>,
        rng: &mut R,
        direction: Direction,
    ) {
        assert!(self.depth == other.depth);
        assert!(self.left.index_in_trajectory() <= self.right.index_in_trajectory());
        match direction {
            Direction::Forward => {
                self.right = other.right;
            }
            Direction::Backward => {
                self.left = other.left;
            }
        }
        let log_size = logaddexp(self.log_size, other.log_size);

        let self_log_size = if self.is_main {
            assert!(self.left.index_in_trajectory() <= 0);
            assert!(self.right.index_in_trajectory() >= 0);
            self.log_size
        } else {
            log_size
        };

        if (other.log_size >= self_log_size)
            || (rng.random_bool((other.log_size - self_log_size).exp()))
        {
            self.draw = other.draw;
        }

        self.depth += 1;
        self.log_size = log_size;
    }

    // Corresponds to `build_leaf` in C++ code
    fn single_step(
        &self,
        math: &mut M,
        hamiltonian: &mut H,
        direction: Direction,
        options: &NutsOptions,
        collector: &mut C,
        early: bool,
    ) -> Result<std::result::Result<NutsTree<M, H, C>, DivergenceInfo>> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let end = match hamiltonian.leapfrog(
            math,
            start,
            direction,
            1.0,
            start.point().initial_energy(),
            options.max_energy_error,
            collector,
        ) {
            LeapfrogResult::Divergence(info) => return Ok(Err(info)),
            LeapfrogResult::Err(err) => return Err(NutsError::LogpFailure(err.into())),
            LeapfrogResult::Ok(end) => end,
        };

        let (log_size, end) = match options.walnuts_options {
            Some(ref options) => {
                // Walnuts implementation
                // TODO: Shouldn't all be in this one big function...
                let mut step_size_factor = 1.0;
                let mut num_steps = 1;
                let mut current = start.clone();

                let mut success = false;

                'step_size_search: for _ in 0..options.max_step_size_halvings {
                    current = start.clone();
                    let mut min_energy = current.energy();
                    let mut max_energy = min_energy;

                    for _ in 0..num_steps {
                        current = match hamiltonian.leapfrog(
                            math,
                            &current,
                            direction,
                            start.point().initial_energy(),
                            options.max_energy_error,
                            step_size_factor,
                            collector,
                        ) {
                            LeapfrogResult::Ok(state) => state,
                            LeapfrogResult::Divergence(_) => {
                                num_steps *= 2;
                                step_size_factor *= 0.5;
                                continue 'step_size_search;
                            }
                            LeapfrogResult::Err(err) => {
                                return Err(NutsError::LogpFailure(err.into()));
                            }
                        };

                        // Update min/max energies
                        let current_energy = current.energy();
                        min_energy = min_energy.min(current_energy);
                        max_energy = max_energy.max(current_energy);
                    }

                    if max_energy - min_energy > options.max_energy_error {
                        num_steps *= 2;
                        step_size_factor *= 0.5;
                        continue 'step_size_search;
                    }

                    success = true;
                    break 'step_size_search;
                }
                let mut last_divergence = None;

                for _ in 0..options.max_step_size_halvings {
                    current = match hamiltonian.split_leapfrog(
                        math,
                        start,
                        direction,
                        num_steps,
                        collector,
                        options.max_energy_error,
                    ) {
                        LeapfrogResult::Ok(state) => {
                            last_divergence = None;
                            state
                        }
                        LeapfrogResult::Err(err) => return Err(NutsError::LogpFailure(err.into())),
                        LeapfrogResult::Divergence(info) => {
                            num_steps *= 2;
                            last_divergence = Some(info);
                            continue;
                        }
                    };
                    break;
                }

                if !success {
                    // TODO: More info
                    return Ok(Err(DivergenceInfo::new_non_reversible()));
                }
                if let Some(info) = last_divergence {
                    let info = DivergenceInfo::new_max_step_size_halvings(math, num_steps, info);
                    return Ok(Err(info));
                }

                let back = direction.reverse();
                let mut current_backward;

                let mut reversible = true;

                'rev_step_size: while num_steps >= 2 {
                    num_steps /= 2;
                    step_size_factor *= 0.5;

                    // TODO: Can we share code for the micro steps in the two directions?
                    current_backward = current.clone();

                    let mut min_energy = current_backward.energy();
                    let mut max_energy = min_energy;

                    for _ in 0..num_steps {
                        current_backward = match hamiltonian.leapfrog(
                            math,
                            &current_backward,
                            back,
                            step_size_factor,
                            start.point().initial_energy(),
                            options.max_energy_error,
                            collector,
                        ) {
                            LeapfrogResult::Ok(state) => state,
                            LeapfrogResult::Divergence(_) => {
                                // We also reject in the backward direction, all is good so far...
                                continue 'rev_step_size;
                            }
                            LeapfrogResult::Err(err) => {
                                return Err(NutsError::LogpFailure(err.into()));
                            }
                        };

                        // Update min/max energies
                        let current_energy = current_backward.energy();
                        min_energy = min_energy.min(current_energy);
                        max_energy = max_energy.max(current_energy);
                        if max_energy - min_energy > options.max_energy_error {
                            // We reject also in the backward direction, all good so far...
                            continue 'rev_step_size;
                        }
                    }

                    match hamiltonian.split_leapfrog(
                        math,
                        &current,
                        back,
                        num_steps,
                        collector,
                        options.max_energy_error,
                    ) {
                        LeapfrogResult::Ok(_) => (),
                        LeapfrogResult::Divergence(_) => {
                            // We also reject in the backward direction, all is good so far...
                            continue;
                        }
                        LeapfrogResult::Err(err) => {
                            return Err(NutsError::LogpFailure(err.into()));
                        }
                    };

                    // We did not reject in the backward direction, so we are not reversible
                    reversible = false;
                    break;
                }

                if reversible || early {
                    let log_size = -current.point().energy_error();
                    (log_size, current)
                } else {
                    // TODO: More info
                    return Ok(Err(DivergenceInfo::new_non_reversible()));
                }
            }
            None => {
                // Classical NUTS
                //
                let end = match hamiltonian.leapfrog(
                    math,
                    start,
                    direction,
                    1.0,
                    start.point().initial_energy(),
                    options.max_energy_error,
                    collector,
                ) {
                    LeapfrogResult::Divergence(info) => return Ok(Err(info)),
                    LeapfrogResult::Err(err) => return Err(NutsError::LogpFailure(err.into())),
                    LeapfrogResult::Ok(end) => end,
                };

                let log_size = -end.point().energy_error();

                (log_size, end)
            }
        };

        Ok(Ok(NutsTree {
            right: end.clone(),
            left: end.clone(),
            draw: end,
            depth: 0,
            log_size,
            is_main: false,
            _phantom2: PhantomData,
        }))
    }

    fn info(&self, maxdepth: bool, divergence_info: Option<DivergenceInfo>) -> SampleInfo {
        SampleInfo {
            depth: self.depth,
            divergence_info,
            reached_maxdepth: maxdepth,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WalnutsOptions {
    pub max_step_size_halvings: u64,
    pub max_energy_error: f64,
}

impl Default for WalnutsOptions {
    fn default() -> Self {
        WalnutsOptions {
            max_step_size_halvings: 10,
            max_energy_error: 5.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NutsOptions {
    pub maxdepth: u64,
    pub mindepth: u64,
    pub check_turning: bool,
    pub store_divergences: bool,
    pub target_integration_time: Option<f64>,
    pub extra_doublings: u64,
    pub max_energy_error: f64,
    pub walnuts_options: Option<WalnutsOptions>,
}

impl Default for NutsOptions {
    fn default() -> Self {
        NutsOptions {
            maxdepth: 10,
            mindepth: 0,
            check_turning: true,
            store_divergences: false,
            target_integration_time: None,
            extra_doublings: 0,
            max_energy_error: 1000.0,
            walnuts_options: None,
        }
    }
}

pub(crate) fn draw<M, H, R, C>(
    math: &mut M,
    init: &mut State<M, H::Point>,
    rng: &mut R,
    hamiltonian: &mut H,
    options: &NutsOptions,
    collector: &mut C,
    early: bool,
) -> Result<(State<M, H::Point>, SampleInfo)>
where
    M: Math,
    H: Hamiltonian<M>,
    R: rand::Rng + ?Sized,
    C: Collector<M, H::Point>,
{
    hamiltonian.initialize_trajectory(math, init, true, rng)?;
    collector.register_init(math, init, options);

    let mut tree = NutsTree::new(init.clone());

    let (mindepth, maxdepth) = if let Some(target_time) = options.target_integration_time {
        let step_size = hamiltonian.step_size();
        let max_steps = (target_time / step_size).ceil() as u64;
        let mindepth = (max_steps as f64)
            .log2()
            .floor()
            .to_u64()
            .unwrap()
            .max(options.mindepth);
        let maxdepth = (max_steps as f64)
            .log2()
            .ceil()
            .to_u64()
            .unwrap()
            .max(mindepth)
            .min(options.maxdepth);

        (mindepth, maxdepth)
    } else {
        (options.mindepth, options.maxdepth)
    };

    if math.dim() == 0 {
        let info = tree.info(false, None);
        collector.register_draw(math, init, &info);
        return Ok((init.clone(), info));
    }

    let options_no_check = NutsOptions {
        check_turning: false,
        ..*options
    };

    while tree.depth < maxdepth {
        let direction: Direction = rng.random();
        let current_options = if tree.depth < mindepth {
            &options_no_check
        } else {
            options
        };
        tree = match tree.extend(
            math,
            rng,
            hamiltonian,
            direction,
            collector,
            current_options,
            early,
        ) {
            ExtendResult::Ok(tree) => tree,
            ExtendResult::Turning(mut tree) => {
                for _ in 0..options.extra_doublings {
                    tree = match tree.extend(
                        math,
                        rng,
                        hamiltonian,
                        direction,
                        collector,
                        &options_no_check,
                        early,
                    ) {
                        ExtendResult::Ok(tree) => tree,
                        ExtendResult::Turning(tree) => tree,
                        ExtendResult::Diverging(tree, info) => {
                            let info = tree.info(false, Some(info));
                            collector.register_draw(math, &tree.draw, &info);
                            return Ok((tree.draw, info));
                        }
                        ExtendResult::Err(error) => {
                            return Err(error);
                        }
                    }
                }
                let info = tree.info(false, None);
                collector.register_draw(math, &tree.draw, &info);
                return Ok((tree.draw, info));
            }
            ExtendResult::Diverging(tree, info) => {
                let info = tree.info(false, Some(info));
                collector.register_draw(math, &tree.draw, &info);
                return Ok((tree.draw, info));
            }
            ExtendResult::Err(error) => {
                return Err(error);
            }
        };
    }
    let info = tree.info(true, None);
    collector.register_draw(math, &tree.draw, &info);
    Ok((tree.draw, info))
}

#[cfg(test)]
mod tests {
    use rand::rng;

    use crate::{
        Chain, Settings, adapt_strategy::test_logps::NormalLogp, math::CpuMath,
        sampler::DiagNutsSettings,
    };

    #[test]
    fn to_arrow() {
        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.);
        let math = CpuMath::new(func);

        let settings = DiagNutsSettings::default();
        let mut rng = rng();

        let mut chain = settings.new_chain(0, math, &mut rng);

        chain.set_position(&vec![0.0; ndim]).unwrap();

        let (_, mut progress) = chain.draw().unwrap();
        for _ in 0..10 {
            let (_, prog) = chain.draw().unwrap();
            progress = prog;
        }

        assert!(!progress.diverging);
    }
}
