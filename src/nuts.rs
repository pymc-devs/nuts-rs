//! Implement the recursive doubling tree expansion that is the heart of the NUTS algorithm.

use rand::RngExt;
use rand_distr::num_traits::ToPrimitive;
use thiserror::Error;

use std::{fmt::Debug, marker::PhantomData};

use crate::dynamics::{Direction, DivergenceInfo, Hamiltonian, LeapfrogResult, Point, State};
use crate::math::{Math, logaddexp};
use crate::walnuts::{MacroStep, WalnutsOptions, macro_step};

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

/// Outcome of integrating a macro step at the base step size.
///
/// With WALNUTS this is the first attempt of the macro step, before any step
/// size halving. It consists of `min_micro_steps` leapfrog steps of size
/// `step_size / min_micro_steps`.
#[derive(Debug, Clone, Copy)]
pub enum BaseStepOutcome {
    /// A plain leapfrog step without WALNUTS.
    Plain,
    /// Finished with this energy error. It may be outside of the WALNUTS
    /// tolerance.
    EnergyError(f64),
    /// Diverged.
    Diverged,
}

/// Information about how a step of the trajectory was computed.
#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    /// Number of gradient evaluations the step needed, including rejected
    /// WALNUTS attempts and reversibility checks.
    pub num_gradients: u64,
    pub base_step: BaseStepOutcome,
}

impl StepInfo {
    /// A single plain leapfrog step.
    pub const PLAIN: StepInfo = StepInfo {
        num_gradients: 1,
        base_step: BaseStepOutcome::Plain,
    };
}

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
        _step: &StepInfo,
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

    /// Whether the trajectory was terminated because a WALNUTS
    /// macro step was not reversible.
    pub irreversible: bool,

    /// Whether the trajectory was terminated because extending it
    /// would have needed more than `max_leapfrog_steps` leapfrog steps.
    pub reached_max_leapfrog_steps: bool,
}

/// A part of the trajectory tree during NUTS sampling.
struct NutsTree<M: Math, H: Hamiltonian<M>, C: Collector<M, H::Point>> {
    /// The left position of the tree.
    ///
    /// The left side always has the smaller index_in_trajectory.
    /// Leapfrogs in backward direction will replace the left.
    left: State<M, H::Point>,
    right: State<M, H::Point>,

    /// A draw from the trajectory between left and right using
    /// multinomial sampling.
    draw: State<M, H::Point>,
    log_size: f64,
    depth: u64,

    /// The number of leapfrog steps between the points of the tree. For
    /// WALNUTS these are the micro steps of the accepted attempts, which,
    /// unlike the gradient evaluations, do not depend on the integration
    /// direction.
    num_steps: u64,

    /// A tree is the main tree if it contains the initial point
    /// of the trajectory.
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
    /// A WALNUTS macro step was not reversible, the trajectory
    /// has to stop.
    Irreversible(NutsTree<M, H, C>),
    /// The extension would have needed more leapfrog steps than allowed. The tree is returned without the extension.
    MaxLeapfrogSteps(NutsTree<M, H, C>),
}

/// The result of a single leaf of the tree.
enum LeafResult<M: Math, H: Hamiltonian<M>, C: Collector<M, H::Point>> {
    /// The new leaf, and the number of leapfrog steps to it.
    Ok(NutsTree<M, H, C>, u64),
    Diverging(DivergenceInfo),
    Irreversible,
}

impl<M: Math, H: Hamiltonian<M>, C: Collector<M, H::Point>> NutsTree<M, H, C> {
    fn new(state: State<M, H::Point>) -> NutsTree<M, H, C> {
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: 0.,
            num_steps: 0,
            is_main: true,
            _phantom2: PhantomData,
        }
    }

    /// Double the tree in `direction`.
    ///
    /// The extension may use at most `max_steps` leapfrog steps.
    /// Otherwise it is discarded, like an extension that contains a U-turn.
    /// This keeps the sampler reversible, because the decision only depends
    /// on the tree and the direction, not on the starting point within the
    /// tree.
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
        max_steps: u64,
    ) -> ExtendResult<M, H, C>
    where
        H: Hamiltonian<M>,
        R: rand::Rng + ?Sized,
    {
        let (mut other, leaf_steps) =
            match self.single_step(math, hamiltonian, direction, options, collector) {
                Ok(LeafResult::Ok(tree, num_steps)) => (tree, num_steps),
                Ok(LeafResult::Diverging(info)) => return ExtendResult::Diverging(self, info),
                Ok(LeafResult::Irreversible) => return ExtendResult::Irreversible(self),
                Err(err) => return ExtendResult::Err(err),
            };
        let Some(max_steps) = max_steps.checked_sub(leaf_steps) else {
            return ExtendResult::MaxLeapfrogSteps(self);
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            let remaining = max_steps - other.num_steps;
            other = match other.extend(
                math,
                rng,
                hamiltonian,
                direction,
                collector,
                options,
                remaining,
            ) {
                Ok(tree) => tree,
                Turning(_) => {
                    return Turning(self);
                }
                Diverging(_, info) => {
                    return Diverging(self, info);
                }
                Irreversible(_) => {
                    return Irreversible(self);
                }
                MaxLeapfrogSteps(_) => {
                    return MaxLeapfrogSteps(self);
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
            if options.uturn_check_first_step || (self.depth > 0) {
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

        self.num_steps += leaf_steps + other.num_steps;
        self.merge_into(math, other, rng, direction);

        if turning {
            ExtendResult::Turning(self)
        } else {
            ExtendResult::Ok(self)
        }
    }

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

    fn single_step(
        &self,
        math: &mut M,
        hamiltonian: &mut H,
        direction: Direction,
        options: &NutsOptions,
        collector: &mut C,
    ) -> Result<LeafResult<M, H, C>> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let (end, num_steps) = match options.walnuts {
            Some(ref walnuts) => {
                match macro_step(math, hamiltonian, start, direction, walnuts, collector) {
                    Ok(MacroStep::Ok { end, num_steps }) => (end, num_steps),
                    Ok(MacroStep::Divergence(info)) => return Ok(LeafResult::Diverging(info)),
                    Ok(MacroStep::Irreversible) => return Ok(LeafResult::Irreversible),
                    Err(err) => return Err(NutsError::LogpFailure(err.into())),
                }
            }
            None => match hamiltonian.leapfrog(
                math,
                start,
                direction,
                1.0,
                start.point().initial_energy(),
                options.max_energy_error,
                collector,
            ) {
                LeapfrogResult::Divergence(info) => return Ok(LeafResult::Diverging(info)),
                LeapfrogResult::Err(err) => return Err(NutsError::LogpFailure(err.into())),
                LeapfrogResult::Ok(end) => (end, 1),
            },
        };

        let log_size = -end.point().energy_error();
        Ok(LeafResult::Ok(
            NutsTree {
                right: end.clone(),
                left: end.clone(),
                draw: end,
                depth: 0,
                log_size,
                num_steps: 0,
                is_main: false,
                _phantom2: PhantomData,
            },
            num_steps,
        ))
    }

    fn info(
        &self,
        maxdepth: bool,
        divergence_info: Option<DivergenceInfo>,
        irreversible: bool,
        max_leapfrog_steps: bool,
    ) -> SampleInfo {
        SampleInfo {
            depth: self.depth,
            divergence_info,
            reached_maxdepth: maxdepth,
            irreversible,
            reached_max_leapfrog_steps: max_leapfrog_steps,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NutsOptions {
    pub maxdepth: u64,
    pub mindepth: u64,
    pub check_turning: bool,
    pub store_divergences: bool,
    pub target_integration_time: Option<f64>,
    pub extra_doublings: u64,
    pub max_energy_error: f64,
    pub uturn_check_first_step: bool,
    /// Use WALNUTS macro steps instead of single leapfrog steps.
    pub walnuts: Option<WalnutsOptions>,
    /// Stop the trajectory before it needs more leapfrog steps than this.
    /// See `NutsSettings::max_leapfrog_steps`.
    pub max_leapfrog_steps: Option<u64>,
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
            uturn_check_first_step: true,
            walnuts: None,
            max_leapfrog_steps: None,
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

    let finish = |math: &mut M,
                  collector: &mut C,
                  tree: NutsTree<M, H, C>,
                  maxdepth: bool,
                  divergence_info: Option<DivergenceInfo>,
                  irreversible: bool,
                  max_leapfrog_steps: bool| {
        let info = tree.info(maxdepth, divergence_info, irreversible, max_leapfrog_steps);
        collector.register_draw(math, &tree.draw, &info);
        Ok((tree.draw, info))
    };

    if math.dim() == 0 {
        return finish(math, collector, tree, false, None, false, false);
    }

    let options_no_check = NutsOptions {
        check_turning: false,
        ..*options
    };

    let remaining_steps = |tree: &NutsTree<M, H, C>| {
        options
            .max_leapfrog_steps
            .map_or(u64::MAX, |max| max.saturating_sub(tree.num_steps))
    };

    // Each step of the trajectory needs at least this many leapfrog steps.
    let min_steps_per_step = options.walnuts.map_or(1, |walnuts| walnuts.min_micro_steps);

    // Whether doubling the tree could stay within `max_leapfrog_steps`. If
    // not, we stop before computing anything. The decision only depends on
    // the tree, so it keeps the sampler reversible.
    let can_double = |tree: &NutsTree<M, H, C>| {
        let min_new_steps = 1u64
            .checked_shl(tree.depth as u32)
            .map_or(u64::MAX, |size| size.saturating_mul(min_steps_per_step));
        min_new_steps <= remaining_steps(tree)
    };

    loop {
        if tree.depth >= maxdepth {
            return finish(math, collector, tree, true, None, false, false);
        }
        if !can_double(&tree) {
            return finish(math, collector, tree, false, None, false, true);
        }
        let direction: Direction = rng.random();
        let current_options = if tree.depth < mindepth {
            &options_no_check
        } else {
            options
        };
        let max_steps = remaining_steps(&tree);
        tree = match tree.extend(
            math,
            rng,
            hamiltonian,
            direction,
            collector,
            current_options,
            max_steps,
        ) {
            ExtendResult::Ok(tree) => tree,
            ExtendResult::Turning(mut tree) => {
                // Extra doublings are not limited by `maxdepth`.
                for _ in 0..options.extra_doublings {
                    if !can_double(&tree) {
                        return finish(math, collector, tree, false, None, false, true);
                    }
                    let max_steps = remaining_steps(&tree);
                    tree = match tree.extend(
                        math,
                        rng,
                        hamiltonian,
                        direction,
                        collector,
                        &options_no_check,
                        max_steps,
                    ) {
                        ExtendResult::Ok(tree) => tree,
                        ExtendResult::Turning(tree) => tree,
                        ExtendResult::Irreversible(tree) => {
                            return finish(math, collector, tree, false, None, true, false);
                        }
                        ExtendResult::Diverging(tree, info) => {
                            return finish(math, collector, tree, false, Some(info), false, false);
                        }
                        ExtendResult::MaxLeapfrogSteps(tree) => {
                            return finish(math, collector, tree, false, None, false, true);
                        }
                        ExtendResult::Err(error) => {
                            return Err(error);
                        }
                    }
                }
                return finish(math, collector, tree, false, None, false, false);
            }
            ExtendResult::Irreversible(tree) => {
                return finish(math, collector, tree, false, None, true, false);
            }
            ExtendResult::Diverging(tree, info) => {
                return finish(math, collector, tree, false, Some(info), false, false);
            }
            ExtendResult::MaxLeapfrogSteps(tree) => {
                return finish(math, collector, tree, false, None, false, true);
            }
            ExtendResult::Err(error) => {
                return Err(error);
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use rand::rng;

    use crate::{
        Chain, Settings, math::CpuMath, math::test_logps::NormalLogp, sampler::DiagNutsSettings,
    };

    #[test]
    fn to_arrow() {
        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.);
        let math = CpuMath::new(func);

        let settings = DiagNutsSettings::default();
        let mut rng = rng();

        let mut chain = settings.new_chain(0, math, &mut rng).unwrap();

        chain.set_position(&vec![0.0; ndim]).unwrap();

        let (_, mut progress) = chain.draw().unwrap();
        for _ in 0..10 {
            let (_, prog) = chain.draw().unwrap();
            progress = prog;
        }

        assert!(!progress.diverging);
    }
}
