use serde::Serialize;
use thiserror::Error;

use std::{fmt::Debug, marker::PhantomData};

use crate::hamiltonian::{Direction, DivergenceInfo, Hamiltonian, LeapfrogResult, Point};
use crate::math::logaddexp;
use crate::state::State;

use crate::math_base::Math;

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

    pub initial_energy: f64,
    pub draw_energy: f64,
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

        let (log_size, end) = match options.walnuts_options {
            Some(ref options) => {
                // Walnuts implementation
                // TODO: Shouldn't all be in this one big function...
                let mut num_steps = 1;
                let mut current = start.clone();

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

                if let Some(info) = last_divergence {
                    let info = DivergenceInfo::new_max_step_size_halvings(math, num_steps, info);
                    return Ok(Err(info));
                }

                let back = direction.reverse();
                let mut reversible = true;

                while num_steps >= 2 {
                    num_steps /= 2;

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
                    return Ok(Err(DivergenceInfo::new_not_reversible(math, start)));
                }
            }
            None => {
                // Classical NUTS.
                // TODO Is equivalent to walnuts with max_step_size_halvings = 0?
                let end = match hamiltonian.leapfrog(math, start, direction, 1, collector) {
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
            initial_energy: self.draw.point().initial_energy(),
            draw_energy: self.draw.energy(),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize)]
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
    pub store_gradient: bool,
    pub store_unconstrained: bool,
    pub check_turning: bool,
    pub store_divergences: bool,
    pub walnuts_options: Option<WalnutsOptions>,
}

impl Default for NutsOptions {
    fn default() -> Self {
        NutsOptions {
            maxdepth: 10,
            mindepth: 0,
            store_gradient: false,
            store_unconstrained: false,
            check_turning: true,
            store_divergences: false,
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
    hamiltonian.initialize_trajectory(math, init, rng)?;
    collector.register_init(math, init, options);

    let mut tree = NutsTree::new(init.clone());

    if math.dim() == 0 {
        let info = tree.info(false, None);
        collector.register_draw(math, init, &info);
        return Ok((init.clone(), info));
    }

    let options_no_check = NutsOptions {
        check_turning: false,
        ..*options
    };

    while tree.depth < options.maxdepth {
        let direction: Direction = rng.random();
        let current_options = if tree.depth < options.mindepth {
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
            ExtendResult::Turning(tree) => {
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
        Chain, Settings, adapt_strategy::test_logps::NormalLogp, cpu_math::CpuMath,
        sampler::DiagGradNutsSettings,
    };

    #[test]
    fn to_arrow() {
        let ndim = 10;
        let func = NormalLogp::new(ndim, 3.);
        let math = CpuMath::new(func);

        let settings = DiagGradNutsSettings::default();
        let mut rng = rng();

        let mut chain = settings.new_chain(0, math, &mut rng);

        let (_, mut progress) = chain.draw().unwrap();
        for _ in 0..10 {
            let (_, prog) = chain.draw().unwrap();
            progress = prog;
        }

        assert!(!progress.diverging);
    }
}
