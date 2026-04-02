//! Implement the recursive doubling tree expansion that is the heart of the NUTS algorithm.

use rand::RngExt;
use rand_distr::num_traits::ToPrimitive;
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
    ) -> ExtendResult<M, H, C>
    where
        H: Hamiltonian<M>,
        R: rand::Rng + ?Sized,
    {
        let mut other = match self.single_step(math, hamiltonian, direction, collector) {
            Ok(Ok(tree)) => tree,
            Ok(Err(info)) => return ExtendResult::Diverging(self, info),
            Err(err) => return ExtendResult::Err(err),
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(math, rng, hamiltonian, direction, collector, options) {
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
        collector: &mut C,
    ) -> Result<std::result::Result<NutsTree<M, H, C>, DivergenceInfo>> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let end = match hamiltonian.leapfrog(math, start, direction, 1.0, collector) {
            LeapfrogResult::Divergence(info) => return Ok(Err(info)),
            LeapfrogResult::Err(err) => return Err(NutsError::LogpFailure(err.into())),
            LeapfrogResult::Ok(end) => end,
        };

        let log_size = -end.point().energy_error();
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

#[derive(Debug, Clone)]
pub struct NutsOptions {
    pub maxdepth: u64,
    pub mindepth: u64,
    pub check_turning: bool,
    pub store_divergences: bool,
    pub target_integration_time: Option<f64>,
    pub extra_doublings: u64,
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
    hamiltonian.initialize_trajectory(math, init, rng)?;
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
