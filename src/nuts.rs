use crate::math::logaddexp;
use crate::statespace::{Direction, StateSpace, LeapfrogInfo};


pub struct SampleInfo<S: StateSpace> {
    pub divergence_info: Option<<S::LeapfrogInfo as LeapfrogInfo>::DivergenceInfo>,
    pub depth: u64,
    pub turning: bool,
}

pub struct NutsTree<S: StateSpace> {
    left: S::StateIdx,
    right: S::StateIdx,
    draw: S::StateIdx,
    log_size: f64,
    log_weighted_accept_prob: f64,
    depth: u64,
    turning: bool,
}

enum ExtendResult<S: StateSpace> {
    Ok(NutsTree<S>),
    Turning(NutsTree<S>),
    Diverging(NutsTree<S>, <S::LeapfrogInfo as LeapfrogInfo>::DivergenceInfo),
}

impl<S: StateSpace> NutsTree<S> {
    fn new(state: S::StateIdx) -> NutsTree<S> {
        NutsTree {
            right: state,
            left: state,
            draw: state,
            depth: 0,
            log_size: 0.,                 // TODO
            log_weighted_accept_prob: 0., // TODO
            turning: false,
        }
    }

    fn free_states(self, space: &mut S) {
        space.free_state(self.right);
        space.free_state(self.left);
        space.free_state(self.draw);
    }

    fn extend<R>(mut self, rng: &mut R, space: &mut S, direction: Direction) -> ExtendResult<S>
    where
        S: StateSpace,
        R: rand::Rng + ?Sized,
    {
        let mut other = match self.single_step(space, direction) {
            Ok(tree) => tree,
            Err(info) => return ExtendResult::Diverging(self, info),
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(rng, space, direction) {
                Ok(tree) => tree,
                Turning(tree) => {
                    tree.free_states(space);
                    return Turning(self);
                }
                Diverging(tree, info) => {
                    tree.free_states(space);
                    return Diverging(self, info);
                }
            };
        }

        let (first, last) = match direction {
            Direction::Forward => (self.left, other.right),
            Direction::Backward => (other.left, self.right),
        };
        let (middle_first, middle_last) = match direction {
            Direction::Forward => (self.right, other.left),
            Direction::Backward => (other.right, self.left),
        };

        let mut turning = space.is_turning(first, last);
        if (!turning) & (self.depth > 1) {
            turning = space.is_turning(self.right, other.right);
        }
        if (!turning) & (self.depth > 1) {
            turning = space.is_turning(self.left, other.left);
        }

        // Merge other tree into self

        space.free_state(middle_first);
        space.free_state(middle_last);
        self.left = first;
        self.right = last;

        let draw = if rng.gen_bool(0.5) {
            space.free_state(other.draw);
            self.draw
        } else {
            space.free_state(self.draw);
            other.draw
        };
        self.draw = draw;
        self.depth += 1;
        self.turning = turning;
        self.log_size = logaddexp(self.log_size, other.log_size);
        self.log_weighted_accept_prob = logaddexp(
            self.log_weighted_accept_prob,
            other.log_weighted_accept_prob,
        );
        if self.turning {
            ExtendResult::Turning(self)
        } else {
            ExtendResult::Ok(self)
        }
    }

    fn single_step(
        &self,
        space: &mut S,
        direction: Direction,
    ) -> Result<NutsTree<S>, <S::LeapfrogInfo as LeapfrogInfo>::DivergenceInfo> {
        let start = match direction {
            Direction::Forward => self.right,
            Direction::Backward => self.left,
        };
        let (end, info) = space.leapfrog(start, direction);

        if let Some(divergence_info) = info.divergence() {
            return Err(divergence_info);
        }

        Ok(NutsTree {
            right: end,
            left: end,
            draw: end,
            depth: 0,
            log_size: 0.,                 // TODO
            log_weighted_accept_prob: 0., // TODO
            turning: false,
        })
    }

    fn info(
        &self,
        divergence_info: Option<<S::LeapfrogInfo as LeapfrogInfo>::DivergenceInfo>,
    ) -> SampleInfo<S> {
        SampleInfo {
            depth: self.depth,
            turning: self.turning,
            divergence_info,
        }
    }
}

pub fn draw<S, R>(rng: &mut R, space: &mut S, maxdepth: u64) -> (S::StateIdx, SampleInfo<S>)
where
    S: StateSpace,
    R: rand::Rng + ?Sized,
{
    let mut tree = NutsTree::new(space.initial_state());
    while tree.depth <= maxdepth {
        use ExtendResult::*;
        let direction: Direction = rng.gen();
        tree = match tree.extend(rng, space, direction) {
            Ok(tree) => tree,
            Turning(tree) => {
                return (tree.draw, tree.info(None));
            }
            Diverging(tree, info) => return (tree.draw, tree.info(Some(info))),
        };
    }
    if tree.draw != tree.left {
        space.free_state(tree.left);
    }
    if tree.draw != tree.right {
        space.free_state(tree.right);
    }
    (tree.draw, tree.info(None))
}

