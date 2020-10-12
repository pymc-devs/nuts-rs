use crate::math::logaddexp;
use crate::integrator::{Direction, Integrator, LeapfrogInfo};

pub struct SampleInfo_<L: LeapfrogInfo> {
    pub divergence_info: Option<L::DivergenceInfo>,
    pub depth: u64,
    pub turning: bool,
}

pub type SampleInfo<I: Integrator + ?Sized> = SampleInfo_<I::LeapfrogInfo>;

pub struct NutsTree<I: Integrator + ?Sized> {
    left: I::StateIdx,
    right: I::StateIdx,
    draw: I::StateIdx,
    log_size: f64,
    log_weighted_accept_prob: f64,
    depth: u64,
    turning: bool,
}

enum ExtendResult<I: Integrator + ?Sized> {
    Ok(NutsTree<I>),
    Turning(NutsTree<I>),
    Diverging(NutsTree<I>, I::LeapfrogInfo),
}

impl<I: Integrator + ?Sized> NutsTree<I> {
    fn new(state: I::StateIdx) -> NutsTree<I> {
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

    fn free_states(self, space: &mut I) {
        space.free_state(self.right);
        space.free_state(self.left);
        space.free_state(self.draw);
    }

    fn extend<R>(mut self, rng: &mut R, space: &mut I, direction: Direction) -> ExtendResult<I>
    where
        I: Integrator,
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
        space: &mut I,
        direction: Direction,
    ) -> Result<NutsTree<I>, I::LeapfrogInfo> {
        let start = match direction {
            Direction::Forward => self.right,
            Direction::Backward => self.left,
        };
        let (end, info) = space.leapfrog(start, direction);

        if info.diverging() {
            return Err(info);
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

    fn info(&self, leapfrog_info: Option<I::LeapfrogInfo>) -> SampleInfo<I> {
        let divergence_info = match leapfrog_info {
            None => { None },
            Some(mut info) => match info.divergence() {
                None => { None },
                Some(val) => { Some(val) }
            }
        };

        SampleInfo_ {
            depth: self.depth,
            turning: self.turning,
            divergence_info,
        }
    }
}

pub fn draw<I, R>(
    rng: &mut R,
    space: &mut I,
    maxdepth: u64,
) -> (I::StateIdx, SampleInfo<I>)
where
    I: Integrator + ?Sized,
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
