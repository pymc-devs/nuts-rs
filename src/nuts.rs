//use crate::integrator::{Direction, Integrator, LeapfrogInfo};
use crate::math::logaddexp;


pub trait DivergenceInfo {}

pub trait LeapfrogInfo {
    type DivergenceInfo: DivergenceInfo;

    fn energy_error(&self) -> f64;
    fn divergence(self) -> Option<Self::DivergenceInfo>;
    fn diverging(&self) -> bool;
}

#[derive(Copy, Clone)]
pub enum Direction {
    Forward,
    Backward,
}

impl rand::distributions::Distribution<Direction> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Direction {
        if rng.gen::<bool>() {
            Direction::Forward
        } else {
            Direction::Backward
        }
    }
}


pub trait Integrator {
    type LeapfrogInfo: LeapfrogInfo;
    type State: Clone;

    fn initial_state(&self) -> Self::State;
    fn leapfrog(
        &mut self,
        start: &Self::State,
        dir: Direction,
    ) -> (Self::State, Self::LeapfrogInfo);
    fn is_turning(&self, start: &Self::State, end: &Self::State) -> bool;
    fn accept(&mut self, state: Self::State, info: SampleInfo<Self>);
    fn write_position(&self, state: &Self::State, out: &mut [f64]);
}


#[derive(Debug)]
pub struct SampleInfo_<L: LeapfrogInfo> {
    pub divergence_info: Option<L::DivergenceInfo>,
    pub depth: u64,
    pub turning: bool,
}

pub type SampleInfo<I> = SampleInfo_<<I as Integrator>::LeapfrogInfo>;


pub struct NutsTree<I: Integrator + ?Sized> {
    left: I::State,
    right: I::State,
    draw: I::State,
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
    fn new(state: I::State) -> NutsTree<I> {
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: 0.,                 // TODO
            log_weighted_accept_prob: 0., // TODO
            turning: false,
        }
    }

    fn extend<R>(mut self, rng: &mut R, integrator: &mut I, direction: Direction) -> ExtendResult<I>
    where
        I: Integrator,
        R: rand::Rng + ?Sized,
    {
        let mut other = match self.single_step(integrator, direction) {
            Ok(tree) => tree,
            Err(info) => return ExtendResult::Diverging(self, info),
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(rng, integrator, direction) {
                Ok(tree) => tree,
                Turning(_) => {
                    return Turning(self);
                }
                Diverging(_, info) => {
                    return Diverging(self, info);
                }
            };
        }

        let (first, last) = match direction {
            Direction::Forward => (self.left.clone(), other.right.clone()),
            Direction::Backward => (other.left.clone(), self.right.clone()),
        };

        let mut turning = integrator.is_turning(&first, &last);
        if (!turning) & (self.depth > 1) {
            turning = integrator.is_turning(&self.right, &other.right);
        }
        if (!turning) & (self.depth > 1) {
            turning = integrator.is_turning(&self.left, &other.left);
        }

        // Merge other tree into self

        self.left = first;
        self.right = last;

        let draw = if rng.gen_bool(0.5) {
            self.draw
        } else {
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
        integrator: &mut I,
        direction: Direction,
    ) -> Result<NutsTree<I>, I::LeapfrogInfo> {
        let start = match direction {
            Direction::Forward => self.right.clone(),
            Direction::Backward => self.left.clone(),
        };
        let (end, info) = integrator.leapfrog(&start, direction);

        if info.diverging() {
            return Err(info);
        }

        Ok(NutsTree {
            right: end.clone(),
            left: end.clone(),
            draw: end,
            depth: 0,
            log_size: 0.,                 // TODO
            log_weighted_accept_prob: 0., // TODO
            turning: false,
        })
    }

    fn info(&self, leapfrog_info: Option<I::LeapfrogInfo>) -> SampleInfo<I> {
        let divergence_info = match leapfrog_info {
            None => None,
            Some(info) => match info.divergence() {
                None => None,
                Some(val) => Some(val),
            },
        };

        SampleInfo_ {
            depth: self.depth,
            turning: self.turning,
            divergence_info,
        }
    }
}

pub fn draw<I, R>(rng: &mut R, integrator: &mut I, maxdepth: u64) -> (I::State, SampleInfo<I>)
where
    I: Integrator + ?Sized,
    R: rand::Rng + ?Sized,
{
    let mut tree = NutsTree::new(integrator.initial_state());
    while tree.depth <= maxdepth {
        use ExtendResult::*;
        let direction: Direction = rng.gen();
        tree = match tree.extend(rng, integrator, direction) {
            Ok(tree) => tree,
            Turning(tree) => {
                let info = tree.info(None);
                return (tree.draw, info);
            }
            Diverging(tree, info) => {
                let info = tree.info(Some(info));
                return (tree.draw, info);
            }
        };
    }
    let info = tree.info(None);
    (tree.draw, info)
}
