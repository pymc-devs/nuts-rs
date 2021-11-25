//use crate::integrator::{Direction, Integrator, LeapfrogInfo};
use crate::math::logaddexp;

pub trait DivergenceInfo: std::fmt::Debug {}

pub trait LeapfrogInfo {
    fn energy_error(&self) -> f64;
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
    type DivergenceInfo: DivergenceInfo;
    type State: Clone;

    fn leapfrog(
        &mut self,
        start: &Self::State,
        dir: Direction,
    ) -> Result<(Self::State, Self::LeapfrogInfo), Self::DivergenceInfo>;
    fn is_turning(&mut self, start: &Self::State, end: &Self::State) -> bool;
    fn randomize_velocity<R: rand::Rng + ?Sized>(&mut self, state: &mut Self::State, rng: &mut R);
    fn write_position(&self, state: &Self::State, out: &mut [f64]);
    fn new_state(&mut self, init: &[f64]) -> Result<Self::State, Self::DivergenceInfo>;
}

#[derive(Debug)]
pub struct SampleInfo<D: DivergenceInfo> {
    pub depth: u64,
    pub divergence_info: Option<D>,
    pub maxdepth: bool,
}

pub struct NutsTree<I: Integrator + ?Sized> {
    left: I::State,
    right: I::State,
    draw: I::State,
    log_size: f64,
    depth: u64,
}

enum ExtendResult<I: Integrator + ?Sized> {
    Ok(NutsTree<I>),
    Turning(NutsTree<I>),
    Diverging(NutsTree<I>, I::DivergenceInfo),
}

impl<I: Integrator + ?Sized> NutsTree<I> {
    fn new(state: I::State) -> NutsTree<I> {
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: 0., // TODO
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
            Direction::Forward => (&self.left, &other.right),
            Direction::Backward => (&other.left, &self.right),
        };

        let mut turning = integrator.is_turning(first, last);
        if (!turning) & (self.depth > 1) {
            turning = integrator.is_turning(&self.right, &other.right);
        }
        if (!turning) & (self.depth > 1) {
            turning = integrator.is_turning(&self.left, &other.left);
        }

        self.merge_into(other, rng, direction);

        if turning {
            ExtendResult::Turning(self)
        } else {
            ExtendResult::Ok(self)
        }
    }

    fn merge_into<R: rand::Rng + ?Sized>(
        &mut self,
        other: NutsTree<I>,
        rng: &mut R,
        direction: Direction,
    ) {
        assert!(self.depth == other.depth);
        match direction {
            Direction::Forward => {
                self.right = other.right;
            }
            Direction::Backward => {
                self.left = other.left;
            }
        }
        if rng.gen_bool(0.5) {
            self.draw = other.draw;
        }
        self.depth += 1;
        self.log_size = logaddexp(self.log_size, other.log_size);
    }

    fn single_step(
        &self,
        integrator: &mut I,
        direction: Direction,
    ) -> Result<NutsTree<I>, I::DivergenceInfo> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let (end, info) = match integrator.leapfrog(start, direction) {
            Err(divergence_info) => return Err(divergence_info),
            Ok((end, info)) => (end, info),
        };

        let energy_error = info.energy_error();

        Ok(NutsTree {
            right: end.clone(),
            left: end.clone(),
            draw: end,
            depth: 0,
            log_size: -energy_error,
        })
    }

    fn info(
        &self,
        maxdepth: bool,
        divergence_info: Option<I::DivergenceInfo>,
    ) -> SampleInfo<I::DivergenceInfo> {
        SampleInfo {
            depth: self.depth,
            divergence_info,
            maxdepth,
        }
    }
}

pub fn draw<I, R>(
    mut init: I::State,
    rng: &mut R,
    integrator: &mut I,
    maxdepth: u64,
) -> (I::State, SampleInfo<I::DivergenceInfo>)
where
    I: Integrator + ?Sized,
    R: rand::Rng + ?Sized,
{
    integrator.randomize_velocity(&mut init, rng);
    let mut tree = NutsTree::new(init);
    while tree.depth <= maxdepth {
        use ExtendResult::*;
        let direction: Direction = rng.gen();
        tree = match tree.extend(rng, integrator, direction) {
            Ok(tree) => tree,
            Turning(tree) => {
                let info = tree.info(false, None);
                return (tree.draw, info);
            }
            Diverging(tree, info) => {
                let info = tree.info(false, Some(info));
                return (tree.draw, info);
            }
        };
    }
    let info = tree.info(true, None);
    (tree.draw, info)
}
