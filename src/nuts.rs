//use crate::integrator::{Direction, Integrator, LeapfrogInfo};
use crate::math::logaddexp;

pub trait DivergenceInfo: std::fmt::Debug {}

pub struct LeapfrogInfo {}

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

pub trait Potential {
    type State: State;
    type DivergenceInfo: DivergenceInfo + 'static;

    fn update_energy(&self, state: &mut Self::State);
    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut Self::State, rng: &mut R);
    fn leapfrog(
        &mut self,
        start: &Self::State,
        dir: Direction,
        step_size: f64,
    ) -> Result<(Self::State, LeapfrogInfo), Self::DivergenceInfo>;
    fn init_state(&mut self, position: &[f64]) -> Result<Self::State, Self::DivergenceInfo>;
}

pub trait State: Clone {
    type Pool;

    fn write_position(&self, out: &mut [f64]);
    fn new(pool: &mut Self::Pool, init: &[f64], init_grad: &[f64]) -> Self;
    fn deep_clone(&self, pool: &mut Self::Pool) -> Self;
    fn is_turning(&self, other: &Self) -> bool;
    fn energy(&self) -> f64;
}

#[derive(Debug)]
pub struct SampleInfo {
    pub depth: u64,
    pub divergence_info: Option<Box<dyn DivergenceInfo>>,
    pub maxdepth: bool,
}

pub struct NutsTree<P: Potential + ?Sized> {
    left: P::State,
    right: P::State,
    draw: P::State,
    log_size: f64,
    depth: u64,
    initial_energy: f64,
}

enum ExtendResult<P: Potential + ?Sized> {
    Ok(NutsTree<P>),
    Turning(NutsTree<P>),
    Diverging(NutsTree<P>, P::DivergenceInfo),
}

impl<P: Potential + ?Sized> NutsTree<P> {
    fn new(state: P::State) -> NutsTree<P> {
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: 0., // TODO
            initial_energy: state.energy(),
        }
    }

    fn extend<R>(
        mut self,
        rng: &mut R,
        potential: &mut P,
        direction: Direction,
        step_size: f64,
    ) -> ExtendResult<P>
    where
        P: Potential,
        R: rand::Rng + ?Sized,
    {
        let mut other = match self.single_step(potential, direction, step_size) {
            Ok(tree) => tree,
            Err(info) => return ExtendResult::Diverging(self, info),
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(rng, potential, direction, step_size) {
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

        let mut turning = first.is_turning(last);
        if (!turning) & (self.depth > 1) {
            turning = self.right.is_turning(&other.right);
        }
        if (!turning) & (self.depth > 1) {
            turning = self.left.is_turning(&other.left);
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
        other: NutsTree<P>,
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
        integrator: &mut P,
        direction: Direction,
        step_size: f64,
    ) -> Result<NutsTree<P>, P::DivergenceInfo> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let (end, info) = match integrator.leapfrog(start, direction, step_size) {
            Err(divergence_info) => return Err(divergence_info),
            Ok((end, info)) => (end, info),
        };

        let energy_error = self.initial_energy - end.energy();

        Ok(NutsTree {
            right: end.clone(),
            left: end.clone(),
            draw: end,
            depth: 0,
            log_size: -energy_error,
            initial_energy: self.initial_energy,
        })
    }

    fn info(&self, maxdepth: bool, divergence_info: Option<P::DivergenceInfo>) -> SampleInfo {
        let info: Option<Box<dyn DivergenceInfo>> = match divergence_info {
            Some(info) => Some(Box::new(info)),
            None => None,
        };
        SampleInfo {
            depth: self.depth,
            divergence_info: info,
            maxdepth,
        }
    }
}

pub fn draw<P, R>(
    mut init: P::State,
    rng: &mut R,
    potential: &mut P,
    maxdepth: u64,
    step_size: f64,
) -> (P::State, SampleInfo)
where
    P: Potential + ?Sized,
    R: rand::Rng + ?Sized,
{
    potential.randomize_momentum(&mut init, rng);
    let mut tree = NutsTree::new(init);
    while tree.depth < maxdepth {
        use ExtendResult::*;
        let direction: Direction = rng.gen();
        tree = match tree.extend(rng, potential, direction, step_size) {
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
