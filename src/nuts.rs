use std::marker::PhantomData;

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

pub trait Collector {
    type State: State;

    fn register_leapfrog(
        &mut self,
        _start: &Self::State,
        _end: &Self::State,
        _step_size: f64,
        _divergence_info: Option<&dyn DivergenceInfo>,
    ) {
    }
    fn register_draw(&mut self, _state: &Self::State, _info: &SampleInfo) {}
    fn register_init(&mut self, _state: &Self::State) {}
}

pub trait Potential {
    type State: State;
    type DivergenceInfo: DivergenceInfo + 'static;

    fn update_potential_gradient(
        &mut self,
        state: &mut Self::State,
    ) -> Result<(), Self::DivergenceInfo>;
    fn update_velocity(&mut self, state: &mut Self::State);
    fn update_kinetic_energy(&mut self, state: &mut Self::State);
    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut Self::State, rng: &mut R);
}

pub trait State: Clone {
    type Pool;

    fn write_position(&self, out: &mut [f64]);
    fn new(pool: &mut Self::Pool, init: &[f64]) -> Self;
    fn new_empty(pool: &mut Self::Pool) -> Self;
    fn is_turning(&self, other: &Self) -> bool;
    fn energy(&self) -> f64;

    fn first_momentum_halfstep(&self, out: &mut Self, epsilon: f64);
    fn position_step(&self, out: &mut Self, epsilon: f64);
    fn second_momentum_halfstep(&mut self, epsilon: f64);
    fn set_psum(&self, target: &mut Self, dir: Direction);
    fn index_in_trajectory(&self) -> i64;
    fn index_in_trajectory_mut(&mut self) -> &mut i64;
}

fn leapfrog<P: Potential + ?Sized, C: Collector<State = P::State>>(
    pool: &mut <P::State as State>::Pool,
    potential: &mut P,
    start: &P::State,
    dir: Direction,
    step_size: f64,
    collector: &mut C,
) -> Result<(P::State, LeapfrogInfo), P::DivergenceInfo> {
    let mut out = P::State::new_empty(pool);

    let sign = match dir {
        Direction::Forward => 1,
        Direction::Backward => -1,
    };

    let epsilon = (sign as f64) * step_size;

    start.first_momentum_halfstep(&mut out, epsilon);
    potential.update_velocity(&mut out);

    start.position_step(&mut out, epsilon);
    if let Err(div_info) = potential.update_potential_gradient(&mut out) {
        collector.register_leapfrog(start, &out, step_size, Some(&div_info));
        return Err(div_info);
    }

    out.second_momentum_halfstep(epsilon);

    potential.update_velocity(&mut out);
    potential.update_kinetic_energy(&mut out);

    *out.index_in_trajectory_mut() = start.index_in_trajectory() + sign;

    start.set_psum(&mut out, dir);

    collector.register_leapfrog(start, &out, step_size, None);

    Ok((out, LeapfrogInfo {}))
}

#[derive(Debug)]
pub struct SampleInfo {
    pub depth: u64,
    pub divergence_info: Option<Box<dyn DivergenceInfo>>,
    pub maxdepth: bool,
}

pub struct NutsTree<P: Potential + ?Sized, C: Collector<State = P::State>> {
    left: P::State,
    right: P::State,
    draw: P::State,
    log_size: f64,
    depth: u64,
    initial_energy: f64,
    collector: PhantomData<C>,
}

enum ExtendResult<P: Potential + ?Sized, C: Collector<State = P::State>> {
    Ok(NutsTree<P, C>),
    Turning(NutsTree<P, C>),
    Diverging(NutsTree<P, C>, P::DivergenceInfo),
}

impl<P: Potential + ?Sized, C: Collector<State = P::State>> NutsTree<P, C> {
    fn new(state: P::State) -> NutsTree<P, C> {
        let initial_energy = state.energy();
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: 0., // TODO
            initial_energy,
            collector: PhantomData,
        }
    }

    fn extend<R>(
        mut self,
        pool: &mut <P::State as State>::Pool,
        rng: &mut R,
        potential: &mut P,
        direction: Direction,
        step_size: f64,
        collector: &mut C,
    ) -> ExtendResult<P, C>
    where
        P: Potential,
        R: rand::Rng + ?Sized,
    {
        let mut other = match self.single_step(pool, potential, direction, step_size, collector) {
            Ok(tree) => tree,
            Err(info) => return ExtendResult::Diverging(self, info),
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(pool, rng, potential, direction, step_size, collector) {
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
        other: NutsTree<P, C>,
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
        pool: &mut <P::State as State>::Pool,
        integrator: &mut P,
        direction: Direction,
        step_size: f64,
        collector: &mut C,
    ) -> Result<NutsTree<P, C>, P::DivergenceInfo> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let (end, info) = match leapfrog(pool, integrator, start, direction, step_size, collector) {
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
            collector: PhantomData,
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

pub fn draw<P, R, C>(
    pool: &mut <P::State as State>::Pool,
    init: P::State,
    rng: &mut R,
    potential: &mut P,
    maxdepth: u64,
    step_size: f64,
    collector: &mut C,
) -> (P::State, SampleInfo)
where
    P: Potential + ?Sized,
    R: rand::Rng + ?Sized,
    C: Collector<State = P::State>,
{
    collector.register_init(&init);

    let mut tree = NutsTree::new(init);
    while tree.depth < maxdepth {
        use ExtendResult::*;
        let direction: Direction = rng.gen();
        tree = match tree.extend(pool, rng, potential, direction, step_size, collector) {
            Ok(tree) => tree,
            Turning(tree) => {
                let info = tree.info(false, None);
                collector.register_draw(&tree.draw, &info);
                return (tree.draw, info);
            }
            Diverging(tree, info) => {
                let info = tree.info(false, Some(info));
                collector.register_draw(&tree.draw, &info);
                return (tree.draw, info);
            }
        };
    }
    let info = tree.info(true, None);
    (tree.draw, info)
}
