fn logaddexp(a: f64, b: f64) -> f64 {
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    a + (b - a).exp().ln_1p()
}

pub trait LeapfrogInfo {
    type DivergenceInfo;

    fn energy_error(&self) -> f64;
    fn divergence(&self) -> Option<Self::DivergenceInfo>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StateIdx {
    pub idx: usize,
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

pub trait StateSpace {
    type LeapfrogInfo: LeapfrogInfo;

    fn initial_state(&self) -> StateIdx;
    fn leapfrog(&mut self, start: StateIdx, dir: Direction) -> (StateIdx, Self::LeapfrogInfo);
    fn is_turning(&self, start: StateIdx, end: StateIdx) -> bool;
    fn free_state(&mut self, state: StateIdx);
}

pub struct SampleInfo<S: StateSpace> {
    pub divergence_info: Option<<S::LeapfrogInfo as LeapfrogInfo>::DivergenceInfo>,
    pub depth: u64,
    pub turning: bool,
}

pub struct NutsTree {
    left: StateIdx,
    right: StateIdx,
    draw: StateIdx,
    log_size: f64,
    log_weighted_accept_prob: f64,
    depth: u64,
    turning: bool,
}

enum ExtendResult<S: StateSpace> {
    Ok(NutsTree),
    Turning(NutsTree),
    Diverging(NutsTree, <S::LeapfrogInfo as LeapfrogInfo>::DivergenceInfo),
}

impl NutsTree {
    fn new(state: StateIdx) -> NutsTree {
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

    fn free_states<S: StateSpace>(self, space: &mut S) {
        space.free_state(self.right);
        space.free_state(self.left);
        space.free_state(self.draw);
    }

    fn extend<S, R>(mut self, rng: &mut R, space: &mut S, direction: Direction) -> ExtendResult<S>
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

    fn single_step<S: StateSpace>(
        &self,
        space: &mut S,
        direction: Direction,
    ) -> Result<NutsTree, <S::LeapfrogInfo as LeapfrogInfo>::DivergenceInfo> {
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

    fn info<S: StateSpace>(
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

pub fn draw<S, R>(rng: &mut R, space: &mut S, maxdepth: u64) -> (StateIdx, SampleInfo<S>)
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

trait FreeListable {
    type Idx;

    fn expect_used(&self);
    fn make_unused(&mut self, next_unused: Self::Idx);
    fn make_used(&mut self) -> Self::Idx;
}

struct State {
    p: Box<[f64]>,
    q: Box<[f64]>,
    v: Box<[f64]>,
    p_sum: Box<[f64]>,
    grad: Box<[f64]>,
    idx_in_trajectory: i64,
    kinetic_energy: f64,
    potential_energy: f64,
    next_free: StateIdx,
    used: bool,
}

struct SimpleStateSpace {
    states: Vec<State>,
    first_free: StateIdx,
    initial_state: Option<StateIdx>,
}

impl SimpleStateSpace {
    fn state(&self, idx: StateIdx) -> &State {
        let state = &self.states[idx.idx];
        if !state.used {
            panic!("Accessing unused state.");
        }
        state
    }

    fn state_mut(&mut self, idx: StateIdx) -> &mut State {
        let state = &mut self.states[idx.idx];
        if !state.used {
            panic!("Accessing unused state.");
        }
        state
    }

    fn next_free(&mut self) -> StateIdx {
        let idx = self.first_free;
        let state = &mut self.states[idx.idx];
        assert!(!state.used);
        state.used = true;
        self.first_free = state.next_free;
        idx
    }
}

fn scalar_prods_of_diff(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> (f64, f64) {
    let n = a.len();
    assert!(b.len() == n);
    assert!(c.len() == n);
    assert!(d.len() == n);
    let mut sum_c: f64 = 0.;
    let mut sum_d: f64 = 0.;

    for i in 0..n {
        sum_c += (a[i] - b[i]) * c[i];
        sum_d += (a[i] - b[i]) * d[i];
    }
    (sum_c, sum_d)
}

impl StateSpace for SimpleStateSpace {
    type LeapfrogInfo = SimpleLeapfrogInfo;

    fn initial_state(&self) -> StateIdx {
        let state = self.initial_state.expect("No initial state.");
        self.state(state);
        state
    }

    fn leapfrog(&mut self, start: StateIdx, dir: Direction) -> (StateIdx, Self::LeapfrogInfo) {
        unimplemented!();
    }

    fn is_turning(&self, start: StateIdx, end: StateIdx) -> bool {
        let start = self.state(start);
        let end = self.state(end);

        let (start, end) = if start.idx_in_trajectory < end.idx_in_trajectory {
            (start, end)
        } else {
            (end, start)
        };

        let (a, b) = scalar_prods_of_diff(&end.p_sum, &start.p_sum, &end.v, &start.v);
        (a < 0.) | (b < 0.)
    }

    fn free_state(&mut self, idx: StateIdx) {
        let state = &mut self.states[idx.idx];
        state.used = false;
        state.next_free = self.first_free;
        self.first_free = idx;
    }
}

struct SimpleLeapfrogInfo {}

impl LeapfrogInfo for SimpleLeapfrogInfo {
    type DivergenceInfo = bool;

    fn energy_error(&self) -> f64 {
        unimplemented!();
    }

    fn divergence(&self) -> Option<bool> {
        unimplemented!();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
