pub trait LeapfrogInfo {
    type DivergenceInfo;

    fn energy_error(&self) -> f64;
    fn divergence(&self) -> Option<Self::DivergenceInfo>;
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
    type StateIdx: Copy + Eq;

    fn initial_state(&self) -> Self::StateIdx;
    fn leapfrog(&mut self, start: Self::StateIdx, dir: Direction) -> (Self::StateIdx, Self::LeapfrogInfo);
    fn is_turning(&self, start: Self::StateIdx, end: Self::StateIdx) -> bool;
    fn free_state(&mut self, state: Self::StateIdx);
}


type CpuStateIndex = generational_arena::Index;


struct CpuState {
    p: Box<[f64]>,
    q: Box<[f64]>,
    v: Box<[f64]>,
    p_sum: Box<[f64]>,
    grad: Box<[f64]>,
    idx_in_trajectory: i64,
    kinetic_energy: f64,
    potential_energy: f64,
    used: bool,
}

struct CpuStateSpace {
    states: generational_arena::Arena<CpuState>,
    initial_state: Option<CpuStateIndex>,
    
}

struct CpuLeapfrogInfo {}

impl LeapfrogInfo for CpuLeapfrogInfo {
    type DivergenceInfo = bool;

    fn energy_error(&self) -> f64 {
        unimplemented!();
    }

    fn divergence(&self) -> Option<bool> {
        unimplemented!();
    }
}

impl StateSpace for CpuStateSpace {
    type LeapfrogInfo = CpuLeapfrogInfo;
    type StateIdx = CpuStateIndex;

    fn initial_state(&self) -> Self::StateIdx {
        let state = self.initial_state.expect("No initial state.");
        assert!(self.states.contains(state));
        state
    }

    fn leapfrog(&mut self, start: Self::StateIdx, dir: Direction) -> (Self::StateIdx, Self::LeapfrogInfo) {
        unimplemented!();
    }

    fn is_turning(&self, start: Self::StateIdx, end: Self::StateIdx) -> bool {
        use crate::math::scalar_prods_of_diff;

        let start = &self.states[start];
        let end = &self.states[end];

        let (start, end) = if start.idx_in_trajectory < end.idx_in_trajectory {
            (start, end)
        } else {
            (end, start)
        };

        let (a, b) = scalar_prods_of_diff(&end.p_sum, &start.p_sum, &end.v, &start.v);
        (a < 0.) | (b < 0.)
    }

    fn free_state(&mut self, idx: Self::StateIdx) {
        unimplemented!();
    }
}
