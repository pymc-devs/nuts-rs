use std::{
    cell::RefCell,
    fmt::Debug,
    ops::Deref,
    rc::{Rc, Weak},
};

use crate::math::{axpy, axpy_out, scalar_prods2, scalar_prods3};

#[derive(Debug)]
struct StateStorage {
    free_states: RefCell<Vec<Rc<InnerStateReusable>>>,
}

impl StateStorage {
    fn new() -> StateStorage {
        StateStorage {
            free_states: RefCell::new(Vec::with_capacity(20)),
        }
    }
}

impl ReuseState for StateStorage {
    fn reuse_state(&self, state: Rc<InnerStateReusable>) {
        self.free_states.borrow_mut().push(state)
    }
}

pub(crate) struct StatePool {
    storage: Rc<StateStorage>,
    dim: usize,
}

impl StatePool {
    pub(crate) fn new(dim: usize) -> StatePool {
        StatePool {
            storage: Rc::new(StateStorage::new()),
            dim,
        }
    }

    pub(crate) fn new_state(&mut self) -> State {
        let inner = match self.storage.free_states.borrow_mut().pop() {
            Some(inner) => {
                if self.dim != inner.inner.q.len() {
                    panic!("dim mismatch");
                }
                inner
            }
            None => {
                let owner: Rc<dyn ReuseState> = self.storage.clone();
                Rc::new(InnerStateReusable::new(self.dim, &owner))
            }
        };
        State {
            inner: std::mem::ManuallyDrop::new(inner),
        }
    }
}

trait ReuseState: Debug {
    fn reuse_state(&self, state: Rc<InnerStateReusable>);
}

#[derive(Debug, Clone)]
pub(crate) struct InnerState {
    pub(crate) p: Box<[f64]>,
    pub(crate) q: Box<[f64]>,
    pub(crate) v: Box<[f64]>,
    pub(crate) p_sum: Box<[f64]>,
    pub(crate) grad: Box<[f64]>,
    pub(crate) idx_in_trajectory: i64,
    pub(crate) kinetic_energy: f64,
    pub(crate) potential_energy: f64,
}

#[derive(Debug)]
pub(crate) struct InnerStateReusable {
    inner: InnerState,
    reuser: Weak<dyn ReuseState>,
}

impl InnerStateReusable {
    fn new(size: usize, owner: &Rc<dyn ReuseState>) -> InnerStateReusable {
        InnerStateReusable {
            inner: InnerState {
                p: vec![0.; size].into(),
                q: vec![0.; size].into(),
                v: vec![0.; size].into(),
                p_sum: vec![0.; size].into(),
                grad: vec![0.; size].into(),
                idx_in_trajectory: 0,
                kinetic_energy: 0.,
                potential_energy: 0.,
            },
            reuser: Rc::downgrade(owner),
        }
    }
}

#[derive(Debug)]
pub(crate) struct State {
    inner: std::mem::ManuallyDrop<Rc<InnerStateReusable>>,
}

impl Deref for State {
    type Target = InnerState;

    fn deref(&self) -> &Self::Target {
        &self.inner.inner
    }
}

#[derive(Debug)]
pub(crate) struct StateInUse {}

type Result<T> = std::result::Result<T, StateInUse>;

impl State {
    pub(crate) fn try_mut_inner(&mut self) -> Result<&mut InnerState> {
        match Rc::get_mut(&mut self.inner) {
            Some(val) => Ok(&mut val.inner),
            None => Err(StateInUse {}),
        }
    }

    pub(crate) fn clone_inner(&self) -> InnerState {
        self.inner.inner.clone()
    }
}

impl Drop for State {
    fn drop(&mut self) {
        let mut rc = unsafe { std::mem::ManuallyDrop::take(&mut self.inner) };
        if let Some(state_ref) = Rc::get_mut(&mut rc) {
            if let Some(reuser) = &mut state_ref.reuser.upgrade() {
                reuser.reuse_state(rc);
            }
        }
    }
}

impl Clone for State {
    fn clone(&self) -> Self {
        State {
            inner: self.inner.clone(),
        }
    }
}

impl crate::nuts::State for State {
    type Pool = StatePool;

    fn is_turning(&self, other: &Self) -> bool {
        let (start, end) = if self.idx_in_trajectory < other.idx_in_trajectory {
            (&*self, other)
        } else {
            (other, &*self)
        };

        let a = start.idx_in_trajectory;
        let b = end.idx_in_trajectory;

        assert!(a < b);
        let (turn1, turn2) = if (a >= 0) & (b >= 0) {
            scalar_prods3(&end.p_sum, &start.p_sum, &start.p, &end.v, &start.v)
        } else if (b >= 0) & (a < 0) {
            scalar_prods2(&end.p_sum, &start.p_sum, &end.v, &start.v)
        } else {
            assert!((a < 0) & (b < 0));
            scalar_prods3(&start.p_sum, &end.p_sum, &end.p, &end.v, &start.v)
        };

        (turn1 < 0.) | (turn2 < 0.)
    }

    fn write_position(&self, out: &mut [f64]) {
        out.copy_from_slice(&self.q)
    }

    fn energy(&self) -> f64 {
        self.kinetic_energy + self.potential_energy
    }

    fn index_in_trajectory(&self) -> i64 {
        self.idx_in_trajectory
    }

    fn make_init_point(&mut self) {
        let inner = self.try_mut_inner().unwrap();
        inner.idx_in_trajectory = 0;
        inner.p_sum.copy_from_slice(&inner.p);
    }

    fn potential_energy(&self) -> f64 {
        self.potential_energy
    }
}

impl State {
    pub(crate) fn new(pool: &mut StatePool, init: &[f64]) -> Self {
        let mut state = pool.new_state();

        let inner = state.try_mut_inner().expect("State already in use");
        for (i, val) in inner.q.iter_mut().enumerate() {
            *val = init[i];
        }

        for val in inner.p_sum.iter_mut() {
            *val = 0.;
        }

        inner.idx_in_trajectory = 0;

        state
    }

    pub(crate) fn first_momentum_halfstep(&self, out: &mut Self, epsilon: f64) {
        axpy_out(
            &self.grad,
            &self.p,
            epsilon / 2.,
            &mut out.try_mut_inner().expect("State already in use").p,
        );
    }

    pub(crate) fn position_step(&self, out: &mut Self, epsilon: f64) {
        let out = out.try_mut_inner().expect("State already in use");
        axpy_out(&out.v, &self.q, epsilon, &mut out.q);
    }

    pub(crate) fn second_momentum_halfstep(&mut self, epsilon: f64) {
        let inner = self.try_mut_inner().expect("State already in use");
        axpy(&inner.grad, &mut inner.p, epsilon / 2.);
    }

    pub(crate) fn set_psum(&self, target: &mut Self, _dir: crate::nuts::Direction) {
        let out = target.try_mut_inner().expect("State already in use");

        assert!(out.idx_in_trajectory != 0);

        if out.idx_in_trajectory == -1 {
            out.p_sum.copy_from_slice(&out.p);
        } else {
            axpy_out(&out.p, &self.p_sum, 1., &mut out.p_sum);
        }
    }

    pub(crate) fn index_in_trajectory(&self) -> i64 {
        self.idx_in_trajectory
    }

    pub(crate) fn index_in_trajectory_mut(&mut self) -> &mut i64 {
        &mut self
            .try_mut_inner()
            .expect("State already in use")
            .idx_in_trajectory
    }

    pub(crate) fn new_empty(pool: &mut StatePool) -> Self {
        pool.new_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_pool() {
        let mut pool = StatePool::new(10);
        let mut state = pool.new_state();
        assert!(state.p.len() == 10);
        state.try_mut_inner().unwrap();

        let mut state2 = state.clone();
        assert!(state.try_mut_inner().is_err());
        assert!(state2.try_mut_inner().is_err());
    }
}
