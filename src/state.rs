use std::{
    cell::RefCell,
    fmt::Debug,
    ops::Deref,
    rc::{Rc, Weak},
};

use crate::math_base::Math;

struct StateStorage<M: Math> {
    free_states: RefCell<Vec<Rc<InnerStateReusable<M>>>>,
}

impl<M: Math> StateStorage<M> {
    fn new(_math: &mut M, capacity: usize) -> StateStorage<M> {
        StateStorage {
            free_states: RefCell::new(Vec::with_capacity(capacity)),
        }
    }
}

pub struct StatePool<M: Math> {
    storage: Rc<StateStorage<M>>,
}

impl<M: Math> StatePool<M> {
    pub fn new(math: &mut M, capacity: usize) -> StatePool<M> {
        StatePool {
            storage: Rc::new(StateStorage::new(math, capacity)),
        }
    }

    pub fn new_state(&self, math: &mut M) -> State<M> {
        let inner = match self.storage.free_states.borrow_mut().pop() {
            Some(inner) => inner,
            None => Rc::new(InnerStateReusable::new(math, self)),
        };
        State {
            inner: std::mem::ManuallyDrop::new(inner),
        }
    }

    pub fn copy_state(&self, math: &mut M, state: &State<M>) -> State<M> {
        let mut new_state = self.new_state(math);

        let InnerState {
            q,
            p,
            p_sum,
            grad,
            v,
            idx_in_trajectory,
            kinetic_energy,
            potential_energy,
        } = new_state
            .try_mut_inner()
            .expect("New state should not have references");

        math.copy_into(&state.q, q);
        math.copy_into(&state.p, p);
        math.copy_into(&state.p_sum, p_sum);
        math.copy_into(&state.grad, grad);
        math.copy_into(&state.v, v);
        *idx_in_trajectory = state.idx_in_trajectory;
        *kinetic_energy = state.kinetic_energy;
        *potential_energy = state.potential_energy;

        new_state
    }
}

#[derive(Debug, Clone)]
pub struct InnerState<M: Math> {
    pub(crate) p: M::Vector,
    pub(crate) q: M::Vector,
    pub(crate) v: M::Vector,
    pub(crate) p_sum: M::Vector,
    pub(crate) grad: M::Vector,
    pub(crate) idx_in_trajectory: i64,
    pub(crate) kinetic_energy: f64,
    pub(crate) potential_energy: f64,
}

pub(crate) struct InnerStateReusable<M: Math> {
    inner: InnerState<M>,
    reuser: Weak<StateStorage<M>>,
}

impl<M: Math> InnerStateReusable<M> {
    fn new(math: &mut M, owner: &StatePool<M>) -> InnerStateReusable<M> {
        InnerStateReusable {
            inner: InnerState {
                p: math.new_array(),
                q: math.new_array(),
                v: math.new_array(),
                p_sum: math.new_array(),
                grad: math.new_array(),
                idx_in_trajectory: 0,
                kinetic_energy: 0.,
                potential_energy: 0.,
            },
            reuser: Rc::downgrade(&Rc::clone(&owner.storage)),
        }
    }
}

pub struct State<M: Math> {
    inner: std::mem::ManuallyDrop<Rc<InnerStateReusable<M>>>,
}

impl<M: Math> Deref for State<M> {
    type Target = InnerState<M>;

    fn deref(&self) -> &Self::Target {
        &self.inner.inner
    }
}

#[derive(Debug)]
pub struct StateInUse {}

type Result<T> = std::result::Result<T, StateInUse>;

impl<M: Math> State<M> {
    pub(crate) fn try_mut_inner(&mut self) -> Result<&mut InnerState<M>> {
        match Rc::get_mut(&mut self.inner) {
            Some(val) => Ok(&mut val.inner),
            None => Err(StateInUse {}),
        }
    }
}

impl<M: Math> Drop for State<M> {
    fn drop(&mut self) {
        let rc = unsafe { std::mem::ManuallyDrop::take(&mut self.inner) };
        if (Rc::strong_count(&rc) == 1) & (Rc::weak_count(&rc) == 0) {
            if let Some(storage) = rc.reuser.upgrade() {
                storage.free_states.borrow_mut().push(rc);
            }
        }
    }
}

impl<M: Math> Clone for State<M> {
    fn clone(&self) -> Self {
        State {
            inner: self.inner.clone(),
        }
    }
}

impl<M: Math> State<M> {
    pub(crate) fn is_turning(&self, math: &mut M, other: &Self) -> bool {
        let (start, end) = if self.idx_in_trajectory < other.idx_in_trajectory {
            (self, other)
        } else {
            (other, self)
        };

        let a = start.idx_in_trajectory;
        let b = end.idx_in_trajectory;

        assert!(a < b);
        let (turn1, turn2) = if (a >= 0) & (b >= 0) {
            math.scalar_prods3(&end.p_sum, &start.p_sum, &start.p, &end.v, &start.v)
        } else if (b >= 0) & (a < 0) {
            math.scalar_prods2(&end.p_sum, &start.p_sum, &end.v, &start.v)
        } else {
            assert!((a < 0) & (b < 0));
            math.scalar_prods3(&start.p_sum, &end.p_sum, &end.p, &end.v, &start.v)
        };

        (turn1 < 0.) | (turn2 < 0.)
    }

    pub(crate) fn write_position(&self, math: &mut M, out: &mut [f64]) {
        math.write_to_slice(&self.q, out)
    }

    pub(crate) fn write_gradient(&self, math: &mut M, out: &mut [f64]) {
        math.write_to_slice(&self.grad, out)
    }

    pub(crate) fn energy(&self) -> f64 {
        self.kinetic_energy + self.potential_energy
    }

    pub(crate) fn index_in_trajectory(&self) -> i64 {
        self.idx_in_trajectory
    }

    pub(crate) fn make_init_point(&mut self, math: &mut M) {
        let inner = self.try_mut_inner().unwrap();
        inner.idx_in_trajectory = 0;
        math.copy_into(&inner.p, &mut inner.p_sum);
    }

    pub(crate) fn potential_energy(&self) -> f64 {
        self.potential_energy
    }

    pub(crate) fn first_momentum_halfstep(&self, math: &mut M, out: &mut Self, epsilon: f64) {
        math.axpy_out(
            &self.grad,
            &self.p,
            epsilon / 2.,
            &mut out.try_mut_inner().expect("State already in use").p,
        );
    }

    pub(crate) fn position_step(&self, math: &mut M, out: &mut Self, epsilon: f64) {
        let out = out.try_mut_inner().expect("State already in use");
        math.axpy_out(&out.v, &self.q, epsilon, &mut out.q);
    }

    pub(crate) fn second_momentum_halfstep(&mut self, math: &mut M, epsilon: f64) {
        let inner = self.try_mut_inner().expect("State already in use");
        math.axpy(&inner.grad, &mut inner.p, epsilon / 2.);
    }

    pub(crate) fn set_psum(&self, math: &mut M, target: &mut Self, _dir: crate::nuts::Direction) {
        let out = target.try_mut_inner().expect("State already in use");

        assert!(out.idx_in_trajectory != 0);

        if out.idx_in_trajectory == -1 {
            math.copy_into(&out.p, &mut out.p_sum);
        } else {
            math.axpy_out(&out.p, &self.p_sum, 1., &mut out.p_sum);
        }
    }

    pub(crate) fn index_in_trajectory_mut(&mut self) -> &mut i64 {
        &mut self
            .try_mut_inner()
            .expect("State already in use")
            .idx_in_trajectory
    }
}

#[cfg(test)]
mod tests {
    use crate::{cpu_math::CpuMath, sampler::test_logps::NormalLogp};

    use super::*;

    #[test]
    fn crate_pool() {
        let logp = NormalLogp { dim: 10, mu: 0.2 };
        let mut math = CpuMath::new(&logp);
        let pool = StatePool::new(&mut math, 10);
        let mut state = pool.new_state(&mut math);
        assert!(state.p.nrows() == 10);
        assert!(state.p.ncols() == 1);
        state.try_mut_inner().unwrap();

        let mut state2 = state.clone();
        assert!(state.try_mut_inner().is_err());
        assert!(state2.try_mut_inner().is_err());
    }

    #[test]
    fn make_state() {
        let dim = 10;
        let logp = NormalLogp { dim, mu: 0.2 };
        let mut math = CpuMath::new(&logp);
        let pool = StatePool::new(&mut math, 10);
        let a = pool.new_state(&mut math);

        assert_eq!(a.idx_in_trajectory, 0);
        assert!(a.p_sum.col_as_slice(0).iter().all(|&x| x == 0f64));
        assert_eq!(a.p_sum.col_as_slice(0).len(), dim);
        assert_eq!(a.grad.col_as_slice(0).len(), dim);
        assert_eq!(a.q.col_as_slice(0).len(), dim);
        assert_eq!(a.p.col_as_slice(0).len(), dim);
    }
}
