use std::{
    cell::RefCell,
    fmt::Debug,
    rc::{Rc, Weak},
};

use crate::{hamiltonian::Point, math_base::Math};

struct StateStorage<M: Math, P: Point<M>> {
    free_states: RefCell<Vec<Rc<InnerStateReusable<M, P>>>>,
}

impl<M: Math, P: Point<M>> StateStorage<M, P> {
    fn new(_math: &mut M, capacity: usize) -> StateStorage<M, P> {
        StateStorage {
            free_states: RefCell::new(Vec::with_capacity(capacity)),
        }
    }
}

pub struct StatePool<M: Math, P: Point<M>> {
    storage: Rc<StateStorage<M, P>>,
}

impl<M: Math, P: Point<M>> StatePool<M, P> {
    pub fn new(math: &mut M, capacity: usize) -> StatePool<M, P> {
        StatePool {
            storage: Rc::new(StateStorage::new(math, capacity)),
        }
    }

    pub fn new_state(&self, math: &mut M) -> State<M, P> {
        let inner = match self.storage.free_states.borrow_mut().pop() {
            Some(inner) => inner,
            None => Rc::new(InnerStateReusable::new(math, self)),
        };
        State {
            inner: std::mem::ManuallyDrop::new(inner),
        }
    }

    pub fn copy_state(&self, math: &mut M, state: &State<M, P>) -> State<M, P> {
        let mut new_state = self.new_state(math);
        let new_point = new_state
            .try_point_mut()
            .expect("New state should not have references");
        state.point().copy_into(math, new_point);
        new_state
    }
}

pub(crate) struct InnerStateReusable<M: Math, P: Point<M>> {
    inner: P,
    reuser: Weak<StateStorage<M, P>>,
}

impl<M: Math, P: Point<M>> InnerStateReusable<M, P> {
    fn new(math: &mut M, owner: &StatePool<M, P>) -> InnerStateReusable<M, P> {
        InnerStateReusable {
            inner: P::new(math),
            reuser: Rc::downgrade(&Rc::clone(&owner.storage)),
        }
    }
}

pub struct State<M: Math, P: Point<M>> {
    inner: std::mem::ManuallyDrop<Rc<InnerStateReusable<M, P>>>,
}

#[derive(Debug)]
pub struct StateInUse {}

type Result<T> = std::result::Result<T, StateInUse>;

impl<M: Math, P: Point<M>> State<M, P> {
    pub fn point(&self) -> &P {
        &self.inner.inner
    }

    pub fn try_point_mut(&mut self) -> Result<&mut P> {
        match Rc::get_mut(&mut self.inner) {
            Some(val) => Ok(&mut val.inner),
            None => Err(StateInUse {}),
        }
    }

    pub fn index_in_trajectory(&self) -> i64 {
        self.inner.inner.index_in_trajectory()
    }

    pub fn write_position(&self, math: &mut M, out: &mut [f64]) {
        math.write_to_slice(self.point().position(), out)
    }

    pub fn write_gradient(&self, math: &mut M, out: &mut [f64]) {
        math.write_to_slice(self.point().gradient(), out)
    }

    pub fn energy(&self) -> f64 {
        self.point().energy()
    }
}

impl<M: Math, P: Point<M>> Drop for State<M, P> {
    fn drop(&mut self) {
        let rc = unsafe { std::mem::ManuallyDrop::take(&mut self.inner) };
        if (Rc::strong_count(&rc) == 1) & (Rc::weak_count(&rc) == 0)
            && let Some(storage) = rc.reuser.upgrade() {
                storage.free_states.borrow_mut().push(rc);
            }
    }
}

impl<M: Math, P: Point<M>> Clone for State<M, P> {
    fn clone(&self) -> Self {
        State {
            inner: self.inner.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cpu_math::CpuMath, euclidean_hamiltonian::EuclideanPoint, sampler::test_logps::NormalLogp,
    };

    use super::*;

    #[test]
    fn crate_pool() {
        let logp = NormalLogp { dim: 10, mu: 0.2 };
        let mut math = CpuMath::new(&logp);
        let pool: StatePool<_, EuclideanPoint<_>> = StatePool::new(&mut math, 10);
        let mut state = pool.new_state(&mut math);
        state.try_point_mut().unwrap();

        let mut state2 = state.clone();
        assert!(state.try_point_mut().is_err());
        assert!(state2.try_point_mut().is_err());
    }

    #[test]
    fn make_state() {
        let dim = 10;
        let logp = NormalLogp { dim, mu: 0.2 };
        let mut math = CpuMath::new(&logp);
        let pool: StatePool<_, EuclideanPoint<_>> = StatePool::new(&mut math, 10);
        let a = pool.new_state(&mut math);
        assert_eq!(a.index_in_trajectory(), 0);
    }
}
