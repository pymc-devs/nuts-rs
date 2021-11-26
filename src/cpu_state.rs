use std::{
    cell::RefCell,
    ops::Deref,
    rc::{Rc, Weak},
};

struct StateStorage {
    free_states: RefCell<Vec<Rc<InnerState>>>,
}

impl StateStorage {
    fn with_capacity(capacity: usize) -> StateStorage {
        StateStorage {
            free_states: RefCell::new(Vec::with_capacity(capacity)),
        }
    }
}

impl ReuseState for StateStorage {
    fn reuse_state(&self, state: Rc<InnerState>) {
        self.free_states.borrow_mut().push(state)
    }
}

pub(crate) struct StatePool {
    storage: Rc<StateStorage>,
    dim: usize,
}

impl StatePool {
    pub(crate) fn with_capacity(dim: usize, capacity: usize) -> StatePool {
        StatePool {
            storage: Rc::new(StateStorage::with_capacity(100)),
            dim,
        }
    }

    pub(crate) fn new_state(&mut self) -> State {
        let inner = match self.storage.free_states.borrow_mut().pop() {
            Some(inner) => {
                if self.dim != inner.q.len() {
                    panic!("dim mismatch");
                }
                inner
            }
            None => {
                let owner: Rc<dyn ReuseState> = self.storage.clone();
                Rc::new(InnerState::new(self.dim, &owner))
            }
        };
        State {
            inner: std::mem::ManuallyDrop::new(inner),
        }
    }
}

trait ReuseState {
    fn reuse_state(&self, state: Rc<InnerState>);
}

pub(crate) struct InnerState {
    pub(crate) p: Box<[f64]>,
    pub(crate) q: Box<[f64]>,
    pub(crate) v: Box<[f64]>,
    pub(crate) p_sum: Box<[f64]>,
    pub(crate) grad: Box<[f64]>,
    pub(crate) idx_in_trajectory: i64,
    pub(crate) kinetic_energy: f64,
    pub(crate) potential_energy: f64,
    reuser: Weak<dyn ReuseState>,
}

impl InnerState {
    fn new(size: usize, owner: &Rc<dyn ReuseState>) -> InnerState {
        InnerState {
            p: vec![0.; size].into(),
            q: vec![0.; size].into(),
            v: vec![0.; size].into(),
            p_sum: vec![0.; size].into(),
            grad: vec![0.; size].into(),
            idx_in_trajectory: 0,
            kinetic_energy: 0.,
            potential_energy: 0.,
            reuser: Rc::downgrade(owner),
        }
    }
}

pub(crate) struct State {
    inner: std::mem::ManuallyDrop<Rc<InnerState>>,
}

impl Deref for State {
    type Target = InnerState;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

#[derive(Debug)]
pub(crate) struct StateInUse {}

type Result<T> = std::result::Result<T, StateInUse>;

impl State {
    pub(crate) fn try_mut_inner(&mut self) -> Result<&mut InnerState> {
        match Rc::get_mut(&mut self.inner) {
            Some(val) => Ok(val),
            None => Err(StateInUse {}),
        }
    }

    pub(crate) fn get_position(&self) -> &[f64] {
        &self.inner.q[..]
    }

    pub(crate) fn get_mut_position(&mut self) -> Result<&mut [f64]> {
        self.try_mut_inner().map(|inner| &mut inner.q[..])
    }

    pub(crate) fn get_gradient(&self) -> &[f64] {
        &self.inner.grad[..]
    }

    pub(crate) fn get_mut_gradient(&mut self) -> Result<&mut [f64]> {
        self.try_mut_inner().map(|inner| &mut inner.grad[..])
    }

    pub(crate) fn get_momentum(&self) -> &[f64] {
        &self.inner.p
    }

    pub(crate) fn get_mut_momentum(&mut self) -> Result<&mut [f64]> {
        self.try_mut_inner().map(|inner| &mut inner.p[..])
    }

    pub(crate) fn get_velocity(&self) -> &[f64] {
        &self.inner.v
    }

    pub(crate) fn get_mut_velocity(&mut self) -> Result<&mut [f64]> {
        self.try_mut_inner().map(|inner| &mut inner.v[..])
    }

    pub(crate) fn potential_energy(&self) -> f64 {
        self.inner.potential_energy
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

    #[inline]
    fn is_turning(&self, other: &Self) -> bool {
        use crate::math::scalar_prods_of_diff;

        let (start, end) = if self.idx_in_trajectory < other.idx_in_trajectory {
            (&*self, other)
        } else {
            (other, &*self)
        };

        let (a, b) = scalar_prods_of_diff(&end.p_sum, &start.p_sum, &end.v, &start.v);
        (a < 0.) | (b < 0.)
    }

    fn write_position(&self, out: &mut [f64]) {
        out.copy_from_slice(&self.q);
    }

    fn new(pool: &mut Self::Pool, init: &[f64], init_grad: &[f64]) -> Self {
        let mut state = pool.new_state();

        let inner = state.try_mut_inner().expect("State already in use");
        for (i, val) in inner.q.iter_mut().enumerate() {
            *val = init[i];
        }

        for (i, val) in inner.grad.iter_mut().enumerate() {
            *val = init_grad[i];
        }

        for val in inner.p_sum.iter_mut() {
            *val = 0.;
        }

        todo!(); // check initialization
    }

    fn deep_clone(&self, pool: &mut Self::Pool) -> Self {
        todo!()
    }

    fn energy(&self) -> f64 {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_pool() {
        let mut pool = StatePool::with_capacity(10, 20);
        let mut state = pool.new_state();
        assert!(state.p.len() == 10);
        state.try_mut_inner().unwrap();
    }
}
