//use intrusive_collections::intrusive_adapter;
//use intrusive_collections::{LinkedListLink, LinkedList};
use std::rc::{Rc, Weak};
use std::ops::Deref;
use std::cell::RefCell;

use crate::nuts::{Integrator, LeapfrogInfo, Direction, DivergenceInfo, SampleInfo};

pub struct InnerState {
    p: Box<[f64]>,
    pub q: Box<[f64]>,
    v: Box<[f64]>,
    p_sum: Box<[f64]>,
    pub grad: Box<[f64]>,
    idx_in_trajectory: i64,
    kinetic_energy: f64,
    potential_energy: f64,
    reuser: Option<Weak<dyn ReuseState>>,
}


impl InnerState {
    fn new(size: usize) -> InnerState {
        InnerState {
            p: vec![0.; size].into(),
            q: vec![0.; size].into(),
            v: vec![0.; size].into(),
            p_sum: vec![0.; size].into(),
            grad: vec![0.; size].into(),
            idx_in_trajectory: 0,
            kinetic_energy: 0.,
            potential_energy: 0.,
            reuser: None,
        }
    }
}


pub struct State {
    //inner: Option<Rc<InnerState>>,
    inner: std::mem::ManuallyDrop<Rc<InnerState>>,
}

/*
impl Drop for State {
    fn drop(&mut self) {
        if self.inner.is_none() {
            return
        }
        if let Some(inner) = self.inner.take() {
            if let Ok(mut inner) = Rc::try_unwrap(inner) {
                if let Some(reuser) = inner.reuser.take() {
                    if let Some(reuser) = reuser.upgrade() {
                        reuser.reuse_state(inner)
                    }
                }
            }
        }
    }
}
*/

impl Drop for State {
    fn drop(&mut self) {
        let inner = unsafe { std::mem::ManuallyDrop::into_inner(std::ptr::read(&self.inner)) };
        if let Ok(mut inner) = Rc::try_unwrap(inner) {
            if let Some(reuser) = inner.reuser.take() {
                if let Some(reuser) = reuser.upgrade() {
                    reuser.reuse_state(inner);
                }
            }
        }
    }
}


/*
impl PartialEq for State {
    fn eq(&self, other: &State) -> bool {
        if let (Some(a), Some(b)) = (self.inner.as_ref(), other.inner.as_ref()) {
            Rc::ptr_eq(&a, &b)
        } else {
            false
        }
    }
}
*/


impl Clone for State {
    fn clone(&self) -> Self {
        State {
            inner: self.inner.clone(),
        }
    }
}


impl Deref for State {
    type Target = InnerState;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
        //self.inner.as_ref().unwrap()
    }
}


trait ReuseState {
    fn reuse_state(&self, state: InnerState);
}

pub trait LogpFunc {
    fn logp(&self, state: &mut InnerState) -> f64;
    fn dim(&self) -> usize;
}


#[derive(Debug)]
pub struct DivergenceInfoImpl {
    
}

impl DivergenceInfo for DivergenceInfoImpl { }

#[derive(Debug)]
pub struct LeapfrogInfoImpl {
    energy_error: f64,
    divergence: Option<DivergenceInfoImpl>,
}

impl LeapfrogInfo for LeapfrogInfoImpl {
    type DivergenceInfo = DivergenceInfoImpl;

    fn energy_error(&self) -> f64 {
        self.energy_error
    }

    fn diverging(&self) -> bool {
        self.divergence.is_some()
    }

    fn divergence(self) -> Option<Self::DivergenceInfo> {
        self.divergence
    }
}


struct StateStorage {
    free_states: RefCell<Vec<InnerState>>,
}


impl StateStorage {
    fn with_capacity(capacity: usize) -> StateStorage {
        StateStorage {
            free_states: RefCell::new(Vec::with_capacity(capacity))
        }
    }
}

impl ReuseState for StateStorage {
    fn reuse_state(&self, state: InnerState) {
        self.free_states.borrow_mut().push(state)
    }
}

fn new_state(this: &mut Rc<StateStorage>, dim: usize) -> State {
    let mut inner = match this.free_states.borrow_mut().pop() {
        Some(inner) => {
            if dim != inner.q.len() {
                panic!("dim mismatch");
            }
            inner
        },
        None => InnerState::new(dim),
    };
    let reuser = Rc::downgrade(&this.clone());
    inner.reuser = Some(reuser);
    State {
        //inner: Some(Rc::new(inner))
        inner: std::mem::ManuallyDrop::new(Rc::new(inner))
    }
}

pub struct StaticIntegrator<F: LogpFunc> {
    states: Rc<StateStorage>,
    logp: F,
    initial_state: State,
}


impl<F: LogpFunc> Integrator for StaticIntegrator<F> {
    type State = State;
    type LeapfrogInfo = LeapfrogInfoImpl;

    fn initial_state(&self) -> State {
        self.initial_state.clone()
    }

    fn leapfrog(&mut self, start: &Self::State, dir: Direction) -> (Self::State, Self::LeapfrogInfo) {
        use crate::math::{axpy, norm};

        let mut out_state = new_state(&mut self.states, self.logp.dim());
        //let out_inner = out_state.inner.as_mut().expect("Use after free");
        //let out = Rc::get_mut(out_inner).expect("State already in use");
        let out = Rc::get_mut(&mut out_state.inner).expect("State already in use");

        let sign = match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };

        let epsilon = (sign as f64) * 1e-6f64;

        out.q.copy_from_slice(&start.q);
        out.p.copy_from_slice(&start.p);

        let dt = epsilon / 2.;

        axpy(&start.grad, &mut out.p, dt);

        // velocity
        out.v.copy_from_slice(&out.p);

        axpy(&out.v, &mut out.q, epsilon);

        let logp = self.logp.logp(out);

        axpy(&out.grad, &mut out.p, dt);

        // velocity
        out.v.copy_from_slice(&out.p);

        // kinetic energy
        out.kinetic_energy = norm(&out.p);
        out.potential_energy = -logp;

        match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };
        match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };
        out.idx_in_trajectory = sign + start.idx_in_trajectory;

        out.p_sum.copy_from_slice(&start.p_sum);
        axpy(&out.p, &mut out.p_sum, sign as f64);

        let info = LeapfrogInfoImpl {
            energy_error: 0.,
            divergence: None,
        };
        (out_state, info)
    }

    fn is_turning(&self, start: &Self::State, end: &Self::State) -> bool {
        use crate::math::scalar_prods_of_diff;

        let (start, end) = if start.idx_in_trajectory < end.idx_in_trajectory {
            (start, end)
        } else {
            (end, start)
        };

        let (a, b) = scalar_prods_of_diff(&end.p_sum, &start.p_sum, &end.v, &start.v);

        //dbg!(&start.p_sum);
        //dbg!(a, b);
        //dbg!(start.idx_in_trajectory, end.idx_in_trajectory);
        (a < 0.) | (b < 0.)
    }

    fn accept(&mut self, state: Self::State, _info: SampleInfo<Self>) {}

    fn write_position(&self, state: &Self::State, out: &mut [f64]) {
        out.copy_from_slice(&state.q);
    }
}


impl<F: LogpFunc> StaticIntegrator<F> {
    pub fn new(func: F, init: &[f64]) -> StaticIntegrator<F> {
        let dim = init.len();
        assert!(dim == func.dim());
        let mut states = Rc::new(StateStorage::with_capacity(100));
        let mut state = new_state(&mut states, dim);

        //let state_inner = state.inner.as_mut().expect("Use after free");
        //let out = Rc::get_mut(state_inner).expect("State already in use");
        let out = Rc::get_mut(&mut state.inner).expect("State already in use");

        for (i, val) in out.q.iter_mut().enumerate() {
            *val = init[i];
        }

        StaticIntegrator {
            logp: func,
            states,
            initial_state: state,
        }
    }

    pub fn set_initial<R: rand::Rng + ?Sized>(&mut self, rng: &mut R, mut state: State) {
        //let state_inner = state.inner.as_mut().expect("Use after free");
        //let out = Rc::get_mut(state_inner).expect("State already in use");
        let out = Rc::get_mut(&mut state.inner).expect("State already in use");

        let dist = rand::distributions::StandardNormal;
        for val in out.p.iter_mut() {
            *val = rng.sample(dist);
        }

        out.p_sum.copy_from_slice(&out.p);

        self.initial_state = state;
    }

    pub fn randomize_initial<R: rand::Rng>(&mut self, rng: &mut R) {
        //let state_inner = self.initial_state.inner.as_mut().expect("Use after free");
        //let out = Rc::get_mut(state_inner).expect("State already in use");
        let out = Rc::get_mut(&mut self.initial_state.inner).expect("State already in use");

        let dist = rand::distributions::StandardNormal;
        for val in out.p.iter_mut() {
            *val = rng.sample(dist);
        }
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn make_state() {
        let _ = InnerState::new(10);
        let mut storage = Rc::new(StateStorage::with_capacity(0));
        let a = new_state(&mut storage, 10);
        assert!(storage.free_states.borrow_mut().len() == 0);
        drop(a);
        assert!(storage.free_states.borrow_mut().len() == 1);
        let a = new_state(&mut storage, 10);
        assert!(storage.free_states.borrow_mut().len() == 0);
        drop(a);
        assert!(storage.free_states.borrow_mut().len() == 1);
    }

    #[test]
    fn sample_normal() {
        struct NormalLogp { dim: usize, mu: f64 };

        impl LogpFunc for NormalLogp {
            fn dim(&self) -> usize {
                self.dim
            }
            fn logp(&self, state: &mut InnerState) -> f64 {
                let position = &state.p;
                let grad = &mut state.grad;
                let n = position.len();
                assert!(grad.len() == n);
                let mut logp = 0f64;
                for i in 0..n {
                    let val = position[i] - self.mu;
                    logp -= val * val;
                    grad[i] = -val;
                }
                logp
            }
        }

        let func = NormalLogp { dim: 10, mu: 3. };
        let init = vec![3.5; func.dim];
        let mut integrator = StaticIntegrator::new(func, &init);

        let mut rng = rand::thread_rng();
        
        integrator.randomize_initial(&mut rng);
        let (out, info) = crate::nuts::draw(&mut rng, &mut integrator, 20);
        dbg!(info);
        dbg!(&out.q);
    }
}
