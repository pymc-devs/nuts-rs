//use intrusive_collections::intrusive_adapter;
//use intrusive_collections::{LinkedListLink, LinkedList};
use std::rc::{Rc, Weak};
use std::ops::Deref;
use std::cell::RefCell;
use std::fmt::Debug;

use crate::nuts::{Integrator, LeapfrogInfo, Direction, DivergenceInfo, SampleInfo};

pub struct State {
    p: Box<[f64]>,
    pub q: Box<[f64]>,
    v: Box<[f64]>,
    p_sum: Box<[f64]>,
    pub grad: Box<[f64]>,
    idx_in_trajectory: i64,
    kinetic_energy: f64,
    pub potential_energy: f64,
    reuser: Option<Weak<dyn ReuseState>>,
}


impl State {
    fn new(size: usize) -> State {
        State {
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


pub struct StateWrapper {
    inner: std::mem::ManuallyDrop<Rc<State>>,
}


impl Drop for StateWrapper {
    fn drop(&mut self) {
        let inner = unsafe { std::mem::ManuallyDrop::into_inner(std::ptr::read(&self.inner)) };
        if let Ok(mut inner) = Rc::try_unwrap(inner) {
            if let Some(reuser) = inner.reuser.take() {
                if let Some(reuser) = reuser.upgrade() {
                    reuser.reuse_state(Rc::new(inner));
                }
            }
        }
    }
}


impl Clone for StateWrapper {
    fn clone(&self) -> Self {
        StateWrapper {
            inner: self.inner.clone(),
        }
    }
}


impl Deref for StateWrapper {
    type Target = State;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}


trait ReuseState {
    fn reuse_state(&self, state: Rc<State>);
}

pub trait LogpFunc {
    type Err: Debug;

    fn logp(&self, state: &mut State) -> Result<(), Self::Err>;
    fn dim(&self) -> usize;
}


#[derive(Debug)]
pub struct DivergenceInfoImpl<E> {
    pub logp_error: Option<E>,
}

impl<E: Debug> DivergenceInfo for DivergenceInfoImpl<E> { }

#[derive(Debug)]
pub struct LeapfrogInfoImpl {
    energy_error: f64,
}

impl LeapfrogInfo for LeapfrogInfoImpl {
    fn energy_error(&self) -> f64 {
        self.energy_error
    }
}


struct StateStorage {
    free_states: RefCell<Vec<Rc<State>>>,
}


impl StateStorage {
    fn with_capacity(capacity: usize) -> StateStorage {
        StateStorage {
            free_states: RefCell::new(Vec::with_capacity(capacity))
        }
    }
}

impl ReuseState for StateStorage {
    fn reuse_state(&self, state: Rc<State>) {
        self.free_states.borrow_mut().push(state)
    }
}

fn new_state(this: &mut Rc<StateStorage>, dim: usize) -> StateWrapper {
    let mut inner = match this.free_states.borrow_mut().pop() {
        Some(inner) => {
            if dim != inner.q.len() {
                panic!("dim mismatch");
            }
            inner
        },
        None => Rc::new(State::new(dim)),
    };
    let reuser = Rc::downgrade(&this.clone());
    Rc::get_mut(&mut inner).unwrap().reuser = Some(reuser);
    StateWrapper {
        //inner: Some(Rc::new(inner))
        inner: std::mem::ManuallyDrop::new(inner)
    }
}

pub struct StaticIntegrator<F: LogpFunc> {
    states: Rc<StateStorage>,
    logp: F,
}


impl<F: LogpFunc> Integrator for StaticIntegrator<F> {
    type State = StateWrapper;
    type LeapfrogInfo = LeapfrogInfoImpl;
    type DivergenceInfo = DivergenceInfoImpl<F::Err>;

    fn leapfrog(&mut self, start: &Self::State, dir: Direction) -> Result<(Self::State, Self::LeapfrogInfo), Self::DivergenceInfo> {
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

        if let Err(error) = self.logp.logp(out) {
            return Err(DivergenceInfoImpl { logp_error: Some(error) });
        }

        axpy(&out.grad, &mut out.p, dt);

        // velocity
        out.v.copy_from_slice(&out.p);

        // kinetic energy
        out.kinetic_energy = norm(&out.p);

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
        };
        Ok((out_state, info))
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

    fn write_position(&self, state: &Self::State, out: &mut [f64]) {
        out.copy_from_slice(&state.q);
    }

    fn randomize_velocity<R: rand::Rng + ?Sized>(&mut self, state: &mut Self::State, rng: &mut R) {
        //let state_inner = self.initial_state.inner.as_mut().expect("Use after free");
        //let out = Rc::get_mut(state_inner).expect("State already in use");
        let out = Rc::get_mut(&mut state.inner).expect("State already in use");

        let dist = rand_distr::StandardNormal;
        for val in out.p.iter_mut() {
            *val = rng.sample(dist);
        }
    }

    fn new_state(&mut self, init: &[f64]) -> Result<Self::State, Self::DivergenceInfo> {
        let mut state = new_state(&mut self.states, self.logp.dim());
        
        let inner = Rc::get_mut(&mut state.inner).expect("State already in use");
        for (i, val) in inner.q.iter_mut().enumerate() {
            *val = init[i];
        }
        
        for val in inner.p_sum.iter_mut() {
            *val = 0.;
        }
        if let Err(error) = self.logp.logp(inner) {
            return Err(DivergenceInfoImpl { logp_error: Some(error) })
        }

        Ok(state)
    }
}


impl<F: LogpFunc> StaticIntegrator<F> {
    pub fn new(func: F, dim: usize) -> StaticIntegrator<F> {
        assert!(dim == func.dim());
        let states = Rc::new(StateStorage::with_capacity(100));

        let mut integrator = StaticIntegrator {
            logp: func,
            states,
        };

        for _ in 0..20 {
            let _ = new_state(&mut integrator.states, integrator.logp.dim());
        };
        integrator
    }
}


pub mod test_logps {
    use super::{LogpFunc, State};

    pub struct NormalLogp { dim: usize, mu: f64 }

    impl NormalLogp {
        pub fn new(dim: usize, mu: f64) -> NormalLogp {
            NormalLogp { dim, mu }
        }
    }

    impl LogpFunc for NormalLogp {
        type Err = ();

        fn dim(&self) -> usize {
            self.dim
        }
        fn logp(&self, state: &mut State) -> Result<(), ()> {
            let position = &state.q;
            let grad = &mut state.grad;
            let n = position.len();
            assert!(grad.len() == n);
            let mut logp = 0f64;
            for i in 0..n {
                let val = position[i] - self.mu;
                logp -= val * val;
                grad[i] = -val;
            }
            state.potential_energy = -logp;
            Ok(())
        }
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use super::test_logps::*;
    use rand::SeedableRng;
    use pretty_assertions::assert_eq;
    
    #[test]
    fn make_state() {
        let _ = State::new(10);
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
    fn deterministic() {
        let dim = 3usize;
        let func = NormalLogp::new(dim, 3.);
        let init = vec![3.5; dim];

        let mut integrator = StaticIntegrator::new(func, dim);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let state = integrator.new_state(&init).unwrap();
        let (out, info) = crate::nuts::draw(state, &mut rng, &mut integrator, 10);


        let state = integrator.new_state(&init).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let (out2, info2) = crate::nuts::draw(state, &mut rng, &mut integrator, 10);

        let mut vals1 = vec![0.; dim];
        let mut vals2 = vec![0.; dim];

        integrator.write_position(&out, &mut vals1);
        integrator.write_position(&out2, &mut vals2);

        dbg!(info);
        dbg!(info2);

        assert_eq!(vals1, vals2);
    }
}
