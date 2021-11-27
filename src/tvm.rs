use std::rc::{Rc, Weak};
use std::ops::Deref;
use std::cell::RefCell;
use std::fmt::Debug;

use tvm::NDArray;
use tvm::runtime::graph_rt::GraphRt;
use crate::nuts::{Integrator, LeapfrogInfo, Direction, DivergenceInfo, SampleInfo};

pub struct State {
    p: NDArray,
    pub q: NDArray,
    v: NDArray,
    p_sum: NDArray,
    pub grad: NDArray,
    scalar_buffer: NDArray,
    pub idx_in_trajectory: i64,
    kinetic_energy: f64,
    pub potential_energy: f32,
    reuser: Weak<dyn ReuseState>,
}


impl State {
    fn new(size: usize, owner: &Rc<dyn ReuseState>, ctx: tvm::Context, dtype: tvm::DataType) -> State {
        State {
            p: NDArray::empty(&[size as i64], ctx, dtype).zeroed(),
            q: NDArray::empty(&[size as i64], ctx, dtype).zeroed(),
            v: NDArray::empty(&[size as i64], ctx, dtype).zeroed(),
            p_sum: NDArray::empty(&[size as i64], ctx, dtype).zeroed(),
            grad: NDArray::empty(&[size as i64], ctx, dtype).zeroed(),
            scalar_buffer: NDArray::empty(&[], ctx, dtype).zeroed(),
            idx_in_trajectory: 0,
            kinetic_energy: 0.,
            potential_energy: 0.,
            reuser: Rc::downgrade(owner),
        }
    }

    fn energy(&self) -> f64 {
        self.potential_energy as f64 + self.kinetic_energy as f64
    }
}


pub struct StateWrapper {
    inner: std::mem::ManuallyDrop<Rc<State>>,
}


impl Drop for StateWrapper {
    fn drop(&mut self) {
        let mut rc = unsafe { std::mem::ManuallyDrop::take(&mut self.inner) };
        if let Some(state_ref) = Rc::get_mut(&mut rc) {
            if let Some(reuser) = &mut state_ref.reuser.upgrade() {
                reuser.reuse_state(rc);
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

fn new_state(this: &mut Rc<StateStorage>, dim: usize, ctx: tvm::Context, dtype: tvm::DataType) -> StateWrapper {
    let inner = match this.free_states.borrow_mut().pop() {
        Some(inner) => {
            if dim != inner.q.len() {
                panic!("dim mismatch");
            }
            inner
        },
        None => {
            let owner: Rc<dyn ReuseState> = this.clone();
            Rc::new(State::new(dim, &owner, ctx, dtype))
        },
    };
    StateWrapper {
        inner: std::mem::ManuallyDrop::new(inner)
    }
}

pub struct StaticIntegrator {
    states: Rc<StateStorage>,
    leapfrog_rt: GraphRt,
    is_turning_rt: GraphRt,
    grad_rt: GraphRt,
    epsilon_pos: NDArray,
    epsilon_neg: NDArray,
    diag_mass: NDArray,
    scalar_buffer: NDArray,
    ctx: tvm::Context,
    dtype: tvm::DataType,
    ndim: usize,
}


impl Integrator for StaticIntegrator {
    type State = StateWrapper;
    type LeapfrogInfo = LeapfrogInfoImpl;
    type DivergenceInfo = DivergenceInfoImpl<()>;

    fn leapfrog(&mut self, start: &Self::State, dir: Direction) -> Result<(Self::State, Self::LeapfrogInfo), Self::DivergenceInfo> {
        let mut out_state = new_state(&mut self.states, self.ndim, self.ctx, self.dtype);
        let out = Rc::get_mut(&mut out_state.inner).expect("State already in use");

        let epsilon = 1e-6f64;

        self.epsilon_pos.fill_from_iter([epsilon as f32].iter().map(|x| *x));
        self.epsilon_pos.fill_from_iter([-epsilon as f32].iter().map(|x| *x));

        self.leapfrog_rt.set_input_zero_copy_idx(0, &start.q).unwrap();
        self.leapfrog_rt.set_input_zero_copy_idx(1, &start.p).unwrap();
        self.leapfrog_rt.set_input_zero_copy_idx(2, &start.grad).unwrap();

        match dir {
            Direction::Forward => {
                self.leapfrog_rt.set_input_zero_copy_idx(3, &self.epsilon_pos).unwrap();
            },
            Direction::Backward => {
                self.leapfrog_rt.set_input_zero_copy_idx(3, &self.epsilon_neg).unwrap();
            }
        };
        self.leapfrog_rt.set_input_zero_copy_idx(4, &self.diag_mass).unwrap();
        self.leapfrog_rt.set_input_zero_copy_idx(5, &start.p_sum).unwrap();

        self.leapfrog_rt.run().map_err(|_| DivergenceInfoImpl { logp_error: None })?;

        self.leapfrog_rt.get_output_into(0, out.scalar_buffer.clone()).unwrap();
        let mut buffer = [0f32];
        out.scalar_buffer.copy_to_buffer(&mut buffer);
        out.potential_energy = -buffer[0];


        self.leapfrog_rt.get_output_into(1, out.scalar_buffer.clone()).unwrap();
        let mut buffer = [0f32];
        out.scalar_buffer.copy_to_buffer(&mut buffer);
        out.kinetic_energy = buffer[0] as f64;

        self.leapfrog_rt.get_output_into(2, out.q.clone()).unwrap();
        self.leapfrog_rt.get_output_into(3, out.p.clone()).unwrap();
        self.leapfrog_rt.get_output_into(4, out.grad.clone()).unwrap();
        self.leapfrog_rt.get_output_into(5, out.v.clone()).unwrap();
        self.leapfrog_rt.get_output_into(6, out.p_sum.clone()).unwrap();

        match dir {
            Direction::Forward => {
                out.idx_in_trajectory = start.idx_in_trajectory + 1;
            },
            Direction::Backward => {
                out.idx_in_trajectory = start.idx_in_trajectory - 1;
            }
        };

        let info = LeapfrogInfoImpl {
            energy_error: out.energy() - start.energy(),
        };
        Ok((out_state, info))
    }

    #[inline]
    fn is_turning(&mut self, start: &Self::State, end: &Self::State) -> bool {
        let (start, end) = if start.idx_in_trajectory < end.idx_in_trajectory {
            (start, end)
        } else {
            (end, start)
        };
        
        self.is_turning_rt.set_input_zero_copy_idx(0, &end.p_sum).unwrap();
        self.is_turning_rt.set_input_zero_copy_idx(1, &start.p_sum).unwrap();
        self.is_turning_rt.set_input_zero_copy_idx(2, &end.v).unwrap();
        self.is_turning_rt.set_input_zero_copy_idx(3, &start.v).unwrap();

        self.is_turning_rt.run().unwrap();

        self.is_turning_rt.get_output_into(0, self.scalar_buffer.clone()).unwrap();

        let mut buffer = [0f32];
        self.scalar_buffer.copy_to_buffer(&mut buffer);

        //buffer[0] < 0.
        false
    }

    fn write_position(&self, state: &Self::State, out: &mut [f64]) {
        let data: Vec<f32> = state.q.to_vec().unwrap();
        assert!(data.len() == out.len());
        for (x, y) in data.iter().zip(out.iter_mut()) {
            *y = *x as f64;
        }
    }

    fn randomize_velocity<R: rand::Rng + ?Sized>(&mut self, state: &mut Self::State, rng: &mut R) {
        //let state_inner = self.initial_state.inner.as_mut().expect("Use after free");
        //let out = Rc::get_mut(state_inner).expect("State already in use");
        let out = Rc::get_mut(&mut state.inner).expect("State already in use");

        let dist = rand_distr::StandardNormal;
        out.p.fill_from_iter((0..self.ndim).map(|_| {
            let val: f32 = rng.sample(dist);
            val
        }));
    }

    fn new_state(&mut self, init: &[f64]) -> Result<Self::State, Self::DivergenceInfo> {
        let mut state = new_state(&mut self.states, self.ndim, self.ctx, self.dtype);
        
        let inner = Rc::get_mut(&mut state.inner).expect("State already in use");
        inner.q.fill_from_iter(init.iter().map(|x| *x as f32));
        
        inner.p_sum.fill_from_iter((0..self.ndim).map(|_| 0f32));

        self.grad_rt.set_input_zero_copy_idx(0, &inner.q).unwrap();
        self.grad_rt.run().unwrap();
        self.grad_rt.get_output_into(0, self.scalar_buffer.clone()).unwrap();
        self.grad_rt.get_output_into(1, inner.grad.clone()).unwrap();

        let mut buffer = [0f32];
        self.scalar_buffer.copy_to_buffer(&mut buffer);
        Ok(state)
    }
}


impl StaticIntegrator {
    pub fn new(leapfrog_rt: GraphRt, is_turning_rt: GraphRt, grad_rt: GraphRt, ndim: usize, ctx: tvm::Context, diag_mass: NDArray, step_size: f32) -> StaticIntegrator {
        let states = Rc::new(StateStorage::with_capacity(100));

        let mut epsilon_pos = ndarray::Array::zeros([]);
        let mut epsilon_neg = ndarray::Array::zeros([]);
        let buffer: ndarray::Array<f32, _> = ndarray::Array::zeros([]);

        *epsilon_pos.get_mut([]).unwrap() = step_size;
        *epsilon_neg.get_mut([]).unwrap() = -step_size;

        let dtype = tvm::DataType::float32();

        let mut integrator = StaticIntegrator {
            leapfrog_rt,
            is_turning_rt,
            grad_rt,
            ctx,
            dtype,
            epsilon_pos: NDArray::from_rust_ndarray(&epsilon_pos.into_dyn(), ctx, dtype).unwrap(),
            epsilon_neg: NDArray::from_rust_ndarray(&epsilon_neg.into_dyn(), ctx, dtype).unwrap(),
            scalar_buffer: NDArray::from_rust_ndarray(&buffer.into_dyn(), ctx, dtype).unwrap(),
            diag_mass,
            states,
            ndim,
        };

        for _ in 0..20 {
            let _ = new_state(&mut integrator.states, ndim, ctx, dtype);
        };
        integrator
    }
}
