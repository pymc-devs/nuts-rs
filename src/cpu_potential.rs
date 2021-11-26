use std::fmt::Debug;

use crate::cpu_state::{InnerState, State, StatePool};
use crate::nuts::DivergenceInfo;
use crate::nuts::{Direction, LeapfrogInfo};

pub(crate) trait LogpFunc {
    type Err: Debug + 'static;

    fn logp(&mut self, state: &mut InnerState) -> Result<(), Self::Err>;
    fn dim(&self) -> usize;
}

pub(crate) trait MassMatrix {
    fn update_velocity(&self, state: &mut InnerState);
    fn update_energy(&self, state: &mut InnerState);
    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut InnerState, rng: &mut R);
}


pub(crate) struct DiagMassMatrix {
    diag: Box<[f64]>,
}


impl MassMatrix for DiagMassMatrix {
    fn update_velocity(&self, state: &mut InnerState) {
        todo!()
    }

    fn update_energy(&self, state: &mut InnerState) {
        todo!()
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut InnerState, rng: &mut R) {
        todo!()
    }
    /*
        let dist = rand_distr::StandardNormal;
        for val in momentum.iter_mut() {
            *val = rng.sample(dist);
        }
        Ok(())
    */
}


#[derive(Debug)]
pub struct DivergenceInfoImpl<E> {
    pub logp_function_error: Option<E>,
}

impl<E: Debug> DivergenceInfo for DivergenceInfoImpl<E> {}

struct Potential<F: LogpFunc, M: MassMatrix> {
    logp: F,
    diag: Box<[f64]>,
    inv_diag: Box<[f64]>,
    state_pool: StatePool,
    mass_matrix: M,
}

impl<F: LogpFunc, M: MassMatrix> Potential<F, M> {
    fn new(logp: F, mass_matrix: M, dim: usize) -> Potential<F, M> {
        let state_pool = StatePool::with_capacity(dim, 20);

        let potential = Potential {
            logp,
            diag: vec![1.; dim].into(),
            inv_diag: vec![1.; dim].into(),
            state_pool,
            mass_matrix,
        };

        potential
    }
}

impl<F: LogpFunc, M: MassMatrix> crate::nuts::Potential for Potential<F, M> {
    type State = State;
    type DivergenceInfo = DivergenceInfoImpl<F::Err>;

    fn update_energy(&self, state: &mut State) {
        todo!()
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut Self::State, rng: &mut R) {
        let inner = state.try_mut_inner().unwrap();
        self.mass_matrix.randomize_momentum(inner, rng);
    }

    fn leapfrog(
        &mut self,
        start: &Self::State,
        dir: Direction,
        step_size: f64,
    ) -> Result<(Self::State, LeapfrogInfo), Self::DivergenceInfo> {
        use crate::math::{axpy, axpy_out};

        let mut out_state = self.state_pool.new_state();
        let mut out = out_state.try_mut_inner().expect("State already in use");

        let sign = match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };

        let epsilon = (sign as f64) * step_size;
        let dt = epsilon / 2.;

        axpy_out(&start.grad, &start.p, dt, &mut out.p);

        self.mass_matrix.update_velocity(out);

        axpy_out(&start.q, &out.v, epsilon, &mut out.q);

        if let Err(error) = self.logp.logp(out) {
            return Err(DivergenceInfoImpl {
                logp_function_error: Some(error),
            });
        }

        axpy(&out.grad, &mut out.p, dt);

        self.mass_matrix.update_velocity(out);
        self.mass_matrix.update_energy(out);

        match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };
        match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };
        out.idx_in_trajectory = start.idx_in_trajectory + sign;

        axpy_out(&start.p_sum, &out.p, sign as f64, &mut out.p_sum);  // TODO check order

        let info = LeapfrogInfo {};
        Ok((out_state, info))
    }

    fn init_state(&mut self, position: &[f64]) -> Result<Self::State, Self::DivergenceInfo> {
        todo!()
    }
}

/*
pub mod test_logps {
    use super::{LogpFunc, State};

    pub struct NormalLogp {
        dim: usize,
        mu: f64,
    }

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
        fn logp(&mut self, state: &mut State) -> Result<(), ()> {
            let position = &state.q;
            let grad = &mut state.grad;
            let n = position.len();
            assert!(grad.len() == n);
            let mut logp = 0f64;

            for (p, g) in position.iter().zip(grad.iter_mut()) {
                let val = *p - self.mu;
                logp -= val * val;
                *g = -val;
            }
            //for i in 0..n {
            //    let val = position[i] - self.mu;
            //    logp -= val * val;
            //    grad[i] = -val;
            //}
            state.potential_energy = -logp;
            Ok(())
        }
    }
}
*/

/*
#[cfg(test)]
mod tests {
    use super::test_logps::*;
    use super::*;
    use pretty_assertions::assert_eq;
    use rand::SeedableRng;

    #[test]
    fn make_state() {
        /*
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
        */
    }

    #[test]
    fn deterministic() {
        let dim = 3usize;
        let func = NormalLogp::new(dim, 3.);
        let init = vec![3.5; dim];

        let mut integrator = StaticIntegrator::new(func, DiagMassMatrix::new(dim), 1., dim);
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
*/
