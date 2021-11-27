use rand::Rng;

use crate::{
    cpu_potential::{CpuLogpFunc, Potential, UnitMassMatrix},
    cpu_state::{State, StatePool},
    nuts::{draw, Collector, SampleInfo},
};

pub struct UnitStaticSampler<F: CpuLogpFunc> {
    potential: Potential<F, UnitMassMatrix>,
    state: State,
    pool: StatePool,
    maxdepth: u64,
    step_size: f64,
}

struct NullCollector {}

impl Collector for NullCollector {
    type State = State;
}

impl<F: CpuLogpFunc> UnitStaticSampler<F> {
    pub fn new(logp: F) -> UnitStaticSampler<F> {
        let mass_matrix = UnitMassMatrix {};
        let mut pool = StatePool::new(logp.dim());
        let potential = Potential::new(logp, mass_matrix);
        let state = pool.new_state();
        UnitStaticSampler {
            potential,
            state,
            pool,
            maxdepth: 10,
            step_size: 1e-2,
        }
    }

    pub fn set_position(&mut self, position: &[f64]) -> Result<(), F::Err> {
        use crate::nuts::Potential;
        {
            let inner = self.state.try_mut_inner().expect("State already in use");
            inner.q.copy_from_slice(position);
        }
        if let Err(err) = self.potential.update_potential_gradient(&mut self.state) {
            return Err(err.logp_function_error.unwrap());
        }
        // TODO check init of p_sum
        Ok(())
    }

    pub fn draw<R: Rng + ?Sized>(&mut self, rng: &mut R) -> (Box<[f64]>, SampleInfo) {
        use crate::nuts::Potential;
        let mut collector = NullCollector {};
        self.potential.randomize_momentum(&mut self.state, rng);
        self.potential.update_velocity(&mut self.state);
        self.potential.update_kinetic_energy(&mut self.state);

        let (state, info) = draw(
            &mut self.pool,
            self.state.clone(),
            rng,
            &mut self.potential,
            self.maxdepth,
            self.step_size,
            &mut collector,
        );
        self.state = state;
        let position: Box<[f64]> = self.state.q.clone().into();
        (position, info)
    }
}

pub mod test_logps {
    use crate::cpu_potential::CpuLogpFunc;

    pub struct NormalLogp {
        dim: usize,
        mu: f64,
    }

    impl NormalLogp {
        pub fn new(dim: usize, mu: f64) -> NormalLogp {
            NormalLogp { dim, mu }
        }
    }

    impl CpuLogpFunc for NormalLogp {
        type Err = ();

        fn dim(&self) -> usize {
            self.dim
        }
        fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, ()> {
            let n = position.len();
            assert!(gradient.len() == n);

            let mut logp = 0f64;
            for (p, g) in position.iter().zip(gradient.iter_mut()) {
                let val = *p - self.mu;
                logp -= val * val;
                *g = -val;
            }
            Ok(logp)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu_sampler::UnitStaticSampler;

    use super::test_logps::*;
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

        let mut sampler = UnitStaticSampler::new(func);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        sampler.set_position(&init).unwrap();
        let (sample1, info1) = sampler.draw(&mut rng);

        let func = NormalLogp::new(dim, 3.);
        let mut sampler = UnitStaticSampler::new(func);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        sampler.set_position(&init).unwrap();
        let (sample2, info2) = sampler.draw(&mut rng);

        dbg!(&sample1);
        dbg!(info1);

        dbg!(&sample2);
        dbg!(info2);

        assert_eq!(sample1, sample2);
    }
}
