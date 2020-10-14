pub fn sample_one(mu: f64, out: &mut [f64]) {
    use nuts_rs::nuts::Integrator;

    struct NormalLogp { dim: usize, mu: f64 };

    impl nuts_rs::cpu::LogpFunc for NormalLogp {
        type Err = ();

        fn dim(&self) -> usize {
            self.dim
        }
        fn logp(&self, state: &mut nuts_rs::cpu::State) -> Result<(), ()>{
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

    let func = NormalLogp { dim: 1000, mu };
    let init = vec![3.5; func.dim];
    let mut integrator = nuts_rs::cpu::StaticIntegrator::new(func, &init).unwrap();

    let mut rng = rand::thread_rng();
    let (state, _) = nuts_rs::nuts::draw(&mut rng, &mut integrator, 20);
    integrator.write_position(&state, out);
}


fn main() {
    let mu = 3.;
    let mut out = vec![0.; 1000];
    sample_one(mu, &mut out);
    println!("{:?}", out[0]);
}
