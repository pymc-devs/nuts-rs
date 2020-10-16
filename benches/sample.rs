use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use nuts_rs::cpu::test_logps::NormalLogp;
use nuts_rs::nuts::Integrator;
use rand::SeedableRng;


fn make_integrator(dim: usize, mu: f64) -> impl Integrator {
    let func = NormalLogp::new(dim, mu);
    nuts_rs::cpu::StaticIntegrator::new(func, dim)
}

pub fn sample_one(mu: f64, out: &mut [f64]) {
    let mut integrator = make_integrator(out.len(), mu);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let init = vec![3.5; out.len()];
    let state = integrator.new_state(&init).unwrap();
    let (state, _) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 10);
    integrator.write_position(&state, out);
}


fn criterion_benchmark(c: &mut Criterion) {
    let mut out = vec![0.; 10];
    c.bench_function("sample + make normal 10", |b| b.iter(|| sample_one(black_box(3.), black_box(&mut out))));

    let mut out = vec![0.; 1000];
    c.bench_function("sample + make normal 1000", |b| b.iter(|| sample_one(black_box(3.), black_box(&mut out))));

    c.bench_function("make_integrator 10", |b| b.iter(|| make_integrator(black_box(10), black_box(3.))));
    c.bench_function("make_integrator 1000", |b| b.iter(|| make_integrator(black_box(1000), black_box(3.))));

    let mut integrator = make_integrator(10, 3.);
    let mut out = vec![0.; 10];
    let init = vec![3.5; out.len()];
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let state = integrator.new_state(&init).unwrap();
    let (state, _) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 10);
    integrator.write_position(&state, &mut out);

    let sum: f64 = out.iter().sum();

    c.bench_function("sample normal 10", |b| b.iter_batched(|| {
        rand::rngs::StdRng::seed_from_u64(42)
    },
    |mut rng| {
        let state = integrator.new_state(&init).unwrap();
        let (state, _) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 10);
        integrator.write_position(&state, &mut out);
        assert_eq!(out.iter().sum::<f64>(), sum);
    },
    BatchSize::SmallInput,
    ));

    let mut integrator = make_integrator(1000, 3.);
    let mut out = vec![0.; 1000];
    let init = vec![3.5; out.len()];
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let state = integrator.new_state(&init).unwrap();
    let (state, _) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 10);
    integrator.write_position(&state, &mut out);

    let sum: f64 = out.iter().sum();

    c.bench_function("sample normal 1000", |b| b.iter_batched(|| {
        rand::rngs::StdRng::seed_from_u64(42)
    },
    |mut rng| {
        let state = integrator.new_state(&init).unwrap();
        let (state, _) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 10);
        integrator.write_position(&state, &mut out);
        assert_eq!(out.iter().sum::<f64>(), sum);
    },
    BatchSize::SmallInput,
    ));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
