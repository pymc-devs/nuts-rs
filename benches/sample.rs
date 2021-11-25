use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
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
    c.bench_function("sample + make normal 10", |b| {
        b.iter(|| sample_one(black_box(3.), black_box(&mut out)))
    });

    let mut out = vec![0.; 1000];
    c.bench_function("sample + make normal 1000", |b| {
        b.iter(|| sample_one(black_box(3.), black_box(&mut out)))
    });

    c.bench_function("make_integrator 10", |b| {
        b.iter(|| make_integrator(black_box(10), black_box(3.)))
    });
    c.bench_function("make_integrator 1000", |b| {
        b.iter(|| make_integrator(black_box(1000), black_box(3.)))
    });

    let mut integrator = make_integrator(10, 3.);
    let mut out = vec![0.; 10];
    let init = vec![3.5; out.len()];
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let state = integrator.new_state(&init).unwrap();
    let (state, info) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 9);
    dbg!(info);
    integrator.write_position(&state, &mut out);

    let sum: f64 = out.iter().sum();

    c.bench_function("sample normal 10", |b| {
        b.iter_batched(
            || rand::rngs::StdRng::seed_from_u64(42),
            |mut rng| {
                let state = integrator.new_state(&init).unwrap();
                let (state, _) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 9);
                integrator.write_position(&state, &mut out);
                assert_eq!(out.iter().sum::<f64>(), sum);
            },
            BatchSize::SmallInput,
        )
    });

    let mut integrator = make_integrator(1000, 3.);
    let mut out = vec![0.; 1000];
    let init = vec![3.5; out.len()];
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let state = integrator.new_state(&init).unwrap();
    let (state, info) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 9);
    dbg!(info);
    integrator.write_position(&state, &mut out);

    let sum: f64 = out.iter().sum();

    c.bench_function("sample normal 1000", |b| {
        b.iter_batched(
            || rand::rngs::StdRng::seed_from_u64(42),
            |mut rng| {
                let state = integrator.new_state(&init).unwrap();
                let (state, _) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 9);
                integrator.write_position(&state, &mut out);
                assert_eq!(out.iter().sum::<f64>(), sum);
            },
            BatchSize::SmallInput,
        )
    });

    /*
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

    let grad_mod = tvm::runtime::Module::load(&base.join("grad_10.so")).unwrap();
    let leapfrog_mod = tvm::runtime::Module::load(&base.join("leapfrog_10.so")).unwrap();
    let turning_mod = tvm::runtime::Module::load(&base.join("turning_10.so")).unwrap();

    let ctx = tvm::Context::cpu(0);
    let grad_rt =
        tvm::runtime::graph_rt::GraphRt::create_from_factory(grad_mod, "default", vec![ctx])
            .unwrap();
    let leapfrog_rt =
        tvm::runtime::graph_rt::GraphRt::create_from_factory(leapfrog_mod, "default", vec![ctx])
            .unwrap();
    let turning_rt =
        tvm::runtime::graph_rt::GraphRt::create_from_factory(turning_mod, "default", vec![ctx])
            .unwrap();

    let ndim = 10;
    let dtype = tvm::DataType::float32();

    let diag_mass_nd: ndarray::Array<f32, _> = ndarray::Array::ones([ndim]);
    let diag_mass = tvm::NDArray::from_rust_ndarray(&diag_mass_nd.into_dyn(), ctx, dtype).unwrap();

    let mut integrator = nuts_rs::tvm::StaticIntegrator::new(
        leapfrog_rt,
        turning_rt,
        grad_rt,
        ndim,
        ctx,
        diag_mass,
        1e-6,
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut out = vec![0.; ndim];
    let init = vec![3.5; out.len()];
    let state = integrator.new_state(&init).unwrap();
    let (state, info) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 9);
    dbg!(state.idx_in_trajectory);
    dbg!(info);
    integrator.write_position(&state, &mut out);

    let sum: f64 = out.iter().sum();

    c.bench_function("sample normal 10 tvm", |b| {
        b.iter_batched(
            || rand::rngs::StdRng::seed_from_u64(42),
            |mut rng| {
                let state = integrator.new_state(&init).unwrap();
                let (state, _) = nuts_rs::nuts::draw(state, &mut rng, &mut integrator, 9);
                integrator.write_position(&state, &mut out);
                assert_eq!(out.iter().sum::<f64>(), sum);
            },
            BatchSize::SmallInput,
        )
    });
    */
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
