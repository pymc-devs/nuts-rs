use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nuts_rs::cpu_sampler::test_logps::NormalLogp;
use nuts_rs::cpu_sampler::UnitStaticSampler;

fn make_sampler(dim: usize, mu: f64) -> UnitStaticSampler<NormalLogp> {
    let func = NormalLogp::new(dim, mu);
    UnitStaticSampler::new(func, 42, 10, 1e-1)
}

pub fn sample_one(mu: f64, out: &mut [f64]) {
    let mut sampler = make_sampler(out.len(), mu);
    let init = vec![3.5; out.len()];
    sampler.set_position(&init).unwrap();
    for _ in 0..1000 {
        let (state, _stats) = sampler.draw();
        out.copy_from_slice(&state);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut out = vec![0.; 10];
    c.bench_function("sample_1000 10", |b| {
        b.iter(|| sample_one(black_box(3.), black_box(&mut out)))
    });

    let mut out = vec![0.; 1000];
    c.bench_function("sample_1000 1000", |b| {
        b.iter(|| sample_one(black_box(3.), black_box(&mut out)))
    });

    let mut out = vec![0.; 100_000];
    c.bench_function("sample_1000 100_000", |b| {
        b.iter(|| sample_one(black_box(3.), black_box(&mut out)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
