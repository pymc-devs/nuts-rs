use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nuts_rs::cpu_sampler::test_logps::NormalLogp;
use nuts_rs::cpu_sampler::{
    sample_parallel, AdaptiveSampler, JitterInitFunc, SamplerArgs, StaticSampler,
};
use nuts_rs::mass_matrix::DiagAdaptExpSettings;
use nuts_rs::math::{axpy, axpy_out, scalar_prods_of_diff, vector_dot};

fn make_sampler(dim: usize, mu: f64) -> StaticSampler<NormalLogp> {
    let func = NormalLogp::new(dim, mu);
    let settings = DiagAdaptExpSettings::default();
    StaticSampler::new(func, SamplerArgs::default(), settings, 0, 42)
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
    for n in [10, 12, 100, 800] {
        let x = vec![2.5; n];
        let mut y = vec![3.5; n];
        let a = vec![3.5; n];
        let d = vec![3.5; n];
        let mut out = vec![0.; n];

        axpy(&x, &mut y, 4.);
        c.bench_function(&format!("axpy {}", n), |b| {
            b.iter(|| axpy(black_box(&x), black_box(&mut y), black_box(4.)));
        });
        axpy_out(&x, &y, 4., &mut out);
        c.bench_function(&format!("axpy_out {}", n), |b| {
            b.iter(|| {
                axpy_out(
                    black_box(&x),
                    black_box(&y),
                    black_box(4.),
                    black_box(&mut out),
                )
            });
        });
        vector_dot(&x, &y);
        c.bench_function(&format!("vector_dot {}", n), |b| {
            b.iter(|| vector_dot(black_box(&x), black_box(&y)));
        });
        scalar_prods_of_diff(&x, &y, &a, &d);
        c.bench_function(&format!("scalar_prods_of_diff {}", n), |b| {
            b.iter(|| {
                scalar_prods_of_diff(black_box(&x), black_box(&y), black_box(&a), black_box(&d))
            });
        });
    }

    let mut out = vec![0.; 10];
    c.bench_function("sample_1000 10", |b| {
        b.iter(|| sample_one(black_box(3.), black_box(&mut out)))
    });

    let mut out = vec![0.; 1000];
    c.bench_function("sample_1000 1000", |b| {
        b.iter(|| sample_one(black_box(3.), black_box(&mut out)))
    });

    for n in [10, 12, 1000] {
        c.bench_function(&format!("sample_adaptive {}", n), |b| {
            b.iter(|| {
                let func = NormalLogp::new(n, 0.);
                let settings = SamplerArgs::default();
                let n_draws = 2000;
                let seed = 42;
                let mut sampler = AdaptiveSampler::new(func, settings, 0, seed);
                let draws: Vec<_> = (0..n_draws).map(|_| sampler.draw()).collect();
                draws
            });
        });

        c.bench_function(&format!("sample_parallel {}", n), |b| {
            b.iter(|| {
                let func = NormalLogp::new(n, 0.);
                let settings = black_box(SamplerArgs::default());
                let mut init_point_func = JitterInitFunc::new();
                let n_chains = black_box(10);
                let n_draws = black_box(1000);
                let seed = black_box(42);
                let n_try_init = 10;
                let (handle, channel) = sample_parallel(
                    func,
                    &mut init_point_func,
                    settings,
                    n_chains,
                    n_draws,
                    seed,
                    n_try_init,
                )
                .unwrap();
                //let draws: Vec<_> = channel.iter().collect();
                let draws = channel.iter().count();
                //assert_eq!(draws.len() as u64, (n_draws + settings.num_tune) * n_chains);
                handle.join().unwrap().unwrap();
                draws
            });
        });
    }

    /*
    let mut out = vec![0.; 100_000];
    c.bench_function("sample_1000 100_000", |b| {
        b.iter(|| sample_one(black_box(3.), black_box(&mut out)))
    });
    */
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
