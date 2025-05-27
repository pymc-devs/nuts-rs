use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use nix::sched::{sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use nuts_rs::{Chain, CpuLogpFunc, CpuMath, LogpError, Math, Settings};
use rand::SeedableRng;
use rayon::ThreadPoolBuilder;
use thiserror::Error;

#[derive(Debug)]
struct PosteriorDensity {
    dim: usize,
}

// The density might fail in a recoverable or non-recoverable manner...
#[derive(Debug, Error)]
enum PosteriorLogpError {}
impl LogpError for PosteriorLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

impl CpuLogpFunc for PosteriorDensity {
    type LogpError = PosteriorLogpError;

    // Only used for transforming adaptation.
    type TransformParams = ();

    // We define a 10 dimensional normal distribution
    fn dim(&self) -> usize {
        self.dim
    }

    // The normal likelihood with mean 3 and its gradient.
    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let mu = 3f64;
        let logp = position
            .iter()
            .copied()
            .zip(grad.iter_mut())
            .map(|(x, grad)| {
                let diff = x - mu;
                *grad = -diff;
                -0.5 * diff * diff
            })
            .sum();
        return Ok(logp);
    }
}

fn make_sampler(dim: usize) -> impl Chain<CpuMath<PosteriorDensity>> {
    let func = PosteriorDensity { dim: dim };

    let settings = nuts_rs::DiagGradNutsSettings {
        num_tune: 1000,
        maxdepth: 3, // small value just for testing...
        ..Default::default()
    };

    let math = nuts_rs::CpuMath::new(func);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42u64);
    settings.new_chain(0, math, &mut rng)
}

pub fn sample_one(out: &mut [f64]) {
    let mut sampler = make_sampler(out.len());
    let init = vec![3.5; out.len()];
    sampler.set_position(&init).unwrap();
    for _ in 0..1000 {
        let (state, _stats) = sampler.draw().unwrap();
        out.copy_from_slice(&state);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    ThreadPoolBuilder::new()
        .num_threads(4)
        .start_handler(|idx| {
            let mut cpu_set = CpuSet::new();
            cpu_set.set(idx).unwrap();
            sched_setaffinity(Pid::from_raw(0), &cpu_set).unwrap();
        })
        .build_global()
        .unwrap();

    let mut cpu_set = CpuSet::new();
    cpu_set.set(0).unwrap();
    sched_setaffinity(Pid::from_raw(0), &cpu_set).unwrap();

    for n in [4, 16, 17, 100, 4567] {
        let mut math = CpuMath::new(PosteriorDensity { dim: n });

        let x = math.new_array();
        let p = math.new_array();
        let p2 = math.new_array();
        let n1 = math.new_array();
        let mut y = math.new_array();
        let mut out = math.new_array();

        let x_vec = vec![2.5; n];
        let mut y_vec = vec![2.5; n];

        c.bench_function(&format!("multiply {}", n), |b| {
            b.iter(|| math.array_mult(black_box(&x), black_box(&y), black_box(&mut out)));
        });

        c.bench_function(&format!("axpy {}", n), |b| {
            b.iter(|| math.axpy(black_box(&x), black_box(&mut y), black_box(4.)));
        });

        c.bench_function(&format!("axpy_ndarray {}", n), |b| {
            b.iter(|| {
                let x = ndarray::aview1(black_box(&x_vec));
                let mut y = ndarray::aview_mut1(black_box(&mut y_vec));
                //y *= &x;// * black_box(4.);
                y.scaled_add(black_box(4f64), &x);
            });
        });

        c.bench_function(&format!("axpy_out {}", n), |b| {
            b.iter(|| {
                math.axpy_out(
                    black_box(&x),
                    black_box(&y),
                    black_box(4.),
                    black_box(&mut out),
                )
            });
        });

        c.bench_function(&format!("vector_dot {}", n), |b| {
            b.iter(|| math.array_vector_dot(black_box(&x), black_box(&y)));
        });

        c.bench_function(&format!("scalar_prods2 {}", n), |b| {
            b.iter(|| {
                math.scalar_prods2(black_box(&p), black_box(&p2), black_box(&x), black_box(&y))
            });
        });

        c.bench_function(&format!("scalar_prods3 {}", n), |b| {
            b.iter(|| {
                math.scalar_prods3(
                    black_box(&p),
                    black_box(&p2),
                    black_box(&n1),
                    black_box(&x),
                    black_box(&y),
                )
            });
        });
    }

    let mut out = vec![0.; 10];
    c.bench_function("sample_1000_10", |b| {
        b.iter(|| sample_one(black_box(&mut out)))
    });

    let mut out = vec![0.; 1000];
    c.bench_function("sample_1000_1000", |b| {
        b.iter(|| sample_one(black_box(&mut out)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
