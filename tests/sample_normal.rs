use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{
    array::{Array, ArrayBuilder, FixedSizeListBuilder, PrimitiveBuilder},
    datatypes::Float64Type,
};
use nuts_rs::{
    CpuLogpFunc, CpuMath, DiagGradNutsSettings, DrawStorage, LogpError, Model, Sampler,
    SamplerWaitResult, Settings, Trace,
};
use rand::prelude::Rng;
use rand_distr::{Distribution, StandardNormal};
use thiserror::Error;

struct NormalLogp<'a> {
    dim: usize,
    mu: &'a [f64],
}

#[derive(Error, Debug)]
enum NormalLogpError {}

impl LogpError for NormalLogpError {
    fn is_recoverable(&self) -> bool {
        true
    }
}

impl<'a> CpuLogpFunc for NormalLogp<'a> {
    type LogpError = NormalLogpError;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let n = position.len();
        assert!(grad.len() == n);
        let mut logp = 0f64;

        position
            .iter()
            .zip(self.mu.iter())
            .zip(grad.iter_mut())
            .for_each(|((&p, &mu), grad)| {
                let diff = p - mu;
                logp -= diff * diff / 2.;
                *grad = -diff;
            });
        Ok(logp)
    }
}

struct Storage {
    draws: FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>,
}

impl Storage {
    fn new(size: usize) -> Storage {
        let values = PrimitiveBuilder::new();
        let draws = FixedSizeListBuilder::new(values, size as i32);
        Storage { draws }
    }
}

impl DrawStorage for Storage {
    fn append_value(&mut self, point: &[f64]) -> anyhow::Result<()> {
        self.draws.values().append_slice(point);
        self.draws.append(true);
        Ok(())
    }

    fn finalize(mut self) -> anyhow::Result<Arc<dyn Array>> {
        Ok(ArrayBuilder::finish(&mut self.draws))
    }

    fn inspect(&self) -> anyhow::Result<Arc<dyn Array>> {
        Ok(ArrayBuilder::finish_cloned(&self.draws))
    }
}

struct NormalModel {
    mu: Box<[f64]>,
}

impl NormalModel {
    fn new(mu: Box<[f64]>) -> Self {
        NormalModel { mu }
    }
}

impl Model for NormalModel {
    type Math<'model> = CpuMath<NormalLogp<'model>>
    where
        Self: 'model;

    type DrawStorage<'model, S: Settings> = Storage
    where
        Self: 'model;

    fn new_trace<'model, S: Settings, R: Rng + ?Sized>(
        &'model self,
        _rng: &mut R,
        _chain_id: u64,
        _settings: &'model S,
    ) -> anyhow::Result<Self::DrawStorage<'model, S>> {
        Ok(Storage::new(self.mu.len()))
    }

    fn math(&self) -> anyhow::Result<Self::Math<'_>> {
        Ok(CpuMath::new(NormalLogp {
            dim: self.mu.len(),
            mu: &self.mu,
        }))
    }

    fn init_position<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> anyhow::Result<()> {
        let normal = StandardNormal;
        position.iter_mut().for_each(|x| *x = normal.sample(rng));
        Ok(())
    }
}

fn sample() -> anyhow::Result<Trace> {
    let mu = vec![0.5; 100];
    let model = NormalModel::new(mu.into());
    let settings = DiagGradNutsSettings {
        seed: 42,
        num_chains: 6,
        ..Default::default()
    };

    let mut sampler = Sampler::new(model, settings, 6, None)?;

    let trace = loop {
        match sampler.wait_timeout(Duration::from_secs(1)) {
            SamplerWaitResult::Trace(trace) => break trace,
            SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
            SamplerWaitResult::Err(err, _trace) => return Err(err),
        };
    };
    Ok(trace)
}

#[test]
fn run() -> anyhow::Result<()> {
    let start = Instant::now();
    let trace = sample()?;
    assert!(trace.chains.len() == 6);
    dbg!(start.elapsed());
    Ok(())
}
