//! Shared model initialization and trace recording for both execution modes.
use crate::sampler_stats::StatsDims;
use crate::storage::ChainStorage;
use crate::{Chain, Math, Model, Progress, Settings, Storable, Value};
use anyhow::{Context, Result, ensure};
use rand::Rng;

pub(crate) fn model_chain<'a, M: Model, S: Settings, R: Rng + ?Sized>(
    model: &'a M,
    settings: &S,
    chain_id: u64,
    rng: &mut R,
) -> Result<S::Chain<M::Math<'a>>> {
    let math = model.math(rng).context("Failed to create model density")?;
    Ok(settings.new_chain(chain_id, math, rng))
}

pub(crate) fn initialize_position<'a, M: Model, C: Chain<M::Math<'a>>, R: Rng + ?Sized>(
    model: &'a M,
    chain: &mut C,
    rng: &mut R,
    attempts: usize,
) -> Result<Vec<f64>> {
    ensure!(attempts > 0, "max_init_attempts must be positive");
    let mut position = vec![0.; chain.dim()];
    for attempt in 0..attempts {
        model
            .init_position(rng, &mut position)
            .context("Failed to generate a new initial position")?;
        match chain.set_position(&position) {
            Ok(()) => return Ok(position),
            Err(error) if attempt + 1 == attempts => {
                return Err(error.context("All initialization points failed"));
            }
            Err(_) => {}
        }
    }
    unreachable!("positive attempt count validated above")
}

/// Consume a draw's storable fields once, retaining a copy only for streaming callers.
pub(crate) fn record_draw<M: Math, C: Chain<M>, T: ChainStorage>(
    chain: &C,
    settings: &impl Settings,
    trace: Option<&mut T>,
    expanded: &mut M::ExpandedVector,
    stats: &mut C::Stats,
    info: &Progress,
    retain_values: bool,
) -> Result<Vec<(String, Option<Value>)>> {
    let math = chain.math();
    let values = expanded.get_all(&*math);
    let Some(trace) = trace else {
        return Ok(if retain_values {
            values
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value))
                .collect()
        } else {
            vec![]
        });
    };
    let retained = if retain_values {
        values
            .iter()
            .map(|(name, value)| ((*name).to_owned(), value.clone()))
            .collect()
    } else {
        vec![]
    };
    trace.record_sample(
        settings,
        stats.get_all(&StatsDims::from(&*math)),
        values,
        info,
    )?;
    Ok(retained)
}
