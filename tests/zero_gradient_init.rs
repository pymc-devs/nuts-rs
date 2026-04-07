/// Regression test: init_state must accept valid initial points whose gradient is exactly zero.
///
/// Scenario (mirrors the eight_schools WASM failure):
///   - Model: standard normal N(0, 1),  dim = 10
///   - Initial position: all zeros
///   - Gradient at x = 0: d/dx(-x²/2) = -x = 0  (every component)
///
/// Before the fix, `check_all` called `array_all_finite_and_nonzero` on the
/// transformed gradient, which rejected this mathematically valid point with
/// `NutsError::BadInitGrad`.  After the fix it only calls `array_all_finite`.
use std::collections::HashMap;

use nuts_rs::{Chain, CpuLogpFunc, CpuMath, DiagGradNutsSettings, LogpError, Settings};
use nuts_storable::HasDims;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Minimal standard-normal model  (mu = 0, sigma = 1 in every dimension)
// ---------------------------------------------------------------------------

struct StandardNormal {
    dim: usize,
}

#[derive(Debug, Error)]
enum StandardNormalError {}

impl LogpError for StandardNormalError {
    fn is_recoverable(&self) -> bool {
        true
    }
}

impl HasDims for StandardNormal {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        [
            ("unconstrained_parameter".to_string(), self.dim as u64),
        ]
        .into_iter()
        .collect()
    }
}

impl CpuLogpFunc for StandardNormal {
    type LogpError = StandardNormalError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let mut logp = 0f64;
        for (x, g) in position.iter().copied().zip(grad.iter_mut()) {
            logp -= x * x / 2.0;
            *g = -x; // gradient is 0 when x == 0
        }
        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        position: &[f64],
    ) -> Result<Self::ExpandedVector, nuts_rs::CpuMathError> {
        Ok(position.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Regression test
// ---------------------------------------------------------------------------

#[test]
fn set_position_at_zero_gradient_should_succeed() {
    let dim = 10;
    let math = CpuMath::new(StandardNormal { dim });

    let settings = DiagGradNutsSettings::default();
    let mut rng = rand::rng();
    let mut chain = settings.new_chain(0, math, &mut rng);

    // All-zero initial position → gradient is identically zero at every component.
    // This is the mathematically valid mode of N(0,1) in each dimension and must
    // not be rejected by init_state.
    let init = vec![0f64; dim];
    chain
        .set_position(&init)
        .expect("set_position should accept a valid point whose gradient is zero");
}
