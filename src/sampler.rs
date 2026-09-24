//! High-level sampler entry points: `Settings` presets, the parallel `Sampler`,
//! and `sample_sequentially` for running one or many chains.

use anyhow::Result;
use nuts_storable::{HasDims, Storable, Value};
use rand::{Rng, SeedableRng, rngs::ChaCha8Rng};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::HashMap, fmt::Debug, time::Duration};

use anyhow::{Context, bail};
use itertools::Itertools;
use std::{
    collections::HashSet,
    ops::Deref,
    sync::{Arc, Mutex},
};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
// `std::time::Instant::now` panics on wasm32-unknown-unknown.
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

#[cfg(feature = "parallel")]
use std::{
    collections::VecDeque,
    sync::mpsc::{
        Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError, channel, sync_channel,
    },
    thread::{self, JoinHandle, spawn},
};

use crate::{
    DiagAdaptExpSettings, Math, StepSizeAdaptMethod,
    adapt_strategy::{EuclideanAdaptOptions, GlobalStrategy, GlobalStrategyStatsOptions},
    chain::{AdaptStrategy, Chain, NutsChain, StatOptions},
    dynamics::{KineticEnergyKind, TransformedHamiltonian, TransformedPointStatsOptions},
    external_adapt_strategy::{ExternalTransformAdaptation, FlowSettings},
    mclmc::MclmcTrajectoryKind,
    nuts::NutsOptions,
    sampler_stats::{SamplerStats, StatsDims},
    transform::{
        DiagAdaptStrategy, DiagMassMatrix, ExternalTransformation, LowRankMassMatrix,
        LowRankMassMatrixStrategy, LowRankSettings,
    },
};

use crate::{
    model::Model,
    storage::{ChainStorage, StorageConfig, TraceStorage},
};

/// All sampler configurations implement this trait
pub trait Settings:
    private::Sealed + Clone + Copy + Default + Sync + Send + Serialize + DeserializeOwned + 'static
{
    type Chain<M: Math>: Chain<M>;

    /// Check for values the sampler cannot work with.
    fn validate(&self) -> Result<()>;

    /// Create a new chain. Fails if the settings are invalid (see [`Settings::validate`]),
    /// or if they need something from `math` it cannot provide.
    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        math: M,
        rng: &mut R,
    ) -> Result<Self::Chain<M>>;

    fn hint_num_tune(&self) -> usize;
    fn hint_num_draws(&self) -> usize;
    fn num_chains(&self) -> usize;
    fn seed(&self) -> u64;
    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions;
    fn sampler_name(&self) -> &'static str;
    fn adaptation_name(&self) -> &'static str;

    /// Stats these settings switch off. They are never written, so declaring them would only
    /// create arrays that stay at their fill value and that every storage backend then carries.
    fn disabled_stats(&self) -> Vec<&'static str>;

    /// The stats that are stored, in the order the sampler emits them. Storage is allocated
    /// from this list, and `ChainProcess` drops anything emitted that is not in it.
    fn stat_names<M: Math>(&self, math: &M) -> Vec<String> {
        let dims = StatsDims::from(math);
        let disabled = self.disabled_stats();
        <<Self::Chain<M> as SamplerStats<M>>::Stats as Storable<_>>::names(&dims)
            .into_iter()
            .filter(|name| !disabled.contains(name))
            .map(String::from)
            .collect()
    }

    /// The posterior variables that are stored, in the order the model expands them. A `Math`
    /// may declare fewer than it produces - a model restricted to a subset of its variables -
    /// and `ChainProcess` drops the rest.
    fn data_names<M: Math>(&self, math: &M) -> Vec<String> {
        <M::ExpandedVector as Storable<_>>::names(math)
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn stat_types<M: Math>(&self, math: &M) -> Vec<(String, nuts_storable::ItemType)> {
        self.stat_names(math)
            .into_iter()
            .map(|name| (name.clone(), self.stat_type::<M>(math, &name)))
            .collect()
    }

    fn stat_type<M: Math>(&self, math: &M, name: &str) -> nuts_storable::ItemType {
        let dims = StatsDims::from(math);
        <<Self::Chain<M> as SamplerStats<M>>::Stats as Storable<_>>::item_type(&dims, name)
    }

    fn data_types<M: Math>(&self, math: &M) -> Vec<(String, nuts_storable::ItemType)> {
        self.data_names(math)
            .into_iter()
            .map(|name| (name.clone(), self.data_type(math, &name)))
            .collect()
    }
    fn data_type<M: Math>(&self, math: &M, name: &str) -> nuts_storable::ItemType {
        <M::ExpandedVector as Storable<_>>::item_type(math, name)
    }

    fn stat_dims_all<M: Math>(&self, math: &M) -> Vec<(String, Vec<String>)> {
        self.stat_names(math)
            .into_iter()
            .map(|name| (name.clone(), self.stat_dims::<M>(math, &name)))
            .collect()
    }

    fn stat_dims<M: Math>(&self, math: &M, name: &str) -> Vec<String> {
        let dims = StatsDims::from(math);
        <<Self::Chain<M> as SamplerStats<M>>::Stats as Storable<_>>::dims(&dims, name)
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn stat_dim_sizes<M: Math>(&self, math: &M) -> HashMap<String, u64> {
        let dims = StatsDims::from(math);
        dims.dim_sizes()
    }

    fn data_dims_all<M: Math>(&self, math: &M) -> Vec<(String, Vec<String>)> {
        self.data_names(math)
            .into_iter()
            .map(|name| (name.clone(), self.data_dims(math, &name)))
            .collect()
    }

    fn data_dims<M: Math>(&self, math: &M, name: &str) -> Vec<String> {
        <M::ExpandedVector as Storable<_>>::dims(math, name)
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn stat_coords<M: Math>(&self, math: &M) -> HashMap<String, Value> {
        let dims = StatsDims::from(math);
        dims.coords()
    }

    fn stat_event_dims<M: Math>(&self, math: &M) -> Vec<(String, Option<String>)> {
        let dims = StatsDims::from(math);
        self.stat_names(math)
            .into_iter()
            .map(|name| {
                let event_dim =
                    <<Self::Chain<M> as SamplerStats<M>>::Stats as Storable<_>>::event_dim(
                        &dims, &name,
                    )
                    .map(String::from);
                (name, event_dim)
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Progress {
    pub draw: u64,
    pub chain: u64,
    pub diverging: bool,
    pub tuning: bool,
    pub step_size: f64,
    pub num_steps: u64,
}

mod private {
    use super::{
        DiagMclmcSettings, DiagNutsSettings, FlowMclmcSettings, FlowNutsSettings,
        LowRankMclmcSettings, LowRankNutsSettings,
    };

    pub trait Sealed {}

    impl Sealed for DiagNutsSettings {}

    impl Sealed for LowRankNutsSettings {}

    impl Sealed for FlowNutsSettings {}

    impl Sealed for DiagMclmcSettings {}

    impl Sealed for LowRankMclmcSettings {}

    impl Sealed for FlowMclmcSettings {}
}

/// Drop the values whose name was not declared to the storage backend. Some backends zip the
/// incoming values against their declared columns, so an undeclared one would shift the rest.
fn retain_declared(declared: &HashSet<String>, values: &mut Vec<(&str, Option<Value>)>) {
    values.retain(|(name, _)| declared.contains(*name));
}

/// The point stats each `store_*` flag suppresses, mirroring `TransformedPoint::extract_stats`.
fn disabled_point_stats(
    store_gradient: bool,
    store_unconstrained: bool,
    store_transformed: bool,
) -> Vec<&'static str> {
    let mut names = Vec::new();
    if !store_gradient {
        names.push("gradient");
    }
    if !store_unconstrained {
        names.push("unconstrained_draw");
    }
    if !store_transformed {
        names.push("transformed_position");
        names.push("transformed_gradient");
    }
    names
}

/// Settings for the NUTS sampler
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NutsSettings<A: Debug + Copy + Default + Serialize> {
    /// The number of tuning steps, where we fit the step size and geometry.
    pub num_tune: u64,
    /// The number of draws after tuning
    pub num_draws: u64,
    /// The maximum tree depth during sampling. The number of leapfrog steps
    /// is smaller than 2 ^ maxdepth.
    pub maxdepth: u64,
    /// The minimum tree depth during sampling. The number of leapfrog steps
    /// is larger than 2 ^ mindepth.
    pub mindepth: u64,
    /// Store the gradient in the SampleStats
    pub store_gradient: bool,
    /// Store each unconstrained parameter vector in the sampler stats
    pub store_unconstrained: bool,
    /// Store the transformed gradient and value in the sampler stats
    pub store_transformed: bool,
    /// If the energy error is larger than this threshold we treat the leapfrog
    /// step as a divergence.
    pub max_energy_error: f64,
    /// Store detailed information about each divergence in the sampler stats
    pub store_divergences: bool,
    /// Settings for geometry adaptation.
    pub adapt_options: A,
    pub check_turning: bool,
    pub target_integration_time: Option<f64>,
    /// Selects the kinetic-energy form and the corresponding integrator.
    ///
    /// - [`KineticEnergyKind::Euclidean`]: standard leapfrog (default for most settings).
    /// - [`KineticEnergyKind::ExactNormal`]: geodesic leapfrog exact for a standard-normal
    ///   potential.
    /// - [`KineticEnergyKind::Microcanonical`]: isokinetic ESH-dynamics leapfrog (microcanonical
    ///   HMC); momentum is constrained to the unit sphere.
    pub trajectory_kind: KineticEnergyKind,
    pub num_chains: usize,
    pub seed: u64,
    /// Number of extra doublings to perform after reaching maxdepth. This can
    /// be used to increase the effective sample size at the cost of more
    /// expensive sampling.
    pub extra_doublings: u64,
    /// Soft clipping for gradients.
    pub gradient_clipping: Option<f64>,
}

pub type DiagNutsSettings = NutsSettings<EuclideanAdaptOptions<DiagAdaptExpSettings>>;
/// Backwards-compatible alias for [`DiagNutsSettings`].
#[deprecated(since = "0.0.0", note = "Use DiagNutsSettings instead")]
pub type DiagGradNutsSettings = DiagNutsSettings;
pub type LowRankNutsSettings = NutsSettings<EuclideanAdaptOptions<LowRankSettings>>;
pub type FlowNutsSettings = NutsSettings<FlowSettings>;
/// Backwards-compatible alias for [`FlowNutsSettings`].
#[deprecated(since = "0.0.0", note = "Use FlowNutsSettings instead")]
pub type TransformedNutsSettings = FlowNutsSettings;

/// Settings for the unadjusted Microcanonical Langevin Monte Carlo (MCLMC) sampler.
///
/// > ⚠️ **Experimental — use with caution**: The MCLMC sampler and all of its
/// > variants are highly experimental. They have not been thoroughly validated
/// > and may **not return correct posteriors**. The API, defaults, and
/// > adaptation behaviour are all subject to breaking changes at any time.
/// > Do not use these samplers in production or for results you rely on.
///
/// Step size `ε` and momentum decoherence length `L` are **constants** — no
/// adaptation of those is performed yet. The geometry is adapted during
/// warmup using the sampler-specific adaptation strategy, while the step size
/// remains fixed.
///
/// Use the type aliases [`DiagMclmcSettings`], [`LowRankMclmcSettings`], and
/// [`FlowMclmcSettings`] for concrete configurations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MclmcSettings<A: Debug + Copy + Default + Serialize> {
    /// Step size ε for the ESH leapfrog integrator.
    pub step_size: f64,
    /// Momentum decoherence length L (controls partial momentum refresh rate).
    /// Set to `f64::INFINITY` to disable momentum refresh entirely.
    pub momentum_decoherence_length: f64,
    /// Number of warmup draws.
    pub num_tune: u64,
    /// Number of sampling draws after warmup.
    pub num_draws: u64,
    /// Number of parallel chains.
    pub num_chains: usize,
    /// RNG seed.
    pub seed: u64,
    /// Maximum energy error before a step is flagged as a divergence.
    pub max_energy_error: f64,
    /// Store each unconstrained parameter vector in the sampler stats.
    pub store_unconstrained: bool,
    /// Store the gradient in the sampler stats.
    pub store_gradient: bool,
    /// Store the transformed gradient and value in the sampler stats
    pub store_transformed: bool,
    /// Store detailed information about each divergence in the sampler stats
    pub store_divergences: bool,
    /// Geometry adaptation options (step-size fields are ignored for Euclidean settings).
    pub adapt_options: A,
    /// Number of leapfrog steps per draw as a fraction of `L / ε`.
    ///
    /// The number of leapfrog steps between collector calls is:
    /// `round(subsample_frequency * L / ε).max(1)`
    ///
    /// - `1.0` (default) — one sample per full trajectory (at the final step).
    /// - `0.0` — every leapfrog step.
    /// - Values in between space samples as a fraction of the decoherence
    ///   length, so the interval scales naturally when `L` or `ε` changes.
    pub subsample_frequency: f64,
    /// When `true`, use the tree-structured step size retry on divergence:
    /// halve the step size factor and try 2 steps before doubling back.
    /// `log_weight` will include `log(step_size)` to correct for the varying
    /// sampling density. When `false`, divergences are recorded immediately
    /// without any retry and `log_weight = -energy_change`.
    pub dynamic_step_size: bool,
    /// Selects which leapfrog integrator and partial-momentum-refresh style
    /// to use.  See [`MclmcTrajectoryKind`] for the available options.
    /// Default: [`MclmcTrajectoryKind::Microcanonical`] (original MCLMC).
    pub trajectory_kind: MclmcTrajectoryKind,
    /// Fraction of `num_tune` draws at which the trajectory is switched from
    /// Euclidean to Microcanonical when
    /// `trajectory_kind == MclmcTrajectoryKind::EuclideanEarlyThenMicrocanonical`.
    /// Ignored for other trajectory kinds.  Default: `0.3`.
    pub trajectory_switch_fraction: f64,
    /// Soft clipping for gradients.
    pub gradient_clipping: Option<f64>,
}

/// MCLMC settings with a diagonal mass matrix adaptation.
///
/// > ⚠️ **Experimental — use with caution**: Highly experimental. Correctness
/// > of the returned posteriors has not been verified. May change at any time.
pub type DiagMclmcSettings = MclmcSettings<EuclideanAdaptOptions<DiagAdaptExpSettings>>;
/// MCLMC settings with a low-rank mass matrix adaptation.
///
/// > ⚠️ **Experimental — use with caution**: Highly experimental. Correctness
/// > of the returned posteriors has not been verified. May change at any time.
pub type LowRankMclmcSettings = MclmcSettings<EuclideanAdaptOptions<LowRankSettings>>;
/// MCLMC settings with a learned flow transformation.
///
/// > ⚠️ **Experimental — use with caution**: Highly experimental. Correctness
/// > of the returned posteriors has not been verified. May change at any time.
pub type FlowMclmcSettings = MclmcSettings<FlowSettings>;
/// Backwards-compatible alias for [`FlowMclmcSettings`].
#[deprecated(since = "0.0.0", note = "Use FlowMclmcSettings instead")]
pub type TransformedMclmcSettings = FlowMclmcSettings;

/// Saturates, because `validate_draw_counts` already rejects counts that do not fit.
fn usize_hint(value: u64) -> usize {
    value.try_into().unwrap_or(usize::MAX)
}

fn validate_draw_counts(num_tune: u64, num_draws: u64) -> Result<()> {
    let total = num_tune
        .checked_add(num_draws)
        .context("num_tune + num_draws is too large")?;
    usize::try_from(total).context("num_tune + num_draws does not fit into usize")?;
    Ok(())
}

fn validate_mclmc<A: Debug + Copy + Default + Serialize>(
    settings: &MclmcSettings<A>,
) -> Result<()> {
    validate_draw_counts(settings.num_tune, settings.num_draws)?;
    if !(settings.step_size.is_finite() && settings.step_size > 0.0) {
        bail!(
            "step_size must be positive and finite, got {}",
            settings.step_size
        );
    }
    Ok(())
}

/// The microcanonical (ESH) dynamics normalize the momentum onto the unit sphere, which
/// needs at least two dimensions.
fn check_microcanonical_dim(microcanonical: bool, dim: usize) -> Result<()> {
    if microcanonical && dim < 2 {
        bail!("Microcanonical dynamics need at least 2 dimensions, but the model has {dim}");
    }
    Ok(())
}

fn default_mclmc_settings<A: Debug + Copy + Default + Serialize>(
    adapt_options: A,
    num_tune: u64,
    num_chains: usize,
    max_energy_error: f64,
) -> MclmcSettings<A> {
    MclmcSettings {
        step_size: 0.5,
        momentum_decoherence_length: 3.0,
        num_tune,
        num_draws: 1000,
        num_chains,
        seed: 0,
        max_energy_error,
        store_unconstrained: false,
        store_gradient: false,
        store_divergences: false,
        store_transformed: false,
        adapt_options,
        subsample_frequency: 1.0,
        dynamic_step_size: true,
        trajectory_kind: MclmcTrajectoryKind::EuclideanEarlyThenMicrocanonical,
        trajectory_switch_fraction: 0.3,
        gradient_clipping: Some(1e10),
    }
}

impl Default for DiagMclmcSettings {
    fn default() -> Self {
        let mut adapt_options = EuclideanAdaptOptions::default();
        adapt_options.step_size_settings.adapt_options.method = StepSizeAdaptMethod::Fixed(0.5);
        default_mclmc_settings(adapt_options, 400, 6, 1000.0)
    }
}

impl Default for LowRankMclmcSettings {
    fn default() -> Self {
        let mut adapt_options = EuclideanAdaptOptions::default();
        adapt_options.early_mass_matrix_switch_freq = 20;
        adapt_options.step_size_settings.adapt_options.method = StepSizeAdaptMethod::Fixed(0.5);
        default_mclmc_settings(adapt_options, 800, 6, 1000.0)
    }
}

impl Default for FlowMclmcSettings {
    fn default() -> Self {
        default_mclmc_settings(FlowSettings::default(), 1500, 1, 20.0)
    }
}

type DiagMclmcChain<M> = crate::mclmc::MclmcChain<
    M,
    ChaCha8Rng,
    GlobalStrategy<M, DiagAdaptStrategy<M>>,
    DiagMassMatrix<M>,
>;
type LowRankMclmcChain<M> = crate::mclmc::MclmcChain<
    M,
    ChaCha8Rng,
    GlobalStrategy<M, LowRankMassMatrixStrategy>,
    LowRankMassMatrix<M>,
>;

impl Settings for DiagMclmcSettings {
    type Chain<M: Math> = DiagMclmcChain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        rng: &mut R,
    ) -> Result<Self::Chain<M>> {
        use crate::dynamics::KineticEnergyKind;
        use crate::mclmc::MclmcChain;
        use crate::stepsize::StepSizeAdaptMethod;

        self.validate()?;
        check_microcanonical_dim(
            !matches!(self.trajectory_kind, MclmcTrajectoryKind::Euclidean),
            math.dim(),
        )?;

        let num_tune = self.num_tune;
        let mut adapt_options = self.adapt_options;
        adapt_options.step_size_settings.adapt_options.method =
            StepSizeAdaptMethod::Fixed(self.step_size);
        let strategy = GlobalStrategy::<M, DiagAdaptStrategy<M>>::new(
            &mut math,
            adapt_options,
            num_tune,
            chain,
        );
        let mass_matrix = DiagMassMatrix::new(
            &mut math,
            self.adapt_options.mass_matrix_options.store_mass_matrix,
        );
        let initial_kind = match self.trajectory_kind {
            MclmcTrajectoryKind::Microcanonical => KineticEnergyKind::Microcanonical,
            MclmcTrajectoryKind::Euclidean
            | MclmcTrajectoryKind::EuclideanEarlyThenMicrocanonical => KineticEnergyKind::Euclidean,
        };
        let mut hamiltonian = TransformedHamiltonian::new(
            &mut math,
            mass_matrix,
            initial_kind,
            self.gradient_clipping,
        );
        hamiltonian.set_momentum_decoherence_length(Some(self.momentum_decoherence_length));
        let switch_draw = (self.trajectory_switch_fraction * self.num_tune as f64) as u64;
        let rng = ChaCha8Rng::try_from_rng(rng).expect("Could not seed rng");
        let stats_options = self.stats_options::<M>();
        Ok(MclmcChain::new(
            math,
            hamiltonian,
            strategy,
            rng,
            chain,
            self.subsample_frequency,
            self.dynamic_step_size,
            self.trajectory_kind,
            switch_draw,
            self.max_energy_error,
            stats_options,
        ))
    }

    fn validate(&self) -> Result<()> {
        validate_mclmc(self)?;
        self.adapt_options.validate()
    }

    fn hint_num_tune(&self) -> usize {
        usize_hint(self.num_tune)
    }

    fn hint_num_draws(&self) -> usize {
        usize_hint(self.num_draws)
    }

    fn num_chains(&self) -> usize {
        self.num_chains
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn disabled_stats(&self) -> Vec<&'static str> {
        disabled_point_stats(
            self.store_gradient,
            self.store_unconstrained,
            self.store_transformed,
        )
    }

    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions {
        StatOptions {
            adapt: GlobalStrategyStatsOptions {
                step_size: (),
                mass_matrix: (),
            },
            hamiltonian: -1,
            point: {
                let store_gradient = self.store_gradient;
                let store_unconstrained = self.store_unconstrained;
                let store_transformed = self.store_transformed;
                TransformedPointStatsOptions {
                    store_gradient,
                    store_unconstrained,
                    store_transformed,
                }
            },
            divergence: crate::dynamics::DivergenceStatsOptions {
                store_divergences: self.store_divergences,
            },
        }
    }

    fn sampler_name(&self) -> &'static str {
        "mclmc"
    }

    fn adaptation_name(&self) -> &'static str {
        "diagonal"
    }
}

fn default_nuts_settings<A: Debug + Copy + Default + Serialize>(
    adapt_options: A,
    num_tune: u64,
    num_chains: usize,
    max_energy_error: f64,
) -> NutsSettings<A> {
    NutsSettings {
        num_tune,
        num_draws: 1000,
        maxdepth: 10,
        mindepth: 0,
        max_energy_error,
        store_gradient: false,
        store_unconstrained: false,
        store_transformed: false,
        store_divergences: false,
        adapt_options,
        check_turning: true,
        seed: 0,
        num_chains,
        target_integration_time: None,
        trajectory_kind: KineticEnergyKind::Euclidean,
        extra_doublings: 0,
        gradient_clipping: Some(1e10),
    }
}

impl Settings for LowRankMclmcSettings {
    type Chain<M: Math> = LowRankMclmcChain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        rng: &mut R,
    ) -> Result<Self::Chain<M>> {
        use crate::dynamics::KineticEnergyKind;
        use crate::mclmc::MclmcChain;
        use crate::stepsize::StepSizeAdaptMethod;

        self.validate()?;
        check_microcanonical_dim(
            !matches!(self.trajectory_kind, MclmcTrajectoryKind::Euclidean),
            math.dim(),
        )?;

        let num_tune = self.num_tune;
        let mut adapt_options = self.adapt_options;
        adapt_options.step_size_settings.adapt_options.method =
            StepSizeAdaptMethod::Fixed(self.step_size);
        let strategy = GlobalStrategy::<M, LowRankMassMatrixStrategy>::new(
            &mut math,
            adapt_options,
            num_tune,
            chain,
        );
        let mass_matrix = LowRankMassMatrix::new(&mut math, self.adapt_options.mass_matrix_options);
        let initial_kind = match self.trajectory_kind {
            MclmcTrajectoryKind::Microcanonical => KineticEnergyKind::Microcanonical,
            MclmcTrajectoryKind::Euclidean
            | MclmcTrajectoryKind::EuclideanEarlyThenMicrocanonical => KineticEnergyKind::Euclidean,
        };
        let mut hamiltonian = TransformedHamiltonian::new(
            &mut math,
            mass_matrix,
            initial_kind,
            self.gradient_clipping,
        );
        hamiltonian.set_momentum_decoherence_length(Some(self.momentum_decoherence_length));
        let switch_draw = (self.trajectory_switch_fraction * self.num_tune as f64) as u64;
        let rng = ChaCha8Rng::try_from_rng(rng).expect("Could not seed rng");
        let stats_options = self.stats_options::<M>();
        Ok(MclmcChain::new(
            math,
            hamiltonian,
            strategy,
            rng,
            chain,
            self.subsample_frequency,
            self.dynamic_step_size,
            self.trajectory_kind,
            switch_draw,
            self.max_energy_error,
            stats_options,
        ))
    }

    fn validate(&self) -> Result<()> {
        validate_mclmc(self)?;
        self.adapt_options.validate()
    }

    fn hint_num_tune(&self) -> usize {
        usize_hint(self.num_tune)
    }

    fn hint_num_draws(&self) -> usize {
        usize_hint(self.num_draws)
    }

    fn num_chains(&self) -> usize {
        self.num_chains
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn disabled_stats(&self) -> Vec<&'static str> {
        disabled_point_stats(
            self.store_gradient,
            self.store_unconstrained,
            self.store_transformed,
        )
    }

    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions {
        StatOptions {
            adapt: GlobalStrategyStatsOptions {
                step_size: (),
                mass_matrix: (),
            },
            hamiltonian: -1,
            point: {
                let store_gradient = self.store_gradient;
                let store_unconstrained = self.store_unconstrained;
                let store_transformed = self.store_transformed;
                TransformedPointStatsOptions {
                    store_gradient,
                    store_unconstrained,
                    store_transformed,
                }
            },
            divergence: crate::dynamics::DivergenceStatsOptions {
                store_divergences: self.store_divergences,
            },
        }
    }

    fn sampler_name(&self) -> &'static str {
        "mclmc"
    }

    fn adaptation_name(&self) -> &'static str {
        "low_rank"
    }
}

impl Default for DiagNutsSettings {
    fn default() -> Self {
        default_nuts_settings(EuclideanAdaptOptions::default(), 400, 6, 1000.0)
    }
}

impl Default for LowRankNutsSettings {
    fn default() -> Self {
        let mut vals = default_nuts_settings(EuclideanAdaptOptions::default(), 800, 6, 1000.0);
        vals.adapt_options.mass_matrix_update_freq = 20;
        vals
    }
}

impl Default for FlowNutsSettings {
    fn default() -> Self {
        default_nuts_settings(FlowSettings::default(), 1500, 1, 20.0)
    }
}

type DiagNutsChain<M> = NutsChain<M, ChaCha8Rng, GlobalStrategy<M, DiagAdaptStrategy<M>>>;
type LowRankNutsChain<M> = NutsChain<M, ChaCha8Rng, GlobalStrategy<M, LowRankMassMatrixStrategy>>;

fn nuts_options(settings: &NutsSettings<impl Debug + Copy + Default + Serialize>) -> NutsOptions {
    NutsOptions {
        maxdepth: settings.maxdepth,
        mindepth: settings.mindepth,
        store_divergences: settings.store_divergences,
        check_turning: settings.check_turning,
        target_integration_time: settings.target_integration_time,
        extra_doublings: settings.extra_doublings,
        max_energy_error: settings.max_energy_error,
        uturn_check_first_step: matches!(settings.trajectory_kind, KineticEnergyKind::ExactNormal),
    }
}

impl Settings for LowRankNutsSettings {
    type Chain<M: Math> = LowRankNutsChain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        mut rng: &mut R,
    ) -> Result<Self::Chain<M>> {
        self.validate()?;
        check_microcanonical_dim(
            matches!(self.trajectory_kind, KineticEnergyKind::Microcanonical),
            math.dim(),
        )?;

        let num_tune = self.num_tune;
        let strategy = GlobalStrategy::new(&mut math, self.adapt_options, num_tune, chain);
        let mass_matrix = LowRankMassMatrix::new(&mut math, self.adapt_options.mass_matrix_options);
        let hamiltonian = TransformedHamiltonian::new(
            &mut math,
            mass_matrix,
            self.trajectory_kind,
            self.gradient_clipping,
        );

        let options = nuts_options(self);

        let rng = ChaCha8Rng::try_from_rng(&mut rng).expect("Could not seed rng");

        Ok(NutsChain::new(
            math,
            hamiltonian,
            strategy,
            options,
            rng,
            chain,
            self.stats_options(),
        ))
    }

    fn validate(&self) -> Result<()> {
        validate_draw_counts(self.num_tune, self.num_draws)?;
        self.adapt_options.validate()
    }

    fn hint_num_tune(&self) -> usize {
        usize_hint(self.num_tune)
    }

    fn hint_num_draws(&self) -> usize {
        usize_hint(self.num_draws)
    }

    fn num_chains(&self) -> usize {
        self.num_chains
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn disabled_stats(&self) -> Vec<&'static str> {
        disabled_point_stats(
            self.store_gradient,
            self.store_unconstrained,
            self.store_transformed,
        )
    }

    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions {
        StatOptions {
            adapt: GlobalStrategyStatsOptions {
                mass_matrix: (),
                step_size: (),
            },
            hamiltonian: -1,
            point: {
                let store_gradient = self.store_gradient;
                let store_unconstrained = self.store_unconstrained;
                let store_transformed = self.store_transformed;
                TransformedPointStatsOptions {
                    store_gradient,
                    store_unconstrained,
                    store_transformed,
                }
            },
            divergence: crate::dynamics::DivergenceStatsOptions {
                store_divergences: self.store_divergences,
            },
        }
    }

    fn sampler_name(&self) -> &'static str {
        "nuts"
    }

    fn adaptation_name(&self) -> &'static str {
        "low_rank"
    }
}

impl Settings for DiagNutsSettings {
    type Chain<M: Math> = DiagNutsChain<M>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        mut rng: &mut R,
    ) -> Result<Self::Chain<M>> {
        self.validate()?;
        check_microcanonical_dim(
            matches!(self.trajectory_kind, KineticEnergyKind::Microcanonical),
            math.dim(),
        )?;

        let num_tune = self.num_tune;
        let strategy = GlobalStrategy::new(&mut math, self.adapt_options, num_tune, chain);
        let mass_matrix = DiagMassMatrix::new(
            &mut math,
            self.adapt_options.mass_matrix_options.store_mass_matrix,
        );
        let potential = TransformedHamiltonian::new(
            &mut math,
            mass_matrix,
            self.trajectory_kind,
            self.gradient_clipping,
        );

        let options = nuts_options(self);

        let rng = ChaCha8Rng::try_from_rng(&mut rng).expect("Could not seed rng");

        Ok(NutsChain::new(
            math,
            potential,
            strategy,
            options,
            rng,
            chain,
            self.stats_options(),
        ))
    }

    fn validate(&self) -> Result<()> {
        validate_draw_counts(self.num_tune, self.num_draws)?;
        self.adapt_options.validate()
    }

    fn hint_num_tune(&self) -> usize {
        usize_hint(self.num_tune)
    }

    fn hint_num_draws(&self) -> usize {
        usize_hint(self.num_draws)
    }

    fn num_chains(&self) -> usize {
        self.num_chains
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn disabled_stats(&self) -> Vec<&'static str> {
        disabled_point_stats(
            self.store_gradient,
            self.store_unconstrained,
            self.store_transformed,
        )
    }

    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions {
        StatOptions {
            adapt: GlobalStrategyStatsOptions {
                mass_matrix: (),
                step_size: (),
            },
            hamiltonian: -1,
            point: {
                let store_gradient = self.store_gradient;
                let store_unconstrained = self.store_unconstrained;
                let store_transformed = self.store_transformed;
                TransformedPointStatsOptions {
                    store_gradient,
                    store_unconstrained,
                    store_transformed,
                }
            },
            divergence: crate::dynamics::DivergenceStatsOptions {
                store_divergences: self.store_divergences,
            },
        }
    }

    fn sampler_name(&self) -> &'static str {
        "nuts"
    }

    fn adaptation_name(&self) -> &'static str {
        "diagonal"
    }
}

impl Settings for FlowNutsSettings {
    type Chain<M: Math> = NutsChain<M, ChaCha8Rng, ExternalTransformAdaptation>;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        mut rng: &mut R,
    ) -> Result<Self::Chain<M>> {
        self.validate()?;
        check_microcanonical_dim(
            matches!(self.trajectory_kind, KineticEnergyKind::Microcanonical),
            math.dim(),
        )?;

        let num_tune = self.num_tune;

        let strategy =
            ExternalTransformAdaptation::new(&mut math, self.adapt_options, num_tune, chain);
        let params = math
            .new_transformation(rng, math.dim(), chain)
            .context("Failed to create external transformation")?;
        let transform = ExternalTransformation::new(params);
        let hamiltonian = TransformedHamiltonian::new(
            &mut math,
            transform,
            self.trajectory_kind,
            self.gradient_clipping,
        );

        let options = nuts_options(self);

        let rng = ChaCha8Rng::try_from_rng(&mut rng).expect("Could not seed rng");
        Ok(NutsChain::new(
            math,
            hamiltonian,
            strategy,
            options,
            rng,
            chain,
            self.stats_options(),
        ))
    }

    fn validate(&self) -> Result<()> {
        validate_draw_counts(self.num_tune, self.num_draws)?;
        self.adapt_options.validate()
    }

    fn hint_num_tune(&self) -> usize {
        usize_hint(self.num_tune)
    }

    fn hint_num_draws(&self) -> usize {
        usize_hint(self.num_draws)
    }

    fn num_chains(&self) -> usize {
        self.num_chains
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn disabled_stats(&self) -> Vec<&'static str> {
        disabled_point_stats(
            self.store_gradient,
            self.store_unconstrained,
            self.store_transformed,
        )
    }

    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions {
        StatOptions {
            adapt: (),
            hamiltonian: (),
            point: {
                let store_gradient = self.store_gradient;
                let store_unconstrained = self.store_unconstrained;
                let store_transformed = self.store_transformed;
                TransformedPointStatsOptions {
                    store_gradient,
                    store_unconstrained,
                    store_transformed,
                }
            },
            divergence: crate::dynamics::DivergenceStatsOptions {
                store_divergences: self.store_divergences,
            },
        }
    }

    fn sampler_name(&self) -> &'static str {
        "nuts"
    }

    fn adaptation_name(&self) -> &'static str {
        "flow"
    }
}

impl Settings for FlowMclmcSettings {
    type Chain<M: Math> = crate::mclmc::MclmcChain<
        M,
        ChaCha8Rng,
        ExternalTransformAdaptation,
        ExternalTransformation<M>,
    >;

    fn new_chain<M: Math, R: Rng + ?Sized>(
        &self,
        chain: u64,
        mut math: M,
        rng: &mut R,
    ) -> Result<Self::Chain<M>> {
        use crate::dynamics::KineticEnergyKind;
        use crate::mclmc::MclmcChain;

        self.validate()?;
        check_microcanonical_dim(
            !matches!(self.trajectory_kind, MclmcTrajectoryKind::Euclidean),
            math.dim(),
        )?;

        let num_tune = self.num_tune;
        let strategy =
            ExternalTransformAdaptation::new(&mut math, self.adapt_options, num_tune, chain);
        let params = math
            .new_transformation(rng, math.dim(), chain)
            .context("Failed to create external transformation")?;
        let transform = ExternalTransformation::new(params);
        let initial_kind = match self.trajectory_kind {
            MclmcTrajectoryKind::Microcanonical => KineticEnergyKind::Microcanonical,
            MclmcTrajectoryKind::Euclidean
            | MclmcTrajectoryKind::EuclideanEarlyThenMicrocanonical => KineticEnergyKind::Euclidean,
        };
        let mut hamiltonian =
            TransformedHamiltonian::new(&mut math, transform, initial_kind, self.gradient_clipping);
        hamiltonian.set_momentum_decoherence_length(Some(self.momentum_decoherence_length));
        let switch_draw = (self.trajectory_switch_fraction * self.num_tune as f64) as u64;
        let rng = ChaCha8Rng::try_from_rng(rng).expect("Could not seed rng");
        let stats_options = self.stats_options::<M>();
        Ok(MclmcChain::new(
            math,
            hamiltonian,
            strategy,
            rng,
            chain,
            self.subsample_frequency,
            self.dynamic_step_size,
            self.trajectory_kind,
            switch_draw,
            self.max_energy_error,
            stats_options,
        ))
    }

    fn validate(&self) -> Result<()> {
        validate_mclmc(self)?;
        self.adapt_options.validate()
    }

    fn hint_num_tune(&self) -> usize {
        usize_hint(self.num_tune)
    }

    fn hint_num_draws(&self) -> usize {
        usize_hint(self.num_draws)
    }

    fn num_chains(&self) -> usize {
        self.num_chains
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn disabled_stats(&self) -> Vec<&'static str> {
        disabled_point_stats(
            self.store_gradient,
            self.store_unconstrained,
            self.store_transformed,
        )
    }

    fn stats_options<M: Math>(&self) -> <Self::Chain<M> as SamplerStats<M>>::StatsOptions {
        StatOptions {
            adapt: (),
            hamiltonian: (),
            point: {
                let store_gradient = self.store_gradient;
                let store_unconstrained = self.store_unconstrained;
                let store_transformed = self.store_transformed;
                TransformedPointStatsOptions {
                    store_gradient,
                    store_unconstrained,
                    store_transformed,
                }
            },
            divergence: crate::dynamics::DivergenceStatsOptions {
                store_divergences: self.store_divergences,
            },
        }
    }

    fn sampler_name(&self) -> &'static str {
        "mclmc"
    }

    fn adaptation_name(&self) -> &'static str {
        "flow"
    }
}

pub fn sample_sequentially<'math, M: Math + 'math, R: Rng + ?Sized>(
    math: M,
    settings: DiagNutsSettings,
    start: &[f64],
    draws: u64,
    chain: u64,
    rng: &mut R,
) -> Result<impl Iterator<Item = Result<(Box<[f64]>, Progress)>> + 'math> {
    let mut sampler = settings.new_chain(chain, math, rng)?;
    sampler.set_position(start)?;
    Ok((0..draws).map(move |_| sampler.draw()))
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ChainProgress {
    pub finished_draws: usize,
    pub total_draws: usize,
    pub divergences: usize,
    pub tuning: bool,
    pub started: bool,
    pub latest_num_steps: usize,
    pub total_num_steps: usize,
    pub step_size: f64,
    pub runtime: Duration,
    pub divergent_draws: Vec<usize>,
}

impl ChainProgress {
    fn new(total: usize) -> Self {
        Self {
            finished_draws: 0,
            total_draws: total,
            divergences: 0,
            tuning: true,
            started: false,
            latest_num_steps: 0,
            step_size: 0f64,
            total_num_steps: 0,
            runtime: Duration::ZERO,
            divergent_draws: Vec::new(),
        }
    }

    fn update(&mut self, stats: &Progress, draw_duration: Duration) {
        if stats.diverging & !stats.tuning {
            self.divergences += 1;
            self.divergent_draws.push(self.finished_draws);
        }
        self.finished_draws += 1;
        self.tuning = stats.tuning;

        self.latest_num_steps = stats.num_steps as usize;
        self.total_num_steps += stats.num_steps as usize;
        self.step_size = stats.step_size;
        self.runtime += draw_duration;
    }
}

/// The parts of a chain the controller reads while the chain is running: the chain's
/// trace, which the controller takes at the end, and its progress.
struct ChainShared<C> {
    trace: Arc<Mutex<Option<C>>>,
    progress: Arc<Mutex<ChainProgress>>,
}

impl<C> Clone for ChainShared<C> {
    fn clone(&self) -> Self {
        Self {
            trace: self.trace.clone(),
            progress: self.progress.clone(),
        }
    }
}

impl<C: ChainStorage> ChainShared<C> {
    fn new(trace: C, total_draws: usize) -> Self {
        Self {
            trace: Arc::new(Mutex::new(Some(trace))),
            progress: Arc::new(Mutex::new(ChainProgress::new(total_draws))),
        }
    }

    fn progress(&self) -> ChainProgress {
        self.progress.lock().expect("Poisoned lock").clone()
    }

    fn flush(&self) -> Result<()> {
        self.trace
            .lock()
            .map_err(|_| anyhow::anyhow!("Could not lock trace mutex"))
            .context("Could not flush trace")?
            .as_mut()
            .map(|v| v.flush())
            .transpose()?;
        Ok(())
    }
}

fn finalize_traces<'a, T: TraceStorage>(
    trace: T,
    chains: impl IntoIterator<Item = &'a ChainShared<T::ChainStorage>>,
) -> Result<(Option<anyhow::Error>, T::Finalized)>
where
    T::ChainStorage: 'a,
{
    let finalized_chain_traces = chains
        .into_iter()
        .filter_map(|chain| chain.trace.lock().expect("Poisoned lock").take())
        .map(|chain| chain.finalize())
        .collect_vec();
    trace.finalize(finalized_chain_traces)
}

fn inspect_traces<'a, T: TraceStorage>(
    trace: &T,
    chains: impl IntoIterator<Item = &'a ChainShared<T::ChainStorage>>,
) -> Result<(Option<anyhow::Error>, T::Finalized)>
where
    T::ChainStorage: 'a,
{
    let traces = chains
        .into_iter()
        .filter_map(|chain| {
            chain
                .trace
                .lock()
                .expect("Poisoned lock")
                .as_ref()
                .map(|v| v.inspect())
        })
        .collect_vec();
    trace.inspect(traces)
}

/// A single initialized chain that records its draws into a `ChainShared`.
///
/// The threaded and the sequential driver both run chains through this; they only
/// differ in how they schedule calls to `step`.
struct ChainRunner<M: Model, S: Settings, C: ChainStorage> {
    chain: S::Chain<M::Math>,
    settings: S,
    shared: ChainShared<C>,
    declared_stats: HashSet<String>,
    declared_data: HashSet<String>,
    draws_left: usize,
}

impl<M: Model, S: Settings, C: ChainStorage> ChainRunner<M, S, C> {
    fn new(
        model: Arc<M>,
        chain_id: u64,
        seed: u64,
        settings: S,
        shared: ChainShared<C>,
    ) -> Result<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(chain_id + 1);

        let logp = Arc::clone(&model)
            .math(&mut rng)
            .context("Failed to create model density")?;
        let dim = logp.dim();

        let mut chain = settings.new_chain(chain_id, logp, &mut rng)?;

        shared.progress.lock().expect("Poisoned mutex").started = true;

        let mut initval = vec![0f64; dim];
        // TODO maxtries
        let mut error = None;
        for _ in 0..500 {
            model
                .init_position(&mut rng, &mut initval)
                .context("Failed to generate a new initial position")?;
            if let Err(err) = chain.set_position(&initval) {
                error = Some(err);
                continue;
            }
            error = None;
            break;
        }

        if let Some(error) = error {
            return Err(error.context("All initialization points failed"));
        }

        // The trace was built from these names, so anything else the chain or the
        // model produces has nowhere to go and is dropped in `step`.
        let (declared_stats, declared_data) = {
            let math = chain.math();
            (
                settings
                    .stat_names(math.deref())
                    .into_iter()
                    .collect::<HashSet<_>>(),
                settings
                    .data_names(math.deref())
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
        };

        Ok(Self {
            chain,
            settings,
            shared,
            declared_stats,
            declared_data,
            draws_left: settings.hint_num_tune() + settings.hint_num_draws(),
        })
    }

    fn is_finished(&self) -> bool {
        self.draws_left == 0
    }

    /// Draw once and record it. Returns `false` once there is nothing left to do,
    /// either because all draws are done or because the controller took the trace.
    fn step(&mut self) -> Result<bool> {
        if self.is_finished() {
            return Ok(false);
        }

        let now = Instant::now();
        let (_point, mut draw_data, mut stats, info) = self.chain.expanded_draw()?;

        let mut guard = self
            .shared
            .trace
            .lock()
            .expect("Could not unlock trace lock. Poisoned mutex");

        let Some(trace_val) = guard.as_mut() else {
            // The trace was removed by the controller. We can stop sampling
            self.draws_left = 0;
            return Ok(false);
        };
        self.shared
            .progress
            .lock()
            .expect("Poisoned mutex")
            .update(&info, now.elapsed());

        let math = self.chain.math();
        let dims = StatsDims::from(math.deref());
        let mut stat_values = stats.get_all(&dims);
        let mut draw_values = draw_data.get_all(math.deref());
        retain_declared(&self.declared_stats, &mut stat_values);
        retain_declared(&self.declared_data, &mut draw_values);
        trace_val.record_sample(&self.settings, stat_values, draw_values, &info)?;

        self.draws_left -= 1;
        Ok(!self.is_finished())
    }
}

/// Create the math once on the controller to build the trace, with the same seed stream
/// in every driver.
fn new_trace<M: Model, S: Settings, C: StorageConfig>(
    model: &Arc<M>,
    settings: &S,
    trace_config: C,
) -> Result<C::Storage> {
    let mut rng = ChaCha8Rng::seed_from_u64(settings.seed());
    rng.set_stream(0);

    let math = Arc::clone(model)
        .math(&mut rng)
        .context("Could not create model density")?;
    trace_config
        .new_trace(settings, &math)
        .context("Could not create trace object")
}

/// `Send` exactly when the `parallel` feature is on.
///
/// The sequential driver keeps its chains between calls, and chains are not `Send`
/// (the state pool uses `Rc`). Without `parallel`, `Sampler` is therefore not `Send`,
/// which keeps every chain on the thread that created it.
#[cfg(feature = "parallel")]
trait MaybeSend: Send {}
#[cfg(feature = "parallel")]
impl<T: Send> MaybeSend for T {}
#[cfg(not(feature = "parallel"))]
trait MaybeSend {}
#[cfg(not(feature = "parallel"))]
impl<T> MaybeSend for T {}

/// Runs the chains behind a `Sampler`.
///
/// `Sampler` only knows the finalized trace type `F`; the model, settings and storage
/// types live inside the driver.
trait Driver<F: Send + 'static>: MaybeSend {
    fn pause(&mut self) -> Result<()>;
    fn resume(&mut self) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
    fn inspect(&mut self) -> Result<(Option<anyhow::Error>, F)>;
    fn progress(&mut self) -> Result<Box<[ChainProgress]>>;
    fn wait_timeout(self: Box<Self>, timeout: Duration) -> SamplerWaitResult<F>;
    fn abort(self: Box<Self>) -> Result<(Option<anyhow::Error>, F)>;
}

fn finished_to_wait_result<F: Send + 'static>(
    result: Result<(Option<anyhow::Error>, F)>,
) -> SamplerWaitResult<F> {
    match result {
        Ok((Some(err), trace)) => SamplerWaitResult::Err(err, Some(trace)),
        Ok((None, trace)) => SamplerWaitResult::Trace(trace),
        Err(err) => SamplerWaitResult::Err(err, None),
    }
}

/// A chain failed with `err`, and the sampler was aborted to stop the others. Report the
/// chain's error together with everything drawn up to that point.
fn chain_failed_wait_result<F: Send + 'static>(
    err: anyhow::Error,
    aborted: Result<(Option<anyhow::Error>, F)>,
) -> SamplerWaitResult<F> {
    match aborted {
        Ok((_, trace)) => SamplerWaitResult::Err(err, Some(trace)),
        Err(abort_err) => SamplerWaitResult::Err(
            anyhow::anyhow!("{err:#}\n\nThe trace could not be finalized: {abort_err:#}"),
            None,
        ),
    }
}

#[cfg(feature = "parallel")]
enum ChainCommand {
    Resume,
    Pause,
}

#[cfg(feature = "parallel")]
struct ChainProcess<C> {
    stop_marker: Sender<ChainCommand>,
    shared: ChainShared<C>,
}

#[cfg(feature = "parallel")]
impl<C: ChainStorage> ChainProcess<C> {
    fn resume(&self) -> Result<()> {
        self.stop_marker.send(ChainCommand::Resume)?;
        Ok(())
    }

    fn pause(&self) -> Result<()> {
        self.stop_marker.send(ChainCommand::Pause)?;
        Ok(())
    }

    /// Set up the controller's handle to a chain, and the job a worker thread runs
    /// to sample it.
    fn new(
        chain_trace: C,
        chain_id: u64,
        settings: &impl Settings,
        results: Sender<Result<()>>,
    ) -> (Self, ChainJob<C>) {
        let (stop_marker_tx, stop_marker_rx) = channel();

        let shared = ChainShared::new(
            chain_trace,
            settings.hint_num_draws() + settings.hint_num_tune(),
        );

        let job = ChainJob {
            chain_id,
            shared: shared.clone(),
            commands: stop_marker_rx,
            results,
        };
        let process = Self {
            stop_marker: stop_marker_tx,
            shared,
        };
        (process, job)
    }
}

/// A chain waiting for a worker thread. Commands sent before it starts queue up in
/// its channel and are handled in order once it runs.
#[cfg(feature = "parallel")]
struct ChainJob<C> {
    chain_id: u64,
    shared: ChainShared<C>,
    commands: Receiver<ChainCommand>,
    results: Sender<Result<()>>,
}

#[cfg(feature = "parallel")]
impl<C: ChainStorage> ChainJob<C> {
    fn run<M: Model, S: Settings>(self, model: Arc<M>, settings: S) {
        let Self {
            chain_id,
            shared,
            commands,
            results,
        } = self;

        let sample = move || {
            let mut runner =
                ChainRunner::<M, S, C>::new(model, chain_id, settings.seed(), settings, shared)?;

            let mut msg = commands.try_recv();
            loop {
                match msg {
                    // The remote end is dead
                    Err(TryRecvError::Disconnected) => {
                        break;
                    }
                    Err(TryRecvError::Empty) => {}
                    Ok(ChainCommand::Pause) => {
                        msg = commands.recv().map_err(|e| e.into());
                        continue;
                    }
                    Ok(ChainCommand::Resume) => {}
                }

                if !runner.step()? {
                    break;
                }

                msg = commands.try_recv();
            }
            Ok(())
        };

        let result = sample();

        // We intentionally ignore errors here, because this means some other
        // chain already failed, and should have reported the error.
        let _ = results.send(result);
    }
}

#[cfg(feature = "parallel")]
#[derive(Debug)]
enum SamplerCommand {
    Pause,
    Continue,
    Progress,
    Flush,
    Inspect,
}

#[cfg(feature = "parallel")]
enum SamplerResponse<T: Send + 'static> {
    Ok(),
    Progress(Box<[ChainProgress]>),
    Inspect(T),
}

pub enum SamplerWaitResult<F: Send + 'static> {
    Trace(F),
    Timeout(Sampler<F>),
    Err(anyhow::Error, Option<F>),
}

/// Runs a set of chains, either on a thread pool or sequentially on the calling thread.
///
/// With the `parallel` feature, chains sample on background threads. Without it, chains
/// only advance while the caller is inside `wait_timeout`, taking turns one draw at a
/// time, and `Sampler` is not `Send`. The interface is the same in both cases.
pub struct Sampler<F: Send + 'static> {
    driver: Box<dyn Driver<F>>,
}

pub struct ProgressCallback {
    pub callback: Box<dyn FnMut(Duration, Box<[ChainProgress]>) + Send>,
    pub rate: Duration,
}

impl<F: Send + 'static> Sampler<F> {
    /// Start sampling on `num_cores` background threads.
    #[cfg(feature = "parallel")]
    pub fn new<M, S, C, T>(
        model: Arc<M>,
        settings: S,
        trace_config: C,
        num_cores: usize,
        callback: Option<ProgressCallback>,
    ) -> Result<Self>
    where
        S: Settings,
        C: StorageConfig<Storage = T>,
        M: Model,
        T: TraceStorage<Finalized = F>,
    {
        settings.validate()?;
        let driver = ThreadedDriver::new(model, settings, trace_config, num_cores, callback)?;
        Ok(Self {
            driver: Box::new(driver),
        })
    }

    /// Set up all chains on the calling thread without starting any threads.
    ///
    /// Chains only draw while the caller is inside `wait_timeout`. `num_cores` is
    /// ignored.
    #[cfg(not(feature = "parallel"))]
    pub fn new<M, S, C, T>(
        model: Arc<M>,
        settings: S,
        trace_config: C,
        _num_cores: usize,
        callback: Option<ProgressCallback>,
    ) -> Result<Self>
    where
        S: Settings,
        C: StorageConfig<Storage = T>,
        M: Model,
        T: TraceStorage<Finalized = F>,
    {
        settings.validate()?;
        let driver = SequentialDriver::new(model, settings, trace_config, callback)?;
        Ok(Self {
            driver: Box::new(driver),
        })
    }

    pub fn pause(&mut self) -> Result<()> {
        self.driver.pause()
    }

    pub fn resume(&mut self) -> Result<()> {
        self.driver.resume()
    }

    pub fn flush(&mut self) -> Result<()> {
        self.driver.flush()
    }

    pub fn inspect(&mut self) -> Result<(Option<anyhow::Error>, F)> {
        self.driver.inspect()
    }

    pub fn abort(self) -> Result<(Option<anyhow::Error>, F)> {
        self.driver.abort()
    }

    pub fn wait_timeout(self, timeout: Duration) -> SamplerWaitResult<F> {
        self.driver.wait_timeout(timeout)
    }

    pub fn progress(&mut self) -> Result<Box<[ChainProgress]>> {
        self.driver.progress()
    }
}

/// Runs every chain on the calling thread, one draw per chain in turn, while the
/// caller is inside `wait_timeout`.
#[cfg(not(feature = "parallel"))]
struct SequentialDriver<M: Model, S: Settings, T: TraceStorage> {
    trace: T,
    chains: Vec<ChainRunner<M, S, T::ChainStorage>>,
    next_chain: usize,
    paused: bool,
    callback: Option<ProgressCallback>,
    /// Time spent drawing, which is the only time the chains make progress.
    sampling_time: Duration,
    last_progress: Option<Instant>,
}

#[cfg(not(feature = "parallel"))]
impl<M: Model, S: Settings, T: TraceStorage> SequentialDriver<M, S, T> {
    fn new<C: StorageConfig<Storage = T>>(
        model: Arc<M>,
        settings: S,
        trace_config: C,
        callback: Option<ProgressCallback>,
    ) -> Result<Self> {
        let trace = new_trace(&model, &settings, trace_config)?;

        let chains = (0..settings.num_chains() as u64)
            .map(|chain_id| {
                let chain_trace = trace
                    .initialize_trace_for_chain(chain_id)
                    .context("Failed to create trace object")?;
                let shared = ChainShared::new(
                    chain_trace,
                    settings.hint_num_draws() + settings.hint_num_tune(),
                );
                ChainRunner::new(
                    Arc::clone(&model),
                    chain_id,
                    settings.seed(),
                    settings,
                    shared,
                )
            })
            .collect::<Result<Vec<_>>>()
            .context("Could not start chains")?;

        Ok(Self {
            trace,
            chains,
            next_chain: 0,
            paused: false,
            callback,
            sampling_time: Duration::ZERO,
            last_progress: None,
        })
    }

    fn report_progress(&mut self, force: bool) {
        let Some(ProgressCallback { callback, rate }) = &mut self.callback else {
            return;
        };
        if !force
            && self
                .last_progress
                .is_some_and(|last| last.elapsed() < *rate)
        {
            return;
        }
        let progress = self
            .chains
            .iter()
            .map(|chain| chain.shared.progress())
            .collect_vec();
        callback(self.sampling_time, progress.into());
        self.last_progress = Some(Instant::now());
    }

    /// The next chain after the one that drew last that still has draws left.
    fn next_unfinished(&self) -> Option<usize> {
        let n = self.chains.len();
        (0..n)
            .map(|offset| (self.next_chain + offset) % n)
            .find(|&idx| !self.chains[idx].is_finished())
    }
}

#[cfg(not(feature = "parallel"))]
impl<M, S, T> Driver<T::Finalized> for SequentialDriver<M, S, T>
where
    M: Model,
    S: Settings,
    T: TraceStorage,
{
    fn pause(&mut self) -> Result<()> {
        self.paused = true;
        Ok(())
    }

    fn resume(&mut self) -> Result<()> {
        self.paused = false;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        for chain in self.chains.iter() {
            chain.shared.flush()?;
        }
        Ok(())
    }

    fn inspect(&mut self) -> Result<(Option<anyhow::Error>, T::Finalized)> {
        inspect_traces(&self.trace, self.chains.iter().map(|chain| &chain.shared))
    }

    fn progress(&mut self) -> Result<Box<[ChainProgress]>> {
        Ok(self
            .chains
            .iter()
            .map(|chain| chain.shared.progress())
            .collect())
    }

    fn wait_timeout(mut self: Box<Self>, timeout: Duration) -> SamplerWaitResult<T::Finalized> {
        // Paused chains cannot make progress, so there is nothing to wait for.
        if self.paused {
            return SamplerWaitResult::Timeout(Sampler { driver: self });
        }

        let start = Instant::now();
        loop {
            self.report_progress(false);

            let Some(idx) = self.next_unfinished() else {
                return finished_to_wait_result(self.abort());
            };

            let draw_start = Instant::now();
            let result = self.chains[idx].step();
            self.sampling_time += draw_start.elapsed();
            self.next_chain = idx + 1;

            if let Err(err) = result {
                return chain_failed_wait_result(err, self.abort());
            }

            if start.elapsed() >= timeout {
                return SamplerWaitResult::Timeout(Sampler { driver: self });
            }
        }
    }

    fn abort(mut self: Box<Self>) -> Result<(Option<anyhow::Error>, T::Finalized)> {
        self.report_progress(true);
        let this = *self;
        finalize_traces(this.trace, this.chains.iter().map(|chain| &chain.shared))
    }
}

/// Runs the chains on `num_cores` worker threads, controlled from a separate
/// controller thread through channels.
#[cfg(feature = "parallel")]
struct ThreadedDriver<F: Send + 'static> {
    main_thread: JoinHandle<Result<(Option<anyhow::Error>, F)>>,
    commands: SyncSender<SamplerCommand>,
    responses: Receiver<SamplerResponse<(Option<anyhow::Error>, F)>>,
    results: Receiver<Result<()>>,
}

#[cfg(feature = "parallel")]
impl<F: Send + 'static> ThreadedDriver<F> {
    fn new<M, S, C, T>(
        model: Arc<M>,
        settings: S,
        trace_config: C,
        num_cores: usize,
        callback: Option<ProgressCallback>,
    ) -> Result<Self>
    where
        S: Settings,
        C: StorageConfig<Storage = T>,
        M: Model,
        T: TraceStorage<Finalized = F>,
    {
        let (commands_tx, commands_rx) = sync_channel(0);
        let (responses_tx, responses_rx) = sync_channel(0);
        let (results_tx, results_rx) = channel();

        let main_thread = spawn(move || {
            let mut callback = callback;
            let results = results_tx;
            let num_chains = settings.num_chains();

            let trace = new_trace(&model, &settings, trace_config)?;

            let mut chains = Vec::with_capacity(num_chains);
            let mut jobs = VecDeque::with_capacity(num_chains);
            for chain_id in 0..num_chains as u64 {
                let chain_trace_val = trace
                    .initialize_trace_for_chain(chain_id)
                    .context("Failed to create trace object")?;
                let (chain, job) =
                    ChainProcess::new(chain_trace_val, chain_id, &settings, results.clone());
                chains.push(chain);
                jobs.push_back(job);
            }
            drop(results);

            // Workers take chains in order, so with more chains than cores the rest
            // wait their turn.
            let jobs = Mutex::new(jobs);
            let jobs = &jobs;
            let model = &model;

            // The scope joins every worker before it returns, and re-raises their panics.
            //
            // `chains` and `trace` are moved in so that they drop before the join: a
            // paused chain only wakes up once its command sender in `chains` is gone.
            thread::scope(move |scope| {
                for worker_id in 0..num_cores.clamp(1, num_chains.max(1)) {
                    let spawned = thread::Builder::new()
                        .name(format!("nutpie-worker-{worker_id}"))
                        .spawn_scoped(scope, move || {
                            loop {
                                let job = jobs.lock().expect("Poisoned lock").pop_front();
                                let Some(job) = job else {
                                    break;
                                };
                                job.run(Arc::clone(model), settings);
                            }
                        });
                    if let Err(err) = spawned {
                        // Taking the traces stops the chains that already run, so the
                        // scope can join their workers.
                        jobs.lock().expect("Poisoned lock").clear();
                        let _ = finalize_traces(trace, chains.iter().map(|chain| &chain.shared));
                        return Err(err).context("Could not start worker thread");
                    }
                }

                let mut main_loop = || {
                    let start_time = Instant::now();
                    let mut pause_start = Instant::now();
                    let mut pause_time = Duration::ZERO;

                    let mut progress_rate = Duration::MAX;
                    if let Some(ProgressCallback { callback, rate }) = &mut callback {
                        let progress = chains
                            .iter()
                            .map(|chain| chain.shared.progress())
                            .collect_vec();
                        callback(start_time.elapsed(), progress.into());
                        progress_rate = *rate;
                    }
                    let mut last_progress = Instant::now();
                    let mut is_paused = false;

                    loop {
                        let timeout = progress_rate.checked_sub(last_progress.elapsed());
                        let timeout = timeout.unwrap_or_else(|| {
                            if let Some(ProgressCallback { callback, .. }) = &mut callback {
                                let progress = chains
                                    .iter()
                                    .map(|chain| chain.shared.progress())
                                    .collect_vec();
                                let mut elapsed = start_time.elapsed().saturating_sub(pause_time);
                                if is_paused {
                                    elapsed = elapsed.saturating_sub(pause_start.elapsed());
                                }
                                callback(elapsed, progress.into());
                            }
                            last_progress = Instant::now();
                            progress_rate
                        });

                        // TODO return when all chains are done
                        match commands_rx.recv_timeout(timeout) {
                            Ok(SamplerCommand::Pause) => {
                                for chain in chains.iter() {
                                    // This failes if the thread is done.
                                    // We just want to ignore those threads.
                                    let _ = chain.pause();
                                }
                                if !is_paused {
                                    pause_start = Instant::now();
                                }
                                is_paused = true;
                                responses_tx.send(SamplerResponse::Ok()).map_err(|e| {
                                    anyhow::anyhow!(
                                        "Could not send pause response to controller thread: {e}"
                                    )
                                })?;
                            }
                            Ok(SamplerCommand::Continue) => {
                                for chain in chains.iter() {
                                    // This failes if the thread is done.
                                    // We just want to ignore those threads.
                                    let _ = chain.resume();
                                }
                                pause_time += pause_start.elapsed();
                                is_paused = false;
                                responses_tx.send(SamplerResponse::Ok()).map_err(|e| {
                                    anyhow::anyhow!(
                                        "Could not send continue response to controller thread: {e}"
                                    )
                                })?;
                            }
                            Ok(SamplerCommand::Progress) => {
                                let progress = chains
                                    .iter()
                                    .map(|chain| chain.shared.progress())
                                    .collect_vec();
                                responses_tx.send(SamplerResponse::Progress(progress.into())).map_err(|e| {
                                    anyhow::anyhow!(
                                        "Could not send progress response to controller thread: {e}"
                                    )
                                })?;
                            }
                            Ok(SamplerCommand::Inspect) => {
                                let finalized_trace = inspect_traces(
                                    &trace,
                                    chains.iter().map(|chain| &chain.shared),
                                )?;
                                responses_tx.send(SamplerResponse::Inspect(finalized_trace)).map_err(|e| {
                                    anyhow::anyhow!(
                                        "Could not send inspect response to controller thread: {e}"
                                    )
                                })?;
                            }
                            Ok(SamplerCommand::Flush) => {
                                for chain in chains.iter() {
                                    chain.shared.flush()?;
                                }
                                responses_tx.send(SamplerResponse::Ok()).map_err(|e| {
                                    anyhow::anyhow!(
                                        "Could not send flush response to controller thread: {e}"
                                    )
                                })?;
                            }
                            Err(RecvTimeoutError::Timeout) => {}
                            Err(RecvTimeoutError::Disconnected) => {
                                if let Some(ProgressCallback { callback, .. }) = &mut callback {
                                    let progress = chains
                                        .iter()
                                        .map(|chain| chain.shared.progress())
                                        .collect_vec();
                                    let mut elapsed =
                                        start_time.elapsed().saturating_sub(pause_time);
                                    if is_paused {
                                        elapsed = elapsed.saturating_sub(pause_start.elapsed());
                                    }
                                    callback(elapsed, progress.into());
                                }
                                return Ok(());
                            }
                        };
                    }
                };
                let result: Result<()> = main_loop();
                // Chains that have not started yet should not start anymore.
                jobs.lock().expect("Poisoned lock").clear();
                // Run finalization even if something failed
                let output = finalize_traces(trace, chains.iter().map(|chain| &chain.shared))?;

                result?;
                Ok(output)
            })
        });

        Ok(Self {
            main_thread,
            commands: commands_tx,
            responses: responses_rx,
            results: results_rx,
        })
    }
}

#[cfg(feature = "parallel")]
impl<F: Send + 'static> Driver<F> for ThreadedDriver<F> {
    fn pause(&mut self) -> Result<()> {
        self.commands
            .send(SamplerCommand::Pause)
            .context("Could not send pause command to controller thread")?;
        let response = self
            .responses
            .recv()
            .context("Could not recieve pause response from controller thread")?;
        let SamplerResponse::Ok() = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(())
    }

    fn resume(&mut self) -> Result<()> {
        self.commands.send(SamplerCommand::Continue)?;
        let response = self.responses.recv()?;
        let SamplerResponse::Ok() = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.commands.send(SamplerCommand::Flush)?;
        let response = self
            .responses
            .recv()
            .context("Could not recieve flush response from controller thread")?;
        let SamplerResponse::Ok() = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(())
    }

    fn inspect(&mut self) -> Result<(Option<anyhow::Error>, F)> {
        self.commands.send(SamplerCommand::Inspect)?;
        let response = self
            .responses
            .recv()
            .context("Could not recieve inspect response from controller thread")?;
        let SamplerResponse::Inspect(trace) = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(trace)
    }

    fn abort(self: Box<Self>) -> Result<(Option<anyhow::Error>, F)> {
        drop(self.commands);
        let result = self.main_thread.join();
        match result {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(Ok(val)) => Ok(val),
            Ok(Err(err)) => Err(err),
        }
    }

    fn wait_timeout(self: Box<Self>, timeout: Duration) -> SamplerWaitResult<F> {
        // `None` if the timeout is too long to represent, which means waiting until done.
        let deadline = Instant::now().checked_add(timeout);
        loop {
            // Each chain reports once when it is done, so wait only for what is left of the
            // timeout, not the whole timeout again after every chain.
            let remaining = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                None => Duration::MAX,
            };
            match self.results.recv_timeout(remaining) {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return chain_failed_wait_result(err, self.abort()),
                // Every chain has reported
                Err(RecvTimeoutError::Disconnected) => {
                    return finished_to_wait_result(self.abort());
                }
                Err(RecvTimeoutError::Timeout) => {
                    return SamplerWaitResult::Timeout(Sampler { driver: self });
                }
            }
        }
    }

    fn progress(&mut self) -> Result<Box<[ChainProgress]>> {
        self.commands.send(SamplerCommand::Progress)?;
        let response = self.responses.recv()?;
        let SamplerResponse::Progress(progress) = response else {
            bail!("Got invalid response from sample controller thread");
        };
        Ok(progress)
    }
}

#[cfg(test)]
pub mod test_logps {
    use std::sync::Arc;

    use crate::{Model, math::CpuLogpFunc, math::CpuMath};
    use anyhow::Result;
    use rand::Rng;

    pub struct CpuModel<F> {
        logp: F,
    }

    impl<F> CpuModel<F> {
        pub fn new(logp: F) -> Self {
            Self { logp }
        }
    }

    impl<F> Model for CpuModel<F>
    where
        F: CpuLogpFunc + Clone + Send + Sync + 'static,
    {
        type Math = CpuMath<F>;

        fn math<R: Rng + ?Sized>(self: Arc<Self>, _rng: &mut R) -> Result<Self::Math> {
            Ok(CpuMath::new(self.logp.clone()))
        }

        fn init_position<R: rand::prelude::Rng + ?Sized>(
            &self,
            _rng: &mut R,
            position: &mut [f64],
        ) -> Result<()> {
            position.iter_mut().for_each(|x| *x = 0.);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::test_logps::NormalLogp;
    use crate::{
        Chain, math::CpuMath, sample_sequentially, sampler::DiagMclmcSettings,
        sampler::DiagNutsSettings, sampler::FlowMclmcSettings, sampler::FlowNutsSettings,
        sampler::LowRankMclmcSettings, sampler::LowRankNutsSettings, sampler::Settings,
    };

    use super::test_logps::CpuModel;

    use anyhow::Result;
    use itertools::Itertools;
    use pretty_assertions::assert_eq;
    use rand::{SeedableRng, rngs::StdRng};

    #[cfg(feature = "zarr")]
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    #[cfg(feature = "zarr")]
    use crate::{Sampler, ZarrConfig};

    #[cfg(feature = "zarr")]
    use zarrs::storage::store::MemoryStore;

    fn assert_settings_smoke<S: Settings>(settings: S) -> Result<()> {
        let logp = NormalLogp { dim: 4, mu: 0.1 };
        let math = CpuMath::new(&logp);
        let mut rng = StdRng::seed_from_u64(42);

        let stat_names = settings.stat_names(&math);
        let stat_types = settings.stat_types(&math);
        assert!(!stat_names.is_empty());
        assert_eq!(stat_names.len(), stat_types.len());

        let mut chain = settings.new_chain(0, math, &mut rng)?;
        chain.set_position(&vec![0.2; 4])?;
        let (_draw, _info) = chain.draw()?;
        Ok(())
    }

    /// Every settings type has to map the `store_*` flags onto the same suppressed stats,
    /// including the flow ones that `all_settings_smoke` cannot build a chain for.
    #[test]
    fn store_flags_disable_point_stats() {
        macro_rules! assert_disabled {
            ($($ty:ty),+ $(,)?) => {$({
                let mut settings = <$ty>::default();
                assert_eq!(
                    settings.disabled_stats(),
                    [
                        "gradient",
                        "unconstrained_draw",
                        "transformed_position",
                        "transformed_gradient",
                    ]
                );
                settings.store_gradient = true;
                settings.store_unconstrained = true;
                settings.store_transformed = true;
                assert!(settings.disabled_stats().is_empty());
            })+};
        }
        assert_disabled!(
            DiagNutsSettings,
            LowRankNutsSettings,
            FlowNutsSettings,
            DiagMclmcSettings,
            LowRankMclmcSettings,
            FlowMclmcSettings,
        );
    }

    #[test]
    fn all_settings_smoke() -> Result<()> {
        assert_settings_smoke(DiagNutsSettings {
            num_tune: 10,
            num_draws: 10,
            ..Default::default()
        })?;
        assert_settings_smoke(LowRankNutsSettings {
            num_tune: 10,
            num_draws: 10,
            ..Default::default()
        })?;
        assert_settings_smoke(DiagMclmcSettings {
            num_tune: 10,
            num_draws: 10,
            ..Default::default()
        })?;
        assert_settings_smoke(LowRankMclmcSettings {
            num_tune: 10,
            num_draws: 10,
            ..Default::default()
        })?;
        Ok(())
    }

    #[test]
    fn sample_chain() -> Result<()> {
        let logp = NormalLogp { dim: 10, mu: 0.1 };
        let math = CpuMath::new(&logp);
        let settings = DiagNutsSettings {
            num_tune: 100,
            num_draws: 100,
            ..Default::default()
        };
        let start = vec![0.2; 10];

        let mut rng = StdRng::seed_from_u64(42);

        let mut chain = settings.new_chain(0, math, &mut rng)?;

        let (_draw, info) = chain.draw()?;
        assert!(info.tuning);
        assert_eq!(info.draw, 0);

        let math = CpuMath::new(&logp);
        let chain = sample_sequentially(math, settings, &start, 200, 1, &mut rng).unwrap();
        let mut draws = chain.collect_vec();
        assert_eq!(draws.len(), 200);

        let draw0 = draws.remove(100).unwrap();
        let (vals, stats) = draw0;
        assert_eq!(vals.len(), 10);
        assert_eq!(stats.chain, 1);
        assert_eq!(stats.draw, 100);
        Ok(())
    }

    #[cfg(feature = "zarr")]
    #[test]
    fn sample_parallel() -> Result<()> {
        let logp = NormalLogp { dim: 100, mu: 0.1 };
        let settings = DiagNutsSettings {
            num_tune: 100,
            num_draws: 100,
            seed: 10,
            ..Default::default()
        };

        let model = CpuModel::new(logp.clone());
        let store = MemoryStore::new();

        let zarr_config = ZarrConfig::new(Arc::new(store));
        let mut sampler = Sampler::new(Arc::new(model), settings, zarr_config, 4, None)?;
        sampler.pause()?;
        sampler.pause()?;
        // TODO flush trace
        sampler.resume()?;
        let (ok, _) = sampler.abort()?;
        if let Some(err) = ok {
            Err(err)?;
        }

        let store = MemoryStore::new();
        let zarr_config = ZarrConfig::new(Arc::new(store));
        let model = CpuModel::new(logp.clone());
        let mut sampler = Sampler::new(Arc::new(model), settings, zarr_config, 4, None)?;
        sampler.pause()?;
        if let (Some(err), _) = sampler.abort()? {
            Err(err)?;
        }

        let store = MemoryStore::new();
        let zarr_config = ZarrConfig::new(Arc::new(store));
        let model = CpuModel::new(logp.clone());
        let start = Instant::now();
        let sampler = Sampler::new(Arc::new(model), settings, zarr_config, 4, None)?;

        let mut sampler = match sampler.wait_timeout(Duration::from_nanos(100)) {
            super::SamplerWaitResult::Trace(_) => {
                dbg!(start.elapsed());
                panic!("finished");
            }
            super::SamplerWaitResult::Timeout(sampler) => sampler,
            super::SamplerWaitResult::Err(_, _) => {
                panic!("error")
            }
        };

        for _ in 0..30 {
            sampler.progress()?;
        }

        match sampler.wait_timeout(Duration::from_secs(1)) {
            super::SamplerWaitResult::Trace(_) => {
                dbg!(start.elapsed());
            }
            super::SamplerWaitResult::Timeout(_) => {
                panic!("timeout")
            }
            super::SamplerWaitResult::Err(err, _) => Err(err)?,
        };

        Ok(())
    }

    #[test]
    fn sample_seq() {
        let logp = NormalLogp { dim: 10, mu: 0.1 };
        let math = CpuMath::new(&logp);
        let settings = DiagNutsSettings {
            num_tune: 100,
            num_draws: 100,
            ..Default::default()
        };
        let start = vec![0.2; 10];

        let mut rng = StdRng::seed_from_u64(42);

        let chain = sample_sequentially(math, settings, &start, 200, 1, &mut rng).unwrap();
        let mut draws = chain.collect_vec();
        assert_eq!(draws.len(), 200);

        let draw0 = draws.remove(100).unwrap();
        let (vals, stats) = draw0;
        assert_eq!(vals.len(), 10);
        assert_eq!(stats.chain, 1);
        assert_eq!(stats.draw, 100);
    }

    /// More chains than worker threads: the extra chains queue up, including their
    /// pause and resume commands, and still all finish.
    #[cfg(feature = "parallel")]
    #[test]
    fn threaded_more_chains_than_cores() -> Result<()> {
        use crate::{HashMapConfig, HashMapValue, Sampler, SamplerWaitResult};
        use std::{sync::Arc, time::Duration};

        let settings = DiagNutsSettings {
            num_tune: 50,
            num_draws: 50,
            num_chains: 5,
            seed: 5,
            ..Default::default()
        };
        let model = Arc::new(CpuModel::new(NormalLogp { dim: 5, mu: 0.1 }));
        let mut sampler = Sampler::new(model, settings, HashMapConfig::new(), 2, None)?;
        sampler.pause()?;
        sampler.resume()?;

        let trace = loop {
            match sampler.wait_timeout(Duration::from_secs(10)) {
                SamplerWaitResult::Trace(trace) => break trace,
                SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
                SamplerWaitResult::Err(err, _) => return Err(err),
            }
        };
        assert_eq!(trace.len(), 5);
        for chain in trace.iter() {
            let HashMapValue::Bool(diverging) = &chain.stats["diverging"] else {
                panic!("diverging stat should be bool");
            };
            assert_eq!(diverging.len(), 100);
        }
        Ok(())
    }

    /// Aborting while paused must not wait for the paused chains, including the ones
    /// still queued behind the worker threads.
    #[cfg(feature = "parallel")]
    #[test]
    fn threaded_abort_while_paused() -> Result<()> {
        use crate::{HashMapConfig, Sampler};
        use std::sync::Arc;

        let settings = DiagNutsSettings {
            num_tune: 1000,
            num_draws: 1000,
            num_chains: 5,
            seed: 5,
            ..Default::default()
        };
        let model = Arc::new(CpuModel::new(NormalLogp { dim: 5, mu: 0.1 }));
        let mut sampler = Sampler::new(model, settings, HashMapConfig::new(), 2, None)?;
        sampler.pause()?;
        let (err, trace) = sampler.abort()?;
        assert!(err.is_none());
        assert_eq!(trace.len(), 5);
        Ok(())
    }

    /// Bad settings and inconsistent models must produce errors instead of panics, both
    /// when building a chain directly and through `Sampler`.
    mod input_errors {
        use std::{sync::Arc, time::Duration};

        use anyhow::Result;
        use rand::{SeedableRng, rngs::StdRng};

        use super::CpuModel;
        use crate::{
            Chain, DiagMclmcSettings, DiagNutsSettings, HashMapConfig, KineticEnergyKind,
            LowRankMclmcSettings, LowRankNutsSettings, MclmcTrajectoryKind, Sampler,
            SamplerWaitResult, Settings, StepSizeAdaptMethod,
            math::CpuMath,
            math::test_logps::{ExpandMismatch, FailingLogp, MismatchedExpandLogp, NormalLogp},
            storage::{StorageConfig, TraceStorage},
        };

        /// A chain that fails stops the others, and `wait_timeout` returns what all chains
        /// have drawn together with the error.
        #[test]
        fn failing_chain_keeps_trace() {
            let settings = DiagNutsSettings {
                num_tune: 100,
                num_draws: 100,
                num_chains: 3,
                seed: 1,
                ..Default::default()
            };
            let logp = FailingLogp {
                inner: NormalLogp { dim: 3, mu: 0.1 },
                fail_after: 100,
                calls: 0,
            };
            let model = Arc::new(CpuModel::new(logp));
            let mut sampler = Sampler::new(model, settings, HashMapConfig::new(), 3, None)
                .expect("Sampler should start");
            let (err, trace) = loop {
                match sampler.wait_timeout(Duration::from_secs(10)) {
                    SamplerWaitResult::Trace(_) => panic!("sampling should fail"),
                    SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
                    SamplerWaitResult::Err(err, trace) => break (err, trace),
                }
            };
            assert!(format!("{err:#}").contains("FailOnPurpose"), "{err:#}");
            let trace = trace.expect("the draws before the failure should be kept");
            assert_eq!(trace.len(), 3);
            assert!(trace.iter().all(|chain| !chain.stats.is_empty()));
        }

        fn new_chain_error<S: Settings>(settings: S, dim: usize) -> String {
            let logp = NormalLogp { dim, mu: 0.1 };
            let mut rng = StdRng::seed_from_u64(42);
            let Err(err) = settings.new_chain(0, CpuMath::new(&logp), &mut rng) else {
                panic!("new_chain should fail");
            };
            format!("{err:#}")
        }

        #[test]
        fn invalid_settings() {
            let base = DiagNutsSettings::default();
            let mut cases = vec![];

            let mut settings = base;
            settings.adapt_options.step_size_settings.jitter = Some(0.0);
            cases.push(("jitter", settings));

            let mut settings = base;
            settings.adapt_options.step_size_settings.initial_step = 0.0;
            cases.push(("initial_step", settings));

            let mut settings = base;
            settings
                .adapt_options
                .step_size_settings
                .adapt_options
                .method = StepSizeAdaptMethod::Fixed(-1.0);
            cases.push(("Fixed step size", settings));

            let mut settings = base;
            settings.adapt_options.early_window = 1.0;
            cases.push(("early_window", settings));

            let mut settings = base;
            settings.adapt_options.mass_matrix_window_growth = 0.5;
            cases.push(("mass_matrix_window_growth", settings));

            for (field, settings) in cases {
                let err = new_chain_error(settings, 4);
                assert!(err.contains(field), "{field}: {err}");
            }

            let settings = DiagMclmcSettings {
                step_size: 0.0,
                ..Default::default()
            };
            let err = new_chain_error(settings, 4);
            assert!(err.contains("step_size"), "{err}");
        }

        #[test]
        fn sampler_rejects_invalid_settings() {
            let mut settings = DiagNutsSettings::default();
            settings.adapt_options.step_size_settings.jitter = Some(-0.1);
            let model = Arc::new(CpuModel::new(NormalLogp { dim: 4, mu: 0.1 }));
            let result = Sampler::new(model, settings, HashMapConfig::new(), 1, None);
            assert!(result.is_err());
        }

        #[test]
        fn microcanonical_needs_two_dims() {
            let settings = DiagNutsSettings {
                trajectory_kind: KineticEnergyKind::Microcanonical,
                ..Default::default()
            };
            assert!(new_chain_error(settings, 1).contains("2 dimensions"));

            let settings = DiagMclmcSettings {
                trajectory_kind: MclmcTrajectoryKind::Microcanonical,
                ..Default::default()
            };
            assert!(new_chain_error(settings, 1).contains("2 dimensions"));
        }

        fn sample_without_tuning<S: Settings>(settings: S) -> Result<()> {
            let logp = NormalLogp { dim: 4, mu: 0.1 };
            let mut rng = StdRng::seed_from_u64(42);
            let mut chain = settings.new_chain(0, CpuMath::new(&logp), &mut rng)?;
            chain.set_position(&[0.2; 4])?;
            for _ in 0..10 {
                let (_draw, info) = chain.draw()?;
                assert!(!info.tuning);
            }
            Ok(())
        }

        #[test]
        fn no_tuning() -> Result<()> {
            sample_without_tuning(DiagNutsSettings {
                num_tune: 0,
                ..Default::default()
            })?;
            sample_without_tuning(LowRankNutsSettings {
                num_tune: 0,
                ..Default::default()
            })?;
            sample_without_tuning(DiagMclmcSettings {
                num_tune: 0,
                ..Default::default()
            })?;
            sample_without_tuning(LowRankMclmcSettings {
                num_tune: 0,
                ..Default::default()
            })?;
            Ok(())
        }

        #[test]
        fn wrong_initial_position_length() -> Result<()> {
            let logp = NormalLogp { dim: 4, mu: 0.1 };
            let mut rng = StdRng::seed_from_u64(42);
            let mut chain =
                DiagNutsSettings::default().new_chain(0, CpuMath::new(&logp), &mut rng)?;
            let err = chain.set_position(&[0.2; 3]).unwrap_err();
            assert!(format!("{err}").contains("length 3"), "{err}");
            Ok(())
        }

        /// Sample a model whose expanded draws do not match their declaration, and return
        /// the error the sampler reports.
        fn sample_error<C, T>(config: C, mismatch: ExpandMismatch) -> String
        where
            C: StorageConfig<Storage = T>,
            T: TraceStorage,
        {
            let settings = DiagNutsSettings {
                num_tune: 5,
                num_draws: 5,
                num_chains: 1,
                seed: 1,
                ..Default::default()
            };
            let logp = MismatchedExpandLogp {
                inner: NormalLogp { dim: 3, mu: 0.1 },
                mismatch,
            };
            let mut sampler =
                Sampler::new(Arc::new(CpuModel::new(logp)), settings, config, 1, None)
                    .expect("Sampler should start");
            loop {
                match sampler.wait_timeout(Duration::from_secs(10)) {
                    SamplerWaitResult::Trace(_) => panic!("sampling should fail for {mismatch:?}"),
                    SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
                    SamplerWaitResult::Err(err, _) => return format!("{err:#}"),
                }
            }
        }

        fn assert_explains(err: &str, mismatch: ExpandMismatch) {
            let expected = match mismatch {
                ExpandMismatch::WrongType => "Got a F32 value",
                ExpandMismatch::WrongLength => "values",
                ExpandMismatch::Missing => "no value for posterior variable x",
            };
            assert!(err.contains(expected), "{mismatch:?}: {err}");
            assert!(
                err.contains("posterior variable x") || err.contains("/x"),
                "error should name the variable: {err}"
            );
        }

        #[test]
        fn hashmap_storage_mismatch() {
            for mismatch in [
                ExpandMismatch::WrongType,
                ExpandMismatch::WrongLength,
                ExpandMismatch::Missing,
            ] {
                assert_explains(&sample_error(HashMapConfig::new(), mismatch), mismatch);
            }
        }

        #[cfg(feature = "ndarray")]
        #[test]
        fn ndarray_storage_mismatch() {
            use crate::NdarrayConfig;
            for mismatch in [
                ExpandMismatch::WrongType,
                ExpandMismatch::WrongLength,
                ExpandMismatch::Missing,
            ] {
                assert_explains(&sample_error(NdarrayConfig::new(), mismatch), mismatch);
            }
        }

        #[cfg(feature = "zarr")]
        #[test]
        fn zarr_storage_mismatch() {
            use crate::ZarrConfig;
            use zarrs::storage::store::MemoryStore;
            for mismatch in [
                ExpandMismatch::WrongType,
                ExpandMismatch::WrongLength,
                ExpandMismatch::Missing,
            ] {
                let config = ZarrConfig::new(Arc::new(MemoryStore::new()));
                assert_explains(&sample_error(config, mismatch), mismatch);
            }
        }

        #[cfg(feature = "arrow")]
        #[test]
        fn arrow_storage_mismatch() {
            use crate::ArrowConfig;
            // Arrow stores a missing draw as null.
            for mismatch in [ExpandMismatch::WrongType, ExpandMismatch::WrongLength] {
                assert_explains(&sample_error(ArrowConfig::default(), mismatch), mismatch);
            }
        }
    }

    /// Tests for the sequential driver, which only exists without `parallel`:
    /// `cargo test --no-default-features`.
    #[cfg(not(feature = "parallel"))]
    mod sequential {
        use std::{
            sync::{
                Arc,
                atomic::{AtomicUsize, Ordering},
            },
            time::Duration,
        };

        use anyhow::Result;
        use itertools::Itertools;

        use super::CpuModel;
        use crate::{
            DiagNutsSettings, HashMapConfig, ProgressCallback, Sampler, SamplerWaitResult,
            math::test_logps::NormalLogp, storage::HashMapResult,
        };

        fn sample_to_end(
            mut sampler: Sampler<Vec<HashMapResult>>,
            timeout: Duration,
        ) -> Result<Vec<HashMapResult>> {
            loop {
                match sampler.wait_timeout(timeout) {
                    SamplerWaitResult::Trace(trace) => return Ok(trace),
                    SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
                    SamplerWaitResult::Err(err, _) => return Err(err),
                }
            }
        }

        /// Debug-print the draws in a stable order, so traces can be compared exactly.
        fn draws_fingerprint(trace: &[HashMapResult]) -> Vec<String> {
            trace
                .iter()
                .map(|chain| {
                    chain
                        .draws
                        .iter()
                        .sorted_by_key(|(name, _)| name.as_str())
                        .map(|(name, value)| format!("{name}: {value:?}"))
                        .join("\n")
                })
                .collect()
        }

        fn new_sampler(callback: Option<ProgressCallback>) -> Result<Sampler<Vec<HashMapResult>>> {
            let settings = DiagNutsSettings {
                num_tune: 50,
                num_draws: 50,
                num_chains: 3,
                seed: 5,
                ..Default::default()
            };
            let model = Arc::new(CpuModel::new(NormalLogp { dim: 5, mu: 0.1 }));
            Sampler::new(model, settings, HashMapConfig::new(), 1, callback)
        }

        #[test]
        fn runs_all_chains() -> Result<()> {
            let trace = sample_to_end(new_sampler(None)?, Duration::from_millis(1))?;
            assert_eq!(trace.len(), 3);
            for chain in trace.iter() {
                let crate::HashMapValue::Bool(diverging) = &chain.stats["diverging"] else {
                    panic!("diverging stat should be bool");
                };
                assert_eq!(diverging.len(), 100);
            }
            Ok(())
        }

        /// Each chain has its own seed stream, so how draws are interleaved across
        /// `wait_timeout` calls must not change them.
        #[test]
        fn draws_do_not_depend_on_scheduling() -> Result<()> {
            let one_draw_per_call = sample_to_end(new_sampler(None)?, Duration::ZERO)?;
            let all_at_once = sample_to_end(new_sampler(None)?, Duration::from_secs(600))?;
            assert_eq!(
                draws_fingerprint(&one_draw_per_call),
                draws_fingerprint(&all_at_once)
            );
            Ok(())
        }

        #[test]
        fn pause_progress_and_inspect() -> Result<()> {
            let calls = Arc::new(AtomicUsize::new(0));
            let callback = ProgressCallback {
                callback: Box::new({
                    let calls = calls.clone();
                    move |_elapsed, _progress| {
                        calls.fetch_add(1, Ordering::Relaxed);
                    }
                }),
                rate: Duration::from_millis(1),
            };
            let mut sampler = new_sampler(Some(callback))?;

            // Nothing draws until `wait_timeout`, and not at all while paused.
            sampler.pause()?;
            let SamplerWaitResult::Timeout(mut sampler) =
                sampler.wait_timeout(Duration::from_millis(50))
            else {
                panic!("paused sampler should time out");
            };
            assert!(
                sampler
                    .progress()?
                    .iter()
                    .all(|chain| chain.finished_draws == 0)
            );
            sampler.resume()?;

            let SamplerWaitResult::Timeout(mut sampler) = sampler.wait_timeout(Duration::ZERO)
            else {
                panic!("a single draw should not finish sampling");
            };
            let drawn: usize = sampler
                .progress()?
                .iter()
                .map(|chain| chain.finished_draws)
                .sum();
            assert_eq!(drawn, 1);

            let (err, partial) = sampler.inspect()?;
            assert!(err.is_none());
            assert_eq!(partial.len(), 3);

            let trace = sample_to_end(sampler, Duration::from_millis(1))?;
            assert_eq!(trace.len(), 3);
            assert!(calls.load(Ordering::Relaxed) >= 2);
            Ok(())
        }
    }
}
