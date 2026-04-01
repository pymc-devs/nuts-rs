//! Hamiltonian dynamics: leapfrog integration, phase-space state management,
//! and the transformed Hamiltonian that operates in a whitened parameter space.

mod hamiltonian;
mod state;
mod transformed_hamiltonian;

pub use hamiltonian::DivergenceInfo;
pub use hamiltonian::DivergenceStats;
pub use hamiltonian::Hamiltonian;
pub use hamiltonian::Point;
pub use hamiltonian::{Direction, LeapfrogResult};
pub use state::{State, StatePool};
pub use transformed_hamiltonian::TransformedHamiltonian;
pub use transformed_hamiltonian::{
    KineticEnergyKind, TransformedPoint, TransformedPointStatsOptions,
};
