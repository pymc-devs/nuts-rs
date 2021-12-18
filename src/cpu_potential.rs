use std::fmt::Debug;

use itertools::{izip, Itertools};

use crate::cpu_state::{InnerState, State};
use crate::math::vector_dot;
use crate::nuts::DivergenceInfo;

pub trait CpuLogpFunc {
    type Err: Debug + Send + 'static;

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::Err>;
    fn dim(&self) -> usize;
}

pub(crate) trait MassMatrix {
    fn update_velocity(&self, state: &mut InnerState);
    fn update_kinetic_energy(&self, state: &mut InnerState);
    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut InnerState, rng: &mut R);
}

#[derive(Debug)]
pub(crate) struct DiagMassMatrix {
    inv_stds: Box<[f64]>,
    pub(crate) variance: Box<[f64]>,
}

impl DiagMassMatrix {
    pub(crate) fn new(variance: Box<[f64]>) -> Self {
        DiagMassMatrix {
            inv_stds: variance.iter().map(|x| 1. / x.sqrt()).collect_vec().into(),
            variance,
        }
    }

    pub(crate) fn update_diag(&mut self, new_variance: impl Iterator<Item = f64>) {
        izip!(
            self.variance.iter_mut(),
            self.inv_stds.iter_mut(),
            new_variance
        )
        .for_each(|(var, inv_std, x)| {
            *var = x;
            *inv_std = (1. / x).sqrt();
        });
    }
}

impl MassMatrix for DiagMassMatrix {
    fn update_velocity(&self, state: &mut InnerState) {
        //axpy_out(&self.variance, &state.p, 1., &mut state.v);
        izip!(state.v.iter_mut(), self.variance.iter(), state.p.iter()).for_each(
            |(out, &var, &p)| {
                *out = var * p;
            },
        );
    }

    fn update_kinetic_energy(&self, state: &mut InnerState) {
        state.kinetic_energy = 0.5 * vector_dot(&state.p, &state.v);
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut InnerState, rng: &mut R) {
        let dist = rand_distr::StandardNormal;
        state
            .p
            .iter_mut()
            .zip(self.inv_stds.iter())
            .for_each(|(p, &s)| {
                let norm: f64 = rng.sample(dist);
                *p = s * norm;
            });
    }
}

#[derive(Debug)]
pub struct DivergenceInfoImpl<E: Send> {
    pub logp_function_error: Option<E>,
    start: Option<InnerState>,
    right: Option<InnerState>,
    energy_error: Option<f64>,
}

impl<E: Debug + Send> DivergenceInfo for DivergenceInfoImpl<E> {
    fn start_location(&self) -> Option<&[f64]> {
        Some(&self.start.as_ref()?.q)
    }

    fn end_location(&self) -> Option<&[f64]> {
        Some(&self.right.as_ref()?.q)
    }

    fn energy_error(&self) -> Option<f64> {
        Some(self.energy_error?)
    }

    fn end_idx_in_trajectory(&self) -> Option<i64> {
        Some(self.right.as_ref()?.idx_in_trajectory)
    }
}

pub(crate) struct Potential<F: CpuLogpFunc, M: MassMatrix> {
    logp: F,
    mass_matrix: M,
}

impl<F: CpuLogpFunc, M: MassMatrix> Potential<F, M> {
    pub(crate) fn new(logp: F, mass_matrix: M) -> Self {
        Potential { logp, mass_matrix }
    }

    pub(crate) fn mass_matrix_mut(&mut self) -> &mut M {
        &mut self.mass_matrix
    }
}

impl<F: CpuLogpFunc, M: MassMatrix> crate::nuts::Potential for Potential<F, M> {
    type State = State;
    type DivergenceInfo = DivergenceInfoImpl<F::Err>;

    fn update_potential_gradient(&mut self, state: &mut State) -> Result<(), Self::DivergenceInfo> {
        // TODO can we avoid the second try_mut_inner?
        let func_return = {
            let inner = state.try_mut_inner().unwrap();
            self.logp.logp(&inner.q, &mut inner.grad)
        };

        let logp = func_return.map_err(|err| DivergenceInfoImpl {
            logp_function_error: Some(err),
            start: Some(state.clone_inner()),
            right: None,
            energy_error: None,
        })?;

        let inner = state.try_mut_inner().unwrap();
        inner.potential_energy = -logp;
        Ok(())
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut Self::State, rng: &mut R) {
        let inner = state.try_mut_inner().unwrap();
        self.mass_matrix.randomize_momentum(inner, rng);
        self.mass_matrix.update_velocity(inner);
        self.mass_matrix.update_kinetic_energy(inner);
        inner.idx_in_trajectory = 0;
        inner.p_sum.copy_from_slice(&inner.p);
    }

    fn update_velocity(&mut self, state: &mut Self::State) {
        self.mass_matrix
            .update_velocity(state.try_mut_inner().expect("State already in us"))
    }

    fn update_kinetic_energy(&mut self, state: &mut Self::State) {
        self.mass_matrix
            .update_kinetic_energy(state.try_mut_inner().expect("State already in us"))
    }

    fn new_divergence_info(
        &mut self,
        left: Self::State,
        end: Self::State,
        energy_error: f64,
    ) -> Self::DivergenceInfo {
        DivergenceInfoImpl {
            logp_function_error: None,
            start: Some(left.clone_inner()),
            right: Some(end.clone_inner()),
            energy_error: Some(energy_error),
        }
    }
}
