use std::collections::VecDeque;

use arrow::array::StructArray;
use faer::{Col, Mat, Scale};

use crate::{
    mass_matrix::{DrawGradCollector, MassMatrix},
    mass_matrix_adapt::MassMatrixAdaptStrategy,
    nuts::{AdaptStats, AdaptStrategy, SamplerStats, StatTraceBuilder},
    potential::EuclideanPotential,
    state::State,
    Math,
};

#[derive(Debug)]
struct InnerMatrix<M: Math> {
    vecs: M::EigVectors,
    vals: M::EigValues,
    vals_inv: M::EigValues,
}

#[derive(Debug)]
pub struct LowRankMassMatrix<M: Math> {
    variance: M::Vector,
    inv_stds: M::Vector,
    store_mass_matrix: bool,
    inner: Option<InnerMatrix<M>>,
}

impl<M: Math> LowRankMassMatrix<M> {
    pub fn new(math: &mut M, store_mass_matrix: bool) -> Self {
        Self {
            variance: math.new_array(),
            inv_stds: math.new_array(),
            store_mass_matrix,
            inner: None,
        }
    }

    fn update_scale(
        &mut self,
        math: &mut M,
        grad: &<M as Math>::Vector,
        fill_invalid: f64,
        clamp: (f64, f64),
    ) {
        math.array_update_var_inv_std_grad(
            &mut self.variance,
            &mut self.inv_stds,
            grad,
            fill_invalid,
            clamp,
        );
    }
}

#[derive(Clone, Debug, Copy, Default)]
pub struct LowRankSettings {
    pub store_mass_matrix: bool,
}

impl<M: Math> SamplerStats<M> for LowRankMassMatrix<M> {
    type Stats = ();
    type Builder = ();

    fn new_builder(&self, settings: &impl crate::Settings, dim: usize) -> Self::Builder {}

    fn current_stats(&self, math: &mut M) -> Self::Stats {}
}

impl<M: Math> MassMatrix<M> for LowRankMassMatrix<M> {
    fn update_velocity(&self, math: &mut M, state: &mut crate::state::InnerState<M>) {
        let Some(inner) = self.inner.as_ref() else {
            math.array_mult(&self.variance, &state.p, &mut state.v);
            return;
        };

        math.array_mult_eigs(&self.stds, &state.p, &mut state.v, &inner.vecs, &inner.vals);
    }

    fn update_kinetic_energy(&self, math: &mut M, state: &mut crate::state::InnerState<M>) {
        state.kinetic_energy = 0.5 * math.array_vector_dot(&state.p, &state.v);
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut crate::state::InnerState<M>,
        rng: &mut R,
    ) {
        let Some(inner) = self.inner.as_ref() else {
            math.array_gaussian(rng, &mut state.p, &self.inv_stds);
            return;
        };

        math.array_gaussian_eigs(rng, &mut state.p, &inner.vals, &inner.vecs);
    }
}

#[derive(Debug, Clone)]
pub struct Stats {
    foreground_length: u64,
    background_length: u64,
    //is_update: bool,
    //diag: Box<[f64]>,
    //eigvalues: Box<[f64]>,
    //eigvectors: Box<[f64]>,
}

#[derive(Debug)]
pub struct Builder {}

impl StatTraceBuilder<Stats> for Builder {
    fn append_value(&mut self, value: Stats) {
        let Stats {
            foreground_length,
            background_length,
        } = value;
    }

    fn finalize(self) -> Option<StructArray> {
        None
    }

    fn inspect(&self) -> Option<StructArray> {
        None
    }
}

#[derive(Debug)]
pub struct LowRankMassMatrixStrategy {
    draws: VecDeque<Vec<f64>>,
    grads: VecDeque<Vec<f64>>,
    ndim: usize,
    background_split: usize,
}

impl LowRankMassMatrixStrategy {
    pub fn new(ndim: usize) -> Self {
        let draws = VecDeque::with_capacity(100);
        let grads = VecDeque::with_capacity(100);

        Self {
            draws,
            grads,
            ndim,
            background_split: 0,
        }
    }

    pub fn add_draw<M: Math>(&mut self, math: &mut M, state: &State<M>) {
        assert!(math.dim() == self.ndim);
        let mut draw = vec![0f64; self.ndim];
        math.write_to_slice(&state.q, &mut draw);
        let mut grad = vec![0f64; self.ndim];
        math.write_to_slice(&state.grad, &mut grad);

        self.draws.push_back(draw);
        self.grads.push_back(grad);
    }

    pub fn clear(&mut self) {
        self.draws.clear();
        self.grads.clear();
    }

    pub fn update<M: Math>(&mut self, math: &mut M, matrix: &mut LowRankMassMatrix<M>, gamma: f64) {
        let draws_vec = &self.draws;
        let grads_vec = &self.grads;

        let ndraws = draws_vec.len();
        assert!(grads_vec.len() == ndraws);

        let mut draws: Mat<f64> = Mat::zeros(self.ndim, ndraws);
        let mut grads: Mat<f64> = Mat::zeros(self.ndim, ndraws);

        for (i, (draw, grad)) in draws_vec.iter().zip(grads_vec.iter()).enumerate() {
            draws.col_as_slice_mut(i).copy_from_slice(&draw[..]);
            grads.col_as_slice_mut(i).copy_from_slice(&grad[..]);
        }

        // Compute diagonal approximation and transform draws and grads
        let stds = Col::from_fn(self.ndim, |col| {
            let draw_mean = draws.col(1).sum() / (self.ndim as f64);
            let grad_mean = grads.col(col).sum() / (self.ndim as f64);
            let draw_std: f64 = draws
                .col(col)
                .iter()
                .map(|&val| (val - draw_mean) * (val - draw_mean))
                .sum::<f64>()
                .sqrt();
            let grad_std: f64 = grads
                .col(col)
                .iter()
                .map(|&val| (val - grad_mean) * (val - grad_mean))
                .sum::<f64>()
                .sqrt();

            let std = (draw_std / grad_std).sqrt();

            let draw_scale = (std * (ndraws as f64)).recip();
            draws
                .col_mut(col)
                .iter_mut()
                .for_each(|val| *val = (*val - draw_mean) * draw_scale);

            let grad_scale = std * (ndraws as f64).recip();
            grads
                .col_mut(col)
                .iter_mut()
                .for_each(|val| *val = (*val - grad_mean) * grad_scale);

            std
        });

        let svd_draws = draws.thin_svd();
        let svd_grads = grads.thin_svd();

        let subspace = faer::concat![[svd_draws.v(), svd_grads.v()]];

        let subspace_qr = subspace.col_piv_qr();

        let subspace_basis = subspace_qr.compute_thin_q();

        let draws_proj = draws * (&subspace_basis);
        let grads_proj = grads * subspace_basis;

        let (vals, vecs) = estimate_mass_matrix(draws_proj, grads_proj, gamma);

        todo!()
    }
}

fn estimate_mass_matrix(draws: Mat<f64>, grads: Mat<f64>, gamma: f64) -> (Col<f64>, Mat<f64>) {
    let mut cov_draws = (&draws) * draws.transpose();
    let mut cov_grads = (&grads) * grads.transpose();

    cov_draws *= Scale(gamma.recip());
    cov_grads *= Scale(gamma.recip());

    cov_draws
        .diagonal_mut()
        .column_vector_mut()
        .iter_mut()
        .for_each(|x| *x += 1f64);
    cov_grads
        .diagonal_mut()
        .column_vector_mut()
        .iter_mut()
        .for_each(|x| *x += 1f64);

    let eigs_grads = cov_grads.selfadjoint_eigendecomposition(faer::Side::Lower);

    let u = eigs_grads.u();
    let mut eigs = eigs_grads.s().column_vector().to_owned();
    eigs.iter_mut().for_each(|val| *val = val.sqrt());

    let cov_grads_sqrt = (&u) * eigs.column_vector_into_diagonal() * u;

    let m = (&cov_grads_sqrt) * cov_draws * cov_grads_sqrt;

    todo!()
}

impl<M: Math> AdaptStats<M> for LowRankMassMatrixStrategy {
    fn num_grad_evals(stats: &Self::Stats) -> usize {
        unimplemented!()
    }
}

impl<M: Math> SamplerStats<M> for LowRankMassMatrixStrategy {
    type Stats = Stats;

    type Builder = Builder;

    fn new_builder(&self, _settings: &impl crate::Settings, _dim: usize) -> Self::Builder {
        Builder {}
    }

    fn current_stats(&self, _math: &mut M) -> Self::Stats {
        Stats {
            foreground_length: self.current_count(),
            background_length: self.background_count(),
        }
    }
}

impl<M: Math> AdaptStrategy<M> for LowRankMassMatrixStrategy {
    type Potential = EuclideanPotential<M, LowRankMassMatrix<M>>;

    type Collector = DrawGradCollector<M>;

    type Options = LowRankSettings;

    fn new(math: &mut M, _options: Self::Options, _num_tune: u64) -> Self {
        Self::new(math.dim())
    }

    fn init<R: rand::Rng + ?Sized>(
        &mut self,
        math: &mut M,
        _options: &mut crate::nuts::NutsOptions,
        potential: &mut Self::Potential,
        state: &State<M>,
        _rng: &mut R,
    ) {
        self.add_draw(math, state);
        potential
            .mass_matrix
            .update_scale(math, &state.grad, 1f64, (1e-20, 1e20))
    }

    fn adapt<R: rand::Rng + ?Sized>(
        &mut self,
        _math: &mut M,
        _options: &mut crate::nuts::NutsOptions,
        _potential: &mut Self::Potential,
        _draw: u64,
        _collector: &Self::Collector,
        _state: &State<M>,
        _rng: &mut R,
    ) {
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        DrawGradCollector::new(math)
    }

    fn is_tuning(&self) -> bool {
        unreachable!()
    }
}

impl<M: Math> MassMatrixAdaptStrategy<M> for LowRankMassMatrixStrategy {
    type MassMatrix = LowRankMassMatrix<M>;

    fn update_estimators(&mut self, math: &mut M, collector: &Self::Collector) {
        if collector.is_good {
            let mut draw = vec![0f64; self.ndim];
            math.write_to_slice(&collector.draw, &mut draw);
            self.draws.push_back(draw);

            let mut grad = vec![0f64; self.ndim];
            math.write_to_slice(&collector.grad, &mut grad);
            self.grads.push_back(grad);
        }
    }

    fn switch(&mut self, _math: &mut M) {
        for _ in 0..self.background_split {
            self.draws.pop_front().expect("Could not drop draw");
            self.grads.pop_front().expect("Could not drop gradient");
        }
        self.background_split = self.draws.len();
        assert!(self.draws.len() == self.grads.len());
    }

    fn current_count(&self) -> u64 {
        self.draws.len() as u64
    }

    fn background_count(&self) -> u64 {
        self.draws.len().checked_sub(self.background_split).unwrap() as u64
    }

    fn update_potential(&self, math: &mut M, potential: &mut Self::Potential) -> bool {
        todo!()
    }
}
