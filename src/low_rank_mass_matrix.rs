use std::collections::VecDeque;

use arrow::{
    array::{ArrayBuilder, FixedSizeListBuilder, ListBuilder, PrimitiveBuilder, StructArray},
    datatypes::{Field, Float64Type, UInt64Type},
};
use faer::{Col, Mat, Scale};
use itertools::Itertools;

use crate::{
    chain::AdaptStrategy,
    euclidean_hamiltonian::{EuclideanHamiltonian, EuclideanPoint},
    hamiltonian::{Hamiltonian, Point},
    mass_matrix::{DrawGradCollector, MassMatrix},
    mass_matrix_adapt::MassMatrixAdaptStrategy,
    sampler_stats::{SamplerStats, StatTraceBuilder},
    state::State,
    Math, NutsError,
};

#[derive(Debug)]
struct InnerMatrix<M: Math> {
    vecs: M::EigVectors,
    vals: M::EigValues,
    vals_sqrt_inv: M::EigValues,
    num_eigenvalues: u64,
}

impl<M: Math> InnerMatrix<M> {
    fn new(math: &mut M, mut vals: Col<f64>, vecs: Mat<f64>) -> Self {
        let vecs = math.new_eig_vectors(
            vecs.col_iter()
                .map(|col| col.try_as_slice().expect("Array not contiguous")),
        );
        let vals_math = math.new_eig_values(vals.as_slice());

        vals.iter_mut().for_each(|x| *x = x.sqrt().recip());
        let vals_inv_math = math.new_eig_values(vals.as_slice());

        Self {
            vecs,
            vals: vals_math,
            vals_sqrt_inv: vals_inv_math,
            num_eigenvalues: vals.nrows() as u64,
        }
    }
}

#[derive(Debug)]
pub struct LowRankMassMatrix<M: Math> {
    variance: M::Vector,
    stds: M::Vector,
    inv_stds: M::Vector,
    inner: Option<InnerMatrix<M>>,
    settings: LowRankSettings,
}

impl<M: Math> LowRankMassMatrix<M> {
    pub fn new(math: &mut M, settings: LowRankSettings) -> Self {
        Self {
            variance: math.new_array(),
            inv_stds: math.new_array(),
            stds: math.new_array(),
            settings,
            inner: None,
        }
    }

    fn update_from_grad(
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
        let mut vals = vec![0f64; math.dim()];
        math.write_to_slice(&self.inv_stds, &mut vals);
        vals.iter_mut().for_each(|x| *x = x.recip());
        math.read_from_slice(&mut self.stds, &vals);
    }

    fn update(&mut self, math: &mut M, mut stds: Col<f64>, vals: Col<f64>, vecs: Mat<f64>) {
        math.read_from_slice(&mut self.stds, stds.as_slice());

        stds.iter_mut().for_each(|x| *x = x.recip());
        math.read_from_slice(&mut self.inv_stds, stds.as_slice());

        stds.iter_mut().for_each(|x| *x = x.recip() * x.recip());
        math.read_from_slice(&mut self.variance, stds.as_slice());

        if vals.is_all_finite() & vecs.is_all_finite() {
            self.inner = Some(InnerMatrix::new(math, vals, vecs));
        } else {
            self.inner = None;
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct LowRankSettings {
    pub store_mass_matrix: bool,
    pub gamma: f64,
    pub eigval_cutoff: f64,
}

impl Default for LowRankSettings {
    fn default() -> Self {
        Self {
            store_mass_matrix: false,
            gamma: 1e-5,
            eigval_cutoff: 100f64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatrixStats {
    eigenvals: Option<Box<[f64]>>,
    stds: Option<Box<[f64]>>,
    num_eigenvalues: u64,
}

pub struct MatrixBuilder {
    eigenvals: Option<ListBuilder<PrimitiveBuilder<Float64Type>>>,
    stds: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    num_eigenvalues: PrimitiveBuilder<UInt64Type>,
}

impl StatTraceBuilder<MatrixStats> for MatrixBuilder {
    fn append_value(&mut self, value: MatrixStats) {
        let MatrixStats {
            eigenvals,
            stds,
            num_eigenvalues,
        } = value;

        if let Some(store) = self.eigenvals.as_mut() {
            if let Some(values) = eigenvals.as_ref() {
                store.values().append_slice(values);
                store.append(true);
            } else {
                store.append(false);
            }
        }
        if let Some(store) = self.stds.as_mut() {
            if let Some(values) = stds.as_ref() {
                store.values().append_slice(values);
                store.append(true);
            } else {
                store.append(false);
            }
        }

        self.num_eigenvalues.append_value(num_eigenvalues);
    }

    fn finalize(self) -> Option<StructArray> {
        let Self {
            eigenvals,
            stds,
            mut num_eigenvalues,
        } = self;

        let num_eigenvalues = ArrayBuilder::finish(&mut num_eigenvalues);

        let mut fields = vec![Field::new(
            "mass_matrix_num_eigenvalues",
            arrow::datatypes::DataType::UInt64,
            false,
        )];
        let mut arrays = vec![num_eigenvalues];

        if let Some(mut eigenvals) = eigenvals {
            let eigenvals = ArrayBuilder::finish(&mut eigenvals);
            fields.push(Field::new(
                "mass_matrix_eigenvals",
                eigenvals.data_type().clone(),
                true,
            ));

            arrays.push(eigenvals);
        }

        if let Some(mut stds) = stds {
            let stds = ArrayBuilder::finish(&mut stds);
            fields.push(Field::new(
                "mass_matrix_stds",
                stds.data_type().clone(),
                true,
            ));

            arrays.push(stds);
        }

        Some(StructArray::new(fields.into(), arrays, None))
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self {
            ref eigenvals,
            ref stds,
            ref num_eigenvalues,
        } = self;

        let num_eigenvalues = ArrayBuilder::finish_cloned(num_eigenvalues);

        let mut fields = vec![Field::new(
            "mass_matrix_num_eigenvalues",
            arrow::datatypes::DataType::UInt64,
            false,
        )];
        let mut arrays = vec![num_eigenvalues];

        if let Some(eigenvals) = &eigenvals {
            let eigenvals = ArrayBuilder::finish_cloned(eigenvals);
            fields.push(Field::new(
                "mass_matrix_eigenvals",
                eigenvals.data_type().clone(),
                true,
            ));

            arrays.push(eigenvals);
        }

        if let Some(stds) = &stds {
            let stds = ArrayBuilder::finish_cloned(stds);
            fields.push(Field::new(
                "mass_matrix_stds",
                stds.data_type().clone(),
                true,
            ));

            arrays.push(stds);
        }
        Some(StructArray::new(fields.into(), arrays, None))
    }
}

impl<M: Math> SamplerStats<M> for LowRankMassMatrix<M> {
    type Stats = MatrixStats;
    type Builder = MatrixBuilder;

    fn new_builder(&self, _settings: &impl crate::Settings, dim: usize) -> Self::Builder {
        let num_eigenvalues = PrimitiveBuilder::new();
        if self.settings.store_mass_matrix {
            let items = PrimitiveBuilder::new();
            let eigenvals = Some(ListBuilder::new(items));

            let items = PrimitiveBuilder::new();
            let stds = Some(FixedSizeListBuilder::new(items, dim as _));

            MatrixBuilder {
                eigenvals,
                stds,
                num_eigenvalues,
            }
        } else {
            MatrixBuilder {
                eigenvals: None,
                stds: None,
                num_eigenvalues,
            }
        }
    }

    fn current_stats(&self, math: &mut M) -> Self::Stats {
        let num_eigenvalues = self
            .inner
            .as_ref()
            .map(|inner| inner.num_eigenvalues)
            .unwrap_or(0);

        if self.settings.store_mass_matrix {
            let mut stds = vec![0f64; math.dim()].into_boxed_slice();
            math.write_to_slice(&self.stds, &mut stds);

            let vals = self
                .inner
                .as_ref()
                .map(|inner| math.eigs_as_array(&inner.vals));

            MatrixStats {
                stds: Some(stds),
                eigenvals: vals,
                num_eigenvalues,
            }
        } else {
            MatrixStats {
                stds: None,
                eigenvals: None,
                num_eigenvalues,
            }
        }
    }
}

impl<M: Math> MassMatrix<M> for LowRankMassMatrix<M> {
    fn update_velocity(&self, math: &mut M, state: &mut EuclideanPoint<M>) {
        let Some(inner) = self.inner.as_ref() else {
            math.array_mult(&self.variance, &state.momentum, &mut state.velocity);
            return;
        };

        math.array_mult_eigs(
            &self.stds,
            &state.momentum,
            &mut state.velocity,
            &inner.vecs,
            &inner.vals,
        );
    }

    fn update_kinetic_energy(&self, math: &mut M, state: &mut EuclideanPoint<M>) {
        state.kinetic_energy = 0.5 * math.array_vector_dot(&state.momentum, &state.velocity);
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut EuclideanPoint<M>,
        rng: &mut R,
    ) {
        let Some(inner) = self.inner.as_ref() else {
            math.array_gaussian(rng, &mut state.momentum, &self.inv_stds);
            return;
        };

        math.array_gaussian_eigs(
            rng,
            &mut state.momentum,
            &self.inv_stds,
            &inner.vals_sqrt_inv,
            &inner.vecs,
        );
    }
}

#[derive(Debug, Clone)]
pub struct Stats {
    //foreground_length: u64,
    //background_length: u64,
    //is_update: bool,
    //diag: Box<[f64]>,
    //eigvalues: Box<[f64]>,
    //eigvectors: Box<[f64]>,
}

#[derive(Debug)]
pub struct Builder {}

impl StatTraceBuilder<Stats> for Builder {
    fn append_value(&mut self, value: Stats) {
        let Stats {} = value;
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
    settings: LowRankSettings,
}

impl LowRankMassMatrixStrategy {
    pub fn new(ndim: usize, settings: LowRankSettings) -> Self {
        let draws = VecDeque::with_capacity(100);
        let grads = VecDeque::with_capacity(100);

        Self {
            draws,
            grads,
            ndim,
            background_split: 0,
            settings,
        }
    }

    pub fn add_draw<M: Math>(&mut self, math: &mut M, state: &State<M, EuclideanPoint<M>>) {
        assert!(math.dim() == self.ndim);
        let mut draw = vec![0f64; self.ndim];
        state.write_position(math, &mut draw);
        let mut grad = vec![0f64; self.ndim];
        state.write_gradient(math, &mut grad);

        self.draws.push_back(draw);
        self.grads.push_back(grad);
    }

    pub fn clear(&mut self) {
        self.draws.clear();
        self.grads.clear();
    }

    pub fn update<M: Math>(&self, math: &mut M, matrix: &mut LowRankMassMatrix<M>) {
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

        let stds = rescale_points(&mut draws, &mut grads);

        let svd_draws = draws.thin_svd();
        let svd_grads = grads.thin_svd();

        let subspace = faer::concat![[svd_draws.u(), svd_grads.u()]];

        let subspace_qr = subspace.col_piv_qr();

        let subspace_basis = subspace_qr.compute_thin_q();

        let draws_proj = subspace_basis.transpose() * (&draws);
        let grads_proj = subspace_basis.transpose() * (&grads);

        let (vals, vecs) = estimate_mass_matrix(draws_proj, grads_proj, self.settings.gamma);

        let filtered = vals
            .iter()
            .zip(vecs.col_iter())
            .filter(|(&val, _)| {
                (val > self.settings.eigval_cutoff) | (val < self.settings.eigval_cutoff.recip())
            })
            .collect_vec();

        let vals = filtered.iter().map(|x| *x.0).collect_vec();
        let vals = faer::col::from_slice(&vals).to_owned();

        let vecs_vec = filtered.into_iter().map(|x| x.1).collect_vec();
        let mut vecs = Mat::zeros(subspace_basis.ncols(), vals.nrows());
        vecs.col_iter_mut()
            .zip(vecs_vec.iter())
            .for_each(|(mut col, vals)| col.copy_from(vals));

        let vecs = subspace_basis * vecs;

        matrix.update(math, stds, vals, vecs);
    }
}

fn rescale_points(draws: &mut Mat<f64>, grads: &mut Mat<f64>) -> Col<f64> {
    let (ndim, ndraws) = draws.shape();
    let stds = Col::from_fn(ndim, |col| {
        let draw_mean = draws.row(col).sum() / (ndraws as f64);
        let grad_mean = grads.row(col).sum() / (ndraws as f64);
        let draw_std: f64 = draws
            .row(col)
            .iter()
            .map(|&val| (val - draw_mean) * (val - draw_mean))
            .sum::<f64>()
            .sqrt();
        let grad_std: f64 = grads
            .row(col)
            .iter()
            .map(|&val| (val - grad_mean) * (val - grad_mean))
            .sum::<f64>()
            .sqrt();

        let std = (draw_std / grad_std).sqrt();

        let draw_scale = (std * (ndraws as f64)).recip();
        draws
            .row_mut(col)
            .iter_mut()
            .for_each(|val| *val = (*val - draw_mean) * draw_scale);

        let grad_scale = std * (ndraws as f64).recip();
        grads
            .row_mut(col)
            .iter_mut()
            .for_each(|val| *val = (*val - grad_mean) * grad_scale);

        std
    });
    stds
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

    let mean = spd_mean(cov_draws, cov_grads);

    let mean_eig = mean.selfadjoint_eigendecomposition(faer::Side::Lower);

    (
        mean_eig.s().column_vector().to_owned(),
        mean_eig.u().to_owned(),
    )
}

fn spd_mean(cov_draws: Mat<f64>, cov_grads: Mat<f64>) -> Mat<f64> {
    let eigs_grads = cov_grads.selfadjoint_eigendecomposition(faer::Side::Lower);

    let u = eigs_grads.u();
    let eigs = eigs_grads.s().column_vector().to_owned();

    let mut eigs_sqrt = eigs.clone();
    eigs_sqrt.iter_mut().for_each(|val| *val = val.sqrt());
    let cov_grads_sqrt = u * eigs_sqrt.column_vector_into_diagonal() * u.transpose();
    let m = (&cov_grads_sqrt) * cov_draws * cov_grads_sqrt;

    let m_eig = m.selfadjoint_eigendecomposition(faer::Side::Lower);

    let m_u = m_eig.u();
    let mut m_s = m_eig.s().column_vector().to_owned();
    m_s.iter_mut().for_each(|val| *val = val.sqrt());

    let m_sqrt = m_u * m_s.column_vector_into_diagonal() * m_u.transpose();

    let mut eigs_grads_inv = eigs;
    eigs_grads_inv
        .iter_mut()
        .for_each(|val| *val = val.sqrt().recip());
    let grads_inv_sqrt = u * eigs_grads_inv.column_vector_into_diagonal() * u.transpose();

    (&grads_inv_sqrt) * m_sqrt * grads_inv_sqrt
}

impl<M: Math> SamplerStats<M> for LowRankMassMatrixStrategy {
    type Stats = Stats;
    type Builder = Builder;

    fn new_builder(&self, _settings: &impl crate::Settings, _dim: usize) -> Self::Builder {
        Builder {}
    }

    fn current_stats(&self, _math: &mut M) -> Self::Stats {
        Stats {}
    }
}

impl<M: Math> AdaptStrategy<M> for LowRankMassMatrixStrategy {
    type Hamiltonian = EuclideanHamiltonian<M, LowRankMassMatrix<M>>;
    type Collector = DrawGradCollector<M>;
    type Options = LowRankSettings;

    fn new(math: &mut M, options: Self::Options, _num_tune: u64, _chain: u64) -> Self {
        Self::new(math.dim(), options)
    }

    fn init<R: rand::Rng + ?Sized>(
        &mut self,
        math: &mut M,
        _options: &mut crate::nuts::NutsOptions,
        hamiltonian: &mut Self::Hamiltonian,
        position: &[f64],
        _rng: &mut R,
    ) -> Result<(), NutsError> {
        let state = hamiltonian.init_state(math, position)?;
        self.add_draw(math, &state);
        hamiltonian.mass_matrix.update_from_grad(
            math,
            state.point().gradient(),
            1f64,
            (1e-20, 1e20),
        );
        Ok(())
    }

    fn adapt<R: rand::Rng + ?Sized>(
        &mut self,
        _math: &mut M,
        _options: &mut crate::nuts::NutsOptions,
        _potential: &mut Self::Hamiltonian,
        _draw: u64,
        _collector: &Self::Collector,
        _state: &State<M, EuclideanPoint<M>>,
        _rng: &mut R,
    ) -> Result<(), NutsError> {
        Ok(())
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

    fn update_potential(&self, math: &mut M, potential: &mut Self::Hamiltonian) -> bool {
        if <LowRankMassMatrixStrategy as MassMatrixAdaptStrategy<M>>::current_count(self) < 3 {
            return false;
        }
        self.update(math, &mut potential.mass_matrix);

        true
    }
}

#[cfg(test)]
mod test {
    use std::ops::AddAssign;

    use faer::{assert_matrix_eq, Col, Mat};
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use rand_distr::StandardNormal;

    use super::{estimate_mass_matrix, spd_mean};

    #[test]
    fn test_spd_mean() {
        let x_diag = faer::col![1., 4., 8.];
        let y_diag = faer::col![1., 1., 0.5];

        let mut x = faer::Mat::zeros(3, 3);
        let mut y = faer::Mat::zeros(3, 3);

        x.diagonal_mut().column_vector_mut().add_assign(x_diag);
        y.diagonal_mut().column_vector_mut().add_assign(y_diag);

        let out = spd_mean(x, y);
        let expected_diag = faer::col![1., 2., 4.];
        let mut expected = faer::Mat::zeros(3, 3);
        expected
            .diagonal_mut()
            .column_vector_mut()
            .add_assign(expected_diag);

        faer::assert_matrix_eq!(out, expected, comp = ulp, tol = 8);
    }

    #[test]
    fn test_estimate_mass_matrix() {
        let distr = StandardNormal;

        let mut rng = SmallRng::seed_from_u64(1);

        let draws: Mat<f64> = Mat::from_fn(20, 3, |_, _| rng.sample(distr));
        //let grads: Mat<f64> = Mat::from_fn(20, 3, |_, _| rng.sample(distr));
        let grads = -(&draws);

        let (vals, vecs) = estimate_mass_matrix(draws, grads, 0.0001);
        assert!(vals.iter().cloned().all(|x| x > 0.));
        assert!(vecs.is_all_finite());
        assert_matrix_eq!(
            vals.as_2d(),
            Col::full(20, 1.).as_2d(),
            comp = abs,
            tol = 1e-4
        );
    }
}
