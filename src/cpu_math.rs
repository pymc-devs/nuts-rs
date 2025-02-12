use std::{error::Error, fmt::Debug, mem::replace};

use faer::{Col, Mat};
use itertools::{izip, Itertools};
use thiserror::Error;

use crate::{
    math::{axpy, axpy_out, multiply, scalar_prods2, scalar_prods3, vector_dot},
    math_base::{LogpError, Math},
};

#[derive(Debug)]
pub struct CpuMath<F: CpuLogpFunc> {
    logp_func: F,
    arch: pulp::Arch,
}

impl<F: CpuLogpFunc> CpuMath<F> {
    pub fn new(logp_func: F) -> Self {
        let arch = pulp::Arch::new();
        Self { logp_func, arch }
    }
}

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum CpuMathError {
    #[error("Error during array operation")]
    ArrayError(),
}

impl<F: CpuLogpFunc> Math for CpuMath<F> {
    type Vector = Col<f64>;
    type EigVectors = Mat<f64>;
    type EigValues = Col<f64>;
    type LogpErr = F::LogpError;
    type Err = CpuMathError;
    type TransformParams = F::TransformParams;

    fn new_array(&mut self) -> Self::Vector {
        Col::zeros(self.dim())
    }

    fn new_eig_vectors<'a>(
        &'a mut self,
        vals: impl ExactSizeIterator<Item = &'a [f64]>,
    ) -> Self::EigVectors {
        let ndim = self.dim();
        let nvecs = vals.len();

        let mut vectors: Mat<f64> = Mat::zeros(ndim, nvecs);
        vectors.col_iter_mut().zip_eq(vals).for_each(|(col, vals)| {
            col.try_as_col_major_mut()
                .expect("Array is not contiguous")
                .as_slice_mut()
                .copy_from_slice(vals)
        });

        vectors
    }

    fn new_eig_values(&mut self, vals: &[f64]) -> Self::EigValues {
        let mut values: Col<f64> = Col::zeros(vals.len());
        values
            .try_as_col_major_mut()
            .expect("Array is not contiguous")
            .as_slice_mut()
            .copy_from_slice(vals);
        values
    }

    fn logp_array(
        &mut self,
        position: &Self::Vector,
        gradient: &mut Self::Vector,
    ) -> Result<f64, Self::LogpErr> {
        self.logp_func.logp(
            position
                .try_as_col_major()
                .expect("Array is not contiguous")
                .as_slice(),
            gradient
                .try_as_col_major_mut()
                .expect("Array is not contiguous")
                .as_slice_mut(),
        )
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpErr> {
        self.logp_func.logp(position, gradient)
    }

    fn dim(&self) -> usize {
        self.logp_func.dim()
    }

    fn scalar_prods3(
        &mut self,
        positive1: &Self::Vector,
        negative1: &Self::Vector,
        positive2: &Self::Vector,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> (f64, f64) {
        scalar_prods3(
            positive1.try_as_col_major().unwrap().as_slice(),
            negative1.try_as_col_major().unwrap().as_slice(),
            positive2.try_as_col_major().unwrap().as_slice(),
            x.try_as_col_major().unwrap().as_slice(),
            y.try_as_col_major().unwrap().as_slice(),
        )
    }

    fn scalar_prods2(
        &mut self,
        positive1: &Self::Vector,
        positive2: &Self::Vector,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> (f64, f64) {
        scalar_prods2(
            positive1.try_as_col_major().unwrap().as_slice(),
            positive2.try_as_col_major().unwrap().as_slice(),
            x.try_as_col_major().unwrap().as_slice(),
            y.try_as_col_major().unwrap().as_slice(),
        )
    }

    fn sq_norm_sum(&mut self, x: &Self::Vector, y: &Self::Vector) -> f64 {
        x.try_as_col_major()
            .unwrap()
            .as_slice()
            .iter()
            .zip(y.try_as_col_major().unwrap().as_slice())
            .map(|(&x, &y)| (x + y) * (x + y))
            .sum()
    }

    fn read_from_slice(&mut self, dest: &mut Self::Vector, source: &[f64]) {
        dest.try_as_col_major_mut()
            .unwrap()
            .as_slice_mut()
            .copy_from_slice(source);
    }

    fn write_to_slice(&mut self, source: &Self::Vector, dest: &mut [f64]) {
        dest.copy_from_slice(source.try_as_col_major().unwrap().as_slice())
    }

    fn copy_into(&mut self, array: &Self::Vector, dest: &mut Self::Vector) {
        dest.clone_from(array)
    }

    fn axpy_out(&mut self, x: &Self::Vector, y: &Self::Vector, a: f64, out: &mut Self::Vector) {
        axpy_out(
            x.try_as_col_major().unwrap().as_slice(),
            y.try_as_col_major().unwrap().as_slice(),
            a,
            out.try_as_col_major_mut().unwrap().as_slice_mut(),
        );
    }

    fn axpy(&mut self, x: &Self::Vector, y: &mut Self::Vector, a: f64) {
        axpy(
            x.try_as_col_major().unwrap().as_slice(),
            y.try_as_col_major_mut().unwrap().as_slice_mut(),
            a,
        );
    }

    fn fill_array(&mut self, array: &mut Self::Vector, val: f64) {
        faer::zip!(array).for_each(|faer::unzip!(pos)| *pos = val);
    }

    fn array_all_finite(&mut self, array: &Self::Vector) -> bool {
        let mut ok = true;
        faer::zip!(array).for_each(|faer::unzip!(val)| ok = ok & val.is_finite());
        ok
    }

    fn array_all_finite_and_nonzero(&mut self, array: &Self::Vector) -> bool {
        self.arch.dispatch(|| {
            array
                .try_as_col_major()
                .unwrap()
                .as_slice()
                .iter()
                .all(|&x| x.is_finite() & (x != 0f64))
        })
    }

    fn array_mult(
        &mut self,
        array1: &Self::Vector,
        array2: &Self::Vector,
        dest: &mut Self::Vector,
    ) {
        multiply(
            array1.try_as_col_major().unwrap().as_slice(),
            array2.try_as_col_major().unwrap().as_slice(),
            dest.try_as_col_major_mut().unwrap().as_slice_mut(),
        )
    }

    fn array_mult_eigs(
        &mut self,
        stds: &Self::Vector,
        rhs: &Self::Vector,
        dest: &mut Self::Vector,
        vecs: &Self::EigVectors,
        vals: &Self::EigValues,
    ) {
        let rhs = stds.as_diagonal() * rhs;
        let trafo = vecs.transpose() * (&rhs);
        let inner_prod = vecs * (vals.as_diagonal() * (&trafo) - (&trafo)) + rhs;
        let scaled = stds.as_diagonal() * inner_prod;

        let _ = replace(dest, scaled);
    }

    fn array_vector_dot(&mut self, array1: &Self::Vector, array2: &Self::Vector) -> f64 {
        vector_dot(
            array1.try_as_col_major().unwrap().as_slice(),
            array2.try_as_col_major().unwrap().as_slice(),
        )
    }

    fn array_gaussian<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Vector,
        stds: &Self::Vector,
    ) {
        let dist = rand_distr::StandardNormal;
        dest.try_as_col_major_mut()
            .unwrap()
            .as_slice_mut()
            .iter_mut()
            .zip(stds.try_as_col_major().unwrap().as_slice().iter())
            .for_each(|(p, &s)| {
                let norm: f64 = rng.sample(dist);
                *p = s * norm;
            });
    }

    fn array_gaussian_eigs<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Vector,
        scale: &Self::Vector,
        vals: &Self::EigValues,
        vecs: &Self::EigVectors,
    ) {
        let mut draw: Col<f64> = Col::zeros(self.dim());
        let dist = rand_distr::StandardNormal;
        draw.try_as_col_major_mut()
            .unwrap()
            .as_slice_mut()
            .iter_mut()
            .for_each(|p| {
                *p = rng.sample(dist);
            });

        let trafo = vecs.transpose() * (&draw);
        let inner_prod = vecs * (vals.as_diagonal() * (&trafo) - (&trafo)) + draw;

        let scaled = scale.as_diagonal() * inner_prod;

        let _ = replace(dest, scaled);
    }

    fn array_update_variance(
        &mut self,
        mean: &mut Self::Vector,
        variance: &mut Self::Vector,
        value: &Self::Vector,
        diff_scale: f64, // 1 / self.count
    ) {
        self.arch.dispatch(|| {
            izip!(
                mean.try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut()
                    .iter_mut(),
                variance
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut()
                    .iter_mut(),
                value.try_as_col_major().unwrap().as_slice()
            )
            .for_each(|(mean, var, x)| {
                let diff = x - *mean;
                *mean += diff * diff_scale;
                *var += diff * diff;
            });
        })
    }

    fn array_update_var_inv_std_draw(
        &mut self,
        variance_out: &mut Self::Vector,
        inv_std: &mut Self::Vector,
        draw_var: &Self::Vector,
        scale: f64,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        self.arch.dispatch(|| {
            izip!(
                variance_out
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut()
                    .iter_mut(),
                inv_std
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut()
                    .iter_mut(),
                draw_var.try_as_col_major().unwrap().as_slice().iter(),
            )
            .for_each(|(var_out, inv_std_out, &draw_var)| {
                let draw_var = draw_var * scale;
                if (!draw_var.is_finite()) | (draw_var == 0f64) {
                    if let Some(fill_val) = fill_invalid {
                        *var_out = fill_val;
                        *inv_std_out = fill_val.recip().sqrt();
                    }
                } else {
                    let val = draw_var.clamp(clamp.0, clamp.1);
                    *var_out = val;
                    *inv_std_out = val.recip().sqrt();
                }
            });
        });
    }

    fn array_update_var_inv_std_draw_grad(
        &mut self,
        variance_out: &mut Self::Vector,
        inv_std: &mut Self::Vector,
        draw_var: &Self::Vector,
        grad_var: &Self::Vector,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        self.arch.dispatch(|| {
            izip!(
                variance_out
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut()
                    .iter_mut(),
                inv_std
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut()
                    .iter_mut(),
                draw_var.try_as_col_major().unwrap().as_slice().iter(),
                grad_var.try_as_col_major().unwrap().as_slice().iter(),
            )
            .for_each(|(var_out, inv_std_out, &draw_var, &grad_var)| {
                let val = (draw_var / grad_var).sqrt();
                if (!val.is_finite()) | (val == 0f64) {
                    if let Some(fill_val) = fill_invalid {
                        *var_out = fill_val;
                        *inv_std_out = fill_val.recip().sqrt();
                    }
                } else {
                    let val = val.clamp(clamp.0, clamp.1);
                    *var_out = val;
                    *inv_std_out = val.recip().sqrt();
                }
            });
        });
    }

    fn array_update_var_inv_std_grad(
        &mut self,
        variance_out: &mut Self::Vector,
        inv_std: &mut Self::Vector,
        gradient: &Self::Vector,
        fill_invalid: f64,
        clamp: (f64, f64),
    ) {
        self.arch.dispatch(|| {
            izip!(
                variance_out
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut()
                    .iter_mut(),
                inv_std
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut()
                    .iter_mut(),
                gradient.try_as_col_major().unwrap().as_slice().iter(),
            )
            .for_each(|(var_out, inv_std_out, &grad_var)| {
                let val = grad_var.abs().clamp(clamp.0, clamp.1).recip();
                let val = if val.is_finite() { val } else { fill_invalid };
                *var_out = val;
                *inv_std_out = val.recip().sqrt();
            });
        });
    }

    fn eigs_as_array(&mut self, source: &Self::EigValues) -> Box<[f64]> {
        source
            .try_as_col_major()
            .unwrap()
            .as_slice()
            .to_vec()
            .into()
    }

    fn inv_transform_normalize(
        &mut self,
        params: &Self::TransformParams,
        untransformed_position: &Self::Vector,
        untransofrmed_gradient: &Self::Vector,
        transformed_position: &mut Self::Vector,
        transformed_gradient: &mut Self::Vector,
    ) -> Result<f64, Self::LogpErr> {
        self.logp_func.inv_transform_normalize(
            params,
            untransformed_position
                .try_as_col_major()
                .unwrap()
                .as_slice(),
            untransofrmed_gradient
                .try_as_col_major()
                .unwrap()
                .as_slice(),
            transformed_position
                .try_as_col_major_mut()
                .unwrap()
                .as_slice_mut(),
            transformed_gradient
                .try_as_col_major_mut()
                .unwrap()
                .as_slice_mut(),
        )
    }

    fn init_from_untransformed_position(
        &mut self,
        params: &Self::TransformParams,
        untransformed_position: &Self::Vector,
        untransformed_gradient: &mut Self::Vector,
        transformed_position: &mut Self::Vector,
        transformed_gradient: &mut Self::Vector,
    ) -> Result<(f64, f64), Self::LogpErr> {
        self.logp_func.init_from_untransformed_position(
            params,
            untransformed_position
                .try_as_col_major()
                .unwrap()
                .as_slice(),
            untransformed_gradient
                .try_as_col_major_mut()
                .unwrap()
                .as_slice_mut(),
            transformed_position
                .try_as_col_major_mut()
                .unwrap()
                .as_slice_mut(),
            transformed_gradient
                .try_as_col_major_mut()
                .unwrap()
                .as_slice_mut(),
        )
    }

    fn init_from_transformed_position(
        &mut self,
        params: &Self::TransformParams,
        untransformed_position: &mut Self::Vector,
        untransformed_gradient: &mut Self::Vector,
        transformed_position: &Self::Vector,
        transformed_gradient: &mut Self::Vector,
    ) -> Result<(f64, f64), Self::LogpErr> {
        self.logp_func.init_from_transformed_position(
            params,
            untransformed_position
                .try_as_col_major_mut()
                .unwrap()
                .as_slice_mut(),
            untransformed_gradient
                .try_as_col_major_mut()
                .unwrap()
                .as_slice_mut(),
            transformed_position.try_as_col_major().unwrap().as_slice(),
            transformed_gradient
                .try_as_col_major_mut()
                .unwrap()
                .as_slice_mut(),
        )
    }

    fn update_transformation<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        rng: &mut R,
        untransformed_positions: impl ExactSizeIterator<Item = &'a Self::Vector>,
        untransformed_gradients: impl ExactSizeIterator<Item = &'a Self::Vector>,
        untransformed_logp: impl ExactSizeIterator<Item = &'a f64>,
        params: &'a mut Self::TransformParams,
    ) -> Result<(), Self::LogpErr> {
        self.logp_func.update_transformation(
            rng,
            untransformed_positions.map(|x| x.try_as_col_major().unwrap().as_slice()),
            untransformed_gradients.map(|x| x.try_as_col_major().unwrap().as_slice()),
            untransformed_logp,
            params,
        )
    }

    fn new_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        untransformed_position: &Self::Vector,
        untransfogmed_gradient: &Self::Vector,
        chain: u64,
    ) -> Result<Self::TransformParams, Self::LogpErr> {
        self.logp_func.new_transformation(
            rng,
            untransformed_position
                .try_as_col_major()
                .unwrap()
                .as_slice(),
            untransfogmed_gradient
                .try_as_col_major()
                .unwrap()
                .as_slice(),
            chain,
        )
    }

    fn transformation_id(&self, params: &Self::TransformParams) -> Result<i64, Self::LogpErr> {
        self.logp_func.transformation_id(params)
    }
}

pub trait CpuLogpFunc {
    type LogpError: Debug + Send + Sync + Error + LogpError + 'static;
    type TransformParams;

    fn dim(&self) -> usize;
    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpError>;

    fn inv_transform_normalize(
        &mut self,
        _params: &Self::TransformParams,
        _untransformed_position: &[f64],
        _untransformed_gradient: &[f64],
        _transformed_position: &mut [f64],
        _transformed_gradient: &mut [f64],
    ) -> Result<f64, Self::LogpError> {
        unimplemented!()
    }

    fn init_from_untransformed_position(
        &mut self,
        _params: &Self::TransformParams,
        _untransformed_position: &[f64],
        _untransformed_gradient: &mut [f64],
        _transformed_position: &mut [f64],
        _transformed_gradient: &mut [f64],
    ) -> Result<(f64, f64), Self::LogpError> {
        unimplemented!()
    }

    fn init_from_transformed_position(
        &mut self,
        _params: &Self::TransformParams,
        _untransformed_position: &mut [f64],
        _untransformed_gradient: &mut [f64],
        _transformed_position: &[f64],
        _transformed_gradient: &mut [f64],
    ) -> Result<(f64, f64), Self::LogpError> {
        unimplemented!()
    }

    fn update_transformation<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        _rng: &mut R,
        _untransformed_positions: impl ExactSizeIterator<Item = &'a [f64]>,
        _untransformed_gradients: impl ExactSizeIterator<Item = &'a [f64]>,
        _untransformed_logp: impl ExactSizeIterator<Item = &'a f64>,
        _params: &'a mut Self::TransformParams,
    ) -> Result<(), Self::LogpError> {
        unimplemented!()
    }

    fn new_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        _untransformed_position: &[f64],
        _untransformed_gradient: &[f64],
        _chain: u64,
    ) -> Result<Self::TransformParams, Self::LogpError> {
        unimplemented!()
    }

    fn transformation_id(&self, _params: &Self::TransformParams) -> Result<i64, Self::LogpError> {
        unimplemented!()
    }
}
