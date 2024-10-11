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
    parallel: faer::Parallelism<'static>,
}

impl<F: CpuLogpFunc> CpuMath<F> {
    pub fn new(logp_func: F) -> Self {
        let arch = pulp::Arch::new();
        let parallel = faer::Parallelism::None;
        Self {
            logp_func,
            arch,
            parallel,
        }
    }

    pub fn with_parallel(logp_func: F, parallel: faer::Parallelism<'static>) -> Self {
        let arch = pulp::Arch::new();
        Self {
            logp_func,
            arch,
            parallel,
        }
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
            col.try_as_slice_mut()
                .expect("Array is not contiguous")
                .copy_from_slice(vals)
        });

        vectors
    }

    fn new_eig_values(&mut self, vals: &[f64]) -> Self::EigValues {
        let mut values: Col<f64> = Col::zeros(vals.len());
        values.as_slice_mut().copy_from_slice(vals);
        values
    }

    fn logp_array(
        &mut self,
        position: &Self::Vector,
        gradient: &mut Self::Vector,
    ) -> Result<f64, Self::LogpErr> {
        self.logp_func
            .logp(position.as_slice(), gradient.as_slice_mut())
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
            positive1.as_slice(),
            negative1.as_slice(),
            positive2.as_slice(),
            x.as_slice(),
            y.as_slice(),
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
            positive1.as_slice(),
            positive2.as_slice(),
            x.as_slice(),
            y.as_slice(),
        )
    }

    fn read_from_slice(&mut self, dest: &mut Self::Vector, source: &[f64]) {
        dest.as_slice_mut().copy_from_slice(source);
    }

    fn write_to_slice(&mut self, source: &Self::Vector, dest: &mut [f64]) {
        dest.copy_from_slice(source.as_slice())
    }

    fn copy_into(&mut self, array: &Self::Vector, dest: &mut Self::Vector) {
        dest.clone_from(array)
    }

    fn axpy_out(&mut self, x: &Self::Vector, y: &Self::Vector, a: f64, out: &mut Self::Vector) {
        axpy_out(x.as_slice(), y.as_slice(), a, out.as_slice_mut());
    }

    fn axpy(&mut self, x: &Self::Vector, y: &mut Self::Vector, a: f64) {
        axpy(x.as_slice(), y.as_slice_mut(), a);
    }

    fn fill_array(&mut self, array: &mut Self::Vector, val: f64) {
        array.fill(val);
    }

    fn array_all_finite(&mut self, array: &Self::Vector) -> bool {
        array.is_all_finite()
    }

    fn array_all_finite_and_nonzero(&mut self, array: &Self::Vector) -> bool {
        self.arch.dispatch(|| {
            array
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
        multiply(array1.as_slice(), array2.as_slice(), dest.as_slice_mut())
    }

    fn array_mult_eigs(
        &mut self,
        stds: &Self::Vector,
        rhs: &Self::Vector,
        dest: &mut Self::Vector,
        vecs: &Self::EigVectors,
        vals: &Self::EigValues,
    ) {
        let rhs = stds.column_vector_as_diagonal() * rhs;
        let trafo = vecs.transpose() * (&rhs);
        let inner_prod = vecs * (vals.column_vector_as_diagonal() * (&trafo) - (&trafo)) + rhs;
        let scaled = stds.column_vector_as_diagonal() * inner_prod;

        let _ = replace(dest, scaled);
    }

    fn array_vector_dot(&mut self, array1: &Self::Vector, array2: &Self::Vector) -> f64 {
        vector_dot(array1.as_slice(), array2.as_slice())
    }

    fn array_gaussian<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Vector,
        stds: &Self::Vector,
    ) {
        let dist = rand_distr::StandardNormal;
        dest.as_slice_mut()
            .iter_mut()
            .zip(stds.as_slice().iter())
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
        draw.as_slice_mut().iter_mut().for_each(|p| {
            *p = rng.sample(dist);
        });

        let trafo = vecs.transpose() * (&draw);
        let inner_prod = vecs * (vals.column_vector_as_diagonal() * (&trafo) - (&trafo)) + draw;

        let scaled = scale.column_vector_as_diagonal() * inner_prod;

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
                mean.as_slice_mut().iter_mut(),
                variance.as_slice_mut().iter_mut(),
                value.as_slice()
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
                variance_out.as_slice_mut().iter_mut(),
                inv_std.as_slice_mut().iter_mut(),
                draw_var.as_slice().iter(),
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
                variance_out.as_slice_mut().iter_mut(),
                inv_std.as_slice_mut().iter_mut(),
                draw_var.as_slice().iter(),
                grad_var.as_slice().iter(),
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
                variance_out.as_slice_mut().iter_mut(),
                inv_std.as_slice_mut().iter_mut(),
                gradient.as_slice().iter(),
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
        source.as_slice().to_vec().into()
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
            untransformed_position.as_slice(),
            untransofrmed_gradient.as_slice(),
            transformed_position.as_slice_mut(),
            transformed_gradient.as_slice_mut(),
        )
    }

    fn transformed_logp(
        &mut self,
        params: &Self::TransformParams,
        untransformed_position: &Self::Vector,
        untransformed_gradient: &mut Self::Vector,
        transformed_position: &mut Self::Vector,
        transformed_gradient: &mut Self::Vector,
    ) -> Result<(f64, f64), Self::LogpErr> {
        self.logp_func.transformed_logp(
            params,
            untransformed_position.as_slice(),
            untransformed_gradient.as_slice_mut(),
            transformed_position.as_slice_mut(),
            transformed_gradient.as_slice_mut(),
        )
    }

    fn update_transformation<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        rng: &mut R,
        untransformed_positions: impl Iterator<Item = &'a Self::Vector>,
        untransformed_gradients: impl Iterator<Item = &'a Self::Vector>,
        params: &'a mut Self::TransformParams,
    ) -> Result<(), Self::LogpErr> {
        self.logp_func.update_transformation(
            rng,
            untransformed_positions.map(|x| x.as_slice()),
            untransformed_gradients.map(|x| x.as_slice()),
            params,
        )
    }

    fn new_transformation(
        &mut self,
        untransformed_position: &Self::Vector,
        untransfogmed_gradient: &Self::Vector,
    ) -> Result<Self::TransformParams, Self::LogpErr> {
        self.logp_func.new_transformation(
            untransformed_position.as_slice(),
            untransfogmed_gradient.as_slice(),
        )
    }

    fn transformation_id(&self, params: &Self::TransformParams) -> i64 {
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
        params: &Self::TransformParams,
        untransformed_position: &[f64],
        untransofrmed_gradient: &[f64],
        transformed_position: &mut [f64],
        transformed_gradient: &mut [f64],
    ) -> Result<f64, Self::LogpError>;

    fn transformed_logp(
        &mut self,
        params: &Self::TransformParams,
        untransformed_position: &[f64],
        untransformed_gradient: &mut [f64],
        transformed_position: &mut [f64],
        transformed_gradient: &mut [f64],
    ) -> Result<(f64, f64), Self::LogpError>;

    fn update_transformation<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        rng: &mut R,
        untransformed_positions: impl Iterator<Item = &'a [f64]>,
        untransformed_gradients: impl Iterator<Item = &'a [f64]>,
        params: &'a mut Self::TransformParams,
    ) -> Result<(), Self::LogpError>;

    fn new_transformation(
        &mut self,
        untransformed_position: &[f64],
        untransfogmed_gradient: &[f64],
    ) -> Result<Self::TransformParams, Self::LogpError>;

    fn transformation_id(&self, params: &Self::TransformParams) -> i64;
}
