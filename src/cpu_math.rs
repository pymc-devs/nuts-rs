use std::{error::Error, fmt::Debug};

use itertools::izip;
use thiserror::Error;

use crate::{
    math::{axpy, axpy_out, multiply, scalar_prods2, scalar_prods3, vector_dot},
    math_base::Math,
    LogpError,
};

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
    type Array = faer_core::Mat<f64>;
    type LogpErr = F::LogpError;
    type Err = CpuMathError;

    fn new_array(&self) -> Self::Array {
        faer_core::Mat::zeros(self.dim(), 1)
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpErr> {
        self.logp_func.logp(position, gradient)
    }

    fn logp_array(
        &mut self,
        position: &Self::Array,
        gradient: &mut Self::Array,
    ) -> Result<f64, Self::LogpErr> {
        self.logp_func
            .logp(position.col_ref(0), gradient.col_mut(0))
    }

    fn dim(&self) -> usize {
        self.logp_func.dim()
    }

    fn scalar_prods3(
        &mut self,
        positive1: &Self::Array,
        negative1: &Self::Array,
        positive2: &Self::Array,
        x: &Self::Array,
        y: &Self::Array,
    ) -> (f64, f64) {
        scalar_prods3(
            positive1.col_ref(0),
            negative1.col_ref(0),
            positive2.col_ref(0),
            x.col_ref(0),
            y.col_ref(0),
        )
    }

    fn scalar_prods2(
        &mut self,
        positive1: &Self::Array,
        positive2: &Self::Array,
        x: &Self::Array,
        y: &Self::Array,
    ) -> (f64, f64) {
        scalar_prods2(
            positive1.col_ref(0),
            positive2.col_ref(0),
            x.col_ref(0),
            y.col_ref(0),
        )
    }

    fn read_from_slice(&mut self, dest: &mut Self::Array, source: &[f64]) {
        dest.col_mut(0).copy_from_slice(source);
    }

    fn write_to_slice(&mut self, source: &Self::Array, dest: &mut [f64]) {
        dest.copy_from_slice(source.col_ref(0))
    }

    fn copy_into(&mut self, array: &Self::Array, dest: &mut Self::Array) {
        dest.clone_from(array)
    }

    fn axpy_out(&mut self, x: &Self::Array, y: &Self::Array, a: f64, out: &mut Self::Array) {
        axpy_out(x.col_ref(0), y.col_ref(0), a, out.col_mut(0));
    }

    fn axpy(&mut self, x: &Self::Array, y: &mut Self::Array, a: f64) {
        axpy(x.col_ref(0), y.col_mut(0), a);
    }

    fn fill_array(&mut self, array: &mut Self::Array, val: f64) {
        array.fill(val);
    }

    fn array_all_finite(&mut self, array: &Self::Array) -> bool {
        array.is_all_finite()
    }

    fn array_all_finite_and_nonzero(&mut self, array: &Self::Array) -> bool {
        array
            .col_ref(0)
            .iter()
            .all(|&x| x.is_finite() & (x != 0f64))
    }

    fn array_mult(&mut self, array1: &Self::Array, array2: &Self::Array, dest: &mut Self::Array) {
        multiply(array1.col_ref(0), array2.col_ref(0), dest.col_mut(0))
    }

    fn array_vector_dot(&mut self, array1: &Self::Array, array2: &Self::Array) -> f64 {
        vector_dot(array1.col_ref(0), array2.col_ref(0))
    }

    fn array_gaussian<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Array,
        stds: &Self::Array,
    ) {
        let dist = rand_distr::StandardNormal;
        dest.col_mut(0)
            .iter_mut()
            .zip(stds.col_ref(0).iter())
            .for_each(|(p, &s)| {
                let norm: f64 = rng.sample(dist);
                *p = s * norm;
            });
    }

    fn array_update_variance(
        &mut self,
        mean: &mut Self::Array,
        variance: &mut Self::Array,
        value: &Self::Array,
        diff_scale: f64, // 1 / self.count
    ) {
        izip!(
            mean.col_mut(0).iter_mut(),
            variance.col_mut(0).iter_mut(),
            value.col_ref(0)
        )
        .for_each(|(mean, mut var, x)| {
            let diff = x - *mean;
            *mean += diff * diff_scale;
            *var += diff * diff;
        });
    }

    fn array_update_var_inv_std_draw_grad(
        &mut self,
        variance_out: &mut Self::Array,
        inv_std: &mut Self::Array,
        draw_var: &Self::Array,
        grad_var: &Self::Array,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        izip!(
            variance_out.col_mut(0).iter_mut(),
            inv_std.col_mut(0).iter_mut(),
            draw_var.col_ref(0).iter(),
            grad_var.col_ref(0).iter(),
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
    }

    fn array_update_var_inv_std_grad(
        &mut self,
        variance_out: &mut Self::Array,
        inv_std: &mut Self::Array,
        gradient: &Self::Array,
        fill_invalid: f64,
        clamp: (f64, f64),
    ) {
        izip!(
            variance_out.col_mut(0).iter_mut(),
            inv_std.col_mut(0).iter_mut(),
            gradient.col_ref(0).iter(),
        )
        .for_each(|(var_out, inv_std_out, &grad_var)| {
            let val = grad_var.abs().clamp(clamp.0, clamp.1).recip();
            let val = if val.is_finite() { val } else { fill_invalid };
            *var_out = val;
            *inv_std_out = val.recip().sqrt();
        });
    }
}

pub trait CpuLogpFunc {
    type LogpError: Debug + Send + Sync + Error + LogpError + 'static;

    fn dim(&self) -> usize;
    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpError>;
}
