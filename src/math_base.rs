use std::{error::Error, fmt::Debug};

use crate::LogpError;

pub trait Math {
    type Array: Debug;
    type LogpErr: Debug + Send + Sync + LogpError + 'static;
    type Err: Debug + Send + Sync + Error + 'static;

    fn new_array(&self) -> Self::Array;

    /// Compute the unnormalized log probability density of the posterior
    ///
    /// This needs to be implemnted by users of the library to define
    /// what distribution the users wants to sample from.
    ///
    /// Errors during that computation can be recoverable or non-recoverable.
    /// If a non-recoverable error occurs during sampling, the sampler will
    /// stop and return an error.
    fn logp_array(
        &mut self,
        position: &Self::Array,
        gradient: &mut Self::Array,
    ) -> Result<f64, Self::LogpErr>;

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpErr>;

    fn dim(&self) -> usize;

    fn scalar_prods3(
        &mut self,
        positive1: &Self::Array,
        negative1: &Self::Array,
        positive2: &Self::Array,
        x: &Self::Array,
        y: &Self::Array,
    ) -> (f64, f64);

    fn scalar_prods2(
        &mut self,
        positive1: &Self::Array,
        positive2: &Self::Array,
        x: &Self::Array,
        y: &Self::Array,
    ) -> (f64, f64);

    fn read_from_slice(&mut self, dest: &mut Self::Array, source: &[f64]);
    fn write_to_slice(&mut self, source: &Self::Array, dest: &mut [f64]);
    fn copy_into(&mut self, array: &Self::Array, dest: &mut Self::Array);
    fn axpy_out(&mut self, x: &Self::Array, y: &Self::Array, a: f64, out: &mut Self::Array);
    fn axpy(&mut self, x: &Self::Array, y: &mut Self::Array, a: f64);

    fn box_array(&mut self, array: &Self::Array) -> Box<[f64]> {
        let mut data = vec![0f64; self.dim()];
        self.write_to_slice(array, &mut data);
        data.into()
    }

    fn fill_array(&mut self, array: &mut Self::Array, val: f64);

    fn array_all_finite(&mut self, array: &Self::Array) -> bool;
    fn array_all_finite_and_nonzero(&mut self, array: &Self::Array) -> bool;
    fn array_mult(&mut self, array1: &Self::Array, array2: &Self::Array, dest: &mut Self::Array);
    fn array_vector_dot(&mut self, array1: &Self::Array, array2: &Self::Array) -> f64;
    fn array_gaussian<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Array,
        stds: &Self::Array,
    );
    fn array_update_variance(
        &mut self,
        mean: &mut Self::Array,
        variance: &mut Self::Array,
        value: &Self::Array,
        diff_scale: f64,
    );
    fn array_update_var_inv_std_draw_grad(
        &mut self,
        variance_out: &mut Self::Array,
        inv_std: &mut Self::Array,
        draw_var: &Self::Array,
        grad_var: &Self::Array,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    );

    fn array_update_var_inv_std_grad(
        &mut self,
        variance_out: &mut Self::Array,
        inv_std: &mut Self::Array,
        gradient: &Self::Array,
        fill_invalid: f64,
        clamp: (f64, f64),
    );
}

trait Array {
    fn write(&self, out: &mut [f64]);
    fn elemwise_mult(&self, other: &Self, out: &mut Self);
    fn len(&self) -> usize;
}
