use std::{error::Error, fmt::Debug};

use crate::LogpError;

pub trait Math {
    type Vector: Debug;
    type EigMatrix: Debug;
    type EigVector: Debug;
    type LogpErr: Debug + Send + Sync + LogpError + 'static;
    type Err: Debug + Send + Sync + Error + 'static;

    fn new_array(&self) -> Self::Vector;

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
        position: &Self::Vector,
        gradient: &mut Self::Vector,
    ) -> Result<f64, Self::LogpErr>;

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpErr>;

    fn dim(&self) -> usize;

    fn scalar_prods3(
        &mut self,
        positive1: &Self::Vector,
        negative1: &Self::Vector,
        positive2: &Self::Vector,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> (f64, f64);

    fn scalar_prods2(
        &mut self,
        positive1: &Self::Vector,
        positive2: &Self::Vector,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> (f64, f64);

    fn read_from_slice(&mut self, dest: &mut Self::Vector, source: &[f64]);
    fn write_to_slice(&mut self, source: &Self::Vector, dest: &mut [f64]);
    fn copy_into(&mut self, array: &Self::Vector, dest: &mut Self::Vector);
    fn axpy_out(&mut self, x: &Self::Vector, y: &Self::Vector, a: f64, out: &mut Self::Vector);
    fn axpy(&mut self, x: &Self::Vector, y: &mut Self::Vector, a: f64);

    fn box_array(&mut self, array: &Self::Vector) -> Box<[f64]> {
        let mut data = vec![0f64; self.dim()];
        self.write_to_slice(array, &mut data);
        data.into()
    }

    fn fill_array(&mut self, array: &mut Self::Vector, val: f64);

    fn array_all_finite(&mut self, array: &Self::Vector) -> bool;
    fn array_all_finite_and_nonzero(&mut self, array: &Self::Vector) -> bool;
    fn array_mult(&mut self, array1: &Self::Vector, array2: &Self::Vector, dest: &mut Self::Vector);
    fn array_vector_dot(&mut self, array1: &Self::Vector, array2: &Self::Vector) -> f64;
    fn array_gaussian<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Vector,
        stds: &Self::Vector,
    );
    fn array_update_variance(
        &mut self,
        mean: &mut Self::Vector,
        variance: &mut Self::Vector,
        value: &Self::Vector,
        diff_scale: f64,
    );
    fn array_update_var_inv_std_draw(
        &mut self,
        variance_out: &mut Self::Vector,
        inv_std: &mut Self::Vector,
        draw_var: &Self::Vector,
        scale: f64,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    );
    fn array_update_var_inv_std_draw_grad(
        &mut self,
        variance_out: &mut Self::Vector,
        inv_std: &mut Self::Vector,
        draw_var: &Self::Vector,
        grad_var: &Self::Vector,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    );

    fn array_update_var_inv_std_grad(
        &mut self,
        variance_out: &mut Self::Vector,
        inv_std: &mut Self::Vector,
        gradient: &Self::Vector,
        fill_invalid: f64,
        clamp: (f64, f64),
    );
}
