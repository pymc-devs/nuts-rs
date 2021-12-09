use itertools::izip;
use multiversion::multiversion;
use std::simd::f64x4;

#[inline]
pub(crate) fn logaddexp(a: f64, b: f64) -> f64 {
    if a == b {
        return a + 2f64.ln();
    }
    let diff = a - b;
    if diff > 0. {
        a + (-diff).exp().ln_1p()
    } else if diff < 0. {
        b + diff.exp().ln_1p()
    } else {
        // diff is NAN
        diff
    }
}

//pub(crate) fn scalar_prods_of_diff(a: &Array1<f64>, b: &Array1<f64>, c: &Array1<f64>, d: &Array1<f64>) -> (f64, f64) {
//#[allow(clippy::many_single_char_names)]
#[multiversion]
#[clone(target = "[x84|x86_64]+avx+avx2")]
#[clone(target = "x86+sse")]
pub fn scalar_prods_of_diff(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> (f64, f64) {
    let n = a.len();
    assert!(b.len() == n);
    assert!(c.len() == n);
    assert!(d.len() == n);

    /*
    (a.par_chunks(1024), b.par_chunks(1024), c.par_chunks(1024), d.par_chunks(1024))
        .into_par_iter()
        //.with_min_len(1024)
        .map(
            |(a, b, c, d)| {
                let mut sum_c: f64 = 0.;
                let mut sum_d: f64 = 0.;
                for i in 0..a.len() {
                    sum_c += (a[i] - b[i]) * c[i];
                    //sum_c = c[i].mul_add(a[i] - b[i], sum_c);
                    sum_d += (a[i] - b[i]) * d[i];
                    //sum_d = d[i].mul_add(a[i] - b[i], sum_d);
                }
                (sum_c, sum_d)
                //(s1 + c * (a - b), s2 + d * (a - b))
            }
        )
        .reduce(|| (0., 0.), |(s1, s2), (s12, s22)| (s1 + s12, s2 + s22))
    */
    /*
    (a, b, c, d)
        .into_par_iter()
        .with_min_len(1024)
        .fold(
            || (0., 0.),
            |(s1, s2), (a, b, c, d)| {
                let mut sum_c: f64 = 0.;
                let mut sum_d: f64 = 0.;
                for i in 0..a.len() {
                    sum_c += (a[i] - b[i]) * c[i];
                    //sum_c = c[i].mul_add(a[i] - b[i], sum_c);
                    sum_d += (a[i] - b[i]) * d[i];
                    //sum_d = d[i].mul_add(a[i] - b[i], sum_d);
                }
                (s1 + sum_c, s2 + sum_d)
                //(s1 + c * (a - b), s2 + d * (a - b))
            }
        ).reduce(
            || (0., 0.),
            |(sum1, sum2), (val1, val2)| (sum1 + val1, sum2 + val2)
        )
    */
    let zero = f64x4::splat(0.);

    let head_length = n - n % 4;

    let (a, a_tail) = a.split_at(head_length);
    let (b, b_tail) = b.split_at(head_length);
    let (c, c_tail) = c.split_at(head_length);
    let (d, d_tail) = d.split_at(head_length);

    let out = izip!(
        a.chunks_exact(4),
        b.chunks_exact(4),
        c.chunks_exact(4),
        d.chunks_exact(4)
    )
    .map(|(a, b, c, d)| {
        (
            f64x4::from_slice(a),
            f64x4::from_slice(b),
            f64x4::from_slice(c),
            f64x4::from_slice(d),
        )
    })
    .fold((zero, zero), |(s1, s2), (a, b, c, d)| {
        (s1 + c * (a - b), s2 + d * (a - b))
    });
    let out_head = (out.0.horizontal_sum(), out.1.horizontal_sum());

    let out = izip!(a_tail, b_tail, c_tail, d_tail,).fold((0., 0.), |(s1, s2), (a, b, c, d)| {
        (s1 + c * (a - b), s2 + d * (a - b))
    });

    (out_head.0 + out.0, out_head.1 + out.1)
}

#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2")]
#[clone(target = "x86+sse")]
pub fn vector_dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    assert!(a.len() == b.len());
    //assert_eq!(&*&a[0] as *const f64 as usize % 16, 0);
    //assert_eq!(&*&b[0] as *const f64 as usize % 16, 0);

    //let head_length = 4 * (n / 4);
    let head_length = n - n % 4;

    let (x, x_tail) = a.split_at(head_length);
    let (y, y_tail) = b.split_at(head_length);

    let sum: f64x4 = izip!(x.chunks_exact(4), y.chunks_exact(4),)
        .map(|(x, y)| {
            let x = f64x4::from_slice(x);
            let y = f64x4::from_slice(y);
            x * y
        })
        .sum();

    let mut result = sum.horizontal_sum();
    for (val1, val2) in x_tail.iter().zip(y_tail) {
        result += *val1 * *val2;
    }
    result
}

#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2")]
#[clone(target = "x86+sse")]
pub fn axpy(x: &[f64], y: &mut [f64], a: f64) {
    let n = x.len();
    assert!(y.len() == n);

    /*
    for i in 0..n {
        y[i] += a * x[i];
    }
    */

    let head_length = n - n % 4;

    let (x, x_tail) = x.split_at(head_length);
    let (y, y_tail) = y.split_at_mut(head_length);

    izip!(x.chunks_exact(4), y.chunks_exact_mut(4),).for_each(|(x, y)| {
        let x = f64x4::from_slice(x);
        let y_val = f64x4::from_slice(y);
        let out = a * x + y_val;
        y.copy_from_slice(&out.to_array())
    });

    izip!(x_tail, y_tail).for_each(|(x, y)| {
        *y += a * x;
    });
}

#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2")]
#[clone(target = "x86+sse")]
pub fn axpy_out(x: &[f64], y: &[f64], a: f64, out: &mut [f64]) {
    let n = x.len();
    assert!(y.len() == n);
    assert!(out.len() == n);

    /*
    for i in 0..n {
        out[i] = y[i] + a * x[i];
    }
    */

    let head_length = 4 * (n / 4);

    let (x, x_tail) = x.split_at(head_length);
    let (y, y_tail) = y.split_at(head_length);
    let (out, out_tail) = out.split_at_mut(head_length);

    izip!(
        x.chunks_exact(4),
        y.chunks_exact(4),
        out.chunks_exact_mut(4),
    )
    .for_each(|(x, y, out)| {
        let x = f64x4::from_slice(x);
        let y_val = f64x4::from_slice(y);
        let out_val = a * x + y_val;
        out.copy_from_slice(&out_val.to_array())
    });

    izip!(x_tail, y_tail, out_tail).for_each(|(x, y, out)| {
        *out = a * x + y;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn check_logaddexp(x in -10f64..10f64, y in -10f64..10f64) {
            let a = (x.exp() + y.exp()).ln();
            let b = logaddexp(x, y);
            let neginf = std::f64::NEG_INFINITY;
            let nan = std::f64::NAN;
            prop_assert!((a - b).abs() < 1e-10);
            prop_assert_eq!(b, logaddexp(y, x));
            prop_assert_eq!(x, logaddexp(x, neginf));
            prop_assert_eq!(logaddexp(neginf, neginf), neginf);
            prop_assert!(logaddexp(nan, x).is_nan());
        }
    }

    #[test]
    fn check_neginf() {
        assert_eq!(logaddexp(std::f64::NEG_INFINITY, 2.), 2.);
        assert_eq!(logaddexp(2., std::f64::NEG_INFINITY), 2.);
    }
}
