use multiversion::multiversion;
use ndarray::{ArrayView1, ShapeBuilder, Zip};
use rayon::prelude::*;

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

#[allow(clippy::many_single_char_names)]
#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2")]
#[clone(target = "x86+sse")]
pub(crate) fn scalar_prods_of_diff(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> (f64, f64) {
    let n = a.len();
    assert_eq!(&*&a[0] as *const f64 as usize % 16, 0);
    assert_eq!(&*&b[0] as *const f64 as usize % 16, 0);
    assert_eq!(&*&c[0] as *const f64 as usize % 16, 0);
    assert_eq!(&*&d[0] as *const f64 as usize % 16, 0);
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

    let mut sum_c: f64 = 0.;
    let mut sum_d: f64 = 0.;
    for i in 0..n {
        sum_c += (a[i] - b[i]) * c[i];
        //sum_c = c[i].mul_add(a[i] - b[i], sum_c);
        sum_d += (a[i] - b[i]) * d[i];
        //sum_d = d[i].mul_add(a[i] - b[i], sum_d);
    }
    (sum_c, sum_d)

    /*
    let a = ArrayView1::from_shape((n,).strides((1,)), a).unwrap();
    let b = ArrayView1::from_shape((n,).strides((1,)), b).unwrap();
    let c = ArrayView1::from_shape((n,).strides((1,)), c).unwrap();
    let d = ArrayView1::from_shape((n,).strides((1,)), d).unwrap();

    let result = Zip::from(&a).and(&b).and(&c).and(&d);
    //result.fold((0., 0.), |(s1, s2), a, b, c, d| (s1 + c * (a - b), s2 + d * (a - b)))
    result.into_par_iter().with_min_len(1024).fold(|| (0., 0.), |(s1, s2), (a, b, c, d)| (s1 + c * (a - b), s2 + d * (a - b))).reduce(|| (0., 0.), |(s1, s2), (s21, s22)| (s1 + s21, s2 + s22))
    */
}

#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2")]
#[clone(target = "x86+sse")]
pub(crate) fn norm2(a: &[f64], b: &[f64]) -> f64 {
    assert!(a.len() == b.len());
    assert_eq!(&*&a[0] as *const f64 as usize % 16, 0);
    assert_eq!(&*&b[0] as *const f64 as usize % 16, 0);
    let mut result = 0.;
    for (val1, val2) in a.iter().zip(b) {
        result += *val1 * *val2;
    }
    result
}

#[inline]
pub(crate) fn norm(a: &[f64]) -> f64 {
    let mut result = 0.;
    for val in a.iter() {
        result += (*val) * (*val);
    }
    result
}

#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2")]
#[clone(target = "x86+sse")]
pub(crate) fn axpy(x: &[f64], y: &mut [f64], a: f64) {
    let n = x.len();
    assert_eq!(&*&x[0] as *const f64 as usize % 16, 0);
    assert_eq!(&*&y[0] as *const f64 as usize % 16, 0);
    assert!(y.len() == n);
    for i in 0..n {
        y[i] += a * x[i];
        //y[i] = a.mul_add(x[i], y[i]);
    }
}

#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2")]
#[clone(target = "x86+sse")]
pub(crate) fn axpy_out(x: &[f64], y: &[f64], a: f64, out: &mut [f64]) {
    let n = x.len();
    assert_eq!(&*&x[0] as *const f64 as usize % 16, 0);
    assert_eq!(&*&y[0] as *const f64 as usize % 16, 0);
    assert_eq!(&*&out[0] as *const f64 as usize % 16, 0);
    assert!(y.len() == n);
    assert!(out.len() == n);
    for i in 0..n {
        out[i] = y[i] + a * x[i];
        //out[i] = a.mul_add(x[i], y[i]);
    }
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
