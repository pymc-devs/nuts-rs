use itertools::izip;
use multiversion::multiversion;

#[cfg(feature = "simd_support")]
use std::simd::{f64x4, num::SimdFloat, StdFloat};

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

#[cfg(feature = "simd_support")]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn multiply(x: &[f64], y: &[f64], out: &mut [f64]) {
    let n = x.len();
    assert!(y.len() == n);
    assert!(out.len() == n);

    let (out, out_tail) = out.as_chunks_mut();
    let (x, x_tail) = x.as_chunks();
    let (y, y_tail) = y.as_chunks();

    izip!(out, x, y).for_each(|(out, x, y)| {
        let x = f64x4::from_array(*x);
        let y = f64x4::from_array(*y);
        *out = (x * y).to_array();
    });

    izip!(out_tail.iter_mut(), x_tail.iter(), y_tail.iter()).for_each(|(out, &x, &y)| {
        *out = x * y;
    });
}

#[cfg(not(feature = "simd_support"))]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn multiply(x: &[f64], y: &[f64], out: &mut [f64]) {
    let n = x.len();
    assert!(y.len() == n);
    assert!(out.len() == n);

    izip!(out.iter_mut(), x.iter(), y.iter()).for_each(|(out, &x, &y)| {
        *out = x * y;
    });
}

#[cfg(feature = "simd_support")]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn scalar_prods2(positive1: &[f64], positive2: &[f64], x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = positive1.len();

    assert!(positive1.len() == n);
    assert!(positive2.len() == n);
    assert!(x.len() == n);
    assert!(y.len() == n);

    let zero = f64x4::splat(0.);

    let (a, a_tail) = positive1.as_chunks();
    let (b, b_tail) = positive2.as_chunks();
    let (c, c_tail) = x.as_chunks();
    let (d, d_tail) = y.as_chunks();

    let out = izip!(a, b, c, d)
        .map(|(&a, &b, &c, &d)| {
            (
                f64x4::from_array(a),
                f64x4::from_array(b),
                f64x4::from_array(c),
                f64x4::from_array(d),
            )
        })
        .fold((zero, zero), |(s1, s2), (a, b, c, d)| {
            let sum = a + b;
            (c.mul_add(sum, s1), d.mul_add(sum, s2))
        });
    let out_head = (out.0.reduce_sum(), out.1.reduce_sum());

    let out = izip!(a_tail, b_tail, c_tail, d_tail,).fold((0., 0.), |(s1, s2), (a, b, c, d)| {
        (s1 + c * (a + b), s2 + d * (a + b))
    });

    (out_head.0 + out.0, out_head.1 + out.1)
}

#[cfg(not(feature = "simd_support"))]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn scalar_prods2(positive1: &[f64], positive2: &[f64], x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = positive1.len();

    assert!(positive1.len() == n);
    assert!(positive2.len() == n);
    assert!(x.len() == n);
    assert!(y.len() == n);

    izip!(positive1, positive2, x, y).fold((0., 0.), |(s1, s2), (a, b, c, d)| {
        (s1 + c * (a + b), s2 + d * (a + b))
    })
}

#[cfg(feature = "simd_support")]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn scalar_prods3(
    positive1: &[f64],
    negative1: &[f64],
    positive2: &[f64],
    x: &[f64],
    y: &[f64],
) -> (f64, f64) {
    let n = positive1.len();

    assert!(positive1.len() == n);
    assert!(positive2.len() == n);
    assert!(negative1.len() == n);
    assert!(x.len() == n);
    assert!(y.len() == n);

    let zero = f64x4::splat(0.);

    let (a, a_tail) = positive1.as_chunks();
    let (b, b_tail) = negative1.as_chunks();
    let (c, c_tail) = positive2.as_chunks();
    let (x, x_tail) = x.as_chunks();
    let (y, y_tail) = y.as_chunks();

    let out = izip!(a, b, c, x, y)
        .map(|(&a, &b, &c, &x, &y)| {
            (
                f64x4::from_array(a),
                f64x4::from_array(b),
                f64x4::from_array(c),
                f64x4::from_array(x),
                f64x4::from_array(y),
            )
        })
        .fold((zero, zero), |(s1, s2), (a, b, c, x, y)| {
            let sum = a - b + c;
            (x.mul_add(sum, s1), y.mul_add(sum, s2))
        });
    let out_head = (out.0.reduce_sum(), out.1.reduce_sum());

    let out = izip!(a_tail, b_tail, c_tail, x_tail, y_tail)
        .fold((0., 0.), |(s1, s2), (a, b, c, x, y)| {
            (s1 + x * (a - b + c), s2 + y * (a - b + c))
        });

    (out_head.0 + out.0, out_head.1 + out.1)
}

#[cfg(not(feature = "simd_support"))]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn scalar_prods3(
    positive1: &[f64],
    negative1: &[f64],
    positive2: &[f64],
    x: &[f64],
    y: &[f64],
) -> (f64, f64) {
    let n = positive1.len();

    assert!(positive1.len() == n);
    assert!(positive2.len() == n);
    assert!(negative1.len() == n);
    assert!(x.len() == n);
    assert!(y.len() == n);

    izip!(positive1, negative1, positive2, x, y).fold((0., 0.), |(s1, s2), (a, b, c, x, y)| {
        (s1 + x * (a - b + c), s2 + y * (a - b + c))
    })
}

#[cfg(feature = "simd_support")]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn vector_dot(a: &[f64], b: &[f64]) -> f64 {
    assert!(a.len() == b.len());

    let (x, x_tail) = a.as_chunks();
    let (y, y_tail) = b.as_chunks();

    let sum: f64x4 = izip!(x, y)
        .map(|(&x, &y)| {
            let x = f64x4::from_array(x);
            let y = f64x4::from_array(y);
            x * y
        })
        .sum();

    let mut result = sum.reduce_sum();
    for (val1, val2) in x_tail.iter().zip(y_tail).take(3) {
        result += *val1 * *val2;
    }
    result
}

#[cfg(not(feature = "simd_support"))]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn vector_dot(a: &[f64], b: &[f64]) -> f64 {
    assert!(a.len() == b.len());

    let mut result = 0f64;
    for (&val1, &val2) in a.iter().zip(b) {
        result += val1 * val2;
    }
    result
}

#[cfg(feature = "simd_support")]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn axpy(x: &[f64], y: &mut [f64], a: f64) {
    let n = x.len();
    assert!(y.len() == n);

    let (x, x_tail) = x.as_chunks();
    let (y, y_tail) = y.as_chunks_mut();

    let a_splat = f64x4::splat(a);

    izip!(x, y).for_each(|(x, y)| {
        let x = f64x4::from_array(*x);
        let y_val = f64x4::from_array(*y);
        let out = x.mul_add(a_splat, y_val);
        *y = out.to_array();
    });

    izip!(x_tail, y_tail).for_each(|(x, y)| {
        *y = x.mul_add(a, *y);
    });
}

#[cfg(not(feature = "simd_support"))]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn axpy(x: &[f64], y: &mut [f64], a: f64) {
    let n = x.len();
    assert!(y.len() == n);

    izip!(x, y).for_each(|(x, y)| {
        *y = x.mul_add(a, *y);
    });
}

#[cfg(feature = "simd_support")]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn axpy_out(x: &[f64], y: &[f64], a: f64, out: &mut [f64]) {
    let n = x.len();
    assert!(y.len() == n);
    assert!(out.len() == n);

    let (x, x_tail) = x.as_chunks();
    let (y, y_tail) = y.as_chunks();
    let (out, out_tail) = out.as_chunks_mut();

    let a_splat = f64x4::splat(a);

    izip!(x, y, out).for_each(|(&x, &y, out)| {
        let x = f64x4::from_array(x);
        let y_val = f64x4::from_array(y);

        *out = x.mul_add(a_splat, y_val).to_array();
    });

    izip!(x_tail, y_tail, out_tail)
        .take(3)
        .for_each(|(&x, &y, out)| {
            *out = a.mul_add(x, y);
        });
}

#[cfg(not(feature = "simd_support"))]
#[multiversion(targets("x86_64+avx+avx2+fma", "arm+neon"))]
pub fn axpy_out(x: &[f64], y: &[f64], a: f64, out: &mut [f64]) {
    let n = x.len();
    assert!(y.len() == n);
    assert!(out.len() == n);

    izip!(x, y, out).for_each(|(&x, &y, out)| {
        *out = a.mul_add(x, y);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    fn assert_approx_eq(a: f64, b: f64) {
        if a.is_nan() && b.is_nan() | b.is_infinite() {
            return;
        }
        if b.is_nan() && a.is_nan() | a.is_infinite() {
            return;
        }
        assert_ulps_eq!(a, b, max_ulps = 8);
    }

    prop_compose! {
        fn array2(maxsize: usize) (size in 0..maxsize) (
            vec1 in prop::collection::vec(prop::num::f64::ANY, size),
            vec2 in prop::collection::vec(prop::num::f64::ANY, size)
        )
        -> (Vec<f64>, Vec<f64>) {
            (vec1, vec2)
        }
    }

    prop_compose! {
        fn array3(maxsize: usize) (size in 0..maxsize) (
            vec1 in prop::collection::vec(prop::num::f64::ANY, size),
            vec2 in prop::collection::vec(prop::num::f64::ANY, size),
            vec3 in prop::collection::vec(prop::num::f64::ANY, size)
        )
        -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            (vec1, vec2, vec3)
        }
    }

    prop_compose! {
        fn array4(maxsize: usize) (size in 0..maxsize) (
            vec1 in prop::collection::vec(prop::num::f64::ANY, size),
            vec2 in prop::collection::vec(prop::num::f64::ANY, size),
            vec3 in prop::collection::vec(prop::num::f64::ANY, size),
            vec4 in prop::collection::vec(prop::num::f64::ANY, size)
        )
        -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
            (vec1, vec2, vec3, vec4)
        }
    }

    prop_compose! {
        fn array5(maxsize: usize) (size in 0..maxsize) (
            vec1 in prop::collection::vec(prop::num::f64::ANY, size),
            vec2 in prop::collection::vec(prop::num::f64::ANY, size),
            vec3 in prop::collection::vec(prop::num::f64::ANY, size),
            vec4 in prop::collection::vec(prop::num::f64::ANY, size),
            vec5 in prop::collection::vec(prop::num::f64::ANY, size)
        )
        -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
            (vec1, vec2, vec3, vec4, vec5)
        }
    }

    proptest! {
        #[test]
        fn check_logaddexp(x in -10f64..10f64, y in -10f64..10f64) {
            let a = (x.exp() + y.exp()).ln();
            let b = logaddexp(x, y);
            let neginf = f64::NEG_INFINITY;
            let nan = f64::NAN;
            prop_assert!((a - b).abs() < 1e-10);
            prop_assert_eq!(b, logaddexp(y, x));
            prop_assert_eq!(x, logaddexp(x, neginf));
            prop_assert_eq!(logaddexp(neginf, neginf), neginf);
            prop_assert!(logaddexp(nan, x).is_nan());
        }

        #[test]
        fn test_axpy((x, y) in array2(10), a in prop::num::f64::ANY) {
            let orig = y.clone();
            let mut y = y.clone();
            axpy(&x[..], &mut y[..], a);
            for ((&x, y), out) in x.iter().zip(orig).zip(y) {
                assert_approx_eq(out, a.mul_add(x, y));
            }
        }

        #[test]
        fn test_scalar_prods2((x1, x2, y1, y2) in array4(10)) {
            let (p1, p2) = scalar_prods2(&x1[..], &x2[..], &y1[..], &y2[..]);
            let x1 = ndarray::Array1::from_vec(x1);
            let x2 = ndarray::Array1::from_vec(x2);
            let y1 = ndarray::Array1::from_vec(y1);
            let y2 = ndarray::Array1::from_vec(y2);
            assert_approx_eq(p1, (&x1 + &x2).dot(&y1));
            assert_approx_eq(p2, (&x1 + &x2).dot(&y2));
        }

        #[test]
        fn test_scalar_prods3((x1, x2, x3, y1, y2) in array5(10)) {
            let (p1, p2) = scalar_prods3(&x1[..], &x2[..], &x3[..], &y1[..], &y2[..]);
            let x1 = ndarray::Array1::from_vec(x1);
            let x2 = ndarray::Array1::from_vec(x2);
            let x3 = ndarray::Array1::from_vec(x3);
            let y1 = ndarray::Array1::from_vec(y1);
            let y2 = ndarray::Array1::from_vec(y2);
            assert_approx_eq(p1, (&x1 - &x2 + &x3).dot(&y1));
            assert_approx_eq(p2, (&x1 - &x2 + &x3).dot(&y2));
        }

        #[test]
        fn test_axpy_out(a in prop::num::f64::ANY, (x, y, out) in array3(10)) {
            let mut out = out.clone();
            axpy_out(&x[..], &y[..], a, &mut out[..]);
            let x = ndarray::Array1::from_vec(x);
            let mut y = ndarray::Array1::from_vec(y);
            y.scaled_add(a, &x);
            for (&out1, out2) in out.iter().zip(y) {
                assert_approx_eq(out1, out2);
            }
        }

        #[test]
        fn test_multiplty((x, y, out) in array3(10)) {
            let mut out = out.clone();
            multiply(&x[..], &y[..], &mut out[..]);
            let x = ndarray::Array1::from_vec(x);
            let y = ndarray::Array1::from_vec(y);
            for (&out1, out2) in out.iter().zip(&x * &y) {
                assert_approx_eq(out1, out2);
            }
        }

        #[test]
        fn test_vector_dot((x, y) in array2(10)) {
            let actual = vector_dot(&x[..], &y[..]);
            let x = ndarray::Array1::from_vec(x);
            let y = ndarray::Array1::from_vec(y);
            let expected = x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum();
            assert_approx_eq(actual, expected);
        }
    }

    #[test]
    fn check_neginf() {
        assert_eq!(logaddexp(f64::NEG_INFINITY, 2.), 2.);
        assert_eq!(logaddexp(2., f64::NEG_INFINITY), 2.);
    }
}
