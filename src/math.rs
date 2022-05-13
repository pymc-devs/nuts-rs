use itertools::izip;
use multiversion::multiversion;
use std::simd::{f64x4, StdFloat};

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

#[multiversion]
#[clone(target = "[x64|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse")]
pub fn multiply(x: &[f64], y: &[f64], out: &mut [f64]) {
    let n = x.len();
    assert!(y.len() == n);
    assert!(out.len() == n);

    let (out, out_tail) = out.as_chunks_mut();
    let (x, x_tail) = x.as_chunks();
    let (y, y_tail) = y.as_chunks();

    izip!(out, x, y)
    .for_each(|(out, x, y)| {
        let x = f64x4::from_array(*x);
        let y = f64x4::from_array(*y);
        *out = (x * y).to_array();
    });

    izip!(out_tail.iter_mut(), x_tail.iter(), y_tail.iter()).for_each(
        |(out, &x, &y)| {
            *out = x * y;
        },
    );
}


#[multiversion]
#[clone(target = "[x84|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse")]
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

#[multiversion]
#[clone(target = "[x84|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse")]
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
        .take(3)
        .fold((0., 0.), |(s1, s2), (a, b, c, x, y)| {
            (s1 + x * (a - b + c), s2 + y * (a - b + c))
        });

    (out_head.0 + out.0, out_head.1 + out.1)
}

#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse")]
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

#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse")]
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

    izip!(x_tail, y_tail).take(3).for_each(|(x, y)| {
        *y = x.mul_add(a, *y);
    });
}

#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2+fma")]
#[clone(target = "x86+sse+fma")]
pub fn axpy_out(x: &[f64], y: &[f64], a: f64, out: &mut [f64]) {
    let n = x.len();
    assert!(y.len() == n);
    assert!(out.len() == n);

    let (x, x_tail) = x.as_chunks();
    let (y, y_tail) = y.as_chunks();
    let (out, out_tail) = out.as_chunks_mut();

    let a_splat = f64x4::splat(a);

    izip!(x, y, out)
    .for_each(|(&x, &y, out)| {
        let x = f64x4::from_array(x);
        let y_val = f64x4::from_array(y);

        //let out_val = a_splat * x + y_val;
        *out = x.mul_add(a_splat, y_val).to_array();
    });

    izip!(x_tail, y_tail, out_tail).take(3).for_each(|(&x, &y, out)| {
        *out = a.mul_add(x, y);
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
