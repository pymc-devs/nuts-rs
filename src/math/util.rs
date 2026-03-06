use itertools::izip;
use pulp::{Arch, WithSimd};

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

struct Multiply<'a> {
    x: &'a [f64],
    y: &'a [f64],
    out: &'a mut [f64],
}

impl<'a> WithSimd for Multiply<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let x = self.x;
        let y = self.y;
        let out = self.out;

        let (x_out, x_tail) = S::as_simd_f64s(x);
        let (y_out, y_tail) = S::as_simd_f64s(y);
        let (out_head, out_tail) = S::as_mut_simd_f64s(out);

        let (out_arrays, out_simd_tail) = pulp::as_arrays_mut::<4, _>(out_head);
        let (x_arrays, x_simd_tail) = pulp::as_arrays::<4, _>(x_out);
        let (y_arrays, y_simd_tail) = pulp::as_arrays::<4, _>(y_out);

        izip!(out_arrays, x_arrays, y_arrays).for_each(
            |([out0, out1, out2, out3], [x0, x1, x2, x3], [y0, y1, y2, y3])| {
                *out0 = simd.mul_f64s(*x0, *y0);
                *out1 = simd.mul_f64s(*x1, *y1);
                *out2 = simd.mul_f64s(*x2, *y2);
                *out3 = simd.mul_f64s(*x3, *y3);
            },
        );

        izip!(
            out_simd_tail.iter_mut(),
            x_simd_tail.iter(),
            y_simd_tail.iter()
        )
        .for_each(|(out, &x, &y)| {
            *out = simd.mul_f64s(x, y);
        });

        izip!(out_tail.iter_mut(), x_tail.iter(), y_tail.iter()).for_each(|(out, &x, &y)| {
            *out = x * y;
        });
    }
}

#[inline(never)]
pub fn multiply(arch: Arch, x: &[f64], y: &[f64], out: &mut [f64]) {
    arch.dispatch(Multiply { x, y, out })
}

struct ScalarProds2<'a> {
    positive1: &'a [f64],
    positive2: &'a [f64],
    x: &'a [f64],
    y: &'a [f64],
}

impl<'a> WithSimd for ScalarProds2<'a> {
    type Output = (f64, f64);

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let positive1 = self.positive1;
        let positive2 = self.positive2;
        let x = self.x;
        let y = self.y;

        let (p1_out, p1_tail) = S::as_simd_f64s(positive1);
        let (p2_out, p2_tail) = S::as_simd_f64s(positive2);
        let (x_out, x_tail) = S::as_simd_f64s(x);
        let (y_out, y_tail) = S::as_simd_f64s(y);

        let mut s1_0 = simd.splat_f64s(0.0);
        let mut s1_1 = simd.splat_f64s(0.0);
        let mut s1_2 = simd.splat_f64s(0.0);
        let mut s1_3 = simd.splat_f64s(0.0);
        let mut s2_0 = simd.splat_f64s(0.0);
        let mut s2_1 = simd.splat_f64s(0.0);
        let mut s2_2 = simd.splat_f64s(0.0);
        let mut s2_3 = simd.splat_f64s(0.0);

        let (p1_out, p1_simd_tail) = pulp::as_arrays::<4, _>(p1_out);
        let (p2_out, p2_simd_tail) = pulp::as_arrays::<4, _>(p2_out);
        let (x_out, x_simd_tail) = pulp::as_arrays::<4, _>(x_out);
        let (y_out, y_simd_tail) = pulp::as_arrays::<4, _>(y_out);

        izip!(p1_out, p2_out, x_out, y_out).for_each(
            |(
                [p1_0, p1_1, p1_2, p1_3],
                [p2_0, p2_1, p2_2, p2_3],
                [x_0, x_1, x_2, x_3],
                [y_0, y_1, y_2, y_3],
            )| {
                let sum0 = simd.add_f64s(*p1_0, *p2_0);
                let sum1 = simd.add_f64s(*p1_1, *p2_1);
                let sum2 = simd.add_f64s(*p1_2, *p2_2);
                let sum3 = simd.add_f64s(*p1_3, *p2_3);
                s1_0 = simd.mul_add_e_f64s(sum0, *x_0, s1_0);
                s1_1 = simd.mul_add_e_f64s(sum1, *x_1, s1_1);
                s1_2 = simd.mul_add_e_f64s(sum2, *x_2, s1_2);
                s1_3 = simd.mul_add_e_f64s(sum3, *x_3, s1_3);
                s2_0 = simd.mul_add_e_f64s(sum0, *y_0, s2_0);
                s2_1 = simd.mul_add_e_f64s(sum1, *y_1, s2_1);
                s2_2 = simd.mul_add_e_f64s(sum2, *y_2, s2_2);
                s2_3 = simd.mul_add_e_f64s(sum3, *y_3, s2_3);
            },
        );

        izip!(p1_simd_tail, p2_simd_tail, x_simd_tail, y_simd_tail).for_each(|(p1, p2, x, y)| {
            let sum = simd.add_f64s(*p1, *p2);
            s1_0 = simd.mul_add_e_f64s(sum, *x, s1_0);
            s2_0 = simd.mul_add_e_f64s(sum, *y, s2_0);
        });

        let mut out = (
            simd.reduce_sum_f64s(
                simd.add_f64s(simd.add_f64s(s1_0, s1_1), simd.add_f64s(s1_2, s1_3)),
            ),
            simd.reduce_sum_f64s(
                simd.add_f64s(simd.add_f64s(s2_0, s2_1), simd.add_f64s(s2_2, s2_3)),
            ),
        );

        izip!(p1_tail.iter(), p2_tail.iter(), x_tail.iter(), y_tail.iter()).for_each(
            |(p1, p2, x, y)| {
                let sum = *p1 + *p2;
                out.0 += sum * *x;
                out.1 += sum * *y;
            },
        );
        out
    }
}

#[inline(never)]
pub fn scalar_prods2(
    arch: Arch,
    positive1: &[f64],
    positive2: &[f64],
    x: &[f64],
    y: &[f64],
) -> (f64, f64) {
    let n = positive1.len();

    assert!(positive1.len() == n);
    assert!(positive2.len() == n);
    assert!(x.len() == n);
    assert!(y.len() == n);

    arch.dispatch(ScalarProds2 {
        positive1,
        positive2,
        x,
        y,
    })
}

struct ScalarProds3<'a> {
    positive1: &'a [f64],
    negative1: &'a [f64],
    positive2: &'a [f64],
    x: &'a [f64],
    y: &'a [f64],
}

impl<'a> WithSimd for ScalarProds3<'a> {
    type Output = (f64, f64);

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let positive1 = self.positive1;
        let negative1 = self.negative1;
        let positive2 = self.positive2;
        let x = self.x;
        let y = self.y;

        let (p1_out, p1_tail) = S::as_simd_f64s(positive1);
        let (n1_out, n1_tail) = S::as_simd_f64s(negative1);
        let (p2_out, p2_tail) = S::as_simd_f64s(positive2);
        let (x_out, x_tail) = S::as_simd_f64s(x);
        let (y_out, y_tail) = S::as_simd_f64s(y);

        let mut s1_0 = simd.splat_f64s(0.0);
        let mut s1_1 = simd.splat_f64s(0.0);
        let mut s1_2 = simd.splat_f64s(0.0);
        let mut s1_3 = simd.splat_f64s(0.0);
        let mut s2_0 = simd.splat_f64s(0.0);
        let mut s2_1 = simd.splat_f64s(0.0);
        let mut s2_2 = simd.splat_f64s(0.0);
        let mut s2_3 = simd.splat_f64s(0.0);

        let (p1_out, p1_simd_tail) = pulp::as_arrays::<4, _>(p1_out);
        let (n1_out, n1_simd_tail) = pulp::as_arrays::<4, _>(n1_out);
        let (p2_out, p2_simd_tail) = pulp::as_arrays::<4, _>(p2_out);
        let (x_out, x_simd_tail) = pulp::as_arrays::<4, _>(x_out);
        let (y_out, y_simd_tail) = pulp::as_arrays::<4, _>(y_out);

        izip!(p1_out, n1_out, p2_out, x_out, y_out).for_each(
            |(
                [p1_0, p1_1, p1_2, p1_3],
                [n1_0, n1_1, n1_2, n1_3],
                [p2_0, p2_1, p2_2, p2_3],
                [x_0, x_1, x_2, x_3],
                [y_0, y_1, y_2, y_3],
            )| {
                let sum0 = simd.sub_f64s(simd.add_f64s(*p1_0, *p2_0), *n1_0);
                let sum1 = simd.sub_f64s(simd.add_f64s(*p1_1, *p2_1), *n1_1);
                let sum2 = simd.sub_f64s(simd.add_f64s(*p1_2, *p2_2), *n1_2);
                let sum3 = simd.sub_f64s(simd.add_f64s(*p1_3, *p2_3), *n1_3);
                s1_0 = simd.mul_add_e_f64s(sum0, *x_0, s1_0);
                s1_1 = simd.mul_add_e_f64s(sum1, *x_1, s1_1);
                s1_2 = simd.mul_add_e_f64s(sum2, *x_2, s1_2);
                s1_3 = simd.mul_add_e_f64s(sum3, *x_3, s1_3);
                s2_0 = simd.mul_add_e_f64s(sum0, *y_0, s2_0);
                s2_1 = simd.mul_add_e_f64s(sum1, *y_1, s2_1);
                s2_2 = simd.mul_add_e_f64s(sum2, *y_2, s2_2);
                s2_3 = simd.mul_add_e_f64s(sum3, *y_3, s2_3);
            },
        );

        izip!(
            p1_simd_tail,
            n1_simd_tail,
            p2_simd_tail,
            x_simd_tail,
            y_simd_tail
        )
        .for_each(|(p1, n1, p2, x, y)| {
            let sum = simd.sub_f64s(simd.add_f64s(*p1, *p2), *n1);
            s1_0 = simd.mul_add_e_f64s(sum, *x, s1_0);
            s2_0 = simd.mul_add_e_f64s(sum, *y, s2_0);
        });

        let mut out = (
            simd.reduce_sum_f64s(
                simd.add_f64s(simd.add_f64s(s1_0, s1_1), simd.add_f64s(s1_2, s1_3)),
            ),
            simd.reduce_sum_f64s(
                simd.add_f64s(simd.add_f64s(s2_0, s2_1), simd.add_f64s(s2_2, s2_3)),
            ),
        );

        izip!(
            p1_tail.iter(),
            n1_tail.iter(),
            p2_tail.iter(),
            x_tail.iter(),
            y_tail.iter()
        )
        .for_each(|(p1, n1, p2, x, y)| {
            let sum = *p1 - *n1 + *p2;
            out.0 += sum * *x;
            out.1 += sum * *y;
        });

        out
    }
}

#[inline(never)]
pub fn scalar_prods3(
    arch: Arch,
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

    arch.dispatch(ScalarProds3 {
        positive1,
        negative1,
        positive2,
        x,
        y,
    })
}

struct VectorDot<'a> {
    x: &'a [f64],
    y: &'a [f64],
}

impl<'a> WithSimd for VectorDot<'a> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let a = self.x;
        let b = self.y;

        assert!(a.len() == b.len());

        let (x, x_tail) = S::as_simd_f64s(a);
        let (y, y_tail) = S::as_simd_f64s(b);

        let mut out0 = simd.splat_f64s(0f64);
        let mut out1 = simd.splat_f64s(0f64);
        let mut out2 = simd.splat_f64s(0f64);
        let mut out3 = simd.splat_f64s(0f64);

        let (x, x_simd_tail) = pulp::as_arrays::<4, _>(x);
        let (y, y_simd_tail) = pulp::as_arrays::<4, _>(y);

        izip!(x, y).for_each(|([x0, x1, x2, x3], [y0, y1, y2, y3])| {
            out0 = simd.mul_add_e_f64s(*x0, *y0, out0);
            out1 = simd.mul_add_e_f64s(*x1, *y1, out1);
            out2 = simd.mul_add_e_f64s(*x2, *y2, out2);
            out3 = simd.mul_add_e_f64s(*x3, *y3, out3);
        });

        izip!(x_simd_tail, y_simd_tail).for_each(|(&x, &y)| {
            out0 = simd.mul_add_e_f64s(x, y, out0);
        });

        out0 = simd.add_f64s(out0, out1);
        out1 = simd.add_f64s(out2, out3);
        out0 = simd.add_f64s(out0, out1);
        let mut result = simd.reduce_sum_f64s(out0);

        x_tail.iter().zip(y_tail).for_each(|(&x, &y)| {
            result += x * y;
        });
        result
    }
}

pub fn vector_dot(arch: Arch, a: &[f64], b: &[f64]) -> f64 {
    arch.dispatch(VectorDot { x: a, y: b })
}

struct Axpy<'a> {
    x: &'a [f64],
    y: &'a mut [f64],
    a: f64,
}

impl<'a> WithSimd for Axpy<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let x = self.x;
        let y = self.y;
        let a = self.a;

        let (x_out, x_tail) = S::as_simd_f64s(x);
        let (y_out, y_tail) = S::as_mut_simd_f64s(y);

        let a_splat = simd.splat_f64s(a);

        let (x_arrays, x_simd_tail) = pulp::as_arrays::<4, _>(x_out);
        let (y_arrays, y_simd_tail) = pulp::as_arrays_mut::<4, _>(y_out);

        izip!(x_arrays, y_arrays).for_each(|([x0, x1, x2, x3], [y0, y1, y2, y3])| {
            *y0 = simd.mul_add_e_f64s(a_splat, *x0, *y0);
            *y1 = simd.mul_add_e_f64s(a_splat, *x1, *y1);
            *y2 = simd.mul_add_e_f64s(a_splat, *x2, *y2);
            *y3 = simd.mul_add_e_f64s(a_splat, *x3, *y3);
        });

        izip!(x_simd_tail.iter(), y_simd_tail.iter_mut()).for_each(|(&x, y)| {
            *y = simd.mul_add_e_f64s(a_splat, x, *y);
        });

        izip!(x_tail.iter(), y_tail.iter_mut()).for_each(|(&x, y)| {
            *y = a.mul_add(x, *y);
        });
    }
}
pub fn axpy(arch: Arch, x: &[f64], y: &mut [f64], a: f64) {
    let n = x.len();
    assert!(y.len() == n);

    arch.dispatch(Axpy { x, y, a });
}

struct AxpyOut<'a> {
    x: &'a [f64],
    y: &'a [f64],
    out: &'a mut [f64],
    a: f64,
}

impl<'a> WithSimd for AxpyOut<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let x = self.x;
        let y = self.y;
        let out = self.out;
        let a = self.a;

        let (x_out, x_tail) = S::as_simd_f64s(x);
        let (y_out, y_tail) = S::as_simd_f64s(y);
        let (out_head, out_tail) = S::as_mut_simd_f64s(out);

        let a_splat = simd.splat_f64s(a);

        let (out_arrays, out_simd_tail) = pulp::as_arrays_mut::<4, _>(out_head);
        let (x_arrays, x_simd_tail) = pulp::as_arrays::<4, _>(x_out);
        let (y_arrays, y_simd_tail) = pulp::as_arrays::<4, _>(y_out);

        izip!(out_arrays, x_arrays, y_arrays).for_each(
            |([out0, out1, out2, out3], [x0, x1, x2, x3], [y0, y1, y2, y3])| {
                *out0 = simd.mul_add_e_f64s(a_splat, *x0, *y0);
                *out1 = simd.mul_add_e_f64s(a_splat, *x1, *y1);
                *out2 = simd.mul_add_e_f64s(a_splat, *x2, *y2);
                *out3 = simd.mul_add_e_f64s(a_splat, *x3, *y3);
            },
        );

        izip!(
            out_simd_tail.iter_mut(),
            x_simd_tail.iter(),
            y_simd_tail.iter()
        )
        .for_each(|(out, &x, &y)| {
            *out = simd.mul_add_e_f64s(a_splat, x, y);
        });

        izip!(x_tail.iter(), y_tail.iter(), out_tail.iter_mut()).for_each(|(&x, &y, out)| {
            *out = a.mul_add(x, y);
        });
    }
}

pub fn axpy_out(arch: Arch, x: &[f64], y: &[f64], a: f64, out: &mut [f64]) {
    let n = x.len();
    assert!(y.len() == n);
    assert!(out.len() == n);

    arch.dispatch(AxpyOut { x, y, out, a });
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
            let arch = pulp::Arch::default();
            let orig = y.clone();
            let mut y = y.clone();
            axpy(arch, &x[..], &mut y[..], a);
            for ((&x, y), out) in x.iter().zip(orig).zip(y) {
                assert_approx_eq(out, a.mul_add(x, y));
            }
        }

        #[test]
        fn test_scalar_prods2((x1, x2, y1, y2) in array4(10)) {
            let arch = pulp::Arch::default();
            let (p1, p2) = scalar_prods2(arch, &x1[..], &x2[..], &y1[..], &y2[..]);
            let x1 = ndarray::Array1::from_vec(x1);
            let x2 = ndarray::Array1::from_vec(x2);
            let y1 = ndarray::Array1::from_vec(y1);
            let y2 = ndarray::Array1::from_vec(y2);
            assert_approx_eq(p1, (&x1 + &x2).dot(&y1));
            assert_approx_eq(p2, (&x1 + &x2).dot(&y2));
        }

        #[test]
        fn test_scalar_prods3((x1, x2, x3, y1, y2) in array5(10)) {
            let arch = Arch::default();
            let (p1, p2) = scalar_prods3(arch, &x1[..], &x2[..], &x3[..], &y1[..], &y2[..]);
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
            let arch = Arch::default();
            let mut out = out.clone();
            axpy_out(arch, &x[..], &y[..], a, &mut out[..]);
            let x = ndarray::Array1::from_vec(x);
            let mut y = ndarray::Array1::from_vec(y);
            y.scaled_add(a, &x);
            for (&out1, out2) in out.iter().zip(y) {
                assert_approx_eq(out1, out2);
            }
        }

        #[test]
        fn test_multiplty((x, y, out) in array3(10)) {
            let arch = pulp::Arch::default();
            let mut out = out.clone();
            multiply(arch, &x[..], &y[..], &mut out[..]);
            let x = ndarray::Array1::from_vec(x);
            let y = ndarray::Array1::from_vec(y);
            for (&out1, out2) in out.iter().zip(&x * &y) {
                assert_approx_eq(out1, out2);
            }
        }

        #[test]
        fn test_vector_dot((x, y) in array2(10)) {
            let arch = pulp::Arch::default();
            let actual = vector_dot(arch, &x[..], &y[..]);
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
