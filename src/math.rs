#[inline]
pub(crate) fn logaddexp(a: f64, b: f64) -> f64 {
    let diff = a - b;
    if diff == 0. {
        a + 2f64.ln()
    } else if diff > 0. {
        a + (-diff).exp().ln_1p()
    } else if diff < 0. {
        b + diff.exp().ln_1p()
    } else {
        // diff is NAN
        diff
    }
}

#[inline]
#[allow(clippy::many_single_char_names)]
pub(crate) fn scalar_prods_of_diff(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> (f64, f64) {
    let n = a.len();
    assert!(b.len() == n);
    assert!(c.len() == n);
    assert!(d.len() == n);
    let mut sum_c: f64 = 0.;
    let mut sum_d: f64 = 0.;

    for i in 0..n {
        sum_c += (a[i] - b[i]) * c[i];
        sum_d += (a[i] - b[i]) * d[i];
    }
    (sum_c, sum_d)
}

#[inline]
pub(crate) fn norm(a: &[f64]) -> f64 {
    let mut result = 0.;
    for val in a.iter() {
        result += *val;
    }
    result
}

#[inline]
pub(crate) fn axpy(x: &[f64], y: &mut [f64], a: f64) {
    let n = x.len();
    assert!(y.len() == n);
    for i in 0..n {
        y[i] += a * x[i];
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
