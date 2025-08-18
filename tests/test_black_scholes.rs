use jaeckel::{
    black, erf_cody, erfcx_cody, implied_black_volatility, normalised_black, normalised_vega, vega,
    volga,
};
use std::f64::consts::{PI, SQRT_2};

// Test-only helper APIs cloned from C++ definitions
fn complementary_normalised_black_test(x: f64, s: f64) -> f64 {
    let h = x / s;
    let t = 0.5 * s;
    0.5 * (erfcx_cody((t + h) / SQRT_2) + erfcx_cody((t - h) / SQRT_2))
        * (-(0.5) * (t * t + h * h)).exp()
}

fn implied_volatility_attainable_accuracy_test(x: f64, s: f64, theta: f64) -> f64 {
    if x == 0.0 {
        return f64::EPSILON
            * (1.0
                + (if s <= f64::EPSILON {
                    1.0
                } else {
                    (erf_cody((0.5 / SQRT_2) * s) * (2.0 * PI).sqrt() * (0.125 * s * s).exp()) / s
                })
                .abs());
    }
    let thetax = if theta < 0.0 { -x } else { x };
    if s <= 0.0 {
        return if thetax > 0.0 { 1.0 } else { f64::EPSILON };
    }
    let b = normalised_black(x, s, theta);
    let v = normalised_vega(x, s);
    if b * v >= f64::MIN_POSITIVE {
        f64::EPSILON * (1.0 + (b / (s * v)).abs())
    } else {
        1.0
    }
}

#[test]
fn test_black_scholes_implied_volatility_round_trip() {
    let test_cases = [
        (100.0, 100.0, 0.2, 1.0, 1.0, "ATM standard"),
        (50.0, 50.0, 0.3, 0.5, 1.0, "ATM low price"),
        (1000.0, 1000.0, 0.15, 2.0, 1.0, "ATM high price"),
        (100.0, 110.0, 0.2, 1.0, 1.0, "10% OTM call"),
        (100.0, 150.0, 0.2, 1.0, 1.0, "50% OTM call"),
        (100.0, 200.0, 0.2, 1.0, 1.0, "100% OTM call (deep)"),
        (100.0, 500.0, 0.3, 1.0, 1.0, "400% OTM call (very deep)"),
        (100.0, 90.0, 0.2, 1.0, 1.0, "10% ITM call"),
        (100.0, 50.0, 0.2, 1.0, 1.0, "50% ITM call"),
        (100.0, 20.0, 0.2, 1.0, 1.0, "80% ITM call (deep)"),
        (100.0, 5.0, 0.3, 1.0, 1.0, "95% ITM call (very deep)"),
        (100.0, 100.0, 0.2, 1.0, -1.0, "ATM put"),
        (100.0, 110.0, 0.2, 1.0, -1.0, "ITM put"),
        (100.0, 90.0, 0.2, 1.0, -1.0, "OTM put"),
        (100.0, 100.0, 0.00001, 1.0, 1.0, "Very low vol"),
        (100.0, 100.0, 0.001, 1.0, 1.0, "Low vol"),
        (100.0, 100.0, 5.0, 1.0, 1.0, "High vol"),
        (100.0, 100.0, 10.0, 1.0, 1.0, "Very high vol"),
        (100.0, 100.0, 0.2, 0.001, 1.0, "Very short maturity"),
        (100.0, 100.0, 0.2, 0.01, 1.0, "Short maturity"),
        (100.0, 100.0, 0.2, 5.0, 1.0, "Long maturity"),
        (100.0, 100.0, 0.2, 10.0, 1.0, "Very long maturity"),
        (100.0, 0.1, 0.2, 1.0, 1.0, "Near-zero strike"),
        (100.0, 10000.0, 0.2, 1.0, 1.0, "Extremely high strike"),
        (0.01, 0.01, 0.2, 1.0, 1.0, "Very low forward and strike"),
        (
            10000.0,
            10000.0,
            0.2,
            1.0,
            1.0,
            "Very high forward and strike",
        ),
        (80.0, 120.0, 0.35, 0.25, 1.0, "OTM short maturity high vol"),
        (150.0, 100.0, 0.15, 3.0, 1.0, "ITM long maturity low vol"),
    ];

    for (f, k, sigma, t, theta, _description) in test_cases {
        let price = black(f, k, sigma, t, theta);
        let implied_vol = implied_black_volatility(price, f, k, t, theta);

        if implied_vol == 0.0 {
            let intrinsic = if theta > 0.0 {
                f64::max(f - k, 0.0)
            } else {
                f64::max(k - f, 0.0)
            };
            if (price - intrinsic).abs() < 1e-12 {
                continue;
            }
        }

        let relative_error = (implied_vol - sigma).abs() / sigma;
        let x = (f / k).ln();
        let s = sigma * t.sqrt();
        let attainable = implied_volatility_attainable_accuracy_test(x, s, theta);
        let tol = (1e-12f64).max(attainable);
        assert!(
            relative_error <= tol,
            "rel {:.3e} > tol {:.3e}",
            relative_error,
            tol
        );
    }
}

#[test]
fn test_call_option_price() {
    let f: f64 = 100.0;
    let k: f64 = 100.0;
    let sigma: f64 = 0.2;
    let t: f64 = 1.0;
    let call_price = black(f, k, sigma, t, 1.0);
    assert!(call_price > 0.0);
    let expected = 7.96557;
    assert!((call_price - expected).abs() < 1e-5);
}

#[test]
fn test_put_option_price() {
    let f: f64 = 100.0;
    let k: f64 = 100.0;
    let sigma: f64 = 0.2;
    let t: f64 = 1.0;
    let call_price = black(f, k, sigma, t, 1.0);
    let put_price = black(f, k, sigma, t, -1.0);
    assert!((call_price - put_price).abs() < 1e-10);
}

#[test]
fn test_zero_volatility_intrinsic() {
    let f: f64 = 100.0;
    let k: f64 = 100.0;
    let t: f64 = 1.0;
    let call_intrinsic = f64::max(f - k, 0.0);
    let put_intrinsic = f64::max(k - f, 0.0);
    assert!((black(f, k, 0.0, t, 1.0) - call_intrinsic).abs() < 1e-12);
    assert!((black(f, k, 0.0, t, -1.0) - put_intrinsic).abs() < 1e-12);
}

#[test]
fn test_zero_time_intrinsic() {
    let f: f64 = 100.0;
    let k: f64 = 100.0;
    let sigma: f64 = 0.2;
    let call_intrinsic = f64::max(f - k, 0.0);
    let put_intrinsic = f64::max(k - f, 0.0);
    assert!((black(f, k, sigma, 0.0, 1.0) - call_intrinsic).abs() < 1e-12);
    assert!((black(f, k, sigma, 0.0, -1.0) - put_intrinsic).abs() < 1e-12);
}

#[test]
fn test_normalised_black_consistency() {
    let f: f64 = 100.0;
    let k: f64 = 100.0;
    let sigma: f64 = 0.2;
    let t: f64 = 1.0;
    let x = (f / k).ln();
    let s = sigma * t.sqrt();
    let call = black(f, k, sigma, t, 1.0);
    let nb = normalised_black(x, s, 1.0);
    assert!((call - nb * (f * k).sqrt()).abs() < 1e-10);
}

#[test]
fn test_implied_volatility_recovery_precise() {
    let f: f64 = 100.0;
    let k: f64 = 100.0;
    let sigma: f64 = 0.2;
    let t: f64 = 1.0;
    let call = black(f, k, sigma, t, 1.0);
    let iv = implied_black_volatility(call, f, k, t, 1.0);
    assert!(((iv - sigma) / sigma).abs() < 1e-12);
}

#[test]
fn test_vega_positivity_and_shape() {
    let f: f64 = 100.0;
    let k: f64 = 100.0;
    let sigma: f64 = 0.2;
    let t: f64 = 1.0;
    let v_atm = vega(f, k, sigma, t);
    assert!(v_atm > 0.0);
    let v_itm = vega(f, k * 0.9, sigma, t);
    let v_otm = vega(f, k * 1.1, sigma, t);
    assert!(v_atm > v_itm);
    assert!(v_atm > v_otm);
}

#[test]
fn test_volga_reasonable() {
    let f: f64 = 100.0;
    let k: f64 = 100.0;
    let sigma: f64 = 0.2;
    let t: f64 = 1.0;
    let vg = vega(f, k, sigma, t);
    let volg = volga(f, k, sigma, t);
    assert!(volg > 0.0);
    assert!((volg - vg).abs() <= 0.1 * vg);
}

#[test]
fn test_extreme_values() {
    let f: f64 = 100.0;
    let sigma: f64 = 0.2;
    let t: f64 = 1.0;
    let deep_itm_call = black(f, 0.01, sigma, t, 1.0);
    let deep_otm_call = black(f, 1000.0, sigma, t, 1.0);
    assert!((deep_itm_call - f).abs() < 0.1);
    assert!(deep_otm_call < 0.01);
}

#[test]
fn test_complementary_normalised_black_basic() {
    let f: f64 = 100.0;
    let k: f64 = 100.0;
    let sigma: f64 = 0.2;
    let t: f64 = 1.0;
    let x = (f / k).ln();
    let s = sigma * t.sqrt();

    let comp = complementary_normalised_black_test(x, s);
    assert!(comp > 0.0);
    let b_max = (x.abs() * 0.5).exp();
    assert!(comp < b_max);
}
