use jaeckel::{erf_cody, erfc_cody, erfcx_cody, inverse_norm_cdf, norm_cdf, norm_pdf};

#[test]
fn cumulative_distribution_function() {
    let tol = 1e-14_f64;
    assert!((norm_cdf(0.0) - 0.5).abs() < tol);
    assert!((norm_cdf(1.0) - 0.8413447460685429).abs() < tol);
    assert!((norm_cdf(-1.0) - 0.1586552539314571).abs() < tol);
    assert!((norm_cdf(2.0) - 0.9772498680518208).abs() < tol);
    assert!((norm_cdf(-2.0) - 0.0227501319481792).abs() < tol);
}

#[test]
fn inverse_cumulative_distribution() {
    let tol = 1e-14_f64;
    assert!((inverse_norm_cdf(0.5) - 0.0).abs() < tol);
    assert!((inverse_norm_cdf(0.8413447460685429) - 1.0).abs() < tol);
    assert!((inverse_norm_cdf(0.1586552539314571) + 1.0).abs() < tol);
    assert!((inverse_norm_cdf(0.9772498680518208) - 2.0).abs() < tol);
    assert!((inverse_norm_cdf(0.0227501319481792) + 2.0).abs() < tol);
}

#[test]
fn cdf_inverse_cdf_roundtrip() {
    let tol = 1e-14_f64;
    let xs = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    for &x in &xs {
        let p = norm_cdf(x);
        let xr = inverse_norm_cdf(p);
        assert!(
            (xr - x).abs() < tol,
            "x={} xr={} diff={}",
            x,
            xr,
            (xr - x).abs()
        );
    }
}

#[test]
fn inverse_cdf_cdf_roundtrip() {
    let tol = 1e-14_f64;
    let ps = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
    for &p in &ps {
        let x = inverse_norm_cdf(p);
        let pr = norm_cdf(x);
        assert!(
            (pr - p).abs() < tol,
            "p={} pr={} diff={}",
            p,
            pr,
            (pr - p).abs()
        );
    }
}

#[test]
fn extreme_values() {
    assert!((norm_cdf(-10.0) - 0.0).abs() < 1e-10);
    assert!((norm_cdf(10.0) - 1.0).abs() < 1e-10);
    assert!(inverse_norm_cdf(1e-10) < -6.0);
    assert!(inverse_norm_cdf(1.0 - 1e-10) > 6.0);
}

#[test]
fn probability_density_function() {
    let tol = 1e-14_f64;
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    assert!((norm_pdf(0.0) - inv_sqrt_2pi).abs() < tol);
    assert!((norm_pdf(1.0) - (-0.5_f64).exp() * inv_sqrt_2pi).abs() < tol);
    assert!((norm_pdf(-1.0) - (-0.5_f64).exp() * inv_sqrt_2pi).abs() < tol);
}

#[test]
fn error_function_checks() {
    let tol = 1e-14_f64;
    assert!(erf_cody(0.0).abs() < tol);
    assert!((erf_cody(1.0) - 0.8427007929497149).abs() < tol);
    assert!((erf_cody(-1.0) + 0.8427007929497149).abs() < tol);

    // Relationship: Phi(x) = 0.5 * (1 + erf(x/sqrt(2)))
    let x = 1.5_f64;
    let expected_cdf = 0.5 * (1.0 + erf_cody(x / 2.0_f64.sqrt()));
    assert!((norm_cdf(x) - expected_cdf).abs() < tol);
}

#[test]
fn complementary_error_function_checks() {
    let tol = 1e-14_f64;
    for &x in &[0.0, 0.5, 1.0, 1.5, 2.0] {
        let erf_val = erf_cody(x);
        let erfc_val = erfc_cody(x);
        assert!((erf_val + erfc_val - 1.0).abs() < tol);
    }
}

#[test]
fn scaled_complementary_error_function() {
    let tol = 1e-14_f64;
    for &x in &[0.0, 0.5, 1.0, 1.5, 2.0] {
        let erfcx_val = erfcx_cody(x);
        let expected = (x * x).exp() * erfc_cody(x);
        // Relative tolerance scaled by expected value
        assert!((erfcx_val - expected).abs() < tol * expected.abs().max(1.0));
    }
}
