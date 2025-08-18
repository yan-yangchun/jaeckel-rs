use jaeckel::{
    convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side,
    minimum_rational_cubic_control_parameter,
    rational_cubic_control_parameter_to_fit_second_derivative_at_left_side,
    rational_cubic_interpolation,
};

#[test]
fn interpolation_at_endpoints() {
    let (x_l, x_r) = (0.0_f64, 1.0_f64);
    let (y_l, y_r) = (2.0_f64, 3.0_f64);
    let (d_l, d_r) = (0.5_f64, 0.5_f64);
    let r = 1.0_f64;
    let tol = 1e-14_f64;

    let y_at_left = rational_cubic_interpolation(x_l, x_l, x_r, y_l, y_r, d_l, d_r, r);
    assert!((y_at_left - y_l).abs() < tol);

    let y_at_right = rational_cubic_interpolation(x_r, x_l, x_r, y_l, y_r, d_l, d_r, r);
    assert!((y_at_right - y_r).abs() < tol);
}

#[test]
fn linear_interpolation() {
    let (x_l, x_r) = (0.0_f64, 2.0_f64);
    let (y_l, y_r) = (1.0_f64, 3.0_f64);
    let slope = (y_r - y_l) / (x_r - x_l);
    let (d_l, d_r) = (slope, slope);
    let r = 1.0_f64;
    let x_mid = 1.0_f64;
    let tol = 1e-14_f64;

    let y_mid = rational_cubic_interpolation(x_mid, x_l, x_r, y_l, y_r, d_l, d_r, r);
    let expected = y_l + slope * (x_mid - x_l);
    assert!((y_mid - expected).abs() < tol);
}

#[test]
fn monotonicity_preservation() {
    let (x_l, x_r) = (0.0_f64, 1.0_f64);
    let (y_l, y_r) = (0.0_f64, 1.0_f64);
    let (d_l, d_r) = (0.5_f64, 0.5_f64);
    let s = (y_r - y_l) / (x_r - x_l);
    let r_min = minimum_rational_cubic_control_parameter(d_l, d_r, s, true);
    let r = r_min + 0.1;

    let mut prev_y = y_l;
    let mut t = 0.1_f64;
    while t <= 1.0 + 1e-12 {
        let x = x_l + t * (x_r - x_l);
        let y = rational_cubic_interpolation(x, x_l, x_r, y_l, y_r, d_l, d_r, r);
        assert!(
            y >= prev_y - 1e-12,
            "monotonicity violated at t={}, y={} prev={}",
            t,
            y,
            prev_y
        );
        prev_y = y;
        t += 0.1;
    }
}

#[test]
fn second_derivative_control() {
    let (x_l, x_r) = (0.0_f64, 1.0_f64);
    let (y_l, y_r) = (0.0_f64, 1.0_f64);
    let (d_l, d_r) = (1.0_f64, 1.0_f64);
    let second_derivative_l = 2.0_f64;
    let r = rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
        x_l,
        x_r,
        y_l,
        y_r,
        d_l,
        d_r,
        second_derivative_l,
    );
    assert!(r > 0.0);
}

#[test]
fn convex_interpolation() {
    let (x_l, x_r) = (0.0_f64, 1.0_f64);
    let (y_l, y_r) = (0.0_f64, 1.0_f64);
    let (d_l, d_r) = (0.0_f64, 2.0_f64);
    let second_derivative_l = 2.0_f64;
    let r = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
        x_l,
        x_r,
        y_l,
        y_r,
        d_l,
        d_r,
        second_derivative_l,
        false,
    );

    let x1 = 0.25_f64;
    let x2 = 0.75_f64;
    let y1 = rational_cubic_interpolation(x1, x_l, x_r, y_l, y_r, d_l, d_r, r);
    let y2 = rational_cubic_interpolation(x2, x_l, x_r, y_l, y_r, d_l, d_r, r);
    let x_mid = 0.5_f64;
    let y_mid = rational_cubic_interpolation(x_mid, x_l, x_r, y_l, y_r, d_l, d_r, r);
    let y_linear = 0.5 * (y1 + y2);
    assert!(y_mid <= y_linear + 1e-14);
}

#[test]
fn symmetry_test() {
    let (x_l, x_r) = (-1.0_f64, 1.0_f64);
    let (y_l, y_r) = (1.0_f64, 1.0_f64);
    let (d_l, d_r) = (-0.5_f64, 0.5_f64);
    let r = 2.0_f64;
    let tol = 1e-14_f64;

    let x1 = -0.5_f64;
    let x2 = 0.5_f64;
    let y1 = rational_cubic_interpolation(x1, x_l, x_r, y_l, y_r, d_l, d_r, r);
    let y2 = rational_cubic_interpolation(x2, x_l, x_r, y_l, y_r, d_l, d_r, r);
    assert!((y1 - y2).abs() < tol);
}

#[test]
fn derivative_consistency() {
    let (x_l, x_r) = (0.0_f64, 1.0_f64);
    let (y_l, y_r) = (0.0_f64, 1.0_f64);
    let (d_l, d_r) = (0.5_f64, 1.5_f64);
    let r = 2.0_f64;
    let h = 1e-8_f64;

    let y_at_l_plus_h = rational_cubic_interpolation(x_l + h, x_l, x_r, y_l, y_r, d_l, d_r, r);
    let d_l_approx = (y_at_l_plus_h - y_l) / h;
    assert!((d_l_approx - d_l).abs() < 1e-6);

    let y_at_r_minus_h = rational_cubic_interpolation(x_r - h, x_l, x_r, y_l, y_r, d_l, d_r, r);
    let d_r_approx = (y_r - y_at_r_minus_h) / h;
    assert!((d_r_approx - d_r).abs() < 1e-6);
}
