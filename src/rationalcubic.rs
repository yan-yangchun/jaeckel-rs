use std::f64;

const MINIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE: f64 = -0.999999985098838806152; // -(1 - sqrt(DBL_EPSILON))
pub const MAXIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE: f64 = 4.0564819207303340e+31; // 2 / (DBL_EPSILON * DBL_EPSILON)

#[inline]
fn is_zero(x: f64) -> bool {
    x.abs() < f64::MIN_POSITIVE
}

/// Rational cubic interpolation
///
/// Interpolates a value at point x between two points (x_l, y_l) and (x_r, y_r)
/// with derivatives d_l and d_r at the endpoints, using control parameter r.
pub fn rational_cubic_interpolation(
    x: f64,
    x_l: f64,
    x_r: f64,
    y_l: f64,
    y_r: f64,
    d_l: f64,
    d_r: f64,
    r: f64,
) -> f64 {
    let h = x_r - x_l;
    if h.abs() <= 0.0 {
        return 0.5 * (y_l + y_r);
    }

    // r should be greater than -1. We do not use assert(r > -1) here in order to allow
    // values such as NaN to be propagated as they should.
    let t = (x - x_l) / h;

    if !(r >= MAXIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE) {
        let omt = 1.0 - t;
        let t2 = t * t;
        let omt2 = omt * omt;

        // Formula (2.4) divided by formula (2.5)
        (y_r * t2 * t
            + (r * y_r - h * d_r) * t2 * omt
            + (r * y_l + h * d_l) * t * omt2
            + y_l * omt2 * omt)
            / (1.0 + (r - 3.0) * t * omt)
    } else {
        // Linear interpolation without over-or underflow.
        y_r * t + y_l * (1.0 - t)
    }
}

/// Calculate control parameter to fit second derivative at left side
pub fn rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
    x_l: f64,
    x_r: f64,
    y_l: f64,
    y_r: f64,
    d_l: f64,
    d_r: f64,
    second_derivative_l: f64,
) -> f64 {
    let h = x_r - x_l;
    let numerator = 0.5 * h * second_derivative_l + (d_r - d_l);

    let denominator = (y_r - y_l) / h - d_l;
    if is_zero(denominator) {
        if numerator > 0.0 {
            MAXIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE
        } else {
            MINIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE
        }
    } else {
        numerator / denominator
    }
}

/// Calculate control parameter to fit second derivative at right side
pub fn rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
    x_l: f64,
    x_r: f64,
    y_l: f64,
    y_r: f64,
    d_l: f64,
    d_r: f64,
    second_derivative_r: f64,
) -> f64 {
    let h = x_r - x_l;
    let numerator = 0.5 * h * second_derivative_r + (d_r - d_l);

    let denominator = d_r - (y_r - y_l) / h;
    if is_zero(denominator) {
        if numerator > 0.0 {
            MAXIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE
        } else {
            MINIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE
        }
    } else {
        numerator / denominator
    }
}

/// Calculate minimum rational cubic control parameter
pub fn minimum_rational_cubic_control_parameter(
    d_l: f64,
    d_r: f64,
    s: f64,
    prefer_shape_preservation_over_smoothness: bool,
) -> f64 {
    let monotonic = d_l * s >= 0.0 && d_r * s >= 0.0;
    let convex = d_l <= s && s <= d_r;
    let concave = d_l >= s && s >= d_r;

    if !monotonic && !convex && !concave {
        // If 3==r_non_shape_preserving_target, this means revert to standard cubic.
        return MINIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE;
    }

    let d_r_m_d_l = d_r - d_l;
    let d_r_m_s = d_r - s;
    let s_m_d_l = s - d_l;

    let mut r1 = -f64::MAX;
    let mut r2 = r1;

    // If monotonicity on this interval is possible, set r1 to satisfy the monotonicity condition (3.8).
    if monotonic {
        if !is_zero(s) {
            // (3.8), avoiding division by zero.
            r1 = (d_r + d_l) / s; // (3.8)
        } else if prefer_shape_preservation_over_smoothness {
            // If division by zero would occur, and shape preservation is preferred,
            // set value to enforce linear interpolation.
            r1 = MAXIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE;
        }
    }

    if convex || concave {
        if !(is_zero(s_m_d_l) || is_zero(d_r_m_s)) {
            // (3.18), avoiding division by zero.
            r2 = f64::max((d_r_m_d_l / d_r_m_s).abs(), (d_r_m_d_l / s_m_d_l).abs());
        } else if prefer_shape_preservation_over_smoothness {
            r2 = MAXIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE;
        }
    } else if monotonic && prefer_shape_preservation_over_smoothness {
        // This enforces linear interpolation along segments that are inconsistent with the slopes
        // on the boundaries, e.g., a perfectly horizontal segment that has negative slopes on either edge.
        r2 = MAXIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE;
    }

    f64::max(
        MINIMUM_RATIONAL_CUBIC_CONTROL_PARAMETER_VALUE,
        f64::max(r1, r2),
    )
}

/// Calculate convex rational cubic control parameter to fit second derivative at left side
pub fn convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
    x_l: f64,
    x_r: f64,
    y_l: f64,
    y_r: f64,
    d_l: f64,
    d_r: f64,
    second_derivative_l: f64,
    prefer_shape_preservation_over_smoothness: bool,
) -> f64 {
    let r = rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
        x_l,
        x_r,
        y_l,
        y_r,
        d_l,
        d_r,
        second_derivative_l,
    );
    let r_min = minimum_rational_cubic_control_parameter(
        d_l,
        d_r,
        (y_r - y_l) / (x_r - x_l),
        prefer_shape_preservation_over_smoothness,
    );
    f64::max(r, r_min)
}

/// Calculate convex rational cubic control parameter to fit second derivative at right side
pub fn convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
    x_l: f64,
    x_r: f64,
    y_l: f64,
    y_r: f64,
    d_l: f64,
    d_r: f64,
    second_derivative_r: f64,
    prefer_shape_preservation_over_smoothness: bool,
) -> f64 {
    let r = rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
        x_l,
        x_r,
        y_l,
        y_r,
        d_l,
        d_r,
        second_derivative_r,
    );
    let r_min = minimum_rational_cubic_control_parameter(
        d_l,
        d_r,
        (y_r - y_l) / (x_r - x_l),
        prefer_shape_preservation_over_smoothness,
    );
    f64::max(r, r_min)
}
