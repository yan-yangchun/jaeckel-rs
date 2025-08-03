use std::f64;

#[inline]
fn d_int(x: f64) -> f64 {
    if x > 0.0 {
        x.floor()
    } else {
        -(-x).floor()
    }
}

#[inline]
fn smoothened_exponential_of_negative_square(y: f64) -> f64 {
    let y_tilde = d_int(y * 16.0) / 16.0;
    (-y_tilde * y_tilde).exp() * (-(y - y_tilde) * (y + y_tilde)).exp()
}

#[inline]
fn smoothened_exponential_of_positive_square(x: f64) -> f64 {
    let x_tilde = d_int(x * 16.0) / 16.0;
    (x_tilde * x_tilde).exp() * ((x - x_tilde) * (x + x_tilde)).exp()
}

// Coefficients for approximation to erf in first interval
const a__: [f64; 5] = [
    3.1611237438705656,
    113.864154151050156,
    377.485237685302021,
    3209.37758913846947,
    0.185777706184603153,
];

const b__: [f64; 4] = [
    23.6012909523441209,
    244.024637934444173,
    1282.61652607737228,
    2844.23683343917062,
];

// Coefficients for approximation to erfc in second interval
const c__: [f64; 9] = [
    0.564188496988670089,
    8.88314979438837594,
    66.1191906371416295,
    298.635138197400131,
    881.95222124176909,
    1712.04761263407058,
    2051.07837782607147,
    1230.33935479799725,
    2.15311535474403846e-8,
];

const d__: [f64; 8] = [
    15.7449261107098347,
    117.693950891312499,
    537.181101862009858,
    1621.38957456669019,
    3290.79923573345963,
    4362.61909014324716,
    3439.36767414372164,
    1230.33935480374942,
];

// Coefficients for approximation to erfc in third interval
const p__: [f64; 6] = [
    0.305326634961232344,
    0.360344899949804439,
    0.125781726111229246,
    0.0160837851487422766,
    6.58749161529837803e-4,
    0.0163153871373020978,
];

const q__: [f64; 5] = [
    2.56852019228982242,
    1.87295284992346047,
    0.527905102951428412,
    0.0605183413124413191,
    0.00233520497626869185,
];

const ONE_OVER_SQRT_PI: f64 = 0.56418958354775628695; // 1/√π
const THRESHOLD: f64 = 0.46875;
const XNEG: f64 = -26.6287357137514; // The original value was -26.628
const XBIG: f64 = 26.543;

#[inline]
fn AB(z: f64) -> f64 {
    ((((a__[4] * z + a__[0]) * z + a__[1]) * z + a__[2]) * z + a__[3])
        / ((((z + b__[0]) * z + b__[1]) * z + b__[2]) * z + b__[3])
}

#[inline]
fn CD(y: f64) -> f64 {
    ((((((((c__[8] * y + c__[0]) * y + c__[1]) * y + c__[2]) * y + c__[3]) * y + c__[4]) * y
        + c__[5])
        * y
        + c__[6])
        * y
        + c__[7])
        / ((((((((y + d__[0]) * y + d__[1]) * y + d__[2]) * y + d__[3]) * y + d__[4]) * y
            + d__[5])
            * y
            + d__[6])
            * y
            + d__[7])
}

#[inline]
fn PQ(z: f64) -> f64 {
    z * (((((p__[5] * z + p__[0]) * z + p__[1]) * z + p__[2]) * z + p__[3]) * z + p__[4])
        / (((((z + q__[0]) * z + q__[1]) * z + q__[2]) * z + q__[3]) * z + q__[4])
}

/// Cody's error function implementation
/// erf(x) = -erf(-x) = 1 - erfc(x) = erfc(-x) - 1
pub fn erf_cody(x: f64) -> f64 {
    let y = x.abs();
    if y <= THRESHOLD {
        // |x| <= 0.46875
        return x * AB(y * y);
    }
    // Compute erfc(|x|)
    let erfc_abs_x = if y >= XBIG {
        0.0 // when |x| ≥ 26.543
    } else if y <= 4.0 {
        CD(y) * smoothened_exponential_of_negative_square(y) // when 0.46875 < |x| <= 4.0
    } else {
        (ONE_OVER_SQRT_PI - PQ(1.0 / (y * y))) / y * smoothened_exponential_of_negative_square(y)
        // when 4.0 < |x| < 26.543
    };
    if x < 0.0 {
        erfc_abs_x - 1.0
    } else {
        1.0 - erfc_abs_x
    }
}

/// Cody's complementary error function implementation
/// erfc(x) = 2 - erfc(-x)
pub fn erfc_cody(x: f64) -> f64 {
    let y = x.abs();
    if y <= THRESHOLD {
        // |x| <= 0.46875
        return 1.0 - x * AB(y * y);
    }
    // Compute erfc(|x|)
    let erfc_abs_x = if y >= XBIG {
        0.0 // when |x| ≥ 26.543
    } else if y <= 4.0 {
        CD(y) * smoothened_exponential_of_negative_square(y) // when 0.46875 < |x| <= 4.0
    } else {
        (ONE_OVER_SQRT_PI - PQ(1.0 / (y * y))) / y * smoothened_exponential_of_negative_square(y)
        // when 4.0 < |x| < 26.543
    };
    if x < 0.0 {
        2.0 - erfc_abs_x
    } else {
        erfc_abs_x
    }
}

pub fn erfcx_cody_above_threshold(y: f64) -> f64 {
    if y <= 4.0 {
        // 0.46875 < |x| <= 4.0
        CD(y)
    } else {
        (ONE_OVER_SQRT_PI - PQ(1.0 / (y * y))) / y
    }
}

/// Cody's scaled complementary error function implementation
/// erfcx(x) = exp(x²) * erfc(x)
pub fn erfcx_cody(x: f64) -> f64 {
    let y = x.abs();
    if y <= THRESHOLD {
        // |x| <= 0.46875
        let z = y * y;
        return (z).exp() * (1.0 - x * AB(z));
    }
    if x < XNEG {
        // x < -26.6287357137514
        return f64::MAX;
    }
    let result = erfcx_cody_above_threshold(y);
    if x < 0.0 {
        // erfcx(-x) = 2*exp(x²) - erfcx(x)
        2.0 * smoothened_exponential_of_positive_square(x) - result
    } else {
        result
    }
}
