use crate::erf_cody::erfc_cody;
use std::f64;

// Mathematical constants
pub const SQRT_TWO: f64 = 1.4142135623730950488016887242096980785696718753769;
pub const SQRT_TWO_PI: f64 = 2.5066282746310005024157652848110452530069867406099;

// Constants for norm_cdf asymptotic expansion
const NORM_CDF_ASYMPTOTIC_EXPANSION_FIRST_THRESHOLD: f64 = -10.0;
const NORM_CDF_ASYMPTOTIC_EXPANSION_SECOND_THRESHOLD: f64 = -67.4489750196082; // -1/sqrt(DBL_EPSILON)

// Constants for PJ-2024-Inverse-Normal algorithm
const U_MAX: f64 = 0.3413447460685429; // Phi(1) - 0.5

/// Standard normal probability density function
/// φ(x) = (1/√(2π))·exp(-x²/2)
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_TWO_PI
}

/// Standard normal cumulative distribution function
pub fn norm_cdf(z: f64) -> f64 {
    if z <= NORM_CDF_ASYMPTOTIC_EXPANSION_FIRST_THRESHOLD {
        // Asymptotic expansion for very negative z following (26.2.12) on page 408
        // in M. Abramowitz and A. Stegun, Pocketbook of Mathematical Functions
        let mut sum = 1.0;
        if z >= NORM_CDF_ASYMPTOTIC_EXPANSION_SECOND_THRESHOLD {
            let zsqr = z * z;
            let mut i = 1.0;
            let mut g = 1.0;
            let mut a = f64::MAX;
            let mut lasta;

            loop {
                lasta = a;
                let x = (4.0 * i - 3.0) / zsqr;
                let y = x * ((4.0 * i - 1.0) / zsqr);
                a = g * (x - y);
                sum -= a;
                g *= y;
                i += 1.0;
                a = a.abs();

                if !(lasta > a && a >= (sum * f64::EPSILON).abs()) {
                    break;
                }
            }
        }
        -norm_pdf(z) * sum / z
    } else {
        0.5 * erfc_cody(-z * (1.0 / SQRT_TWO))
    }
}

// PJ-2024-Inverse-Normal

/// Specialisation of x = Phi⁻¹(p) for x ≤ -1, p = Phi(x) ≤ 0.1586552539314570514148
fn inverse_norm_cdf_for_low_probabilities(p: f64) -> f64 {
    // Five branches based on r = sqrt(-ln(p))
    let r = (-p.ln()).sqrt();

    // All of the below are Remez-optimized minimax rational function approximations of order (5,5)
    if r < 6.7 {
        if r < 3.41 {
            if r < 2.05 {
                // Branch I. Accuracy better than 7.6E-17 in perfect arithmetic
                (3.691562302945566191
                    + r * (4.7170590600740689449e1
                        + r * (6.5451292110261454609e1
                            + r * (-7.4594687726045926821e1
                                + r * (-8.3383894003636969722e1 - 1.3054072340494093704e1 * r)))))
                    / (1.0
                        + r * (2.0837211328697753726e1
                            + r * (7.1813812182579255459e1
                                + r * (5.9270122556046077717e1
                                    + r * (9.2216887978737432303 + 1.8295174852053530579e-4 * r)))))
            } else {
                // Branch II. Accuracy better than 9.4E-17 in perfect arithmetic
                (3.2340179116317970288
                    + r * (1.449177828689122096e1
                        + r * (6.8397370256591532878e-1
                            + r * (-1.81254427791789183e1
                                + r * (-1.005916339568646151e1 - 1.2013147879435525574 * r)))))
                    / (1.0
                        + r * (8.8820931773304337525
                            + r * (1.4656370665176799712e1
                                + r * (7.1369811056109768745
                                    + r * (8.4884892199149255469e-1
                                        + 1.0957576098829595323e-5 * r)))))
            }
        } else {
            // Branch III. Accuracy better than 9.1E-17 in perfect arithmetic
            (3.1252235780087584807
                + r * (9.9483724317036560676
                    + r * (-5.1633929115525534628
                        + r * (-1.1070534689309368061e1
                            + r * (-2.8699061335882526744 - 1.5414319494013597492e-1 * r)))))
                / (1.0
                    + r * (7.076769154309171622
                        + r * (8.1086341122361532407
                            + r * (2.0307076064309043613
                                + r * (1.0897972234131828901e-1 + 1.3565983564441297634e-7 * r)))))
        }
    } else {
        if r < 12.9 {
            // Branch IV. Accuracy better than 9E-17 in perfect arithmetic
            (2.6161264950897283681
                + r * (2.250881388987032271
                    + r * (-3.688196041019692267
                        + r * (-2.9644251353150605663
                            + r * (-4.7595169546783216436e-1 - 1.612303318390145052e-2 * r)))))
                / (1.0
                    + r * (3.2517455169035921495
                        + r * (2.1282030272153188194
                            + r * (3.3663746405626400164e-1
                                + r * (1.1400087282177594359e-2 + 3.0848093570966787291e-9 * r)))))
        } else {
            // Branch V. Accuracy better than 9.5E-17 in perfect arithmetic
            (2.3226849047872302955
                + r * (-4.2799650734502094297e-2
                    + r * (-2.5894451568465728432
                        + r * (-8.6385181219213758847e-1
                            + r * (-6.5127593753781672404e-2 - 1.0566357727202585402e-3 * r)))))
                / (1.0
                    + r * (1.9361316119254412206
                        + r * (6.1320841329197493341e-1
                            + r * (4.6054974512474443189e-2
                                + r * (7.471447992167225483e-4 + 2.3135343206304887818e-11 * r)))))
        }
    }
}

/// Inverse of Phi(x)-0.5, i.e., for a given u ∈ [-u_max,u_max] find x such that u = Phi(x)-0.5
fn inverse_norm_cdfm_half_for_midrange_probabilities(u: f64) -> f64 {
    // Accuracy better than 9.8E-17 in perfect arithmetic within this branch
    let s = U_MAX * U_MAX - u * u;
    u * ((2.92958954698308805
        + s * (5.0260572167303103e1
            + s * (3.01870541922933937e2
                + s * (7.4997781456657924e2
                    + s * (6.90489242061408612e2
                        + s * (1.34233243502653864e2 - 7.58939881401259242 * s))))))
        / (1.0
            + s * (1.8918538074574598e1
                + s * (1.29404120448755281e2
                    + s * (3.86821208540417453e2
                        + s * (4.79123914509756757e2 + 1.79227008508102628e2 * s))))))
}

/// Inverse standard normal cumulative distribution function
/// Given p, return x such that p = Phi(x), i.e., x = Phi⁻¹(p)
pub fn inverse_norm_cdf(p: f64) -> f64 {
    let u = p - 0.5;
    if u.abs() < U_MAX {
        inverse_norm_cdfm_half_for_midrange_probabilities(u)
    } else {
        // Tail probability min(p,1-p) = 0.15866. r = √(-ln(min(p,1-p))) > 1.3568425277
        if u > 0.0 {
            -inverse_norm_cdf_for_low_probabilities(1.0 - p)
        } else {
            inverse_norm_cdf_for_low_probabilities(p)
        }
    }
}

/// Inverse error function
/// We can use the internal branches of Phi⁻¹(·) to implement erfinv()
/// avoiding catastrophic subtractive cancellation for small arguments
pub fn erfinv(e: f64) -> f64 {
    // Phi(x) = erfc(-x/√2)/2 = (1+erf(x/√2))/2 = erf(x/√2)/2 + 0.5
    // erf(z) = 2 · ( Phi(√2·z) - 0.5 )
    // Hence, if e = erf(z), and y is the solution to Phi(y) - 0.5 = e/2, then z = y/√2
    if e.abs() < 2.0 * U_MAX {
        inverse_norm_cdfm_half_for_midrange_probabilities(0.5 * e) * (1.0 / SQRT_TWO)
    } else {
        (if e < 0.0 {
            inverse_norm_cdf_for_low_probabilities(0.5 * e + 0.5)
        } else {
            -inverse_norm_cdf_for_low_probabilities(-0.5 * e + 0.5)
        }) * (1.0 / SQRT_TWO)
    }
}
