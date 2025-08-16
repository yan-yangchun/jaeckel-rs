use crate::erf_cody::{erf_cody, erfc_cody, erfcx_cody};
use crate::normaldistribution::{erfinv, inverse_norm_cdf, norm_pdf};
use crate::rationalcubic::{
    convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side,
    convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side,
    rational_cubic_interpolation,
};
use std::f64;

// Mathematical constants
const SQRT_PI_OVER_TWO: f64 = 1.253314137315500251207882642405522626503493370305;
const SQRT_ONE_OVER_THREE: f64 = 0.577350269189625764509148780501957455647601751270;
const TWO_PI_OVER_SQRT_TWENTY_SEVEN: f64 = 1.209199576156145233729385505094770488189377498728;
const SQRT_THREE_OVER_THIRD_ROOT_TWO_PI: f64 = 0.938643487427383566075051356115075878414688769574;
const SQRT_TWO_PI: f64 = 2.5066282746310005024157652848110452530069867406099;
const SQRT_TWO: f64 = 1.4142135623730950488016887242096980785696718753769;
const TWO_PI: f64 = 6.283185307179586476925286766559005768394338798750;
const LN_TWO_PI: f64 = 1.8378770664093454835606594728112352989033387018095;
const PI_OVER_SIX: f64 = 0.523598775598298873077107230546583814032861566563;
const SQRT_THREE: f64 = 1.732050807568877293527446341505872366942805253810;

// Derived constants
const FOURTH_ROOT_DBL_EPSILON: f64 = 0.0001220703125; // sqrt(SQRT_DBL_EPSILON)
const SQRT_DBL_MIN: f64 = 1.4916681462400413e-154; // sqrt(f64::MIN_POSITIVE)

// Special volatility values
pub const VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC: f64 = -f64::MAX;
pub const VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM: f64 = f64::MAX;

// Region definitions
const ETA: f64 = -13.0; // η
const TAU: f64 = 0.21022410381342866; // τ = 2 * SIXTEENTH_ROOT_DBL_EPSILON

// Maximum iterations for implied volatility
const IMPLIED_VOLATILITY_MAXIMUM_ITERATIONS: i32 = 2;

#[inline]
fn is_region_i(thetax: f64, s: f64) -> bool {
    thetax < s * ETA && s * (0.5 * s - (TAU + 0.5 + ETA)) + thetax < 0.0
}

#[inline]
fn is_region_ii(thetax: f64, s: f64) -> bool {
    s * (s - (2.0 * TAU)) - thetax / ETA < 0.0
}

#[inline]
fn square(x: f64) -> f64 {
    x * x
}

#[inline]
pub fn householder3_factor(nu: f64, h_2: f64, h_3: f64) -> f64 {
    (1.0 + 0.5 * h_2 * nu) / (1.0 + nu * (h_2 + h_3 * nu / 6.0))
}

#[inline]
pub fn householder4_factor(nu: f64, h_2: f64, h_3: f64, h_4: f64) -> f64 {
    (1.0 + nu * (h_2 + nu * h_3 / 6.0))
        / (1.0 + nu * (1.5 * h_2 + nu * (h_2 * h_2 * 0.25 + h_3 / 3.0 + nu * h_4 / 24.0)))
}

#[inline]
fn implied_volatility_output(_count: i32, volatility: f64) -> f64 {
    volatility
}

/// Normalised intrinsic value
/// θ·[exp(θ·x/2) - exp(-θ·x/2)]
#[inline]
pub fn normalised_intrinsic(thetax: f64) -> f64 {
    if thetax <= 0.0 {
        return 0.0;
    }
    let x2 = thetax * thetax;
    if x2 < 98.0 * FOURTH_ROOT_DBL_EPSILON {
        // Use Taylor series for small x
        thetax
            * (1.0
                + x2 * ((1.0 / 24.0)
                    + x2 * ((1.0 / 1920.0) + x2 * ((1.0 / 322560.0) + (1.0 / 92897280.0) * x2))))
    } else {
        // Use direct formula
        (0.5 * thetax).exp() - (-0.5 * thetax).exp()
    }
}

// Helper function for Y'(h) calculation
#[inline]
fn yprime_tail_expansion_rational_function_part(w: f64) -> f64 {
    w * (-2.9999999999994663866
        + w * (-1.7556263323542206288E2
            + w * (-3.4735035445495633334E3
                + w * (-2.7805745693864308643E4
                    + w * (-8.3836021460741980839E4 - 6.6818249032616849037E4 * w)))))
        / (1.0
            + w * (6.3520877744831739102E1
                + w * (1.4404389037604337538E3
                    + w * (1.4562545638507033944E4
                        + w * (6.6886794165651675684E4
                            + w * (1.2569970380923908488E5 + 6.9286518679803751694E4 * w))))))
}

// Y'(h) = 1+h·Y(h) avoiding subtractive cancellation
fn yprime(h: f64) -> f64 {
    if h < -4.0 {
        let w = 1.0 / (h * h);
        w * (1.0 + yprime_tail_expansion_rational_function_part(w))
    } else if h <= -0.46875 {
        // Remez-optimized minimax rational function
        (1.0000000000594317229
            - h * (6.1911449879694112749E-1
                - h * (2.2180844736576013957E-1
                    - h * (4.5650900351352987865E-2
                        - h * (5.545521007735379052E-3
                            - h * (3.0717392274913902347E-4
                                - h * (4.2766597835908713583E-8
                                    + 8.4592436406580605619E-10 * h)))))))
            / (1.0
                - h * (1.8724286369589162071
                    - h * (1.5685497236077651429
                        - h * (7.6576489836589035112E-1
                            - h * (2.3677701403094640361E-1
                                - h * (4.6762548903194957675E-2
                                    - h * (5.5290453576936595892E-3
                                        - 3.0822020417927147113E-4 * h)))))))
    } else {
        1.0 + h * SQRT_PI_OVER_TWO * erfcx_cody(-(1.0 / SQRT_TWO) * h)
    }
}

/// Computational regions for selecting optimal algorithms
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ComputationRegion {
    /// Region I: Asymptotic expansion region
    RegionI,
    /// Region II: Small-t expansion region (most critical for precision)
    RegionII,
    /// Region III: Standard computation region
    RegionIII,
}

/// Detect the computational region for optimal algorithm selection
pub fn detect_computation_region(thetax: f64, s: f64) -> ComputationRegion {
    if is_region_i(thetax, s) {
        return ComputationRegion::RegionI;
    }
    if is_region_ii(thetax, s) {
        return ComputationRegion::RegionII;
    }
    ComputationRegion::RegionIII
}

pub fn ln_normalised_vega(x: f64, s: f64) -> f64 {
    let h = x / s;
    let t = 0.5 * s;
    -(LN_TWO_PI * 0.5) - 0.5 * (h * h + t * t)
}

pub fn scaled_normalised_black_and_ln_vega(thetax: f64, s: f64) -> (f64, f64) {
    match detect_computation_region(thetax, s) {
        ComputationRegion::RegionII => {
            let h = thetax / s;
            let t = 0.5 * s;
            let bx_scaled = small_t_expansion_of_scaled_normalised_black(h, t);
            let ln_vega = ln_normalised_vega(thetax, s);
            (bx_scaled, ln_vega)
        }
        _ => {
            // For other regions, use standard formula
            let ln_vega = (normalised_vega(thetax, s)).ln();
            let bx_scaled = normalised_black(thetax, s) * (-ln_vega).exp();
            (bx_scaled, ln_vega)
        }
    }
}

fn compute_f_lower_map_and_first_two_derivatives(x: f64, s: f64) -> (f64, f64, f64) {
    let ax = x.abs();
    let z = SQRT_ONE_OVER_THREE * ax / s;
    let y = z * z;
    let s2 = s * s;
    let phi = 0.5 * erfc_cody((1.0 / SQRT_TWO) * z); // norm_cdf(-z)
    let phi_pdf = norm_pdf(z);

    let fpp = PI_OVER_SIX * y / (s2 * s)
        * phi
        * (8.0 * SQRT_THREE * s * ax + (3.0 * s2 * (s2 - 8.0) - 8.0 * x * x) * phi / phi_pdf)
        * (2.0 * y + 0.25 * s2).exp();

    let phi2 = phi * phi;
    let phi3 = phi2 * phi;
    let fp = TWO_PI * y * phi2 * (y + 0.125 * s * s).exp();
    let f = TWO_PI_OVER_SQRT_TWENTY_SEVEN * ax * phi3;

    (f, fp, fpp)
}

fn inverse_f_lower_map(x: f64, f: f64) -> f64 {
    if f <= f64::MIN_POSITIVE {
        return f64::MIN_POSITIVE;
    }
    // Avoid catastrophic cancellation for small f
    let ax = x.abs();
    let f_cbrt = f.cbrt();
    let ax_cbrt = ax.cbrt();
    let ratio = SQRT_THREE_OVER_THIRD_ROOT_TWO_PI * f_cbrt / ax_cbrt;

    // Check for numerical issues
    if ratio >= 1.0 {
        // This shouldn't happen mathematically, but can due to rounding
        // Return a large value to signal the problem
        return 100.0 * ax;
    }

    let inv_phi = inverse_norm_cdf(ratio);
    if inv_phi.is_infinite() || inv_phi.is_nan() {
        return 100.0 * ax;
    }

    (ax / (SQRT_THREE * inv_phi)).abs()
}

// f(x) := 1 - erfcx(x)  ≈  2/sqrt_π·x - x² + 4/(3sqrt_π)·x³ + ... for small x
fn one_minus_erfcx(x: f64) -> f64 {
    if x < -1.0 / 5.0 || x > 1.0 / 3.0 {
        return 1.0 - erfcx_cody(x);
    }
    // Remez-optimized minimax rational function of order (4,5) for g(x) := (2/sqrt_π-f(x)/x)/x
    // The relative accuracy of f(x) ≈ x·(2/sqrt_π-x·g(x)) is better than 2.5E-17
    x * (1.128379167095512573896
        - x * (1.0000000000000002
            + x * (1.1514967181784756
                + x * (0.57689001208873741
                    + x * (0.14069188744609651 + 0.014069285713634565 * x))))
            / (1.0
                + x * (1.9037494962421563
                    + x * (1.5089908593742723
                        + x * (0.62486081658640257
                            + x * (0.1358008134514386 + 0.012463320728346347 * x))))))
}

/// Black (1976) option value
/// B(F,K,σ,T,θ=±1) = θ·[F·N(θ·d₁) - K·N(θ·d₂)]
/// where d₁ = [ln(F/K) + σ²T/2]/(σ√T) and d₂ = d₁ - σ√T
pub fn Black(f: f64, k: f64, sigma: f64, t: f64, theta: f64) -> f64 {
    let s = sigma * t.sqrt();
    // Specialisation for x = 0 where b(s) = 1-2·Φ(-s/2) = erf(s/√8)
    if k == f {
        return f * erf_cody((0.5 / SQRT_TWO) * s);
    }
    // Map in-the-money to out-of-the-money
    let intrinsic = if theta < 0.0 { k - f } else { f - k }.max(0.0);
    if s <= 0.0 {
        intrinsic
    } else {
        intrinsic + (f.sqrt() * k.sqrt()) * normalised_black(-(f / k).ln().abs(), s)
    }
}

/// Normalised Black option value
/// β(x,s) := B(F,K,σ,T,θ=±1)/√(F·K) with x=ln(F/K) and s=σ√T
pub fn NormalisedBlack(x: f64, s: f64, theta: f64) -> f64 {
    // Specialisation for x = 0 where b(s) = 1-2·Φ(-s/2) = erf(s/√8)
    if x == 0.0 {
        return erf_cody((0.5 / SQRT_TWO) * s);
    }
    let thetax = if theta < 0.0 { -x } else { x };
    normalised_intrinsic(thetax)
        + if s <= 0.0 {
            0.0
        } else {
            normalised_black(-x.abs(), s)
        }
}

/// Calculate implied Black volatility from option price
pub fn ImpliedBlackVolatility(price: f64, f: f64, k: f64, t: f64, theta: f64) -> f64 {
    let max_price = if theta < 0.0 { k } else { f };
    if price >= max_price {
        return implied_volatility_output(0, VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM);
    }
    let mu = if theta < 0.0 { k - f } else { f - k }; // Map in-the-money to out-of-the-money
    let beta = if mu > 0.0 { price - mu } else { price } / (f.sqrt() * k.sqrt());
    let x = if f == k { 0.0 } else { -(f / k).ln().abs() };
    lets_be_rational(beta, x, IMPLIED_VOLATILITY_MAXIMUM_ITERATIONS) / t.sqrt()
}

/// Calculate normalised implied Black volatility
pub fn NormalisedImpliedBlackVolatility(beta: f64, x: f64, theta: f64) -> f64 {
    // Map in-the-money to out-of-the-money
    let thetax = if theta < 0.0 { -x } else { x };
    lets_be_rational(
        beta - normalised_intrinsic(thetax),
        -x.abs(),
        IMPLIED_VOLATILITY_MAXIMUM_ITERATIONS,
    )
}

/// Calculate Vega: ∂Black(F,K,σ,T)/∂σ
pub fn Vega(f: f64, k: f64, sigma: f64, t: f64) -> f64 {
    (f.sqrt() * k.sqrt()) * NormalisedVega((f / k).ln(), sigma * t.sqrt()) * t.sqrt()
}

/// Calculate normalised Vega: ∂β(x,s)/∂s
pub fn NormalisedVega(x: f64, s: f64) -> f64 {
    let ax = x.abs();
    if ax <= 0.0 {
        return (1.0 / SQRT_TWO_PI) * (-0.125 * s * s).exp();
    }
    if s <= 0.0 || s <= ax * SQRT_DBL_MIN {
        return 0.0;
    }
    normalised_vega(x, s)
}

/// Calculate Volga: ∂²Black(F,K,σ,T)/∂σ²
pub fn Volga(f: f64, k: f64, sigma: f64, t: f64) -> f64 {
    (f.sqrt() * k.sqrt()) * normalised_volga((f / k).ln(), sigma * t.sqrt()) * t
}

/// Calculate normalised Volga: ∂²β(x,s)/∂s²
pub fn NormalisedVolga(x: f64, s: f64) -> f64 {
    normalised_volga(x, s)
}

// Asymptotic expansion of the 'scaled normalised Black' function
#[rustfmt::skip]
fn asymptotic_expansion_of_scaled_normalised_black(h: f64, t: f64) -> f64 {
    let e = square(t / h);
    let r = (h + t) * (h - t);
    let q = square(h / r);

    // Coefficients for the asymptotic expansion (up to A20 for maximum precision)
    const A0: f64 = 2.0;
    let a1 = -6.0 - 2.0 * e;
    let a2 = 30.0 + e * (60.0 + 6.0 * e);
    let a3 = -2.1E2 + e * (-1.05E3 + e * (-6.3E2 - 30.0 * e));
    let a4 = 1.89E3 + e * (1.764E4 + e * (2.646E4 + e * (7.56E3 + 2.1E2 * e)));
    let a5 = -2.079E4 + e * (-3.1185E5 + e * (-8.7318E5 + e * (-6.237E5 + e * (-1.0395E5 - 1.89E3 * e))));
    let a6 = 2.7027E5 + e * (5.94594E6 + e * (2.675673E7 + e * (3.567564E7 + e * (1.486485E7 + e * (1.62162E6 + 2.079E4 * e)))));
    let a7 = -4.05405E6 + e * (-1.2297285E8 + e * (-8.1162081E8 + e * (-1.73918745E9 + e * (-1.35270135E9 + e * (-3.6891855E8 + e * (-2.837835E7 - 2.7027E5 * e))))));
    let a8 = 6.891885E7 + e * (2.756754E9 + e * (2.50864614E10 + e * (7.88431644E10 + e * (9.85539555E10 + e * (5.01729228E10 + e * (9.648639E9 + e * (5.513508E8 + 4.05405E6 * e)))))));
    let a9 = -1.30945815E9 + e * (-6.678236565E10 + e * (-8.013883878E11 + e * (-3.4726830138E12 + e * (-6.3665855253E12 + e * (-5.2090245207E12 + e * (-1.8699062382E12 + e * (-2.671294626E11 + e * (-1.178512335E10 - 6.891885E7 * e))))))));
    let a10 = 2.749862115E10 + e * (1.7415793395E12 + e * (2.664616389435E13 + e * (1.52263793682E14 + e * (3.848890340295E14 + e * (4.618668408354E14 + e * (2.664616389435E14 + e * (7.10564370516E13 + e * (7.83710702775E12 + e * (2.749862115E11 + 1.30945815E9 * e)))))))));
    let a11 = -6.3246828645E11 + e * (-4.870005805665E13 + e * (-9.2530110307635E14 + e * (-6.74147946527055E15 + e * (-2.24715982175685E16 + e * (-3.71802806872497E16 + e * (-3.14602375045959E16 + e * (-1.34829589305411E16 + e * (-2.77590330922905E15 + e * (-2.4350029028325E14 + e * (-6.95715115095E12 - 2.749862115E10 * e))))))))));
    let a12 = 1.581170716125E13 + e * (1.454677058835E15 + e * (3.36030400590885E16 + e * (3.04027505296515E17 + e * (1.29211689751018875E18 + e * (2.81916414002223E18 + e * (3.289024830025935E18 + e * (2.067387036016302E18 + e * (6.8406188691715875E17 + e * (1.12010133530295E17 + e * (8.0007238235925E15 + e * (1.89740485935E14 + 6.3246828645E11 * e)))))))))));
    let a13 = -4.2691609335375E14 + e * (-4.624924344665625E16 + e * (-1.2764791191277125E18 + e * (-1.40412703104048375E19 + e * (-7.41067044160255312E19 + e * (-2.06151377739125569E20 + e * (-3.17155965752500875E20 + e * (-2.74868503652167425E20 + e * (-1.33392067948845956E20 + e * (-3.51031757760120938E19 + e * (-4.6804234368016125E18 + e * (-2.774954606799375E17 + e * (-5.54990921359875E15 - 1.581170716125E13 * e))))))))))));
    let a14 = 1.238056670725875E16 + e * (1.5599514051146025E18 + e * (5.06984206662245812E19 + e * (6.66322100184665925E20 + e * (4.27556680951827302E21 + e * (1.47701398874267613E22 + e * (2.89721974714909549E22 + e * (3.31110828245610914E22 + e * (2.2155209831140142E22 + e * (8.55113361903654604E21 + e * (1.83238577550783129E21 + e * (2.02793682664898325E20 + e * (1.01396841332449162E19 + e * (1.733279339016225E17 + 4.2691609335375E14 * e)))))))))))));
    let a15 = -3.8379756792502125E17 + e * (-5.56506473491280812E19 + e * (-2.10359446979704147E21 + e * (-3.25556286992399275E22 + e * (-2.49593153360839444E23 + e * (-1.04829124411552567E24 + e * (-2.55352995361474201E24 + e * (-3.72085793241005264E24 + e * (-3.28310994036181115E24 + e * (-1.74715207352587611E24 + e * (-5.49104937393846778E23 + e * (-9.76668860977197826E22 + e * (-9.11557603578717971E21 + e * (-3.89554531443896569E20 + e * (-5.75696351887531875E18 - 1.238056670725875E16 * e))))))))))))));
    let a16 = 1.26653197415257012E19 + e * (2.09399953059891594E21 + e * (9.10889795810528434E22 + e * (1.63960163245895118E24 + e * (1.48019591819210871E25 + e * (7.42789224401858187E25 + e * (2.19979885688242617E26 + e * (3.98058840769200926E26 + e * (4.47816195865351041E26 + e * (3.1425697955463231E26 + e * (1.36178024473674001E26 + e * (3.55247020366106089E25 + e * (5.32870530549159134E24 + e * (4.25081904711579936E23 + e * (1.57049964794918696E22 + e * (2.0264511586441122E20 + 3.8379756792502125E17 * e)))))))))))))));
    let a17 = -4.43286190953399544E20 + e * (-8.28945177082857147E22 + e * (-4.11156807833097145E24 + e * (-8.51681959082844086E25 + e * (-8.9426605703698629E26 + e * (-5.28429942794582808E27 + e * (-1.86982902835006224E28 + e * (-4.11362386237013693E28 + e * (-5.74697451360533836E28 + e * (-5.14202982796267117E28 + e * (-2.9383027588358121E28 + e * (-1.05685988558916562E28 + e * (-2.32509174829616435E27 + e * (-2.9808868567899543E26 + e * (-2.05578403916548572E25 + e * (-6.63156141666285717E23 + e * (-7.53586524620779224E21 - 1.26653197415257012E19 * e))))))))))))))));
    let a18 = 1.64015890652757831E22 + e * (3.44433370370791445E24 + e * (1.93227120778014001E26 + e * (4.56384056694737831E27 + e * (5.51464068506141545E28 + e * (3.79006214355130008E29 + e * (1.5791925598130417E30 + e * (4.15102044293713818E30 + e * (7.05063031116528617E30 + e * (7.83403367907254019E30 + e * (5.707653109038565E30 + e * (2.70718724539378577E30 + e * (8.21180131102781683E29 + e * (1.54409939181719633E29 + e * (1.71144021260526687E28 + e * (1.030544644149408E27 + e * (2.92768364815172729E25 + e * (2.95228603174964096E23 + 4.43286190953399544E20 * e)))))))))))))))));
    let a19 = -6.39661973545755542E23 + e * (-1.49894122467555382E26 + e * (-9.44332971545598906E27 + e * (-2.52271808112895708E29 + e * (-3.4757449117776742E30 + e * (-2.74899824840597868E31 + e * (-1.33220684345828198E32 + e * (-4.12349737260896802E32 + e * (-8.36827407970643511E32 + e * (-1.13045105989016755E33 + e * (-1.02278905418634207E33 + e * (-6.18524605891345204E32 + e * (-2.47409842356538081E32 + e * (-6.41432924628061693E31 + e * (-1.04272347353330226E31 + e * (-1.00908723245158283E30 + e * (-5.35122017209172713E28 + e * (-1.34904710220799844E27 + e * (-1.21535774973693553E25 - 1.64015890652757831E22 * e))))))))))))))))));
    let a20 = 2.62261409153759772E25 + e * (6.81879663799775407E27 + e * (4.79361403651242111E29 + e * (1.43808421095372633E31 + e * (2.24101456206955687E32 + e * (2.02098767779363674E33 + e * (1.12708928184645126E34 + e * (4.05752141464722454E34 + e * (9.69628279235549981E34 + e * (1.56501406473106313E35 + e * (1.72151547120416944E35 + e * (1.29283770564739997E35 + e * (6.59347229880173987E34 + e * (2.25417856369290252E34 + e * (5.05246919448409185E33 + e * (7.17124659862258199E32 + e * (6.11185789655333692E31 + e * (2.87616842190745267E30 + e * (6.47785680609786637E28 + e * (5.24522818307519544E26 + 6.39661973545755542E23 * e)))))))))))))))))));

    // Adaptive term selection based on thresholds (UP_TO_20_TERMS version)
    let mut omega = 0.0;
    let thresholds = [10.589, 10.876, 11.22, 11.635, 12.143, 12.771, 13.559, 14.566, 15.884, 17.656, 20.129, 23.743, 29.365, 38.892, 57.148, 99.336];
    let threshold_value = -h - t + TAU;

    // Find which threshold range we're in
    let mut case = 16; // default case
    for (i, &threshold) in thresholds.iter().enumerate() {
        if threshold_value < threshold {
            case = i;
            break;
        }
    }

    // Apply coefficients based on threshold case (mimicking C++ switch with fallthrough)
    match case {
        0 => { omega = q * (a20 + omega); omega = q * (a19 + omega); omega = q * (a18 + omega); omega = q * (a17 + omega); omega = q * (a16 + omega); omega = q * (a15 + omega); omega = q * (a14 + omega); omega = q * (a13 + omega); omega = q * (a12 + omega); omega = q * (a11 + omega); omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        1 => { omega = q * (a19 + omega); omega = q * (a18 + omega); omega = q * (a17 + omega); omega = q * (a16 + omega); omega = q * (a15 + omega); omega = q * (a14 + omega); omega = q * (a13 + omega); omega = q * (a12 + omega); omega = q * (a11 + omega); omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        2 => { omega = q * (a18 + omega); omega = q * (a17 + omega); omega = q * (a16 + omega); omega = q * (a15 + omega); omega = q * (a14 + omega); omega = q * (a13 + omega); omega = q * (a12 + omega); omega = q * (a11 + omega); omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        3 => { omega = q * (a17 + omega); omega = q * (a16 + omega); omega = q * (a15 + omega); omega = q * (a14 + omega); omega = q * (a13 + omega); omega = q * (a12 + omega); omega = q * (a11 + omega); omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        4 => { omega = q * (a16 + omega); omega = q * (a15 + omega); omega = q * (a14 + omega); omega = q * (a13 + omega); omega = q * (a12 + omega); omega = q * (a11 + omega); omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        5 => { omega = q * (a15 + omega); omega = q * (a14 + omega); omega = q * (a13 + omega); omega = q * (a12 + omega); omega = q * (a11 + omega); omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        6 => { omega = q * (a14 + omega); omega = q * (a13 + omega); omega = q * (a12 + omega); omega = q * (a11 + omega); omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        7 => { omega = q * (a13 + omega); omega = q * (a12 + omega); omega = q * (a11 + omega); omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        8 => { omega = q * (a12 + omega); omega = q * (a11 + omega); omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        9 => { omega = q * (a11 + omega); omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        10 => { omega = q * (a10 + omega); omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        11 => { omega = q * (a9 + omega); omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        12 => { omega = q * (a8 + omega); omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        13 => { omega = q * (a7 + omega); omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        14 => { omega = q * (a6 + omega); omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        15 => { omega = q * (a5 + omega); omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
        _ => { omega = A0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + omega)))); },
    }

    let bx = (t / r) * omega;
    bx
}

// Small t expansion of scaled normalised Black
fn small_t_expansion_of_scaled_normalised_black(h: f64, t: f64) -> f64 {
    let a = yprime(h);
    let h_square = h * h;
    let t_square = t * t;

    let b0 = 2.0 * a;
    let b1 = (-1.0 + a * (3.0 + h_square)) / 3.0;
    let b2 = (-7.0 - h_square + a * (15.0 + h_square * (10.0 + h_square))) / 60.0;
    let b3 = (-57.0
        + (-18.0 - h_square) * h_square
        + a * (105.0 + h_square * (105.0 + h_square * (21.0 + h_square))))
        / 2520.0;
    let b4 = (-561.0
        + h_square * (-285.0 + (-33.0 - h_square) * h_square)
        + a * (945.0 + h_square * (1260.0 + h_square * (378.0 + h_square * (36.0 + h_square)))))
        / 181440.0;
    let b5 = (-6555.0
        + h_square * (-4680.0 + h_square * (-840.0 + (-52.0 - h_square) * h_square))
        + a * (10395.0
            + h_square
                * (17325.0
                    + h_square * (6930.0 + h_square * (990.0 + h_square * (55.0 + h_square))))))
        / 19958400.0;
    let b6 = (-89055.0
        + h_square
            * (-82845.0
                + h_square * (-20370.0 + h_square * (-1926.0 + (-75.0 - h_square) * h_square)))
        + a * (135135.0
            + h_square
                * (270270.0
                    + h_square
                        * (135135.0
                            + h_square
                                * (25740.0
                                    + h_square * (2145.0 + h_square * (78.0 + h_square)))))))
        / 3113510400.0;

    t * (b0
        + t_square
            * (b1
                + t_square
                    * (b2 + t_square * (b3 + t_square * (b4 + t_square * (b5 + b6 * t_square))))))
}

// Optimal use of Cody's functions for normalised Black
fn normalised_black_with_optimal_use_of_codys_functions(thetax: f64, s: f64) -> f64 {
    const CODYS_THRESHOLD: f64 = 0.46875;
    let h = thetax / s;
    let t = 0.5 * s;
    let q_1 = -(1.0 / SQRT_TWO) * (h + t);
    let q_2 = -(1.0 / SQRT_TWO) * (h - t);

    let two_b = if q_1 < CODYS_THRESHOLD {
        if q_2 < CODYS_THRESHOLD {
            (0.5 * thetax).exp() * erfc_cody(q_1) - (-0.5 * thetax).exp() * erfc_cody(q_2)
        } else {
            (0.5 * thetax).exp() * erfc_cody(q_1) - (-0.5 * (h * h + t * t)).exp() * erfcx_cody(q_2)
        }
    } else {
        if q_2 < CODYS_THRESHOLD {
            (-0.5 * (h * h + t * t)).exp() * erfcx_cody(q_1)
                - (-0.5 * thetax).exp() * erfc_cody(q_2)
        } else {
            (-0.5 * (h * h + t * t)).exp() * (erfcx_cody(q_1) - erfcx_cody(q_2))
        }
    };

    f64::max(0.5 * two_b, 0.0)
}

// Normalised vega: ∂b(x,s)/∂s
#[inline]
fn normalised_vega(x: f64, s: f64) -> f64 {
    let ax = x.abs();
    if ax <= 0.0 {
        return (1.0 / SQRT_TWO_PI) * (-0.125 * s * s).exp();
    }
    if s <= 0.0 || s <= ax * SQRT_DBL_MIN {
        return 0.0;
    }

    match detect_computation_region(x, s) {
        ComputationRegion::RegionII => {
            let (_bx_scaled, ln_vega) = scaled_normalised_black_and_ln_vega(x, s);
            ln_vega.exp()
        }
        _ => {
            let h = x / s;
            let t = 0.5 * s;
            (1.0 / SQRT_TWO_PI) * (-0.5 * (h * h + t * t)).exp()
        }
    }
}

// Inverse of normalised vega for numerical stability
#[inline]
fn inv_normalised_vega(x: f64, s: f64) -> f64 {
    let ax = x.abs();
    if ax <= 0.0 {
        return SQRT_TWO_PI * (0.125 * s * s).exp();
    }
    if s <= 0.0 || s <= ax * SQRT_DBL_MIN {
        return f64::MAX;
    }
    let h = x / s;
    let t = 0.5 * s;
    SQRT_TWO_PI * (0.5 * (h * h + t * t)).exp()
}

// Normalised volga: ∂²b(x,s)/∂s²
fn normalised_volga(x: f64, s: f64) -> f64 {
    let ax = x.abs();
    if ax <= 0.0 {
        return (1.0 / SQRT_TWO_PI) * (-0.125 * s * s).exp();
    }
    if s <= 0.0 || s <= ax * SQRT_DBL_MIN {
        return 0.0;
    }
    let h = x / s;
    let t = 0.5 * s;
    let h_square = h * h;
    let t_square = t * t;
    (1.0 / SQRT_TWO_PI) * (-0.5 * (h_square + t_square)).exp() * (h_square - t_square) / s
}

// Main normalised Black function
fn normalised_black(thetax: f64, s: f64) -> f64 {
    if s <= 0.0 {
        return 0.0;
    }

    match detect_computation_region(thetax, s) {
        ComputationRegion::RegionI => {
            asymptotic_expansion_of_scaled_normalised_black(thetax / s, 0.5 * s)
                * normalised_vega(thetax, s)
        }
        ComputationRegion::RegionII => {
            let (bx_scaled, ln_vega) = scaled_normalised_black_and_ln_vega(thetax, s);
            bx_scaled * ln_vega.exp()
        }
        ComputationRegion::RegionIII => {
            normalised_black_with_optimal_use_of_codys_functions(thetax, s)
        }
    }
}

// Univariate rational function approximations for b_l(x)/b_max(x)
#[inline]
fn b_l_over_b_max(s_c: f64) -> f64 {
    if s_c < 2.6267851073127395 {
        if s_c < 0.7099295739719539 {
            // Branch I: For small |x|, i.e., small s_c
            // f ≈ (exp(-1/π)/4-Phi(-sqrt_(π/2))/2)·s_c² + exp(-1/π)/(3·sqrt_(2π))·s_c³ + O(s_c⁴)
            //   c₂ := (exp(-1/π)/4-Phi(-sqrt_(π/2))/2) =  0.07560996640296361767172
            //   c₃ := exp(-1/π)/(3·sqrt_(2π))        = -0.09672719281339436290858
            // Nonlinear-Remez optimized minimax rational function of order (5,4)
            let g = (8.0741072372882856924E-2
                + s_c
                    * (9.8078911786358897272E-2
                        + s_c
                            * (3.9760631445677058375E-2
                                + s_c
                                    * (5.9716928459589189876E-3
                                        + s_c
                                            * (-6.4036399341479799981E-6
                                                + 4.5425102093616062245E-7 * s_c)))))
                / (1.0
                    + s_c
                        * (1.8594977672287664353
                            + s_c
                                * (1.3658801475711790419
                                    + s_c
                                        * (4.6132707108655653215E-1
                                            + 6.1254597049831720643E-2 * s_c))));
            // Branch I. Accuracy better than 7.43E-17 in perfect arithmetic
            (s_c * s_c) * (0.07560996640296361767172 + s_c * (s_c * g - 0.09672719281339436290858))
        } else {
            // Branch II: Intermediate range
            // Remez optimized minimax rational function of order (6,6)
            // Branch II. Accuracy better than 8.77E-17 in perfect arithmetic
            (1.9795737927598581235E-9
                + s_c
                    * (-2.7081288564685588037E-8
                        + s_c
                            * (7.5610142272549044609E-2
                                + s_c
                                    * (6.917130174466834016E-2
                                        + s_c
                                            * (2.9537058950963019803E-2
                                                + s_c
                                                    * (6.5849252702302307774E-3
                                                        + 6.9711400639834715731E-4 * s_c))))))
                / (1.0
                    + s_c
                        * (2.1941448525586579756
                            + s_c
                                * (2.1297103549995181357
                                    + s_c
                                        * (1.1571483187179784072
                                            + s_c
                                                * (3.7831622253060456794E-1
                                                    + s_c
                                                        * (7.1714862448829349869E-2
                                                            + 6.6361975827861200167E-3 * s_c))))))
        }
    } else if s_c < 7.348469228349534 {
        // Branch III: Higher intermediate range
        // Remez optimized minimax rational function of order (6,6)
        // Branch III. Accuracy better than 7.49E-17 in perfect arithmetic
        (-9.3325115354837883291E-5
            + s_c
                * (5.3118033972794648837E-4
                    + s_c
                        * (7.4114855448345002595E-2
                            + s_c
                                * (7.4039658186822817454E-2
                                    + s_c
                                        * (3.9225177407687604785E-2
                                            + s_c
                                                * (1.0022913378254090083E-2
                                                    + 1.7012579407246055469E-3 * s_c))))))
            / (1.0
                + s_c
                    * (2.2217238132228132256
                        + s_c
                            * (2.3441816707087403282
                                + s_c
                                    * (1.3912323646271141826
                                        + s_c
                                            * (5.3231258443501838354E-1
                                                + s_c
                                                    * (1.1744005919716101572E-1
                                                        + 1.6195405895930935811E-2 * s_c))))))
    } else {
        // Branch IV: Far tail - transform to reciprocal evaluation for numerical stability
        // Default version without USE_RECIPROCAL_EVALUATION_IN_FAR_TAIL_OF_BL_OVER_BMAX
        // Transformed back to (6,6) rational function of s_c via analytical simplification
        // Branch IV. Accuracy better than 8.4E-17 in perfect arithmetic
        (1.4500072297240603183E-3
            + s_c
                * (-1.5116692485011195757E-3
                    + s_c
                        * (7.1682178310936334831E-2
                            + s_c
                                * (3.921610857820463493E-2
                                    + s_c
                                        * (2.9342405658628443931E-2
                                            + s_c
                                                * (5.1832526171631521426E-3
                                                    + 1.6930208078421474854E-3 * s_c))))))
            / (1.0
                + s_c
                    * (1.6176313502305414664
                        + s_c
                            * (1.6823159175281531664
                                + s_c
                                    * (8.4878307567372222113E-1
                                        + s_c
                                            * (3.7543742137375791321E-1
                                                + s_c
                                                    * (7.126137099644302999E-2
                                                        + 1.6116992546788676159E-2 * s_c))))))
    }
}

// Univariate rational function approximations for b_u(x)/b_max(x)
// Four-branch implementation matching C++ exactly for maximum precision
#[inline]
fn b_u_over_b_max(s_c: f64) -> f64 {
    if s_c < 1.7888543819998317 {
        if s_c < 0.7745966692414833 {
            // Branch I: For small |x|, i.e., small s_c
            // f ≈ 1-2·Phi(-sqrt_(π/2)) + (exp(-π/4)/4-Phi(-sqrt_(π/2))/2)·s_c² + O(s_c³)
            //   c₀ := 1-2·Phi(-sqrt_(π/2))             = 0.7899085945560627246288
            //   c₂ := (exp(-π/4)/4-Phi(-sqrt_(π/2))/2) = 0.0614616805805147403487
            // Nonlinear-Remez optimized minimax rational function of order (5,4)
            let g = (-6.063099881233561706E-2
                + s_c
                    * (-8.1011946637120604985E-2
                        + s_c
                            * (-4.2505564862438753828E-2
                                + s_c
                                    * (-8.9880000946868691788E-3
                                        + s_c
                                            * (-7.5603072110443268356E-6
                                                + 4.3879556621540147458E-7 * s_c)))))
                / (1.0
                    + s_c
                        * (1.8400371530721828756
                            + s_c
                                * (1.5709283443886143691
                                    + s_c
                                        * (6.8913245453611400484E-1
                                            + 1.4703173061720980923E-1 * s_c))));
            // Branch I. Accuracy better than 9.2E-17 in perfect arithmetic
            0.7899085945560627246288 + (s_c * s_c) * (0.0614616805805147403487 + s_c * g)
        } else {
            // Branch II: Intermediate range
            // Remez optimized minimax rational function of order (6,5)
            // Branch II. Accuracy better than 8.4E-17 in perfect arithmetic
            (7.8990944435755287611E-1
                + s_c
                    * (-1.2655410534988972886
                        + s_c
                            * (-2.8803040699221003256
                                + s_c
                                    * (-2.6936198689113258727
                                        + s_c
                                            * (-1.1213067281643205754
                                                + s_c
                                                    * (-2.1277793801691629892E-1
                                                        + 5.1486445905299802703E-6 * s_c))))))
                / (1.0
                    + s_c
                        * (-1.6021222722060444448
                            + s_c
                                * (-3.7242680976480704555
                                    + s_c
                                        * (-3.2083117718907365085
                                            + s_c
                                                * (-1.2922333835930958583
                                                    - 2.3762328334050001161E-1 * s_c)))))
        }
    } else if s_c < 6.164414002968976 {
        // Branch III: Higher intermediate range
        // Remez optimized minimax rational function of order (6,6)
        // Branch III. Accuracy better than 7.7E-17 in perfect arithmetic
        (7.8990640048967596475E-1
            + s_c
                * (1.5993699253596663678
                    + s_c
                        * (1.6481729039140370242
                            + s_c
                                * (9.8227188109869200166E-1
                                    + s_c
                                        * (3.6313557966186936883E-1
                                            + s_c
                                                * (7.8277036261179606301E-2
                                                    + 9.3404307364538726214E-3 * s_c))))))
            / (1.0
                + s_c
                    * (2.0247407005640401446
                        + s_c
                            * (2.0087454279103740489
                                + s_c
                                    * (1.1627561803056961973
                                        + s_c
                                            * (4.2004672123723823581E-1
                                                + s_c
                                                    * (8.9130862793887234546E-2
                                                        + 1.0436767768858021717E-2 * s_c))))))
    } else {
        // Branch IV: Far tail - transform to reciprocal evaluation for numerical stability
        // Default version without USE_RECIPROCAL_EVALUATION_IN_FAR_TAIL_OF_BU_OVER_BMAX
        // Transformed back to (6,6) rational function of s_c via analytical simplification
        // Branch IV. Accuracy better than 3.9E-17 in perfect arithmetic
        (7.91133825948419359E-1
            + s_c
                * (1.24653733210880042
                    + s_c
                        * (1.32747426980537386
                            + s_c
                                * (6.95009705717846778E-1
                                    + s_c
                                        * (3.05965944268228457E-1
                                            + s_c
                                                * (6.02200363391352887E-2
                                                    + 1.29050244454344842E-2 * s_c))))))
            / (1.0
                + s_c
                    * (1.58117486714634672
                        + s_c
                            * (1.60144713247629644
                                + s_c
                                    * (8.30040185836882436E-1
                                        + s_c
                                            * (3.53071863813401531E-1
                                                + s_c
                                                    * (6.95901684131758475E-2
                                                        + 1.44197580643890011E-2 * s_c))))))
    }
}

// Core implied volatility solver using Let's Be Rational algorithm
fn lets_be_rational(beta: f64, thetax: f64, n: i32) -> f64 {
    if beta <= 0.0 {
        return implied_volatility_output(
            0,
            if beta == 0.0 {
                0.0
            } else {
                VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC
            },
        );
    }

    // Special case for exact ATM (thetax = 0)
    if thetax == 0.0 {
        // b_atm(s) = 1-2·Φ(-s/2) = 2·Φ(s/2)-1 = erf(s/√8)
        // So s = 2√2 · erfinv(β)
        return implied_volatility_output(0, 2.0 * SQRT_TWO * erfinv(beta));
    }

    let b_max = (0.5 * thetax).exp();
    if beta >= b_max {
        return implied_volatility_output(0, VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM);
    }

    let mut iterations = 0;
    let mut s: f64;
    let mut ds: f64;
    let mut s_left = f64::MIN_POSITIVE;
    let mut s_right = f64::MAX;
    let mut is_lowest_branch = false;

    // Critical point calculations
    let sqrt_ax = (-thetax).sqrt();
    let s_c = SQRT_TWO * sqrt_ax;
    let ome = one_minus_erfcx(sqrt_ax);
    let b_c = 0.5 * b_max * ome;

    if beta < b_c {
        // LOWER HALF: s < s_c

        let s_l = s_c - SQRT_PI_OVER_TWO * ome;
        let b_l = b_l_over_b_max(s_c) * b_max;

        let x_over_s_c_square_plus_0p25 = -0.5 * thetax + 0.25 * s_c * s_c;
        let _x_over_s_c_square_plus_0p125 = x_over_s_c_square_plus_0p25 - 0.125 * s_c * s_c;

        // Check if we're in the lowest branch
        if beta < b_l {
            // LOWEST BRANCH: Use f_lower_map approach for numerical stability
            is_lowest_branch = true;
            s_right = s_l;

            let (f_lower_map_l, d_f_lower_map_l_d_beta, d2_f_lower_map_l_d_beta2) =
                compute_f_lower_map_and_first_two_derivatives(thetax, s_l);

            let r_ll =
                convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
                    0.0,
                    b_l,
                    0.0,
                    f_lower_map_l,
                    1.0,
                    d_f_lower_map_l_d_beta,
                    d2_f_lower_map_l_d_beta2,
                    true,
                );

            let mut f = rational_cubic_interpolation(
                beta,
                0.0,
                b_l,
                0.0,
                f_lower_map_l,
                1.0,
                d_f_lower_map_l_d_beta,
                r_ll,
            );

            if !(f > 0.0) {
                // Fallback to quadratic interpolation
                let t = beta / b_l;
                f = (f_lower_map_l * t + b_l * (1.0 - t)) * t;
            }

            s = inverse_f_lower_map(thetax, f);

            // Ensure s is reasonable
            if s <= 0.0 || s.is_infinite() || s.is_nan() {
                // Fallback to a reasonable guess
                s = s_l * 0.5;
            }
        } else {
            // Lower-middle branch
            let inv_v_c = SQRT_TWO_PI / b_max;
            let inv_v_l = inv_normalised_vega(thetax, s_l);
            let r_lm =
                convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
                    b_l, b_c, s_l, s_c, inv_v_l, inv_v_c, 0.0, false,
                );
            s_left = s_l;
            s_right = s_c;
            s = rational_cubic_interpolation(beta, b_l, b_c, s_l, s_c, inv_v_l, inv_v_c, r_lm);
        }

        // Ensure initial guess is positive
        if s <= 0.0 {
            s = 0.5 * (s_l + s_c);
        }
    } else {
        // UPPER HALF: s >= s_c
        let v_c = b_max / SQRT_TWO_PI;

        let s_h = if v_c <= f64::MIN_POSITIVE {
            s_c + (-thetax).abs() * (-thetax).abs().sqrt() / v_c
        } else {
            s_c - b_c / v_c
        };

        let _b_h = normalised_black(thetax, s_h);

        let s_u = if thetax < ETA {
            let eta = if 1.0 < -thetax { 1.0 } else { -thetax };
            s_c + eta * (-thetax + square(thetax).sqrt())
        } else {
            s_h
        };

        let b_u = b_u_over_b_max(s_c) * b_max;

        if beta <= b_u {
            // Upper-middle
            let inv_v_c = SQRT_TWO_PI / b_max;
            let inv_v_u = inv_normalised_vega(thetax, s_u);
            let r_um =
                convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
                    b_c, b_u, s_c, s_u, inv_v_c, inv_v_u, 0.0, false,
                );
            s_left = s_c;
            s_right = s_u;
            s = rational_cubic_interpolation(beta, b_c, b_u, s_c, s_u, inv_v_c, inv_v_u, r_um);
        } else {
            // Upper-right
            if thetax == 0.0 {
                return implied_volatility_output(
                    0,
                    VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM,
                );
            }

            let f_upper_map = |x: f64, s: f64| -> f64 { normalised_black(x, s) };

            let d_f_upper_map_d_s = |x: f64, s: f64| -> f64 { normalised_vega(x, s) };

            let d2_f_upper_map_d_s2 = |x: f64, s: f64| -> f64 { normalised_volga(x, s) };

            let f_u = f_upper_map(thetax, s_u);
            let d_u = d_f_upper_map_d_s(thetax, s_u);
            let d2_u = d2_f_upper_map_d_s2(thetax, s_u) / 2.0;

            s_left = s_u;
            if d2_u > 0.0 && d_u.powf(2.0) + d2_u * (beta - f_u) >= 0.0 {
                s = s_u + (beta - f_u) / (d_u + (d_u.powf(2.0) + d2_u * (beta - f_u)).sqrt());
            } else {
                let f_h = f_upper_map(thetax, s_h);
                let d_h = d_f_upper_map_d_s(thetax, s_h);
                s = rational_cubic_interpolation(beta, f_h, f_u, s_h, s_u, d_h, d_u, 0.0);
            }
        }
    }

    // Householder refinement iterations
    for _iteration in 0..n {
        if !(s > 0.0) {
            break;
        }

        let b = normalised_black(thetax, s);
        let v = normalised_vega(thetax, s);

        // For LOWEST BRANCH: also compute scaled version to get bpob = 1/bx
        let (bx_scaled, _ln_vega) = if is_lowest_branch {
            // Correctly compute scaled normalised black like C++
            let ln_vega = (normalised_vega(thetax, s)).ln();
            let bx_scaled = normalised_black(thetax, s) * (-ln_vega).exp();
            (bx_scaled, ln_vega)
        } else {
            (0.0, 0.0) // Unused for non-lowest branch
        };

        if !(b > f64::MIN_POSITIVE && v > f64::MIN_POSITIVE) {
            // Binary search fallback
            ds = 0.5 * (s_left + s_right) - s;
        } else {
            if is_lowest_branch {
                // Special objective function for lowest branch
                // g(s) = 1/ln(b(s)) - 1/ln(beta)
                let ln_b = b.ln();
                let ln_beta = beta.ln();

                if ln_b.abs() < f64::EPSILON || ln_beta.abs() < f64::EPSILON {
                    // Fallback to binary search if logarithms are too small
                    ds = 0.5 * (s_left + s_right) - s;
                } else {
                    let h = thetax / s;
                    let x_square_over_s_cube = h * h / s;
                    let b_h_2 = x_square_over_s_cube - s / 4.0;
                    let lambda = 1.0 / ln_b;
                    let bpob = 1.0 / bx_scaled;

                    // Correct C++ formula for nu in LOWEST BRANCH
                    let nu = (ln_beta - ln_b) * ln_b / ln_beta / bpob;

                    let otl = 1.0 + 2.0 * lambda;
                    let h_2 = b_h_2 - bpob * otl;
                    let c = 3.0 * (x_square_over_s_cube / s); // = 3 * (h/s)²
                    let b_h_3 = b_h_2 * b_h_2 - c - 0.25;
                    let sq_bpob = bpob * bpob;
                    let bppob = b_h_2 * bpob;
                    let mu = 6.0 * lambda * (1.0 + lambda);
                    let h_3 = b_h_3 + sq_bpob * (2.0 + mu) - bppob * 3.0 * otl;

                    let householder_factor_value = householder3_factor(nu, h_2, h_3);
                    ds = nu * householder_factor_value;
                }
            } else {
                // Standard Householder iteration
                let h = thetax / s;
                let nu = (beta - b) / v;
                let x_square_over_s_cube = (h * h) / s;
                let h_2 = x_square_over_s_cube - s * 0.25;
                let h_3 = h_2 * h_2 - 3.0 * (x_square_over_s_cube / s) - 0.25;

                ds = nu * householder3_factor(nu, h_2, h_3);
            }

            if b > beta && s < s_right {
                s_right = s;
            } else if b < beta && s > s_left {
                s_left = s;
            }
        }

        s += ds;
        iterations += 1;

        // Ensure s stays positive
        if s <= 0.0 {
            s = 0.5 * (s_left + s_right);
            break;
        }
    }

    implied_volatility_output(iterations, s)
}
