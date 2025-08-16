//! # Jaeckel
//!
//! A Rust port of Peter Jäckel's algorithms on http://www.jaeckel.org
//!
//! ## Let's Be Rational
//! The Rust crate is based on the latest (2024) version of the C++ reference implementation
//! The conditionally compiled features in the C++ reference implementation are not included in this crate
//! The PJ-2024-Inverse-Normal algorithm are used to inverse the normal cumulative distribution funciton and the AS241 algorighm is omitted
//!
//! ### Example
//!
//! ```rust
//! use jaeckel::{black, implied_black_volatility};
//!
//! // Calculate option price
//! let forward = 100.0;
//! let strike = 110.0;
//! let volatility = 0.2;
//! let time_to_expiry = 1.0;
//! let is_call = 1.0; // 1.0 for call, -1.0 for put
//!
//! let price = black(forward, strike, volatility, time_to_expiry, is_call);
//!
//! // Calculate implied volatility from price
//! let implied_vol = implied_black_volatility(price, forward, strike, time_to_expiry, is_call);
//! assert!((implied_vol - volatility).abs() < 1e-14);
//! ```
//！
//! ## Copyright
//!
//! Original C++ implementation Copyright © 2013-2023 Peter Jäckel.
//!
//! Permission to use, copy, modify, and distribute this software is freely granted,
//! provided that this notice is preserved.

#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

// Internal modules - not exposed in public API
mod erf_cody;
mod lets_be_rational;
mod normaldistribution;
mod rationalcubic;

/// Error function using Cody's algorithm.
///
/// Computes erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// The error function value
pub use crate::erf_cody::erf_cody;

/// Complementary error function using Cody's algorithm.
///
/// Computes erfc(x) = 1 - erf(x) = (2/√π) ∫ₓ^∞ e^(-t²) dt
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// The complementary error function value
pub use crate::erf_cody::erfc_cody;

/// Scaled complementary error function using Cody's algorithm.
///
/// Computes erfcx(x) = e^(x²) × erfc(x)
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// The scaled complementary error function value
pub use crate::erf_cody::erfcx_cody;

/// Inverse error function.
///
/// Computes the inverse of erf(x), i.e., finds y such that erf(y) = x
///
/// # Arguments
/// * `x` - Input value in range (-1, 1)
///
/// # Returns
/// The inverse error function value
///
/// # Panics
/// Panics if |x| >= 1
pub use crate::normaldistribution::erfinv;

/// Inverse cumulative normal distribution function.
///
/// Computes the inverse of the standard normal CDF, also known as the quantile function
/// or probit function.
///
/// # Arguments
/// * `u` - Probability value in range (0, 1)
///
/// # Returns
/// The value x such that P(X ≤ x) = u, where X ~ N(0,1)
pub use crate::normaldistribution::inverse_norm_cdf;

/// Cumulative normal distribution function.
///
/// Computes P(X ≤ x) where X ~ N(0,1)
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// The cumulative probability
pub use crate::normaldistribution::norm_cdf;

/// Standard normal probability density function.
///
/// Computes φ(x) = (1/√(2π)) × e^(-x²/2)
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// The probability density value
pub use crate::normaldistribution::norm_pdf;

/// Rational cubic interpolation function.
///
/// Performs monotone convex interpolation using rational cubic splines.
/// This is used internally for implied volatility calculations.
///
/// # Arguments
/// * `x` - Evaluation point
/// * `x_l` - Left node x-coordinate
/// * `x_r` - Right node x-coordinate
/// * `y_l` - Left node y-coordinate
/// * `y_r` - Right node y-coordinate
/// * `d_l` - Left node derivative
/// * `d_r` - Right node derivative
/// * `r` - Control parameter for shape
///
/// # Returns
/// Interpolated value at x
pub use crate::rationalcubic::rational_cubic_interpolation;

/// Calculate control parameter to fit second derivative at left side.
///
/// # Arguments
/// * `x_l` - Left node x-coordinate
/// * `x_r` - Right node x-coordinate
/// * `y_l` - Left node y-coordinate
/// * `y_r` - Right node y-coordinate
/// * `d_l` - Left node derivative
/// * `d_r` - Right node derivative
/// * `second_derivative_l` - Target second derivative at left side
///
/// # Returns
/// Control parameter r for rational cubic interpolation
pub use crate::rationalcubic::rational_cubic_control_parameter_to_fit_second_derivative_at_left_side;

/// Calculate control parameter to fit second derivative at right side.
///
/// # Arguments
/// * `x_l` - Left node x-coordinate
/// * `x_r` - Right node x-coordinate
/// * `y_l` - Left node y-coordinate
/// * `y_r` - Right node y-coordinate
/// * `d_l` - Left node derivative
/// * `d_r` - Right node derivative
/// * `second_derivative_r` - Target second derivative at right side
///
/// # Returns
/// Control parameter r for rational cubic interpolation
pub use crate::rationalcubic::rational_cubic_control_parameter_to_fit_second_derivative_at_right_side;

/// Calculate minimum rational cubic control parameter.
///
/// # Arguments
/// * `d_l` - Left derivative
/// * `d_r` - Right derivative
/// * `s` - Slope between nodes
/// * `prefer_shape_preservation_over_smoothness` - Prioritize shape preservation
///
/// # Returns
/// Minimum control parameter to preserve desired shape properties
pub use crate::rationalcubic::minimum_rational_cubic_control_parameter;

/// Calculate convex rational cubic control parameter to fit second derivative at left side.
///
/// # Arguments
/// * `x_l` - Left node x-coordinate
/// * `x_r` - Right node x-coordinate
/// * `y_l` - Left node y-coordinate
/// * `y_r` - Right node y-coordinate
/// * `d_l` - Left node derivative
/// * `d_r` - Right node derivative
/// * `second_derivative_l` - Target second derivative at left side
/// * `prefer_shape_preservation_over_smoothness` - Prioritize shape preservation
///
/// # Returns
/// Control parameter ensuring both second derivative fit and convexity
pub use crate::rationalcubic::convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side;

/// Calculate convex rational cubic control parameter to fit second derivative at right side.
///
/// # Arguments
/// * `x_l` - Left node x-coordinate
/// * `x_r` - Right node x-coordinate
/// * `y_l` - Left node y-coordinate
/// * `y_r` - Right node y-coordinate
/// * `d_l` - Left node derivative
/// * `d_r` - Right node derivative
/// * `second_derivative_r` - Target second derivative at right side
/// * `prefer_shape_preservation_over_smoothness` - Prioritize shape preservation
///
/// # Returns
/// Control parameter ensuring both second derivative fit and convexity
pub use crate::rationalcubic::convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side;

/// Calculate Black-Scholes option price.
///
/// Computes the undiscounted Black-Scholes price for a European option.
///
/// # Arguments
/// * `forward` - Forward price of the underlying asset
/// * `strike` - Strike price of the option
/// * `volatility` - Implied volatility (annualized)
/// * `time` - Time to expiration in years
/// * `theta` - Option type: 1.0 for call, -1.0 for put
///
/// # Returns
/// The undiscounted option price
///
/// # Example
/// ```
/// use jaeckel::black;
///
/// let call_price = black(100.0, 110.0, 0.2, 1.0, 1.0);
/// let put_price = black(100.0, 110.0, 0.2, 1.0, -1.0);
/// ```
pub use crate::lets_be_rational::Black as black;

/// Calculate Black-Scholes vega (price sensitivity to volatility).
///
/// Computes ∂Price/∂σ for the Black-Scholes formula.
///
/// # Arguments
/// * `forward` - Forward price of the underlying asset
/// * `strike` - Strike price of the option
/// * `volatility` - Implied volatility (annualized)
/// * `time` - Time to expiration in years
///
/// # Returns
/// Vega (always positive for both calls and puts)
pub use crate::lets_be_rational::Vega as vega;

/// Calculate Black-Scholes volga (second derivative with respect to volatility).
///
/// Computes ∂²Price/∂σ² for the Black-Scholes formula.
///
/// # Arguments
/// * `forward` - Forward price of the underlying asset
/// * `strike` - Strike price of the option
/// * `volatility` - Implied volatility (annualized)
/// * `time` - Time to expiration in years
///
/// # Returns
/// Volga (convexity of price with respect to volatility)
pub use crate::lets_be_rational::Volga as volga;

/// Calculate implied volatility from Black-Scholes price.
///
/// Uses Peter Jäckel's "Let's Be Rational" algorithm for extremely accurate
/// and fast implied volatility calculations.
///
/// # Arguments
/// * `price` - Option price (undiscounted)
/// * `forward` - Forward price of the underlying asset
/// * `strike` - Strike price of the option
/// * `time` - Time to expiration in years
/// * `theta` - Option type: 1.0 for call, -1.0 for put
///
/// # Returns
/// Implied volatility. Special values:
/// - Returns `f64::MAX` if price is above maximum possible
/// - Returns `-f64::MAX` if price is below intrinsic value
/// - Returns 0.0 if price equals intrinsic value (no time value)
///
/// # Example
/// ```
/// use jaeckel::{black, implied_black_volatility};
///
/// let price = black(100.0, 110.0, 0.2, 1.0, 1.0);
/// let iv = implied_black_volatility(price, 100.0, 110.0, 1.0, 1.0);
/// ```
pub use crate::lets_be_rational::ImpliedBlackVolatility as implied_black_volatility;

/// Calculate normalised Black-Scholes price.
///
/// Computes the Black-Scholes price in normalised form where the forward
/// is 1 and moneyness is expressed as x = ln(F/K).
///
/// # Arguments
/// * `x` - Log-moneyness: ln(forward/strike)
/// * `s` - Total volatility: σ√t
///
/// # Returns
/// Normalised option price
pub use crate::lets_be_rational::NormalisedBlack as normalised_black;

/// Calculate normalised vega.
///
/// Computes vega in normalised form.
///
/// # Arguments
/// * `x` - Log-moneyness: ln(forward/strike)
/// * `s` - Total volatility: σ√t
///
/// # Returns
/// Normalised vega
pub use crate::lets_be_rational::NormalisedVega as normalised_vega;

/// Calculate normalised volga.
///
/// Computes volga in normalised form.
///
/// # Arguments
/// * `x` - Log-moneyness: ln(forward/strike)
/// * `s` - Total volatility: σ√t
///
/// # Returns
/// Normalised volga
pub use crate::lets_be_rational::NormalisedVolga as normalised_volga;

/// Calculate implied volatility from normalised Black-Scholes price.
///
/// # Arguments
/// * `beta` - Normalised option price
/// * `x` - Log-moneyness: ln(forward/strike)
/// * `theta` - Option type: 1.0 for call, -1.0 for put
///
/// # Returns
/// Total volatility s = σ√t
pub use crate::lets_be_rational::NormalisedImpliedBlackVolatility as normalised_implied_black_volatility;

/// Householder's third-order convergence factor for root finding.
///
/// # Arguments
/// * `nu` - Newton step: -f/f'
/// * `h_2` - Second derivative ratio: f''/f'
/// * `h_3` - Third derivative ratio: f'''/f'
///
/// # Returns
/// Householder convergence factor
pub use crate::lets_be_rational::householder3_factor;

/// Householder's fourth-order convergence factor for root finding.
///
/// # Arguments
/// * `nu` - Newton step: -f/f'
/// * `h_2` - Second derivative ratio: f''/f'
/// * `h_3` - Third derivative ratio: f'''/f'
/// * `h_4` - Fourth derivative ratio: f''''/f'
///
/// # Returns
/// Householder convergence factor
pub use crate::lets_be_rational::householder4_factor;
