use jaeckel::{black, implied_black_volatility};

#[test]
fn test_black_scholes_implied_volatility_round_trip() {
    // Test that Black-Scholes price and implied volatility are inverses of each other
    // This verifies both functions work correctly together across various scenarios
    let test_cases = [
        // (forward, strike, volatility, time, call/put, description)
        // ATM cases
        (100.0, 100.0, 0.2, 1.0, 1.0, "ATM standard"),
        (50.0, 50.0, 0.3, 0.5, 1.0, "ATM low price"),
        (1000.0, 1000.0, 0.15, 2.0, 1.0, "ATM high price"),
        // OTM call cases
        (100.0, 110.0, 0.2, 1.0, 1.0, "10% OTM call"),
        (100.0, 150.0, 0.2, 1.0, 1.0, "50% OTM call"),
        (100.0, 200.0, 0.2, 1.0, 1.0, "100% OTM call (deep)"),
        (100.0, 500.0, 0.3, 1.0, 1.0, "400% OTM call (very deep)"),
        // ITM call cases
        (100.0, 90.0, 0.2, 1.0, 1.0, "10% ITM call"),
        (100.0, 50.0, 0.2, 1.0, 1.0, "50% ITM call"),
        (100.0, 20.0, 0.2, 1.0, 1.0, "80% ITM call (deep)"),
        (100.0, 5.0, 0.3, 1.0, 1.0, "95% ITM call (very deep)"),
        // Put cases
        (100.0, 100.0, 0.2, 1.0, -1.0, "ATM put"),
        (100.0, 110.0, 0.2, 1.0, -1.0, "ITM put"),
        (100.0, 90.0, 0.2, 1.0, -1.0, "OTM put"),
        // Different volatility levels
        (100.0, 100.0, 0.00001, 1.0, 1.0, "Very low vol"),
        (100.0, 100.0, 0.001, 1.0, 1.0, "Low vol"),
        (100.0, 100.0, 5.0, 1.0, 1.0, "High vol"),
        (100.0, 100.0, 10.0, 1.0, 1.0, "Very high vol"),
        // Different time to maturity
        (100.0, 100.0, 0.2, 0.001, 1.0, "Very short maturity"),
        (100.0, 100.0, 0.2, 0.01, 1.0, "Short maturity"),
        (100.0, 100.0, 0.2, 5.0, 1.0, "Long maturity"),
        (100.0, 100.0, 0.2, 10.0, 1.0, "Very long maturity"),
        // Edge cases with extreme strikes
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
        // Mixed scenarios
        (80.0, 120.0, 0.35, 0.25, 1.0, "OTM short maturity high vol"),
        (150.0, 100.0, 0.15, 3.0, 1.0, "ITM long maturity low vol"),
    ];

    let mut max_error: f64 = 0.0;

    for (f, k, sigma, t, theta, _description) in test_cases {
        let price = black(f, k, sigma, t, theta);
        let implied_vol = implied_black_volatility(price, f, k, t, theta);

        //For zero implied volatility, check if price equals intrinsic
        if implied_vol == 0.0 {
            let intrinsic = if theta > 0.0 {
                f64::max(f - k, 0.0)
            } else {
                f64::max(k - f, 0.0)
            };
            if (price - intrinsic).abs() < 1e-12 {
                // Expected behavior when price equals intrinsic value
                continue;
            }
        }

        let relative_error = (implied_vol - sigma).abs() / sigma;
        max_error = max_error.max(relative_error);
    }

    // Assert that we achieve machine precision for all computable cases
    assert!(
        max_error < 1e-10,
        "Maximum relative error {:.6e} exceeds tolerance",
        max_error
    );
}
