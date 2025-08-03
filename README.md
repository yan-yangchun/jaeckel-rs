# Jaeckel

A Rust port of Peter Jäckel's algorithms on http://www.jaeckel.org

## Let's Be Rational
- The Rust crate is based on the latest (2024) version of the C++ reference implementation
- The conditionally compiled features in the C++ reference implementation are not included in this crate
- The PJ-2024-Inverse-Normal algorithm are used to inverse the normal cumulative distribution funciton and the AS241 algorighm is omitted

### Usage

```rust
use jaeckel::{black, implied_black_volatility};

// Calculate option price
let price = black(100.0, 200.0, 0.2, 1.0, 1.0);

// Calculate implied volatility
let implied_vol = implied_black_volatility(price, 100.0, 200.0, 1.0, 1.0);
```

## License

Copyright © 2013-2023 Peter Jäckel.

Permission to use, copy, modify, and distribute this software is freely granted,
provided that this notice is preserved.

