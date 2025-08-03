use criterion::{criterion_group, criterion_main, Criterion};
use jaeckel::{black, implied_black_volatility};
use std::hint::black_box;

fn benchmark_black_price(c: &mut Criterion) {
    let mut group = c.benchmark_group("Black Price");

    group.bench_function("black price", |b| {
        b.iter(|| {
            black(
                black_box(100.0), // forward
                black_box(100.0), // strike
                black_box(0.2),   // volatility
                black_box(1.0),   // time
                black_box(1.0),   // call
            )
        })
    });

    group.finish();
}

fn benchmark_implied_volatility(c: &mut Criterion) {
    let mut group = c.benchmark_group("Implied Volatility");

    let price = black(100.0, 100.0, 0.2, 1.0, 1.0);

    group.bench_function("implied volatility", |b| {
        b.iter(|| {
            implied_black_volatility(
                black_box(price),
                black_box(100.0), // forward
                black_box(100.0), // strike
                black_box(1.0),   // time
                black_box(1.0),   // call
            )
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_black_price, benchmark_implied_volatility,);
criterion_main!(benches);
