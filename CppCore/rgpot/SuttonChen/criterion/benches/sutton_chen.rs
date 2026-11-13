//! Energy and analytic gradient of the Sutton-Chen 12-6 direct kernel.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::cell::Cell;

extern "C" {
    fn rgpot_sc_bench(
        pos: *const f64,
        n: usize,
        epsilon: f64,
        c: f64,
        a: f64,
        n_exp: i32,
        m_exp: i32,
        cutoff: f64,
        forces: *mut f64,
        energy: *mut f64,
    );
}

fn cluster(n: usize) -> Vec<f64> {
    // xorshift matching a compact grid, not the C++ mt19937 geometry.
    // The timing depends on the pair count, which this grid fixes.
    let mut side = 1usize;
    while side * side * side < n {
        side += 1;
    }
    let mut pos = vec![0.0; 3 * n];
    let mut state = 12u32;
    let mut k = 0usize;
    for ix in 0..side {
        for iy in 0..side {
            for iz in 0..side {
                if k >= n {
                    return pos;
                }
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let jx = (state >> 8) as f64 / (u32::MAX as f64);
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let jy = (state >> 8) as f64 / (u32::MAX as f64);
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let jz = (state >> 8) as f64 / (u32::MAX as f64);
                pos[3 * k] = ix as f64 * 1.2 + 0.05 * jx;
                pos[3 * k + 1] = iy as f64 * 1.2 + 0.05 * jy;
                pos[3 * k + 2] = iz as f64 * 1.2 + 0.05 * jz;
                k += 1;
            }
        }
    }
    pos
}

fn bench_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sutton_chen_12_6");
    group.sample_size(30);
    for n in [13usize, 38, 80, 147] {
        let pos = cluster(n);
        let mut forces = vec![0.0; 3 * n];
        let energy = Cell::new(0.0);
        group.bench_function(format!("n{n}"), |b| {
            b.iter(|| {
                forces.fill(0.0);
                let mut e = 0.0;
                unsafe {
                    rgpot_sc_bench(
                        pos.as_ptr(),
                        n,
                        black_box(1.0),
                        black_box(144.41),
                        black_box(1.0),
                        12,
                        6,
                        40.0,
                        forces.as_mut_ptr(),
                        &mut e,
                    );
                }
                energy.set(black_box(e));
            });
        });
        let _ = energy.get();
    }
    group.finish();
}

criterion_group!(benches, bench_sizes);
criterion_main!(benches);
