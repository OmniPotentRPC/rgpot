// MIT License
// Copyright 2023--present rgpot developers

//! Serial wall time of one Mishin copper force call.
//!
//! `cargo bench -p rgpot-core --bench mishin_cu -- [repeats]` prints one
//! line per cell: the atom count, the repeats, and the median and minimum
//! seconds per call. The repeats default to 31 and are at least 6.

use std::hint::black_box;
use std::time::Instant;

use rgpot_core::eam::{mishin_cu, MISHIN_CU_LATTICE};

fn fcc_vacancy(n: usize, a: f64) -> (Vec<[f64; 3]>, [[f64; 3]; 3]) {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut positions = Vec::new();
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                for site in basis {
                    positions.push([
                        (ix as f64 + site[0]) * a,
                        (iy as f64 + site[1]) * a,
                        (iz as f64 + site[2]) * a,
                    ]);
                }
            }
        }
    }
    positions.remove(0);
    let length = n as f64 * a;
    (
        positions,
        [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
    )
}

fn main() {
    let repeats = std::env::args()
        .skip(1)
        .find_map(|arg| arg.parse::<usize>().ok())
        .unwrap_or(31)
        .max(6);
    println!("atoms,repeats,median_s,min_s");
    for cells in [2usize, 3, 4] {
        let (positions, cell) = fcc_vacancy(cells, MISHIN_CU_LATTICE);
        black_box(mishin_cu(black_box(&positions), black_box(cell)));
        let mut samples: Vec<f64> = (0..repeats)
            .map(|_| {
                let started = Instant::now();
                black_box(mishin_cu(black_box(&positions), black_box(cell)));
                started.elapsed().as_secs_f64()
            })
            .collect();
        samples.sort_by(f64::total_cmp);
        println!(
            "{},{},{:.9},{:.9}",
            positions.len(),
            repeats,
            samples[samples.len() / 2],
            samples[0]
        );
    }
}
