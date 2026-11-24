// MIT License
// Copyright 2023--present rgpot developers

use std::sync::Mutex;

use crate::potential::PotentialStressCallback;

/// Per-handle stress state, kept in the handle's opaque cache so the C
/// layout of `rgpot_potential_t` does not change.
#[derive(Default)]
pub(crate) struct StressState {
    callback: Mutex<Option<PotentialStressCallback>>,
}

impl StressState {
    pub(crate) fn set(&self, callback: Option<PotentialStressCallback>) {
        *self.callback.lock().unwrap_or_else(|p| p.into_inner()) = callback;
    }

    pub(crate) fn get(&self) -> Option<PotentialStressCallback> {
        *self.callback.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// Opt-in pair virial accumulator for Rust-native potential callbacks.
///
/// The six entries are `dE/dε` in `xx, yy, zz, yz, xz, xy` order. Callers
/// publish only after every enabled energy term has contributed.
pub(crate) struct StressAccumulator {
    enabled: bool,
    strain: [f64; 6],
}

impl StressAccumulator {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            strain: [0.0; 6],
        }
    }

    /// Add a central pair whose force on i is `fscale * d`, `d = r_i - r_j`.
    pub(crate) fn pair_force(&mut self, fscale: f64, d: [f64; 3]) {
        if !self.enabled {
            return;
        }
        let c = -fscale;
        self.strain[0] += c * d[0] * d[0];
        self.strain[1] += c * d[1] * d[1];
        self.strain[2] += c * d[2] * d[2];
        self.strain[3] += c * d[1] * d[2];
        self.strain[4] += c * d[0] * d[2];
        self.strain[5] += c * d[0] * d[1];
    }

    pub(crate) fn publish(self, cell: &[f64; 9]) -> Option<[f64; 9]> {
        if !self.enabled {
            return None;
        }
        let volume = (cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
            - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
            + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]))
            .abs();
        if volume.is_nan() || volume <= 0.0 {
            return None;
        }
        let inv = 1.0 / volume;
        let [xx, yy, zz, yz, xz, xy] = self.strain.map(|v| v * inv);
        Some([xx, xy, xz, xy, yy, yz, xz, yz, zz])
    }
}
