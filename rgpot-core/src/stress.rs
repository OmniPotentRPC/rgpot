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

    /// Publish the Cauchy stress, or `None` when stress was not requested or
    /// the cell has no positive finite volume.
    pub(crate) fn publish(self, cell: &[f64; 9]) -> Option<[f64; 9]> {
        if !self.enabled {
            return None;
        }
        let volume = cell_volume(cell);
        if volume <= 0.0 {
            return None;
        }
        let inv = 1.0 / volume;
        let [xx, yy, zz, yz, xz, xy] = self.strain.map(|v| v * inv);
        Some([xx, xy, xz, xy, yy, yz, xz, yz, zz])
    }
}

/// Absolute cell volume of the row-major 3x3 `cell`.
///
/// Zero is the error value, as for the C++ `cellVolume`: a cell with a
/// non-finite entry and a cell whose determinant overflows both give exactly
/// `0.0`, never NaN or infinity.
pub(crate) fn cell_volume(cell: &[f64; 9]) -> f64 {
    if cell.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    let volume = (cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]))
        .abs();
    if volume.is_finite() {
        volume
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    fn accumulator() -> StressAccumulator {
        let mut acc = StressAccumulator::new(true);
        acc.pair_force(2.0, [1.0, 0.0, 0.0]);
        acc
    }

    #[test]
    fn publishes_a_symmetric_tensor_for_a_finite_cell() {
        let cell = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let stress = accumulator()
            .publish(&cell)
            .expect("stress for a unit cell");
        assert_eq!(stress[0], -2.0 / 8.0);
        assert_eq!(stress[1], stress[3]);
        assert_eq!(stress[2], stress[6]);
        assert_eq!(stress[5], stress[7]);
    }

    #[test]
    fn a_disabled_accumulator_publishes_nothing() {
        assert!(StressAccumulator::new(false).publish(&UNIT).is_none());
    }

    #[test]
    fn a_singular_cell_leaves_stress_unpublished() {
        let mut cell = UNIT;
        cell[8] = 0.0;
        assert!(accumulator().publish(&cell).is_none());
    }

    #[test]
    fn an_overflowing_volume_leaves_stress_unpublished() {
        let big = f64::MAX;
        let cell = [big, 0.0, 0.0, 0.0, big, 0.0, 0.0, 0.0, big];
        assert!(cell.iter().all(|v| v.is_finite()));
        assert_eq!(cell_volume(&cell), 0.0);
        assert!(accumulator().publish(&cell).is_none());
    }

    #[test]
    fn a_non_finite_cell_leaves_stress_unpublished() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut cell = UNIT;
            cell[1] = bad;
            assert_eq!(cell_volume(&cell), 0.0);
            assert!(accumulator().publish(&cell).is_none());
        }
    }
}
