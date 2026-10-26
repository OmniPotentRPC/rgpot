// MIT License
// Copyright 2023--present rgpot developers

//! Unshifted cluster pair kernels.
//!
//! The unit Lennard-Jones branch is the Wales and Doye expression tree.
//! Callers that inline these functions do not cross a library boundary
//! per pair. The same tree is in `rgpot_lj_pair`. Gradients are
//! derivatives of the energy.

/// Unit Lennard-Jones tree. `r2` must be finite and positive.
///
/// Inverse powers of `r^2`, energy `4 (r^-12 - r^-6)`, and
/// `(dV/dr) / r = -24 r^-2 (2 r^-12 - r^-6)`.
#[inline(always)]
pub fn lj_unit(r2: f64) -> (f64, f64, f64) {
    let inv2 = 1.0 / r2;
    let inv6 = inv2 * inv2 * inv2;
    let inv12 = inv6 * inv6;
    let energy = 4.0 * (inv12 - inv6);
    let coef = -24.0 * inv2 * (2.0 * inv12 - inv6);
    let curv = inv2 * (624.0 * inv12 - 168.0 * inv6);
    (energy, coef, curv)
}

/// Pair energy, `(dV/dr) / r`, and `d2V/dr2`.
///
/// `r2 <= 0` returns zeros. `epsilon == 1` and `sigma == 1` takes the
/// unit branch.
#[inline(always)]
pub fn lj_pair(epsilon: f64, sigma: f64, r2: f64) -> (f64, f64, f64) {
    if !(r2 > 0.0) || !r2.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    if epsilon == 1.0 && sigma == 1.0 {
        return lj_unit(r2);
    }
    let inv2 = 1.0 / r2;
    let scaled = (sigma * sigma) * inv2;
    let s6 = scaled * scaled * scaled;
    let s12 = s6 * s6;
    let energy = 4.0 * epsilon * (s12 - s6);
    let coef = -24.0 * epsilon * inv2 * (2.0 * s12 - s6);
    let curv = epsilon * inv2 * (624.0 * s12 - 168.0 * s6);
    (energy, coef, curv)
}

/// Energy, `(dV/dr) / r`, and `d2V/dr2` for a list of squared distances.
#[inline(always)]
pub fn lj_curvature_pairs(epsilon: f64, sigma: f64, r2: &[f64], energy: &mut [f64], coef: &mut [f64], curv: &mut [f64]) {
    let n = r2.len().min(energy.len()).min(coef.len()).min(curv.len());
    for i in 0..n {
        let (v, c, k) = lj_pair(epsilon, sigma, r2[i]);
        energy[i] = v;
        coef[i] = c;
        curv[i] = k;
    }
}

fn inside(r2: f64, cutoff: f64) -> bool {
    if !(r2 > 0.0) {
        return false;
    }
    if !(cutoff > 0.0) {
        return true;
    }
    r2 < cutoff * cutoff
}

/// Whole-configuration energy and gradient. `cutoff <= 0` keeps every pair.
#[inline(always)]
pub fn lj_gradient(epsilon: f64, sigma: f64, cutoff: f64, positions: &[f64], gradient: &mut [f64]) -> f64 {
    let n = positions.len() / 3;
    for g in gradient.iter_mut().take(3 * n) {
        *g = 0.0;
    }
    let mut energy = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            let mut r2 = 0.0;
            for k in 0..3 {
                d[k] = positions[3 * i + k] - positions[3 * j + k];
                r2 += d[k] * d[k];
            }
            if !inside(r2, cutoff) {
                continue;
            }
            let (v, coef, _) = lj_pair(epsilon, sigma, r2);
            energy += v;
            for k in 0..3 {
                let term = coef * d[k];
                gradient[3 * i + k] += term;
                gradient[3 * j + k] -= term;
            }
        }
    }
    energy
}

/// Pair energy added to both atoms. The total is half the sum.
#[inline(always)]
pub fn lj_atom_energies(epsilon: f64, sigma: f64, positions: &[f64], atom_energy: &mut [f64]) {
    let n = positions.len() / 3;
    for slot in atom_energy.iter_mut().take(n) {
        *slot = 0.0;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = positions[3 * i + k] - positions[3 * j + k];
                r2 += d * d;
            }
            if !(r2 > 0.0) {
                continue;
            }
            let scaled = (sigma * sigma) / r2;
            let s6 = scaled * scaled * scaled;
            let v = 4.0 * epsilon * (s6 * s6 - s6);
            atom_energy[i] += v;
            atom_energy[j] += v;
        }
    }
}

/// Gradient on the moved atoms only.
#[inline(always)]
pub fn lj_partial(epsilon: f64, sigma: f64, positions: &[f64], moved: &[usize], gradient: &mut [f64]) {
    let n = positions.len() / 3;
    for g in gradient.iter_mut().take(3 * n) {
        *g = 0.0;
    }
    for &i in moved {
        for j in 0..n {
            if j == i {
                continue;
            }
            let mut d = [0.0; 3];
            let mut r2 = 0.0;
            for k in 0..3 {
                d[k] = positions[3 * i + k] - positions[3 * j + k];
                r2 += d[k] * d[k];
            }
            if !(r2 > 0.0) {
                continue;
            }
            let (_, coef, _) = lj_pair(epsilon, sigma, r2);
            for k in 0..3 {
                gradient[3 * i + k] += coef * d[k];
            }
        }
    }
}

/// Arithmetic-mean diameters. Energy is `4 * sum (s^12 - s^6)`.
#[inline(always)]
pub fn lj_arithmetic(sigma: &[f64], cutoff: f64, positions: &[f64], gradient: &mut [f64]) -> f64 {
    let n = positions.len() / 3;
    for g in gradient.iter_mut().take(3 * n) {
        *g = 0.0;
    }
    let mut acc = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            let mut r2 = 0.0;
            for k in 0..3 {
                d[k] = positions[3 * i + k] - positions[3 * j + k];
                r2 += d[k] * d[k];
            }
            if !inside(r2, cutoff) {
                continue;
            }
            let s = 0.5 * (sigma[i] + sigma[j]);
            let s2 = s * s;
            let sr2 = s2 / r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            acc += sr12 - sr6;
            let slope = 24.0 * (sr6 - 2.0 * sr12) / r2;
            for k in 0..3 {
                let term = slope * d[k];
                gradient[3 * i + k] += term;
                gradient[3 * j + k] -= term;
            }
        }
    }
    4.0 * acc
}

/// Unit Morse tree at `epsilon = 1` and `re = 1`. `r2` must be positive.
#[inline(always)]
pub fn morse_unit(rho: f64, r2: f64) -> (f64, f64, f64) {
    let r = r2.sqrt();
    let a = (rho * (1.0 - r)).exp();
    let energy = a * (a - 2.0);
    let coef = 2.0 * rho * (a - a * a) / r;
    let curv = 2.0 * rho * rho * (2.0 * a * a - a);
    (energy, coef, curv)
}

/// Morse pair. `epsilon == 1` and `re == 1` takes the Doye branch.
#[inline(always)]
pub fn morse_pair(epsilon: f64, rho: f64, re: f64, r2: f64) -> (f64, f64, f64) {
    if !(r2 > 0.0) || !(re > 0.0) {
        return (0.0, 0.0, 0.0);
    }
    if epsilon == 1.0 && re == 1.0 {
        return morse_unit(rho, r2);
    }
    let r = r2.sqrt();
    let b = (rho * (1.0 - r / re)).exp();
    let energy = epsilon * b * (b - 2.0);
    let coef = epsilon * 2.0 * rho / re * b * (1.0 - b) / r;
    let db = -b * rho / re;
    let d2b = b * (rho / re) * (rho / re);
    let curv = epsilon * (2.0 * db * db + 2.0 * (b - 1.0) * d2b);
    (energy, coef, curv)
}

/// Morse cluster energy and gradient.
#[inline(always)]
pub fn morse_gradient(epsilon: f64, rho: f64, re: f64, cutoff: f64, positions: &[f64], gradient: &mut [f64]) -> f64 {
    let n = positions.len() / 3;
    for g in gradient.iter_mut().take(3 * n) {
        *g = 0.0;
    }
    let mut energy = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            let mut r2 = 0.0;
            for k in 0..3 {
                d[k] = positions[3 * i + k] - positions[3 * j + k];
                r2 += d[k] * d[k];
            }
            if !inside(r2, cutoff) {
                continue;
            }
            let (v, coef, _) = morse_pair(epsilon, rho, re, r2);
            energy += v;
            for k in 0..3 {
                let term = coef * d[k];
                gradient[3 * i + k] += term;
                gradient[3 * j + k] -= term;
            }
        }
    }
    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell(n: usize) -> Vec<f64> {
        let mut x = vec![0.0; 3 * n];
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let radius = (1.0 - z * z).max(0.0).sqrt();
            let theta = golden * i as f64;
            let shell = 2.5;
            x[3 * i] = shell * radius * theta.cos();
            x[3 * i + 1] = shell * radius * theta.sin();
            x[3 * i + 2] = shell * z;
        }
        x
    }

    #[test]
    fn unit_lj38_shell_matches_the_anneal_bits() {
        let x = shell(38);
        let mut g = vec![0.0; x.len()];
        let energy = lj_gradient(1.0, 1.0, 0.0, &x, &mut g);
        assert_eq!(energy.to_bits(), 0xc043_0d64_cf22_d516);
        let bits = g.iter().fold(0u64, |acc, value| {
            acc.wrapping_mul(0x9E37_79B1_85EB_CA87)
                .wrapping_add(value.to_bits())
        });
        assert_eq!(bits, 0x5294_11a1_48ae_0660);
        let mut pair_energy = 0.0;
        for i in 0..38 {
            for j in (i + 1)..38 {
                let mut r2 = 0.0;
                for k in 0..3 {
                    let d = x[3 * i + k] - x[3 * j + k];
                    r2 += d * d;
                }
                pair_energy += lj_pair(1.0, 1.0, r2).0;
            }
        }
        assert_eq!(pair_energy.to_bits(), energy.to_bits());
    }

    #[test]
    fn dimer_at_one_is_zero_with_gradient_24() {
        let x = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut g = [0.0; 6];
        let energy = lj_gradient(1.0, 1.0, 0.0, &x, &mut g);
        assert_eq!(energy, 0.0);
        assert_eq!(g[0], 24.0);
        assert_eq!(g[3], -24.0);
    }
}
