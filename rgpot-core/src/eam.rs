// MIT License
// Copyright 2023--present rgpot developers

//! Analytic embedded-atom potential for copper.
//!
//! [`mishin_cu`] is the EAM1 copper of Mishin, Mehl, Papaconstantopoulos,
//! Voter and Kress, Phys. Rev. B 63, 224106 (2001), evaluated from its
//! closed forms rather than a tabulated `setfl` file. It returns the energy,
//! the forces and the Cauchy stress of a periodic cell whose lattice vectors
//! are the rows of `cell`, in Å and eV.
//!
//! Every pair inside the cutoff is listed once over all periodic images, so
//! a cell shorter than the cutoff also sees the images of each atom. The
//! cost is quadratic in the atom count, which suits cells of a few hundred
//! atoms.
//!
//! [`mishin_cu_potential`] wraps the same evaluation as a
//! [`PotentialImpl`] callback, for callers that speak the rgpot tensors.

use std::os::raw::c_void;

use crate::potential::PotentialImpl;
use crate::status::rgpot_status_t;
use crate::tensor::{rgpot_tensor_data, rgpot_tensor_owned_cpu_f64_2d};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Cutoff radius, Å.
pub const MISHIN_CU_CUTOFF: f64 = 5.50679;
/// Zero-temperature fcc lattice constant of the potential, Å.
pub const MISHIN_CU_LATTICE: f64 = 3.615;

const H_CUT: f64 = 0.50037;
const E1: f64 = 2.01458e2;
const E2: f64 = 6.59288e-3;
const R01: f64 = 0.83591;
const R02: f64 = 4.46867;
const ALPHA1: f64 = 2.97758;
const ALPHA2: f64 = 1.54927;
const DELTA: f64 = 0.86225e-2;
const RS: [f64; 3] = [2.24000, 1.80000, 1.20000];
const S_REP: [f64; 3] = [4.0, 40.0, 1.15000e3];
const DENSITY_A: f64 = 3.80362;
const R03: f64 = -2.19885;
const R04: f64 = -2.61984e2;
const BETA1: f64 = 0.17394;
const BETA2: f64 = 5.35661e2;
const F0: f64 = -2.28235;
const F2: f64 = 1.35535;
const Q1: f64 = -1.27775;
const Q2: f64 = -0.86074;
const Q3: f64 = 1.78804;
const Q4: f64 = 2.97571;
const EMB_Q1: f64 = 0.40000;
const EMB_Q2: f64 = 0.30000;

/// Energy, forces and stress of one copper cell.
#[derive(Clone, Debug, PartialEq)]
pub struct EamEvaluation {
    /// Potential energy, eV.
    pub energy: f64,
    /// Force on each atom, eV/Å.
    pub forces: Vec<[f64; 3]>,
    /// Cauchy stress `σ = (1/V) ∂E/∂ε`, eV/Å³.
    pub stress: [[f64; 3]; 3],
}

struct Bond {
    i: usize,
    j: usize,
    d: [f64; 3],
    r: f64,
}

/// Evaluate the Mishin copper EAM1 on a periodic cell.
///
/// `positions` are Cartesian, in Å. `cell` holds the lattice vectors as
/// rows, in Å. A cell with zero volume gives a non-finite stress.
pub fn mishin_cu(positions: &[[f64; 3]], cell: [[f64; 3]; 3]) -> EamEvaluation {
    let n = positions.len();
    let lengths = [0, 1, 2].map(|row| {
        (cell[row][0] * cell[row][0] + cell[row][1] * cell[row][1] + cell[row][2] * cell[row][2])
            .sqrt()
    });
    let min_len = lengths.into_iter().fold(f64::MAX, f64::min);
    let nimg = ((MISHIN_CU_CUTOFF / min_len).ceil() as i32).max(1);
    let cutoff2 = MISHIN_CU_CUTOFF * MISHIN_CU_CUTOFF;
    let mut bonds = Vec::new();
    for i in 0..n {
        for j in i..n {
            for sx in -nimg..=nimg {
                for sy in -nimg..=nimg {
                    for sz in -nimg..=nimg {
                        if i == j && !positive_image(sx, sy, sz) {
                            continue;
                        }
                        let shift = [
                            sx as f64 * cell[0][0]
                                + sy as f64 * cell[1][0]
                                + sz as f64 * cell[2][0],
                            sx as f64 * cell[0][1]
                                + sy as f64 * cell[1][1]
                                + sz as f64 * cell[2][1],
                            sx as f64 * cell[0][2]
                                + sy as f64 * cell[1][2]
                                + sz as f64 * cell[2][2],
                        ];
                        let d = [
                            positions[j][0] + shift[0] - positions[i][0],
                            positions[j][1] + shift[1] - positions[i][1],
                            positions[j][2] + shift[2] - positions[i][2],
                        ];
                        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if r2 > 1e-16 && r2 < cutoff2 {
                            bonds.push(Bond {
                                i,
                                j,
                                d,
                                r: r2.sqrt(),
                            });
                        }
                    }
                }
            }
        }
    }
    let mut rho = vec![0.0; n];
    for bond in &bonds {
        let weight = density(bond.r).0;
        rho[bond.i] += weight;
        rho[bond.j] += weight;
    }
    let mut energy = 0.0;
    let mut fp = vec![0.0; n];
    for i in 0..n {
        let (value, slope) = embed(rho[i]);
        energy += value;
        fp[i] = slope;
    }
    // The gradient is accumulated and negated at the end, so an exact zero
    // force keeps the sign it has as the negative of a gradient.
    let mut gradient = vec![[0.0; 3]; n];
    let mut virial = [[0.0; 3]; 3];
    for bond in &bonds {
        let (value, dv) = pair(bond.r);
        let (_, drho) = density(bond.r);
        let inv_r = 1.0 / bond.r;
        if bond.i != bond.j {
            energy += value;
            let deriv = dv + (fp[bond.i] + fp[bond.j]) * drho;
            for c in 0..3 {
                let force = deriv * bond.d[c] * inv_r;
                gradient[bond.i][c] -= force;
                gradient[bond.j][c] += force;
            }
            for (a, row) in virial.iter_mut().enumerate() {
                for (b, slot) in row.iter_mut().enumerate() {
                    *slot += deriv * bond.d[a] * bond.d[b] * inv_r;
                }
            }
        } else {
            energy += value;
            let deriv = dv + 2.0 * fp[bond.i] * drho;
            for (a, row) in virial.iter_mut().enumerate() {
                for (b, slot) in row.iter_mut().enumerate() {
                    *slot += deriv * bond.d[a] * bond.d[b] * inv_r;
                }
            }
        }
    }
    let volume = det(cell).abs();
    let mut stress = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            stress[a][b] = virial[a][b] / volume;
        }
    }
    let forces = gradient
        .into_iter()
        .map(|row| [-row[0], -row[1], -row[2]])
        .collect();
    EamEvaluation {
        energy,
        forces,
        stress,
    }
}

/// The Mishin copper EAM1 as an rgpot callback.
///
/// The callback reads `positions` as `[n_atoms, 3]` and `box_matrix` as a
/// row-major `3 × 3` with the lattice vectors as rows. It ignores the
/// atomic numbers; every atom is copper. The variance is zero.
pub fn mishin_cu_potential() -> PotentialImpl {
    PotentialImpl::new(mishin_cu_callback, std::ptr::null_mut(), None)
}

unsafe extern "C" fn mishin_cu_callback(
    _user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    if input.is_null() || output.is_null() {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let input = unsafe { &*input };
    let Some(n) = (unsafe { input.n_atoms() }) else {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    };
    if input.box_matrix.is_null() {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let coords = unsafe { rgpot_tensor_data(input.positions) } as *const f64;
    let box_data = unsafe { rgpot_tensor_data(input.box_matrix) } as *const f64;
    if coords.is_null() || box_data.is_null() {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let coords = unsafe { std::slice::from_raw_parts(coords, n * 3) };
    let box_data = unsafe { std::slice::from_raw_parts(box_data, 9) };
    let positions: Vec<[f64; 3]> = coords
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let cell = [
        [box_data[0], box_data[1], box_data[2]],
        [box_data[3], box_data[4], box_data[5]],
        [box_data[6], box_data[7], box_data[8]],
    ];
    let result = mishin_cu(&positions, cell);
    let flat: Vec<f64> = result.forces.iter().flatten().copied().collect();
    unsafe {
        (*output).energy = result.energy;
        (*output).variance = 0.0;
        (*output).forces = rgpot_tensor_owned_cpu_f64_2d(flat.as_ptr(), n as i64, 3);
    }
    rgpot_status_t::RGPOT_SUCCESS
}

fn cutoff_weight(r: f64) -> (f64, f64) {
    let x = (r - MISHIN_CU_CUTOFF) / H_CUT;
    if x >= 0.0 {
        return (0.0, 0.0);
    }
    let x2 = x * x;
    let x4 = x2 * x2;
    let den = 1.0 + x4;
    (x4 / den, 4.0 * x * x2 / (den * den) / H_CUT)
}

fn morse(r: f64, r0: f64, alpha: f64) -> (f64, f64) {
    let z = -alpha * (r - r0);
    let e1 = z.exp();
    let e2 = (2.0 * z).exp();
    (e2 - 2.0 * e1, 2.0 * alpha * (e1 - e2))
}

fn pair(r: f64) -> (f64, f64) {
    if r >= MISHIN_CU_CUTOFF {
        return (0.0, 0.0);
    }
    let (m1, dm1) = morse(r, R01, ALPHA1);
    let (m2, dm2) = morse(r, R02, ALPHA2);
    let (psi, dpsi) = cutoff_weight(r);
    let core = E1 * m1 + E2 * m2 + DELTA;
    let mut value = core * psi;
    let mut slope = (E1 * dm1 + E2 * dm2) * psi + core * dpsi;
    for (&rs, &strength) in RS.iter().zip(S_REP.iter()) {
        if r < rs {
            let gap = rs - r;
            let gap3 = gap * gap * gap;
            value -= strength * gap * gap3;
            slope += 4.0 * strength * gap3;
        }
    }
    (value, slope)
}

fn density(r: f64) -> (f64, f64) {
    if r >= MISHIN_CU_CUTOFF {
        return (0.0, 0.0);
    }
    let (psi, dpsi) = cutoff_weight(r);
    let d3 = r - R03;
    let g1 = (-BETA1 * d3 * d3).exp();
    let dg1 = g1 * (-2.0 * BETA1 * d3);
    let arg = -BETA2 * (r - R04);
    let (g2, dg2) = if arg < -700.0 {
        (0.0, 0.0)
    } else {
        let g2 = arg.exp();
        (g2, -BETA2 * g2)
    };
    let inner = DENSITY_A * g1 + g2;
    let dinner = DENSITY_A * dg1 + dg2;
    (inner * psi, dinner * psi + inner * dpsi)
}

fn embed(rho: f64) -> (f64, f64) {
    let x = rho - 1.0;
    let x2 = x * x;
    let x3 = x2 * x;
    if rho < 1.0 {
        let value = F0 + 0.5 * F2 * x2 + Q1 * x3 + Q2 * x2 * x2 + Q3 * x2 * x3 + Q4 * x3 * x3;
        let slope =
            F2 * x + 3.0 * Q1 * x2 + 4.0 * Q2 * x3 + 5.0 * Q3 * x2 * x2 + 6.0 * Q4 * x2 * x3;
        (value, slope)
    } else {
        let num = F0 + 0.5 * F2 * x2 + Q1 * x3 + EMB_Q1 * x2 * x2;
        let den = 1.0 + EMB_Q2 * x3;
        let dnum = F2 * x + 3.0 * Q1 * x2 + 4.0 * EMB_Q1 * x3;
        let dden = 3.0 * EMB_Q2 * x2;
        (num / den, (dnum * den - num * dden) / (den * den))
    }
}

fn positive_image(sx: i32, sy: i32, sz: i32) -> bool {
    sx > 0 || (sx == 0 && (sy > 0 || (sy == 0 && sz > 0)))
}

fn det(h: [[f64; 3]; 3]) -> f64 {
    h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{
        rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
        rgpot_tensor_free,
    };

    fn fcc(n: usize, a: f64) -> (Vec<[f64; 3]>, [[f64; 3]; 3]) {
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
        let length = n as f64 * a;
        (
            positions,
            [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
        )
    }

    /// A vacancy cell with every atom pushed off its site.
    fn rattled() -> (Vec<[f64; 3]>, [[f64; 3]; 3]) {
        let (mut positions, cell) = fcc(2, MISHIN_CU_LATTICE);
        positions.remove(0);
        for (atom, row) in positions.iter_mut().enumerate() {
            let k = atom as f64;
            row[0] += 0.03 * (1.3 * k).sin();
            row[1] += 0.03 * (2.1 * k).cos();
            row[2] -= 0.02 * (0.7 * k).sin();
        }
        (positions, cell)
    }

    #[test]
    fn perfect_crystal_has_the_mishin_cohesive_energy_and_no_force() {
        let (positions, cell) = fcc(2, MISHIN_CU_LATTICE);
        let result = mishin_cu(&positions, cell);
        let per_atom = result.energy / positions.len() as f64;
        // Mishin et al. fit E_coh = 3.54 eV at a = 3.615 Å.
        assert!((per_atom + 3.54).abs() < 5.0e-3, "{per_atom}");
        for force in &result.forces {
            for component in force {
                assert!(component.abs() < 1.0e-10, "{component}");
            }
        }
    }

    #[test]
    fn one_conventional_cell_sees_its_own_images() {
        let (small, small_cell) = fcc(1, MISHIN_CU_LATTICE);
        let (large, large_cell) = fcc(3, MISHIN_CU_LATTICE);
        let e_small = mishin_cu(&small, small_cell).energy / small.len() as f64;
        let e_large = mishin_cu(&large, large_cell).energy / large.len() as f64;
        assert!((e_small - e_large).abs() < 1.0e-12, "{e_small} {e_large}");
    }

    #[test]
    fn forces_are_the_negative_energy_gradient() {
        let (positions, cell) = rattled();
        let result = mishin_cu(&positions, cell);
        let h = 1.0e-5;
        for atom in [0, 7, 30] {
            for axis in 0..3 {
                let mut plus = positions.clone();
                let mut minus = positions.clone();
                plus[atom][axis] += h;
                minus[atom][axis] -= h;
                let slope =
                    (mishin_cu(&plus, cell).energy - mishin_cu(&minus, cell).energy) / (2.0 * h);
                let force = result.forces[atom][axis];
                assert!(
                    (force + slope).abs() < 1.0e-6,
                    "{atom} {axis}: {force} {slope}"
                );
            }
        }
    }

    #[test]
    fn forces_sum_to_zero() {
        let (positions, cell) = rattled();
        let result = mishin_cu(&positions, cell);
        for axis in 0..3 {
            let total: f64 = result.forces.iter().map(|row| row[axis]).sum();
            assert!(total.abs() < 1.0e-10, "{axis}: {total}");
        }
    }

    #[test]
    fn stress_is_the_strain_derivative_over_the_volume() {
        let (positions, cell) = rattled();
        let result = mishin_cu(&positions, cell);
        let volume = det(cell).abs();
        let h = 1.0e-6;
        let strained = |a: usize, b: usize, e: f64| {
            let mut eps = [[0.0; 3]; 3];
            eps[a][b] += 0.5 * e;
            eps[b][a] += 0.5 * e;
            let deform = |row: [f64; 3]| {
                let mut out = row;
                for (j, slot) in out.iter_mut().enumerate() {
                    *slot += row[0] * eps[0][j] + row[1] * eps[1][j] + row[2] * eps[2][j];
                }
                out
            };
            let moved: Vec<[f64; 3]> = positions.iter().map(|&row| deform(row)).collect();
            let box_rows = [deform(cell[0]), deform(cell[1]), deform(cell[2])];
            mishin_cu(&moved, box_rows).energy
        };
        for (a, b) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)] {
            let slope = (strained(a, b, h) - strained(a, b, -h)) / (2.0 * h);
            let sigma = result.stress[a][b];
            assert!(
                (sigma - slope / volume).abs() < 1.0e-6,
                "{a}{b}: {sigma} {}",
                slope / volume
            );
            let skew = (result.stress[a][b] - result.stress[b][a]).abs();
            assert!(skew <= 1.0e-15 * sigma.abs().max(1.0), "{a}{b}: {skew}");
        }
    }

    #[test]
    fn callback_matches_the_direct_evaluation_bit_for_bit() {
        let (positions, cell) = rattled();
        let direct = mishin_cu(&positions, cell);
        let pot = mishin_cu_potential();
        let n = positions.len();
        let mut flat: Vec<f64> = positions.iter().flatten().copied().collect();
        let mut numbers = vec![29; n];
        let mut box_matrix: Vec<f64> = cell.iter().flatten().copied().collect();
        let positions_tensor = unsafe { rgpot_tensor_cpu_f64_2d(flat.as_mut_ptr(), n as i64, 3) };
        let numbers_tensor = unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), n as i64) };
        let box_tensor = unsafe { rgpot_tensor_cpu_f64_matrix3(box_matrix.as_mut_ptr()) };
        let input = rgpot_force_input_t {
            positions: positions_tensor,
            atomic_numbers: numbers_tensor,
            box_matrix: box_tensor,
        };
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 1.0,
        };
        let status = unsafe { pot.calculate(&input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(output.energy.to_bits(), direct.energy.to_bits());
        assert_eq!(output.variance, 0.0);
        let data = unsafe { rgpot_tensor_data(output.forces) } as *const f64;
        let forces = unsafe { std::slice::from_raw_parts(data, n * 3) };
        for (got, want) in forces.iter().zip(direct.forces.iter().flatten()) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
        unsafe {
            rgpot_tensor_free(positions_tensor);
            rgpot_tensor_free(numbers_tensor);
            rgpot_tensor_free(box_tensor);
            rgpot_tensor_free(output.forces);
        }
    }

    #[test]
    fn callback_refuses_a_missing_box() {
        let pot = mishin_cu_potential();
        let mut flat = vec![0.0; 3];
        let positions_tensor = unsafe { rgpot_tensor_cpu_f64_2d(flat.as_mut_ptr(), 1, 3) };
        let input = rgpot_force_input_t {
            positions: positions_tensor,
            atomic_numbers: std::ptr::null_mut(),
            box_matrix: std::ptr::null_mut(),
        };
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { pot.calculate(&input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        assert!(output.forces.is_null());
        unsafe { rgpot_tensor_free(positions_tensor) };
    }
}
