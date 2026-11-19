// MIT License
// Copyright 2023--present rgpot developers

//! Tabulated EAM through the safe Rust wrapper and the C ABI, against LAMMPS.
//!
//! The references are the LAMMPS runs of `CppCore/tests/data/eam_table`:
//!
//! - `cu_u3_vacancy.ref`, `pair_style eam` on `Cu_u3.eam` for a jittered
//!   3 x 3 x 3 fcc copper cell with one atom removed, 107 atoms, written by
//!   `make_vacancy_ref.py` with LAMMPS 22 Jul 2025;
//! - `cu_u3_cubic.ref`, the same file for the whole 108 atom cell;
//! - `cuni_sheared.ref`, `pair_style eam/alloy` on `CuNi.eam.alloy` for a
//!   sheared cell, which exercises setfl and a triclinic box.
//!
//! `Cu_u3.eam` has sha256
//! `3436c491a4c75ea8b7141adbc6ee382a118f5fdb47f609c2a660fc1eb772599f`.

use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rgpot_core::c_api::eam_table::rgpot_potential_new_eam_table;
use rgpot_core::c_api::potential::{rgpot_potential_calculate, rgpot_potential_free};
use rgpot_core::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
use rgpot_core::eam_table::{
    EamTableError, EamTableFormat, EamTablePotential, RGPOT_EAM_TABLE_AUTO,
    RGPOT_EAM_TABLE_FINNIS_SINCLAIR, RGPOT_EAM_TABLE_FUNCFL, RGPOT_EAM_TABLE_SETFL,
};
use rgpot_core::status::{rgpot_last_error, rgpot_status_t};
use rgpot_core::tensor::{rgpot_tensor_data, rgpot_tensor_free};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Energy agreement with LAMMPS, eV. The largest gap of the four cells is
/// 2.6e-12.
const ENERGY_TOLERANCE: f64 = 1.0e-11;
/// Force agreement with LAMMPS, eV/Angstrom. The largest gap is 3.0e-14.
const FORCE_TOLERANCE: f64 = 1.0e-12;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../CppCore/tests/data/eam_table")
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(rgpot_last_error()) }
        .to_string_lossy()
        .into_owned()
}

struct Reference {
    cell: [f64; 9],
    energy: f64,
    z: Vec<i32>,
    positions: Vec<f64>,
    forces: Vec<f64>,
}

fn reference(name: &str) -> Reference {
    let path = data_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|why| panic!("cannot read {}: {why}", path.display()));
    let mut cell = [0.0; 9];
    let mut energy = None;
    let mut count = None;
    let mut z = Vec::new();
    let mut positions = Vec::new();
    let mut forces = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("box") => {
                for (slot, v) in cell.iter_mut().zip(fields) {
                    *slot = v.parse().expect("box value");
                }
            }
            Some("energy") => energy = Some(fields.next().expect("energy").parse().expect("f64")),
            Some("atoms") => count = Some(fields.next().expect("count").parse().expect("usize")),
            Some("stress") | None => {}
            Some(first) => {
                z.push(first.parse().expect("atomic number"));
                let row: Vec<f64> = fields.map(|v| v.parse().expect("f64")).collect();
                assert_eq!(row.len(), 6, "{line}");
                positions.extend_from_slice(&row[..3]);
                forces.extend_from_slice(&row[3..]);
            }
        }
    }
    assert_eq!(Some(z.len()), count, "{name}: atom count");
    Reference {
        cell,
        energy: energy.expect("energy line"),
        z,
        positions,
        forces,
    }
}

fn assert_matches(name: &str, energy: f64, forces: &[f64], reference: &Reference) {
    assert!(
        (energy - reference.energy).abs() < ENERGY_TOLERANCE,
        "{name}: energy {energy} against LAMMPS {}",
        reference.energy
    );
    assert_eq!(forces.len(), reference.forces.len(), "{name}: force count");
    let worst = forces
        .iter()
        .zip(&reference.forces)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < FORCE_TOLERANCE,
        "{name}: largest force gap {worst} against LAMMPS"
    );
}

#[test]
fn copper_vacancy_matches_lammps_through_the_wrapper() {
    let reference = reference("cu_u3_vacancy.ref");
    assert_eq!(reference.z.len(), 107);
    let potential =
        EamTablePotential::open(&data_dir().join("Cu_u3.eam"), EamTableFormat::Funcfl, true)
            .expect("open Cu_u3.eam");
    let answer = potential
        .force(&reference.positions, &reference.z, &reference.cell)
        .expect("evaluate the vacancy cell");
    assert_matches("vacancy", answer.energy, &answer.forces, &reference);
}

#[test]
fn the_vacancy_carries_a_net_force_free_cell() {
    let reference = reference("cu_u3_vacancy.ref");
    let potential =
        EamTablePotential::open(&data_dir().join("Cu_u3.eam"), EamTableFormat::Auto, true)
            .expect("open Cu_u3.eam by its suffix");
    let answer = potential
        .force(&reference.positions, &reference.z, &reference.cell)
        .unwrap();
    for axis in 0..3 {
        let total: f64 = answer.forces.iter().skip(axis).step_by(3).sum();
        assert!(
            total.abs() < FORCE_TOLERANCE,
            "net force {total} on axis {axis}"
        );
    }
}

#[test]
fn the_whole_cell_and_the_setfl_cell_match_lammps() {
    let cubic = reference("cu_u3_cubic.ref");
    let funcfl =
        EamTablePotential::open(&data_dir().join("Cu_u3.eam"), EamTableFormat::Funcfl, true)
            .unwrap();
    let answer = funcfl
        .force(&cubic.positions, &cubic.z, &cubic.cell)
        .unwrap();
    assert_matches("cubic", answer.energy, &answer.forces, &cubic);

    let sheared = reference("cuni_sheared.ref");
    let setfl = EamTablePotential::open(
        &data_dir().join("CuNi.eam.alloy"),
        EamTableFormat::Auto,
        true,
    )
    .unwrap();
    let answer = setfl
        .force(&sheared.positions, &sheared.z, &sheared.cell)
        .unwrap();
    assert_matches("sheared", answer.energy, &answer.forces, &sheared);
}

#[test]
fn one_potential_serves_several_threads() {
    let reference = Arc::new(reference("cu_u3_vacancy.ref"));
    let potential = Arc::new(
        EamTablePotential::open(&data_dir().join("Cu_u3.eam"), EamTableFormat::Funcfl, true)
            .unwrap(),
    );
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let (reference, potential) = (Arc::clone(&reference), Arc::clone(&potential));
            std::thread::spawn(move || {
                potential
                    .force(&reference.positions, &reference.z, &reference.cell)
                    .unwrap()
            })
        })
        .collect();
    let answers: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
    for answer in &answers[1..] {
        assert_eq!(answer, &answers[0]);
    }
}

fn open_error(path: &Path, format: EamTableFormat) -> EamTableError {
    EamTablePotential::open(path, format, true).expect_err("the table must be refused")
}

#[test]
fn a_missing_file_is_an_open_error_naming_the_path() {
    let path = data_dir().join("no_such_table.eam");
    let err = open_error(&path, EamTableFormat::Funcfl);
    assert!(matches!(err, EamTableError::Open { .. }), "{err}");
    assert!(err.to_string().contains("no_such_table.eam"), "{err}");
    assert_eq!(err.status(), rgpot_status_t::RGPOT_INVALID_PARAMETER);
}

fn scratch(name: &str, contents: &[u8]) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, contents).expect("write the scratch table");
    path
}

#[test]
fn a_truncated_table_is_malformed() {
    let whole = std::fs::read(data_dir().join("Cu_u3.eam")).unwrap();
    let path = scratch("truncated.eam", &whole[..whole.len() / 2]);
    let err = open_error(&path, EamTableFormat::Funcfl);
    assert!(matches!(err, EamTableError::Malformed(_)), "{err}");
}

#[test]
fn a_file_that_is_not_a_table_is_malformed() {
    let path = scratch("garbage.eam", b"not a table\nat all\nreally\n");
    let err = open_error(&path, EamTableFormat::Funcfl);
    assert!(matches!(err, EamTableError::Malformed(_)), "{err}");
    let err = open_error(&path, EamTableFormat::Setfl);
    assert!(matches!(err, EamTableError::Malformed(_)), "{err}");
}

#[test]
fn a_funcfl_file_read_as_setfl_is_malformed() {
    let err = open_error(&data_dir().join("Cu_u3.eam"), EamTableFormat::Setfl);
    assert!(matches!(err, EamTableError::Malformed(_)), "{err}");
}

struct Handle(*mut rgpot_core::eindir::rgpot_potential_t);

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { rgpot_potential_free(self.0) };
    }
}

fn new_handle(path: &str, format: i32, periodic: bool) -> Handle {
    let path = CString::new(path).unwrap();
    let handle = unsafe { rgpot_potential_new_eam_table(path.as_ptr(), format, periodic) };
    assert!(!handle.is_null(), "constructor failed: {}", last_error());
    Handle(handle)
}

/// Owns the arrays a `rgpot_force_input_t` borrows.
struct Input {
    _positions: Vec<f64>,
    _numbers: Vec<i32>,
    _cell: Box<[f64; 9]>,
    input: rgpot_force_input_t,
}

impl Input {
    fn new(reference: &Reference, compute_stress: bool) -> Self {
        let mut positions = reference.positions.clone();
        let mut numbers = reference.z.clone();
        let mut cell = Box::new(reference.cell);
        let mut input = unsafe {
            rgpot_force_input_create(
                numbers.len(),
                positions.as_mut_ptr(),
                numbers.as_mut_ptr(),
                cell.as_mut_ptr(),
            )
        };
        input.compute_stress = compute_stress;
        Self {
            _positions: positions,
            _numbers: numbers,
            _cell: cell,
            input,
        }
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        unsafe { rgpot_force_input_free(&mut self.input) };
    }
}

fn calculate(
    handle: &Handle,
    reference: &Reference,
    compute_stress: bool,
) -> (rgpot_status_t, rgpot_force_out_t) {
    let input = Input::new(reference, compute_stress);
    let mut output = rgpot_force_out_t::default();
    let status = unsafe { rgpot_potential_calculate(handle.0, &input.input, &mut output) };
    (status, output)
}

#[test]
fn copper_vacancy_matches_lammps_through_the_c_abi() {
    let reference = reference("cu_u3_vacancy.ref");
    let table = data_dir().join("Cu_u3.eam");
    for format in [RGPOT_EAM_TABLE_FUNCFL, RGPOT_EAM_TABLE_AUTO] {
        let handle = new_handle(table.to_str().unwrap(), format, true);
        let (status, output) = calculate(&handle, &reference, false);
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
        let forces = unsafe {
            let data = rgpot_tensor_data(output.forces).cast::<f64>();
            std::slice::from_raw_parts(data, reference.forces.len())
        };
        assert_matches("C ABI vacancy", output.energy, forces, &reference);
        assert_eq!(output.variance, 0.0);
        unsafe { rgpot_tensor_free(output.forces) };
    }
}

fn strained(reference: &Reference, a: usize, b: usize, amount: f64) -> Reference {
    let mut changed = Reference {
        cell: reference.cell,
        energy: reference.energy,
        z: reference.z.clone(),
        positions: reference.positions.clone(),
        forces: reference.forces.clone(),
    };
    for r in changed.positions.chunks_exact_mut(3) {
        let old = [r[0], r[1], r[2]];
        r[a] += amount * old[b];
        if a != b {
            r[b] += amount * old[a];
        }
    }
    for row in changed.cell.chunks_exact_mut(3) {
        let old = [row[0], row[1], row[2]];
        row[a] += amount * old[b];
        if a != b {
            row[b] += amount * old[a];
        }
    }
    changed
}

fn output_forces(output: &rgpot_force_out_t, n: usize) -> Vec<f64> {
    unsafe {
        let data = rgpot_tensor_data(output.forces).cast::<f64>();
        std::slice::from_raw_parts(data, 3 * n).to_vec()
    }
}

#[test]
fn c_abi_eam_table_stress_matches_triclinic_energy_strain() {
    // A non-degenerate subset avoids a pair exactly on a nearest-image
    // boundary while retaining the published triclinic cell.
    let mut reference = reference("cuni_sheared.ref");
    reference.z.truncate(17);
    reference.positions.truncate(3 * reference.z.len());
    reference.forces.truncate(3 * reference.z.len());
    let handle = new_handle(
        data_dir().join("CuNi.eam.alloy").to_str().unwrap(),
        RGPOT_EAM_TABLE_SETFL,
        true,
    );
    let (status, requested) = calculate(&handle, &reference, true);
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
    assert_eq!(requested.has_stress, 1);
    assert_eq!(requested.stress[1], requested.stress[3]);
    assert_eq!(requested.stress[2], requested.stress[6]);
    assert_eq!(requested.stress[5], requested.stress[7]);

    let volume = reference.cell[0]
        * (reference.cell[4] * reference.cell[8] - reference.cell[5] * reference.cell[7])
        - reference.cell[1]
            * (reference.cell[3] * reference.cell[8] - reference.cell[5] * reference.cell[6])
        + reference.cell[2]
            * (reference.cell[3] * reference.cell[7] - reference.cell[4] * reference.cell[6]);
    let h = 1.0e-5;
    let mut worst = 0.0_f64;
    for (a, b) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)] {
        let (plus_status, plus) = calculate(&handle, &strained(&reference, a, b, h), false);
        let (minus_status, minus) = calculate(&handle, &strained(&reference, a, b, -h), false);
        assert_eq!(plus_status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(minus_status, rgpot_status_t::RGPOT_SUCCESS);
        let finite_difference = (plus.energy - minus.energy) / (2.0 * h);
        let multiplicity = if a == b { 1.0 } else { 2.0 };
        let analytic = volume.abs() * multiplicity * requested.stress[3 * a + b];
        let relative = (analytic - finite_difference).abs()
            / analytic.abs().max(finite_difference.abs()).max(1.0e-8);
        eprintln!(
            "component {a}{b}: analytic {analytic:.12e} finite_difference {finite_difference:.12e} relative {relative:.12e}"
        );
        worst = worst.max(relative);
        unsafe {
            rgpot_tensor_free(plus.forces);
            rgpot_tensor_free(minus.forces);
        }
    }
    eprintln!("EAMTable C ABI triclinic FD max relative error {worst:.12e}");
    assert!(worst < 1.0e-6, "max relative stress error {worst}");

    let requested_forces = output_forces(&requested, reference.z.len());
    let (plain_status, plain) = calculate(&handle, &reference, false);
    assert_eq!(plain_status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(plain.has_stress, 0);
    assert_eq!(plain.stress, [0.0; 9]);
    assert_eq!(plain.energy.to_bits(), requested.energy.to_bits());
    assert_eq!(output_forces(&plain, reference.z.len()), requested_forces);
    unsafe {
        rgpot_tensor_free(requested.forces);
        rgpot_tensor_free(plain.forces);
    }
}

#[test]
fn the_c_abi_refuses_a_cell_smaller_than_the_cutoff_allows() {
    let mut reference = reference("cu_u3_vacancy.ref");
    reference.cell = [8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0];
    let handle = new_handle(
        data_dir().join("Cu_u3.eam").to_str().unwrap(),
        RGPOT_EAM_TABLE_FUNCFL,
        true,
    );
    let (status, output) = calculate(&handle, &reference, false);
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert!(output.forces.is_null());
    assert!(
        last_error().contains("half a face spacing"),
        "{}",
        last_error()
    );
}

#[test]
fn the_c_abi_constructor_reports_each_refusal_with_null() {
    let missing = CString::new("/nonexistent/dir/Cu_u3.eam").unwrap();
    let empty = CString::new("").unwrap();
    let unknown_suffix = CString::new("Cu_u3.dat").unwrap();
    let table = CString::new(data_dir().join("Cu_u3.eam").to_str().unwrap()).unwrap();
    let cases: [(&str, *const std::os::raw::c_char, i32, &str); 6] = [
        (
            "null path",
            std::ptr::null(),
            RGPOT_EAM_TABLE_FUNCFL,
            "path is NULL",
        ),
        (
            "empty path",
            empty.as_ptr(),
            RGPOT_EAM_TABLE_FUNCFL,
            "path is empty",
        ),
        (
            "missing file",
            missing.as_ptr(),
            RGPOT_EAM_TABLE_FUNCFL,
            "cannot open /nonexistent/dir/Cu_u3.eam",
        ),
        ("bad format code", table.as_ptr(), 17, "format code 17"),
        (
            "suffix that names no format",
            unknown_suffix.as_ptr(),
            RGPOT_EAM_TABLE_AUTO,
            "cannot tell the table format",
        ),
        (
            "wrong layout",
            table.as_ptr(),
            RGPOT_EAM_TABLE_FINNIS_SINCLAIR,
            "EAM table",
        ),
    ];
    for (name, path, format, expect) in cases {
        let handle = unsafe { rgpot_potential_new_eam_table(path, format, true) };
        assert!(handle.is_null(), "{name}: a handle came back");
        assert!(last_error().contains(expect), "{name}: {}", last_error());
    }
    let setfl =
        unsafe { rgpot_potential_new_eam_table(table.as_ptr(), RGPOT_EAM_TABLE_SETFL, true) };
    assert!(setfl.is_null(), "a funcfl file read as setfl");
}

#[test]
fn a_truncated_file_is_refused_by_the_c_abi() {
    let whole = std::fs::read(data_dir().join("Cu_u3.eam")).unwrap();
    let path = scratch("truncated_c_abi.eam", &whole[..whole.len() / 3]);
    let path = CString::new(path.to_str().unwrap()).unwrap();
    let handle =
        unsafe { rgpot_potential_new_eam_table(path.as_ptr(), RGPOT_EAM_TABLE_FUNCFL, true) };
    assert!(handle.is_null());
    assert!(
        last_error().contains("short or unreadable array"),
        "{}",
        last_error()
    );
}

#[test]
fn a_free_cluster_ignores_the_box_and_loses_the_bonds_past_its_surface() {
    let mut reference = reference("cu_u3_cubic.ref");
    let table = data_dir().join("Cu_u3.eam");
    let periodic = EamTablePotential::open(&table, EamTableFormat::Funcfl, true)
        .unwrap()
        .force(&reference.positions, &reference.z, &reference.cell)
        .unwrap();
    reference.cell = [0.0; 9];
    let handle = new_handle(table.to_str().unwrap(), RGPOT_EAM_TABLE_FUNCFL, false);
    let (status, output) = calculate(&handle, &reference, false);
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
    assert!(output.energy > periodic.energy, "{}", output.energy);
    unsafe { rgpot_tensor_free(output.forces) };
}
