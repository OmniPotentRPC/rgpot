// MIT License
// Copyright 2023--present rgpot developers

use std::path::{Path, PathBuf};

use rgpot_core::c_api::potential::rgpot_potential_calculate;
use rgpot_core::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
use rgpot_core::eam_table::{EamTable, EamTableFormat};
use rgpot_core::eam_table_factory::{
    eam_table_from_data, eam_table_from_data_with, eam_table_from_file, eam_table_from_file_with,
    EamTableFactoryOptions,
};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::{rgpot_tensor_data, rgpot_tensor_free};
use rgpot_core::types::rgpot_force_out_t;
use rgpot_core::RgpotError;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../CppCore/tests/data/eam_table")
}

fn evaluate(
    handle: &rgpot_core::potential::PotentialImpl,
    positions: &[f64],
    atomic_numbers: &[i32],
    cell: [f64; 9],
) -> (f64, Vec<f64>) {
    let mut positions = positions.to_vec();
    let mut atomic_numbers = atomic_numbers.to_vec();
    let mut cell = cell;
    let mut input = unsafe {
        rgpot_force_input_create(
            atomic_numbers.len(),
            positions.as_mut_ptr(),
            atomic_numbers.as_mut_ptr(),
            cell.as_mut_ptr(),
        )
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    let status = unsafe { rgpot_potential_calculate(handle.as_handle(), &input, &mut output) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let forces = unsafe {
        let ptr = rgpot_tensor_data(output.forces).cast::<f64>();
        std::slice::from_raw_parts(ptr, positions.len()).to_vec()
    };
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_force_input_free(&mut input);
    }
    (output.energy, forces)
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
    for line in text.lines().filter(|line| !line.starts_with('#')) {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("box") => {
                for (slot, value) in cell.iter_mut().zip(fields) {
                    *slot = value.parse().expect("box value");
                }
            }
            Some("energy") => energy = Some(fields.next().expect("energy").parse().expect("f64")),
            Some("atoms") => count = Some(fields.next().expect("count").parse().expect("usize")),
            Some("stress") => {}
            None => {}
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

fn finite_difference_forces(
    handle: &rgpot_core::potential::PotentialImpl,
    positions: &[f64],
    atomic_numbers: &[i32],
    cell: [f64; 9],
    step: f64,
) -> Vec<f64> {
    let mut fd = vec![0.0; positions.len()];
    for k in 0..positions.len() {
        let mut plus = positions.to_vec();
        let mut minus = positions.to_vec();
        plus[k] += step;
        minus[k] -= step;
        let (e_plus, _) = evaluate(handle, &plus, atomic_numbers, cell);
        let (e_minus, _) = evaluate(handle, &minus, atomic_numbers, cell);
        fd[k] = -(e_plus - e_minus) / (2.0 * step);
    }
    fd
}

#[test]
fn from_file_funcfl_matches_reference_and_finite_difference() {
    let path = data_dir().join("Cu_u3.eam");
    let reference = reference("cu_u3_cubic.ref");
    let handle = eam_table_from_file(&path, EamTableFormat::Funcfl, true, 1.0).unwrap();
    let (energy, forces) = evaluate(&handle, &reference.positions, &reference.z, reference.cell);
    assert!((energy - reference.energy).abs() < 1.0e-10);
    for (got, want) in forces.iter().zip(&reference.forces) {
        assert!((got - want).abs() < 1.0e-10);
    }
    let positions = [0.0, 0.0, 0.0, 2.35, 0.1, -0.2];
    let z = [29_i32, 29_i32];
    let cell = [0.0; 9];
    let local = eam_table_from_file(&path, EamTableFormat::Funcfl, false, 1.0).unwrap();
    let (_, local_forces) = evaluate(&local, &positions, &z, cell);
    let fd = finite_difference_forces(&local, &positions, &z, cell, 1.0e-6);
    for (got, want) in local_forces.iter().zip(fd) {
        assert!((got - want).abs() < 1.0e-4);
    }
}

#[test]
fn from_data_setfl_matches_reference_and_finite_difference() {
    let table = EamTable::open(&data_dir().join("CuNi.eam.alloy"), EamTableFormat::Setfl).unwrap();
    let reference = reference("cuni_sheared.ref");
    let handle = eam_table_from_data(table.clone(), true, 1.0).unwrap();
    let (energy, forces) = evaluate(&handle, &reference.positions, &reference.z, reference.cell);
    assert!((energy - reference.energy).abs() < 1.0e-10);
    for (got, want) in forces.iter().zip(&reference.forces) {
        assert!((got - want).abs() < 1.0e-10);
    }
    let positions = [0.0, 0.0, 0.0, 2.45, 0.0, 0.0];
    let z = [29_i32, 28_i32];
    let cell = [0.0; 9];
    let local = eam_table_from_data(table, false, 1.0).unwrap();
    let (_, local_forces) = evaluate(&local, &positions, &z, cell);
    let fd = finite_difference_forces(&local, &positions, &z, cell, 1.0e-6);
    for (got, want) in local_forces.iter().zip(fd) {
        assert!((got - want).abs() < 1.0e-4);
    }
}

#[test]
fn factory_with_options_validates_skin_and_constructs_handles() {
    let path = data_dir().join("Cu_u3.eam");
    let bad = EamTableFactoryOptions::default().with_skin(-0.5);
    let err = match eam_table_from_file_with(&path, EamTableFormat::Funcfl, bad) {
        Ok(_) => panic!("expected invalid skin error"),
        Err(err) => err,
    };
    assert!(matches!(err, RgpotError::Unsupported(_)), "{err}");

    let table = EamTable::open(&path, EamTableFormat::Funcfl).unwrap();
    let opts = EamTableFactoryOptions::default()
        .with_periodic(false)
        .with_skin(0.5);
    let handle = eam_table_from_data_with(table, opts).unwrap();
    let (energy, forces) = evaluate(
        &handle,
        &[0.0, 0.0, 0.0, 2.35, 0.0, 0.0],
        &[29, 29],
        [0.0; 9],
    );
    assert!(energy.is_finite());
    assert_eq!(forces.len(), 6);
}
