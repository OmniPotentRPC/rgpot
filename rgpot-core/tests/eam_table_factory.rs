// MIT License
// Copyright 2023--present rgpot developers

use std::path::{Path, PathBuf};

use rgpot_core::c_api::potential::rgpot_potential_calculate;
use rgpot_core::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
use rgpot_core::eam_table::{EamTable, EamTableFormat, EamTablePotential};
use rgpot_core::eam_table_factory::{eam_table_from_data, eam_table_from_file};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::{rgpot_tensor_data, rgpot_tensor_free};
use rgpot_core::types::rgpot_force_out_t;

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
    let positions = [0.0, 0.0, 0.0, 2.35, 0.1, -0.2];
    let z = [29_i32, 29_i32];
    let cell = [0.0; 9];

    let handle = eam_table_from_file(&path, EamTableFormat::Funcfl, false, false, 1.0).unwrap();
    let (energy, forces) = evaluate(&handle, &positions, &z, cell);

    let reference = EamTablePotential::open(&path, EamTableFormat::Funcfl, false).unwrap();
    let expected = reference.force(&positions, &z, &cell).unwrap();
    assert!((energy - expected.energy).abs() < 1.0e-12);
    for (got, want) in forces.iter().zip(&expected.forces) {
        assert!((got - want).abs() < 1.0e-10);
    }

    let fd = finite_difference_forces(&handle, &positions, &z, cell, 1.0e-6);
    for (got, want) in forces.iter().zip(fd) {
        assert!((got - want).abs() < 1.0e-4);
    }
}

#[test]
fn from_data_setfl_matches_reference_and_finite_difference() {
    let table = EamTable::open(&data_dir().join("CuNi.eam.alloy"), EamTableFormat::Setfl).unwrap();
    let positions = [0.0, 0.0, 0.0, 2.45, 0.0, 0.0];
    let z = [29_i32, 28_i32];
    let cell = [0.0; 9];

    let handle = eam_table_from_data(table.clone(), false, false, 1.0).unwrap();
    let (energy, forces) = evaluate(&handle, &positions, &z, cell);

    let reference = EamTablePotential::new(table, false).unwrap();
    let expected = reference.force(&positions, &z, &cell).unwrap();
    assert!((energy - expected.energy).abs() < 1.0e-12);
    for (got, want) in forces.iter().zip(&expected.forces) {
        assert!((got - want).abs() < 1.0e-10);
    }

    let fd = finite_difference_forces(&handle, &positions, &z, cell, 1.0e-6);
    for (got, want) in forces.iter().zip(fd) {
        assert!((got - want).abs() < 1.0e-4);
    }
}
