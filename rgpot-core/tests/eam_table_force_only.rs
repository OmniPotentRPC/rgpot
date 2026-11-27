// MIT License
// Copyright 2023--present rgpot developers

//! The force-only EAM table call on a triclinic cell is bit-identical to the
//! numbers the library produced before stress existed.
//!
//! Each constant below is a hash over the bit patterns of the energy and
//! every force component, taken from the library with no stress support.
//! The test uses only entry points that library already had, so the same
//! file builds and passes against it.
use std::ffi::CString;
use std::path::{Path, PathBuf};

use rgpot_core::c_api::eam_table::rgpot_potential_new_eam_table;
use rgpot_core::c_api::potential::{rgpot_potential_calculate, rgpot_potential_free};
use rgpot_core::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
use rgpot_core::eam_table::RGPOT_EAM_TABLE_SETFL;
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::{rgpot_tensor_data, rgpot_tensor_free};
use rgpot_core::types::rgpot_force_out_t;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../CppCore/tests/data/eam_table")
}

fn read_cell(name: &str) -> ([f64; 9], Vec<i32>, Vec<f64>) {
    let text = std::fs::read_to_string(data_dir().join(name)).expect("reference file");
    let mut cell = [0.0; 9];
    let mut z = Vec::new();
    let mut positions = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("box") => {
                for (slot, v) in cell.iter_mut().zip(fields) {
                    *slot = v.parse().expect("box value");
                }
            }
            Some("energy" | "atoms" | "stress") | None => {}
            Some(first) => {
                z.push(first.parse().expect("atomic number"));
                let row: Vec<f64> = fields.map(|v| v.parse().expect("f64")).collect();
                positions.extend_from_slice(&row[..3]);
            }
        }
    }
    (cell, z, positions)
}

fn fnv(values: impl IntoIterator<Item = f64>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        hash ^= value.to_bits();
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn force_only_hash(cell: [f64; 9], mut z: Vec<i32>, mut positions: Vec<f64>) -> (u64, u64) {
    let table = CString::new(data_dir().join("CuNi.eam.alloy").to_str().unwrap()).unwrap();
    let handle =
        unsafe { rgpot_potential_new_eam_table(table.as_ptr(), RGPOT_EAM_TABLE_SETFL, true) };
    assert!(!handle.is_null());
    let mut cell = Box::new(cell);
    let mut input = unsafe {
        rgpot_force_input_create(
            z.len(),
            positions.as_mut_ptr(),
            z.as_mut_ptr(),
            cell.as_mut_ptr(),
        )
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    let status = unsafe { rgpot_potential_calculate(handle, &input, &mut output) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let forces = unsafe {
        let data = rgpot_tensor_data(output.forces).cast::<f64>();
        std::slice::from_raw_parts(data, 3 * z.len()).to_vec()
    };
    let energy = fnv([output.energy]);
    let all = fnv(std::iter::once(output.energy).chain(forces));
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_force_input_free(&mut input);
        rgpot_potential_free(handle);
    }
    (energy, all)
}

#[test]
fn triclinic_force_only_result_matches_the_pre_stress_library_bit_for_bit() {
    let (cell, mut z, mut positions) = read_cell("cuni_sheared.ref");
    z.truncate(17);
    positions.truncate(3 * z.len());
    let (energy, all) = force_only_hash(cell, z, positions);
    assert_eq!(
        (energy, all),
        (0x14ae_2b90_a96f_9136, 0xc9d8_1c0e_bc13_cb1c)
    );
}
