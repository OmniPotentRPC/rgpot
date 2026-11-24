// MIT License
// Copyright 2023--present rgpot developers

//! A stress request to the RPC server fails when the potential cannot answer
//! it, the same way the C++ server fails it, and never succeeds with a zero
//! tensor or with `has_stress` clear.
//!
//! The potential is the EAM table, which computes stress for a periodic
//! table in a cell it can measure. Three requests it cannot answer are
//! checked: a singular cell, a table that is not periodic, and a cell the
//! kernel has no stress for. A request it can answer, a sheared cell, is
//! checked alongside so the refusals are not a server that never serves
//! stress. A force-only call on the same server stays unaffected where the
//! potential can evaluate the geometry at all.

use std::ffi::{c_void, CStr, CString};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rgpot_core::c_api::eam_table::rgpot_potential_new_eam_table;
use rgpot_core::c_api::potential::{
    rgpot_potential_new, rgpot_potential_set_stress_callback, rgpot_stress_out_create,
};
use rgpot_core::c_api::rpc::{
    rgpot_rpc_calculate, rgpot_rpc_calculate_stress, rgpot_rpc_client_free, rgpot_rpc_client_new,
};
use rgpot_core::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
use rgpot_core::eam_table::{RGPOT_EAM_TABLE_FUNCFL, RGPOT_EAM_TABLE_SETFL};
use rgpot_core::eindir::rgpot_potential_t;
use rgpot_core::rpc::server::rgpot_rpc_server_start;
use rgpot_core::status::{rgpot_last_error, rgpot_status_t};
use rgpot_core::tensor::{rgpot_tensor_free, rgpot_tensor_owned_cpu_f64_2d};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t, rgpot_stress_out_t};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../CppCore/tests/data/eam_table")
}

struct Geometry {
    cell: [f64; 9],
    numbers: Vec<i32>,
    positions: Vec<f64>,
}

/// A LAMMPS reference geometry of the C++ EAM table tests, cut to `atoms`.
fn read_geometry(name: &str, atoms: usize) -> Geometry {
    let text = std::fs::read_to_string(data_dir().join(name)).expect("reference file");
    let mut cell = [0.0; 9];
    let mut numbers = Vec::new();
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
                numbers.push(first.parse().expect("atomic number"));
                let row: Vec<f64> = fields.map(|v| v.parse().expect("f64")).collect();
                positions.extend_from_slice(&row[..3]);
            }
        }
    }
    numbers.truncate(atoms);
    positions.truncate(3 * atoms);
    Geometry {
        cell,
        numbers,
        positions,
    }
}

fn cubic() -> Geometry {
    read_geometry("cu_u3_cubic.ref", 32)
}

fn sheared() -> Geometry {
    read_geometry("cuni_sheared.ref", 17)
}

struct SendPtr(*const rgpot_potential_t);
unsafe impl Send for SendPtr {}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("addr")
        .port()
}

fn serve(pot: *const rgpot_potential_t) -> u16 {
    assert!(!pot.is_null(), "{}", last_error());
    let port = free_port();
    let handle = SendPtr(pot);
    std::thread::spawn(move || {
        let handle = handle;
        let host = CString::new("127.0.0.1").unwrap();
        unsafe { rgpot_rpc_server_start(handle.0, host.as_ptr(), port) };
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        std::thread::sleep(Duration::from_millis(20));
    }
    port
}

fn eam_server(file: &str, format: i32, periodic: bool) -> u16 {
    let path = CString::new(data_dir().join(file).to_str().unwrap()).unwrap();
    serve(unsafe { rgpot_potential_new_eam_table(path.as_ptr(), format, periodic) })
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(rgpot_last_error()) }
        .to_string_lossy()
        .into_owned()
}

struct Outcome {
    status: rgpot_status_t,
    error: String,
    stress: rgpot_stress_out_t,
}

/// One call from a fresh client: force-only, or with a stress request.
fn call(port: u16, geometry: &Geometry, with_stress: bool) -> Outcome {
    let host = CString::new("127.0.0.1").unwrap();
    let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), port) };
    assert!(!client.is_null(), "{}", last_error());
    let mut positions = geometry.positions.clone();
    let mut numbers = geometry.numbers.clone();
    let mut cell = Box::new(geometry.cell);
    let mut input: rgpot_force_input_t = unsafe {
        rgpot_force_input_create(
            numbers.len(),
            positions.as_mut_ptr(),
            numbers.as_mut_ptr(),
            cell.as_mut_ptr(),
        )
    };
    let mut output = rgpot_force_out_t::default();
    let mut stress = rgpot_stress_out_create();
    let status = if with_stress {
        unsafe { rgpot_rpc_calculate_stress(client, &input, &mut output, &mut stress) }
    } else {
        unsafe { rgpot_rpc_calculate(client, &input, &mut output) }
    };
    let error = if status == rgpot_status_t::RGPOT_SUCCESS {
        String::new()
    } else {
        last_error()
    };
    unsafe {
        if !output.forces.is_null() {
            rgpot_tensor_free(output.forces);
        }
        rgpot_force_input_free(&mut input);
        rgpot_rpc_client_free(client);
    }
    Outcome {
        status,
        error,
        stress,
    }
}

fn assert_refused(outcome: &Outcome, what: &str) {
    assert_eq!(outcome.status, rgpot_status_t::RGPOT_RPC_ERROR, "{what}");
    assert!(
        outcome.error.to_lowercase().contains("stress")
            || outcome
                .error
                .contains("potential callback returned an error"),
        "{what}: {}",
        outcome.error
    );
    assert_eq!(outcome.stress.has_stress, 0, "{what}");
    assert_eq!(outcome.stress.stress, [0.0; 9], "{what}");
}

#[test]
fn a_stress_request_on_a_sheared_cell_is_served() {
    let port = eam_server("CuNi.eam.alloy", RGPOT_EAM_TABLE_SETFL, true);
    let shear = sheared();
    let outcome = call(port, &shear, true);
    assert_eq!(
        outcome.status,
        rgpot_status_t::RGPOT_SUCCESS,
        "{}",
        outcome.error
    );
    assert_eq!(outcome.stress.has_stress, 1);
    let s = outcome.stress.stress;
    assert!(s.iter().all(|v| v.is_finite()));
    assert!(s.iter().any(|v| *v != 0.0));
    assert_eq!(s[1], s[3]);
    assert_eq!(s[2], s[6]);
    assert_eq!(s[5], s[7]);

    let forces_only = call(port, &shear, false);
    assert_eq!(forces_only.status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(forces_only.stress.has_stress, 0);
}

#[test]
fn a_stress_request_on_a_singular_cell_fails() {
    let port = eam_server("Cu_u3.eam", RGPOT_EAM_TABLE_FUNCFL, true);
    let mut flat = cubic();
    flat.cell[8] = 0.0;
    assert_refused(&call(port, &flat, true), "singular cell");

    let mut empty = cubic();
    empty.cell = [0.0; 9];
    assert_refused(&call(port, &empty, true), "zero cell");

    let served = call(port, &cubic(), true);
    assert_eq!(
        served.status,
        rgpot_status_t::RGPOT_SUCCESS,
        "{}",
        served.error
    );
    assert_eq!(served.stress.has_stress, 1);
}

#[test]
fn a_stress_request_on_a_non_periodic_table_fails_and_force_only_still_serves() {
    let port = eam_server("Cu_u3.eam", RGPOT_EAM_TABLE_FUNCFL, false);
    let cluster = cubic();
    let refused = call(port, &cluster, true);
    assert_refused(&refused, "non-periodic table");
    assert!(
        refused.error.contains("returned no stress"),
        "the message matches the C++ server: {}",
        refused.error
    );

    let forces_only = call(port, &cluster, false);
    assert_eq!(
        forces_only.status,
        rgpot_status_t::RGPOT_SUCCESS,
        "{}",
        forces_only.error
    );
    assert_eq!(forces_only.stress.has_stress, 0);
}

/// The Fortran-style EAM kernel differentiates only an orthorhombic cell
/// and leaves `has_stress` clear on any other. The server must turn that
/// into a failed request rather than a success without stress.
mod orthorhombic_only {
    use super::*;
    use rgpot_core::tensor::rgpot_tensor_data;

    unsafe extern "C" fn force(
        _user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let n = unsafe { (*input).n_atoms() }.expect("positions");
        let values = vec![0.0; 3 * n];
        unsafe {
            (*output).energy = -1.0;
            (*output).forces = rgpot_tensor_owned_cpu_f64_2d(values.as_ptr(), n as i64, 3);
        }
        rgpot_status_t::RGPOT_SUCCESS
    }

    unsafe extern "C" fn force_and_stress(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
        stress: *mut rgpot_stress_out_t,
    ) -> rgpot_status_t {
        let status = unsafe { force(user_data, input, output) };
        let cell = unsafe { rgpot_tensor_data((*input).box_matrix) } as *const f64;
        let cell = unsafe { std::slice::from_raw_parts(cell, 9) };
        let orthorhombic = [1, 2, 3, 5, 6, 7].iter().all(|&k| cell[k] == 0.0);
        if orthorhombic {
            unsafe {
                (*stress).stress = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
                (*stress).has_stress = 1;
            }
        }
        status
    }

    #[test]
    fn a_stress_request_on_a_non_orthorhombic_cell_fails() {
        let pot = unsafe { rgpot_potential_new(force, std::ptr::null_mut(), None) };
        assert!(!pot.is_null());
        let status = unsafe { rgpot_potential_set_stress_callback(pot, Some(force_and_stress)) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        let port = serve(pot);

        let box_cell = cubic();
        let served = call(port, &box_cell, true);
        assert_eq!(
            served.status,
            rgpot_status_t::RGPOT_SUCCESS,
            "{}",
            served.error
        );
        assert_eq!(served.stress.has_stress, 1);

        let mut tilted = cubic();
        tilted.cell[3] = 1.5;
        let refused = call(port, &tilted, true);
        assert_refused(&refused, "non-orthorhombic cell");
        assert!(
            refused.error.contains("returned no stress"),
            "{}",
            refused.error
        );

        let forces_only = call(port, &tilted, false);
        assert_eq!(forces_only.status, rgpot_status_t::RGPOT_SUCCESS);
    }
}
