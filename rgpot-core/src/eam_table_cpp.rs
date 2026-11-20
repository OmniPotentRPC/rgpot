// MIT License
// Copyright 2023--present rgpot developers

//! C++-backed EAM table constructor bridge.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use crate::eam_table::{EamTable, EamTableError, EamTableFormat, RGPOT_EAM_TABLE_FINNIS_SINCLAIR};
use crate::error::RgpotError;
use crate::potential::PotentialImpl;
use crate::status::{rgpot_status_t, set_last_error};
use crate::tensor::{copy_f64_tensor, copy_i32_tensor, create_owned_f64_tensor};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

unsafe extern "C" {
    fn rgpot_core_eam_cpp_new_from_file(
        path: *const c_char,
        format: c_int,
        periodic: bool,
        stress: bool,
        skin: f64,
    ) -> *mut c_void;
    fn rgpot_core_eam_cpp_new_from_text(
        text: *const c_char,
        format: c_int,
        periodic: bool,
        stress: bool,
        skin: f64,
    ) -> *mut c_void;
    fn rgpot_core_eam_cpp_free(handle: *mut c_void);
    fn rgpot_core_eam_cpp_force(
        handle: *mut c_void,
        n_atoms: usize,
        positions: *const f64,
        atomic_numbers: *const i32,
        cell: *const f64,
        forces: *mut f64,
        energy: *mut f64,
    ) -> c_int;
    fn rgpot_core_eam_cpp_last_error() -> *const c_char;
    #[cfg(test)]
    fn rgpot_core_eam_cpp_drop_count() -> i64;
    #[cfg(test)]
    fn rgpot_core_eam_cpp_reset_drop_count();
}

fn cpp_error(fallback: &str) -> RgpotError {
    let message = unsafe {
        let ptr = rgpot_core_eam_cpp_last_error();
        if ptr.is_null() {
            fallback.to_owned()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    RgpotError::EamTable(EamTableError::Invalid(message))
}

fn table_to_text(table: &EamTable) -> String {
    table.to_finnis_sinclair_text()
}

pub(crate) fn cpp_potential_from_file(
    path: &Path,
    format: EamTableFormat,
    periodic: bool,
    stress: bool,
    skin: f64,
) -> Result<PotentialImpl, RgpotError> {
    let path = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
        RgpotError::EamTable(EamTableError::Invalid("path has interior NUL".into()))
    })?;
    let user_data = unsafe {
        rgpot_core_eam_cpp_new_from_file(path.as_ptr(), format as c_int, periodic, stress, skin)
    };
    if user_data.is_null() {
        return Err(cpp_error("EAM table: C++ constructor failed"));
    }
    Ok(PotentialImpl::new(
        cpp_eam_table_callback,
        user_data,
        Some(drop_cpp_eam_table),
    ))
}

pub(crate) fn cpp_potential_from_data(
    table: EamTable,
    periodic: bool,
    stress: bool,
    skin: f64,
) -> Result<PotentialImpl, RgpotError> {
    let text = CString::new(table_to_text(&table)).map_err(|_| {
        RgpotError::EamTable(EamTableError::Invalid("table text has interior NUL".into()))
    })?;
    let user_data = unsafe {
        rgpot_core_eam_cpp_new_from_text(
            text.as_ptr(),
            RGPOT_EAM_TABLE_FINNIS_SINCLAIR,
            periodic,
            stress,
            skin,
        )
    };
    if user_data.is_null() {
        return Err(cpp_error("EAM table: C++ constructor from data failed"));
    }
    Ok(PotentialImpl::new(
        cpp_eam_table_callback,
        user_data,
        Some(drop_cpp_eam_table),
    ))
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn cpp_eval_file(
    path: &Path,
    format: EamTableFormat,
    periodic: bool,
    stress: bool,
    skin: f64,
    positions: &[f64],
    atomic_numbers: &[i32],
    cell: &[f64; 9],
) -> Result<(f64, Vec<f64>), RgpotError> {
    let path = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
        RgpotError::EamTable(EamTableError::Invalid("path has interior NUL".into()))
    })?;
    let handle = unsafe {
        rgpot_core_eam_cpp_new_from_file(path.as_ptr(), format as c_int, periodic, stress, skin)
    };
    if handle.is_null() {
        return Err(cpp_error("EAM table: C++ constructor failed"));
    }
    let mut forces = vec![0.0_f64; positions.len()];
    let mut energy = 0.0_f64;
    let status = unsafe {
        rgpot_core_eam_cpp_force(
            handle,
            atomic_numbers.len(),
            positions.as_ptr(),
            atomic_numbers.as_ptr(),
            cell.as_ptr(),
            forces.as_mut_ptr(),
            &mut energy,
        )
    };
    unsafe { rgpot_core_eam_cpp_free(handle) };
    if status == 0 {
        Ok((energy, forces))
    } else {
        Err(cpp_error("EAM table: C++ direct evaluation failed"))
    }
}

#[cfg(test)]
pub(crate) fn cpp_drop_count() -> i64 {
    unsafe { rgpot_core_eam_cpp_drop_count() }
}

#[cfg(test)]
pub(crate) fn cpp_reset_drop_count() {
    unsafe { rgpot_core_eam_cpp_reset_drop_count() };
}

unsafe extern "C" fn drop_cpp_eam_table(user_data: *mut c_void) {
    unsafe { rgpot_core_eam_cpp_free(user_data) };
}

unsafe extern "C" fn cpp_eam_table_callback(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    if user_data.is_null() || input.is_null() || output.is_null() {
        set_last_error("EAM table potential: null user data, input or output");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let input = unsafe { &*input };
    let output = unsafe { &mut *output };
    let Some(n_atoms) = (unsafe { input.n_atoms() }) else {
        set_last_error("EAM table potential: positions tensor has no shape");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    };
    let positions = copy_f64_tensor(input.positions, "positions", 3 * n_atoms, true);
    let numbers = copy_i32_tensor(input.atomic_numbers, "atomic_numbers", n_atoms);
    let cell = copy_f64_tensor(input.box_matrix, "box_matrix", 9, true);
    let (positions, numbers, cell) = match (positions, numbers, cell) {
        (Ok(positions), Ok(numbers), Ok(cell)) => (positions, numbers, cell),
        (Err(why), _, _) | (_, Err(why), _) | (_, _, Err(why)) => {
            set_last_error(&why);
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
    };
    let mut forces = vec![0.0_f64; 3 * n_atoms];
    let mut energy = 0.0_f64;
    let status = unsafe {
        rgpot_core_eam_cpp_force(
            user_data,
            n_atoms,
            positions.as_ptr(),
            numbers.as_ptr(),
            cell.as_ptr(),
            forces.as_mut_ptr(),
            &mut energy,
        )
    };
    match status {
        0 => {
            output.energy = energy;
            output.variance = 0.0;
            output.forces = create_owned_f64_tensor(forces, vec![n_atoms as i64, 3]);
            rgpot_status_t::RGPOT_SUCCESS
        }
        1 => {
            let why = cpp_error("EAM table: C++ force failed");
            set_last_error(&why.to_string());
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        }
        _ => {
            let why = cpp_error("EAM table: C++ force internal error");
            set_last_error(&why.to_string());
            rgpot_status_t::RGPOT_INTERNAL_ERROR
        }
    }
}
