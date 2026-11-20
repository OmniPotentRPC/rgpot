// MIT License
// Copyright 2023--present rgpot developers

//! C++-backed EAM table constructor bridge.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use crate::eam_table::{EamTable, EamTableError, EamTableFormat, RGPOT_EAM_TABLE_FINNIS_SINCLAIR};
use crate::error::RgpotError;
use crate::potential::PotentialImpl;
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
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

struct CppEamTable {
    handle: *mut c_void,
    periodic: bool,
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
    let handle = unsafe {
        rgpot_core_eam_cpp_new_from_file(path.as_ptr(), format as c_int, periodic, stress, skin)
    };
    if handle.is_null() {
        return Err(cpp_error("EAM table: C++ constructor failed"));
    }
    let user_data = Box::into_raw(Box::new(CppEamTable { handle, periodic })).cast();
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
    let handle = unsafe {
        rgpot_core_eam_cpp_new_from_text(
            text.as_ptr(),
            RGPOT_EAM_TABLE_FINNIS_SINCLAIR,
            periodic,
            stress,
            skin,
        )
    };
    if handle.is_null() {
        return Err(cpp_error("EAM table: C++ constructor from data failed"));
    }
    let user_data = Box::into_raw(Box::new(CppEamTable { handle, periodic })).cast();
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
    cell: &[f64],
) -> Result<(f64, Vec<f64>), RgpotError> {
    if positions.len() != 3 * atomic_numbers.len() {
        return Err(RgpotError::EamTable(EamTableError::Geometry(format!(
            "{} coordinates for {} atoms",
            positions.len(),
            atomic_numbers.len()
        ))));
    }
    if cell.len() != 9 {
        return Err(RgpotError::EamTable(EamTableError::Geometry(format!(
            "cell has {} entries, not 9",
            cell.len()
        ))));
    }
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

#[cfg(test)]
pub(crate) static CPP_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

unsafe extern "C" fn drop_cpp_eam_table(user_data: *mut c_void) {
    let potential = unsafe { Box::from_raw(user_data.cast::<CppEamTable>()) };
    unsafe { rgpot_core_eam_cpp_free(potential.handle) };
}

unsafe extern "C" fn cpp_eam_table_callback(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
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
        let potential = unsafe { &*user_data.cast::<CppEamTable>() };
        let cell = if potential.periodic {
            copy_f64_tensor(input.box_matrix, "box_matrix", 9, true)
        } else {
            Ok(vec![0.0; 9])
        };
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
                potential.handle,
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
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_i32_1d, rgpot_tensor_free};
    use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

    fn minimal_table() -> EamTable {
        let text = concat!(
            "c\nc\nc\n",
            "1 Cu\n",
            "5 0.5 5 0.25 1.0\n",
            "29 63.546 3.615 fcc\n",
            "0.0 0.0 0.0 0.0 0.0\n",
            "0.0 0.0 0.0 0.0 0.0\n",
            "0.0 0.0 0.0 0.0 0.0\n",
        );
        EamTable::from_text(text, EamTableFormat::FinnisSinclair).unwrap()
    }

    fn input_without_cell() -> (rgpot_force_input_t, rgpot_force_out_t) {
        let positions = Box::leak(Box::new([0.0_f64, 0.0, 0.0, 0.5, 0.0, 0.0]));
        let numbers = Box::leak(Box::new([29_i32, 29]));
        let input = rgpot_force_input_t {
            positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 2, 3) },
            atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 2) },
            box_matrix: std::ptr::null_mut(),
        };
        let output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        (input, output)
    }

    unsafe fn free_io(input: &mut rgpot_force_input_t, output: &mut rgpot_force_out_t) {
        unsafe {
            rgpot_tensor_free(input.positions);
            rgpot_tensor_free(input.atomic_numbers);
            rgpot_tensor_free(output.forces);
        }
    }

    #[test]
    fn nonperiodic_callback_does_not_require_a_box() {
        let _lock = CPP_TEST_LOCK.lock().unwrap();
        let potential = cpp_potential_from_data(minimal_table(), false, false, 1.0).unwrap();
        let (mut input, mut output) = input_without_cell();
        let status = unsafe { potential.calculate(&input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        unsafe { free_io(&mut input, &mut output) };
    }

    #[test]
    fn callback_catches_panics() {
        let _lock = CPP_TEST_LOCK.lock().unwrap();
        let potential = cpp_potential_from_data(minimal_table(), false, false, 1.0).unwrap();
        let (mut input, mut output) = input_without_cell();
        unsafe {
            *(*input.positions).dl_tensor.shape = i64::MAX;
        }
        let status = unsafe { potential.calculate(&input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INTERNAL_ERROR);
        unsafe { free_io(&mut input, &mut output) };
    }

    #[test]
    fn cpp_force_rejects_a_null_cell() {
        let _lock = CPP_TEST_LOCK.lock().unwrap();
        let text = CString::new(minimal_table().to_finnis_sinclair_text()).unwrap();
        let handle = unsafe {
            rgpot_core_eam_cpp_new_from_text(
                text.as_ptr(),
                RGPOT_EAM_TABLE_FINNIS_SINCLAIR,
                false,
                false,
                1.0,
            )
        };
        assert!(!handle.is_null());
        let positions = [0.0_f64, 0.0, 0.0];
        let numbers = [29_i32];
        let mut forces = [0.0_f64; 3];
        let mut energy = 0.0;
        let status = unsafe {
            rgpot_core_eam_cpp_force(
                handle,
                1,
                positions.as_ptr(),
                numbers.as_ptr(),
                std::ptr::null(),
                forces.as_mut_ptr(),
                &mut energy,
            )
        };
        assert_eq!(status, 1);
        unsafe { rgpot_core_eam_cpp_free(handle) };
    }
}
