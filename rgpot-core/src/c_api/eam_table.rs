// MIT License
// Copyright 2023--present rgpot developers

//! C API constructor of the tabulated EAM potential.
//!
//! ```c
//! rgpot_potential_t *pot =
//!     rgpot_potential_new_eam_table("Cu_u3.eam", RGPOT_EAM_TABLE_FUNCFL, true);
//! if (pot == NULL) {
//!     fprintf(stderr, "rgpot error: %s\n", rgpot_last_error());
//! }
//! rgpot_status_t s = rgpot_potential_calculate(pot, &input, &output);
//! rgpot_potential_free(pot);
//! ```

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;

use crate::eam_table::{
    drop_eam_table, eam_table_callback, EamTableError, EamTableFormat, EamTablePotential,
};
use crate::eindir::rgpot_potential_t;
use crate::status::set_last_error;

use super::potential::rgpot_potential_new;

fn path_of(path: *const c_char) -> Result<PathBuf, EamTableError> {
    if path.is_null() {
        return Err(EamTableError::Invalid("path is NULL".into()));
    }
    let bytes = unsafe { CStr::from_ptr(path) }.to_bytes();
    if bytes.is_empty() {
        return Err(EamTableError::Invalid("path is empty".into()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Ok(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
    }
    #[cfg(not(unix))]
    {
        match std::str::from_utf8(bytes) {
            Ok(text) => Ok(PathBuf::from(text)),
            Err(_) => Err(EamTableError::Invalid("path is not UTF-8".into())),
        }
    }
}

fn build(
    path: *const c_char,
    format: c_int,
    periodic: bool,
) -> Result<EamTablePotential, EamTableError> {
    let format = EamTableFormat::from_code(format)?;
    EamTablePotential::open(&path_of(path)?, format, periodic)
}

/// Create a potential that evaluates a tabulated EAM table in process.
///
/// Reads the DYNAMO table at `path` and returns a handle that evaluates its
/// energy in eV and its forces in eV/Angstrom through
/// `rgpot_potential_calculate`. Positions and the box are in Angstrom, the
/// box is the lattice vectors as rows, and an atom takes its element from its
/// atomic number, matched against the first field of the element lines of the
/// file.
///
/// - `path`: NUL-terminated path of the table file.
/// - `format`: `RGPOT_EAM_TABLE_FUNCFL` (`pair_style eam`),
///   `RGPOT_EAM_TABLE_SETFL` (`eam/alloy`), `RGPOT_EAM_TABLE_FINNIS_SINCLAIR`
///   (`eam/fs`), or `RGPOT_EAM_TABLE_AUTO` to choose from the suffix of the
///   file name (`.eam`, `.eam.alloy` or `.setfl`, `.eam.fs`).
/// - `periodic`: keep the nearest image of each partner. A calculation then
///   needs a box whose face spacings all exceed twice the cutoff of the file,
///   and is refused with `RGPOT_INVALID_PARAMETER` for a smaller one. With
///   `false` the box is ignored and the atoms are a free cluster.
///
/// Returns a heap-allocated `rgpot_potential_t*`, or `NULL` on failure with
/// the reason in `rgpot_last_error`: a NULL or empty `path`, an unknown
/// `format`, a file that cannot be opened, or one that is not a table of that
/// format. The caller must eventually pass the returned pointer to
/// `rgpot_potential_free`.
///
/// # Safety
///
/// `path` must be NULL or point to a NUL-terminated string that stays valid
/// for the call.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_new_eam_table(
    path: *const c_char,
    format: c_int,
    periodic: bool,
) -> *mut rgpot_potential_t {
    let made = std::panic::catch_unwind(|| build(path, format, periodic));
    let potential = match made {
        Ok(Ok(potential)) => potential,
        Ok(Err(why)) => {
            set_last_error(&why.to_string());
            return std::ptr::null_mut();
        }
        Err(_) => {
            set_last_error("rgpot_potential_new_eam_table: internal error");
            return std::ptr::null_mut();
        }
    };
    let user_data = Box::into_raw(Box::new(potential)).cast();
    let handle =
        unsafe { rgpot_potential_new(eam_table_callback, user_data, Some(drop_eam_table)) };
    if handle.is_null() {
        unsafe { drop_eam_table(user_data) };
        set_last_error("rgpot_potential_new_eam_table: the handle could not be allocated");
    }
    handle
}
