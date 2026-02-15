// MIT License
// Copyright 2023--present rgpot developers

//! Convenience constructors for the core C types.
//!
//! The struct definitions themselves live in [`crate::types`]. This module
//! provides `extern "C"` factory functions so that C callers can construct the
//! structs without relying on designated initializers (which are C99 only and
//! not available in some MSVC configurations).
//!
//! These functions are thin wrappers that perform no validation — the caller
//! is responsible for providing valid pointers with sufficient backing storage.

use crate::types::{rgpot_force_input_t, rgpot_force_out_t};
use std::os::raw::c_int;

/// Create a `rgpot_force_input_t` from raw pointers.
///
/// This is a convenience constructor for C callers. All pointers
/// are borrowed — the caller retains ownership.
///
/// # Safety
/// All pointers must be valid for the lifetime of the returned struct.
/// `pos` must point to at least `n_atoms * 3` doubles.
/// `atmnrs` must point to at least `n_atoms` ints.
/// `box_` must point to at least 9 doubles.
#[no_mangle]
pub unsafe extern "C" fn rgpot_force_input_create(
    n_atoms: usize,
    pos: *const f64,
    atmnrs: *const c_int,
    box_: *const f64,
) -> rgpot_force_input_t {
    rgpot_force_input_t {
        n_atoms,
        pos,
        atmnrs,
        box_,
    }
}

/// Create a `rgpot_force_out_t` with zeroed energy and variance.
///
/// # Safety
/// `forces` must point to a buffer of at least `n_atoms * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn rgpot_force_out_create(
    forces: *mut f64,
) -> rgpot_force_out_t {
    rgpot_force_out_t {
        forces,
        energy: 0.0,
        variance: 0.0,
    }
}
