// MIT License
// Copyright 2023--present rgpot developers

//! C API for the potential handle lifecycle: create, calculate, free.
//!
//! These three functions form the core public interface for potential energy
//! calculations. The typical usage pattern from C/C++ is:
//!
//! ```c
//! // 1. Create a handle from a callback
//! rgpot_potential_t *pot = rgpot_potential_new(my_callback, my_data, NULL);
//!
//! // 2. Prepare input/output
//! rgpot_force_input_t  input  = rgpot_force_input_create(n, pos, types, box);
//! double forces[n * 3];
//! rgpot_force_out_t    output = rgpot_force_out_create(forces);
//!
//! // 3. Calculate
//! rgpot_status_t s = rgpot_potential_calculate(pot, &input, &output);
//! if (s != RGPOT_SUCCESS) { /* handle error */ }
//!
//! // 4. Clean up
//! rgpot_potential_free(pot);
//! ```

use std::os::raw::c_void;

use crate::potential::{PotentialCallback, PotentialImpl, rgpot_potential_t};
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Create a new potential handle from a callback function pointer.
///
/// - `callback`: the function that performs the force/energy calculation.
/// - `user_data`: opaque pointer forwarded to every callback invocation
///   (typically a pointer to the C++ potential object).
/// - `free_fn`: optional destructor for `user_data`. Pass `NULL` if the
///   caller manages the lifetime externally.
///
/// Returns a heap-allocated `rgpot_potential_t*`, or `NULL` on failure.
/// The caller must eventually pass the returned pointer to
/// `rgpot_potential_free`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_new(
    callback: PotentialCallback,
    user_data: *mut c_void,
    free_fn: Option<unsafe extern "C" fn(*mut c_void)>,
) -> *mut rgpot_potential_t {
    let pot = PotentialImpl::new(callback, user_data, free_fn);
    Box::into_raw(Box::new(pot))
}

/// Perform a force/energy calculation using the potential handle.
///
/// - `pot`: a valid handle obtained from `rgpot_potential_new`.
/// - `input`: pointer to the input configuration.
/// - `output`: pointer to the output buffer (forces must be pre-allocated).
///
/// Returns `RGPOT_SUCCESS` on success, or an error status code.
/// On error, call `rgpot_last_error()` for details.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_calculate(
    pot: *const rgpot_potential_t,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() {
            set_last_error("rgpot_potential_calculate: pot is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        if input.is_null() {
            set_last_error("rgpot_potential_calculate: input is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        if output.is_null() {
            set_last_error("rgpot_potential_calculate: output is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }

        let pot_ref = unsafe { &*pot };
        unsafe { pot_ref.calculate(input, output) }
    }))
}

/// Free a potential handle previously obtained from `rgpot_potential_new`.
///
/// If `pot` is `NULL`, this function is a no-op.
/// After this call, `pot` must not be used again.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_free(pot: *mut rgpot_potential_t) {
    if !pot.is_null() {
        drop(unsafe { Box::from_raw(pot) });
    }
}
