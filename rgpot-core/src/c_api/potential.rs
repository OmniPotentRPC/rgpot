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
//! rgpot_force_out_t    output = rgpot_force_out_create();
//!
//! // 3. Calculate
//! rgpot_status_t s = rgpot_potential_calculate(pot, &input, &output);
//! if (s != RGPOT_SUCCESS) { /* handle error */ }
//! // output.forces is now a DLPack tensor — use it, then free:
//! rgpot_tensor_free(output.forces);
//!
//! // 4. Clean up
//! rgpot_force_input_free(&input);
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
/// - `input`: pointer to the input configuration (DLPack tensors).
/// - `output`: pointer to the output struct. The callback sets `output->forces`
///   to a callee-allocated DLPack tensor.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{create_owned_f64_tensor, rgpot_tensor_free};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    /// A trivial callback that sets energy = n_atoms and returns success.
    unsafe extern "C" fn sum_callback(
        _ud: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };
        let n = unsafe { inp.n_atoms() }.unwrap_or(0);
        out.energy = n as f64;
        out.variance = 0.0;
        out.forces = create_owned_f64_tensor(vec![0.0; n * 3], vec![n as i64, 3]);
        rgpot_status_t::RGPOT_SUCCESS
    }

    /// A callback that always returns an error.
    unsafe extern "C" fn failing_callback(
        _ud: *mut c_void,
        _input: *const rgpot_force_input_t,
        _output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        crate::status::set_last_error("deliberately failed");
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    }

    /// A callback that reads user_data as a counter and increments it.
    unsafe extern "C" fn counting_callback(
        ud: *mut c_void,
        _input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let counter = unsafe { &*(ud as *const AtomicU32) };
        counter.fetch_add(1, Ordering::SeqCst);
        let out = unsafe { &mut *output };
        out.energy = counter.load(Ordering::SeqCst) as f64;
        rgpot_status_t::RGPOT_SUCCESS
    }

    fn make_test_input() -> (
        [f64; 6],
        [i32; 2],
        [f64; 9],
        rgpot_force_input_t,
    ) {
        let mut pos = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut atmnrs = [1_i32, 1];
        let mut box_ = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let input = unsafe {
            crate::c_api::types::rgpot_force_input_create(
                2,
                pos.as_mut_ptr(),
                atmnrs.as_mut_ptr(),
                box_.as_mut_ptr(),
            )
        };
        (pos, atmnrs, box_, input)
    }

    unsafe fn cleanup(input: &mut rgpot_force_input_t, output: &mut rgpot_force_out_t) {
        unsafe {
            rgpot_tensor_free(output.forces);
            crate::c_api::types::rgpot_force_input_free(input);
        }
        output.forces = std::ptr::null_mut();
    }

    // --- rgpot_potential_new / rgpot_potential_free ---

    #[test]
    fn new_returns_non_null() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        assert!(!pot.is_null());
        unsafe { rgpot_potential_free(pot) };
    }

    #[test]
    fn free_null_is_noop() {
        unsafe { rgpot_potential_free(std::ptr::null_mut()) };
    }

    // --- rgpot_potential_calculate: null argument handling ---

    #[test]
    fn calculate_null_pot_returns_invalid_parameter() {
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };

        let status = unsafe {
            rgpot_potential_calculate(std::ptr::null(), &input, &mut output)
        };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe { crate::c_api::types::rgpot_force_input_free(&mut input) };
    }

    #[test]
    fn calculate_null_input_returns_invalid_parameter() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };

        let status = unsafe {
            rgpot_potential_calculate(pot, std::ptr::null(), &mut output)
        };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe { rgpot_potential_free(pot as *mut _) };
    }

    #[test]
    fn calculate_null_output_returns_invalid_parameter() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();

        let status = unsafe {
            rgpot_potential_calculate(pot, &input, std::ptr::null_mut())
        };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot as *mut _);
        }
    }

    // --- rgpot_potential_calculate: success path ---

    #[test]
    fn full_lifecycle_new_calculate_free() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };

        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(output.energy, 2.0); // n_atoms = 2
        assert!(!output.forces.is_null());

        unsafe {
            cleanup(&mut input, &mut output);
            rgpot_potential_free(pot as *mut _);
        }
    }

    // --- Error propagation from callback ---

    #[test]
    fn callback_error_propagates() {
        let pot = unsafe {
            rgpot_potential_new(failing_callback, std::ptr::null_mut(), None)
        };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };

        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INTERNAL_ERROR);

        let msg = unsafe { std::ffi::CStr::from_ptr(crate::status::rgpot_last_error()) };
        assert_eq!(msg.to_str().unwrap(), "deliberately failed");

        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot as *mut _);
        }
    }

    // --- user_data passthrough ---

    #[test]
    fn user_data_is_forwarded_to_callback() {
        let counter = AtomicU32::new(0);
        let pot = unsafe {
            rgpot_potential_new(
                counting_callback,
                &counter as *const _ as *mut c_void,
                None,
            )
        };

        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };

        unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(output.energy, 1.0);

        unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert_eq!(output.energy, 2.0);

        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot as *mut _);
        }
    }

    // --- free_fn is called on drop ---

    static FREE_CALLED: AtomicBool = AtomicBool::new(false);

    unsafe extern "C" fn track_free(_ptr: *mut c_void) {
        FREE_CALLED.store(true, Ordering::SeqCst);
    }

    #[test]
    fn free_fn_is_invoked_on_drop() {
        FREE_CALLED.store(false, Ordering::SeqCst);

        let mut dummy: u8 = 42;
        let pot = unsafe {
            rgpot_potential_new(
                sum_callback,
                &mut dummy as *mut u8 as *mut c_void,
                Some(track_free),
            )
        };

        assert!(!FREE_CALLED.load(Ordering::SeqCst));
        unsafe { rgpot_potential_free(pot) };
        assert!(FREE_CALLED.load(Ordering::SeqCst));
    }

    // --- Multiple sequential calculations on the same handle ---

    #[test]
    fn multiple_calculations_same_handle() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };

        for n in 1..=5_usize {
            let mut pos = vec![0.0_f64; n * 3];
            let mut atmnrs = vec![1_i32; n];
            let mut box_ = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

            let mut input = unsafe {
                crate::c_api::types::rgpot_force_input_create(
                    n,
                    pos.as_mut_ptr(),
                    atmnrs.as_mut_ptr(),
                    box_.as_mut_ptr(),
                )
            };
            let mut output = rgpot_force_out_t {
                forces: std::ptr::null_mut(),
                energy: 0.0,
                variance: 0.0,
            };

            let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
            assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
            assert_eq!(output.energy, n as f64);

            unsafe { cleanup(&mut input, &mut output) };
        }

        unsafe { rgpot_potential_free(pot as *mut _) };
    }
}
