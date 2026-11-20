// MIT License
// Copyright 2023--present rgpot developers

//! C API for the potential handle lifecycle: create, calculate, free.
//!
//! `rgpot_potential_t` is the superset struct defined in `crate::eindir`:
//! it embeds `eindir_objective_t` as its first member so every
//! `rgpot_potential_t*` IS-A `eindir_objective_t*` at the C ABI.
//!
//! The three lifecycle functions here use the direct callback path
//! (`pot.callback`); the eindir evaluation path goes through
//! `eindir_objective_eval` with the zero-cost cast.
//!
//! ```c
//! // Direct rgpot path:
//! rgpot_potential_t *pot = rgpot_potential_new(my_callback, my_data, NULL);
//! rgpot_status_t s = rgpot_potential_calculate(pot, &input, &output);
//! rgpot_potential_free(pot);
//!
//! // eindir path (IS-A cast, no conversion method):
//! eindir_objective_t *obj = (eindir_objective_t*)pot;
//! double val;
//! eindir_objective_eval(obj, x_tensor, &val);
//! ```

use std::os::raw::c_void;

use crate::eindir::rgpot_potential_t;
use crate::potential::PotentialCallback;
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t, rgpot_stress_out_t};

/// Create a new potential handle from a callback function pointer.
///
/// Creates a `rgpot_potential_t` with no molecular context (n_atoms = 0) and
/// no eindir bounds.  The potential evaluates through `callback` directly.
/// Use `rgpot_potential_new_eindir` when the IS-A eindir embedding is needed.
///
/// Returns a heap-allocated `rgpot_potential_t*`, or `NULL` on failure.
/// The caller must eventually pass the returned pointer to `rgpot_potential_free`.
///
/// # Safety
///
/// `callback` and `free_fn` must be safe to call with `user_data` for as
/// long as the handle lives.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_new(
    callback: PotentialCallback,
    user_data: *mut c_void,
    free_fn: Option<unsafe extern "C" fn(*mut c_void)>,
) -> *mut rgpot_potential_t {
    use crate::eindir::rgpot_potential_new_eindir;
    // Create a superset struct with no molecular context.
    unsafe {
        rgpot_potential_new_eindir(
            callback,
            user_data,
            free_fn,
            0,                // n_atoms = 0
            std::ptr::null(), // atomic_numbers
            std::ptr::null(), // box_matrix
            std::ptr::null(), // bounds_low
            std::ptr::null(), // bounds_high
        )
    }
}

/// Perform a force/energy calculation using the potential handle.
///
/// Routes through the direct rgpot callback (`pot.callback`), bypassing the
/// eindir evaluation path.
///
/// # Safety
///
/// A NULL `pot`, `input` or `output` returns `RGPOT_INVALID_PARAMETER`.
/// Otherwise `pot` must be a live handle from `rgpot_potential_new` or
/// `rgpot_potential_new_eindir`, `input` must point to a valid
/// `rgpot_force_input_t` whose tensors stay alive for the call, and `output`
/// must point to writable `rgpot_force_out_t` storage.
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
        let p = unsafe { &*pot };
        if let Some(why) = p.fused_cache_peer_refusal() {
            set_last_error(&why);
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        unsafe { (p.callback)(p.pot_user_data, input, output) }
    }))
}

/// Install the callback that also computes stress for `pot`, or clear it
/// with NULL.
///
/// The callback receives the `user_data` the handle was created with, so one
/// potential object answers both the force-only and the stress request. A
/// potential that computes no stress registers none, and
/// `rgpot_potential_calculate_stress` then reports that instead of returning
/// zeros.
///
/// # Safety
///
/// A NULL `pot` returns `RGPOT_INVALID_PARAMETER`. `callback` must be safe
/// to call with the handle's `user_data` for as long as it stays registered.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_set_stress_callback(
    pot: *const rgpot_potential_t,
    callback: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            input: *const rgpot_force_input_t,
            output: *mut rgpot_force_out_t,
            stress: *mut rgpot_stress_out_t,
        ) -> rgpot_status_t,
    >,
) -> rgpot_status_t {
    if pot.is_null() {
        set_last_error("rgpot_potential_set_stress_callback: pot is NULL");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    unsafe { &*pot }.set_stress_callback(callback);
    rgpot_status_t::RGPOT_SUCCESS
}

/// Nonzero when `pot` computes Cauchy stress, that is when a stress
/// callback is registered. A NULL `pot` returns zero.
///
/// # Safety
///
/// `pot` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_supports_stress(pot: *const rgpot_potential_t) -> i32 {
    if pot.is_null() {
        return 0;
    }
    i32::from(unsafe { &*pot }.stress_callback().is_some())
}

/// Build a zeroed stress block with `struct_size` set for this build.
#[no_mangle]
pub extern "C" fn rgpot_stress_out_create() -> rgpot_stress_out_t {
    rgpot_stress_out_t::default()
}

/// Perform a force/energy calculation that also returns the Cauchy stress.
///
/// This is `rgpot_potential_calculate` plus the stress block. It is a
/// separate entry point so `rgpot_force_input_t` and `rgpot_force_out_t`
/// keep the layout callers built against ABI minor 6 already use.
///
/// `stress->struct_size` must be at least `sizeof(rgpot_stress_out_t)` of
/// the header the library was built with, or the call returns
/// `RGPOT_INVALID_PARAMETER` without touching the block. A potential with no
/// stress callback also returns `RGPOT_INVALID_PARAMETER` and a message that
/// names the missing capability; the block stays zero. On success
/// `stress->has_stress` is nonzero only when every component was computed,
/// for example it stays zero for a singular cell.
///
/// # Safety
///
/// A NULL `pot`, `input`, `output` or `stress` returns
/// `RGPOT_INVALID_PARAMETER`. Otherwise the pointers must satisfy the
/// requirements of `rgpot_potential_calculate`, and `stress` must point to
/// writable storage of `stress->struct_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_calculate_stress(
    pot: *const rgpot_potential_t,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
    stress: *mut rgpot_stress_out_t,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() || input.is_null() || output.is_null() || stress.is_null() {
            set_last_error("rgpot_potential_calculate_stress: NULL argument");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let size = unsafe { std::ptr::addr_of!((*stress).struct_size).read() };
        if size < rgpot_stress_out_t::SIZE {
            set_last_error(&format!(
                "rgpot_potential_calculate_stress: stress block of {size} bytes is smaller than \
                 the {} bytes this library writes; set struct_size = sizeof(rgpot_stress_out_t)",
                rgpot_stress_out_t::SIZE
            ));
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let p = unsafe { &*pot };
        if let Some(why) = p.fused_cache_peer_refusal() {
            set_last_error(&why);
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let Some(callback) = p.stress_callback() else {
            set_last_error(
                "rgpot_potential_calculate_stress: this potential does not compute stress",
            );
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        };
        let block = unsafe { &mut *stress };
        block.has_stress = 0;
        block.stress = [0.0; 9];
        let status = unsafe { callback(p.pot_user_data, input, output, stress) };
        if status != rgpot_status_t::RGPOT_SUCCESS {
            block.has_stress = 0;
            block.stress = [0.0; 9];
        }
        status
    }))
}

/// Free a potential handle previously obtained from `rgpot_potential_new` or
/// `rgpot_potential_new_eindir`.
///
/// Calls `pot_free_fn(pot_user_data)` if provided, frees all owned arrays,
/// then frees the struct.
///
/// # Safety
///
/// `pot` must be NULL or a pointer returned by one of those constructors
/// that has not been freed yet. The pointer is invalid after the call.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_free(pot: *mut rgpot_potential_t) {
    if pot.is_null() {
        return;
    }
    let p = unsafe { &*pot };
    if let Some(ff) = p.pot_free_fn {
        if !p.pot_user_data.is_null() {
            unsafe { ff(p.pot_user_data) };
        }
    }
    let dim = p.base.dim;
    if !p.base.low.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.base.low, dim, dim)) };
    }
    if !p.base.high.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.base.high, dim, dim)) };
    }
    if !p.atomic_numbers.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.atomic_numbers, p.n_atoms, p.n_atoms)) };
    }
    unsafe { drop(Box::from_raw(pot)) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{create_owned_f64_tensor, rgpot_tensor_free};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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

    unsafe extern "C" fn failing_callback(
        _ud: *mut c_void,
        _input: *const rgpot_force_input_t,
        _output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        crate::status::set_last_error("deliberately failed");
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    }

    unsafe extern "C" fn stress_callback(
        _ud: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
        stress: *mut rgpot_stress_out_t,
    ) -> rgpot_status_t {
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };
        let n = unsafe { inp.n_atoms() }.unwrap_or(0);
        out.forces = create_owned_f64_tensor(vec![0.0; n * 3], vec![n as i64, 3]);
        out.energy = n as f64;
        let block = unsafe { &mut *stress };
        block.stress = [1.0, 0.2, 0.3, 0.2, 2.0, 0.4, 0.3, 0.4, 3.0];
        block.has_stress = 1;
        rgpot_status_t::RGPOT_SUCCESS
    }

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

    fn make_test_input() -> ([f64; 6], [i32; 2], [f64; 9], rgpot_force_input_t) {
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

    #[test]
    fn calculate_null_pot_returns_invalid_parameter() {
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t::default();
        let status = unsafe { rgpot_potential_calculate(std::ptr::null(), &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe { crate::c_api::types::rgpot_force_input_free(&mut input) };
    }

    #[test]
    fn calculate_null_input_returns_invalid_parameter() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let mut output = rgpot_force_out_t::default();
        let status = unsafe { rgpot_potential_calculate(pot, std::ptr::null(), &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe { rgpot_potential_free(pot as *mut _) };
    }

    #[test]
    fn calculate_null_output_returns_invalid_parameter() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let status = unsafe { rgpot_potential_calculate(pot, &input, std::ptr::null_mut()) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot as *mut _);
        }
    }

    #[test]
    fn full_lifecycle_new_calculate_free() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t::default();
        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(output.energy, 2.0);
        assert!(!output.forces.is_null());
        unsafe {
            cleanup(&mut input, &mut output);
            rgpot_potential_free(pot as *mut _);
        }
    }

    #[test]
    fn stress_has_its_own_entry_point_and_force_calls_never_carry_it() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        unsafe { rgpot_potential_set_stress_callback(pot, Some(stress_callback)) };
        assert_eq!(unsafe { rgpot_potential_supports_stress(pot) }, 1);
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();

        let mut output = rgpot_force_out_t::default();
        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(output.energy, 2.0);
        unsafe { rgpot_tensor_free(output.forces) };
        output.forces = std::ptr::null_mut();

        let mut stress = rgpot_stress_out_create();
        assert_eq!(stress.struct_size, rgpot_stress_out_t::SIZE);
        let status =
            unsafe { rgpot_potential_calculate_stress(pot, &input, &mut output, &mut stress) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(stress.has_stress, 1);
        assert_eq!(stress.stress[0], 1.0);
        assert_eq!(stress.stress[8], 3.0);
        unsafe {
            cleanup(&mut input, &mut output);
            rgpot_potential_free(pot);
        }
    }

    #[test]
    fn stress_request_without_a_stress_callback_is_an_error_not_zeros() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        assert_eq!(unsafe { rgpot_potential_supports_stress(pot) }, 0);
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t::default();
        let mut stress = rgpot_stress_out_create();
        let status =
            unsafe { rgpot_potential_calculate_stress(pot, &input, &mut output, &mut stress) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        assert!(output.forces.is_null());
        assert_eq!(stress.has_stress, 0);
        assert_eq!(stress.stress, [0.0; 9]);
        let msg = unsafe { std::ffi::CStr::from_ptr(crate::status::rgpot_last_error()) };
        assert!(msg.to_string_lossy().contains("does not compute stress"));
        unsafe {
            cleanup(&mut input, &mut output);
            rgpot_potential_free(pot);
        }
    }

    #[test]
    fn undersized_stress_block_is_refused_before_it_is_written() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        unsafe { rgpot_potential_set_stress_callback(pot, Some(stress_callback)) };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t::default();
        let mut stress = rgpot_stress_out_create();
        stress.struct_size = rgpot_stress_out_t::SIZE - 1;
        stress.stress = [7.0; 9];
        let status =
            unsafe { rgpot_potential_calculate_stress(pot, &input, &mut output, &mut stress) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        assert!(output.forces.is_null());
        assert_eq!(stress.stress, [7.0; 9]);
        unsafe {
            cleanup(&mut input, &mut output);
            rgpot_potential_free(pot);
        }
    }

    #[test]
    fn null_stress_block_is_refused() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t::default();
        let status = unsafe {
            rgpot_potential_calculate_stress(pot, &input, &mut output, std::ptr::null_mut())
        };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe {
            cleanup(&mut input, &mut output);
            rgpot_potential_free(pot);
        }
    }

    /// The two ABI structs of minor 6, frozen here as a caller built against
    /// that header lays them out. A canary follows each one in memory: a
    /// library that wrote a field it added later would clobber it.
    #[repr(C)]
    struct Minor6Input {
        positions: *mut dlpk::sys::DLManagedTensorVersioned,
        atomic_numbers: *mut dlpk::sys::DLManagedTensorVersioned,
        box_matrix: *mut dlpk::sys::DLManagedTensorVersioned,
        canary: [u64; 16],
    }

    #[repr(C)]
    struct Minor6Out {
        forces: *mut dlpk::sys::DLManagedTensorVersioned,
        energy: f64,
        variance: f64,
        canary: [u64; 16],
    }

    const CANARY: u64 = 0xA5A5_5A5A_DEAD_BEEF;

    #[test]
    fn minor_6_structs_keep_their_size_and_still_evaluate() {
        assert_eq!(
            std::mem::size_of::<rgpot_force_input_t>(),
            3 * std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<rgpot_force_out_t>(),
            std::mem::size_of::<usize>() + 2 * std::mem::size_of::<f64>()
        );
        assert_eq!(
            std::mem::offset_of!(Minor6Input, canary),
            std::mem::size_of::<rgpot_force_input_t>()
        );
        assert_eq!(
            std::mem::offset_of!(Minor6Out, canary),
            std::mem::size_of::<rgpot_force_out_t>()
        );

        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let (_pos, _atmnrs, _box_, mut built) = make_test_input();
        let input = Minor6Input {
            positions: built.positions,
            atomic_numbers: built.atomic_numbers,
            box_matrix: built.box_matrix,
            canary: [CANARY; 16],
        };
        let mut out = Minor6Out {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
            canary: [CANARY; 16],
        };
        let status = unsafe {
            rgpot_potential_calculate(
                pot,
                (&input as *const Minor6Input).cast(),
                (&mut out as *mut Minor6Out).cast(),
            )
        };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(out.energy, 2.0);
        assert!(!out.forces.is_null());
        assert_eq!(input.canary, [CANARY; 16]);
        assert_eq!(out.canary, [CANARY; 16]);
        unsafe {
            rgpot_tensor_free(out.forces);
            crate::c_api::types::rgpot_force_input_free(&mut built);
            rgpot_potential_free(pot);
        }
    }

    #[test]
    fn callback_error_propagates() {
        let pot = unsafe { rgpot_potential_new(failing_callback, std::ptr::null_mut(), None) };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t::default();
        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INTERNAL_ERROR);
        let msg = unsafe { std::ffi::CStr::from_ptr(crate::status::rgpot_last_error()) };
        assert_eq!(msg.to_str().unwrap(), "deliberately failed");
        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot as *mut _);
        }
    }

    #[test]
    fn user_data_is_forwarded_to_callback() {
        let counter = AtomicU32::new(0);
        let pot = unsafe {
            rgpot_potential_new(counting_callback, &counter as *const _ as *mut c_void, None)
        };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t::default();
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
            let mut output = rgpot_force_out_t::default();
            let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
            assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
            assert_eq!(output.energy, n as f64);
            unsafe { cleanup(&mut input, &mut output) };
        }
        unsafe { rgpot_potential_free(pot as *mut _) };
    }
}
