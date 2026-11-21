// MIT License
// Copyright 2023--present rgpot developers

//! C API for the dense energy Hessian of a potential handle.
//!
//! A registered Hessian callback is used as written. With only an analytic
//! Hessian-vector callback, the matrix is that product on each unit
//! direction. With neither, the call returns `RGPOT_NO_HESSIAN` and does not
//! difference forces.

use std::os::raw::c_void;

use crate::eindir::rgpot_potential_t;
use crate::hessian::{rgpot_hessian_counts_t, HessianCallback, HessianError};
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
use crate::types::rgpot_force_input_t;

fn fail(err: HessianError) -> rgpot_status_t {
    match &err {
        HessianError::Invalid(why) | HessianError::NoHessian(why) => set_last_error(why),
        HessianError::Callback { message, .. } => set_last_error(message),
    }
    err.status()
}

fn refuse(msg: &str) -> rgpot_status_t {
    set_last_error(msg);
    rgpot_status_t::RGPOT_INVALID_PARAMETER
}

/// Register `callback` as the analytic dense energy Hessian of `pot`, or
/// clear it with NULL.
///
/// The callback receives the `user_data` the handle was created with. It
/// writes `(3 * n_atoms)²` row-major doubles, `d²E/dx_i dx_j`. With a
/// callback registered, `rgpot_potential_hessian` calls it and
/// `RGPOT_HVP_AUTO` products multiply the matrix when no Hessian-vector
/// callback is registered. Neither path differences forces.
///
/// Replacing or clearing the callback returns only after every call of the
/// previous callback that was already running has returned, and the previous
/// callback is never called after this function returns. The caller may
/// release the old callback and its state as soon as this function returns.
/// Because it waits for those calls, this function must not be called from
/// inside a Hessian callback, nor from a thread that is inside this handle's
/// Hessian or Hessian-vector entry points: it would wait for itself.
///
/// # Safety
///
/// `pot` must be NULL or a live handle. A NULL `pot` returns
/// `RGPOT_INVALID_PARAMETER`. `callback` must be safe to call with the
/// handle's `user_data` for as long as it stays registered, and must not
/// re-enter this handle's Hessian entry points or replace the callback.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_set_hessian_callback(
    pot: *const rgpot_potential_t,
    callback: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            input: *const rgpot_force_input_t,
            hessian_out: *mut f64,
        ) -> rgpot_status_t,
    >,
) -> rgpot_status_t {
    let callback: Option<HessianCallback> = callback;
    if pot.is_null() {
        return refuse("rgpot_potential_set_hessian_callback: pot is NULL");
    }
    unsafe { &*pot }.set_hessian_callback(callback);
    rgpot_status_t::RGPOT_SUCCESS
}

/// Install or clear the caller's dense-Hessian byte budget.
///
/// A NULL `max_bytes` clears the budget. No component count or byte size is
/// assumed in its place: a request is then refused only when the byte count
/// does not fit in `size_t`, or when the allocation itself fails. A non-NULL
/// pointer installs that many bytes, and zero is a budget that refuses every
/// positive matrix. The pointer is read once; it need not stay alive.
///
/// # Safety
///
/// `pot` must be NULL or a live handle. A NULL `pot` returns
/// `RGPOT_INVALID_PARAMETER`. A non-NULL `max_bytes` must point at a readable
/// `uint64_t`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_set_hessian_byte_budget(
    pot: *const rgpot_potential_t,
    max_bytes: *const u64,
) -> rgpot_status_t {
    if pot.is_null() {
        return refuse("rgpot_potential_set_hessian_byte_budget: pot is NULL");
    }
    let budget = if max_bytes.is_null() {
        None
    } else {
        Some(unsafe { *max_bytes })
    };
    unsafe { &*pot }.set_hessian_byte_budget(budget);
    rgpot_status_t::RGPOT_SUCCESS
}

/// Write the energy Hessian at the geometry of `input` to `hessian_out`.
///
/// `hessian_out` holds `(3 * n_atoms)²` doubles, row-major. The matrix comes
/// from the Hessian callback when one is registered, otherwise from the
/// analytic Hessian-vector callback applied to each unit direction. A handle
/// with neither returns `RGPOT_NO_HESSIAN`. A repeated request at the same
/// geometry, species, cell and engine settings is the cached matrix.
///
/// # Safety
///
/// NULL pointers return `RGPOT_INVALID_PARAMETER`. Otherwise `pot` must be a
/// live handle, `input` must point to a valid `rgpot_force_input_t` whose
/// tensors stay alive for the call, and `hessian_out` must point to
/// `(3 * n_atoms)²` writable doubles that do not overlap the position buffer.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_hessian(
    pot: *const rgpot_potential_t,
    input: *const rgpot_force_input_t,
    hessian_out: *mut f64,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() {
            return refuse("rgpot_potential_hessian: pot is NULL");
        }
        if input.is_null() {
            return refuse("rgpot_potential_hessian: input is NULL");
        }
        if hessian_out.is_null() {
            return refuse("rgpot_potential_hessian: hessian_out is NULL");
        }
        let input = unsafe { &*input };
        let Some(dim) = (unsafe { input.n_atoms() }).and_then(|n| n.checked_mul(3)) else {
            return refuse("rgpot_potential_hessian: positions tensor is NULL or too large");
        };
        let Some(n2) = dim.checked_mul(dim) else {
            return refuse("rgpot_potential_hessian: Hessian dimension overflows");
        };
        let Some(bytes) = n2.checked_mul(std::mem::size_of::<f64>()) else {
            return refuse("rgpot_potential_hessian: Hessian byte count overflows");
        };
        if let Some(limit) = unsafe { &*pot }.hessian_byte_budget() {
            let Ok(bytes) = u64::try_from(bytes) else {
                return refuse(
                    "rgpot_potential_hessian: Hessian byte count does not fit in 64 bits",
                );
            };
            if bytes > limit {
                return refuse(&format!(
                    "rgpot_potential_hessian: Hessian of {bytes} bytes exceeds the caller budget of {limit} bytes"
                ));
            }
        }
        if dim > 0 && !input.positions.is_null() {
            let positions = unsafe { (*input.positions).dl_tensor.data } as usize;
            let Some(position_bytes) = dim.checked_mul(std::mem::size_of::<f64>()) else {
                return refuse("rgpot_potential_hessian: position byte count overflows");
            };
            let out = hessian_out as usize;
            if positions != 0
                && positions < out.saturating_add(bytes)
                && out < positions.saturating_add(position_bytes)
            {
                return refuse("rgpot_potential_hessian: hessian_out overlaps positions");
            }
        }
        match unsafe { (*pot).energy_hessian(input) } {
            Ok(matrix) => {
                if matrix.len() != n2 {
                    return refuse(&format!(
                        "rgpot_potential_hessian: expected {n2} elements, got {}",
                        matrix.len()
                    ));
                }
                unsafe { std::ptr::copy_nonoverlapping(matrix.as_ptr(), hessian_out, n2) };
                rgpot_status_t::RGPOT_SUCCESS
            }
            Err(err) => fail(err),
        }
    }))
}

/// Read the dense-Hessian counters of `pot` into `out`.
///
/// # Safety
///
/// `pot` must be NULL or a live handle, and `out` NULL or writable. NULL
/// for either returns `RGPOT_INVALID_PARAMETER`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_hessian_counts(
    pot: *const rgpot_potential_t,
    out: *mut rgpot_hessian_counts_t,
) -> rgpot_status_t {
    if pot.is_null() || out.is_null() {
        return refuse("rgpot_potential_hessian_counts: NULL argument");
    }
    unsafe { *out = (*pot).hessian_counts() };
    rgpot_status_t::RGPOT_SUCCESS
}

/// Zero the dense-Hessian counters of `pot`. The cached matrix stays.
///
/// # Safety
///
/// `pot` must be NULL or a live handle. NULL is a no-op.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_reset_hessian_counts(pot: *const rgpot_potential_t) {
    if !pot.is_null() {
        unsafe { &*pot }.reset_hessian_counts();
    }
}
