// MIT License
// Copyright 2023--present rgpot developers

//! C API for dense Hessians of a potential handle, and for the capabilities
//! and counts of its derivative entry points.
//!
//! ```c
//! rgpot_derivative_caps_t caps;
//! rgpot_potential_derivative_caps(pot, &caps);
//! int scheme = caps.hessian_analytic ? RGPOT_HESSIAN_ANALYTIC
//!                                    : RGPOT_HESSIAN_CENTRAL;
//! rgpot_hessian_scheme_t used;
//! rgpot_potential_hessian(pot, &input, 0.0, scheme, h, &used);
//! ```
//!
//! See [`crate::hessian`] for the schemes.

use std::os::raw::c_void;

use crate::c_api::hvp::{fail, refuse};
use crate::eindir::rgpot_potential_t;
use crate::hessian::{
    hessian_len, refuse_overlap, rgpot_derivative_caps_t, rgpot_derivative_counts_t,
    rgpot_hessian_scheme_t, HessianCallback,
};
use crate::status::{catch_unwind, rgpot_status_t};
use crate::types::rgpot_force_input_t;

/// Register `callback` as the analytic Hessian of `pot`, or clear it with
/// NULL.
///
/// The callback receives the `user_data` the handle was created with and
/// writes `(3 * n_atoms)^2` doubles, row-major. `RGPOT_HESSIAN_ANALYTIC`
/// calls it, and so does `RGPOT_HESSIAN_EXACT`.
///
/// Replacing or clearing the callback returns only after every call of the
/// previous callback that was already running has returned, and the previous
/// callback is never called after this function returns. The caller may
/// release the old callback and its state as soon as this function returns.
/// Because it waits for those calls, this function must not be called from
/// inside a Hessian callback, nor from a thread that is inside this handle's
/// Hessian entry points: it would wait for itself.
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

/// Register `callback` as the dual-number Hessian of `pot`, or clear it
/// with NULL.
///
/// It has the signature of the analytic Hessian callback.
/// `RGPOT_HESSIAN_AD` calls it, and `RGPOT_HESSIAN_EXACT` when no analytic
/// Hessian callback is registered.
///
/// # Safety
///
/// As for `rgpot_potential_set_hessian_callback`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_set_ad_hessian_callback(
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
        return refuse("rgpot_potential_set_ad_hessian_callback: pot is NULL");
    }
    unsafe { &*pot }.set_ad_hessian_callback(callback);
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

/// Write the dense Hessian of the energy at the geometry of `input` to
/// `hessian_out` and the scheme that computed it to `scheme_used`.
///
/// `hessian_out` holds `(3 * n_atoms)^2` doubles, row-major, coordinate
/// `3 i + k` for axis `k` of atom `i`. `scheme` is a
/// `rgpot_hessian_scheme_t` value, and there is no automatic one:
///
/// - `RGPOT_HESSIAN_ANALYTIC`: the registered Hessian callback;
///   `RGPOT_INVALID_PARAMETER` when none is registered.
/// - `RGPOT_HESSIAN_AD`: the registered dual-number Hessian callback;
///   `RGPOT_INVALID_PARAMETER` when none is registered.
/// - `RGPOT_HESSIAN_EXACT`: the analytic callback, else the dual-number
///   one; `RGPOT_INVALID_PARAMETER` when neither is registered.
/// - `RGPOT_HESSIAN_CENTRAL`: column `c` is
///   `(F(x - h e_c) - F(x + h e_c)) / (2 h)`, from `6 * n_atoms` force
///   calls that bypass the cached centre, and the matrix written is the
///   symmetric part of those columns. Error `O(h^2)`.
///
/// `step` is `h`, a length in the units of the positions; 0 selects
/// `RGPOT_HVP_DEFAULT_STEP`. `RGPOT_HESSIAN_EXACT` reports
/// `RGPOT_HESSIAN_ANALYTIC` or `RGPOT_HESSIAN_AD`, and any other scheme
/// reports itself. `scheme_used` is written only on success; on a failure
/// the content of `hessian_out` is unspecified. A failing callback's status
/// is returned with the message it set, or `<callback> returned <status>`
/// when it set none. A non-finite entry from any scheme returns
/// `RGPOT_INVALID_PARAMETER`.
///
/// # Safety
///
/// NULL pointers, an unknown scheme, a negative or non-finite step and a
/// `hessian_out` that overlaps the positions, atomic numbers or box
/// return `RGPOT_INVALID_PARAMETER` without a call. So does a matrix whose
/// byte count `(3 * n_atoms)^2 * sizeof(double)` overflows or exceeds the
/// budget set by `rgpot_potential_set_hessian_byte_budget`, before
/// `hessian_out` is touched: a caller whose buffer may be short installs its
/// length in bytes as the budget. Otherwise `pot` must be a live handle,
/// `input` must point to a valid `rgpot_force_input_t` whose tensors stay
/// alive for the call, `hessian_out` must point to `(3 * n_atoms)^2`
/// writable doubles and `scheme_used` must be writable.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_hessian(
    pot: *const rgpot_potential_t,
    input: *const rgpot_force_input_t,
    step: f64,
    scheme: i32,
    hessian_out: *mut f64,
    scheme_used: *mut rgpot_hessian_scheme_t,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() || input.is_null() {
            return refuse("rgpot_potential_hessian: pot or input is NULL");
        }
        if hessian_out.is_null() || scheme_used.is_null() {
            return refuse("rgpot_potential_hessian: hessian_out or scheme_used is NULL");
        }
        let Some(scheme) = rgpot_hessian_scheme_t::from_raw(scheme) else {
            return refuse(&format!("rgpot_potential_hessian: unknown scheme {scheme}"));
        };
        let input = unsafe { &*input };
        let Some((dim, entries)) = unsafe { input.n_atoms() }
            .and_then(|n| n.checked_mul(3))
            .and_then(|dim| Some((dim, dim.checked_mul(dim)?)))
            .filter(|&(_, entries)| entries <= isize::MAX as usize / std::mem::size_of::<f64>())
        else {
            return refuse("rgpot_potential_hessian: positions tensor is NULL or too large");
        };
        // The budget is the only length the caller states for hessian_out,
        // so a short buffer must be refused before the slice is formed.
        if let Err(err) = hessian_len(dim, unsafe { (*pot).hessian_byte_budget() }) {
            return fail(err);
        }
        if let Err(err) = unsafe { refuse_overlap(input, hessian_out, entries) } {
            return fail(err);
        }
        let hessian_out = unsafe { std::slice::from_raw_parts_mut(hessian_out, entries) };
        match unsafe { (*pot).hessian(input, step, scheme, hessian_out) } {
            Ok(ran) => {
                unsafe { scheme_used.write(ran) };
                rgpot_status_t::RGPOT_SUCCESS
            }
            Err(err) => fail(err),
        }
    }))
}

/// Read which derivative schemes `pot` has into `out`.
///
/// The analytic and dual-number flags follow the callbacks registered at
/// the time of the call; the finite-difference flags are set for every
/// handle, which always has its force callback.
///
/// # Safety
///
/// `pot` must be NULL or a live handle, and `out` NULL or writable. NULL
/// for either returns `RGPOT_INVALID_PARAMETER`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_derivative_caps(
    pot: *const rgpot_potential_t,
    out: *mut rgpot_derivative_caps_t,
) -> rgpot_status_t {
    if pot.is_null() || out.is_null() {
        return refuse("rgpot_potential_derivative_caps: NULL argument");
    }
    unsafe { out.write((*pot).derivative_caps()) };
    rgpot_status_t::RGPOT_SUCCESS
}

/// Read the counts of requests each derivative scheme of `pot` answered
/// into `out`. `rgpot_potential_reset_hvp_counts` zeros them.
///
/// # Safety
///
/// `pot` must be NULL or a live handle, and `out` NULL or writable. NULL
/// for either returns `RGPOT_INVALID_PARAMETER`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_derivative_counts(
    pot: *const rgpot_potential_t,
    out: *mut rgpot_derivative_counts_t,
) -> rgpot_status_t {
    if pot.is_null() || out.is_null() {
        return refuse("rgpot_potential_derivative_counts: NULL argument");
    }
    unsafe { out.write((*pot).derivative_counts()) };
    rgpot_status_t::RGPOT_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_api::potential::{rgpot_potential_free, rgpot_potential_new};
    use crate::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
    use crate::tensor::create_owned_f64_tensor;
    use crate::types::rgpot_force_out_t;
    use std::sync::atomic::{AtomicUsize, Ordering};

    unsafe extern "C" fn counted_forces(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        unsafe { &*(user_data as *const AtomicUsize) }.fetch_add(1, Ordering::SeqCst);
        let n = unsafe { (*input).n_atoms() }.unwrap_or(0);
        let out = unsafe { &mut *output };
        out.energy = 0.0;
        out.variance = 0.0;
        out.forces = create_owned_f64_tensor(vec![0.0; 3 * n], vec![n as i64, 3]);
        rgpot_status_t::RGPOT_SUCCESS
    }

    unsafe extern "C" fn counted_hessian(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        hessian_out: *mut f64,
    ) -> rgpot_status_t {
        unsafe { &*(user_data as *const AtomicUsize) }.fetch_add(1, Ordering::SeqCst);
        let dim = 3 * unsafe { (*input).n_atoms() }.unwrap_or(0);
        unsafe { std::slice::from_raw_parts_mut(hessian_out, dim * dim) }.fill(0.0);
        rgpot_status_t::RGPOT_SUCCESS
    }

    #[test]
    fn a_buffer_shorter_than_the_matrix_is_refused_before_any_write() {
        let calls = AtomicUsize::new(0);
        let user_data = &calls as *const AtomicUsize as *mut c_void;
        let pot = unsafe { rgpot_potential_new(counted_forces, user_data, None) };
        unsafe { rgpot_potential_set_hessian_callback(pot, Some(counted_hessian)) };
        let mut pos = vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0];
        let mut atmnrs = vec![1, 1];
        let mut box_ = vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let mut input = unsafe {
            rgpot_force_input_create(2, pos.as_mut_ptr(), atmnrs.as_mut_ptr(), box_.as_mut_ptr())
        };

        // Two atoms need 36 doubles; the caller owns only three.
        let mut short = [7.0_f64; 3];
        let budget = std::mem::size_of_val(&short) as u64;
        assert_eq!(
            unsafe { rgpot_potential_set_hessian_byte_budget(pot, &budget) },
            rgpot_status_t::RGPOT_SUCCESS
        );
        for scheme in [
            rgpot_hessian_scheme_t::RGPOT_HESSIAN_ANALYTIC,
            rgpot_hessian_scheme_t::RGPOT_HESSIAN_CENTRAL,
        ] {
            let mut used = rgpot_hessian_scheme_t::RGPOT_HESSIAN_EXACT;
            let status = unsafe {
                rgpot_potential_hessian(
                    pot,
                    &input,
                    0.0,
                    scheme as i32,
                    short.as_mut_ptr(),
                    &mut used,
                )
            };
            assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
            assert!(crate::status::last_error_message().contains("caller budget"));
            assert_eq!(used, rgpot_hessian_scheme_t::RGPOT_HESSIAN_EXACT);
        }
        assert_eq!(short, [7.0; 3]);
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let mut full = vec![7.0_f64; 36];
        let budget = std::mem::size_of_val(full.as_slice()) as u64;
        unsafe { rgpot_potential_set_hessian_byte_budget(pot, &budget) };
        let mut used = rgpot_hessian_scheme_t::RGPOT_HESSIAN_EXACT;
        let status = unsafe {
            rgpot_potential_hessian(
                pot,
                &input,
                0.0,
                rgpot_hessian_scheme_t::RGPOT_HESSIAN_ANALYTIC as i32,
                full.as_mut_ptr(),
                &mut used,
            )
        };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(used, rgpot_hessian_scheme_t::RGPOT_HESSIAN_ANALYTIC);
        assert!(full.iter().all(|&h| h == 0.0));
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        unsafe {
            rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot);
        }
    }

    #[test]
    fn a_byte_count_that_overflows_is_refused_without_a_call() {
        let calls = AtomicUsize::new(0);
        let user_data = &calls as *const AtomicUsize as *mut c_void;
        let pot = unsafe { rgpot_potential_new(counted_forces, user_data, None) };
        unsafe { rgpot_potential_set_hessian_callback(pot, Some(counted_hessian)) };
        let mut pos = [0.0_f64; 3];
        let mut atmnrs = [1];
        let mut box_ = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        // Only the shape is read: (3 n)^2 overflows usize, and its byte
        // count passes isize::MAX for a smaller n, before any tensor data is
        // used.
        for n_atoms in [1_usize << 31, 1_usize << 30] {
            let mut input = unsafe {
                rgpot_force_input_create(
                    n_atoms,
                    pos.as_mut_ptr(),
                    atmnrs.as_mut_ptr(),
                    box_.as_mut_ptr(),
                )
            };
            let mut out = [7.0_f64; 1];
            let mut used = rgpot_hessian_scheme_t::RGPOT_HESSIAN_EXACT;
            let status = unsafe {
                rgpot_potential_hessian(
                    pot,
                    &input,
                    0.0,
                    rgpot_hessian_scheme_t::RGPOT_HESSIAN_ANALYTIC as i32,
                    out.as_mut_ptr(),
                    &mut used,
                )
            };
            assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
            assert_eq!(
                crate::status::last_error_message(),
                "rgpot_potential_hessian: positions tensor is NULL or too large"
            );
            assert_eq!(out, [7.0]);
            unsafe { rgpot_force_input_free(&mut input) };
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        unsafe { rgpot_potential_free(pot) };
    }
}
