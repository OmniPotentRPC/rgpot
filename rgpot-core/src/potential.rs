// MIT License
// Copyright 2023--present rgpot developers

//! Callback-based potential dispatch.
//!
//! This module defines [`PotentialImpl`], an opaque handle that wraps a C
//! function pointer callback together with a `void* user_data` and an optional
//! destructor. This is the core abstraction that lets existing C++ potentials
//! (LJ, CuH2, or any future implementation) plug into the Rust infrastructure
//! without the Rust side knowing the concrete type.
//!
//! ## How it Works
//!
//! 1. The C++ side creates a potential object (e.g., `LJPot`).
//! 2. A trampoline function with the [`PotentialCallback`] signature is
//!    registered, casting `user_data` back to the concrete type and calling
//!    `forceImpl`.
//! 3. The Rust core dispatches through the function pointer, receiving results
//!    via the [`rgpot_force_out_t`] output struct.
//!
//! ## Lifetime Contract
//!
//! - The `user_data` pointer is borrowed by `PotentialImpl`. The caller must
//!   keep the underlying object alive for the lifetime of the handle.
//! - If a `free_fn` is provided, it is called on drop when `user_data` is
//!   non-null, transferring ownership to `PotentialImpl`.
//! - The handle is exposed to C as `rgpot_potential_t` — an opaque pointer
//!   managed via `rgpot_potential_new` / `rgpot_potential_free`.

use std::os::raw::c_void;

use crate::status::rgpot_status_t;
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Function pointer type for a potential energy calculation.
///
/// The callback receives:
/// - `user_data`: opaque pointer to the C++ object (e.g. `LJPot*`)
/// - `input`: the atomic configuration
/// - `output`: the buffer for results
///
/// Returns `RGPOT_SUCCESS` on success, or an error status code.
pub type PotentialCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t;

/// Destructor for the user_data pointer.
pub type FreeFn = unsafe extern "C" fn(*mut c_void);

/// Opaque potential handle wrapping a callback + user data.
pub struct PotentialImpl {
    pub(crate) callback: PotentialCallback,
    pub(crate) user_data: *mut c_void,
    pub(crate) free_fn: Option<FreeFn>,
}

// PotentialImpl stores a raw pointer but we guarantee exclusive access
// through the opaque handle pattern.
unsafe impl Send for PotentialImpl {}

impl PotentialImpl {
    /// Create a new potential from a callback, user data, and optional destructor.
    pub fn new(
        callback: PotentialCallback,
        user_data: *mut c_void,
        free_fn: Option<FreeFn>,
    ) -> Self {
        Self {
            callback,
            user_data,
            free_fn,
        }
    }

    /// Invoke the underlying callback.
    ///
    /// # Safety
    /// The caller must ensure `input` and `output` point to valid, properly
    /// sized structures.
    pub unsafe fn calculate(
        &self,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        (self.callback)(self.user_data, input, output)
    }
}

impl Drop for PotentialImpl {
    fn drop(&mut self) {
        if let Some(free) = self.free_fn {
            if !self.user_data.is_null() {
                unsafe { free(self.user_data) };
            }
        }
    }
}

/// Opaque handle exposed to C as `rgpot_potential_t`.
///
/// This is a type alias used by cbindgen to generate a forward declaration.
pub type rgpot_potential_t = PotentialImpl;

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn mock_callback(
        _user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        // Simple mock: set energy to n_atoms as f64
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };
        out.energy = inp.n_atoms as f64;
        out.variance = 0.0;
        rgpot_status_t::RGPOT_SUCCESS
    }

    #[test]
    fn test_potential_callback() {
        let pot = PotentialImpl::new(mock_callback, std::ptr::null_mut(), None);

        let positions = [0.0_f64; 9]; // 3 atoms × 3
        let atmnrs = [1_i32; 3];
        let box_ = [1.0_f64; 9];
        let mut forces = [0.0_f64; 9];

        let input = rgpot_force_input_t {
            n_atoms: 3,
            pos: positions.as_ptr(),
            atmnrs: atmnrs.as_ptr(),
            box_: box_.as_ptr(),
        };
        let mut output = rgpot_force_out_t {
            forces: forces.as_mut_ptr(),
            energy: 0.0,
            variance: 0.0,
        };

        let status = unsafe { pot.calculate(&input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(output.energy, 3.0);
    }
}
