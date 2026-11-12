// MIT License
// Copyright 2023--present rgpot developers

//! C-compatible core data types for force/energy calculations.
//!
//! These two `#[repr(C)]` structs are the fundamental data exchange types at the
//! FFI boundary.  Each field that previously held a raw pointer to a flat array
//! now holds a `*mut DLManagedTensorVersioned`, making the types device-agnostic
//! (CPU, CUDA, ROCm, etc.) via the DLPack tensor exchange protocol.
//!
//! ## Memory Model
//!
//! - **Input tensors** are *borrowed* — the caller retains ownership.
//! - **Output forces** are *callee-allocated* — the callback sets
//!   `output.forces` to a DLPack tensor it creates.  After the call, the
//!   caller takes ownership and must free it via `rgpot_tensor_free`.
//! - **Energy and variance** are plain `f64` scalars, always on the host.
//!
//! ## DLPack Tensor Shapes
//!
//! | Field | dtype | ndim | shape |
//! |-------|-------|------|-------|
//! | `positions` | f64 | 2 | `[n_atoms, 3]` |
//! | `atomic_numbers` | i32 | 1 | `[n_atoms]` |
//! | `box_matrix` | f64 | 2 | `[3, 3]` |
//! | `forces` | f64 | 2 | `[n_atoms, 3]` |

use dlpk::sys::DLManagedTensorVersioned;

/// Input configuration for a potential energy evaluation.
///
/// All tensor fields are *borrowed* — the caller retains ownership and must
/// keep them alive for the lifetime of this struct.
///
/// # Example (from C)
///
/// ```c
/// double positions[] = {0.0, 0.0, 0.0,  1.0, 0.0, 0.0};
/// int    types[]     = {1, 1};
/// double cell[]      = {10.0,0,0, 0,10.0,0, 0,0,10.0};
///
/// rgpot_force_input_t input = rgpot_force_input_create(2, positions, types, cell);
/// // ... use input ...
/// rgpot_force_input_free(&input);
/// ```
#[repr(C)]
pub struct rgpot_force_input_t {
    /// Positions tensor: `[n_atoms, 3]`, dtype f64, any device.
    pub positions: *mut DLManagedTensorVersioned,
    /// Atomic numbers tensor: `[n_atoms]`, dtype i32, any device.
    pub atomic_numbers: *mut DLManagedTensorVersioned,
    /// Box matrix tensor: `[3, 3]`, dtype f64, any device.
    pub box_matrix: *mut DLManagedTensorVersioned,
}

/// Results from a potential energy evaluation.
///
/// The `forces` field starts as `NULL` and is set by the potential callback to
/// a callee-allocated DLPack tensor.  After the call, the caller owns the
/// tensor and must free it via `rgpot_tensor_free`.
///
/// # Fields
///
/// - `forces`: output force tensor, same shape/dtype as `positions`.
/// - `energy`: the calculated potential energy.
/// - `variance`: uncertainty estimate; zero when not applicable.
/// - `hessian`: optional Cartesian Hessian `[n_dof, n_dof]`. Null when the
///   potential does not provide one.
/// - `capabilities`: bit flags. [`RGPOT_CAP_HESSIAN`] is set only together
///   with a non-null `hessian`.
///
/// A potential that returns energy and forces leaves `hessian` null and
/// `capabilities` at zero.
#[repr(C)]
pub struct rgpot_force_out_t {
    /// Forces tensor `[n_atoms, 3]`, f64 — set by the callback.
    pub forces: *mut DLManagedTensorVersioned,
    /// Calculated potential energy.
    pub energy: f64,
    /// Variance / uncertainty of the calculation (0.0 when unused).
    pub variance: f64,
    /// Hessian tensor `[n_dof, n_dof]`, f64, or null.
    pub hessian: *mut DLManagedTensorVersioned,
    /// Capability bits. See [`RGPOT_CAP_HESSIAN`].
    pub capabilities: u32,
}

/// The callback filled `hessian`.
pub const RGPOT_CAP_HESSIAN: u32 = 1;

impl rgpot_force_out_t {
    /// True when the callback both set the flag and provided a tensor.
    pub fn has_hessian(&self) -> bool {
        self.capabilities & RGPOT_CAP_HESSIAN != 0 && !self.hessian.is_null()
    }
}

impl rgpot_force_input_t {
    /// Extract `n_atoms` from the positions tensor's `shape[0]`.
    ///
    /// Returns `None` if `positions` is null or has no shape.
    ///
    /// # Safety
    ///
    /// The `positions` tensor must be a valid DLPack tensor with `ndim >= 1`.
    pub unsafe fn n_atoms(&self) -> Option<usize> {
        if self.positions.is_null() {
            return None;
        }
        let t = unsafe { &(*self.positions).dl_tensor };
        if t.ndim < 1 || t.shape.is_null() {
            return None;
        }
        Some(unsafe { *t.shape } as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;

    #[test]
    fn n_atoms_from_positions_tensor() {
        let mut pos = [0.0_f64; 6]; // 2 atoms * 3
        let tensor = unsafe { rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), 2, 3) };
        let input = rgpot_force_input_t {
            positions: tensor,
            atomic_numbers: std::ptr::null_mut(),
            box_matrix: std::ptr::null_mut(),
        };
        assert_eq!(unsafe { input.n_atoms() }, Some(2));
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn n_atoms_null_positions() {
        let input = rgpot_force_input_t {
            positions: std::ptr::null_mut(),
            atomic_numbers: std::ptr::null_mut(),
            box_matrix: std::ptr::null_mut(),
        };
        assert_eq!(unsafe { input.n_atoms() }, None);
    }

    #[test]
    fn force_out_starts_with_null_forces() {
        let out = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
            hessian: std::ptr::null_mut(),
            capabilities: 0,
        };
        assert!(out.forces.is_null());
        assert!(out.hessian.is_null());
        assert_eq!(out.capabilities, 0);
        assert!(!out.has_hessian());
        assert_eq!(out.energy, 0.0);
        assert_eq!(out.variance, 0.0);
    }

    #[test]
    fn hessian_is_present_only_when_the_callback_sets_the_flag() {
        let data = [2.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let tensor = unsafe { rgpot_tensor_owned_cpu_f64_2d(data.as_ptr(), 3, 3) };
        let mut out = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
            hessian: tensor,
            capabilities: 0,
        };
        assert!(!out.has_hessian());
        out.capabilities = RGPOT_CAP_HESSIAN;
        assert!(out.has_hessian());
        let ptr = unsafe { rgpot_tensor_data(tensor) } as *const f64;
        assert!((unsafe { *ptr } - 2.0).abs() < 1.0e-15);
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn force_out_energy_and_variance_are_independent() {
        let mut out = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 1.5,
            variance: 2.5,
            hessian: std::ptr::null_mut(),
            capabilities: 0,
        };
        assert_eq!(out.energy, 1.5);
        assert_eq!(out.variance, 2.5);
        out.energy = -3.0;
        assert_eq!(out.energy, -3.0);
        assert_eq!(out.variance, 2.5);
    }

    #[test]
    fn full_input_with_tensors() {
        let mut pos = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut atmnrs = [1_i32, 1];
        let mut box_ = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

        let pos_t = unsafe { rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), 2, 3) };
        let atm_t = unsafe { rgpot_tensor_cpu_i32_1d(atmnrs.as_mut_ptr(), 2) };
        let box_t = unsafe { rgpot_tensor_cpu_f64_matrix3(box_.as_mut_ptr()) };

        let input = rgpot_force_input_t {
            positions: pos_t,
            atomic_numbers: atm_t,
            box_matrix: box_t,
        };

        assert_eq!(unsafe { input.n_atoms() }, Some(2));

        // Verify data is accessible through tensors.
        let pt = unsafe { &(*pos_t).dl_tensor };
        let data = unsafe { std::slice::from_raw_parts(pt.data as *const f64, 6) };
        assert_eq!(data, &pos);

        unsafe {
            rgpot_tensor_free(pos_t);
            rgpot_tensor_free(atm_t);
            rgpot_tensor_free(box_t);
        }
    }
}
