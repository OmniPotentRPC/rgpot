// MIT License
// Copyright 2023--present rgpot developers

//! C-compatible core data types for force/energy calculations.
//!
//! These two `#[repr(C)]` structs are the fundamental data exchange types at the
//! FFI boundary. They mirror the C++ `ForceInput` and `ForceOut` structures
//! defined in `CppCore/rgpot/ForceStructs.hpp` and are layout-compatible with
//! them (identical field order and types, different field names).
//!
//! ## Memory Model
//!
//! Both structs use *raw pointers* to borrowed memory. The Rust side never
//! allocates or frees the underlying buffers — that responsibility stays with
//! the C/C++ caller. The helper methods on each struct produce safe slices for
//! internal Rust use, but require the caller to uphold validity invariants.
//!
//! ## Correspondence with C++ Types
//!
//! | Rust field | C++ field | Type |
//! |-----------|----------|------|
//! | `rgpot_force_input_t::n_atoms` | `ForceInput::nAtoms` | `size_t` |
//! | `rgpot_force_input_t::pos` | `ForceInput::pos` | `const double*` |
//! | `rgpot_force_input_t::atmnrs` | `ForceInput::atmnrs` | `const int*` |
//! | `rgpot_force_input_t::box_` | `ForceInput::box` | `const double*` |
//! | `rgpot_force_out_t::forces` | `ForceOut::F` | `double*` |
//! | `rgpot_force_out_t::energy` | `ForceOut::energy` | `double` |
//! | `rgpot_force_out_t::variance` | `ForceOut::variance` | `double` |

use std::os::raw::c_int;

/// Input configuration for a potential energy evaluation.
///
/// All pointer fields are *borrowed* — the caller retains ownership and must
/// keep the underlying arrays alive for the lifetime of this struct.
///
/// # Fields
///
/// - `pos`: flat array of atomic coordinates, `[n_atoms * 3]` doubles in
///   row-major order (x0, y0, z0, x1, y1, z1, ...).
/// - `atmnrs`: atomic numbers, `[n_atoms]` ints.
/// - `box_`: simulation cell vectors as a flat 3x3 row-major matrix, `[9]`
///   doubles.
///
/// # Example (from C)
///
/// ```c
/// double positions[] = {0.0, 0.0, 0.0,  1.0, 0.0, 0.0};
/// int    types[]     = {1, 1};
/// double cell[]      = {10.0,0,0, 0,10.0,0, 0,0,10.0};
///
/// rgpot_force_input_t input = rgpot_force_input_create(2, positions, types, cell);
/// ```
#[repr(C)]
pub struct rgpot_force_input_t {
    /// Total number of atoms in the configuration.
    pub n_atoms: usize,
    /// Pointer to flat position array `[n_atoms * 3]`.
    pub pos: *const f64,
    /// Pointer to atomic number array `[n_atoms]`.
    pub atmnrs: *const c_int,
    /// Pointer to the 3x3 cell matrix `[9]`, row-major.
    pub box_: *const f64,
}

/// Results from a potential energy evaluation.
///
/// The `forces` pointer must be pre-allocated by the caller with at least
/// `n_atoms * 3` doubles. The `energy` and `variance` fields are written by the
/// potential callback.
///
/// # Fields
///
/// - `forces`: output force array, same layout as `pos` in
///   [`rgpot_force_input_t`].
/// - `energy`: the calculated potential energy (eV or consistent unit).
/// - `variance`: uncertainty estimate; zero when not applicable.
#[repr(C)]
pub struct rgpot_force_out_t {
    /// Pointer to force output buffer `[n_atoms * 3]`.
    pub forces: *mut f64,
    /// Calculated potential energy.
    pub energy: f64,
    /// Variance / uncertainty of the calculation (0.0 when unused).
    pub variance: f64,
}

impl rgpot_force_input_t {
    /// Returns the positions as a slice, or `None` if the pointer is null.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `self.pos` is valid and points to at
    /// least `self.n_atoms * 3` contiguous `f64` values.
    pub unsafe fn positions(&self) -> Option<&[f64]> {
        if self.pos.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(self.pos, self.n_atoms * 3) })
        }
    }

    /// Returns the atomic numbers as a slice, or `None` if the pointer is null.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `self.atmnrs` is valid and points to at
    /// least `self.n_atoms` contiguous `c_int` values.
    pub unsafe fn atomic_numbers(&self) -> Option<&[c_int]> {
        if self.atmnrs.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(self.atmnrs, self.n_atoms) })
        }
    }

    /// Returns the box matrix as a 9-element slice, or `None` if the pointer
    /// is null.
    ///
    /// The 9 elements represent a 3x3 row-major matrix:
    /// `[a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z]`
    /// where a, b, c are the cell vectors.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `self.box_` is valid and points to at
    /// least 9 contiguous `f64` values.
    pub unsafe fn box_matrix(&self) -> Option<&[f64]> {
        if self.box_.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(self.box_, 9) })
        }
    }
}

impl rgpot_force_out_t {
    /// Returns the forces buffer as a mutable slice, or `None` if the pointer
    /// is null.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `self.forces` is valid and points to at
    /// least `n_atoms * 3` contiguous `f64` values.
    pub unsafe fn forces_mut(&mut self, n_atoms: usize) -> Option<&mut [f64]> {
        if self.forces.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts_mut(self.forces, n_atoms * 3) })
        }
    }
}
