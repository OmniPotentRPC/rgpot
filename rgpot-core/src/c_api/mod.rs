// MIT License
// Copyright 2023--present rgpot developers

//! Public C API entry points.
//!
//! Each submodule exposes `extern "C"` functions that cbindgen collects into
//! `rgpot-core/include/rgpot.h`. All functions in this module follow three
//! invariants:
//!
//! 1. **Return [`rgpot_status_t`](crate::status::rgpot_status_t)** (or a
//!    pointer / void for constructors and destructors).
//! 2. **Wrap the body in [`catch_unwind`](crate::status::catch_unwind)** to
//!    prevent panics from crossing the FFI boundary.
//! 3. **Validate pointer arguments** and call
//!    [`set_last_error`](crate::status::set_last_error) before returning a
//!    non-success status.
//!
//! ## Submodules
//!
//! - [`types`] — Convenience constructors for
//!   [`rgpot_force_input_t`](crate::types::rgpot_force_input_t) and
//!   [`rgpot_force_out_t`](crate::types::rgpot_force_out_t).
//! - [`potential`] — Lifecycle functions for
//!   [`rgpot_potential_t`](crate::potential::rgpot_potential_t): create,
//!   calculate, free.
//! - [`rpc`] — RPC client functions (feature-gated on `rpc`): connect,
//!   calculate, disconnect.

pub mod types;
pub mod potential;

#[cfg(feature = "rpc")]
pub mod rpc;
