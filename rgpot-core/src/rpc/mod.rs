// MIT License
// Copyright 2023--present rgpot developers

//! Cap'n Proto RPC module for distributed potential calculations.
//!
//! This module is only compiled when the `rpc` Cargo feature is enabled.
//! It reuses the existing `CppCore/rgpot/rpc/Potentials.capnp` schema that
//! the C++ server also uses, ensuring full wire-format compatibility between
//! Rust and C++ endpoints.
//!
//! **Schema**
//!
//! The [`schema`] submodule contains the Rust code generated from
//! ``Potentials.capnp`` by ``capnpc`` during ``build.rs``. The schema defines:
//!
//! - ``ForceInput`` -- positions, atomic numbers, simulation cell.
//! - ``PotentialResult`` -- energy and forces.
//! - ``Potential`` interface -- a single ``calculate`` RPC method.
//!
//! **Client**
//!
//! [`client::RpcClient`] connects to a remote rgpot server and provides a
//! synchronous ``calculate()`` method. Internally it owns a tokio runtime so
//! that the blocking C API can drive async I/O. Exposed to C via
//! ``rgpot_rpc_client_new`` / ``rgpot_rpc_calculate`` / ``rgpot_rpc_client_free``.
//!
//! **Server**
//!
//! [`server::rgpot_rpc_server_start`] accepts a ``rgpot_potential_t`` handle
//! (callback-backed) and listens for incoming Cap'n Proto RPC connections.
//! Each ``calculate`` call is dispatched to the callback. The server blocks the
//! calling thread.

/// Re-export of the generated Cap'n Proto schema from the crate root.
///
/// The generated code uses `crate::Potentials_capnp` internally, so the
/// module must live at the crate root.  This re-export provides a
/// convenient `rpc::schema` alias.
pub use crate::Potentials_capnp as schema;

pub mod client;
pub mod server;
