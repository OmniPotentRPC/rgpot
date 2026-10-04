// MIT License
// Copyright 2023--present rgpot developers

//! Capability handshake shared by every force dispatch path.
//!
//! A peer's `Capabilities` message is compared with what this build requires.
//! The refusal names the field, the required value and the received value.
//! Numeric fields left at the Cap'n Proto default of zero are absent. An empty
//! `buildVersion` or `buildRevision` is not a mismatch: those fields name the
//! producing build and are optional.

use capnp::message::ReaderOptions;
use capnp::serialize;

use crate::Potentials_capnp::capabilities::{Builder, Operation, Reader};

/// Protocol family this build speaks.
pub const PROTOCOL_FAMILY: &str = "rgpot.potentials";
/// Wire-incompatible protocol revision.
pub const PROTOCOL_MAJOR: u16 = 1;
/// Additive protocol revision this build produces.
pub const PROTOCOL_MINOR: u16 = 0;
/// Cap'n Proto file id of `Potentials.capnp`.
pub const SCHEMA_ID: &str = "0xbd1f89fa17369103";

/// What this build requires of a peer before it dispatches a force evaluation.
#[derive(Debug, Clone)]
pub struct Expectation {
    pub family: &'static str,
    pub protocol_major: u16,
    /// Lowest protocol minor the host accepts. A higher minor is additive.
    pub protocol_minor_min: u16,
    pub schema_id: &'static str,
    pub bridge_abi_major: u16,
    /// Highest eindir bridge minor this build can consume.
    pub bridge_abi_minor_max: u16,
    pub bridge_layout: u32,
    pub dlpack_major: u16,
    /// Highest DLPack minor this build can consume.
    pub dlpack_minor_max: u16,
    /// Bridge feature bits this build understands. A peer must be a subset.
    pub bridge_features: u64,
    pub required_operations: Vec<Operation>,
}

impl Default for Expectation {
    fn default() -> Self {
        let stamp = eindir_core::ffi::eindir_core_abi_stamp();
        Self {
            family: PROTOCOL_FAMILY,
            protocol_major: PROTOCOL_MAJOR,
            protocol_minor_min: 0,
            schema_id: SCHEMA_ID,
            bridge_abi_major: stamp.abi_major as u16,
            bridge_abi_minor_max: stamp.abi_minor as u16,
            bridge_layout: stamp.objective_layout,
            dlpack_major: stamp.dlpack_major as u16,
            dlpack_minor_max: stamp.dlpack_minor as u16,
            bridge_features: stamp.features,
            required_operations: vec![Operation::Energy, Operation::Forces],
        }
    }
}

/// Build identity carried by a peer. Empty strings are unknown, not refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerIdentity {
    pub build_version: String,
    pub build_revision: String,
}

fn mismatch(
    field: &str,
    required: impl std::fmt::Display,
    received: impl std::fmt::Display,
) -> String {
    format!("{field}: required {required}, received {received}")
}

fn text(result: capnp::Result<capnp::text::Reader<'_>>, field: &str) -> Result<String, String> {
    result
        .map_err(|e| mismatch(field, "UTF-8 text", e))?
        .to_string()
        .map_err(|e| mismatch(field, "UTF-8 text", e))
}

fn operation_name(op: Operation) -> &'static str {
    match op {
        Operation::Energy => "energy",
        Operation::Forces => "forces",
        Operation::Gradient => "gradient",
        Operation::Hessian => "hessian",
        Operation::Dipole => "dipole",
        Operation::Polarizability => "polarizability",
        Operation::Quadrupole => "quadrupole",
        Operation::Stress => "stress",
        Operation::Optimize => "optimize",
        Operation::Frequencies => "frequencies",
    }
}

fn operation_list(ops: &[Operation]) -> String {
    if ops.is_empty() {
        return "none".to_owned();
    }
    ops.iter()
        .map(|op| operation_name(*op))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Check `caps` against `want`. The error names the first refused field.
pub fn check_capabilities(caps: Reader<'_>, want: &Expectation) -> Result<(), String> {
    let family = text(caps.get_protocol_family(), "protocolFamily")?;
    if family != want.family {
        return Err(mismatch("protocolFamily", want.family, family));
    }
    if caps.get_protocol_major() != want.protocol_major {
        return Err(mismatch(
            "protocolMajor",
            want.protocol_major,
            caps.get_protocol_major(),
        ));
    }
    if caps.get_protocol_minor() < want.protocol_minor_min {
        return Err(mismatch(
            "protocolMinor",
            format!(">= {}", want.protocol_minor_min),
            caps.get_protocol_minor(),
        ));
    }
    let schema_id = text(caps.get_schema_id(), "schemaId")?;
    if schema_id != want.schema_id {
        return Err(mismatch("schemaId", want.schema_id, schema_id));
    }
    if caps.get_bridge_abi_major() != want.bridge_abi_major {
        return Err(mismatch(
            "bridgeAbiMajor",
            want.bridge_abi_major,
            caps.get_bridge_abi_major(),
        ));
    }
    if caps.get_bridge_abi_minor() > want.bridge_abi_minor_max {
        return Err(mismatch(
            "bridgeAbiMinor",
            format!("<= {}", want.bridge_abi_minor_max),
            caps.get_bridge_abi_minor(),
        ));
    }
    if caps.get_bridge_layout() != want.bridge_layout {
        return Err(mismatch(
            "bridgeLayout",
            want.bridge_layout,
            caps.get_bridge_layout(),
        ));
    }
    if caps.get_dlpack_major() != want.dlpack_major {
        return Err(mismatch(
            "dlpackMajor",
            want.dlpack_major,
            caps.get_dlpack_major(),
        ));
    }
    if caps.get_dlpack_minor() > want.dlpack_minor_max {
        return Err(mismatch(
            "dlpackMinor",
            format!("<= {}", want.dlpack_minor_max),
            caps.get_dlpack_minor(),
        ));
    }
    if caps.get_bridge_features() & !want.bridge_features != 0 {
        return Err(mismatch(
            "bridgeFeatures",
            format!("subset of {:#x}", want.bridge_features),
            format!("{:#x}", caps.get_bridge_features()),
        ));
    }
    let ops = caps
        .get_operations()
        .map_err(|e| mismatch("operations", "a list", e))?;
    let served: Vec<Operation> = ops.iter().filter_map(Result::ok).collect();
    for required in &want.required_operations {
        if !served.contains(required) {
            return Err(mismatch(
                "operations",
                operation_name(*required),
                operation_list(&served),
            ));
        }
    }
    Ok(())
}

/// Decode a flat `Capabilities` message and apply [`check_capabilities`].
pub fn inspect(bytes: &[u8], want: &Expectation) -> Result<PeerIdentity, String> {
    if bytes.is_empty() || bytes.len() % 8 != 0 {
        return Err(mismatch(
            "capabilities",
            "a word-aligned Capabilities message",
            format!("{} bytes", bytes.len()),
        ));
    }
    let mut slice = bytes;
    let message = serialize::read_message_from_flat_slice(&mut slice, ReaderOptions::new())
        .map_err(|e| mismatch("capabilities", "a Capabilities message", e))?;
    let caps = message
        .get_root::<Reader>()
        .map_err(|e| mismatch("capabilities", "a Capabilities message", e))?;
    check_capabilities(caps, want)?;
    Ok(PeerIdentity {
        build_version: text(caps.get_build_version(), "buildVersion")?,
        build_revision: text(caps.get_build_revision(), "buildRevision")?,
    })
}

/// Fill the compatibility and build-identity fields of `caps`.
///
/// Callers still set backend name, availability, operations and config kinds.
/// `buildRevision` is empty when the build has no source revision, which is
/// the tarball fallback.
pub fn fill_compatibility(mut caps: Builder<'_>) {
    let want = Expectation::default();
    caps.set_protocol_family(want.family);
    caps.set_protocol_major(want.protocol_major);
    caps.set_protocol_minor(PROTOCOL_MINOR);
    caps.set_schema_id(want.schema_id);
    caps.set_bridge_abi_major(want.bridge_abi_major);
    caps.set_bridge_abi_minor(want.bridge_abi_minor_max);
    caps.set_bridge_layout(want.bridge_layout);
    caps.set_dlpack_major(want.dlpack_major);
    caps.set_dlpack_minor(want.dlpack_minor_max);
    caps.set_bridge_features(want.bridge_features);
    caps.set_build_version(env!("CARGO_PKG_VERSION"));
    caps.set_build_revision(env!("RGPOT_SOURCE_REVISION"));
}

#[cfg(test)]
pub(crate) fn flat_message(edit: impl FnOnce(&mut Builder<'_>)) -> Vec<u8> {
    let mut msg = capnp::message::Builder::new_default();
    {
        let mut caps = msg.init_root::<Builder<'_>>();
        fill_compatibility(caps.reborrow());
        let mut ops = caps.reborrow().init_operations(2);
        ops.set(0, Operation::Energy);
        ops.set(1, Operation::Forces);
        edit(&mut caps);
    }
    serialize::write_message_to_words(&msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(edit: impl FnOnce(&mut Builder<'_>), want: &Expectation) -> Result<(), String> {
        let bytes = flat_message(edit);
        inspect(&bytes, want).map(|_| ())
    }

    #[test]
    fn matching_metadata_is_accepted() {
        assert_eq!(verdict(|_| {}, &Expectation::default()), Ok(()));
    }

    #[test]
    fn absent_build_identity_is_accepted() {
        let identity = inspect(
            &flat_message(|caps| {
                caps.set_build_version("");
                caps.set_build_revision("");
            }),
            &Expectation::default(),
        )
        .unwrap();
        assert_eq!(identity.build_version, "");
        assert_eq!(identity.build_revision, "");
    }

    #[test]
    fn producer_reports_the_build_identity() {
        let identity = inspect(&flat_message(|_| {}), &Expectation::default()).unwrap();
        assert_eq!(identity.build_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(identity.build_revision, env!("RGPOT_SOURCE_REVISION"));
    }

    #[test]
    fn expectation_default_tracks_the_eindir_stamp() {
        let want = Expectation::default();
        assert_eq!(
            (
                want.bridge_abi_major,
                want.bridge_abi_minor_max,
                want.bridge_layout,
                want.dlpack_major,
                want.dlpack_minor_max,
                want.bridge_features
            ),
            (1, 0, 1, 1, 0, 0x3)
        );
    }

    #[test]
    fn rejection_matrix_names_field_required_and_received() {
        let want = Expectation::default();
        let mut minor = want.clone();
        minor.protocol_minor_min = 1;
        let mut bridge_minor = want.clone();
        bridge_minor.bridge_abi_minor_max = 2;
        let mut dlpack_minor = want.clone();
        dlpack_minor.dlpack_minor_max = 2;

        let refuse = |edit: fn(&mut Builder<'_>), want: &Expectation, exact: &str| {
            assert_eq!(verdict(edit, want).unwrap_err(), exact);
        };
        let allow = |edit: fn(&mut Builder<'_>), want: &Expectation| {
            assert_eq!(verdict(edit, want), Ok(()));
        };

        // protocolFamily: compatible is the filled message.
        refuse(
            |c| c.set_protocol_family("old.family"),
            &want,
            "protocolFamily: required rgpot.potentials, received old.family",
        );
        refuse(
            |c| c.set_protocol_family("new.family"),
            &want,
            "protocolFamily: required rgpot.potentials, received new.family",
        );
        refuse(
            |c| c.set_protocol_family(""),
            &want,
            "protocolFamily: required rgpot.potentials, received ",
        );

        allow(|c| c.set_protocol_major(1), &want);
        refuse(
            |c| c.set_protocol_major(0),
            &want,
            "protocolMajor: required 1, received 0",
        );
        refuse(
            |c| c.set_protocol_major(2),
            &want,
            "protocolMajor: required 1, received 2",
        );

        allow(|c| c.set_protocol_minor(1), &minor);
        refuse(
            |c| c.set_protocol_minor(0),
            &minor,
            "protocolMinor: required >= 1, received 0",
        );
        allow(|c| c.set_protocol_minor(3), &minor);
        // Unset minor is the zero default, the same word as an explicit 0.
        refuse(|_| {}, &minor, "protocolMinor: required >= 1, received 0");

        allow(|c| c.set_schema_id(SCHEMA_ID), &want);
        refuse(
            |c| c.set_schema_id("0x0000000000000001"),
            &want,
            "schemaId: required 0xbd1f89fa17369103, received 0x0000000000000001",
        );
        refuse(
            |c| c.set_schema_id("0xffffffffffffffff"),
            &want,
            "schemaId: required 0xbd1f89fa17369103, received 0xffffffffffffffff",
        );
        refuse(
            |c| c.set_schema_id(""),
            &want,
            "schemaId: required 0xbd1f89fa17369103, received ",
        );

        allow(|c| c.set_bridge_abi_major(1), &want);
        refuse(
            |c| c.set_bridge_abi_major(0),
            &want,
            "bridgeAbiMajor: required 1, received 0",
        );
        refuse(
            |c| c.set_bridge_abi_major(2),
            &want,
            "bridgeAbiMajor: required 1, received 2",
        );

        allow(|c| c.set_bridge_abi_minor(2), &bridge_minor);
        allow(|c| c.set_bridge_abi_minor(1), &bridge_minor);
        refuse(
            |c| c.set_bridge_abi_minor(3),
            &bridge_minor,
            "bridgeAbiMinor: required <= 2, received 3",
        );
        allow(|c| c.set_bridge_abi_minor(0), &bridge_minor);

        allow(|c| c.set_bridge_layout(1), &want);
        refuse(
            |c| c.set_bridge_layout(0),
            &want,
            "bridgeLayout: required 1, received 0",
        );
        refuse(
            |c| c.set_bridge_layout(2),
            &want,
            "bridgeLayout: required 1, received 2",
        );

        allow(|c| c.set_bridge_features(0x3), &want);
        allow(|c| c.set_bridge_features(0x1), &want);
        refuse(
            |c| c.set_bridge_features(0x7),
            &want,
            "bridgeFeatures: required subset of 0x3, received 0x7",
        );
        allow(|c| c.set_bridge_features(0), &want);

        allow(|c| c.set_dlpack_major(1), &want);
        refuse(
            |c| c.set_dlpack_major(0),
            &want,
            "dlpackMajor: required 1, received 0",
        );
        refuse(
            |c| c.set_dlpack_major(2),
            &want,
            "dlpackMajor: required 1, received 2",
        );

        allow(|c| c.set_dlpack_minor(2), &dlpack_minor);
        allow(|c| c.set_dlpack_minor(1), &dlpack_minor);
        refuse(
            |c| c.set_dlpack_minor(3),
            &dlpack_minor,
            "dlpackMinor: required <= 2, received 3",
        );
        allow(|c| c.set_dlpack_minor(0), &dlpack_minor);

        allow(|_| {}, &want);
        refuse(
            |c| {
                let mut ops = c.reborrow().init_operations(1);
                ops.set(0, Operation::Energy);
            },
            &want,
            "operations: required forces, received energy",
        );
        allow(
            |c| {
                let mut ops = c.reborrow().init_operations(3);
                ops.set(0, Operation::Energy);
                ops.set(1, Operation::Forces);
                ops.set(2, Operation::Hessian);
            },
            &want,
        );
        refuse(
            |c| {
                c.reborrow().init_operations(0);
            },
            &want,
            "operations: required energy, received none",
        );
    }

    #[test]
    fn ragged_message_names_the_size() {
        let err = inspect(&[1, 2, 3], &Expectation::default()).unwrap_err();
        assert_eq!(
            err,
            "capabilities: required a word-aligned Capabilities message, received 3 bytes"
        );
    }
}
