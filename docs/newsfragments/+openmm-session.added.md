Add an engine-agnostic `rgpot_md_session_*` C ABI and `rgpot_core::md_session`
contract in Angstrom/eV/ps/amu units, backed by an in-process OpenMM session
with `3N` hyperplane BAOAB, specular planar walls, deterministic 64-bit seeding,
and unconstrained Verlet NVE. `CAbiMdSession::from_engine_ops` adopts the
operation table an engine's create call writes: a null required callback is
refused with `MdSessionErrorKind::MissingOperation`, and a refused table has
its context released through `destroy`.
