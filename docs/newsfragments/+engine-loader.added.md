`rgpot-core` loads any library that exports the generic engine C ABI (`rgpot/engine_c_abi.h`) from Rust.
The `engine` feature adds `rgpot_core::engine::EnginePotential`: it checks ABI compatibility, creates an instance from the backend's configuration message, and answers energy and forces per geometry or batch.
The optional `rgpot_engine_hvp` symbol exposes analytic Hessian-vector products through the same potential handle.
A positive return code is a recoverable `EngineError`, while a negative code permanently marks the instance fatal.
`EnginePotential::into_potential` wraps the instance as a `PotentialImpl`.
