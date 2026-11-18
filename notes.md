# Units follow-up notes

Base: `132f9f45` on `feat/units-constants`.

## Why `CppCore/rgpot/units_rust_export.cc` was deleted

`units_rust_export.cc` is not built by Meson or CMake, and no target or test
references it. Keeping an unbuilt generator creates drift risk (it can silently
rot while still looking authoritative). The Rust constants are now maintained
directly in `rgpot-core/src/units.rs` and are already bit-checked against
`CppCore/rgpot/units.hpp` by `rgpot-core/tests/units.rs`, so deleting the stale
helper keeps one tested source of truth.

## Constant deduplication choices

- Added missing canonical constants to `units.hpp` (with CODATA 2018 / SI 2019
  citations) and rewired duplicate literal sites to those names.
- Kept numeric values bit-identical by reusing the exact previous literals in
  the canonical declarations, then referencing those declarations at use sites.
- Added/extended tests to lock bit-level parity for the refactored conversion
  factors and Rust/C++ constant exports.
