# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [1.1.0](https://github.com/OmniPotentRPC/rgpot/tree/1.1.0) - 2026-03-29

### Added

- **XTBPot** (``-Dwith_xtb=true``): GFN tight-binding via the [xtb](https://github.com/grimme-lab/xtb) C API. Supports GFNFF, GFN0-xTB, GFN1-xTB, and GFN2-xTB. Configure method, accuracy, electronic temperature, charge, and spin through ``XTBConfig``. RPC server accepts ``XTB``, ``GFNFF``, ``GFN0xTB``, and ``GFN1xTB`` potential names. ([#35](https://github.com/OmniPotentRPC/rgpot/issues/35))
- **TBLitePot** (``-Dwith_tblite=true``): GFN tight-binding via the [tblite](https://github.com/tblite/tblite) C API. Supports GFN1, GFN2, and IPEA1. Same config pattern as XTBPot with a lighter dependency footprint. RPC server accepts ``TBLite`` / ``TBLiteGFN2``, ``TBLiteGFN1``, and ``TBLiteIPEA1``. ([#36](https://github.com/OmniPotentRPC/rgpot/issues/36))
- **MetatomicPot** (``-Dwith_metatomic=true``): load [metatomic](https://github.com/metatensor/metatomic) TorchScript atomistic models directly in C++ (no Python at runtime). Neighbor lists from [vesin](https://github.com/Luthaf/vesin); forces via PyTorch autograd. Thread-safe with an internal mutex; caches atomic-types tensors between geometry steps. Configurable device (CPU/CUDA), dtype override, length unit, and consistency checks via ``MetatomicConfig``. RPC server accepts ``Metatomic:<model_path>``. ([#37](https://github.com/OmniPotentRPC/rgpot/issues/37))
- **Units module** (``rgpot/units.hpp`` / ``units.cc``):
  - Compile-time CODATA 2018 constants (Angstrom/Bohr, Hartree/eV, Boltzmann, fused force conversion).
  - Runtime Shunting-Yard unit expression parser with SI dimensional analysis (derived from metatomic-torch PR #173, BSD-3-Clause). Supports compound expressions such as ``kJ/mol/A^2`` and ``(eV*u)^(1/2)``.
  - ``validate_unit(quantity, unit)`` for length/energy/force/pressure/momentum/mass/velocity/charge checks. ([#38](https://github.com/OmniPotentRPC/rgpot/issues/38))
- **RPC unit negotiation**: ``ForceInput`` Cap'n Proto schema gains optional ``lengthUnit`` and ``energyUnit`` fields (defaults ``"angstrom"`` / ``"eV"`` for backward compatibility). The server converts caller positions/box into Angstrom before evaluation and converts energy/forces into the caller's requested units on the way out.
- **pixi environments**: ``xtbbld``, ``tblitebld``, ``tbbld`` (xtb+tblite), and ``metatomicbld`` (linux-64; torch + metatomic-torch + vesin via PyPI) for feature-gated builds.
- Catch2 regression tests for XTBPot, TBLitePot, MetatomicPot (LJ model fixture from eOn), and the unit parser.

### Changed

- Cache hits move the force matrix out of the cache instead of copying it (saves a full ``n_atoms × 3`` allocation on every hit).
- RocksDB cache keys use binary ``Slice`` from hash bytes instead of ``std::to_string``, preserving entropy and avoiding string allocation.
- Eigen adapter copies atom matrices via ``Eigen::Map`` and box matrices via ``memcpy`` instead of element-wise loops.
- Meson metatomic linking supplements CMake Torch discovery with explicit ``-L`` / ``-rpath`` / ``-ltorch_cpu`` from ``torch.__file__`` (avoids unresolved ``libtorch_cpu.so`` when Meson's CMake trace is incomplete).
- Vesin integration uses ``VesinDevice{VesinCPU, 0}`` brace-init so current PyPI
  vesin wheels (struct ``{type, device_id}``) compile; vesin 0.3–0.4 briefly
  aliased ``VesinDevice`` to the device-kind enum.

### Fixed

- Metatomic link failures on conda/pixi layouts where Torch CMake config does not fully propagate ``libtorch_cpu`` into Meson's dependency graph.


## [1.0.3](https://github.com/OmniPotentRPC/rgpot/tree/1.0.3) - 2026-03-01

### Fixed

- MSVC/clang-cl compatibility: use ``/W3`` instead of GCC/Clang warning flags and skip ``-lstdc++`` link arg on Windows ([#31](https://github.com/OmniPotentRPC/rgpot/issues/31))


## [1.0.2](https://github.com/OmniPotentRPC/rgpot/tree/1.0.2) - 2026-03-01

### Added

- External integration guide covering namespace collision mitigation when embedding rgpot as a subproject ([#30](https://github.com/OmniPotentRPC/rgpot/issues/30))

### Changed

- CI dependency bumps and cleanup to prevent trailing whitespace in generated headers ([#29](https://github.com/OmniPotentRPC/rgpot/issues/29))


## [1.0.0](https://github.com/OmniPotentRPC/rgpot/tree/1.0.0) - 2026-02-15

### Added

- doc(arch): add architecture guide covering layer diagram, error conventions, and how to register new potentials
- feat(build): add release CI workflow, meson subproject install guards, CMake FetchContent readiness, and package version config
- feat(cache): add a RocksDB integration
- feat(ci): add `rust_tests` job using cargo-nextest on Ubuntu and macOS with default and `--all-features` configurations
- feat(cpp): add C++ RAII wrappers (`include/rgpot/`) — `PotentialHandle`, `InputSpec`, `CalcResult`, `RpcClient`, `Error`, with full Doxygen documentation
- feat(rpc): add feature-gated Cap'n Proto RPC client and server in Rust, sharing the existing `Potentials.capnp` schema with the C++ side
- feat(rpc): initialize a C style integration to the server
- feat(rpc): initialize a server component
- feat(rust): add Rust core library (`rgpot-core/`) with `#[repr(C)]` types, callback-based potential dispatch, status codes, thread-local error handling, and auto-generated C header via cbindgen
- feat(rust): integrate DLPack tensor exchange protocol via `dlpk` crate — core types now use `DLManagedTensorVersioned*` for device-agnostic data exchange, with borrowed (non-owning) and owned tensor helpers in new `tensor` module
- test(rust): add 39 unit tests covering types, status codes, potential lifecycle, C API, null-pointer handling, error propagation, and `free_fn` invocation

### Changed

- chore(build): add Meson `with_rust_core` option, `rust-test` / `rust-test-all` pixi tasks, and `cargo-nextest` dependency

### Fixed

- fix(cpp): remove incorrect `extern "C"` trampolines from `LJPot.hpp` and `CuH2Pot.hpp` that used raw `int` returns and `void*` params; use typed `PotentialHandle::from_impl<>()` template instead


## [0.0.1](https://github.com/OmniPotentRPC/rgpot/tree/v0.0.1) - 2024-01-26

### Added

- Initial release with C++ core: Lennard-Jones and CuH2 EAM potentials.
- CRTP-based `Potential<Derived>` template with optional caching.
- Cap'n Proto RPC server and client bridge.
- Meson and CMake build systems.
- CI build matrix (Meson/CMake x Linux/macOS x RPC/Cache feature flags).
- RPC integration tests and client bridge stress tests.
