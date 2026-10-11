# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [3.5.0](https://github.com/OmniPotentRPC/rgpot/tree/v3.5.0) - 2026-10-11

### Breaking

- An rgpot client from this release calls `Potential.getCapabilities` (ordinal `@12`) before the first `calculate` and refuses a server that does not implement the call or whose capabilities are incompatible.
  A server built on an older `Potentials.capnp` needs that method before such a client can use it.

- `PotCaps` gained `groupCollective` and `stress`.
  A C++ caller compiled against the 3.4.0 headers has a different `PotCaps` layout from a `librgpot` built from this release.
  `rgpot_cxx_layout_stamp()` reports the loaded layout.

- `LJConfig`, `LJClusterConfig` and `MorseConfig` gained `switch_width`, so their layout changed.
  The C++ layout stamp covers them.

- `librgpot` no longer links MPI.
  Calculator groups need `librgpot_mpi` installed next to `librgpot`, or `RGPOT_MPI_LIBRARY`.
  A consumer that took MPI flags from the `rgpot` meson or pkg-config dependency now asks for `rgpot-mpi` or for MPI itself.

- The CMake package config moved from `lib/cmake/pot` to `lib/cmake/rgpot`.
  `find_package(rgpot)` finds that directory.
  An `rgpot_DIR` that points at `lib/cmake/pot` does not.

- `CppCore/rgpot/nlist/PairListCache.hpp` is installed.
  `PairListCache::accumulate` takes a radial kernel that returns a `PairTerm`, and `CachedPairList::FoldMode` gained `General`.
  The 3.4.0 header exposed `CachedPairList::forEach` instead.

### Deprecated

- `libmetatomic_engine.so` is kept for one release and removed after it.

### Added

- A `umaexport` pixi environment pins `fairchem-core` 2.23.0 and `torch` 2.13, so `pixi run -e umaexport python scripts/export_uma_aoti.py --help` is reproducible.
  The `export-uma-aoti` task runs the exporter in it.

- A standalone C plugin ABI (`include/rgpot/plugin.h`, `RGPOT_PLUGIN_API_VERSION` 1) lets a shared library export `rgpot_plugin_init()`.
  The host scans `RGPOT_PLUGIN_PATH`, checks the API version and registers the plugin.
  `RGPOT_CAP_INSTANCE_SAFE` means instances may calculate concurrently; without it the host serializes every instance of that plugin.

- Backends loaded through the potential profile are checked before dispatch.
  `ProfileSession::load` and `rgpot::abi::checked_load` read the backend's `Capabilities` and refuse a different protocol family or major, a lower protocol minor than required, another schema identity, an incompatible eindir bridge ABI, layout or feature set, another DLPack revision, or a missing energy or forces operation.
  Metadata a backend leaves unset is not checked.

- Effective medium theory and the cell-list embedded-atom potential now live in this library.
  EMTConfig.rasmussen selects the Rasmussen parameter set.
  The copper cluster energy remains 5.129167 eV.

- Every backend gets the generic `rgpot_engine_*` table through one adapter over `PotentialBase` (`rgpot/engine/EngineTable.cc`): new engine libraries `classical_engine` (lj, ljcluster, morse, zbl and the Fortran potentials), `mopac_engine`, `nwchem_engine` and `skala_engine`, and the generic table added to the UMA, Metatomic and XTB engines.
  The table adds `rgpot_engine_name`, `rgpot_engine_version` and `rgpot_engine_model_digest` (SHA-256 of the loaded model file), a stated thread-safety policy per reentrancy class, and engine ABI minor 2.
  `UmaParams` and `MetatomicParams` carry the torch thread counts (default 1) and determinism requests, `MetatomicParams` also the rotation and probe options, and the engine applies them at create.
  `potentials-schema` is repinned to the revision that adds them.

- Lennard-Jones, Morse, and ZBL write the Cauchy stress from the same pair force.
  xTB and tblite copy the virial the single point already stores.

- Metatomic evaluates a batch in one model forward and reports that capability.
  The engine file is librgpot_metatomic_engine.so.

- TIP4P, SPC/E and TIP4P on platinum compile when `-Dwith_water=true`.
  The option defaults false.
  Each kernel keeps the 8.5 angstrom cutoff and the 1.0 angstrom switching width.

- The C++ classes that cross the library boundary have a layout stamp: `rgpot::abi::header_stamp()` is a `constexpr` of `rgpot/abi/CxxLayout.hpp`, `rgpot_cxx_layout_stamp()` reports the loaded library, and `rgpot::abi::layout_compatible()` compares them.
  It covers the force structures, `PotCaps`, the pair potential classes and their configs, the `RGPOT_HAS_CACHE` bit and the virtual-function counts of `Potential`, and exists in builds without the Rust core.

- The NWChem multi-rank force host and the collective RPC server own their MPI runtime through `MpiHost`.
  A throw on one rank is reduced across the communicator and rethrown on every rank with that rank's message, so the ranks leave the call together.

- The generic engine table gains optional symbols and a minor version: `rgpot_engine_abi_stamp` / `rgpot_engine_abi_compatible` (major, minor, layout revision), `rgpot_engine_caps`, `rgpot_engine_last_error`, `rgpot_engine_set_num_threads` and `rgpot_engine_set_charge_spin`, all exported by `libuma_engine.so`, which is now linked with `-z nodelete`.
  The `umaenginebld` pixi environment and a CI leg build the engine and run `UmaEngineTest`.
  `potserv` builds when metatomic and rpc are both enabled.

- `CPMDPot::selectOrbitals(key)` names the image or bead the next forces belong to and passes it to `cpmdc_session_select_orbitals`, so an engine that keeps orbitals per key starts each SCF from that image's or bead's own previous orbitals.

- `Capabilities` carries `buildVersion` and `buildRevision` (appended at `@18` and `@19`), filled by the Rust and C++ producers, and `rgpot_source_revision()` returns the source revision of the loaded library.
  The revision comes from `RGPOT_SOURCE_REVISION` or the git checkout and is empty when neither exists.

- `LJConfig`, `LJClusterConfig` and `MorseConfig` take `switch_width`: a positive width replaces the shifted truncation with the C^2 quintic switch `S(t) = 1 - 10t^3 + 15t^4 - 6t^5` over the last `switch_width` Angstrom below the cutoff, so the energy, force and curvature reach zero there.
  The default 0 keeps the shifted truncation and its cache keys.
  `scripts/validation/pair_kernels.py` (sympy) checks every pair kernel's force against `-dE/dx` and the switch's C^2 conditions; `scripts/validation/quintic_switch.sollya` bounds its binary64 evaluation error.

- `PotentialCache` counts computed and served evaluations per handle (`counts()`, `reset_counts()`).
  The same split is available on eindir potentials through `rgpot_potential_eval_counts` and `rgpot_potential_reset_eval_counts`, and from Python as `PotentialCache.counts()` (bound when built with the cache), so two runs under different cache states report the same computed count.

- `XcKernel` serves the `mgga_lapl` family as well as `lda`, `gga` and `mgga_tau`.
  Laplacian contractions need the AO Laplacians in `lapl_chi` and return 2 when those are missing, leaving the output unchanged.

- `pixi run check-eonviz-pins` (`scripts/check_eonviz_pins.sh`) fails when an `eonviz` dependency is a git branch or tag, lacks a full commit `rev`, or when `chemparseplot` or `rgpycrumbs` is not an exact `==` pin.

- `rgpot.h` carries `RGPOT_ABI_VERSION_MAJOR`, `RGPOT_ABI_VERSION_MINOR` and `RGPOT_ABI_LAYOUT_REVISION` with an `rgpot_abi_stamp_t` struct, and the library exports `rgpot_abi_stamp()` and `rgpot_abi_compatible(const rgpot_abi_stamp_t *)`.
  A loaded `librgpot` is compatible with a caller when the major and layout revision are equal and the library minor is at least the caller's.
  The C ABI reference page states when each number changes.

- `rgpot_potential_set_peer_capabilities` installs the peer's `Capabilities` message on a callback handle.
  An incompatible peer makes `rgpot_potential_calculate`, `eindir_objective_eval` and `eindir_objective_grad` fail before the callback runs, with `peer refused: <first mismatch>` in `rgpot_last_error()`.
  A handle that never receives a message keeps the callback contract.

- `rgpot_version()`, `rgpot_version_major()`, `rgpot_version_minor()` and `rgpot_version_patch()` report the version of the loaded library, so a C caller can check it against the `RGPOT_VERSION*` macros of the header it compiled with.
  `rgpot.h` carried 3.3.0 through the 3.4.0 release; `potctl release sync` now stamps its macros and `potctl release assert` fails when they drift.

- `scripts/export_uma_aoti.py` embeds `natoms`, per-element `counts` and the `model`, `torch_version` and `fairchem_version` that produced the package.
  UmaPot checks `natoms` and `counts` against every input, so a package for C2H2 refuses C2H4 although both share `z_set` `[1, 6]`.

- `scripts/export_uma_aoti.py` records which export path produced a package (`export_path`), documents that torch 2.13 with fairchem-core 2.23 always takes the static `make_fx` fallback, and warns when that fallback runs without `--molecular-box`.

- rgpot::BatchEvaluator evaluates many configurations in parallel: it builds one potential per worker from a factory, splits the batch into contiguous ranges and returns results in input order, with no re-entrancy requirement on the potential.
  A worker failure is rethrown after every worker has joined.

### Developer

- A plugin CI leg (potctl ci plugin-test) runs PluginTest, PluginContractTest and ExternalPluginTest on Ubuntu and macOS and is required by the CI gate.

- CI runs the check-eonviz-pins pixi task as a blocking job.

- Coverage generation fails on test failures and lcov errors; only the Codecov upload remains advisory.
  The coverage scripts accept lcov 2.

- Every GitHub Action reference is pinned to a full commit SHA with the version in a trailing comment; ci/gha/check-drift.sh rejects unpinned references.

- Pull request CI runs hold read-only permissions: docs deployment and the Codecov OIDC upload run only on pushes, in their own jobs.

- The MPI contract client runs the capability handshake against the collective RPC server before it configures or evaluates, and a second run requires a protocol major the server does not speak and must stop at the handshake with the reason, for both the serial and the MPI server layouts.

- The Python contract tests cover the recursive `ExprParams` / `PotSpec` tree: the schema names the tree, `PotentialConfig` stays a single-backend union, and a nested `0.5*inner + d3` composition survives a Cap'n Proto round trip.

- The macOS wheel job is required and validates the wheel contents, and publishing waits for it.

- The wheel test command runs `tests/test_rgpot_pip_core.py` against each repaired wheel, so the runpath, import and abi3 checks gate the wheel jobs.

- `CapabilitiesWire` decodes `Capabilities` messages across the complete pre-build-identity `Potentials.capnp` (`CppCore/tests/fixtures`) and requires every member that file declares to exist in the current schema with the same ordinal and type.

- `RPCSchemaSync` regenerates the Cap'n Proto artifacts from both vendored `Potentials.capnp` copies, requires the output and the code generator request to match, and requires the build's generated header and source to equal a fresh regeneration.

- tests/check_wheel_contents.py fails when a built wheel contains the xckernel generator files (catalog.py, cbackend.py) or depends on sympy; both wheel jobs run it.

- The Ubuntu Meson job passes -Dwith_mpi=enabled and runs calculator_group_test under mpirun with four ranks.

### Changed

- Every Fortran potential (SW, EDIP, Lenosky, Tersoff, EAM Al, FeHe, CuH2, Water-H) keeps its neighbour table and error message in a per-instance workspace and reports `Reentrancy::PerInstance` with `perImageInstances`, replacing `ProcessSerial`.
  Separate instances evaluate on separate threads, and a NEB that keeps one instance per image no longer rebuilds one shared table as the images alternate (220 of 240 calls rebuilt it in a 601-atom EAM aluminium NEB, 27.5 percent of its samples).
  The C entry points without a workspace remain for the CuH2 plugin and stay process-serial.
  eOn 3.6.0's EAM_AL needs one instance per image, which this provides ([#95](https://github.com/OmniPotentRPC/rgpot/issues/95)).

- LJPot, LJClusterPot, MorsePot and ZBLPot search pairs with linked cells once the box holds 64 or more cells of one cutoff plus skin, so a fresh search costs O(n) instead of O(n^2), and fully periodic boxes past 20000 atoms keep their cached pair list.
  The cached force loop runs over a sorted CSR half list with the periodic image recorded at build when cutoff + skin is below half the box, and keeps each atom's force in registers across its row.
  `PairListCache::accumulate` takes a radial kernel returning a `PairTerm` (energy and `-V'(r)/r`) and returns the energy.

- The EAM aluminium kernel evaluates four exponentials per neighbour entry instead of up to eighteen: the density pass keeps rho(r) and rho'(r) for the force pass, the shifts at the cutoff are computed once per call, and integer powers and the embedding polynomial run without `pow`.
  `scripts/validation/eam_al.py` (sympy) checks the rearranged forms and the gather force against the closed form ([#95](https://github.com/OmniPotentRPC/rgpot/issues/95)).

- The Fortran potentials ask vesin for unsorted neighbour lists and order each atom's row by partner index with two linear counting sorts, which replaces vesin's global sort (10.6 percent of an EAM aluminium NEB) and keeps every force sum in one order whether vesin reused its cached topology or rebuilt it ([#95](https://github.com/OmniPotentRPC/rgpot/issues/95)).

- The Rust `RpcClient` and the `pot_calculate` bridge call `Potential.getCapabilities` before the first `calculate` and refuse a server with incompatible capabilities.
  A server that does not implement the call is refused with an error naming `Potential.getCapabilities` and the upgrade path; the rgpot servers (`potserv`, `rgpot_rpc_server_start`) implement it.
  Servers built on `Potentials.capnp` elsewhere need the call before an rgpot client built from this release can use them.

- The `eonviz` pixi environment pins `chemparseplot` and `rgpycrumbs` to the 1.11.0 releases instead of git branches, and raises the `readcon` floor to 0.14.7.

- `librgpot` no longer links MPI.
  With MPI found, the build adds `librgpot_mpi` (meson dependency and pkg-config name `rgpot-mpi`) holding every MPI call of the calculator groups, and `bindCalculators` or `calculatorsUseMpi` load it with `dlopen` on first use through one C entry point, `rgpot_mpi_api_v1` (`rgpot/calculator_mpi_abi.h`).
  A host that never asks for calculator groups no longer maps libmpi, whose UCX load-time hooks cost 0.4 s per process start.
  `loadCalculatorMpi`, `calculatorMpiLoaded` and `calculatorMpiLoadError` let a host load it from a chosen path (or `RGPOT_MPI_LIBRARY`) and report failures.

- `uma_ase_parity` requires the `.pt2` model path as its argument and prints a usage line without it; the former default was a TorchScript path `UmaPot` rejects.

- Energies and forces of the pair potentials and the Fortran potentials can differ from 3.4.0 in the last bits, because sums now run in sorted CSR and partner order.

### Fixed

- TBLitePot builds against tblite 0.6.0.
  The GFN calculator constructors take an optional config pointer there, and a null config keeps the previous behaviour.
  Older tblite keeps the two-argument call.

- The UmaPot README and `UmaConfig::cutoff` described a `model.pt2.json` sidecar that overrides the config.
  UmaPot reads only the metadata the exporter embeds in the `.pt2`; the docs say so and list the keys it uses.

- The eindir callbacks validate what a force callback returns with `RGPOT_SUCCESS` before reading it: a null forces tensor or data pointer, a wrong dtype, rank or element count, a non-finite force and a non-finite energy are refused with `EINDIR_INVALID_PARAMETER` and a `rgpot_last_error()` message, and the gradient output is left untouched.
  `n_atoms * 3` overflow is refused at construction.

- The result cache no longer deadlocks a bound MPI world.
  A potential that reports the new `PotCaps::groupCollective` (`CPMDPot` does) has its cache hit or miss decided jointly over its calculator through the new `calculatorAgree`, so a geometry one rank holds and another misses is computed on every rank of the calculator instead of leaving the missing rank alone in the engine's collective.

- UmaPot checks the charge, spin, task and composition embedded in an AOTI package against its `UmaConfig` and the input atoms on every force call, and throws `rgpot::UmaContractError` on a mismatch.
  Without the check, a config whose charge or spin differs from the package's evaluates a different potential energy surface and reports no error.

- `CPMDPot::forceImpl` exchanges force errors on its calculator's communicator instead of `MPI_COMM_WORLD`.
  A host that hands calculator groups an uneven batch (two NEB endpoints on four groups) left the computing groups in that world exchange while the idle groups waited in `shareFromCalculator`, and every grouped run hung at its first such batch.
  A failure still prints on every rank of the failing calculator and aborts the whole world.

- `UmaPot` refuses an edge count outside the range recorded in the `.pt2` metadata (`nedges_min`, `nedges_max`) with an `UmaContractError` on field `nedges`, instead of passing the compiled graph a shape it was not exported for.
  `scripts/export_uma_aoti.py` writes both keys: 1 to 65536 for a dynamic export, the traced edge count for a static one.

- `scripts/export_baker_uma_aoti.py` deduplicates Baker systems by exact composition, charge and spin.
  Keying by `z_set` merged systems that share an element set but not their atom counts, and their packages cannot evaluate each other.

- In 3.4.0 a periodic pair list assumed an orthogonal box, and LJPot warned that the box is assumed to be orthogonal.
  Periodic images now use the full-lattice minimum image, and an ill-conditioned cell throws.
  Energies and forces on a non-orthogonal cell change.

- A cache hit restores the stored variance and stress.
  Records keep the energy and force prefix and append versioned variance and stress metadata.
  A record from an older build still reads, and a potential that reports stress treats a record without that metadata as a miss.

- The CMake install exports the neighbor-list library, xxHash, the Fortran runtime, the public Fortran potentials and their kernels, the CuH2 plugin entry points, the calculator schemas, the engine ABI headers, and the optional MPI calculator companion.
  A static core is usable from a shared consumer, and static kernels stay hidden at that link.
  MPI host targets are enabled only when mpi.h compiles.
  On Apple the Fortran Coulomb plugin links its C driver.

- Core potentials compile with their feature definitions.
  Engine plugins link the core archive on demand.
  potserv takes the torch dependencies when metatomic and rpc are both enabled.

- The UMA exporter rewrites a strided force-slice write so a snapshot of the base stays independent of later writes.
  Molecular inputs are centred before they are converted to the model's precision.
  A rewritten AOTI archive is published without clobbering a package another reader still has open.
  An export candidate is kept only when its tensors run and match the reference.
  Package publication and reads retry a transient Windows sharing error.

- An LDA XcKernel accepts basis values that have no gradients.
  The Python bindings check XC operand arrays and keep them alive for the call.

- The CPMD engine is given the calculator group's communicator.
  Communicator adopters are retained and deduplicated across binding orders.
  A refused calculator binding stops the next CPMD force evaluation.
  Registered CPMD callbacks stay on one process-lifetime engine handle.
  NWChem keeps one engine handle for each resolved function table for the life of the process.

- Pair sums of one geometry now agree bit for bit between a cold scan and a cached CSR list.
  In 3.4.0 those two orders could differ in the last bits, and repeated eOn NEB runs differed at 1e-12 eV.

## [3.4.0](https://github.com/OmniPotentRPC/rgpot/tree/3.4.0) - 2026-09-30

### Added

- Calculator groups report how many there are and broadcast a result from one group to every rank; the meson build takes MPI through `-Dwith_mpi=enabled`.

### Developer

- The twelve-rank refused calculator split calls finalizeMpiAtExit, so mpirun can exit 0 after MPI_Init.
- The two-rank calculator share that matches calls finalizeMpiAtExit, so mpirun can exit 0 after MPI_Init.
- mpirun -n 4 calculator_group_test checks that rank 3 holds the buffer rank 0 filled. An out-of-range calculator owner aborts every rank.

### Changed

- Read the engine documentation at https://cpmdc.rgoswami.me.

### Fixed

- A failed CPMD engine call sets an abort flag, and the exit handler calls `MPI_Abort` instead of a collective `MPI_Finalize` that peers stuck in a broadcast would never reach.
- A force that throws on one rank is printed on every rank in that call, then MPI_Abort runs on MPI_COMM_WORLD. MPI_Finalize at exit runs only when this library called MPI_Init.
- A release tag no longer fails when `publish.yml` has already uploaded `rgpot-core` to crates.io: `release.yml` skips `cargo publish` for a version that exists, so its GitHub release job still runs.
- A shareFromCalculator call that cannot enter the broadcast aborts every bound rank after those ranks print the error.
- The HTML sitemap lists each page at the site root.
- The Morse cutoff-shift test compares `energyShift()` with an absolute tolerance at the operands' rounding scale instead of a 1e-14 relative pin on a cancelled difference, so the suite passes on aarch64 (EESSI 2026.06, Neoverse-N2), where `exp` and multiply-add contraction differ from x86 by an ulp.
- The calculator namespace page names the rank that fills ForceOut and the first rank of a share broadcast.
- The one-shot CPMD gradient path keeps a nine-component stress tensor when the engine computed one.
- `CPMDPot::setParams` with byte-identical parameters keeps the engine session and its stored wavefunction; the engine loaded by `bindCalculators` stays loaded for the process; `paramsKey` fingerprints the CPMD parameters for the result cache.
- calculatorsUseMpi follows MPI_Initialized. A refused calculator split reports the MPI_COMM_WORLD size and a negative group index.


## [3.3.0](https://github.com/OmniPotentRPC/rgpot/tree/3.3.0) - 2026-09-28

### Added

- CPMD ranks can be split into one calculator group per image.
- The new `rgpot_eindir_abi_stamp()` returns the eindir-core ABI stamp of the objective base in `rgpot_potential_t`. The generated `rgpot.h` stays valid C, with the fused evaluation cache behind an opaque pointer.

### Changed

- Always compile the xTB dlopen frontend (`XTBDlopen`) without `-Dwith_xtb`, matching Metatomic/NWChem: hosts ship without NEEDED `libxtb` and load `libxtb_engine.so` via `RGPOT_XTB_ENGINE` at runtime. Linked `XTBPot` and the engine plugin still require `with_xtb`. Config types moved to `XTBConfig.hpp` (no `xtb.h`). ([#53](https://github.com/OmniPotentRPC/rgpot/issues/53))

### Fixed

- `CPMDPot` copies `PotentialResult.stress` into `ForceOut` when the engine returns nine components.
- RocksDB 11 builds of the cache work: `PotentialCache` opens the database through the `std::unique_ptr` `DB::Open` overload on RocksDB 10.4 and later. ([#54](https://github.com/OmniPotentRPC/rgpot/issues/54))
- A `-Dwith_cache=true` build lists `-DRGPOT_HAS_CACHE=TRUE` and `rocksdb` in `rgpot.pc`, so consumers of the cache API find its flags and headers through pkg-config. ([#55](https://github.com/OmniPotentRPC/rgpot/issues/55))
- Configuring rgpot, on its own or as a Meson subproject, no longer warns that `add_languages` is missing `native:`.
- Linux wheels import without a system OpenBLAS. auditwheel repaired the real librgpot library but not the librgpot.so.3 and librgpot.so copies that _core loads, so those still named libopenblas.so.0.
- Linux wheels vendor the OpenMP runtime: `import rgpot` no longer needs a system `libgomp.so.1`.
- rgpot configures offline: the `potentials-schema` subproject is optional, and the two schema-sync tests run only when it is available, so `--wrap-mode=nodownload` builds no longer fail at `meson setup`.

### Miscellaneous

- OIDC trusted publishing on tag v* and workflow_dispatch. ([#76](https://github.com/OmniPotentRPC/rgpot/issues/76))


## [3.2.0](https://github.com/OmniPotentRPC/rgpot/tree/3.2.0) - 2026-09-13

### Added

- MOPACPot dlopens the split [mopacc](https://github.com/OmniPotentRPC/mopacc)
  engine (`libmopacc.so`, OpenMOPAC). Packed `MopacCParams`.
  Default model is AM1.
- NWChem RPC specifications accept basis, theory, exchange-correlation, and SCF
  iteration controls for reproducible electronic-structure requests.
- Skala XC is available as C++ ``SkalaPot``: NWChem DFT through ``libnwchemc``
  with ``dft.xc`` defaulting to ``skala-1.1``. Enable with ``-Dwith_skala=true``
  (requires ``-Dwith_rpc=true``).
- UmaPot evaluates UMA / OMol from an AOTInductor ``.pt2`` (``scripts/export_uma_aoti.py``): vesin builds the neighbor list, the compiled graph returns energy and forces. Charge and spin are per-call tensor inputs. One ``.pt2`` is exported per composition (``merge_mole``); the sidecar records cutoff, neighbors, task, charge, spin, ``z_set``, and label. ``scripts/export_baker_uma_aoti.py`` walks Baker endpoints and deduplicates by ``(z_set, charge, spin)``. potserv accepts ``Uma:<model.pt2>`` and ``Uma:<model.pt2>:<task>``.

### Fixed

- CI pins snapper-fmt to v0.10.0. Latest 0.11.2 panics on UTF-8 in org files.
- Metatomic-enabled build-tree executables locate Torch and metatomic shared
  libraries without an ``LD_LIBRARY_PATH`` override.
- UmaPot clears PT_GNU_STACK PF_X on AOTI wrapper.so before dlopen, so Elja-traced ``.pt2`` packages load on hosts that refuse an executable stack. ``scripts/export_uma_aoti.py`` applies the same rewrite at mint time.


## [3.1.2](https://github.com/OmniPotentRPC/rgpot/tree/3.1.2) - 2026-09-04

### Fixed

- D3/D4 in the wheel are built from pinned meson wraps (static s-dftd3 / dftd4), not a conda prefix. The dftd4 wrap uses OpenBLAS as its LAPACK. ``pip install`` no longer needs vendored ``libblas`` from micromamba.


## [3.1.1](https://github.com/OmniPotentRPC/rgpot/tree/3.1.1) - 2026-09-04

### Added

- The PyPI wheel compiles ExprPot, D3Pot, D4Pot, and XcKernel and vendors ``s-dftd3`` / ``dftd4``. ``import rgpot`` exposes those classes (``has_expr`` / ``has_dftd3`` / ``has_dftd4`` / ``has_xckernel``).


## [3.1.0](https://github.com/OmniPotentRPC/rgpot/tree/3.1.0) - 2026-09-04

### Added

- A wheels workflow builds and publishes the Python package. rgpot had none:
  2.5.4 reached PyPI hand-built from a branch, with no tag and no
  main-branch source behind it, which is how the version surfaces drifted
  apart in the first place. A `v*` tag now builds manylinux and macOS
  wheels plus an sdist with cibuildwheel and publishes them; a manual run
  defaults to TestPyPI so an upload can be rehearsed before the
  irreversible one.

  The build step refuses a wheel whose `librgpot` carries fewer than the
  eight Fortran `bind(c)` entry points, so the potentials cannot silently
  fall out of the artifact again.
- Add `scripts/time_lj_rgpot_vs_pyeonclient.py` to time in-process `LJPot`
  against pyeonclient `Matter` on the ExprPotTest two-atom LJ fixture.
- D3Pot wraps the s-dftd3 ISO C API (``-Dwith_dftd3=true``, pixi env ``dftd3``). Config selects BJ/zero damping, a functional key, and an explicit ATM three-body flag. In-process Potential summand; no ``PotentialConfig.d3`` arm.
- D4Pot wraps the dftd4 ISO C API (``-Dwith_dftd4=true``, pixi env ``dftd4``). Config selects a functional key, total charge, and an explicit ATM/many-body flag. In-process Potential summand; no ``PotentialConfig.d4`` arm.
- ExprPot algebra goldens pin identity ``lj``, ``0.5*lj + morse``, ``2*lj - lj``, and ``0.5*(lj+morse)`` under ``CppCore/tests/data/expr/`` (energy 1e-14 eV, forces 1e-12 eV/A). Missing fixtures fail closed. Optional ``0.5*lj + d3`` when ``-Dwith_dftd3=true``.
- ExprPot compiles a Lepton energy string over named ``unique_ptr<PotentialBase>`` children (``-Dwith_expr=true``). Construct-time name checks fail closed. No PotentialConfig arm, no XcKernel term.
- ExprPot forces use analytic Lepton ``differentiate(name)`` weights compiled at construct: ``F = sum_i (df/dE_i) F_i``. No finite difference, no muParser.
- ExprPot wraps through ``PotentialHandle::from_impl`` / ``rgpot_potential_new_eindir`` as one eindir objective. A single ``rgpot_potential_calculate`` returns ``0.5*lj + morse``.
- Golden masters pin D3 BJ ATM on/off and D4 PBE energy and forces against s-dftd3 / dftd4 on a water-octamer cube (``meson test --suite dftd``).
- In-process `XcKernel` contracts Libxc derivative arrays into AO XC Fock and fxc matrices via the libxckernel C ABI (`-Dwith_xckernel=true`, off by default). This is not a `Potential` and has no `PotentialConfig` arm.
- Optional conda-forge s-dftd3 / dftd4 gates (``-Dwith_dftd3`` / ``-Dwith_dftd4``, pixi features ``dftd3`` / ``dftd4``). Default workspace and wheel stay free of those libraries.
- Optional vendored OpenMM Lepton gate (``-Dwith_expr``, ``third_party/lepton``). Default workspace and wheel stay free of Lepton object code.
- RPC/config can name a recursive ``ExprParams`` / ``PotSpec`` tree (for example ``0.5*lj + d3``). ``PotentialConfig`` stays a single-backend configure union. The C++ ``ExprPot`` constructor remains the first-slice API.
- TDA pin operator (H2O/sto-3g singlet roots) is also compared to libnwchemc ``task tddft`` / TDA via ``scripts/compare_xckernel_tda_nwchemc.py``. Exclusive sigma contraction stays 1e-17; the nwchemc root residual is the measured engine value (1e-6).
- `Potential::forceBatch` evaluates several independent systems in one call.
  The default `forceBatchImpl` loops over `forceImpl`, so every existing kernel answers a batch without an override.
  When the result cache is on, hits are served per system and only the misses reach the kernel as one compact batch.
- `XcKernel` assembles TDA and RPA sigma over the singlet `st_o2_p` C kernels plus host Coulomb, matching the PySCF `lib.einsum` contraction path and `eval_rho` (`ao @ dm`). The meson golden and live `TDA.gen_vind` / `gen_tdhf_operation` both meet exclusive 1e-17 against those committed sigma pins.
- `rgpot-core` can load any minimum-profile backend by prefix, keep one persistent session, and exchange the shared `PotentialConfig`, `ForceInput`, and `PotentialResult` carriers in-process.

### Changed

- Docs credit Antics and load only `antics.js`; Umami tracker and label are gone.
- The 1 MiB large-file CI gate now allowlists ``CppCore/tests/data/``, ``third_party/``, and ``pixi.lock`` (XcKernel goldens and the lockfile). Other tracked files still fail at 1 MiB.

### Fixed

- D3Pot now passes a sigma buffer to ``dftd3_get_dispersion`` so s-dftd3 fills the gradient (forces were previously left at zero).
- GGA TDA/RPA ``applyFxc`` keeps the generated 7-term monomials on the host long-double evaluator so C matches live ``gen_vind`` at exclusive 1e-17.
- LDA TDA/RPA fxc forms `wv = w * rho * (v2rho2_0 + v2rho2_1)` and tiles
  stage B at PySCF `BLKSIZE` (128). `--tda-rpa` pins host J from the
  `get_j` call inside `gen_vind` / `gen_tdhf_operation`.
- The cp312 wheel now carries the `abi3` tag: `tool.meson-python.limited-api`
  declares the stable-ABI build to meson-python, so one cp312-abi3 wheel
  installs on every CPython >= 3.12 instead of 3.12 alone.
- The metatomic torch dependency links on macOS. Its link arguments carried
  `-rpath-link` and `--as-needed`, which are GNU ld only: Apple's linker
  rejects them outright, so any Darwin build with `-Dwith_metatomic=true`
  died with "unknown options" while producing `librgpot.3.dylib`. Those
  flags now apply on GNU linkers only; Apple's resolves the dylibs from the
  library path and their install names.
- Wheel publish falls back to the `pypi-rgpot` / `testpypi-rgpot`
  `PYPI_TOKEN` until a matching trusted publisher is registered.

  cibuildwheel `build` must be a space-separated string so cp310, cp311,
  and the cp312 abi3 wheel all ship (a TOML array only built cp310).
- `--pyscf` Fock live-PySCF regen now fails above exclusive 1e-15 after long-double stage A/B (mGGA-tau float64 einsum was 1.50e-15).
- `--tda-rpa` keeps committed TDA/RPA host-J pins. Live `get_j` jitters
  about 1e-15 on this case and is reported, not written over the Coulomb
  pin the exclusive 1e-17 C compare uses.
- `--tda-rpa` now writes same-SCF extras and MANIFEST before the exclusive 1e-17 live-vs-committed gate; sigma/J pins are the MO-replay `gen_vind` / `get_j` (a fresh RKS is not bit-stable at that scale).
- `--tda-rpa` regen pins PySCF to one OpenMP thread and replays committed MOs so live `gen_vind` / `gen_tdhf_operation` can meet exclusive 1e-17 against the sigma pins.


## [3.0.2](https://github.com/OmniPotentRPC/rgpot/tree/3.0.2) - 2026-08-08

### Fixed

- Windows MSVC builds Cap'n Proto with lean windows.h defines, and links Flang pot kernels through MSVC lib.exe archives so llvm-ar no longer triggers LNK1107. ([#61](https://github.com/OmniPotentRPC/rgpot/issues/61))
- MetatomicPot no longer fails to load exported PET-MAD (and similar) models on metatensor-torch 0.10.3: scripted modules without `_mts_buffer_names` no longer trip the mixed-dict `.to()` walk.
- Python 3.x wheels import without torch: fat MetatomicPot is no longer
  `link_whole`'d into `librgpot` when `with_python` (engine stays dlopen-only),
  and the repair step ships `librgpot.so.3` (mesonpy zip dropped the soname link)
  plus a RUNPATH into site-packages torch/metatensor/vesin.


## [3.0.1](https://github.com/OmniPotentRPC/rgpot/tree/3.0.1) - 2026-07-26

### Fixed

- The Python wheel builds the Fortran 2018 potentials. `[tool.meson-python.args]`
  never passed `-Dwith_fortran_pots`, so a wheel carried a `librgpot` with no
  Fortran-backed potential in it -- SW, EDIP, Lenosky, Tersoff, EAM-Al, FeHe,
  CuH2, and TIP4P-H were all missing from the one artifact most users install.
  The cibuildwheel configuration installs a Fortran compiler in the build
  image to match, and the repair step vendors `libgfortran` as it would any
  other dependency.


## [3.0.0](https://github.com/OmniPotentRPC/rgpot/tree/3.0.0) - 2026-07-26

### Added

- Pot-hosting groundwork for absorbing eOn's potential kernels:
  `PotCaps` capability descriptors (`PotentialBase::caps()`) replace
  caller-side thread-safety lists; `paramsKey()` parameter fingerprints
  (FNV-1a + kernel-version salt) join the result-cache key so instances
  with different parameters never share entries; `LJConfig` establishes
  the plain-aggregate parameter convention; a `pot_bench` Catch2
  microbenchmark harness gates future pot migrations; `potctl` locksteps
  `pyproject.toml`. ([#57](https://github.com/OmniPotentRPC/rgpot/issues/57))
- Eight potentials absorbed from eOn arrive as Fortran 2018 kernels under
  `CppCore/rgpot/fortran/`: Stillinger-Weber, EDIP, Lenosky, Tersoff, EAM
  aluminium, FeHe, CuH2, and TIP4P-H. Each is a rewrite rather than a
  wrapper -- modules with `implicit none`, derived-type parameters in place
  of COMMON blocks, kinds from `iso_fortran_env` asserted against the C
  types at compile time, `intent` on every argument, `pure` kernels,
  structured control flow, and status returns instead of `stop`.

  Neighbours come from vesin through `rgpot_neighbors`, a CSR full-list
  table carrying pair vectors and distances, and the vendored vesin Fortran
  interface gained the C API's Verlet `skin` option so the kernels keep the
  list caching their predecessors had. Pair sums are restated as gathers:
  each atom accumulates the whole force acting on it and writes one column,
  so the atom loops run under `do concurrent`. Bond-order and embedding
  kernels reach the same form in three passes, since the environment
  derivative is not known until its sum completes.

  Each kernel carries a Fortran-only test (translation invariance, zero net
  force, analytic forces against central differences of the energy) and is
  pinned in `FortranPotsTest` to the reference energies of the kernels it
  replaces. The kernel archives link under `--exclude-libs`, so no Fortran
  symbol reaches the library interface; a test fails the suite if one ever
  does.
- Three classical potentials absorbed from eOn, always built: `MorsePot`
  (pairwise Morse with the shifted cutoff, platinum defaults through
  `MorseConfig`), `LJClusterPot` (12-6 Lennard-Jones on free boundaries,
  `LJClusterConfig`), and `ZBLPot` (screened nuclear repulsion with the
  LAMMPS switching function, `ZBLConfig{cut_inner, cut_global}`, which
  rejects a cutoff pair outside `0 < cut_inner < cut_global`). All three
  take their pairs from the shared `PairListCache`, carry `paramsKey()`
  fingerprints, and appear in `pot_bench`. `PotType` gains `Morse`,
  `LJCluster` and `ZBL`.

### Changed

- `LJPot` pair search goes through `rgpot::nlist::PairListCache` (header-only
  port of eOn's PairListCache): Verlet-skin cached candidate lists in a
  process-global proximity-matched pool, lazy list capture so one-shot
  evaluations run a single fused scan, exact per-call MIC folding, and the
  inverse-r2 pair math instead of `pow()`/`sqrt()`. Boxes where MIC caching
  is unsound keep the historical per-call scan. ([#56](https://github.com/OmniPotentRPC/rgpot/issues/56))
- One versioned `librgpot.so.3` umbrella: every pot builds as a PIC static
  convenience library folded in via `link_whole`; only dlopen'd engine
  plugins ship as separate shared objects. `rgpot.pc` always installs when
  rgpot is the top-level project, exporting the umbrella with
  `RGPOT_HAS_*` feature cflags. vesin resolves once at top level
  (installed >=0.6 preferred, vendored `third_party/vesin` fallback with
  the Fortran interface), and the new `with_fortran_pots` feature
  decouples Fortran-backed potentials from the RPC role options.
  Statistics counters in `registry<T>` are atomic, and public headers no
  longer inject `AtomMatrix` at namespace scope. CI gains the first
  Windows leg and an offline `--wrap-mode=nodownload` rehearsal of the
  release-tarball build. ([#57](https://github.com/OmniPotentRPC/rgpot/issues/57))
- The Cap'n Proto schema ships inside `librgpot` instead of a separate
  `libptlrpc.so`. Consumers that bundle the umbrella alone -- pip wheels
  especially -- no longer need a second shared object beside it, and
  `rgpot.pc` correspondingly stops listing `-lptlrpc`; the symbols are
  unchanged and still exported from `librgpot`. `potserv` and
  `pot_client_bridge` take the generated translation unit from the shared
  archive rather than compiling their own copy.
- The CuH2 potential is the in-tree Fortran 2018 kernel
  (`CppCore/rgpot/fortran/rgpot_cuh2.f90`), reached through
  `rgpot::fortranpots::CuH2Pot` from `rgpot/fortran/FortranPots.hpp`. It
  matches the energy and forces of the kernel it replaces to 1e-6 on the
  geometry `CuH2PotTest` pins. `rgpot::CuH2Pot` and
  `rgpot/CuH2/CuH2Pot.hpp` are gone, and with them the `fortcuh2`
  subproject and its wrap-git entry, so a source build needs no download
  for the Fortran pots and the offline `--wrap-mode=nodownload` CI leg
  covers them. `rgpot/CuH2/cuh2Utils.hpp` keeps the slab-geometry helpers
  unchanged. In-tree sources gate the Fortran pots on
  `RGPOT_HAS_FORTRAN_POTS`, the same cflag `rgpot.pc` exports to
  consumers.

### Fixed

- `potctl`'s lockstep check now covers `pyproject.toml`, so the Python
  package version cannot drift away from meson, CMake, cargo, pixi, and
  towncrier again. It had: 2.5.4 went to PyPI as a multi-ABI wheel set
  (torch 2.7 through 2.13) while every other surface stayed at 2.5.3. All
  six now read the same version, and the check fails the build when they
  do not. ([#52](https://github.com/OmniPotentRPC/rgpot/issues/52))
- Four link-layout defects in the umbrella build, each of which only
  surfaced in a profile that combines the RPC stack with the Fortran
  potentials:

  - Every convenience archive carried its own copy of the vendored vesin
    translation unit, because meson copies a static library's objects into
    each static library that links it, so the umbrella saw multiple
    definitions of vesin's thread-local error state. The archives take a
    headers-only view now and the objects enter `librgpot` -- and each leaf
    binary -- exactly once.
  - `librgpot` linked without the Fortran runtime: it is a C++ target that
    takes the kernels through `link_whole`, which leaves meson no Fortran
    source to infer the runtime from. It is now named per compiler id.
  - `ptlrpc_dep` exported the generated capnp `.cpp` as a dependency source,
    compiling a second copy of that translation unit into every consumer.
    Only the generated header propagates.
  - The umbrella exported the RPC entry points only when something inside
    `librgpot` happened to reference them; they are folded in with
    `link_whole` now, since it is consumers that call them.
- The RPC integration test compares the CuH2 energy and forces at the same
  1e-6 tolerance `CuH2PotTest` uses, instead of asserting exact float
  equality against values from the kernel the Fortran 2018 rewrite replaced.
  Its error path also kills the server before draining its output: reading a
  live server's stderr blocks until that process exits, so an assertion
  failure did not report itself, it sat until the six-hour CI timeout. The
  handler now prints the exception type and traceback as well, since
  `AssertionError` stringifies to nothing.
- The strict-determinism metatomic test no longer requires two matched force
  calls to agree bit for bit, because the provider does not offer that.
  Instrumenting the path shows rgpot handing the model identical inputs --
  positions, cell, and neighbour list all hash the same on every SO3 pass of
  both calls -- and getting energies back that differ by one to two ulp on a
  call chosen at random. It reproduces with `OMP_NUM_THREADS=1`, with the
  TorchScript profiling executor frozen, and with the pair vectors copied
  into torch-owned storage, so it is neither thread-count reassociation, nor
  graph specialization, nor alignment of the buffers rgpot passes in. The
  test now bounds the energy at four ulp and the force components at an
  absolute floor scaled to the largest component, which still fails on any
  drift larger than the provider's own noise.
- The vendored vesin Fortran interface builds under LLVM Flang on Windows.
  It bound `vesin_neighbors` directly, which takes the device and options
  structs by value; Flang cannot lower a by-value `BIND(C)` derived type on
  the `x86_64-pc-windows-msvc` target and aborts the compiler outright
  ("not yet implemented: passing VALUE BIND(C) derived type for this
  target"), so no Fortran consumer of vesin could be built there at all.
  vesin gains `vesin_neighbors_byref`, which takes both structs through
  pointers and forwards them by value, and the Fortran module binds to that.
  Marked as an upstream candidate alongside the other local vesin patches.


## [2.5.3](https://github.com/OmniPotentRPC/rgpot/tree/2.5.3) - 2026-07-20

### Fixed

- Fix ``-Dwith_rpc_client_only=true -Dwith_tests=true``: use ``pot_bridge_dep``
  (was undefined ``rgpot_bridge_dep``), link ``units.cc`` into unit tests when
  ``rgpot_core`` is not built, and soft-skip bridge stress cases when potserv is
  not running (lazy client connect). Catch2 configure/link/test succeed for the
  client frontend product without a live server.


## [2.5.2](https://github.com/OmniPotentRPC/rgpot/tree/2.5.2) - 2026-07-18

### Fixed

- Rebuild Metatomic engines against vesin 0.6+ ``VesinOptions`` (``skin`` /
  ``n_threads``) so ``vesin_neighbors`` no longer fails with a bare error when
  runtime libvesin is 0.6 and the engine was built for 0.5.


## [2.5.0](https://github.com/OmniPotentRPC/rgpot/tree/2.5.0) - 2026-07-17

### Added

- Multi-language Codecov coverage (Rust, C++, Python, Fortran) with OIDC uploads. ([#47](https://github.com/OmniPotentRPC/rgpot/issues/47))
- Install headers and ``rgpot.pc`` (``nwchempot`` / ``cpmdpot`` / ``ptlrpc``; no
  torch or xTB at link time) so eOn and other hosts can prefer
  ``dependency('rgpot')`` over the Meson subproject wrap. Engines stay runtime
  dlopen. See ``docs/eon_pkgconfig.md``.
- Keep Metatomic **dlopen** product on pip: portable `libmetatomic_engine.so` plugin (no eonclib) + `evaluate_metatomic` frontend (not LJ-only wheels).
- Metatomic dual path: linked ``MetatomicPot`` (fast) and ``MetatomicDlopen``
  frontend plus optional ``libmetatomic_engine.so`` C ABI (slow/plugin path).
- Metatomic engines are built for each supported torch major
  (``rgpot/lib/torch-X.Y/libmetatomic_engine.so``) and selected at runtime from
  the installed torch version — same multi-ABI model as metatomic-torch itself.
  The pip product covers **torch 2.7 and newer**; earlier majors are out of scope.
- MetatomicConfig gains an explicit ``torch_determinism`` policy
  (``TorchDeterminismPolicy::Fast`` default, ``Strict`` opt-in). Strict mode
  enables deterministic LibTorch algorithms, math-only scaled-dot-product
  attention (flash / memory-efficient / cuDNN SDP disabled), deterministic
  cuDNN with benchmarking off, deterministic fill of uninitialized memory, and
  disables TF32 for cuBLAS and cuDNN. These flags are process-global via
  ``at::globalContext()``; Fast never mutates them. CUDA hosts still need
  ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (or ``:16:8``) before the first cuBLAS
  call for bit-stable matmuls.
- Portable Metatomic engine wheels: `$ORIGIN` RUNPATH to site-packages torch/metatomic
  (single build-time torch major; multi-ABI package dirs cannot be listed oldest-first).
  `scripts/rgpot_build_wheel.sh` always runs RPATH repair after `python -m build`.
- Python bindings use **nanobind** with **stable ABI** (abi3 / Py_LIMITED_API 3.12)
  when built on Python >= 3.12 (same policy as pyeonclient). Metatomic engines are
  packed multi-ABI under ``rgpot/lib/torch-X.Y/`` and selected from the installed
  torch major at runtime. Supported libtorch majors start at **2.7** (engines for
  2.7–2.13 ship in the manylinux wheel); torch 2.6 and older are not bundled.
- Python package `rgpot` is pip-installable: core Lennard-Jones bindings via
  meson-python wheels (`import rgpot; rgpot.evaluate_lj(...)`), plus optional
  Metatomic multi-ABI engines for **torch 2.7+**.
- xTB dual backends: keep linked ``XTBPot`` and add ``XTBDlopen`` +
  ``libxtb_engine.so`` C ABI plugin (same pattern as metatomic engine).

### Fixed

- Metatomic C ABI engines soft-release the GIL so pyeonclient Job.run can evaluate forces under torch autograd without a fat metatomic link.
- SoftGilRelease in the metatomic C ABI only calls PyEval_SaveThread when PyGILState_Check is true, so nested GIL release from pyeonclient Job.run is safe.


## [2.2.1](https://github.com/OmniPotentRPC/rgpot/tree/2.2.1) - 2026-07-06

### Fixed

- Windows/MSVC builds no longer fail on unused ``cxxabi.h`` includes or nested
  ``std::array`` box flattening in ``Potential`` (``C1083`` / ``C2676``).


## [2.2.0](https://github.com/OmniPotentRPC/rgpot/tree/2.2.0) - 2026-07-05

### Added

- A profile-driven ABI loader (`rgpot::abi::ProfileLoader`) resolves the minimum
  potential ABI available from a single install prefix.

### Changed

- The canonical potentials-schema contract is pinned at v1.13.0, adding the
  Capabilities discovery surface; the in-tree `Potentials.capnp` copies track it
  byte-for-byte.

### Fixed

- `rgpot-core` publishes to crates.io again: `dlpk` (0.1.5) and `eindir-core`
  (0.5.0, `capi`) now resolve from the registry instead of git pins, and the
  `publish = false` guard is gone.


## [2.1.0](https://github.com/OmniPotentRPC/rgpot/tree/2.1.0) - 2026-07-03

### Added

- eOn integration: eOn ships an in-process `RgpotPot` potential (`-Dwith_rgpot=true`) that consumes rgpot's `NWChemPot` / `CPMDPot` frontends as a Meson subproject and `dlopen`s `libnwchemc` / `libcpmdc` directly, with `potserv` remaining available for out-of-process RPC. See `docs/orgmode/howto/eon-rgpot.org`.

### Changed

- The Cap'n Proto schema is pinned to canonical [potentials-schema](https://github.com/OmniPotentRPC/potentials-schema) v1.12.0: `MetatomicParams` arm (union ordinal 4) with the upstream requested-outputs surface, typed NWChem and CPMD parity batches (dplot/esp, prop/linres/pimd/path/tddft), `CommonMethodSpec` overlay, and a schema-sync CI gate that fails when the vendored copies diverge from the pinned release.

### Fixed

- Release-prepare CI no longer runs `cargo publish --dry-run` for `rgpot-core` while it depends on git-only `dlpk` / `eindir-core` (not on crates.io). The job uses `cargo check -p rgpot-core --locked`, and `package.publish` is `false` until registry deps exist. ([#42](https://github.com/OmniPotentRPC/rgpot/issues/42))


## [2.0.0](https://github.com/OmniPotentRPC/rgpot/tree/2.0.0) - 2026-06-26

> Channel note: `v2.0.0` was prepared on `main` but never tagged or published;
> the content below first ships in `v2.1.0`.

### Added

- NWChemPot backend: stable message-based C ABI (`nwchem_c_abi.h`), always-built frontend with `dlopen` of optional `libnwchemc`, stub ABI for CI without NWChem, Cap'n Proto `NWChemParams`/`configure @1`, and `potserv ... NWChem`.
- CPMDPot backend: always-built frontend with `dlopen` of optional `libcpmdc` (split [`cpmdc`](https://github.com/OmniPotentRPC/cpmdc) engine), Cap'n Proto `CPMDParams` / `PotentialConfig.cpmd` / `configure`, structured `CPMDInputSection` arms, in-tree `cpmdc_fake_engine` for CI without CPMD, and `potserv ... CPMD`. Engine lookup: explicit `enginePath`, then `CPMDC_LIBRARY`, `RGPOT_CPMDC_ENGINE`, `RGPOT_CPMD_ENGINE`, then `libcpmdc.so` and the other default names.
- rgpot potentials are now eindir objectives: `rgpot_potential_t` embeds eindir's `eindir_objective_t` as its first member (zero-cost IS-A), with the embedded eval/grad callbacks routed through the rgpot force callback (gradient = -force). rgpot-core consumes `eindir-core` as a shared Cargo crate rather than a prebuilt static lib, so downstream Rust consumers (e.g. `anneal-core`) can minimize an rgpot potential through `eindir_core::Objective<f64>` without a two-Rust-runtime conflict. See `docs/orgmode/howto/eindir-anneal.org`.

### Developer

- CI authoring uses a Nickel `ci/gha/` library with a single hand-maintained
  `ci-orchestrator.yml` for PR/main jobs (prepare/plan, hygiene, build/rust/bridge,
  potentials, CI gate), and nickel-exported workflows only for release, docs,
  cosmo potctl, and doc-commenter. Removed redundant `build`/`prek`/`potentials`/
  `docs_quality`/`towncrier`/`cosmo-potctl-spike` committed workflows in favor of
  the orchestrator plus `gen-gha`/`gha-drift` for the remaining exports. ([#40](https://github.com/OmniPotentRPC/rgpot/issues/40))

### Changed

- NWChemPot is now a pure consumer of the split [`nwchemc`](https://github.com/OmniPotentRPC/nwchemc) engine: the in-tree NWChem embed (`nwchem_c_abi.c`, `nwchem_embed_c_api.f90`, `nwchem_embed_legacy.F`) and the `-Dwith_nwchem`/`-Dnwchem_root`/`-Dnwchem_target` build options are removed. The frontend always builds and `dlopen`s `libnwchemc.so`; the capnp schema is synced byte-for-byte to nwchemc's canonical superset so a flat `NWChemParams` round-trips with no field loss.


## [1.2.0](https://github.com/OmniPotentRPC/rgpot/tree/1.2.0) - 2026-06-24

### Added

- XTBPot: GFN tight-binding via the xtb C API (GFNFF, GFN0/1/2-xTB). Feature-gated with ``-Dwith_xtb=true``. RPC names ``XTB``, ``GFNFF``, ``GFN0xTB``, ``GFN1xTB``. ([#35](https://github.com/OmniPotentRPC/rgpot/issues/35))
- TBLitePot: GFN tight-binding via the tblite C API (GFN1, GFN2, IPEA1). Feature-gated with ``-Dwith_tblite=true``. ([#36](https://github.com/OmniPotentRPC/rgpot/issues/36))
- MetatomicPot: load metatomic TorchScript models directly in C++ (vesin neighbor lists, autograd forces). Feature-gated with ``-Dwith_metatomic=true``. Requires vesin 0.5+ (``VesinDevice{VesinCPU, 0}``). ([#37](https://github.com/OmniPotentRPC/rgpot/issues/37))
- Units module (``rgpot/units.hpp``): CODATA 2018 constants plus a runtime unit expression parser, with Cap'n Proto ``lengthUnit``/``energyUnit`` negotiation on the RPC boundary. ([#38](https://github.com/OmniPotentRPC/rgpot/issues/38))

### Developer

- Lockstep monorepo release via cocogitto 7 + towncrier + ``potctl`` (``potctl/``,
  not published): ``cog bump`` runs ``potctl release sync`` then towncrier then
  ``potctl release assert --require-changelog``; ``release.yml`` publishes
  ``rgpot-core`` on stable ``v*`` tags. CI ``potentials.yml`` builds xtb/tblite and
  metatomic/vesin backends.
- Release tooling in ``potctl`` (Rust workspace crate): lockstep assert
  (meson/CMake/cargo/towncrier/pixi), fail-fast CHANGELOG gate, ``release sync``
  writes all surfaces, cargo publish --locked only, RC tags skip crates.io,
  release-prepare cargo dry-run, SECURITY.md and CODEOWNERS for release surfaces.

### Fixed

- MetatomicPot: target vesin 0.5+ ``VesinDevice`` struct (``{VesinDeviceKind, device_id}``, pass ``VesinDevice{VesinCPU, 0}``) and zero-initialized ``VesinOptions`` (``sorted`` / ``algorithm``).


## [1.1.0](https://github.com/OmniPotentRPC/rgpot/tree/1.1.0) - 2026-03-29

### Added

- XTBPot: GFN tight-binding potential via xtb (GFNFF, GFN0, GFN1, GFN2). Feature-gated with ``-Dwith_xtb=true``. ([#35](https://github.com/OmniPotentRPC/rgpot/issues/35))
- TBLitePot: GFN tight-binding potential via tblite (GFN1, GFN2, IPEA1). Feature-gated with ``-Dwith_tblite=true``. ([#36](https://github.com/OmniPotentRPC/rgpot/issues/36))
- MetatomicPot: ML atomistic models via metatomic/PyTorch with autograd forces and vesin neighbor lists. Feature-gated with ``-Dwith_metatomic=true``. ([#37](https://github.com/OmniPotentRPC/rgpot/issues/37))
- Shared unit conversion header (``rgpot/units.hpp``) with CODATA 2018 physical constants for Angstrom/Bohr, Hartree/eV, and Boltzmann conversions. ([#38](https://github.com/OmniPotentRPC/rgpot/issues/38))


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
