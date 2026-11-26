# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [3.8.0] - 2026-11-26

[3.8.0]: https://github.com/OmniPotentRPC/rgpot/tree/v3.8.0

### Breaking

- `Potentials.capnp` is now rgconf's `potential.capnp`, vendored byte for byte at the commit `CppCore/rgpot/rpc/schema.pin` names, with the `common.capnp` it imports beside it in both `CppCore/rgpot/rpc` and `rgpot-core/schema`. The `potentials-schema` wrap and the local diff series are gone: rgconf declares every member rgpot compiled from them at the same ordinal, so rgpot's own ordinals are unchanged and a message an earlier rgpot wrote decodes the same. The generated `Potentials.capnp.h` now includes `common.capnp.h`, which meson and CMake generate and install next to it in `include/rgpot/rpc`; a consumer that compiles the schema itself needs `common.capnp` beside `Potentials.capnp`, and `rgpot-core` gains a `common_capnp` module. The schema adds rgconf's `gpr` (27), `potlib` (28), `external` (29), `plugin` (30) and `orca` (31) arms, `PotentialSpec.orca` (7), `ORCAParams`, and the shared `PotentialConfig` fields `commonPatch` to `emtRasmussenParameters` (22 to 26). rgconf renumbered its own members to make room: its `gpr`, `potlib`, `external` and `plugin` arms moved from 18 to 21 to 27 to 30, and its `ForceInput` `hessianDisplacement`, `wantStress` and `wantHessianVector` from 10 to 12 to 12 to 14. A peer built against rgconf 0.1.0's numbering does not interoperate with this schema for those members; peers built against earlier rgpot or potentials-schema numbering do.

### Added

- A build option `with_uma` builds the UMA engine plugin against torch alone, without the metatensor and metatomic packages that `with_metatomic` needs. Torch is then a Python package or the C++ distribution: `RGPOT_TORCH_ROOT` may name a libtorch tree (`lib`, `include` and a `build-version` file), and the plugin records that library directory as its run path.
- A how-to, "Add dispersion to a DFT engine", walks through running a DFT engine such as cpmdc BLYP with its own dispersion off beside a `d2` or `d4` composite term from Python, `potserv` and `libcomposite_engine`, and through correcting existing CPMD BLYP-D2 data with `d2cpmd` at weight -1.
- A potential handle answers validated Taylor jets along a line: `rgpot_potential_taylor_jet` writes enclosures `lo_k <= D^k V(x)[v^k] / k! <= hi_k` from a jet registered with `rgpot_potential_set_taylor_jet` (`TaylorJet` and `serve_taylor_jet` on the Rust side).
  A handle without a jet answers the new status `RGPOT_NO_JET` (6), and a jet that meets an undecided branch or leaves a function's domain answers `RGPOT_JET_BRANCH` (7) or `RGPOT_JET_DOMAIN` (8); nothing is differenced in their place.
  The ABI minor is 4.
- Add a lammpc-backed constructor path for `LammpsPot` so rgpot can evaluate full
  LAMMPS `pair_style` setups from `PotentialConfig.lammps` (`LammpsParams`), while
  preserving the existing LJ plugin constructor.
  The Python `rgpot_lammps_potential()` binding now has an overload that accepts
  flat `PotentialConfig` bytes, and the external C ABI adds
  `rgpot_lammpc_potential()`.
  The lammpc backend passes the same `Capabilities` handshake as the other
  profile backends and is refused, naming its path, on a mismatch. `LammpsPot`
  reports stress only when that backend declares the `stress` operation, and its
  result-cache key covers the resolved backend file and the configuration in
  both modes.
- Add an engine-agnostic `rgpot_md_session_*` C ABI and `rgpot_core::md_session`
  contract in Angstrom/eV/ps/amu units, backed by an in-process OpenMM session
  with `3N` hyperplane BAOAB, specular planar walls, deterministic 64-bit seeding,
  and unconstrained Verlet NVE. `CAbiMdSession::from_engine_ops` adopts the
  operation table an engine's create call writes: a null required callback is
  refused with `MdSessionErrorKind::MissingOperation`, and a refused table has
  its context released through `destroy`.
- Add the optional `rgpot_engine_taylor_jet` entry point, the
  `RgpotEngineDerivativeReport` roundoff report and the trailing
  `RgpotEngineCaps.jet_order` capability to the generic engine C ABI. One call
  serves a uniform batch of systems and returns the exact first `jet_order`
  directional derivatives of the gradient along a line for each, the first block
  being the Hessian-vector product, with the summed operation count and largest
  scale as a roundoff estimate. An engine without the jet reports
  `RGPOT_ENGINE_ERR_UNSUPPORTED`, and the Rust loader maps that, a missing symbol
  or a lower `jet_order` to `EngineError::Unsupported` through
  `EngineSession::taylor_jet` and `EnginePotential::taylor_jet`. The loader asks
  for the capabilities once per instance. A potential serves the jet by deriving
  `rgpot::GradientJet`.
- Add the public Rust `EngineSession` host with typed engine status, owned
  callbacks, model-state and hint support, serialized coordinate transforms, and
  batched force evaluation. Extend public `EngineSymbols` with the optional
  `model_state`, `set_callbacks`, and `hint` entry points.
- Added `rgpot.EAMTablePot(file, format, periodic=True, stress=False, skin=1.0)`
  and `rgpot.has_eam_table`. The class reads a funcfl, setfl or fs table, has a
  `cutoff` property, and works with `RgpotCalculator("eam_table", ...)`. The
  wheel build now enables `with_eam_table`.
- Added `rgpot::PotentialHandle::eam_table(path, format, periodic)`, the C++
  face of `rgpot_potential_new_eam_table`. It throws `rgpot::Error` where the C
  constructor returns NULL.
- Added `rgpot_potential_new_eam_table(path, format, periodic)` to the C ABI
  of `rgpot-core`, with `rgpot_core::eam_table::EamTablePotential` as its safe
  Rust wrapper and `rgpot::PotentialHandle::eam_table` for C++. A host reads a
  LAMMPS funcfl, setfl or Finnis-Sinclair table in process, with no C++ kernel
  and no engine plugin. The evaluation matches LAMMPS `pair_style eam` on
  `Cu_u3.eam` for a copper cell with a vacancy to 3e-12 eV and 3e-14 eV per
  Angstrom.
- Added a GROMACS nblib engine for the generic MD session ABI, built with
  =-Dwith_nblib=enabled=. nblib supplies the forces of a topology given as text;
  Langevin dynamics on a hyperplane, planar walls, the constraint force per step
  and the seeded random stream are the driver the OpenMM engine runs, so both
  engines answer the same calls with the same noise. =nblib_prefix= names a
  GROMACS install, and =engines/nblib/build_gromacs.sh= builds the pinned release
  with the nblib API.
- Added optional MD-session neighbour exchange slots to `rgpot_md_session_ops`
  using the shared `rgpot_nlist_request` / `rgpot_nlist_soa` shape, with clear
  unsupported status reporting. In Rust, `MdSession::get_neighbors` and
  `MdSession::set_neighbors` take `MdNlistBuffers` and `MdNlistPairs`, which
  borrow one slice per column and check their lengths against the pair count and
  the request before the engine is called.

  The OpenMM session now serves neighbour queries from the current Context
  geometry through the shared rgpot neighbour builder fallback and reports
  external neighbour injection as unsupported.
- Added the controls a rare-event sampler needs to the generic MD session
  ABI: `rgpot_md_session_run_langevin` returns the Lagrange multiplier of a
  hyperplane per step (the `dF/dc` integrand of PAFI and the force of
  Bennett-Chandler), accepts reflecting or rejecting walls, and keeps its
  seeded random stream across calls; `rgpot_md_session_reseed_velocities` draws
  Maxwell-Boltzmann velocities, with `n . v = 0` on a plane, identically for
  every engine; and the periodic box and the first force are set and read through
  `rgpot_md_session_set_periodic_box`, `rgpot_md_session_get_periodic_box`,
  `rgpot_md_session_initialize`, and the getters for positions, velocities,
  forces and energies. The new callbacks follow `destroy` in the vtable, so an
  engine built against the previous header still loads. `rgpot_core::md_session`
  mirrors them with `MdSession::run_langevin` and the matching
  Maxwell-Boltzmann draw.
- Added user-facing analytic model support end to end: potserv now serves
  `DoubleWell` and `CoupledProtonWell`, clients can call `calculateHvp` over
  RPC for their analytic Hessian-vector products, and the C model API
  documents constructor and evaluation usage.
- An opt-in MDI compatibility frontend now serves every `rgpot_potential_t` to drivers such as LAMMPS `fix mdi/qm`, with atomic-unit wire conversion, shared energy/force evaluations, and TCP low-latency mode. Direct in-process calls and Cap'n Proto RPC remain the default paths. Meson `-Dwith_mdi=true` and CMake `-DRGPOT_WITH_MDI=ON` build it, TCP is refused with an error where POSIX sockets are missing, and `rgpot_mdi_server_start` is exported by every build, so the C ABI minor of 9 is true of each library; a build without the `mdi` feature returns `RGPOT_RPC_ERROR`.
- CPMD engine and RPC callers can select a stable orbital key per image or bead, including through the Python RPC calculator.
- Engine plugins expose optional C calls to attach an rgpot result cache and report how many geometry results the kernel computed and how many the cache served.
- Load the optional `rgpot_engine_hvp` and `analytic_hvp` capability through
  `rgpot_core::engine::EnginePotential`. Engine-backed potentials use the
  existing `Capabilities.hessianVector` and `Potential.calculateHvp` RPC path;
  an engine without the analytic operation returns
  `RGPOT_ENGINE_ERR_UNSUPPORTED` and never falls back to finite differences, and
  overlapping `direction` and `hv_out` buffers are rejected with
  `RGPOT_ENGINE_ERR_INVALID`.
- Meson consumers can take `rgpot_dep_no_openmp` in place of `rgpot_dep` to compile and link against rgpot without the OpenMP flag. librgpot is still built with OpenMP and keeps its thread teams: a shared librgpot records the runtime as `NEEDED`, and a static one now links the runtime library by name, so its archives no longer put `-fopenmp` on the consumer's link line. `rgpot_dep` still passes the flag to its consumers.
- Setting `use_openmp` to false builds the cell walk and the potentials serially. `librgpot` and a consumer of its dependency then get neither the OpenMP flag nor the runtime, and the regions guarded by `_OPENMP` run on one thread.
- The C API of `rgpot-core` computes dense Hessians with
  `rgpot_potential_hessian`, under an explicit `rgpot_hessian_scheme_t`
  (`RGPOT_HESSIAN_ANALYTIC`, `RGPOT_HESSIAN_AD`, `RGPOT_HESSIAN_EXACT` or
  `RGPOT_HESSIAN_CENTRAL`, the last making `6 * n_atoms` force calls), and writes
  the scheme that ran to `scheme_used`. `rgpot_potential_set_hessian_callback`
  and `rgpot_potential_set_ad_hessian_callback` register the analytic and
  dual-number Hessians, and `rgpot_potential_set_ad_hvp_callback` a dual-number
  Hessian-vector product, which `RGPOT_HVP_AD` calls. `RGPOT_HVP_EXACT` takes the
  analytic product, else the dual-number one, and refuses rather than
  differencing. `rgpot_potential_hvp_resolved` reports the scheme an
  `RGPOT_HVP_AUTO` product resolved to. `rgpot_potential_derivative_caps` reports
  which analytic, dual-number and finite-difference schemes a handle has, with
  their force-call costs, and `rgpot_potential_derivative_counts` counts the
  requests each scheme answered.
- The Cap'n Proto ASE calculator now provides remote stress for periodic cells, including cell filters and barostats.
- The Hessian-vector explanation page now covers the dual-number and exact product schemes, how the scheme that answered is reported, the dense Hessian schemes and the capability query, and when to take the dense Hessian instead of products: the break-even `k* = (t_dense - t_hvp)/t_apply + 1`, which is `6N - 1` with finite differences only, and the correction when a multiply by the stored Hessian is not free.
- The Muller-Brown analytic surface now has configurable parameters and C++, C, Rust, Python, RPC, and generic-engine interfaces.
  Each configure call describes the whole surface, and an empty parameter list takes the published default; the PotentialConfig arms reference has the rule.
  A PotentialConfig arm that does not belong on that builtin is refused with an error that names the arm.
  MullerBrownPot::paramsKey follows the six parameter lists, so two surfaces do not share a result-cache entry.
- The Python potentials gain `hessian(positions, atom_types, box, scheme)`, which
  returns the dense `(3N, 3N)` Hessian and the name of the scheme that computed
  it; `scheme` has no default, and `"analytic"`, `"ad"` and `"exact"` are refused
  on a kernel without them while `"central"` makes `6N` force calls.
  `hvp_resolved` returns the product with the scheme that ran, so an `"auto"`
  product that resolved to forward differences reports it, and `hvp` accepts
  `"ad"` and `"exact"`. `derivative_caps()` reports which analytic, dual-number
  and finite-difference schemes a kernel has, with their force-call costs, and
  `derivative_counts()` counts the requests each scheme answered.
- The always-built `libcoupled_well_engine.so` is an analytic engine plugin for one atom on a two-coordinate coupled well, with a quartic barrier along x and a transverse spring shifted by the barrier coordinate. It takes a versioned create message with the atomic number and the barrier energy, well position, transverse spring and coupling, and provides exact energy, forces, batch evaluation and Hessian-vector products. `rgpot_core::engine::CoupledWellConfig` and `PotlibcConfig` are the typed create messages for it and for the potlibc plugin; both are `#[non_exhaustive]`, have `Default` and `with_*` builders, and `validate`. `libpotlibc_engine.so` now evaluates a batch natively.
- The built-in plugin registry now includes classical-morse and sutton-chen, and eam-funcfl and eam-setfl when tabulated EAM is built. Each one reads a flat JSON configuration. An external plugin whose name is already registered is refused, and the existing plugin is kept.
- The default-disabled `with_potlibc` option builds `libpotlibc_engine.so`, which selects fixed-size and variable-size POTLIB surfaces through a versioned engine create message.
  The plugin converts the engine ABI units, accepts species where potlibc metadata marks them unspecified, and exposes analytic forces and Hessian-vector products reported by the selected surface.
  The plugin refuses an atom count whose coordinate buffers would not fit in `size_t`, at create and at every force or HVP call, instead of letting the buffer size wrap.
- The generic engine C ABI adds `RgpotEngineComponent`, `rgpot_engine_last_components` and `rgpot_engine_component_name` for the per-term record of a composite engine, and bumps `RGPOT_ENGINE_ABI_MINOR` to 5.
- The generic engine C ABI adds rgpot_engine_select_orbitals and rgpot_engine_copy_orbitals, bumps RGPOT_ENGINE_ABI_MINOR to 3, and reports orbital capabilities through RgpotEngineCaps.
- The generic engine C ABI adds the optional `rgpot_engine_unload_policy` export with `RGPOT_ENGINE_UNLOAD_RESIDENT` and `RGPOT_ENGINE_UNLOAD_SAFE`. This is an additive change to the engine ABI, so `RGPOT_ENGINE_ABI_MINOR` and `ENGINE_ABI_MINOR` are now 7. No existing layout or call changes, a host of minor 7 still loads plugins of minor 0 to 6, and a host older than minor 7 refuses a plugin that reports minor 7. A plugin of minor 7 or later that returns `RGPOT_ENGINE_UNLOAD_SAFE` can be unloaded with the new `unsafe` `EnginePotential::unload(path)` once no instance uses it, and the unload closes the library after it releases the registry lock. Every other plugin stays loaded, and `unload` refuses it with `EngineError::Unsupported`. The C++ library soversion stays 9.
- The in-process and RPC ASE calculators are discoverable as `rgpot` and `rgpot_rpc` through ASE's calculator registry, and installed wheels expose lazy ASE 3.29 calculator plugins without breaking ASE 3.23–3.28.
- The nblib session engine opens a GROMACS run input file. The spec `tpr PATH`
  evaluates the nbnxm, PME, listed and position-restraint forces of the file in
  its triclinic cell, spreads the forces of virtual sites, exposes the massive
  atoms, and holds the SETTLE and LINCS distances in every run. Features of the
  file it does not evaluate are refused by name. Build with `-Dnblib_source` and
  `-Dnblib_build` naming the GROMACS trees of `nblib_prefix`.
- `CBEPot` now evaluates the CH5 2008 surface through potlibc's `ch52008` instead
  of the external cbe-pes library, and serves the engine gradient jet to order 3
  through `potlibc_gradient_jet`, in one call for a whole batch, with the
  forward-mode operation count and scale of the call, so the `cbe` backend of the
  classical engine answers `rgpot_engine_taylor_jet` with `jet_order` 3. It also
  evaluates energies and forces for a batch in one `potlibc_evaluate` call and
  reports `batched`. Building with `with_cbe` needs a potlibc that provides
  `potlibc_gradient_jet` and its `potlibc` pkg-config file. Bit goldens of the
  energy, force, Hessian product and jet at the saddle and at a displaced
  geometry pin the engine.
- `D2CpmdConfig` takes a `symmetry` code, the SYMMETRY value of the CPMD run, which selects the branch of `pbc3` that `D2CpmdPot` follows. Code 12 takes the monoclinic branch, which its box cannot show because it has the shape of code 14, and a code that `pbc3` does not handle is refused. `-1` keeps reading the branch from the box. `D2CpmdParams` carries it at ordinal 4, and `rgpot.D2CpmdPot` takes a `symmetry` argument.
- `EngineSession` gains `attach_cache`, `cache_counts` and `clear_cache`, so session users can attach a result cache and read its counts. On Windows the loader refuses a cache path that is not valid Unicode, because the engine opens it as UTF-8.
- `ExprPot` and `SumPot` have a read-only `cutoff` property: the largest cutoff
  among their terms, in Angstrom.
- `LJPot` has a dense analytic Hessian, `hessianImpl`, and a dense Hessian and
  Hessian-vector product from dual numbers, `hessianAdImpl` and
  `hessianVectorAdImpl`, and advertises them as `analyticHessian`,
  `adHessian` and `adHvp`. The dual-number pair term is the templated
  `lj::pairEnergy` of `rgpot/LennardJones/LJPairEnergy.hpp`, the energy the
  force kernel sums, so its second derivative comes from the energy itself
  rather than from the hand-written one. Both Hessians are symmetric to the
  last bit and agree to 1e-12 of their largest entry. Coincident atoms throw
  `std::domain_error`.
- `LJPot` has an analytic Hessian-vector product, the pair second derivatives
  of the shifted or switched 12-6 term contracted with the direction, so
  `HvpScheme::Auto` and `rgpot.LJPot.hvp` make no force calls. Two atoms on
  one point make the product NaN, which `HessianVectorCache` reports as
  `std::domain_error`.
- `MEAMPot` is the modified embedded-atom method of Baskes, Phys. Rev. B 46, 2727 (1992), doi:10.1103/PhysRevB.46.2727, with the second-neighbour form of Lee and Baskes, Phys. Rev. B 62, 8564 (2000), doi:10.1103/PhysRevB.62.8564. It reads the library and parameter files of the LAMMPS `pair_style meam`, handles periodic and triclinic cells with every image inside the cutoff, and writes the energy, the forces and the stress. Energy, forces and virial match LAMMPS to 1e-11 on the reference sets of the test suite.
- `MorsePot` has a dense analytic Hessian, `hessianImpl`, built from the pair
  derivatives of its Hessian-vector product, and advertises it as
  `analyticHessian`. It is symmetric to the last bit. Morse has no
  dual-number path; its `hessianAdImpl` and `hessianVectorAdImpl` throw.
- `ORCAPot` (`PotType::ORCA`) runs ORCA single-point energies and gradients behind the normal `Potential` interface, built with `-Dwith_orca=true` (CMake `RGPOT_WITH_ORCA`). Each call writes an input with `! EnGrad`, runs the `orca` executable (including symlinks to executables) found by `ORCAConfig::executable` or on `PATH` in a working directory of its own with extra file descriptors closed before `exec`, and parses the `.engrad` energy and gradient in the C locale using `units.hpp`. Stale `.engrad` and `.gbw` files are removed before each run, treating any removal failure other than a missing file as `ORCAError`. The converged `.gbw` is passed to the next call through `%moinp` and `! MORead` unless `restart` is turned off, and a call without one starts from ORCA's own guess. The potential is per instance and non-periodic, and a box with a non-zero entry is refused. A failed run throws `ORCAError` with the tail of ORCA's standard error. One run is bounded by `ORCAConfig::timeout_s` (one hour by default; 0 waits with no limit): past that limit the process group is sent SIGTERM and then SIGKILL if any process in the group is still alive two seconds later (even if the leader has exited), the child is reaped, and the call throws `ORCAError`. ORCA is described in Neese, WIREs Comput. Mol. Sci. 2, 73 (2012), doi:10.1002/wcms.81, and Neese, WIREs Comput. Mol. Sci. 12, e1606 (2022), doi:10.1002/wcms.1606. The tests run a fake `orca` script, and a test against a real ORCA skips when `orca` is not on `PATH`.
- `RGPOT_SCC_NOT_CONVERGED` (5) is a new `rgpot_status_t` for an engine whose self-consistent field did not converge, `rgpot_set_last_error` lets a callback store the message that goes with its status, and `rgpot_potential_fused_error_kind` tells an SCC failure from another failure on the eindir path. The ABI minor is 3; no existing call changes.
  `rgpot::SccNotConverged` is the exception a C++ potential throws for it: `XTBPot` and `XTBDlopen` do, `PotentialHandle::from_impl` returns the new status with the message, and `rgpot::Error::status()` carries it. The xtb engine ABI returns `RGPOT_XTB_RC_SCC_NOT_CONVERGED` (-4) from `rgpot_xtb_force`.
  `rgpot_fused_settings_t` with `rgpot_fused_settings_default` and `rgpot_potential_set_fused_settings` install the engine settings that key the fused cache. `XTBPot` gains `setAccuracy`, `setElectronicTemperature`, `setMaxIterations` and `fusedSettings()`.
- `StillingerWeberSi` and `TersoffSiC` can be evaluated on a local region plus ghost atoms. Each declares a full neighbour list and a ghost reach of twice its cutoff. A driver sums the energy of the owned atoms and adds the forces written on ghost atoms back to their owners. Results for a whole cell are unchanged.
- `fortranpots::CuH2Pot` has an analytic Hessian-vector product from the pair,
  density and embedding second derivatives of the copper-hydrogen EAM, through
  the new `rgpot_cuh2_hvp_ws` entry point, so `HvpScheme::Auto` and
  `rgpot.CuH2Pot.hvp` make no force calls. It reports the same species and
  geometry faults as the force.
- `fortranpots::TersoffPot` now advertises and publishes analytic Cauchy stress
  on both dispatch paths of `PotType::TersoffSi`: the all-silicon Fortran gather
  and the mixed-species fallback to `TersoffSiC`. The gather computes `dE/dε`
  analytically in the same Voigt order and sign convention as `stress.hpp`.
- `rgpot-core` loads any library that exports the generic engine C ABI
  (`rgpot/engine_c_abi.h`) from Rust. The `engine` feature adds
  `rgpot_core::engine::EnginePotential`: it opens the library, checks both the
  ABI major and `rgpot_engine_abi_stamp` (major, minor and layout revision),
  creates an instance from the backend's configuration message, and answers
  energy and forces per geometry, or per batch when the library exports
  `rgpot_engine_force_batch`. A positive return code is a recoverable
  `EngineError`, and a negative one poisons the handle and returns a fatal
  `EngineError` with the engine's own message without re-entering native code on
  later calls. `EnginePotential::into_potential` wraps the instance as a
  `PotentialImpl`.
- `rgpot-ipi-driver` serves `CPMD` through the generic i-PI driver, using `--pot "CPMD params <file>"`, with `--preserve-open-cell` for an isolated deck that must keep its own cell, and it rejects an init string that requests a `batch_size` above 1.
- `rgpot.EMTPot` brings the Asap effective-medium potential to Python, with `config`, `caps`, `evaluate`, `evaluate_batch` and the `hvp` family, and `RgpotCalculator("emt")` uses it with a kernel cutoff of 6.855 Angstrom, so a small cell such as cubic copper with `a=3.6` is tiled. Energies and forces agree with ASE's `EMT` when that is built with `asap_cutoff=True`.
- `rgpot.lammps.FixExternal` now runs any Python rgpot potential through LAMMPS `fix external`, including unit conversion, triclinic cells, image unwrapping, analytic virials, deadlock-safe MPI error propagation, and complete callback teardown, without importing LAMMPS or mpi4py until needed.
- `rgpot/Dual.hpp` adds forward-mode dual numbers, `rgpot::ad::Dual<T>`, with
  `Dual2` for second derivatives and `secondDerivative` to read a value and
  its first two derivatives from one evaluation. A kernel written as a
  template over its scalar type is differentiated from its own source, with
  no step and no truncation error.
- `rgpot::D2CpmdPot` reproduces OpenCPMD's built-in empirical D2 (`EMPIRICAL CORRECTION` / `ALL DFT-D2`) term for term: CPMD's tables, constants, damping, 6 R0 truncation and minimum image convention, so a composite can subtract it from existing CPMD BLYP-D2 results and put the periodic `rgpot::D2Pot` in its place. It is configurable through the new `PotentialConfig.d2cpmd` arm (`potserv <port> D2Cpmd`), as a `d2cpmd` composite term, and as `rgpot.D2CpmdPot` and a `"d2cpmd"` kind in Python. Its forces jump when a pair crosses half the cell; `rgpot::D2Pot` does not.
- `rgpot::D2Pot` is Grimme's DFT-D2 (J. Comput. Chem. 27, 1787 (2006)) with the tables and per-functional parameters of the dftd3 program, summed over every periodic image within a cutoff in any cell, sheared or not, with energy, forces and stress. The default cutoff of 150 Angstrom converges the energy of liquid water to below `sqrt(eps)` Hartree per atom. It is configurable through the new `PotentialConfig.d2` arm (`potserv <port> D2`), as a `d2` builtin term of a composite, and as `rgpot.D2Pot` and a `"d2"` term kind in Python.
- `rgpot::D4Pot` takes a `periodic` flag (`D4Config::periodic`, `D4Params.periodic`, `rgpot.D4Pot(periodic=True)`) that passes the box to dftd4 as a 3D lattice and returns the stress from dftd4's virial; the default stays molecular. `potserv`'s D4 backend now applies the `PotentialConfig.d4` arm instead of refusing it, and `d4` is a builtin composite term, so a DFT engine with its own dispersion off can run beside D4 with dftd4's BLYP parameters (`functional = "blyp"`).
- `rgpot::HessianVectorCache::apply` returns the scheme that computed the
  product, so a `HvpScheme::Auto` request that resolved to `Forward` reports it,
  and `derivativeCounts()` counts products by that scheme. `HvpScheme::Ad`
  uses a potential's dual-number product and `HvpScheme::Exact` takes the
  analytic product, else the dual-number one, and refuses rather than
  differencing. `HessianVectorCache::hessian` writes the dense Hessian under an
  explicit `HessianScheme` (`Analytic`, `Ad`, `Exact` or `Central`, the last
  making `6N` force calls) and returns the one that ran. `rgpot::derivativeCaps`
  reports which analytic, dual-number and finite-difference ways a potential
  has, with their force-call costs.
- `rgpot::PotentialHandle::hessian` computes the dense Hessian through the Rust
  core under an explicit scheme and returns the one that ran, and
  `PotentialHandle::hvp` now returns the scheme that computed the product, so an
  `RGPOT_HVP_AUTO` request that resolved to forward differences reports it.
  `from_impl` registers the analytic and dual-number Hessian and Hessian-vector
  hooks of a potential from its `caps()` bits, and `derivative_caps` and
  `derivative_counts` report what the handle can do and has done.
- `rgpot::SumPot` is a weighted sum of child potentials with a name, a weight (sign allowed) and an on/off switch per term, a per-term record of energies, forces and stress, and logging through `RGPOT_SUM_LOG` or a sink. It is configurable through the new `PotentialConfig.composite` arm (`potserv <port> Composite`), `libcomposite_engine` and `rgpot.SumPot` in Python; `rgpot::lr::SumPot` now builds on it.
- `rgpot::makePotential` (`rgpot/PotentialFactory.hpp`) builds the potential a `PotentialConfig` names, from a reader or from the flat-array bytes, and returns it as `std::unique_ptr<rgpot::PotentialBase>`. The in-process arms use the backends compiled into the build. The `orca` arm builds the new `rgpot::ORCACPot`, a frontend of the split orcac shim on the pattern of `MOPACPot`: it dlopens `liborcac` (`ORCAC_LIBRARY`, `RGPOT_ORCAC_ENGINE`, then `liborcac.so.1` and `liborcac.so`), checks that it exports the thirteen functions of orcac C ABI 1 and reports ABI 1, and creates one session from an `OrcaCConfig` lowered from `ORCAParams`. The `metatomic`, `uma` and `plugin` arms, and `xtb` in a build without `-Dwith_xtb`, load an engine library over the generic engine C ABI as an `rgpot::EnginePot`; `plugin` opens `libraryPath`, or `lib<pluginName>_engine` through `RGPOT_<PLUGINNAME>_ENGINE`, with `opaquePayload` as the configuration. An arm the build cannot construct throws `std::invalid_argument` naming the arm and the option that enables it.
- `rgpot::nlist::Builder::frames()` counts the times an update folded the positions into the cell. Each accepted update folds once, whether it keeps the table or searches again, so `frames()` equals the number of accepted updates and `builds()` the number of searches among them. The state stays behind the builder's pointer, so the layout of `Builder` and the soversion of `librgpot` do not change.
- `rgpot_ch5_2008_taylor_jet` (`rgpot_core::ch5_2008::Ch52008` in Rust) is the exact Taylor jet of the CH5 2008 surface of Corchado, Bravo and Espinosa-Garcia, in Bohr and Hartree, for registration with `rgpot_potential_set_taylor_jet` on a handle that evaluates that surface.
  Its saturation guards, the `rch < 3.8` switch and the handedness swap are decided on enclosures; an undecided one is `RGPOT_JET_BRANCH`.
- `rgpot_core::taylor::Taylor::from_coefficients` builds a truncated series from supplied interval coefficients and refuses an empty list with `JetError::Shape`.
  The `TaylorCoefficient` trait names the coefficient operations that the exponential recurrence needs, and `TaylorExponential<C>` evaluates `exp(u)` one coefficient at a time in the Griewank-Walther order.
  `Taylor::exp` now runs through that recurrence and returns the same enclosures as before.
- `rgpot_core::taylor` carries truncated Taylor series with outward-rounded interval coefficients (`maryada`): sums, products, quotients, `sqrt`, `exp`, `tanh`, `acos` and integer powers, with branch tests decided on enclosures.
  A potential whose energy is an explicit formula evaluates it on `Taylor::line` coordinates to implement `TaylorJet`; a quotient, root or arc cosine at the edge of its domain is `JetError::Domain`, and an undecided comparison is `JetError::Branch`.
- `rgpot_md_langevin_result` ends with `constraint_metric`, the plane metric of
  every step, and `MdLangevinOutcome::constraint_metric` returns it. Callers and
  engines built against the shorter struct keep working.
- librgpot now exports unmangled C calculator-group entry points rgpot_mpi_host_abi_version, rgpot_mpi_bind_calculators, and rgpot_mpi_share_bytes for hosts that dlopen symbols.

### Developer

- Added ASE `Calculator` and `BaseCalculator` contract conformance tests (`calculation_required`, `calculate_properties`, `export_properties`, `reset`, `atoms`/`results` copy isolation, `todict` JSON round-trip, `PropertyNotImplementedError`, `ase.calculators.test.gradient_test`, `BFGS`, `FrechetCellFilter`, and `CppCore/rgpot/rpc/ase_helpers.py` / `ase_calculator.py` shim coverage). `RgpotCalculator` already conforms on `develop`.
- Added a Linux CI leg that builds with OpenMP disabled and runs `LongRangeTest` and `consumer_no_openmp`, keeping the serial long-range path and the OpenMP-free consumer link under coverage.
- Added a regression test that verifies external potential constructors do not
  leave heap allocations behind when backend factory setup fails.
- Build and test workflows skip documentation-only changes. A superseded run is cancelled except on main and tags. Wheels, thread sanitizer, address and leak sanitizer, and the macOS ISA audit run on main, tags, a manual dispatch, or a pull request labeled ci:full.
- CBE bit-exact engine goldens are compared on x86-64 with glibc. Other hosts keep the tolerance checks.
- Consolidated `DynLib.hpp`, candidate-path search, and `ProfileLoader`
  resolution across `CP2KPot`, `CPMDPot`, `MOPACPot`, `NWChemPot`,
  `XTBDlopen`, and `MetatomicDlopen`, sharing Cap'n Proto serialization and
  communicator-adopter registration helpers in `CppCore/rgpot/abi/`.
- Conventional commit checks accept the `dev` type. `potctl ci cog-check`
  now matches commit types case-insensitively, as cog does, so `BLD:`
  subjects pass.
- Deduplicated `energy_gradient`, LJ-like bindings, and GROMACS/OpenMM
  wrappers in `python/bind/bind_module.cpp`, and extracted shared geometry,
  cell, and OpenMP thread-count fixtures into `CppCore/tests/TestFixtures.hpp`.
- Deduplicated three-body inner loops and neighbour-cache synchronization in
  `SWSi.cc` and `TersoffSiC.cc` via `ThreeBodyCommon.hpp`, shared cell-team
  pair walks, free-box fallbacks, and the 12-6 Lennard-Jones pair kernel across
  `LJPot`, `LJClusterPot`, `MorsePot`, `ZBLPot`, `SuttonChenPot`, and
  `EAMTablePot` via `PairEval.hpp` and `LJPair.hpp`, and unified orthorhombic
  minimum-image and Morse/density evaluation in `EAM.cpp`.
- Enforced binary-only SciPy resolution on free-threaded Python 3.13 in the reduced build helper, resolved rustdoc intra-doc link warnings in `potctl` and `rgpot-core`, and avoided `SIGPIPE` failures during wheel symbol-table inspection.
- Engine plugins export the C ABI only. A coverage build no longer publishes
  the GCC gcov runtime from those libraries.
- Engine, tight-binding and Fortran kernel pages now cite the method papers, and the libxckernel note cites its journal article.
- Give the neighbour-list test binary an explicit 120s timeout and trim the dense field-invariance test boxes so coverage runs stay well within that budget even under parallel suite contention.
- Hessian-vector tests compare the LJ product with a closed-form 12-6 Hessian and a force finite difference, the CuH2 product with wrapped periodic pairs, and the RPC unit smoke with the energy gradient and curvature.
- Install `llvm` during free-threaded CPython bootstrap in `py-tsan.yml` and ensure `llvm-symbolizer` is on `PATH` so ThreadSanitizer symbolizes frames and applies upstream and Python 3.13 suppressions during parallel pytest runs.
- NWChem configuration message tests now distinguish a loaded engine refusal from a missing engine.
- Refines the Tersoff stress tests by clarifying the thread-setting check
  as a determinism smoke test and deduplicating Voigt index constants.
- Release docs describe the fragment checks and which pull requests skip them.
- Rust sources now pass the repository formatting and clippy checks.
- Sutton-Chen workspace tests now assert serial behavior when OpenMP is disabled.
- The C++ coverage build enables `-fprofile-update=atomic` for concurrent OpenMP gcov counters, and the capture script disables synthetic exception-unwind branches under lcov 2 so capture and extract run without error suppressions.
- The EMT leak-accounting test now pairs each counted deallocation with its
  allocation, so it stays valid when MinGW routes a delete across runtime modules.
  The pairing uses a hashed pointer table, so the test keeps its earlier runtime.
- The EMT reference now states which failures keep the cached kernel and which drop it. A call that fails before the kernel runs, and a non-finite energy or force, keep a kernel from an earlier successful call. An Asap failure and any other exception raised by the kernel drop it.
- The Fortran coverage job uses the same test timeout multiplier as the C++ coverage job. NeighbourLayerTest gets one extra multiple of its registry timeout in that build, because four multiples left it about forty seconds inside an eight minute limit.
- The FortranStaticAudit test fails when a Fortran kernel archive holds
  procedure-local static data, which every thread would share. gfortran
  emits such a static for the hidden length of a deferred-length function
  result inside a concatenation, even under -frecursive. That static
  truncated Tersoff error messages and corrupted the heap when several
  free-threaded Python callers failed at once. New native and Python tests
  run the Tersoff stress path from many threads at once, interleaved with
  simultaneous zero-volume failures, and require results identical to one
  thread.
- The Gupta and reduced pair reference now records the `!(re > 0.0)` Morse checks and the error conditions of each C entry point.
- The Gupta, reduced-pair and Sutton-Chen C entry points now have reference entries, so the documented signatures stay aligned with the headers.
- The LeakSanitizer lane runs `llvm-symbolizer` through `tools/lsan/llvm-symbolizer`, which clears `LD_LIBRARY_PATH` so the system binary does not load pixi's libraries. The libomp suppression matches function names, so a symbolizer that cannot start left libomp's 128-byte allocation reported as a bare `libomp.so` offset. `tools/check_lsan_libomp.sh` now fails if frames are not symbolized, and leaks 137 bytes from `main` and 53 bytes from inside an OpenMP region on purpose; it requires LeakSanitizer to report those two and no libomp allocation.
- The Meson build options page lists `with_uma`, which builds the UMA engine plugin against torch alone.
- The NWChem reference now says a non-empty engine path that names another library replaces the loaded one.
  The Skala reference now says a non-empty engine path is forwarded whenever it is set.
- The Python typing job installs mpi4py, which ships its own type hints, so `rgpot.lammps` is checked by mypy and pyright against the real package.
- The `engine_abi_stamps` test now opens `libpotlibc_engine` when `with_potlibc` is enabled, and the new `potlibc_engine_rust_loader` test loads it with `EnginePotential::load`, evaluates a surface, and drives the lock-failure build to check that `EngineError::Lock` latches the handle.
- The engine ABI reference wraps the shared export list and the Rust symbol table on semantic lines.
- The engine config size checks compare against the public minimum message sizes,
  so the lint run with warnings denied passes at the minimum supported Rust
  version with the engine feature on.
- The engine loader tests now check that a refused load destroys every engine instance it created and creates none when it is refused before creation. The cases are an engine that reports itself unavailable, another ABI major, a newer minor, another layout revision, and a lock or fatal code from `rgpot_engine_caps`. The failed-first-load test also reopens the unavailable, newer minor and caps failure cases.
- The manual index includes the design notes and the documentation style guide. `doccheck` fails when an org page or a pinned citation drops out of the manual.
- The neighbour-layer tests now cover orthorhombic periodic coordinates outside the primary box, verifying both Stream and Stored neighbour modes against the direct minimum-image reference on the boundary pair and randomized multi-wrap trials.
- The potential and backend pages now name the parameter source and a checked reference value for Lennard-Jones, Morse, ZBL, EAM, EMT, water, D3, D4, xtb, metatomic and CPMD.
- The potlibc engine howto now pins the potlibc checkout used for plugin regression tests to the main revision that makes the LSTH Hessian symmetric.
- The release guide wraps its fragment rules to the documentation line limit, so the org formatting check passes again.
- The release-notes template now drops copyright lines and comment blocks from fragments and writes the bullet itself, so a fragment that starts with `- ` no longer renders as a doubled bullet. Existing fragments are rewritten to match, and `scripts/check_news_fragments.py` (`pixi r news-fragments-check`) checks the rule.
- The root `meson.build` matches the output of the pinned Meson formatter, so the meson-format hook no longer rewrites it.
- The root `meson.build` now requires Meson 1.10 or newer (`meson_version: '>=1.10'`), matching use of `array.slice()` for registry command arguments.
- The six manual pages snapper still wanted to wrap now match its org formatter.
- The timeouts of `MorsePotTest`, `SuttonChenPotTest`, `ThreadTeamTest` and its `OMP_THREAD_LIMIT` reruns (300 s) and `D2PotTest` (600 s) leave at least 3.5 times the wall time they took on a 2-CPU runner, where meson runs one test per CPU and each test's OpenMP team shares those CPUs. The 120 s of `omp_crossover_v3_analysis`, a pure Python bootstrap that takes 8 s alone, leaves the same margin. A test that is only slow no longer fails the lane, and a hang still does.
- The version-drift check fails when `rgpot-bundle/Cargo.lock` has lost the cog
  markers around the `rgpot-core` version. The ABI stamp page records that a
  whole-archive bundle exports 127 `rgpot_*` symbols, and `rgpot_so_exports` pins
  that count next to the 73 of a normal link. Of those 73, two are
  `rgpot_builtin_muller_brown_new` and `rgpot_muller_brown_config_default`, and
  one is `rgpot_lammpc_potential`.
- The wheel workflow installs each repaired wheel with its `ase` extra and runs `tests/test_ase_plugin_wheel.py`, which skips itself on ASE releases without the plugin API.
- Tighten Sutton-Chen serial-path tests so OpenMP-only counters are asserted
  only where they can change and serial builds stay warning-clean.
- Workflows run on pushes to main and develop and on pull requests, so a branch with an open pull request starts each workflow once. A newer push to a pull request cancels its older run.
- `EMTPot` keeps its installed class layout. The cached Asap kernel lives behind the existing cell pointer, so the C++ soversion and the layout stamp stay as they are.
- `NWChemPotMessageAbiTest` now counts `nwchemc_set_params` calls through `nwchemc_fake_engine` and pins `NWChemPot::setPotentialConfig` across initial parameter refusal, repeated refusal, applied parameters, `none`, and unhandled configuration arms.
- `check_cxx_soversion.py` compares the soversion ledger with `main` for every pull request, as the ABI guide states, so a pull request into `develop` may edit the soversion line `develop` has not released. In a CI clone without `main`, its fallback fetch now writes `origin/main`, so the comparison runs instead of being skipped.
- `doccheck_examples` in a `-Dwith_rpc_client_only=true` build skips the C and C++ examples that link `librgpot`, with the reason `client-only build has no librgpot`, instead of failing with `ld: cannot find -lrgpot`. The examples are still compiled, and a build that is not client-only still fails when `librgpot` is missing.
- `hessian_bench`, a meson benchmark, times the LJPot dense Hessian (analytic, dual-number, central) and Hessian-vector product (analytic, dual-number, central, forward at a new and at the cached centre) on free clusters through `rgpot::HessianVectorCache`, with each result's error against the analytic Hessian and the break-even `k*` of each pair.
- `rgpot_so_exports` is in the test registry, under the feature that is on
  only when that pin is registered. The tabulated EAM and vendored vesin
  pytest modules read their fixtures from the checkout named by
  `RGPOT_PROJECT` or from pytest's rootdir when that directory is the
  checkout, and they skip only when no checkout is present. A file missing
  from a checkout that is present fails the test.
- `schema_ledger.py` pins `Potentials.ordinals` to the rgconf commit in `CppCore/rgpot/rpc/schema.pin`: `check --fetch` compares both vendored copies of `Potentials.capnp` and `common.capnp` with it, and `append` retired the former local entries by recording them again as upstream. The `apply` command, the series checks and the `schema-sync-canonical-*` meson tests are gone.
- `tools/check_header_drift.sh`, run by the `rust_tests` job and by `pixi r check-header-drift`, fails when `rgpot-core/include/rgpot.h` differs from the cbindgen output. The `gen-header` build now reruns when the Rust source changes, and the header carries the `# Safety` notes of the RPC client functions.
- Coverage builds shorten the projected Langevin free-energy samples so
  the run finishes inside the registry timeout. Release runs keep the
  full trajectory. The documentation self-test reads schema member names
  and the ledger wording of a missing member.
- Org pages that failed the snapper width check are wrapped to that
  width. The wording is unchanged.
- The MinGW Wine job skips AotiPublicationStressTest, which replaces a
  file that is still open. Wine is detected only from the ntdll export
  wine_get_version. A directory clock Wine will not move no longer fails
  the stale-directory sweep, and path checks accept two spellings of one
  directory.
- The documentation now covers every public potential, C and C++ header,
  Rust item, Cap'n Proto field, Python name and command line program. Each
  potential has a how-to with its functional form, parameter sources, units,
  a runnable example and its literature citation. `llms.txt` and
  `llms-full.txt` are generated from the sources by `scripts/gen_llms.py`,
  and `doccheck` fails when a public item has no entry or those files are
  stale.
- The plain text builder treats = and ~ as verbatim only when the
  characters around them match org rules, and it leaves \( \) and $$
  math unchanged. llms-full.txt is checked for a backtick inside inline
  math and for an unbalanced backtick on a line.

### Changed

- A batch is spread across the threads requested by `OMP_NUM_THREADS` when no system would open a team of its own. `StillingerWeberSi::energyAndGradientBatch`, `TersoffSiC::energyAndGradientBatch`, and the default `Potential::forceBatchImpl` for a shared instance that is not per-image and not group-collective use `teamSize` with that potential's `kTeamMinAtoms` and `kTeamFrom`, or `rgpot::kTeamMinAtoms` when the potential has no counts of its own. A system that would open a team stays on the calling thread, so the team is not nested and a spread worker does not open one. The results match the one-thread loop. Each Stillinger-Weber or Tersoff configuration uses its own neighbour list.
- CPMD and engine ABI docs clarify that seeding an orbital key chooses the next SCF initial guess from the session's current or stored converged orbitals.
- Repeated loads of one generic engine plugin share a single resident entry. The Rust loader keys plugins by device and inode, opens each file once with `RTLD_NODELETE`, never unloads it, and reuses it for every later load of that file through any path. A path whose file was replaced after it was loaded, or a bare library name, is refused with `EngineError::Load`. The `LammpsPot`, `GromacsPot` and `OpenMMPot` constructors keep their engine loaded the same way, and the metatomic, xtb, OpenMM, NWChem and Skala engines are linked with `-z nodelete` like the UMA engine. A load that fails validation, or finds its file replaced while it loads, closes its handle and leaves no registry entry, so a later load opens the file again and reports the same error.
- The C ABI minor of `rgpot-core` is now 10, in `RGPOT_ABI_VERSION_MINOR`, for the
  additive `rgpot_rpc_client_select_orbitals` RPC export. A caller built against
  minor 0 to 9 still matches; a caller built against minor 10 is refused by an
  older library.
- The C ABI minor of `rgpot-core` is now 5, in `RGPOT_ABI_VERSION_MINOR`, for
  the additive engine and RPC Hessian-vector exports. A caller built against
  minor 0, 1, 2, 3 or 4 still matches; a caller built against minor 5 is refused
  by an older library.
- The C ABI minor of `rgpot-core` is now 6, in `RGPOT_ABI_VERSION_MINOR`, for the
  additive `rgpot_potential_new_eam_table` constructor and the four
  `RGPOT_EAM_TABLE_*` format codes. A caller built against minor 0 to 5 still
  matches; a caller built against minor 6 is refused by an older library.
- The C ABI minor of `rgpot-core` is now 8, in `RGPOT_ABI_VERSION_MINOR`, for the
  additive dense-Hessian, dual-number and derivative-capability functions, the
  `rgpot_hessian_scheme_t` enum, the `rgpot_derivative_caps_t` and
  `rgpot_derivative_counts_t` structs, and the `RGPOT_HVP_AD` and
  `RGPOT_HVP_EXACT` values. A caller built against minor 0 to 7 still matches; a
  caller built against minor 8 is refused by an older library.
- The OpenMP team activation threshold of calls without one of their own, such as
  a neighbour-list build or `EMTPot`, can now be configured with Meson's
  `team_min_atoms` option or the `RGPOT_TEAM_MIN_ATOMS` compile definition while
  retaining 8192 atoms as the default; each classical potential keeps its own
  counts as `kTeamFrom`.
- The RPC protocol minor is now 2 in the C++, Rust and Python mirrors and in the generated C header.
  The wire check reads the minor from all three sources and fails when they disagree, and it keeps the orbital-key introduction minor from exceeding the current minor.
- The engine C ABI header and reference now say that `rgpot_engine_create` takes no callbacks, so a host registers an oracle with `rgpot_engine_set_callbacks` on the handle create returned, and a learning engine that needs one fails its evaluations until then. `EngineSession::load` documents the same order.
- The generic engine C ABI minor is now 11, in `RGPOT_ENGINE_ABI_MINOR` and
  `ENGINE_ABI_MINOR`, for the additive `rgpot_engine_taylor_jet` entry point and
  the trailing `RgpotEngineCaps.jet_order` field. A host serves any engine of the
  same major and layout whose minor is at most its own, so an engine of minor 10
  or earlier without the jet still loads and reports `EngineError::Unsupported`
  only when the jet is requested.
- The generic engine C ABI minor is now 4, in `RGPOT_ENGINE_ABI_MINOR` and
  `ENGINE_ABI_MINOR`, for the additive `rgpot_engine_hvp` entry point and the
  trailing `RgpotEngineCaps.analytic_hvp` capability flag. An engine at minor 4
  serves a host built against minor 4 or newer; a host built against minor 3
  refuses that newer engine.
- The generic engine C ABI minor is now 8, in `RGPOT_ENGINE_ABI_MINOR` and
  `ENGINE_ABI_MINOR`, for the additive return code `RGPOT_ENGINE_ERR_LOCK` (-2).
  An engine at minor 8 serves a host built against minor 8 or newer; a host built
  against minor 7 refuses that newer engine.
- The molecular-dynamics session entry points and operation tables add optional
  neighbour-query and neighbour-setter tail slots while preserving compatibility
  with older callers. The slots are on the C++ session operation table. They are
  not exports of the rgpot-core C ABI, so that minor is unchanged and the C++
  soversion stays 9.
- The workspace declares `rust-version = "1.88"` for `rgpot-core`, `rgpot-pair` and `potctl`, and the pixi `rust` requirement is `>=1.88.0`. An older toolchain now stops with a version message.
- With an empty path, `rgpot::abi::ProfileLoader` and the Rust
  `rgpot_core::profile::library_candidates` try only the `<PREFIX>_LIBRARY` and
  `RGPOT_<PREFIX>_ENGINE` values that are set and not empty, and try the default
  library names only when neither is set. A hint that does not open now fails
  with a message naming each library tried instead of loading a default one.
- `EngineStatus` now carries the magnitude of a failure code as a `NonZeroU32`, so a recoverable failure is always a positive C code, a fatal one is always negative, and no failure can encode as success. `recoverable` and `fatal` build a status from a code of the matching sign, `from_code` and `code` round-trip every `c_int`, and `or_invalid` turns a success into `ENGINE_ERR_INVALID`.
- `EngineSymbols` is now `#[non_exhaustive]`, so a symbol added to the engine ABI no longer breaks code that builds one.
  Build it with `EngineSymbols::new`, which takes the four required symbols, and add each optional symbol with a `with_*` method such as `with_set_callbacks`.
  The optional `model_state`, `set_callbacks` and `hint` symbols this release adds are set the same way.
- `MetatomicPot` now reports `ForceOut::variance` as a variance in squared units rather than an RMS: the square of the mean model `energy_uncertainty` in eV^2, or, when a multi-pass run has no model uncertainty, the mean-square force residual over orientations in (eV/Angstrom)^2.
  The value is passed through unchanged by the C++, C and Python entry points.
  Direct users who relied on the previously documented eV or eV/Angstrom must take the square root of `variance` to get the old units back.
- `librgpot` takes soversion 9 in `kCxxSoversion`. `lr::SumPot` now derives from `SumPot` and grows from 56 to 120 bytes, and its old constructor and `add` symbols are gone, and `D4Config` gains `periodic`, and `D2CpmdConfig` gains `symmetry`. `PotCaps` gains `analyticHessian`, `adHessian` and `adHvp`, and `PotentialBase` gains the virtuals `hessianImpl`, `hessianAdImpl` and `hessianVectorAdImpl`, so `kCxxLayoutRevision` is 2. `nlist::Builder` and `nlist::Walk` keep their state behind a pointer, so each is one pointer wide. The layout stamp now covers `SumPot`, `lr::SumPot`, `PotType::LrSum`, `D2Config`, `D2Pot`, `PotType::D2`, `D2CpmdConfig`, `D2CpmdPot`, `PotType::D2Cpmd`, `D4Config`, `MullerBrownConfig`, `MullerBrownPot`, `nlist::Builder` and `nlist::Walk`.
- `rgpot.lammps.FixExternal` does less work per step: it skips unit factors that equal one, passes atom IDs that already run 1..N straight through without a sort and a scatter, takes the volume from the diagonal of LAMMPS' lower-triangular cell instead of a determinant, and symmetrizes the virial with plain floats. Results are unchanged, and the virial no longer depends on a potential leaving its cell argument alone.
- `rgpot.lammps.FixExternal` now raises `ValueError` when the LAMMPS box is non-periodic in every direction (`boundary f f f`) and the potential reports `caps()["periodic"]`. Such a potential, for example `rgpot.LJPot`, wrapped pairs across the box extents and returned a wrong energy without a warning; use a non-periodic potential such as `rgpot.LJClusterPot`. The new keyword `allow_periodic_on_free_box=True` restores the previous behavior.
- `rgpot::abi::ProfileLoader` now tries only the library a non-empty explicit
  path names. When it does not open, `load()` throws a message that names the
  path instead of falling back to the `<PREFIX>_LIBRARY` and
  `RGPOT_<PREFIX>_ENGINE` hints and the default library names. The lammpc
  `LammpsPot` constructor, `rgpot_lammpc_potential()` and the Python
  `rgpot_lammps_potential(config_bytes, engine_path)` overload inherit this.
  The lammpc constructor also refuses truncated bytes and a `PotentialConfig`
  without the `lammps` arm with `std::invalid_argument` (`ValueError` in Python).

### Fixed

- A Hessian-vector product or Hessian callback of `rgpot-core` that fails without calling `rgpot_set_last_error` is reported as `<callback> returned <status>`, instead of with a message an earlier call on the thread left in `rgpot_last_error`, and a successful derivative call, forward and central differences included, leaves that earlier message in place.
- A MinGW build, native or cross, builds the fast pair list for `x86_64-pc-windows-gnu` and links Rust with the C++ compiler from the native or cross file (`CARGO_TARGET_<TRIPLE>_LINKER`). Before, a cross build took the Rust target from `CARGO_BUILD_TARGET` in the environment, so a reconfigure without it linked Linux system libraries into the DLL. A cross file can name another triple with the `rust_target` property. The `nlist_rust` test builds its DLL and Rust caller with the same compiler and target and runs the caller under the cross file's `exe_wrapper`.
- A Windows `librgpot` DLL exports the whole public C ABI. Before, it exported the version, the ReducedPair and Sutton-Chen functions and the C++ layout stamp, and none of the neighbour-list, MD session, fast pair list or external-potential functions. The public C headers mark their functions with `RGPOT_API` from `rgpot/rgpot_api.h`, and the fast pair list names reach the DLL through a module definition file. On Windows the engines that call only the C ABI link the DLL. `lj_eval`, `pot_spec` and the drivers built on them link the static archives instead, because the DLL exports no C++ class. The `abi_symbols` test runs on Windows and reads the DLL export table, and its baseline now lists every `RGPOT_API` function.
- A `Neighbours::update` that rejects a cell or a coordinate, or finds the
  generation counter exhausted, throws before anything changes, so a built
  object keeps its list and generation and a default-constructed object stays
  empty without a generation.
  An allocation that fails part way through an update leaves a
  default-constructed object with a cleared generation, so the next update
  rebuilds instead of reusing a partial list, and a copy assignment that throws
  leaves its target unchanged.
  A moved-from `Neighbours` is left as a default-constructed one with a cleared
  generation, so its next update builds from its own coordinates instead of
  reading coordinates it no longer holds.
- A cached metatomic engine no longer serves results computed under a previous charge or spin after `rgpot_engine_set_charge_spin`. The metatomic potential now fingerprints the settings its model sees, so a change of charge or spin is a cache miss with a new key.
- A classical potential call too small to gain from an OpenMP team no longer starts one, whatever `OMP_NUM_THREADS` asks for.
  `SuttonChenPot` started one on every call: on 8 threads a 55-atom cluster took 41.9 to 62.4 µs per call over four runs, against 14.0 to 14.1 µs on one thread, and now takes 13.0 to 13.1 µs on either.
  From the `kTeamMinAtoms` of their class up, `SuttonChenPot`, `LJPot`, `LJClusterPot`, `MorsePot`, `ZBLPot`, EAM, `StillingerWeberSi` and `TersoffSiC` sum each atom's energy, force and strain on their own and add the atoms in index order, so the energy, forces, stress and per-atom terms have the same bits on any `OMP_NUM_THREADS`.
  A team used to add one partial sum per thread: on 8 threads the energy moved in the last bits on 10 to 16 of 16 noisy cluster frames for every potential but ZBL, by up to 2.06e-4 epsilon for a 10000-atom reduced Sutton-Chen cluster, and the forces moved for every potential but Sutton-Chen and ZBL.
  `rgpot::teamSize` gives the thread count of a call from the counts of the potential's class, `rgpot::TeamThreshold` sets one threshold for every potential called on its thread, for benchmarks, and `rgpot::teamsOpened` counts the OpenMP regions the calling thread has opened.
  The installed headers add `teamSize(std::size_t, std::size_t, TeamFrom)` and change the signatures of `nlist::sumPairs` and `sumCentres`; `teamSize(std::size_t)` remains exported and deprecated as a forwarding wrapper to `teamSize(n, kTeamMinAtoms, {})` under `kCxxSoversion` 9, and the C and engine ABIs (`RGPOT_ABI_VERSION_MINOR` 10, `RGPOT_ENGINE_ABI_MINOR` 11) are unchanged.
  `kEamThreadedTolerance` is gone from `EAM.h`, since every thread count now gives the bits of one thread.
- A fatal engine failure is no longer hidden behind a recoverable one in `EngineSession`.
  A callback or transform failure and the code the engine returns are judged together, and a fatal failure from either side wins.
  The first fatal failure, with its message, is the one that poisons the handle, so every later call returns the identical `EngineError::Fatal`.
  A later recoverable callback failure no longer replaces an earlier panic, and a fatal hint whose error buffer is empty reports a message naming its code instead of an empty one.
- A negative return code of `RGPOT_ENGINE_ERR_LOCK` (`-2`) now reports
  `EngineError::Lock` with the operation name and poisons the handle like
  other fatal codes. Handles of an engine that declares `reentrancy` 2 now
  serialize their native calls process wide, so two handles on the same
  engine library no longer run native code at the same time. The
  `libpotlibc_engine` plugin frees the handle in `rgpot_engine_destroy` even
  when its lock cannot be taken.
  A negative `rgpot_engine_caps` code at load now rejects the engine and
  destroys its handle instead of being read as a reentrant engine, and a
  positive code or a truncated answer keeps process wide serialization.
  A handle that finds the process wide lock poisoned by a panic in native code
  reports `EngineError::Lock` and stays unusable, so every later call returns
  that error instead of retrying the lock.
- A neighbour query on coordinates that lie whole cells outside the box no longer drops pairs. `rgpot::nlist::build`, `Builder`, `Walk`, the C ABI and `rgpot::lr::DualList` fold periodic coordinates in fractional space before the image search. The shift each pair reports is for the coordinates as given, so `r_j - r_i + S H` is the pair vector for unwrapped input too, which is what the Fortran bridge, metatomic and UMA rebuild their vectors from. In a 10 Å box, atoms at 0 and 18.5 Å with a 2 Å cutoff now give their one pair, at shift -2. Moving an atom by whole cells keeps the same pairs and changes their shifts by the same whole cells. A periodic coordinate that is not finite, or lies more than 2^29 cells from the origin, is rejected with `Code::Box`. `rgpot::lr::DualList` rejects a non-finite position with `std::invalid_argument` ("positions must be finite").
- A non-periodic OpenMM session no longer wraps neighbour queries through the
  placeholder box. A neighbour buffer shorter than the list is an error from
  that session, and the error names the pair count the caller must provide.
- A periodic `Cluster4` neighbour list now holds the same rows whichever output fields it stores. Rows of one cluster and partner were merged by their stored shift, so a query without shifts merged every periodic image of a partner into one row and lost the rest. In a 3 Å cubic cell with four atoms and a 4 Å cutoff, `build`, `Walk` and `Builder` now give 74 rows for every choice of shifts, vectors and distances, where a query without shifts gave 4. Pair lists were not affected.
- A rejected `shareFromCalculator` now waits until every rank has written the mismatch before `MPI_Abort`, so a peer killed by the first abort no longer leaves an empty log.
- A static MinGW build with the fast pair list builds. It installs `librgpot_fastnlist.a` beside `librgpot.a` and lists it in `rgpot.pc`; before, `ar` took the Rust archive for the archive it was writing. `print_cxx_layout_stamp` declares fmt, so `CxxSoversion` has its printer when fmt is not on the default include path. `PluginLifetimeTest` and `CxxSoversion` run their program under the cross file's `exe_wrapper`, and `test_list_parity` lists a cross build's Catch2 tests under it. The `abi_symbols` test is a Python script and needs no POSIX shell, `dumpbin` or `objdump`.
- A static Windows `librgpot` now defines `RGPOT_STATIC` for in-tree targets.
  Targets in the same CMake build no longer compile its headers as dllimport.
  `scripts/abi_symbols.py` reads a PE export table with no names as empty.
  It used to fail on the zero name table address of such a table.
- A test name that contains spaces stays one test when the project is configured.
  The accepted version range now includes every release this tree is tested with, through 3.31.8.
- A variable-cell neighbour update now measures image shifts of both signs before it reuses a stored list. A lattice change whose signs differ no longer drops a pair that has moved into the cutoff. A negative, infinite or NaN skin does not count as a reuse in either `withinSkin` overload. Periodic positions are folded before the skin comparison, so adding whole lattice vectors does not move an atom.
- An engine oracle callback can no longer hand native code a success it did not earn. A failure that carries a success status is reported to the engine as `ENGINE_ERR_INVALID`, and an oracle answer with a non-finite energy or force is refused with the same code, so the evaluation fails with a recoverable `EngineError` instead of returning the unwritten buffers.
- CPMDPot drops a session when the engine refuses the selected orbital key, so a later setParams with the same parameters creates a session and applies the key again.
- CPMDPot now applies selectOrbitals immediately on a live cpmdc session and re-applies the key after session recreation, so select-select-force sequences are unambiguous.
  CPMDPot adds copyOrbitals(from_key, to_key) and uses cpmdc_session_copy_orbitals when the engine exports it, while engines without that primitive report unsupported instead of silently succeeding.
- CPMDPot now carries the embed refusal detail when session creation fails, and `CPMDPotMessageAbiTest` pins the exact `rejected by embed` wording with that reason.
- CPMDPot reports the orbital key refusal on a later force when configure could not open a session, instead of an empty libcpmdc not loaded suffix.
- CPMDPot::setPotentialConfig reports a config rejected by embed, with the orbital key refusal, when configure drops the session after the engine refuses a key, instead of saying the engine is not loaded.
- Cache and parameter contract tests no longer hard-code shared temporary paths.
  They now use per-test temporary locations, which avoids cross-test collisions in
  parallel runs and works across sandboxed environments.
- Configuring the build no longer fails when the Python prefix has no include directory and GROMACS cannot be found. A build that requests GROMACS and cannot find it still stops with an error.
- Constructing and freeing a LAMMPS, GROMACS or OpenMM potential no longer takes another dynamic-linker reference for an engine file that is already open. An open that does not become the resident engine is closed, and the resident engine stays mapped.
- Coverage instrumentation multiplies every C++ test timeout by 4. That multiplier is what lets the instrumented suite pass: the thread-team matrix and `NeighbourListTest` both run past their ordinary limits, and concurrency among the thread-team copies was only part of the thread-team failures.
  Each thread-team registration runs one Catch2 tag. Open Sutton-Chen is split across tags, one thread count of the large cluster at a time, because a coarser registration stayed over half its coverage limit on 4 cores. A coverage build runs those registrations one at a time. An ordinary build still runs them in parallel. The limit on each one is 480 seconds: two instrumented runs each had a single registration take about 330 seconds after that same binary had taken under a minute.
  `NeighbourListTest` takes about 40 seconds once it is instrumented, past the 30 second default, so its limit is 120 seconds.
- Direct `forcePerAtom` calls, the C ABI Hessian-vector entry points, and `cellVolume` now reject missing or non-finite inputs before invoking kernels or publishing stress, while `FusedCache` no longer stores validation failures.
- Each classical potential opens an OpenMP team from atom counts measured for it, one each for teams of two, three, and four or more threads (`kTeamFrom`, a `rgpot::TeamFrom`), where one threshold of 8192 atoms held for every potential.
  On 8 threads, from those counts to 4000 atoms, a call takes 0.20 to 0.86 of its time under that threshold: `StillingerWeberSi` at 4000 atoms takes 431 µs per call, against 2042 µs.
  Only `StillingerWeberSi`, from 112 atoms, and `TersoffSiC`, from 80, open a team of two, since the other potentials evaluate each pair twice on a team and gained nothing from two threads: at 10000 atoms on 2 threads `LJPot`, `LJClusterPot`, `MorsePot` and `SuttonChenPot` now take 0.80 to 0.93 of their time.
  One thread sums atom by atom from the `kTeamMinAtoms` of the class, the least of its counts, so the bits still do not depend on `OMP_NUM_THREADS`; below 8192 atoms that costs one thread up to 12% (`StillingerWeberSi`).
  A batch spreads a system only when that potential would leave it on one thread, so a system between the class floor and 8192 atoms is not spread and a spread worker does not open an inner team.
  In `Stream` mode one thread evaluates each pair once again, with the bits of a team: at 10000 atoms `LJPot`, `LJClusterPot`, `MorsePot`, `ZBLPot` and EAM take 0.49 to 0.70 of their time, and `LJClusterPot` at 100000 atoms 0.62.
  The one-argument `rgpot::teamSize(std::size_t)` overload is deprecated and remains exported as a forwarding wrapper under `kCxxSoversion` 9, and the C and engine ABIs (`RGPOT_ABI_VERSION_MINOR` 10, `RGPOT_ENGINE_ABI_MINOR` 11) are unchanged.
- Every engine plugin built from this repository now exports
  `rgpot_engine_abi_stamp`, which reports the engine ABI major, minor and layout
  revision. The stamp is defined once, in `CppCore/rgpot/engine/EngineAbiStamp.cc`,
  and the table-based libraries, the standalone Lennard-Jones, native-unit and
  OpenMM plugins, and the test stub all add that file. A loader such as
  `EnginePotential` in `rgpot-core` can now read the ABI minor and layout revision
  of these plugins directly instead of assuming minor 0.
- Exported rgpot_lammpc_potential from the shared library and added it to
  the ABI baseline checks so hidden-visibility and Windows builds keep the
  constructor in the public C surface.
- Fixed building the Python extension when `with_python` is enabled: the
  `has_analytic_hvp()` query no longer calls a non-const thread-slot accessor
  on a const handle (the same `primary().caps()` path as `caps()`).
- Fixed strict-aliasing undefined behavior in `rgpot::Potential::cacheKey` under GCC 13 `-O3` Release builds by defining `XXH_FORCE_MEMORY_ACCESS 0` before including `<xxhash.h>` and copying the hashed scalar keys into byte arrays via `std::memcpy`.
- Fixed the `RgpotEngineAbiStamp` minor-version rule in `engine_abi.org` to match
  `rgpot_engine_abi_compatible` and `RgpotEngineAbiStamp::serves`: a host built
  against minor `N` accepts an engine of the same major and layout revision whose
  minor is at most `N` (rejecting plugins with a newer minor than the host's).
- Force and energy calls now reject a missing coordinate, species or force buffer, a missing cell on a periodic potential that needs one (a molecular dispersion sum such as D2 still takes none), and a system with no atoms. A system with no atoms is reported on its own, including when the force buffer is also absent.
- Fortran-backed potentials now preserve complete error messages when separate instances fail concurrently.
- Generic and external engine initializers may load another engine without deadlocking the process. Engine handles enter the resident registry only after their initializer, required-symbol checks, ABI validation and first successful construction complete.
- Langevin and energy-conserving steps with distance constraints add the impulse
  of the position constraint to the velocity. The driver projected the velocity
  onto the new tangent space alone, which removed kinetic energy from every
  rotating rigid group, so a free rigid rotor lost energy.
- Listed `d2cpmd-symmetry.diff` (`D2CpmdParams.symmetry`) among the local schema diffs in `docs/orgmode/reference/potential_config.org`.
- Listed `libcomposite_engine`, its Windows exclusion, and `RGPOT_ENGINE_COMPOSITE` in `docs/orgmode/howto/engine-plugins.org`.
- MEAMPot now rejects a null force buffer when the system has atoms, with a clear error before it writes any result.
- MEAMPot now rejects a null output pointer with a clear error before it writes any result.
- MOPACPot now refuses a libmopacc whose C ABI or library ABI is not exactly
  2, one that lacks mopacc_c_abi_version, mopacc_abi_version,
  mopacc_result_size or mopacc_params_size, or one whose reported MopacCResult
  or MopacCParams size does not match its own, before calling into the engine.
  The load error names the expected and found values. A nonzero status from
  mopacc_set_params also leaves the engine unloaded.
- Meson registers the `ipi_driver_ase` test only when the `python3` that runs it can import pytest, numpy and ASE, so a build whose Python lacks numpy, such as an offline `--wrap-mode=nodownload` build, no longer fails the test with `ModuleNotFoundError`.
- Moved the vendored vesin lockstep test into the Python CI test path and aligned potctl/CI vesin wheel constraints with the vendored 0.6 series.
- On MinGW, `simd::exact_fma`, `mul_add`, `fma_quick` and `fma_array` return the bits of a hardware fused multiply-add for every input. They handed zeros, subnormal results and inputs with extreme exponents to the C library's `fma`, and mingw-w64's `fma` is not fused, so those results could be off in the last bit or return the default NaN instead of the input NaN. A MinGW build now uses a correctly rounded software fma there. In an unoptimized build the hardware bodies of the LJ, Morse and EAM kernels and of `exp` also called mingw-w64's `fma`, because nothing was inlined into their `target("fma")` functions; they now use the instruction, or the same software fma when the copy stays out of line. `exp_hw_n` and the EAM hardware body are flattened like the other hardware bodies. `FmaExactTest` and `ExactFmaEq` compare against the instruction on MinGW instead of mingw-w64's `fma`, and `ExactFmaEq` checks the software fma against the instruction on every x86-64 GCC or Clang build running on a CPU with FMA.
- On Windows, the `LammpsPot` result-cache key now hashes the final path of the
  module that loaded, read from the loader, instead of the requested path. A
  backend selected through the environment passes an empty path, so two such
  backends with the same version, ABI revision and configuration used to share
  one key.
- One fused-evaluation cache now stands in front of every fused potential, and it stores a failed call like a result: `eindir_objective_eval` and `eindir_objective_grad` at one geometry, and the Hessian-vector centre, run the engine once, and a failed energy followed by its gradient used to run a GFN2 SCC twice.
  The cache key is the position bits plus the atomic numbers, the cell and the settings of the engine (charge, spin, accuracy, electronic temperature, SCC iterations and mixer, engine build), and a request with the stored key returns the stored success or failure with no engine call. `rgpot::HessianVectorCache` follows the same rule for potentials used without the Rust core.
  A failed energy is not retried at the same bits: a warm-start retry after an SCC failure needs a changed setting, installed with `rgpot_potential_set_fused_settings` or `PotentialHandle::sync_fused_settings`, or `rgpot_potential_hvp_invalidate`.
  The gradient of a failed energy returns the same status and the same `rgpot_last_error` message as the energy.
- Periodic `EAMTablePot` evaluation sums every image inside the cutoff, so a cell whose face spacing is at most twice the cutoff is valid.
  An atom can interact with its own lattice translations.
  Evaluation still throws `std::invalid_argument` when one axis would search past an image extent of 1024, or when the three axes together would visit more than 262144 lattice shifts for one pair.
  A fractional separation that is not finite, or whose nearest integer plus the image extent does not fit in `int`, throws `std::invalid_argument` as well.
- Periodic and partially periodic cells with any nonzero off-diagonal entry are binned in fractional coordinates from the internal minimum image inverse, shifted to a coordinate reference and wrapped on periodic axes before projection so translation stability holds at large coordinates, while non-periodic fractional axes span the actual coordinate range without wrapping. Fractional projections are retained only as reusable update workspace. Orthorhombic cells keep a Cartesian grid. Cells with a zero or negative diagonal entry on a periodic axis, such as a cube rotated by 90, 120 or 180 degrees, also take the fractional binning path and stay periodic along that axis. Uniformly scaled tiny cells keep their scaled inverse in `MinimumImage`, while a singular or ill conditioned cell in fractional binning, or a non-finite fractional coordinate, raises an error instead of falling back to Cartesian bins.
- Persistent result-cache keys are a version byte followed by a 64-bit hash in little-endian order, independent of the width of `size_t`. The engine ABI reference documents that the cache has no size cap and is bounded by removing its directory.
- Persistent result-cache keys preserve the order of the potential type, parameter fingerprint, and geometry fields. Scalar and batched callers reuse the same results under optimized builds. An existing database retains its old entries and recomputes each geometry on its first visit using the new key format.
- Python potential calls now map kernel `std::domain_error` failures, including coincident `LJPot` atoms across `__call__`, `evaluate`, `evaluate_per_atom`, `evaluate_batch`, `evaluate_lj`, `GromacsPot`, `OpenMMPot`, and `ExprPot`, to `rgpot.NumericalError`, and `rgpot_lj_eval` catches C++ exceptions so the C engine path returns an evaluation error code instead of unwinding across `extern "C"`.
- RPC clients now check the advertised protocol minor before sending the orbital-key `ForceInput` fields.
  Those fields first appeared in minor 2, and a keyed request to an older server fails with the required and advertised minors.
  `ForceInput.neighbors` needs no minor above 0, and requests without an orbital key stay valid against a minor-1 server.
- Remote RPC failures now return error statuses instead of aborting: closed peers, truncated replies, callback-side server exceptions, and NULL RPC tensor data are all reported through the C ABI and Python RPC calculator as ordinary call failures.
- Rust C-ABI EAM table tests keep their borrowed periodic cell storage alive across evaluation.
- Sync the vendored mopacc C ABI header in MOPACPot with upstream mopacc,
  adding missing fields to MopacCParams and MopacCResult to prevent stack
  corruption on by-value result returns. Assert struct sizes and field
  offsets at compile time, and add an upstream-layout canary test.
- Taylor jet callbacks now reject non-finite directions, interval operations
  handle empty inputs consistently, and the API reference lists the complete
  Taylor jet error and entry point sets.
- Tersoff and ZBL build a stored neighbour list on the same OpenMP atom slices as the other classical potentials once a system has 8192 atoms and the stencil holds at least 4096 candidates, more than one per atom.
  The index matches one thread.
  An open axis whose nominal grid does not fit in memory is binned on the atom span, so a cluster in a huge box keeps a real cell list.
- Tersoff reference documentation now describes rejected cells as
  zero-volume or non-finite to match `cellVolume` using `|det|`, and the
  test suite adds C++ left-handed mirror checks alongside Fortran and
  Python strain finite-difference and thread-invariance stress tests.
- The AOTI package export enables deterministic algorithms while it compiles, so a scatter that used to lower to unordered atomic adds gives one output for one input, and the caller's deterministic setting is restored afterwards.
- The Asap `Exception` class used by `EMT` no longer shares one `thread_local` buffer between `GetMessage` calls, so the messages of two live exceptions no longer overwrite each other, and copy assignment no longer leaves two exceptions deleting the same message stream. `GetMessage` still returns `const char *`, `sizeof(Exception)` is still 8 bytes, and every symbol the class exported before is still exported; copy assignment is added.
- The C++ neighbour table comments, the neighbour list tutorial and the neighbour layer notes now state that a reused update rewrites only the shifts. The pair vectors and distances stay those of the last search until a full rebuild, as the C header already says.
- The C-ABI potential tests in `rgpot-core` keep the position, atomic number and cell arrays on the heap, so the borrowed force input still points at live storage when it is evaluated.
- The Meson configure step reads the libc++ version again on Windows toolchains built on clang (clang, clang-cl, llvm-mingw, MSYS2 CLANG64). A libc++ older than 19 once more gets the `__atomic_fetch_or` fallback check, and configure stops with a clear error when that fallback cannot link. MSVC is still skipped, and the probe passes the C++20 standard flag again.
- The OpenMM MD session takes its first Langevin force on the hyperplane, keeps
  its random stream across calls, holds serialized distance constraints, and
  reflects at walls until the state is inside, where it used to return after
  eight passes.
- The Python RPC calculator no longer replaces a protocol minor mismatch with a generic calculate failure.
  It raises `ProtocolMinorMismatchError`, a `RuntimeError` whose message and `required_minor`, `server_minor` and `client_minor` attributes report the minor the request needs, the one the server advertised and the one the client implements.
- The Python extension exports `evaluate_metatomic_dlopen` in every build.
  `has_metatomic_dlopen` is true whether or not torch is linked, and the
  engine library is still opened at runtime.
  Installing the package includes the `librgpot` SONAME as a regular file,
  so the extension can load the library from a wheel.
- The Rust engine loader rejects a non-finite cell and a negative jet scale with `EngineError::Invalid`.
- The `RgpotRpcCalculator` ASE calculator now lists `free_energy` in `implemented_properties`, so `get_potential_energy(force_consistent=True)` returns the energy instead of raising `PropertyNotImplementedError`.
- The `orca` arm of `rgpot::makePotential` now loads a `liborcac` built from orcac main, which exports the frozen C ABI 1 (`orcac_config_init`, `orcac_create`, `orcac_energy_gradient` and the rest of the thirteen functions in `include/orcac.h`). Before, `rgpot::ORCACPot` looked for session functions such a library does not export, so `available()` was false. `rgpot::ORCACConfig` now holds the `OrcaCConfig` fields (method, basis, keywords, blocks, executable, scratch directory, charge, multiplicity, processes, memory, timeout and restart), the arm lowers `ORCAParams` onto them without reading `common`, `keepFiles` is refused, and the force is the negative of the gradient `liborcac` returns. `ORCACPot::runDirectory` is replaced by `ORCACPot::engineVersion`. rgpot builds against a vendored `orcac.h` pinned to that orcac commit. The build option `-Dorcac_main=enabled` builds `liborcac` at the pinned commit and runs the arm against it.
- The `rgpot-pair` C kernels now build with GCC 4.8, the compiler in the
  manylinux2014 aarch64 cross image. The build asks for `-std=gnu99` where the
  compiler is not MSVC, and the Gupta buffers use `__thread` on GCC and Clang.
- The `rgpot-pair` x86 kernels now select their AVX2/FMA clones with CPUID and
  XGETBV instead of compiler runtime tables. This keeps accelerated kernels
  available when Rust links the static library with `-nodefaultlibs`, including
  on x86_64 macOS, while retaining the scalar fallback.
- The built-in LEPS presets now use the extended-LEPS singlet and anti-Morse
  triplet factors, restoring their published well depths and barriers.
- The capability handshake, in Rust and in C++, treats a schema id
  with an optional `0x` prefix, in either case, as the same Cap'n
  Proto file id. The C++ check uses one helper for that comparison.
- The composite engine and the CMake CuH2 plugin link the umbrella library.
  They no longer define their own `Neighbours` generation counter.
- The documentation of `EngineStatus::code` no longer promises an exact round trip for every status.
  A status built from a return code round-trips exactly, while a failure built directly with a magnitude beyond the range of `c_int` saturates to `c_int::MAX` or `c_int::MIN`.
- The engine drivers build on Windows. `lj_eval.cpp` and the engine tests load engine libraries through the cross-platform loader in `rgpot/plugin/DynLib.hpp` instead of `dlfcn.h`. `capi_threads` uses Win32 threads on Windows. The native Lennard-Jones engine exports `rgpot_md_native_force`, which a Windows DLL did not. The ThreadSanitizer binary is skipped on Windows. The `lj_eval` library declares the fmt headers `LJPot.hpp` includes, so the engines build when fmt is not on the default include path.
- The fast neighbour list now uses `linkcell` 0.4.2 and `minimage` 0.2.3,
  without `minimage` compatibility C symbols that can conflict with mimalloc.
- The generated C header `rgpot-core/include/rgpot.h` now declares `ENGINE_ERR_INVALID`, the engine code for an invalid argument, next to `ENGINE_ERR_UNSUPPORTED`, so C hosts can name the code that the Rust loader returns.
- The installed CMake package of a static rgpot built with OpenMP now finds OpenMP itself. `find_package(rgpot)` used to fail at generate time because `rgpot_nlist` named `OpenMP::OpenMP_CXX` and nothing defined it. A consumer links the runtime library and is not compiled or linked with `-fopenmp`.
- The lazy ASE 3.29 calculator plugins keep the module ASE binds when `rgpot.ase.__ase_plugins__` is first read, including when several threads read that attribute at once.
- The plane multiplier of `ProjectedLangevin` includes the centripetal force that
  holds rotating distance constraints, and the run records the metric
  `Z = 1 / [(J M^-1 J^T)^-1]_plane` of every step. The free energy along the plane
  offset is the integral of the mean multiplier minus `kT ln <Z^(-1/2)>`. A plane
  normal that translates rigid groups as a whole is unaffected.
- The two argument `rejectInputCall` and `guardInputCall` and the three argument `rejectForceCall` and `guardForceCall` stay exported from `librgpot` as wrappers over the new forms with `needsSpecies` true, so binaries built against the old signatures still load under `kCxxSoversion` 9. The `needsSpecies` parameter no longer has a default, and the C ABI in `rgpot.h` is unchanged.
- The variance that `ExprPot` returns now scales each child variance by the square of its weight, as `SumPot` does, where it used the weight itself. The two agree only for a weight of one, and a negative weight no longer gives a negative variance.
- UMA tasks, including `omol`, accept explicit spin `0` without rewriting it to `1`. `rgpot::validate_uma_spin_for_task` in `rgpot/UmaPot/UmaConfig.hpp` now enforces `[0, 100]` for every UMA task, with only negative spins and spins above `100` rejected. The same bounds are enforced in `UmaPot` and `libuma_engine`, while `rgpot_engine_set_charge_spin` still reports unsupported values through `rgpot_engine_last_error`. Also tightened drop-in behavior so `librgpot_gromacs` keeps an explicit engine spin when the qmi `MULTIPLICITY` is the grompp default `1`, including when `CHARGE` is not `0`. The setter takes charge and spin together, so the drop-in reads that spin back out of the engine `CONFIG` and applies the qmi charge with it. Only a non-default `MULTIPLICITY` replaces the spin. A config that leaves spin at its default falls back to the qmi charge and multiplicity, and the default pair `CHARGE 0 / MULTIPLICITY 1` is still left on the engine config. Metatomic and Skala reject spin values below `1`, and `librgpot_gromacs` refuses an engine `CONFIG` whose spin is below the minimum of its engine (`0` for UMA, `1` for metatomic, Skala and NWChem) instead of forwarding it.
- UmaPot refuses a device that the AOTI package was not exported for, passes a CUDA index to the runner and rejects an unknown device string.
- Updated the README, the XC-kernel note and the architecture reference to
  describe how `PotentialConfig.d4` configures `potserv <port> D4` (including
  periodic 3D lattice evaluation and stress) and `d4` builtin composite terms,
  rather than stating that the `d4` arm is unapplied and refused everywhere.
- `CBEPot` serializes potlibc calls on one handle. Distinct handles may run at the same time, and the classical engine reports per-instance reentrancy for CBE.
- `EMTPot` no longer crashes after a refused call. An element the provider does not cover, a cell too thin for the kernel, a singular or non-finite cell and non-finite positions raise `std::invalid_argument` in C++ and `ValueError` in Python, and the next call starts from a clean state. A refused call no longer leaks the Asap neighbor list index, which AddressSanitizer reported. A call with zero atoms raises `std::invalid_argument` (`invalid parameter: nAtoms must be positive`) in C++ and `ValueError` in Python.

  `EMTPot` raises `std::domain_error` when atoms sit on top of each other and the forces are not finite, and Python reports it as `rgpot.NumericalError` from `__call__`, `evaluate` and `evaluate_batch`. Before, the call returned NaN forces without an error.
- `EMTPot` no longer keeps force evaluation serial above `rgpot::kTeamMinAtoms`.
  It now evaluates the Asap half list in two OpenMP phases, one owner-row pass
  and one incoming-partner pass, so the force sum order is fixed and the
  energy and forces are bit-identical at 1, 2, 4 and 8 threads.
  Below `rgpot::kTeamMinAtoms` it still runs on the calling thread and opens
  no OpenMP region.
- `EMTPot` no longer recomputes forces for a second `EMT::GetCartesianForces` call on unchanged atoms above `rgpot::kTeamMinAtoms`. The two phase OpenMP force path now records the atoms change counter the way the serial paths do, and the forces are bit-identical to before at every thread count.
- `EMTPot` returns the energy and forces of the new cell when the cell changes between calls with the same species. Asap's neighbour list does not notice a new cell, so the kernel kept the old pairs and a strain of 5% gave an energy tens of eV off. Numerical stress through `RgpotCalculator` with `"emt"` now agrees with ASE's `EMT`.
- `EnginePotential::load` now calls `rgpot_engine_available` before
  `rgpot_engine_create` when the plugin exports it, and refuses an engine that
  reports itself unavailable with `EngineError::Unavailable`. A plugin that does
  not export the symbol, or exports a null entry for it, is treated as available,
  so such a plugin keeps loading. The symbol is the optional
  `EngineSymbols::available` field, set with `EngineSymbols::with_available`, and
  `EngineSymbols::new` keeps taking the four required symbols. Every other
  resolved symbol is checked for a null address, so a library that exports a null
  required entry is reported as a missing symbol instead of being called.
- `EnginePotential` now validates `rgpot_engine_abi_stamp` (major, minor and
  layout revision) on load and poisons its handle after a negative return code so
  subsequent calls never re-enter a failed native engine, and `rgpot_engine_hvp`
  rejects overlapping `direction` and `hv_out` buffers with
  `RGPOT_ENGINE_ERR_INVALID`.
- `EngineSession::set_callbacks` no longer frees a callback context the engine may still hold.
  A successful replacement installs the new callbacks before the old ones are dropped, and a panic in the old callbacks' destructor is caught and reported as a fatal failure instead of unwinding out of the call.
  The callbacks of a refused replacement are kept alive until the engine is destroyed, because an engine built before the header forbade it may have retained them.
  The `rgpot_engine_set_callbacks` header and reference now state the lifetime of the previous and the supplied `ctx` after a successful and a refused call.
- `EngineSession` no longer returns a false success when a refused callback state that the engine kept registered fails.
  A panic in the oracle, acquisition or model-update callback of such a state is now collected like one from the registered callbacks and reported as a fatal failure.
  The same holds for `set_callbacks`, `hint` and the force calls.
- `LJPot` keeps its `nlist::Neighbours` scratch per thread through `rgpot::Kept`, so concurrent `const` evaluations on a shared `LJPot` instance run without locks or races, and `LJPot::caps().reentrancy` is now `Reentrancy::SharedInstance`.
- `LJPot` keys the neighbour list it keeps on each thread on
  `Neighbours::generation()` instead of its own address.
  A destructor releases only the lists of the thread it runs on, so an `LJPot`
  built at the address of a destroyed one used to find that one's list on every
  other thread.
- `MinImageIsaVariants` runs its binaries under the cross file's `exe_wrapper` and accepts the CRLF line a Windows binary prints, so it passes in a MinGW cross build and under MSYS2.
- `MorsePot` on a stored neighbour list no longer evaluates each pair from both ends on an OpenMP team, and `Neighbours::fastTiles` no longer bypasses `walkStored` on stored lists.
  When `MorsePot::kTeamFrom` opens a team for the requested thread count, from 2048 atoms on three threads and from 24 (`MorsePot::kTeamMinAtoms`) on four or more but never on two, and the stored list holds at least 4096 candidate upper pairs, `MorsePot` evaluates each upper pair once in a single two-phase region with work-balanced atom slices rounded to 8-atom boundaries and vectorised `simd::exp_fixed_n` batches, while keeping the one-thread energy, force, stress and per-atom bits across all thread counts (including 1, 2, 4, 8 and 16 threads).
  The private split-snapshot test hook is no longer exported from `librgpot`, and an installed tree no longer includes implementation sources such as `MorsePot.cc`.
- `Neighbours.hpp` no longer marks `Neighbours::nextGeneration`
  `__declspec(dllimport)` for consumers on Windows, so a program linked to a
  static `librgpot` resolves it.
  A consumer of the shared library still resolves it through the import library.
- `Neighbours::generation()` is a process-unique identifier that changes at
  construction and on every rebuild, copy and move.
  It comes from one counter exported by `librgpot`.
  The `MorsePot` stored-pair workspace and the `fastTiles` stencil cache are keyed
  on it instead of an object address, so an object built at the address of a
  destroyed one does not reuse that object's pair index, slice splits or stencil.
  Periodic and partially periodic skewed cells are binned in fractional coordinates
  from the internal minimum-image inverse, wrapped on periodic axes and spanned
  on open axes, so a sheared stream with zero skin does not miss pairs across
  cell faces.
  A singular or ill-conditioned cell in fractional binning, or a non-finite
  fractional coordinate, is an error instead of falling back to Cartesian
  binning.
  Orthorhombic cells and fully open skewed cells use Cartesian binning.
  The C++ soversion of `librgpot` is 9.
- `Neighbours` counts the cells of a grid in 64-bit arithmetic and bins a box
  whose grid has more cells than an `int` can name as one cell.
  A periodic box of 3e9 Å with a 1 Å search used to overflow the cell count and
  throw `std::length_error` instead.
- `Neighbours` keeps nothrow default construction and guards the 64-bit
  generation counter against wraparound at `UINT64_MAX`, throwing
  `std::overflow_error` instead of returning the `0` sentinel.
- `RgpotCalculator("mullerbrown")` resolves a cutoff of 0. Muller-Brown is a
  non-periodic surface, so a periodic cell is refused and a free cluster is not
  tiled. A caller-supplied potential still has to publish its own cutoff.
- `RgpotCalculator` now advertises only constructible potential kinds, exposes
  the analytic `TersoffSiC` stress, forwards i-PI driver stress, rejects periodic
  D3/D4 geometries, and is checked against ASE's numerical derivatives and
  standard relaxation tools for every potential type in the build.
- `RgpotCalculator` now tiles a short periodic cell by the cutoff of the
  potential it was given. An `ExprPot` or `SumPot` built from kind strings or
  terms without a `cutoff` key, and a GROMACS or OpenMM engine object, resolved
  to a cutoff of 0 and gave the wrong energy and forces in a cell narrower than
  twice the cutoff. A potential whose cutoff cannot be determined now raises
  `ValueError` when the calculator is built.
- `RgpotRpcCalculator.close` now drops the Python capability handle before tearing down the `kj` event loop, handles already-closed loops and unawaited teardown coroutines, always closes its loop and managed `potserv` process in `finally`, avoids unclosed or blocking subprocess pipes in `spawn`, and never masks an active `with`-block exception in `__exit__`.
- `SuttonChenPot` keys its stored-pair workspace on `Neighbours::generation()`
  instead of the neighbour-list object's address, so a potential constructed at a
  recycled address does not reuse another object's pair index or slice splits.
- `SuttonChenPot` now requires both `total_up <= term_limit` (from `setPairTermLimit(pairs)`) and `pairScratchFits` before entering the three-phase stored OpenMP region, skips `buildStoredHalfIndex` up front when `est_half_pairs > term_limit`, falls back to the threaded two-pass row walk with `PairTerms<3>` capped at `term_limit / lanes` per lane when the three-phase path is skipped, releases the per-call three-phase scratch vectors from the thread-local workspace when the call returns, and includes `SuttonChenConfig`, `PotType::SuttonChen`, and `SuttonChenPot` in the `CxxLayout.hpp` ABI layout stamp (`kCxxSoversion` 8).
- `SuttonChenPot` on a stored neighbour list no longer evaluates each pair from both ends or runs serial embedding and reduction loops between separate OpenMP regions, which made 2 threads slower than 1 thread and capped multi-core scaling.
  When `SuttonChenPot::kTeamFrom` opens a team for the requested thread count, from 2048 atoms on three threads and from 64 (`SuttonChenPot::kTeamMinAtoms`) on four or more but never on two, and the stored list holds at least 4096 candidate upper pairs, `SuttonChenPot` evaluates each upper pair once in a single three-phase `forEachSlicePhase` region over work-balanced atom slices rounded to 8-atom boundaries, computing pair terms once, accumulating densities and embedding weights in parallel, and reducing forces, stress, and per-atom virials in one-thread summation order so every thread count keeps the exact one-thread bits.
- `TersoffSiC` names a missing atomic-number array as a missing species buffer. A null position buffer, cell or force buffer is reported on its own.
- `UmaPot` compiles against torch 2.8 again. Loading a package's constants blob called `update_constant_buffer_from_blob`, which that torch does not have; the call is now made only where the member exists, and a package with a blob reports that the linked torch cannot load it.
- `ZBLPot` on a stored neighbour list keeps the energy, forces, stress and per-atom terms of one thread on every `OMP_NUM_THREADS`, including on arm64.
  The row walk rounds `fscale * d` and then adds that product, as the one-thread walk does.
  `ZBLPot.cc` is built with contraction off, as the other classical kernels are, so a compiler that fuses across statements rounds both walks the same way.
- `aoti_unpack::detail::unique_token` now has hidden symbol visibility so `librgpot` does not export its function-local static symbols (`counter`, `seed` and its guard variable) in Metatomic builds.
- `libpotlibc_engine` and `libcoupled_well_engine` accept the original version-1 create messages (144 and 48 bytes) and the version-2 messages that carry a size and reserved words. `RGPOT_POTLIBC_CONFIG_VERSION` and `RGPOT_COUPLED_WELL_CONFIG_VERSION` now name version 2, so callers that recompile a version-1 struct against the current headers must set `version` to `RGPOT_POTLIBC_CONFIG_VERSION_V1` or `RGPOT_COUPLED_WELL_CONFIG_VERSION_V1` (with `RGPOT_POTLIBC_CONFIG_V1_SIZE` and `RGPOT_COUPLED_WELL_CONFIG_V1_SIZE` for the 144- and 48-byte sizes). Any other version is refused with its own error.
- `librgpot` exports `rgpot_fast_pairs`, `rgpot_fast_wrap`, `rgpot_fast_wrap_axes` and `rgpot_fast_last_error` whenever the fast pair list is built. Before, a shared `librgpot` exported them only when the Fortran water-H kernel was built too, because nothing else in the library called them.
- `meson setup -Dwith_examples=true` builds again: `time_pair_scaling` links the fast pair list, so the water-H kernels no longer leave `rgpot_fast_wrap` undefined.
- `meson setup -Dwith_rust_core=true` now works in a clean build directory. The
  Rust library is built by a target that ninja runs before anything links it, so
  setup no longer needs a prior cargo run. With the fast pair list on,
  `rgpot-core` and the pair list are linked as one archive, `librgpot_bundle.a`,
  so the Rust runtime is defined once. The EAM table free function also catches
  a panic and records it in `rgpot_last_error` instead of unwinding into the
  host.
- `meson setup -Dwith_rust_core=true` without the fast pair list builds again on
  the workspace Rust version, 1.88. The build asked for a 1.89 toolchain only
  the pair list needs, and stopped when rustup had none. That toolchain is now
  used only when the pair list is on. On macOS every Rust archive, with or
  without the pair list, is still written with `/usr/bin/ar`.
- `nlist::build` lists every periodic image inside the cutoff for a one-atom cell and for a periodic cell narrower than twice the cutoff, including an atom's images of itself. A periodic orthorhombic neighbour search folds each coordinate into the cell before Cartesian binning, so an atom outside the box is found with its image inside. Pair kernels still sum one nearest image of each distinct pair. Sheared cells stay on the fractional grid.
- `potserv` now serves the `EAMAl` potential when the Fortran potentials are built, and lists it in its usage message. Before, asking for `EAMAl` failed with an unknown potential type.
- `potserv` usage now names `TBLiteGFN2` and the optional Metatomic device, and lists `CuH2` only when the Fortran potentials are built.
- `rgpot.lammps.FixExternal` now imports `mpi4py` through `importlib.import_module` when LAMMPS runs on multiple ranks and verifies the broadcast error report type, so `mypy --strict` and `pyright` pass without `mpi4py` installed.
- `rgpot::lr::DualList` now builds its half-shell candidate lists through `rgpot::nlist::Builder`, enumerating every periodic image inside the cutoff once (including one canonical direction for self images when the cutoff exceeds a cubic or triclinic cell vector), and filters candidates on the active OpenMP team while skipping the distance pass when coordinates and the cell are unchanged.
  `rgpot::nlist::Builder::update` checks both signs of image shifts and measures folded coordinates against the nearest reference image, so variable-cell and NPT steps within the skin reuse the stored candidate table instead of rebuilding on every box change.
  D3 and D4 now restore the opposite directed neighbor for each canonical periodic self image, preserving their energies, coordination numbers and forces; D3 also restores self-image stress.
- `rgpot::lr::DualList` rebuilds after an update with no atoms, or one molecular atom. Passing the earlier atoms again used to leave both lists empty.
- `rgpot::lr::DualList` rejects an infinite or NaN skin with `std::invalid_argument` ("skin must be finite and non-negative"), as it already did for a negative skin.
- `rgpot::nlist::build` and `buildByAtom` now leave the previous table in place when they reject a periodic coordinate that is not finite or lies more than 2^29 cells from the origin, as the header promises for a rejected geometry. Both cleared the table before that check and left it empty.
- `rgpot_engine_taylor_jet` keeps the caller's jet and derivative report unchanged when the evaluation fails.
- `scripts/export_uma_aoti.py` now compiles each `.pt2` archive inside an isolated `TORCHINDUCTOR_CACHE_DIR`, preventing concurrent exports of identical AOTInductor graphs from overwriting each other's embedded package metadata.
- `scripts/run_coverage_fortran.sh` again enables branch coverage with `lcov_branch_coverage=1` on lcov 1.x and `branch_coverage=1` on 2.x, matching `scripts/run_coverage_cpp.sh`, so the Fortran Codecov upload on Ubuntu 22.04 keeps BRDA data instead of line hits only.
- lammpc and LAMMPS engine loading errors now omit an empty engine path prefix,
  so wrapped failures read `lammpc: ...` instead of `lammpc: : ...`.
- RgpotCalculator raises PropertyNotImplementedError for stress and per-atom
  stresses when the cell has fewer than three nonzero lattice vectors, matching
  ASE's LennardJones and EMT calculators. Energy and forces are unchanged.
- The in-process ASE calculator follows the ASE calculator contract on five
  points. calculate_many omits stress when the cell has fewer than three
  nonzero lattice vectors, and raises PropertyNotImplementedError if stress
  was requested. numerical_stress and stress_step are calculator parameters:
  set updates them and todict round-trips a non-default value. calculate
  with no properties list fills every implemented property. ase.db stores
  the name rgpot. An empty system_changes list replaces results, so a key
  this call did not produce does not remain.
- The plain text builder copies backtick code through unchanged and
  keeps a marker that belongs inside a code span. A slash, a plus or a
  star still ends a span, and math is left unchanged. The check reports
  an inline code span from the sources that the plain text changed.

### Miscellaneous

- Restored the develop CI checks across Meson formatting, link validation, LeakSanitizer libomp symbolization, lcov 2 coverage capture, isolated readcon golden environments, optional `mpi4py` typing, wheel EAM table fixtures, ThreadSanitizer container execution, benchmark analysis timeout headroom, and synchronized `MPI_Abort` error flushing across calculator ranks.


## [3.7.0] - 2026-11-17

[3.7.0]: https://github.com/OmniPotentRPC/rgpot/tree/v3.7.0

### Breaking

- C++ member functions of the potentials now use camelCase, like the
  other members of those classes.
  Rename the calls; no old name remains.

  - `PotentialBase::get_type()` is `PotentialBase::getType()`.
  - `PotentialBase::set_cache()` and `Potential::set_cache()` are
    `setCache()`.
    An override in a derived class must use the new name.
    The Python method stays `set_cache`.
  - `CP2KPot::probe_available()`, `CPMDPot::probe_available()`,
    `MOPACPot::probe_available()` and `NWChemPot::probe_available()` are
    `probeAvailable()`.
  - `CPMDPot::abi_available()`, `MOPACPot::abi_available()` and
    `NWChemPot::abi_available()` are `abiAvailable()`.
  - `lr::CoulombPot::set_neighbours()`,
    `lr::DispersionPot::set_neighbours()` and
    `lr::NlhPot::set_neighbours()` are `setNeighbours()`.
  - `lr::SumPot::share_neighbours()` is `shareNeighbours()`.
  - The namespace `rgpot::details` in `include/rgpot/errors.hpp` is
    `rgpot::detail`, like every other internal namespace.

- `RGPOT` no longer takes `U0`, `CUTOFF` and `PSI` in PLUMED units.
  Write `RGPOT ATOMS=... POTENTIAL=LJ OPTIONS={u0 1.0 psi 1.0 cutoff
  8.0} TYPES=Ar` instead, with the options in eV and Angstrom:
  `U0=96.48533212331002 CUTOFF=0.8 PSI=0.1` under GROMACS units is
  `u0 1.0 cutoff 8.0 psi 1.0`.
  `TYPES` is required.

### Added

- `StillingerWeberSi::forceImpl` writes the Cauchy stress and
  `caps().stress` is true.
  The stress is (1/V) dE/dε in eV/Angstrom^3, with the sign and layout
  of the `LJPot`, `MorsePot` and `ZBLPot` stress.
  The virial sums each folded bond vector times the energy gradient with
  respect to that vector.
  That sum is the many-body virial of Thompson, Plimpton and Mattson,
  J. Chem. Phys. 131, 154107 (2009).
  Its digital object identifier (DOI) is 10.1063/1.3245303.
  Energies and forces keep their bits.
  `SWSiTest` now also runs under CMake.

- The Fortran `SWPot` (`PotType::SWSi`) writes the Cauchy stress and
  `caps().stress` is true.
  It sums the same bond-vector virial as `StillingerWeberSi`, so its
  stress is the `StillingerWeberSi` stress times the epsilon ratio
  1.068176741337.
  Its energies and forces keep their bits.
  `SWPot::forceBatch`, the call eOn's engine makes, skips the virial and
  leaves `has_stress` at 0.
  In Python, `StillingerWeberSi` gains `caps()` and `evaluate()`, which
  returns the stress, so `RgpotCalculator` reports the Atomic Simulation
  Environment (ASE) stress for it and for `SWPot`.

- `rgpot_lammps_potential`, `rgpot_gromacs_potential` and
  `rgpot_openmm_potential` are the public constructors in C++, the C ABI
  and Python.
  LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator)
  dlopens the engine plugin.
  GROMACS and OpenMM store the nanometre and kJ/mol conversion once.

- The engine drivers are `pair_style rgpot`, a PLUMED `RGPOT` collective
  variable used by `gmx mdrun -plumed`, an OpenMM shifted Lennard-Jones
  comparison, and an `rgpot_engine_force` Lennard-Jones plugin.
  Each one is a meson test when its library is present.

- UmaPot evaluates a same-composition stencil in chunks of the package
  `batch_max` and keeps the per-system loop when `batch_max` is 0 or 1.
  A static band (`--batch-style static`) is traced at exactly
  `batch_max` systems, and a short chunk repeats its last system.
  `--batch-style packed` is one disjoint graph with a dynamic atom count
  and a dynamic edge count (`batch_packed`).
  UmaPot sends the real system count and does not repeat a system.
  `scripts/export_uma_aoti.py --batch-max N --molecular-box L` traces
  either graph.
  The force gradient is traced on torch builds that gate
  `torch.autograd.grad`, and `OMP_NUM_THREADS` is applied to torch
  before the compile because the CPU general matrix multiply (GEMM)
  kernel is built for that thread count.

- `scripts/cpu_scaling.py` writes strong, one-cell and weak CPU scaling
  tables as comma-separated values (CSV).
  Each CSV header names the host, the compilers and the build options.
  Every row is checked bit for bit against a 1-thread call.
  It runs only when called and changes no library behaviour.

- Free-cluster Cambridge Morse and reduced Lennard-Jones kernels,
  `rgpot_cambridge_morse` and `rgpot_cluster_lj`.
  Both sums run over every pair: a neighbour list would drop a tail that
  is not zero.
  Energy and gradient are one pass.

- Free-cluster Cleri–Rosato kernel `rgpot_gupta`.
  The sum visits every pair: a neighbour list would drop a tail the
  Cambridge minima include.
  The density is accumulated once and the force reuses that pair
  derivative.

- Hessian-vector products for any potential, for minimum-mode searches
  (Lanczos, dimer) that apply the Hessian to many directions at one
  geometry.
  In the C ABI, `rgpot_potential_hvp` computes `H v` on a potential
  handle.
  `RGPOT_HVP_FORWARD`, the default, reuses the forces of the last centre
  the handle evaluated, so k products at one geometry cost k + 1 force
  calls instead of 2k, at first order in the step.
  `RGPOT_HVP_CENTRAL` costs two calls a product at second order.
  A potential that knows its Hessian registers an analytic product with
  `rgpot_potential_set_hvp_callback` and the products make no force
  call.
  `rgpot_potential_hvp_centre` returns the centre energy and forces from
  the same cache, `rgpot_potential_hvp_invalidate` drops it, and
  `rgpot_potential_hvp_counts` reports the calls made.
  The centre is keyed on the bits of the positions, atomic numbers and
  cell.
  In C++, `PotentialHandle` gains `hvp`, `hvp_centre`,
  `set_hvp_callback`, `invalidate_hvp` and `hvp_counts`, and
  `rgpot/HessianVector.hpp` gives the same products to potentials used
  without the Rust core.
  `MorsePot::hessianVectorImpl` is the analytic product of the shifted
  and switched Morse potential, and `PotentialHandle::from_impl`
  registers it.
  In Python, every potential gains `hvp`, `hvp_centre`,
  `invalidate_hvp`, `hvp_counts`, `reset_hvp_counts` and
  `has_analytic_hvp`.

- Long-range summands behind `-Dwith_long_range=true`: Ewald, smooth
  particle mesh Ewald and damped shifted force electrostatics, D3(BJ),
  D4, MBD and QDO dispersion, SO3LR NLH repulsion, and a sum that adds
  them to any other potential on the same `ForceInput`.

- The PLUMED action `RGPOT` evaluates any potential `pair_style rgpot`
  accepts: `POTENTIAL=Name OPTIONS={key value ...} TYPES=El,...`, with
  the names and options of `pair_coeff` (now listed once in the
  potential names reference).
  Positions, the cell, the value and its derivatives follow PLUMED's
  units, converted with its `Units` object.
  Periodic cells, including triclinic GROMACS boxes, go to the
  potential, and the box derivatives come from its stress, so
  `BIASVALUE` under `gmx mdrun -plumed` gives the right virial and
  pressure in NPT runs.
  A potential without a stress stops on a periodic cell; `NOPBC`
  evaluates an isolated cluster.
  New meson tests check the forces and box derivatives against
  `plumed driver --debug-forces`, the units, and the C and Python
  interfaces, and `gromacs_mdrun` now also checks the bias, virial and
  pressure of a short Sutton-Chen NPT run.

- The header `rgpot/Staged.hpp` lets a host that builds its own
  neighbour list, such as LAMMPS, evaluate a finite-range potential on
  that list in two stages, the scheme of `pair_style eam`:
  `densityStage` sums each atom's densities, the host completes them
  across its copies of neighbouring atoms (`Locality::densities` values
  per atom), and `forceStage` adds the forces.
  `HalfList` takes a half list in the layout of a LAMMPS `NeighList`,
  with newton on or off, and `StageTally` takes the host's share of the
  energy and virial and the per-atom split.
  `LJPot`, `MorsePot`, `SuttonChenPot` and `EAMTablePot` have the
  stages, the concept `rgpot::StagedPotential` detects them, and
  `rgpot::spec::Built` carries them for a potential built by name.
  `PotCaps` and the C++ layout are unchanged.

- `CP2KPot` runs CP2K Quickstep through `libcp2kc_rgpot`, a shim over
  libcp2k built with `-Dwith_cp2k=enabled`.
  `CP2KParams` (`PotentialConfig.cp2k`) renders a default GPW deck or
  fills `@RGPOT_COORD@` and `@RGPOT_CELL@` in a verbatim template.
  The shim keeps one force environment per orbital key, and an
  unconverged SCF raises `rgpot::CP2KScfError` on every rank instead of
  aborting the MPI world.
  potserv serves it as `CP2K`.

- `LJPot`, `LJClusterPot` and `MorsePot` gain `evaluate_per_atom`, which
  returns everything `evaluate` does plus per-atom `energies` (shape
  `(N,)`, eV, summing to the energy) and `virials` (shape `(N, 6)`, eV,
  Voigt order, summing to minus the volume times the stress).
  It exists only on potentials whose C++ kernel has `forcePerAtom`.

- `compute pe/atom`, `compute stress/atom` and
  `compute centroid/stress/atom` work with `pair_style rgpot` for LJ,
  Morse, SuttonChen and EAMTable, on any number of ranks; they match
  `lj/cut`, `morse` and `eam/alloy` atom by atom.
  `LJPot`, `LJClusterPot`, `MorsePot`, `ZBLPot`, `SuttonChenPot` and
  `EAMTablePot` gain `forcePerAtom`, which also writes per-atom energies
  (pair terms split half to each atom, EAM and Sutton-Chen embedding to
  its own atom) and per-atom virials; `rgpot::PerAtomPotential` in
  `rgpot/PerAtom.hpp` detects it.
  No virtual or data member changes, so the C++ layout is unchanged.
  Potentials without the split stop the run with an error naming the
  potential.

- `librgpot_gromacs` (`-Dwith_gromacs_dropin=enabled`, off by default)
  serves any rgpot potential through the libcp2k C API of the GROMACS
  QM/MM interface.
  A double-precision GROMACS 2024 to 2026 built with `-DGMX_CP2K=ON`
  against it (`engines/gromacs/build_gromacs.sh`) evaluates the QM group
  in `gmx mdrun` without PLUMED, with the energy in GROMACS'
  `Quantum En.` term.
  An `&RGPOT` section of the `grompp -qmi` file chooses a `pot_spec`
  name and options or an `rgpot_engine_*` library and its configuration;
  elements come from the topology.
  Periodic (including triclinic) and isolated QM regions work; there is
  no virial, so NPT is not supported, and MM charges do not act on the
  QM region.
  Bad input stops `mdrun` with an `rgpot-gromacs:` message.
  New meson tests `gromacs_dropin_api` and `gromacs_dropin_mdrun`
  compare with `rgpot-eval` through real `mdrun` runs, and
  `engines/gromacs/bench_dropin.py` measures the per-step cost.

- `pair_style rgpot` evaluates finite-range potentials by domain on more
  than one rank, the way `pair_style eam` does.
  For LJ, the `u0 psi` form, Morse, SuttonChen with a positive cutoff
  and EAMTable, each rank runs the density and force stages of the
  potential on the half neighbour list of LAMMPS, whose ghosts reach one
  cutoff plus the skin, and the densities of a many-body potential pass
  between the ranks between the two stages, so the force call no longer
  runs on rank 0 alone.
  On one rank with `mode domain`, the 16384-atom Lennard-Jones box of
  `bench_pair_rgpot.py` takes 8.27 ms per step (`lj/cut` 8.38, the
  gather path 15.10) and the 32000-atom CuNi box 46.84 ms (`eam/alloy`
  44.47, the gather path 70.10).
  Energies, forces, the global virial, `pe/atom` and `stress/atom` match
  the gather path and `lj/cut`, `eam/alloy` and `morse` on 1, 2 and 4
  ranks, under `newton on` and `off`, including ranks that own no atoms.
  Engines, potentials without a finite range or the two stages,
  `virial fd` and one-rank runs keep the gather on rank 0.
  The new `pair_style rgpot cutoff mode auto|gather|domain` keyword
  forces either path, and `init` logs which one runs and why.
  In C++, `rgpot/Locality.hpp` adds `rgpot::Locality` and the concept
  `rgpot::LocalPotential`, a non-virtual `locality()` on `LJPot`,
  `MorsePot`, `SuttonChenPot` and `EAMTablePot`; `PotCaps` and the C++
  layout are unchanged.

- `pair_style rgpot` works as a native LAMMPS pair style.
  It fills the global virial from the potential's stress, or from a
  strain difference when there is none, so pressure, NPT and
  `fix box/relax` work in orthogonal and triclinic boxes.
  `pair_coeff * * Name [key value ...] El ...` selects an rgpot
  potential by name and maps types to elements.
  MPI runs gather the box on rank 0.
  Other unit styles stop with an error.
  `rgpot-eval` evaluates a named potential on one geometry.

- `rgpot.ase.RgpotCalculator` accepts any potential class the bindings
  expose, by class name or kind string (`"morse"`, `"ljcluster"`,
  `"tersoffsic"`, ...; `rgpot.ase.available_kinds()` lists them), and
  passes keyword arguments to that class's constructor.
  It reports per-atom `energies` and `stresses` for LJ, LJ-cluster and
  Morse, offers a finite-difference `stress` for kernels without one
  (`numerical_stress=True`), rebuilds the potential on `calc.set(...)`,
  and pickles, `copy`s and `deepcopy`s by rebuilding the potential from
  its parameters.
  It matches ASE's `LennardJones`, `MorsePotential` and an analytic
  `EAM` for EAM-Al to 1e-10 relative, and runs under the ASE optimizers,
  cell filters, Velocity Verlet, NEB, `Vibrations` and `SumCalculator`.

- `rgpot::IPIDriver` and the `rgpot-ipi-driver` program are the client
  side of the i-PI socket protocol (built with `with_ipi`).
  They connect to i-PI, ASE's `SocketIOCalculator` or `IPIPot` over a
  UNIX socket or TCP and answer `STATUS`, `INIT`, `POSDATA`, `GETFORCE`
  and `EXIT` with any potential `engines/pot_spec.hpp` builds, returning
  hartree, hartree/bohr and the virial `-sigma V`.
  Open clusters get a padded box, thin periodic cells are tiled to the
  cutoff, and `--stress-step` gives a finite-difference virial for
  kernels without stress.
  The flags follow `i-pi-py_driver`.
  `rgpot-ipi-bench` times the socket round trip against a direct call;
  `docs/orgmode/howto/ipi.org` now covers both directions and lists the
  timings.

- `scripts/export_uma_aoti.py` exports a dynamic UMA package by default:
  the atom count (2 to `--max-atoms`, default 1024) and the edge count
  (1 to `--max-edges`, default 300 per atom) are symbolic, so one
  package keeps working when an NEB image or a dimer step changes the
  neighbour list, and, with the MoLE experts kept in the graph, for
  other molecules of the same task head.
  The package is kept only after it matches `FAIRChemCalculator` on the
  input, on the input with one atom moved past the cutoff, and on a
  second molecule.
  The checkpoint can come from `--checkpoint` or `RGPOT_UMA_CHECKPOINT`.
  `UmaPot` refuses an atomic number above the package's `z_max` before
  the compiled graph runs.

- `scripts/export_uma_aoti.py` resolves the checkpoint once
  (`--checkpoint`, `RGPOT_UMA_CHECKPOINT` or a Hugging Face download)
  and records it in the package: `checkpoint_source` (`cli`, `env` or
  `hf`), `checkpoint_path` or `checkpoint_hf_repo` and
  `checkpoint_hf_filename`, the resolved commit as
  `checkpoint_hf_revision`, and `checkpoint_sha256` of the file it
  loaded.
  `UmaPot::checkpointProvenance()` and the exporter's
  `checkpoint_provenance()` read them back; packages exported earlier
  load as before and report the keys absent.

### Developer

- The benchmark description states the comparison in plain words.
  The upper bound is at least the widest calibration ratio.
  The comparison that fails on a slowdown is the one to run.
  Continuous integration should use that comparison's exit status.

- The C ABI page lists the public C functions.
  The coverage check fails when one of them is missing from that page.
  The ABI baseline keeps the neighbour and fast-list exports.

- The develop continuous integration job builds a serial LAMMPS with the
  PLUGIN package from the same headers as `pair_style rgpot`.
  It then requires the LAMMPS, GROMACS, OpenMM and PLUMED tests to pass.

- Configuring the engine drivers accepts an include path that lies
  inside the source tree.
  `include_directories()` rejects that absolute path, so a pixi prefix
  or a LAMMPS checkout under the workspace could not configure.
  The path is made relative to `engines/meson.build`.
  A path outside the tree stays absolute.

- CMake's `nlist_rust` test passes the script the OpenMP libraries
  FindOpenMP chose, and the runtime's directory as an rpath, so the
  shared library the script links finds the OpenMP runtime with
  AppleClang, which has no `-fopenmp`.
  Called without them, as meson calls it, the script uses `-fopenmp`
  only if a test program links with it, so a build without OpenMP no
  longer fails where clang takes the flag but finds no libomp.

- Every CI job that runs on a runner sets `timeout-minutes`: its
  measured p95 plus a margin that covers the longest per-test timeout it
  runs, so meson or CTest still kills and names a hung test before the
  job times out.
  `ci/gha/README.md` gives the rule and the numbers, and shows how to
  tell a job no runner took, which a timeout cannot catch, from one that
  hung.
  The reusable `isa-audit.yml` takes a `timeout-minutes` input, 10 by
  default.

- Every OpenMP region in the library now opens in
  `CppCore/rgpot/Slices.hpp`, through `forEachSlice`,
  `forEachSlicePhase` or `onFullTeam`, so a result depends on its slice
  count and not on how many threads the runtime starts.
  `CppCore/tests/test_openmp_confined.py` fails the test suite when
  OpenMP appears in any other library source, `ThreadTeamTest` checks
  the threaded potentials and neighbour tables on short, dynamic and
  nested teams, and the `thread_limit` CI job runs the suite under
  `OMP_THREAD_LIMIT=1` and `OMP_THREAD_LIMIT=2`.
  The long-range summands follow the same rule, so particle mesh Ewald,
  D3 and D4 no longer stall or skip work when the runtime starts fewer
  threads than they ask for.

- The C ABI stamp of `rgpot-core` moves to 1, 2 and 1: the
  Hessian-vector entry points are additions, and `rgpot_potential_t`
  keeps its layout.
  `scripts/abi_core_baseline.txt` lists the `rgpot_*` functions the
  `rgpot-core` shared library exports.
  The meson test `abi_core_symbols` fails when one of them is no longer
  exported.
  `scripts/abi_symbols.sh` takes the name pattern with `-p`, ahead of
  the library and its baselines.

- The ISA audit proves dispatched clones from the linked library instead
  of reading them from `rgpot/isa-allow.toml`.
  A local function reached only behind a `__builtin_cpu_supports` test
  or from a `target_clones` resolver may use the features that test
  covers, so the AVX2 Cambridge Morse, cluster Lennard-Jones and Gupta
  bodies pass the wheel audit with no entry.
  AVX in any other function is still a violation, and the report lists
  each clone with its dispatcher.

- The ISA self-test links its GCC dispatch fixtures with
  `-nodefaultlibs`, so they import libgcc's CPU model and the proof
  reads `__cpu_model` through the GOT.
  The static copy is built for the toolchain's own `-march`, and on the
  Ubuntu 26.04 runner its `__cpu_indicator_init` has two BMI2 `shlx`.
  Clang reads `__cpu_model` PC-relative, so its fixtures keep the copy.

- A clone the ISA audit proves for `avx512_vbmi2` or `avx512_bitalg`
  may also use AVX512BW, which `-mavx512vbmi2` and `-mavx512bitalg`
  enable from GCC 14, as in Clang.
  `test_dispatch.py` still checks the implications exactly against GCC
  12 and 13, where those options enable only AVX512F.

- The `abi_symbols` test picks the `nm` mode from the library it is
  given: `nm -D --defined-only` for a shared object and
  `nm -g --defined-only` for a static `lib*.a` or `*.lib` archive, which
  has no dynamic symbol table.
  macOS still uses `nm -gU`.
  A small fixture built as a shared object and as both archive kinds
  runs every mode in every `with_tests` build, and checks that a shared
  object's hidden names and an archive's missing names still fail.

- `CppCore/rgpot/rpc/Potentials.ordinals` records every field, union
  arm, enumerant and method ordinal of `Potentials.capnp` with its name
  and type, append only.
  `schema_ledger.py check` (meson suite `schema`, CI job
  `schema ordinal ledger`) holds the pinned `potentials-schema` commit,
  the local series diffs and both vendored copies to it, and refuses a
  series diff that removes or rewrites an `@N` line or adds an ordinal
  that is not a `local` ledger entry.

- `tests/test_expr_params.py` checks that `PotentialConfig.expr` is the
  upstream `ExprParams` arm at @16, as `test_rpc_schema_sync.py` does,
  instead of refusing it.

- `CppCore/tests/registry.txt` lists every test with the features that
  gate it, its kind and its options, and meson and CMake both register
  their tests from it.
  CMake now registers each test meson does except those only meson can
  run, whose lines give the reason: in the build the `thread_limit` CI
  job tests, that is 99 of meson's 107 tests, where CMake registered 31.
  CMake's unit tests link OpenMP as meson's do, and its ABI symbol tests
  run their script through `sh`.
  `CuH2PotTest`, `CapnpAdapterTest` and `BridgeStressTest` take meson's
  names, `CuH2test`, `RPCTest` and `RPCClientTest`.
  Each build registers `test_list_parity`, which fails when the build's
  tests or their options differ from the registry.
  `tools/check_test_parity.sh` configures the plain, the RPC and cache,
  and the client-only builds with meson and CMake, without building
  them, and fails when the two register different tests.
  The `test_parity` CI job runs it, and `ci_gate` waits on that job.

- `EAMCellVerletTest` requires a NaN wherever the direct pair loop gives
  one, whatever its sign bit, and every other value bit for bit.
  IEEE 754 does not specify the sign of a NaN result, and optimised
  Clang builds flip it for the forces between coincident atoms, so the
  test failed in the macOS CMake builds and passed in the unoptimised
  meson builds.

- `docs/doccheck.py` checks the documentation against the source.
  It requires a reference entry with Inputs, Units, Outputs and Errors
  for every public C, C++ and Python name, and a classification for
  every code block in the README and the docs.
  The C names include the Metatomic and XTB engine tables and the
  libcp2k API of the GROMACS drop-in.
  The entry of each `*Config` struct must name every field, a reference
  page must name every field the rgpot-local schema diffs add, and the
  `&RGPOT` key table must list the keys the drop-in reads.
  It runs, compiles or statically checks each block according to that
  classification, and flags option names, paths and `RGPOT_*` names that
  do not exist; paths that `.gitignore` covers, such as `docs/build`,
  count as build outputs.
  `pixi r -e docs doccheck` runs it, and the docs CI job runs it with
  the Sphinx build, which now uses `-W --keep-going`.
  It replaces `docs/check_c_symbols.py`; the engines CI job no longer
  asks for the `c_symbols` test.

- `potctl ci cmake-test` and `potctl ci bridge-client-build` give
  `cmake --build` an explicit `--parallel` count:
  `CMAKE_BUILD_PARALLEL_LEVEL` when it is set, the usable CPUs
  otherwise.
  A bare `-j` let make start every ready compile at once, and with the
  whole test registry to build, the macOS CMake legs (3 cores, 7 GB)
  spent 7 to 49 minutes compiling, where the meson legs of the same run
  built and tested in 3 to 6.

- `rgpot::simd::exp_hw_n` is declared on every target.
  MSVC, Arm and other builds without the x86-64 hardware body get one
  that calls `rgpot::simd::exp`, so a caller that picks the body at run
  time compiles everywhere.

- `sc_call_bench` (`meson test --benchmark sc_call_bench`) times
  Sutton-Chen calls on free clusters of 13 to 561 atoms with no cutoff:
  the C entry `rgpot_sutton_chen_energy_forces`, a `SuttonChenPot` built
  per call, one kept across calls in the default and the streaming
  neighbour mode, and a plain half-pair loop.
  Each mode also prints its largest energy and force gaps to that loop
  and a hash of the energy and force bits, so two builds can be compared
  bit for bit.

- `tools/check_test_parity.sh`, and so the `test_parity` CI job, also
  reads the `compile_commands.json` meson and CMake write when
  configured, and fails when a source both compile gets different
  contraction or fast-math flags from each.
  `tools/check_fp_parity.py` does the comparison.

### Changed

- A stored `nlist::Neighbours` list is kept when the caller wraps atoms
  back into the periodic cell.
  The half-skin test measures each atom's move to the nearest image of
  its reference position, so an atom that crossed a face no longer
  counts as having moved a cell vector, and the bins are redone after
  such a wrap.
  The potentials that keep such a list (`LJPot`, `LJClusterPot`,
  `MorsePot`, `ZBLPot`, `SuttonChenPot`, `EAMPot`, `StillingerWeberSi`,
  `TersoffSiC` and `EAMTablePot`) rebuild it less often on wrapped
  input: on the gather path of `pair_style rgpot`, 3 instead of 40
  builds in 112 calls for 16384 Lennard-Jones atoms.
  One thread walks the pairs in index order, so its results keep their
  bits; with `OMP_NUM_THREADS` above 1 the order follows the bins and
  results agree to rounding.
  `nlist::Builder`, whose table stores image shifts, is unchanged.

- An infinite cutoff or skin in `rgpot::nlist::Neighbours` walks every
  pair straight from the coordinates, with no bins and no stored list,
  `i` ascending and then `j` ascending, and keeps the update for any
  move while the atom count, the cell and the spec stay the same;
  `allPairs()` reports it.
  That walk takes a copy of its callback, so a caller reads back only
  the state it captured by reference.
  `SuttonChenPot` with `cutoff <= 0` searches that way.
  It used to stand in a 1e6 search, which binned a cluster into one
  cell, read `MemAvailable` twice and stored and sorted every pair on
  each call.
  Energies and forces keep their bits on one thread.
  A team of threads used to leave that one cell to one thread and now
  splits the atoms into runs, one per thread: the forces keep their
  bits, and the energy, a sum over threads, can move in the last bits.

- `EAMTablePot` keeps its pair list between force calls, with a Verlet
  skin of `EAMTableConfig::skin` (1 Angstrom by default, `skin` in
  `pair_coeff * * EAMTable ...` and the other hosts of
  `rgpot::spec::build`), instead of building a list with no skin on
  every call.
  Energies, forces, stress and per-atom terms keep their bits for any
  skin.
  The spline tables were and are made once, in the constructor.
  Because the list is per-instance scratch, `caps().reentrancy` is now
  `Reentrancy::PerInstance`: a caller that evaluates one `EAMTablePot`
  from several threads at once needs one instance per thread, as for
  `LJPot`.
  On the gather path of `pair_style rgpot` a 32000-atom CuNi step went
  from 338.63 to 70.74 ms on one rank, with one list build in 62 calls
  (`docs/orgmode/howto/engine-lammps.org`).

- `EAMTablePot` locates each pair distance on its radial grid once per
  pass.
  It located it once per table before: twice per pair in the density
  pass and three times in the force pass.
  It also keeps the three radial tables of each element pair together
  instead of indexing nested vectors for every pair.
  The arithmetic is the same, so the results do not change.
  On one rank with `mode domain`, a 32000-atom CuNi step of
  `pair_style rgpot` went from 54.13 to 47.03 ms, against 43.87 ms for
  `eam/alloy`.

- `SuttonChenPot` keeps the powers and `1/r^2` that the force of each
  pair takes from the density pass, and the force pass reads them back
  instead of raising the distance to `n` and `m` a second time.
  `setPairTermLimit` bounds the store, 32 MiB by default, and the force
  pass computes the pairs past it again.
  The pair lambdas inline into the walks and take their constants by
  value, and the density and weight arrays are kept with the instance.
  `SuttonChenPot.cc` is built with `-ffp-contract=off`, as the other
  kernels are, so reading a term back and computing it again round
  alike.
  Energies, forces, the stress and the per-atom split keep their bits
  wherever the old build fused no multiply into an add.

- `potentials-schema` is pinned at 190fee6, upstream main 7fa9046 plus
  the CP2K arm, and both vendored `Potentials.capnp` copies are that
  file byte for byte.
  `PotentialConfig` gains the upstream arms `xtb` @6, `tblite` @7, `uma`
  @8, `lj` @9, `morse` @10, `zbl` @11, `d3` @12, `d4` @13, `mopac` @14,
  `skala` @15, `expr` @16 and `cp2k` @17, with their parameter structs,
  so rgpot numbers them as gpridge and every other consumer does.
  `BuiltinParams` gains `config` @2, `PotentialSpec` gains `cp2k` @6,
  `PotSpec` gains `potential` @2 and `MetatomicParams` gains
  `lengthUnit` through `attachSystemExtras` (@13 to @18).

- `rgpot_sutton_chen_energy_forces` and `rgpot_lj_eval` keep the
  potential of their last call on each thread instead of building one
  per call, so the next call with the same parameters and atom count,
  and for `rgpot_lj_eval` the same atomic numbers, reuses its neighbour
  update and scratch.
  Any other call, and one that throws, drops it and builds a new one.
  Energies and forces keep the bits of a potential built for the call.
  `rgpot::Kept` in `rgpot/KeptPotential.hpp` holds one potential per
  thread and potential type, keyed by the bits of its parameters.
  With this and the unbounded search and kept pair terms, a Sutton-Chen
  12-6 call on a free 55-atom cluster with no cutoff went from 60.901 to
  13.690 microseconds on one core (`sc_call_bench`,
  `docs/orgmode/reference/sutton_chen.org`).

### Fixed

- UmaPot treats a package with neither `nedges_min` nor `nedges_max`
  as fixed-shape when `shapes` records `edge_index`.
  The second extent is the traced edge count, and the neighbour list
  must match it.
  A published gpridge export records that list, for example
  `edge_index` `[2, 1332]`, and it used to be refused.
  A package with only one of the two keys is still refused.

- A debug or sanitizer build no longer aborts when a vesin neighbour
  list with no pairs is reset or recomputed.
  The reset had called memset on a null pairs pointer, which is
  undefined even when the byte count is zero.
  It now poisons only the storage the list allocated, and it stops at
  capacity.

- On a cell narrower than three cutoffs, `rgpot_fast_pairs` answers a
  short buffer from n(n-1)/2 and fills the caller's buffer directly.
  For the 10k-atom bridge cell, it no longer builds a 400 MB pair list
  just to report the count.
  A short-buffer return also clears the thread-local error message.

- `RgpotCalculator` reports the stress of a non-periodic structure as
  the cluster virial divided by the caller's cell volume, so it matches
  ASE's finite-difference stress.
  A non-periodic evaluation still uses a padded cube so images stay
  outside the cutoff.
  ASE's finite-difference stress divides the same virial by the caller's
  cell, so the cube stress is rescaled by V_cube / V_caller.

- The C++ soversion row is found by index.
  A pointer into `kCxxSoversionTable` is not a constant expression under
  `-fsanitize=undefined`, so the layout static asserts failed that
  build.
  The recorded hash is unchanged.
  `tools/check_cxx_layout_ubsan.sh` configures the address and undefined
  sanitizers and runs `CxxLayoutTest` and `CxxLayoutVirtualCounts`.

- The nine doubles stored in `rgpot_potential_t` are lattice vectors as
  rows, the same layout as `ForceInput.box`.
  The header comment had called that matrix column-major.

- The LAMMPS and OpenMM configure probes ignore `engines/lammps` and
  `engines/openmm`.
  Those directories sat on the interpreter's path, so the probes
  reported the modules present and the meson suite ran tests that need
  the libraries.

- A CMake build compiles the core library sources and the reduced pair,
  Lennard-Jones, Sutton-Chen and Tersoff potentials without
  floating-point contraction, as meson does.
  On arm64, where the compiler fuses `a*b+c` into one multiply-add by
  default, the CMake library's results differed from meson's in the last
  bits, and `ReducedPairTest` and the skewed-cell case of
  `FastNlistTest` failed on macOS.

- A CMake build with the cache links the shared RocksDB library when
  there is one, as meson does.
  conda-forge's static RocksDB put jemalloc ahead of libc in every
  program that links `librgpot`, and such a program crashed in `free`
  once it loaded `libmpi`, as the MPI calculator group and collective
  RPC tests do.

- A GFN2 single point that does not converge at the configured
  electronic temperature is retried with smeared occupation and
  evaluated again at that temperature.
  The returned energy is the cooled one.

- A meson build with `with_eigen=true` configures again when neither
  pkg-config nor CMake finds `eigen3`.
  The lookup passed `dependency()` a `subproject:` keyword, which
  current meson rejects; it now names the fallback
  `['eigen', 'eigen_dep']`, so Eigen comes from
  `subprojects/eigen.wrap`.

- A neighbour walk split across more slices than the OpenMP runtime
  started threads now walks every slice.
  `nlist::Neighbours::forCellTeam`, the threaded `nlist` builds and
  `simd::fill_view` handed slice `t` only to thread `t`, so a build
  without OpenMP, a call inside another OpenMP region or a run under
  `OMP_THREAD_LIMIT` dropped the pairs of the missing threads: with
  `OMP_NUM_THREADS=3` and `OMP_THREAD_LIMIT=2`, the threaded Sutton-Chen
  energy came out wrong, and the macOS meson build, which has no OpenMP,
  failed `NeighbourLayerTest`.
  `nlist::forEachSlice` runs the slices on the threads the runtime
  starts, so a sum taken in slice order is the same for any team.
  A build inside another OpenMP region is no longer ruled out.

- A project that vendors rgpot with `subproject()` no longer installs
  `rgpot-lj` and the engine plugins from `engines/`.
  They are installed only when rgpot is the top-level project, like the
  rest of rgpot, so `meson install` in the parent no longer stops on an
  engine output the parent never built.

- A threaded pair-list `build` (`OMP_NUM_THREADS` above 1, at least 8192
  atoms) now stores the same pairs as one thread when a pair has more
  than one image inside the cutoff, as in a periodic box thinner than
  two search radii.
  Before, it kept only one image of each pair and lost the rest.

- Both vendored `Potentials.capnp` copies are again the pinned
  `potentials-schema` 190fee6 file byte for byte, so
  `schema-sync-canonical-rpc` and `schema-sync-canonical-core` pass when
  a parent such as gpridge resolves `potentials-schema` from its own
  wrap.
  The rgpot-local neighbour fields (`ForceInput.neighbors` @9,
  `PotentialResult.neighborList` @28, `Potential.neighbours` @13 and the
  `NeighborRequest`, `NeighborSystem` and `NeighborList` structs) moved
  from the wrap's `diff_files` to the add-only series
  `CppCore/rgpot/rpc/local/series`, mirrored in
  `rgpot-core/schema/local` for the crate.
  Meson, CMake and `rgpot-core/build.rs` apply the series and compile,
  install and serve that effective schema, which is unchanged on the
  wire.
  `schema_ledger.py check` now refuses a vendored copy that differs from
  the pinned file or carries a `local` ledger entry, a wrap that sets
  `diff_files`, and a series mirror that drifts;
  `schema_ledger.py apply` writes the effective schema.

- Repeating `bindCalculators` with an argument of 0 or less returns the
  stored group when it names the same split.
  Before, the repeat compared the raw argument with the resolved size
  and returned index -1.

- The C++ potentials that share the neighbour list (LJ, Morse,
  Sutton-Chen, ZBL, EAM, Stillinger-Weber, Tersoff) are several times
  faster on a sheared or otherwise non-orthogonal periodic cell, with
  bit-identical results.
  The minimum image of a pair no longer searches the nearby lattice
  images when the rounded image is provably the shortest, and the
  neighbour list drops a pair without that search when every image lies
  beyond the cutoff plus skin.
  Through `rgpot.ase.RgpotCalculator`, a sheared 32-atom copper cell,
  which the Morse cutoff tiles into 864 atoms, went from 203 to 37 ms
  for the first evaluation and from 48 to 6.1 ms for each one after a
  small move.

- The Rust RPC server rejects a `calculate` request whose positions do
  not hold three values per atom or whose box does not hold 9 values,
  and a callback result whose forces tensor has the wrong size.
  Before, the callback read past the end of a short request and the
  server read past the end of short forces.

- The headers in `include/rgpot` include each other by file name, so
  `rgpot/rgpot.hpp` compiles on a case-insensitive file system such as
  APFS or NTFS.
  There, with `CppCore` ahead of `include` on the search path,
  `rgpot/potential.hpp` also named `CppCore/rgpot/Potential.hpp`, and
  code using `rgpot.hpp` failed with "no member named
  'PotentialHandle'".

- Threaded stored EAM, `LJClusterPot`, `SWSi` and `TersoffSiC` now
  evaluate every atom when the OpenMP runtime starts fewer threads than
  `OMP_NUM_THREADS` asks for.
  Each split the atoms into one slice per requested thread and handed
  slice `t` only to thread `t`, so a call inside another OpenMP region
  or under `OMP_THREAD_LIMIT` left out the atoms of the missing threads:
  with `OMP_NUM_THREADS=2` and `OMP_THREAD_LIMIT=1`, the stored EAM
  energy of a 500-atom aluminium crystal was 151.6 eV off.
  They now run their slices through `nlist::forEachSlice`, so a fixed
  thread count gives the same bits whatever team the runtime starts.

- Two `EAMPot` objects evaluated at the same time on different threads
  no longer race on one process-wide table of element splines.
  Each potential now keeps the tables of the elements it has seen.

- `CP2KPot`, `CPMDPot`, `MOPACPot` and `NWChemPot` skip an engine
  variable that is set but empty, as the other engine-backed potentials
  do.
  Before, the empty path opened the running program and stopped the
  search with a missing-symbol error.

- `D3Pot` and the `"d3"` term of `ExprPot` accept `damping` as `"bj"`,
  `"BJ"`, `"zero"` or `"Zero"` and raise `ValueError` for any other
  string.
  Before, an unknown or misspelled name silently selected Becke-Johnson
  damping.

- `EAMPot` throws `std::invalid_argument` for an atomic number it has no
  parameters for.
  Before, it threw the bare `int` 14324, which callers catching
  `std::exception` could not handle.

- `EAMTable` builds again: its pair walk uses the `Neighbours` cell list
  after the removal of the old pair-list cache header.

- `EMTPot`, `D3Pot` and `D4Pot` rebuild their cached structure when the
  atomic numbers change.
  Before, they did this only when the atom count changed, so a reused
  instance gave the energy and forces of the previous species.

- `IPIPot` no longer fails with "timed out" just before its deadline.
  The wait it passed to `poll()` was the time left rounded down to whole
  milliseconds, so `poll()` could return while time remained; it is now
  rounded up.

- `StillingerWeberSi`, `TersoffSiC` and `IPIPot` raise
  `rgpot.NumericalError` from `__call__` for a non-finite energy, and
  `StillingerWeberSi` and `TersoffSiC` release the interpreter during
  the call, like the other potentials.
  Before, they returned the energy unchecked, and the two kernels held
  the interpreter.

- `UmaPot::caps()` reports `batched`.
  `UmaPot` already evaluated a batch of images in one AOTInductor call,
  but the flag was false, so a caller that trusts it, such as a nudged
  elastic band, evaluated the images one at a time.

- `UmaPot` and `libuma_engine` no longer unpack the `.pt2` into a new
  directory under the system temporary directory, which torch left
  behind whenever a process died and which could fill or be missing on a
  fresh machine.
  The package is unpacked once into
  `<root>/aoti/<sha256 of the .pt2>.model/` and reused by later runs,
  and concurrent processes share one unpack.
  The root is `UmaConfig::cache_dir` or the new `UmaParams.cacheDir`,
  then `RGPOT_CACHE_DIR`, then `$XDG_CACHE_HOME/rgpot`, then
  `.rgpot-cache` in the working directory.

- `XTBPot` and `XTBDlopen` no longer start an SCC from the iterate of
  one that failed, and a session's GFN2 energy on hot Si clusters
  matches a fresh start and standalone xtb.
  The new `XTBConfig::warm_min_gap` (1 eV) uses xtb's own guess when the
  HOMO-LUMO gap is too small for a unique SCC solution.
  The engine exports `rgpot_xtb_set_warm_min_gap` and
  `rgpot_xtb_warm_min_gap`.
  `TBLitePot` drops the wavefunction of a failed SCF.

- `evaluate_batch` raises `TypeError` for a positions or box entry that
  is not a C-contiguous float64 array, as the other entry points do.
  Before, it raised `RuntimeError` with the message `std::bad_cast`.

- `from rgpot import *` now also imports `LammpsPot`, `GromacsPot`,
  `OpenMMPot`, `rgpot_lammps_potential`, `rgpot_gromacs_potential` and
  `rgpot_openmm_potential`.
  They were exported but missing from `rgpot.__all__`.

- `potserv` on the Morse backend applies
  `configure(PotentialConfig.morse)`, rebuilding the pot with the sent
  `de`, `a`, `re` and `cutoff` and keeping its switch width; it used to
  refuse every configure call that was not for NWChem or CPMD.
  A value it cannot use, such as a cutoff below the switch width, gets
  `ok=false` and leaves the pot unchanged.
  On every backend `none` is a no-op, and any other arm the backend does
  not apply, including one from a newer schema, gets `ok=false` with a
  message that names the arm.

- `rgpot-eval LJ`, `Morse` and `SuttonChen` without a `cutoff` option
  use the potential's own default cutoff (15, 9.5 and 40 Angstrom)
  instead of 0.
  `pair_style rgpot` still passes its cutoff.

- `rgpot.ase.RgpotCalculator` gives the right energy, forces and stress
  for a periodic cell narrower than twice the potential's cutoff.
  The kernels take the minimum image of each pair, so before this a
  32-atom argon cell with an 8.5 Angstrom LJ cutoff came out 12% too
  high and a 4-atom EAM-Al cell gave -2.996 instead of -3.356 eV/atom.
  The calculator now tiles such a cell into a supercell that is wide
  enough and returns the energy per copy and the forces of the first
  copy.
  Structures with periodic and non-periodic axes (slabs, wires) are
  evaluated instead of raising `ValueError`.

- `rgpot.h` declares `rgpot_potential_set_peer_capabilities` only when
  `RGPOT_HAS_SCHEMA` is defined, which the meson build does with RPC on.
  Before, the header declared it in builds whose library did not define
  it, so a call failed at link time.

- `rgpot_engine_last_error` in the in-tree Lennard-Jones engine plugin
  writes the terminating NUL into a one-byte buffer, as `engine_c_abi.h`
  promises.
  Before, it left that buffer untouched.

- `rgpot_engine_set_charge_spin` on the Metatomic engine reaches the
  model: each system carries the `rgpot::charge`,
  `rgpot::spin_multiplicity` and `rgpot::spin` data the model declares
  in `requested_inputs`, and the engine reads `charge`, `spin` and
  `attachSystemExtras` from `MetatomicParams`.
  It used to return 0 and drop the values.
  For a value the model does not read, other than charge 0 or
  multiplicity 1, `rgpot_engine_set_charge_spin` now returns 3 and
  `rgpot_engine_create` fails.

- `rgpot_external_potential_energy_forces` returns 1 with its usual
  message for a null `box9` when `n` is 0, instead of reading through
  the null pointer.
  The LAMMPS, GROMACS and OpenMM constructors also reject an infinite
  `u0`, `cutoff` or `psi`, which they used to accept.

- `rgpot_tensor_owned_cpu_f64_2d` returns NULL with an error message for
  NULL data, a negative extent or an overflowing size, and
  `rgpot_tensor_device`, `rgpot_tensor_data` and `rgpot_tensor_shape`
  return a safe value for a NULL tensor.
  Before, these cases were undefined behaviour.

- On Windows, publishing an AOTI package replaces a file that a reader
  still has open, as the 3.6.0 notes said it did.
  `MoveFileExW` refuses to replace a file with any open handle, so a
  reader that kept the package open could make the publish fail with
  access denied after its two-second deadline.
  The replace now falls back to POSIX rename semantics (Windows 10 1607
  or later, NTFS), and the reader keeps the complete package it opened.
  A read, reservation or publish that gives up reports the error code,
  the attempts and the elapsed time.

## [3.6.0] - 2026-11-13

[3.6.0]: https://github.com/OmniPotentRPC/rgpot/tree/v3.6.0

### Breaking

- librgpot's ELF soversion is 5.
  LJPot and MorsePot each hold a neighbour list and a cached pair table,
  so a binary built against soversion 4 does not load librgpot.so.5.
  librgpot_mpi stays at soversion 3.
- librgpot's ELF soversion is 6.
  LJ and Morse no longer keep a copied half-list, and ZBL and the open
  Lennard-Jones cluster hold a neighbour list.
  A caller built against soversion 5 does not load librgpot.so.6.
  librgpot_mpi stays at soversion 3.

### Added

- A public neighbour query returns a structure-of-arrays table or a
  chunk of a walk.
  A pair list is a half shell or a full shell.
  A cluster of four is a separate shape and stores every neighbour.
  The C ABI is version 4.
  `rgpot_nlist_request_init` sets the defaults, `bytes` is the struct
  size, and `rgpot_nlist_strerror` names a code.
  Potentials.capnp carries the same columns, and a client streams chunks
  with `cursor` and `next`.
  The tutorial is `docs/orgmode/tutorials/neighbour-list.org`.
  The query walks the same cell list as the classical potentials.
  Fewer than 8192 atoms stay on the calling thread.
  A larger build writes one shared table.
  Several independent systems are one batch, and the C builder keeps a
  Verlet list.
  A positive skin with a strict cut is rejected.
  The classical potentials call that same walk.
  `rgpot.neighbor_list` is the Python half list.
  The Fortran gather uses that list and the builder's skin cache.
  `quantities` selects columns, and `offsets=True` adds a sorted CSR row
  pointer.
- `EAMTablePot` reads embedded atom method (EAM) and Finnis-Sinclair
  potentials from the LAMMPS (Large-scale Atomic/Molecular Massively
  Parallel Simulator) `funcfl`, `setfl` and `eam/fs` tables, with
  energy, forces and an optional Cauchy stress.
  It is built only with `-Dwith_eam_table=true` (CMake
  `RGPOT_WITH_EAM_TABLE`), so the default build is unchanged.
  On `Cu_u3.eam` and `CuNi.eam.alloy` from the LAMMPS `potentials`
  directory it matches `pair_style eam` and `eam/alloy` to 3.1e-12 eV in
  the energy and 1.7e-14 eV/Angstrom in the forces.
- `LJClusterPot` and `MorsePot` are on the Python extension.
  They call the same force implementations as the C++ classes already
  linked into `librgpot`.
- `SuttonChenPot` writes the Cauchy stress when
  `SuttonChenConfig::stress` is set.
  The flag is off by default, and with it clear the energy, forces,
  capabilities and parameter key are unchanged.
  The stress matches LAMMPS `pair_style eam/fs` to 2e-13 eV/Angstrom^3
  on a cubic and a sheared platinum cell.
- `TersoffSiC::forceImpl` writes the Cauchy stress and `caps().stress`
  is true.
  The stress is (1/V) dE/dε in eV/Angstrom^3, with the sign and layout
  of the `LJPot`, `MorsePot` and `ZBLPot` stress.
  The virial sums each folded bond vector times the energy gradient with
  respect to that vector, the many-body form of Thompson, Plimpton and
  Mattson, J. Chem. Phys. 131, 154107 (2009).
  Its digital object identifier (DOI) is 10.1063/1.3245303.
  Energies and forces keep their bits.
  `TersoffPot` keeps `caps().stress` false and writes no stress on any
  call, with `has_stress` 0 and a zero tensor, since its all-silicon
  calls stay on the Fortran gather.
- `XTBPot` and `XTBDlopen` reuse the self-consistent charge (SCC)
  wavefunction across calls.
  Reuse holds while every atom stays within `warm_move` angstroms.
  The default guard is 1 angstrom.
  `reset()` drops that guess, as does a larger move or a new
  composition.
  The next single point then uses xtb's own guess.
  The engine exports `rgpot_xtb_reset` and `rgpot_xtb_set_warm_move`.
  It also exports `rgpot_xtb_warm_move` and
  `rgpot_xtb_reused_wavefunction`.
- Sutton-Chen potential, with the reduced platinum and gold 10-8
  parameters as the default.
  Energy and analytic forces, cluster and periodic evaluation, and
  `rgpot_sutton_chen_energy_forces` for callers that load the library.
- Physical Sutton-Chen presets for platinum and gold, and bulk
  nearest-neighbor and cohesive-well scales.
- Lennard-Jones, Morse, the analytical embedded-atom method (EAM)
  kernel, and the orthorhombic fold select an instruction set at run
  time through Google Highway.
  `RGPOT_SIMD` names the target and `rgpot::simd::chosen_target_name()`
  reports it.
  A product-sum uses a hardware fused multiply-add (FMA) only on a
  target the runtime check selected, and only when that target has the
  instruction.
  SSE2 and SSE4 use it when glibc reports a usable FMA, and a Dekker
  product otherwise.
  Both potentials still take their pairs from the pair-list cache.
- Unshifted Lennard-Jones and Morse cluster calls, binary Lennard-Jones,
  and TIP4P site energy.
  `rgpot_lj_gradient` returns the energy gradient.
  `rgpot_lj_curvature_pairs` evaluates a pair list in one call.
  The `rgpot-pair` crate inlines the same unit trees.
- `rgpot.LJPot` takes the `LJConfig` fields as keywords (`u0`, `cutoff`,
  `psi`, `switch_width`), with the C++ defaults, so Python can turn on
  the C^2 cutoff switch of the Lennard-Jones pair.
  An `ExprPot` child given as a dict reads the same keys for `"lj"` and
  `De`, `a`, `re`, `cutoff` and `switch_width` for `"morse"`.
  Both pair kinds were built from their defaults before.
- `PotType::TersoffSi` is one potential.
  All-silicon configurations, and a null species pointer, use the
  Fortran gather.
  Any other atomic number uses `TersoffSiC`, the 1989 silicon-carbon
  kernel.
  That kernel is from Tersoff, Phys. Rev. B 39, 5566 (1989).
  Its digital object identifier (DOI) is 10.1103/PhysRevB.39.5566.
  The erratum, Phys. Rev. B 41, 3248 (1990), sets carbon B to 346.74 eV
  and d to 4.3484.
  The DOI of the erratum is 10.1103/PhysRevB.41.3248.2.
  Carbon beta is dimensionless and is not printed by the erratum.
  An atomic number other than 6 or 14 is rejected.
  There is no separate silicon-carbide PotType.
  Calling `TersoffSiC` directly always evaluates the 1989 kernel.
  `energy_gradient` evaluates one configuration or a batch of
  configurations that share a cell and a species list.
- `StillingerWeberSi` (`PotType::SWSi1985`) is the Stillinger-Weber
  silicon potential of Stillinger and Weber, Phys. Rev. B 31, 5262
  (1985), doi:10.1103/PhysRevB.31.5262.
  Epsilon is the 3.4723e-12 erg printed in equation (2.9), 2.16723920
  eV.
  `PotType::SWSi` keeps eOn's rescaled epsilon, 2.16826 times
  1.06767385, which is 2.3149945 eV.
  The ratio of the eOn epsilon to the published value is 1.068176741337.
  The factor 1.06767385 is copied from eOn's SWFortran.f90 and is
  untraced.
  `energy_gradient` evaluates one configuration or a batch of
  configurations that share a cell and a species list.
- `TersoffSiC::setHeldSkin` keeps a skinned neighbour list across
  `forceImpl` calls.
  The list is rebuilt when any atom has moved by more than half the skin
  since the last build, or when the cell or the atom count changes.
  The skin is 0 by default, which rebuilds on every call as before.
  `TersoffPot::setHeldSkin` passes the skin to the 1989 kernel;
  all-silicon calls stay on the Fortran gather.
  With a positive skin the instance reports `Reentrancy::PerInstance`.
- MOPACPot dlopens the split
  [mopacc](https://github.com/OmniPotentRPC/mopacc) engine
  (`libmopacc.so`, OpenMOPAC AM1).
  Packed `MopacCParams`, not Cap'n Proto.
  It is not the molecular orbital package (MOPAC) engine of the
  Amsterdam Modeling Suite.
- NWChem remote procedure call (RPC) specifications accept basis,
  theory, exchange-correlation, and self-consistent field (SCF)
  iteration controls for reproducible electronic-structure requests.
- Skala XC is available as C++ ``SkalaPot``: NWChem density functional
  theory (DFT) through ``libnwchemc`` with ``dft.xc`` defaulting to
  ``skala-1.1``.
  Enable with ``-Dwith_skala=true`` (requires ``-Dwith_rpc=true``).
- UmaPot evaluates the OMol universal model for atoms (UMA) from an
  AOTInductor `.pt2` package.
  The package comes from `scripts/export_uma_aoti.py`.
  The vesin library builds the neighbor list, and the compiled graph
  returns energy and forces.
  Charge and spin are per-call tensor inputs.
  One `.pt2` is exported per composition with `merge_mole`.
  Its sidecar records cutoff, neighbors, task, charge, spin, `z_set` and
  label.
  The Baker exporter walks Baker endpoints and deduplicates them by
  `z_set`, charge and spin.
  It is `scripts/export_baker_uma_aoti.py`.
  potserv accepts `Uma:<model.pt2>` and `Uma:<model.pt2>:<task>`.
- `IPIPot` listens for an i-PI driver on a UNIX or TCP socket and
  returns energy, forces, and Cauchy stress.
  `potserv` selects it with `IPI:unix:<path>` or
  `IPI:tcp:<host>:<port>`.
  Meson `with_ipi` defaults to auto and stays off on Windows.
  CMake `RGPOT_WITH_IPI` follows the same rule.
- The Python extension builds a free-threaded cp314t wheel and declares
  that support as beta.
  `rgpot.ase.RgpotCalculator` evaluates potentials in process, including
  a batch of structures.
- `rgpot_eindir_abi_compatible(&stamp)` checks an `eindir_abi_stamp_t`
  against the eindir-core compiled into `librgpot`, so a caller that
  takes an rgpot potential as an eindir objective needs no second link
  to eindir-core.
  It returns what `eindir_core_abi_compatible` returns for the same
  argument.
  Existing calls do not change.

### Developer

- `FastNlistTest` links with `-Dwith_fortran_pots=disabled`.
  Its two water-H cases call the Fortran kernels and are compiled only
  when those kernels are built.
- The regeneration, comparison and timing scripts read the hosts they
  may run on from `RGPOT_REGEN_HOSTS`, a comma-separated list of host
  names that is empty by default.
  Until it lists the current host, `scripts/regen_dftd_goldens.py`,
  `scripts/regen_expr_goldens.py`, `scripts/regen_xckernel_goldens.py`,
  `scripts/gen_libxckernel.sh`,
  `scripts/compare_xckernel_tda_nwchemc.py` and
  `scripts/time_lj_rgpot_vs_pyeonclient.py` refuse to run.
  `DFTD_ALLOW_REGEN=1`, `EXPR_ALLOW_REGEN=1`, `XCKERNEL_ALLOW_REGEN=1`
  and `TIME_LJ_ALLOW=1` still override the check.
  `scripts/gen_libxckernel.sh` records the host in
  `third_party/libxckernel/PIN` as the label from `RGPOT_HOST_LABEL`, or
  as `unspecified` when that is unset or blank, and never writes the
  system hostname there.
- Coverage jobs stay green when the Codecov OpenID Connect (OIDC) token
  request times out.
  A failed OIDC upload is retried with the repository token.
- Pull requests into develop run the CI orchestrator, including the
  build matrix and the Windows leg.
- Documentation builds use a concurrency group per ref, so one pull
  request does not cancel another's build.
  A running GitHub Pages deploy is left to finish.
- The release version is written by hand only in the root `Cargo.toml`.
  The key is `version` in `[workspace.package]`.
  `rgpot-core` and `rgpot-pair` inherit it.
  cogapp regenerates every other copy from it.
  Copies sit in the meson, CMake, pyproject, pixi, towncrier and Sphinx
  files.
  The C header and the rgpot-core README carry copies as well.
  A prek hook fails a pull request when a copy drifts.
- The release-prepare commit check ignores merge commits.
  Cog 7 is invoked as `cog check --ignore-merge-commits <base>..HEAD`.
  Single-parent `merge:` sync commits are ignored as well.
- The shared `py-tsan` workflow sets `vm.mmap_rnd_bits=28`
  unconditionally and runs its container with `seccomp=unconfined`, and
  the instruction set architecture (ISA) self-test compiles its x86
  fixtures at `-march=x86-64`.
  Both follow the `ubuntu-latest` runner's move to Ubuntu 26.04, where
  the sysctl is root-only to read and the default compiler emits vector
  extension (VEX) scalar moves.
- py-tsan takes an optional rust-archive input for the Rust staticlib
  name.
  The default remains librust_core_demo.a.
- The tight Lennard-Jones cluster kernel's AVX2 bodies run only after
  `__builtin_cpu_supports("avx2")`.
  `rgpot/isa-allow.toml` names `lj_avx2`, `eval_quad`, and `acc_quad`
  for Advanced Vector Extensions (AVX) and AVX2.
  It also names `__cpu_indicator_init` from GCC for the `xgetbv` that
  initializer uses.
- Wheel builds keep an unstripped copy of librgpot and _core and reject
  instructions outside the x86-64 and armv8-a baselines, except vendored
  libraries and the FMA kernels behind the glibc check named in
  rgpot/isa-allow.toml.
- `librgpot` on x86-64 now carries one copy of `exact_fma_hw`, the one
  in `exp.cc`.
  The weak copy in `exact_fma_hw.cc` stayed in `.text` with no symbol,
  and the instruction set audit of the Linux x86-64 wheels failed on its
  `vfmadd132sd`.
- The Baker hydrogen cyanide (HCN) UmaPot case skips when
  ``uma-s-1p1-omol-hcn.pt2`` is absent.
  Set ``RGPOT_UMA_REQUIRE_FIXTURES`` to fail that case instead.
- `CppCore/meson.build` follows `meson format` and seven org
  documentation pages follow the semantic line breaks of `snapper`.
- The shared `py-wheels` workflow defaults to the `macos-15` runner
  image instead of `macos-14`, which GitHub Actions retires on
  November 2, 2026.
  The `wheels` workflow and the C++ layout check request `macos-15`
  as well.
- Twelve pinned GitHub Actions in the CI workflows move to releases
  that run on the Node 24 runtime, including checkout 7.0.1,
  setup-python 7.1.0 and upload-artifact 7.0.2.

### Changed

- `EAMPot` keeps a Verlet neighbour list and reuses it while every atom
  moves by at most half of one fifth of the widest cutoff.
  One thread matches the previous energy and forces bit for bit.
  `OMP_NUM_THREADS` above 1 splits atoms across threads.
  A fixed count repeats bit for bit and stays within a relative
  tolerance of 1e-9 of the one-thread result.
- macOS wheels of the main package are built on the `macos-15` runner
  and tagged `macosx_15_0`.
  Homebrew on that image builds libgfortran, libquadmath, OpenBLAS and
  libomp for macOS 15.0, and delocate rejects them under a 14.0 target.
  The bind and rust-core template wheels stay `macosx_14_0`.
- EAM, Lennard-Jones, Sutton-Chen, Stillinger-Weber and Tersoff share
  one neighbour layer.
  It bins atoms into cells, folds the minimum image, and either stores a
  Verlet list or walks the cells without storing pairs.
  The mode follows the memory the operating system reports as available,
  or a caller can select it.
  `OMP_NUM_THREADS` above 1 splits atoms across threads.
  One thread matches the previous energy and forces bit for bit.
  A fixed thread count repeats bit for bit and stays within a relative
  tolerance of 1e-9 of the one-thread result.
- Morse and ZBL take pairs from the shared neighbour layer.
  The separate pair-list cache is gone.
  Stillinger-Weber and Tersoff use that layer for every mode, including
  the default.
  One thread stays within 1e-12 of the previous ordered walk.
  A streaming evaluation keeps a cell binning and a per-atom scratch,
  and several threads walk those cells and build bonds together.
- Sutton-Chen streaming keeps the per-atom density between two cell
  walks.
  It no longer stores a partner index for every pair.
- A short cutoff reuses the shared stream stencil, keeping only cells
  that hold a pair inside the search when a 27-slot entry per cell is
  larger than the pair index.
  Morse and EAM stay on the full stencil.
  `Mode::Auto` still stores when the pair index fits.
- The CPMDPot README states how a force call sets the cell.
  The box of the call is the cell, and a cpmdc release after `0.3.0`
  replaces the `CELL` of the input deck with it.
- The Morse kernel computes `exp` with `rgpot::simd::exp_fixed`, a fixed
  sequence of IEEE operations that returns the same bits on every
  instruction set and C library (within 1.24 ulp), and evaluates each
  pair-list row in vector clones for AVX2 and AVX-512F.
  Morse energies and forces can move in the last bits, and the Morse
  cache key changes.
- The general-cell minimum image is computed in double.
  An ill-conditioned cell is Minkowski-reduced as in Nguyen and Stehle
  (2009), whose DOI is 10.1145/1597036.1597050.
  A reduced basis that is still too ill-conditioned searches the 27
  images with coefficients in {-1, 0, 1}.
  Each product-sum is a hardware fused multiply-add, inlined in the
  fold, when glibc reports FMA.
  Otherwise it is the exactly rounded TwoProduct and TwoSum emulation of
  Boldo and Muller (2011), whose DOI is 10.1109/tc.2010.139.
  That file is compiled with contraction off, so the folded bits do not
  depend on the instruction set.
- The software fused multiply-add used on CPUs without FMA is about five
  times faster in the minimum-image fold, with the same bits as
  `std::fma`.
- The rgpot-core C ABI minor version is now 1 for the new
  `rgpot_eindir_abi_compatible`, in `RGPOT_ABI_VERSION_MINOR`.
  A caller built against minor 0 still passes `rgpot_abi_compatible`.
  The ABI major and the layout revision do not change.
  The `librgpot` SONAME does: this release ships soversion 6.
- The XcKernel stage-B contraction runs on dgemm and dsyr2k, blocked
  over the grid, linked through FlexiBLAS or the generic blas
  dependency.
- The eonviz environment installs readcon 0.17.1 from PyPI.
  That wheel's RoundTrip writer matches the vendored restart golden.
  Its `atoms_to_con` writes one `.con` frame with the RoundTrip writer,
  precision omitted and canonical left off.
  The UMA exporter reads `.con` files with `atoms_from_con`.
  CI compares those writes with the byte-exact restart golden.

### Fixed

- Coincident Lennard-Jones atoms raise NumericalError again.
  Potentials that keep a neighbour scratch ask for one instance per
  thread.
  The layout-stamp job reads `kCxxSoversion` from the header.
  ThreadSanitizer leaves the fused multiply-add (FMA) resolver units
  uninstrumented so the extension can load.
  The macOS wheel links libgfortran from the path the compiler reports.
  A stored neighbour list is kept when free memory cannot be read.
  Flang is not given the MSVC OpenMP compile flag.
  Apple clang is not given `-fopenmp` by the Rust neighbour caller.
  The installed CMake package exports the neighbour archive the
  libraries link.
  The instruction set architecture (ISA) audit names the hardware
  Lennard-Jones span.
  The socket neighbour example is built only where POSIX processes
  exist.
  The stream scaling bench is too: it forks a child per row.
  The Rust neighbour caller on Windows links the COFF archives with
  link.exe and exports the C entry points.
  An empty JSON member in a UMA package is written as an empty object,
  which the AOTInductor loader can parse.
  Hypothesis checks stay serial, because the strategy cache is
  process-wide.
  The 108000-atom neighbour list is allowed 30 seconds.
  The Sutton-Chen cluster relaxation is allowed 120 seconds.
  rustc on that host uses the archive directory as its temp directory.
- The classical engine now links the water potentials when
  `-Dwith_water=true` is set, so `libclassical_engine` no longer fails
  to link.
  It carries the water kernels that its `tip4p`, `spce` and `tip4p_pt`
  names construct.
- `librgpot_cuh2` links the Fortran runtime again.
  The plugin is a C++ link, so the runtime was not inferred and the
  shared library was missing `_gfortran_*`.
- The manylinux wheel jobs retry the gfortran and OpenBLAS install.
  Each retry reads fresh metadata from the primary AlmaLinux repository.
  The mirror list could return the same broken mirror on every attempt.
  The jobs stop when the install fails, instead of failing on a missing
  OpenBLAS.
- `forceBatch` now zeroes the energy and forces of every result slot
  before the kernel runs, as it already did for the variance and stress.
  Before, a reused buffer kept an earlier call's energy and forces when
  the kernel threw, for example when a plugin reported an error or
  returned a non-finite value.
- UmaPot builds positions, the cell and cell offsets in the package
  dtype.
  The package is AOTInductor (PyTorch ahead-of-time compiled).
  It records `pos_dtype`, and `model_dtype` when `pos_dtype` is absent.
  The exporter converts the model with `hydra.double()` for `--dtype
  float64`, and calls `hydra.float()` for `--dtype float32`.
  A package with neither key stays float32.
  UmaPot logs that fallback and both dtypes at load.
- A cell narrower than three cutoffs is every unordered pair, including
  lists past 4096 atoms.
  The client bridge stress test (10k atoms, a 10 A box, Lennard-Jones
  cutoff 15 A) no longer asks linkcell for that stencil.
- Pair-list force updates each use one fused multiply-add, so a cold
  evaluation and a captured pair list agree to the last bit.
- The pair-list loops and the minimum-image fold use the hardware fused
  multiply-add on any CPU with FMA, detected from the `cpuid`
  instruction, where they took the software path on C libraries older
  than glibc 2.33 (Enterprise Linux 8).
  `RGPOT_FMA=software` selects the software path, which returns the same
  bits.
- The aarch64 Linux wheels build again.
  The minimum-image fold takes `target("fma")` on x86-64 only.
  That is the one target where `fma_hw_active()` can select the hardware
  body.
- A ThreadSanitizer build of `librgpot` compiles each
  `RGPOT_SIMD_CLONES` loop once, for the baseline, so
  `import rgpot._core` no longer crashes in the loader.
  The sanitizer had instrumented the clone resolver.
  It called the sanitizer hooks through procedure linkage table slots.
  An `RTLD_NOW` load had not bound those slots yet.
  Every other build keeps the clones, and every clone returns the same
  bits.
- `MinimumImage::fold` now finishes on thin or strongly sheared cells,
  and on long displacements along open directions; those cases search an
  LLL-reduced basis of the periodic rows.
- The H-in-water 26-image scan is one routine.
  Under Apple Clang the Fortran copy and the test copy differed by an
  ulp, so a skewed cell disagreed with the cluster folded from it.
- The linkcell and minimage pair list builds only when cargo and rustc
  1.89 or newer are available, and the MSVC archive is
  `rgpot_fastnlist.lib` with the system libraries that compiler prints.
  On a skewed cell with a longer fractional wrap, WaterH keeps the
  26-image nearest vector.
- When the fast pair-list library is built, it is linked into potserv,
  the examples, and client-only builds.
- A static build links the fast pair list as one relocatable object.
  Apple ar was taking the rust archive on its command line as the
  archive to write, so the macOS umbrella library was never created.
- The fast pair-list archive on macOS is written with ``/usr/bin/ar``.
  The conda archiver pixi exports as ``AR`` produces an archive Apple ld
  rejects.
- An eindir gradient is served from the fused energy result only when
  the positions have the same bits as the energy call.
  Before, a gradient at `-0.0` was served the result computed at `+0.0`,
  because the two compare equal.
  The pair list cache and the half-shell neighbour list now compare the
  box bit for bit as well, so a box that differs only in the sign of a
  zero gets its own build.
  Inputs without a signed zero take the same path as before.
- The MOPAC frontend stays inside `librgpot` and is not installed as its
  own archive.
  Installing that archive copied the core helpers into it and the shared
  library then failed to link.
- The NWChem and Car-Parrinello molecular dynamics (CPMD) frontends stay
  inside ``librgpot`` and are not installed as their own archives.
  Installing those archives could copy the core helpers into them and
  make the shared library fail to link.
  A client-only build, which has no umbrella library, still installs the
  archives.
- The Lennard-Jones cluster kernel no longer crashes at thread exit when
  RocksDB links jemalloc.
  The conda-forge `rocksdb` links it, and `RGPOT_WITH_CACHE=ON` uses
  that build.
  Its aligned scratch buffer comes from `posix_memalign`, which jemalloc
  replaces along with `free`.
- The open Lennard-Jones cluster kernel allocates its scratch buffer
  with ``_aligned_malloc`` on Windows.
  MSVC does not provide ``std::aligned_alloc``.
- The half-shell pair walk builds with the Visual C++ compiler again.
  That compiler gets `__forceinline`, `__declspec(noinline)` and a plain
  test, and GCC and Clang keep their attributes and `__builtin_expect`.
  The pair and Sutton-Chen headers declare their C entry points for
  export, so `cl.exe` accepts the exported definitions.
  The Tersoff neighbour-list bench prints `_MSC_FULL_VER` where the
  compiler has no `__VERSION__`.
- The Meson build compiles `ReducedPairTest` with `-ffp-contract=off`,
  as it already compiles the `ReducedPair` library.
  Apple Clang had fused the expected Morse curvature in the test into an
  `fma`, which missed the library value by one ulp.
- Publishing an AOTInductor package on Windows replaces a file a reader
  still has open.
  The reader keeps the complete package it opened, and the replace no
  longer fails with access denied.
- On Windows, UmaPot can publish a rewritten AOTInductor package while
  another reader or writer has the destination open.
  Access denied and sharing violations during the replace are retried
  until a deadline.
  The published file is one complete payload.
- UmaPot clears PT_GNU_STACK PF_X on the AOTInductor wrapper.so before
  dlopen, so Elja-traced ``.pt2`` packages load on hosts that refuse an
  executable stack.
  ``scripts/export_uma_aoti.py`` applies the same rewrite at mint time.
- The XcKernel golden pins use the stage-B bound `k*eps*|a|^T|b|` (k is
  the grid-block length, at most 256) plus half an ulp of the pin, for
  the C-versus-NumPy and random-grid Fock contractions.
  A fixed 1e-16 cutoff rejected a correct dsyr2k sum.
  CI runs XcKernelGoldens with `-Dwith_xckernel=true`.
- XcKernel links OpenBLAS when FlexiBLAS and a pkg-config file named
  blas are both absent.
  Both manylinux `openblas-devel` and Homebrew publish `openblas.pc`.
- `XcKernel.contract` still treats a missing `dchi` as omitted.
  `scal` stays required.
  The stub is valid Python: passing `None` for `scal` raises
  `TypeError`.
- ``XTBDlopen`` no longer loads a different
  ``libxtb_engine`` when ``engine_path`` is set and that file cannot be
  opened.
  Environment variables and the bare soname are used only when
  ``engine_path`` is empty.
- ``XTBDlopen`` rejects a slash-less ``engine_path``.
  A bare soname would be opened through the loader search path and could
  substitute a different ``libxtb_engine``.
  Pass an absolute or relative path.
  When that path does not open, the error names it and the loader
  diagnostic instead of environment variables that are ignored.
- `IPIPot` sets the UNIX socket address length on macOS.
  The mock driver is joined before the capture it writes is released.
- The `rgpot._core` type stub lists `StillingerWeberSi` and
  `TersoffSiC`, and carries the current `LJPot` and `ExprPot`
  docstrings.
- `ase_helpers` imports with only ASE and numpy again.
  `atoms_from_con` and `atoms_to_con` work without the compiled
  `rgpot._core`.
  `AnnealRunner` and `AnnealOptimizer` are loaded from
  `rgpot.ase.anneal` on first use.
- Metatomic-enabled build-tree executables locate Torch and metatomic
  shared libraries without an ``LD_LIBRARY_PATH`` override.
- `eam_al` stores the eOn r2549 `potinit.f` default-real literals at
  single precision.
  Keeping the extra printed digits moves the four-atom aluminium cluster
  by 1.76e-6 eV.
- `sutton_chen_scales` summed a bcc lattice.
  The nearest-neighbor distance is now the fcc value a / sqrt(2), not
  0.5634 a, and the reduced 10-8 well is 295.4645 epsilon, not 292.9794.
  `fcc_power_sum` in C++ and `rgpot-pair` changes the same way.

### Miscellaneous

- Questions and bug reports now go to the rgpot issue tracker.
  Security reports go to `SECURITY.md`.
  The Code of Conduct names rgoswami@ieee.org for enforcement reports.
  A root `CONTRIBUTING.md` points to the contributing pages.
  Links between pages no longer point at missing reStructuredText files.
  The navigation now lists the thread-safety page and the Fortran
  porting rules.
- The tutorial "Scan a potential energy surface over RPC" evaluates
  Lennard-Jones and Morse pair curves and an H2-over-copper EAM surface
  through `potserv`, and ships the scripts that draw its figures.
  The remote procedure call (RPC) model page now draws the client,
  server and backend layout.

## [3.5.1](https://github.com/OmniPotentRPC/rgpot/tree/v3.5.1) - 2026-10-11

### Breaking

- 3.5.0 changed the C++ layout of PotCaps, LJConfig, LJClusterConfig and
  MorseConfig and should have bumped the librgpot soversion.
  It kept soversion 3, so a caller built against 3.4.0 loaded 3.5.0's
  librgpot.so.3 and crashed.
  3.5.1 installs that layout as librgpot.so.4.
  Callers built against 3.4.0 fail to load it.
  Callers built against 3.5.0 need to relink.
  The C ABI stamp stays at major 1, minor 0, layout revision 1, because no C
  struct changed layout and the C API has no shared library of its own.
  librgpot_mpi stays at soversion 3, because its C ABI did not change.

### Developer

- The rgbind header is the shared nanobind policy for free-threaded extension modules, with reusable py-* workflows for wheels, free-threading, ThreadSanitizer, and typing.
- py-tsan takes an optional rust-archive input for the Rust staticlib name.
  The default remains librust_core_demo.a.

### Changed

- The documentation site uses the v2 logo and favicons, explains the RPC model,
  and lists the surrounding ecosystem in the navigation.

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

### Deprecated

- The wheel's `libmetatomic_engine.so`, a symlink to `librgpot_metatomic_engine.so`, is kept for one release and removed after it.
  Source builds still install `libmetatomic_engine.so`.

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
  The pip wheel packs the engine as `librgpot_metatomic_engine.so`.

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
  `PairListCache::evaluate` and `CachedPairList::forEach` keep their 3.4.0 signatures.

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

## [3.4.0](https://github.com/OmniPotentRPC/rgpot/tree/v3.4.0) - 2026-09-30

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


## [3.3.0](https://github.com/OmniPotentRPC/rgpot/tree/v3.3.0) - 2026-09-28

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


## [3.2.0](https://github.com/OmniPotentRPC/rgpot/tree/v3.2.0) - 2026-09-13

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


## [3.1.2](https://github.com/OmniPotentRPC/rgpot/tree/v3.1.2) - 2026-09-04

### Fixed

- D3/D4 in the wheel are built from pinned meson wraps (static s-dftd3 / dftd4), not a conda prefix. The dftd4 wrap uses OpenBLAS as its LAPACK. ``pip install`` no longer needs vendored ``libblas`` from micromamba.


## [3.1.1](https://github.com/OmniPotentRPC/rgpot/tree/v3.1.1) - 2026-09-04

### Added

- The PyPI wheel compiles ExprPot, D3Pot, D4Pot, and XcKernel and vendors ``s-dftd3`` / ``dftd4``. ``import rgpot`` exposes those classes (``has_expr`` / ``has_dftd3`` / ``has_dftd4`` / ``has_xckernel``).


## [3.1.0](https://github.com/OmniPotentRPC/rgpot/tree/v3.1.0) - 2026-09-04

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


## [3.0.2](https://github.com/OmniPotentRPC/rgpot/tree/v3.0.2) - 2026-08-08

### Fixed

- Windows MSVC builds Cap'n Proto with lean windows.h defines, and links Flang pot kernels through MSVC lib.exe archives so llvm-ar no longer triggers LNK1107. ([#61](https://github.com/OmniPotentRPC/rgpot/issues/61))
- MetatomicPot no longer fails to load exported PET-MAD (and similar) models on metatensor-torch 0.10.3: scripted modules without `_mts_buffer_names` no longer trip the mixed-dict `.to()` walk.
- Python 3.x wheels import without torch: fat MetatomicPot is no longer
  `link_whole`'d into `librgpot` when `with_python` (engine stays dlopen-only),
  and the repair step ships `librgpot.so.3` (mesonpy zip dropped the soname link)
  plus a RUNPATH into site-packages torch/metatensor/vesin.


## [3.0.1](https://github.com/OmniPotentRPC/rgpot/tree/v3.0.1) - 2026-07-26

### Fixed

- The Python wheel builds the Fortran 2018 potentials. `[tool.meson-python.args]`
  never passed `-Dwith_fortran_pots`, so a wheel carried a `librgpot` with no
  Fortran-backed potential in it -- SW, EDIP, Lenosky, Tersoff, EAM-Al, FeHe,
  CuH2, and TIP4P-H were all missing from the one artifact most users install.
  The cibuildwheel configuration installs a Fortran compiler in the build
  image to match, and the repair step vendors `libgfortran` as it would any
  other dependency.


## [3.0.0](https://github.com/OmniPotentRPC/rgpot/tree/v3.0.0) - 2026-07-26

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


## [2.5.3](https://github.com/OmniPotentRPC/rgpot/tree/v2.5.3) - 2026-07-20

### Fixed

- Fix ``-Dwith_rpc_client_only=true -Dwith_tests=true``: use ``pot_bridge_dep``
  (was undefined ``rgpot_bridge_dep``), link ``units.cc`` into unit tests when
  ``rgpot_core`` is not built, and soft-skip bridge stress cases when potserv is
  not running (lazy client connect). Catch2 configure/link/test succeed for the
  client frontend product without a live server.


## [2.5.2](https://github.com/OmniPotentRPC/rgpot/tree/v2.5.2) - 2026-07-18

### Fixed

- Rebuild Metatomic engines against vesin 0.6+ ``VesinOptions`` (``skin`` /
  ``n_threads``) so ``vesin_neighbors`` no longer fails with a bare error when
  runtime libvesin is 0.6 and the engine was built for 0.5.


## [2.5.0](https://github.com/OmniPotentRPC/rgpot/tree/v2.5.0) - 2026-07-17

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


## [2.2.1](https://github.com/OmniPotentRPC/rgpot/tree/v2.2.1) - 2026-07-06

### Fixed

- Windows/MSVC builds no longer fail on unused ``cxxabi.h`` includes or nested
  ``std::array`` box flattening in ``Potential`` (``C1083`` / ``C2676``).


## [2.2.0](https://github.com/OmniPotentRPC/rgpot/tree/v2.2.0) - 2026-07-05

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


## [2.1.0](https://github.com/OmniPotentRPC/rgpot/tree/v2.1.0) - 2026-07-03

### Added

- eOn integration: eOn ships an in-process `RgpotPot` potential (`-Dwith_rgpot=true`) that consumes rgpot's `NWChemPot` / `CPMDPot` frontends as a Meson subproject and `dlopen`s `libnwchemc` / `libcpmdc` directly, with `potserv` remaining available for out-of-process RPC. See `docs/orgmode/howto/eon-rgpot.org`.

### Changed

- The Cap'n Proto schema is pinned to canonical [potentials-schema](https://github.com/OmniPotentRPC/potentials-schema) v1.12.0: `MetatomicParams` arm (union ordinal 4) with the upstream requested-outputs surface, typed NWChem and CPMD parity batches (dplot/esp, prop/linres/pimd/path/tddft), `CommonMethodSpec` overlay, and a schema-sync CI gate that fails when the vendored copies diverge from the pinned release.

### Fixed

- Release-prepare CI no longer runs `cargo publish --dry-run` for `rgpot-core` while it depends on git-only `dlpk` / `eindir-core` (not on crates.io). The job uses `cargo check -p rgpot-core --locked`, and `package.publish` is `false` until registry deps exist. ([#42](https://github.com/OmniPotentRPC/rgpot/issues/42))


## [2.0.0](https://github.com/OmniPotentRPC/rgpot/commit/18861eb01915a02a477fd40e87a921c675ae6200) - 2026-06-26

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


## [1.2.0](https://github.com/OmniPotentRPC/rgpot/tree/v1.2.0) - 2026-06-24

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


## [1.1.0](https://github.com/OmniPotentRPC/rgpot/tree/v1.1.0) - 2026-03-29

### Added

- XTBPot: GFN tight-binding potential via xtb (GFNFF, GFN0, GFN1, GFN2). Feature-gated with ``-Dwith_xtb=true``. ([#35](https://github.com/OmniPotentRPC/rgpot/issues/35))
- TBLitePot: GFN tight-binding potential via tblite (GFN1, GFN2, IPEA1). Feature-gated with ``-Dwith_tblite=true``. ([#36](https://github.com/OmniPotentRPC/rgpot/issues/36))
- MetatomicPot: ML atomistic models via metatomic/PyTorch with autograd forces and vesin neighbor lists. Feature-gated with ``-Dwith_metatomic=true``. ([#37](https://github.com/OmniPotentRPC/rgpot/issues/37))
- Shared unit conversion header (``rgpot/units.hpp``) with CODATA 2018 physical constants for Angstrom/Bohr, Hartree/eV, and Boltzmann conversions. ([#38](https://github.com/OmniPotentRPC/rgpot/issues/38))


## [1.0.3](https://github.com/OmniPotentRPC/rgpot/tree/v1.0.3) - 2026-03-01

### Fixed

- MSVC/clang-cl compatibility: use ``/W3`` instead of GCC/Clang warning flags and skip ``-lstdc++`` link arg on Windows ([#31](https://github.com/OmniPotentRPC/rgpot/issues/31))


## [1.0.2](https://github.com/OmniPotentRPC/rgpot/tree/v1.0.2) - 2026-03-01

### Added

- External integration guide covering namespace collision mitigation when embedding rgpot as a subproject ([#30](https://github.com/OmniPotentRPC/rgpot/issues/30))

### Changed

- CI dependency bumps and cleanup to prevent trailing whitespace in generated headers ([#29](https://github.com/OmniPotentRPC/rgpot/issues/29))


## [1.0.0](https://github.com/OmniPotentRPC/rgpot/tree/v1.0.0) - 2026-02-15

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
