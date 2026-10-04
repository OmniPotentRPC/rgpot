# NWChemPot - backend under rgpot `PotentialConfig` params

## User options = rgpot `PotentialConfig` (Cap'n Proto)

rgpot has **one** user-facing parameter carrier: `PotentialConfig` in
`Potentials.capnp`, an extensible **union** of backend-specific option structs.
`NWChemParams` is only the **nwchem arm**, not a separate config ecosystem.

Same schema for RPC and in-process; add future arms without new TOML/JSON:

| Arm (today / planned) | Payload | Backend |
|----------------------|---------|---------|
| `none` | void | no backend knobs / no-op configure |
| `nwchem` | `NWChemParams` | NWChemPot |
| *(later)* `metatomic` | `MetatomicParams` | MetatomicPot |
| *(later)* `xtb` / `tblite` | ... | XTBPot / TBLitePot |

```
user / client
    PotentialConfig  { nwchem = NWChemParams{...} }   rgpot params (in/out)
            │
            ├── RPC:  configure(config)
            └── C++:  setPotentialConfig(config)  or  setParams(nwchem only)
            │
            ▼  (nwchem arm only on this pot)
    serialized flat Cap'n Proto NWChemParams bytes
            │  nwchemc_set_params / nwchemc_energy_gradient
            ▼
    libnwchemc.so  (the split nwchemc engine, resolved by dlopen)
      C parser + iso_c_binding embed -> NWChem rtdb/task_energy/task_gradient
      (no user .nw / subprocess CLI)
```

Geometry for `calculate` stays on `ForceInput`; `PotentialConfig` is method/backend setup only.

**Not** a subprocess `nwchem` CLI. The engine needs `NWCHEM_TOP` and a built
`libnwchemc.so` on the dlopen path.

rgpot is a pure consumer of the split engine project
<https://github.com/OmniPotentRPC/nwchemc>: it serializes `NWChemParams`,
`dlopen`s `libnwchemc.so`, and passes the message bytes directly. rgpot builds
no in-process NWChem embed of its own; build `libnwchemc.so` from `nwchemc`.

## Layers

| Piece | Built when | Role |
|-------|------------|------|
| `NWChemPot.cc` frontend | RPC schema support is enabled | Serialize `NWChemParams`, `dlopen` engine, units to eV/Angstrom |
| `nwchem_c_abi.h` | header | stable consumer C-ABI contract (`nwchemc_*`) |
| `nwchem_c_abi_stub.c` | NWChem ABI tests are enabled | no-op ABI: `nwchemc_available()==0`, used by the ABI conformance test |

The real engine (`nwchemc_*` implementation, NWChem embed, Fortran) lives in the
split [`nwchemc`](https://github.com/OmniPotentRPC/nwchemc) project, not here.

```
app / potserv
    │
    ▼
NWChemPot (static, always in librgpot)
    │  dlopen(RTLD_GLOBAL)  NWCHEMC_LIBRARY / RGPOT_NWCHEMC_ENGINE / enginePath
    ▼
libnwchemc.so  (split nwchemc engine)
    nwchemc_set_params / nwchemc_energy_gradient / nwchemc_available
    │
    ▼
NWChem embed  ->  geom/basis via embed API + task_energy/gradient  (NWCHEM_TOP libs)
```

## Meson

```bash
# Frontend and schema support; resolves libnwchemc.so by dlopen at runtime.
meson setup bbdir -Dwith_rpc=true
meson compile -C bbdir
```

rgpot builds no NWChem engine. To get `libnwchemc.so`, build the split
[`nwchemc`](https://github.com/OmniPotentRPC/nwchemc) project against an NWChem
source tree, then point `NWCHEMC_LIBRARY` or `RGPOT_NWCHEMC_ENGINE` at the
resulting shared library. conda/pixi `nwchem` packages ship the **driver
binary**, not the embed SDK; `nwchemc` needs an NWChem source tree.

## Runtime

| Variable | Purpose |
|----------|---------|
| `NWCHEMC_LIBRARY` | Path to `libnwchemc.so` |
| `RGPOT_NWCHEMC_ENGINE` | Path to `libnwchemc.so` |
| `RGPOT_NWCHEM_ENGINE` | Alternate path to `libnwchemc.so` |
| `NWCHEM_TOP` | Hint for engine/data paths (optional; also `NWChemParams.nwchemRoot`) |
| `LD_LIBRARY_PATH` | NWChem `lib/<target>` (and deps) that `libnwchemc.so` resolves against |

## `NWChemParams` fields (payload inside `PotentialConfig.nwchem`)

| field | default | meaning |
|-------|---------|---------|
| `basis` | `sto-3g` | Gaussian basis |
| `theory` | `scf` | Method: `scf`, `dft`, `blyp`, `b3lyp`, ... |
| `scfType` | `rhf` | HF: `rhf`/`uhf`; with DFT: XC functional (`blyp`, ...) |
| `charge` | `0` | Molecular charge |
| `multiplicity` | `1` | 2S+1 |
| `enginePath` | `""` | Frontend: explicit `libnwchemc.so` path; empty -> env/probe |
| `nwchemRoot` | `""` | Frontend: `NWCHEM_TOP`; empty -> env |
| `task` | `gradient` | NWChem task hint; rgpot force calls use gradient |
| `title` | `""` | Optional NWChem title/start prefix |
| `memoryMb` | `0` | 0 -> NWChem defaults / environment |
| `scratchDir` | `""` | Optional NWChem scratch directory |
| `permanentDir` | `""` | Optional NWChem permanent directory |
| `inputBlocks` | `[]` | Raw NWChem directive blocks applied by `nwchemc` before task execution |

Defaults are the Cap'n Proto schema defaults.

### DFT: two equivalent forms

1. **Preferred:** `theory="dft"`, `scfType="blyp"` (or `b3lyp`, ...): explicit DFT + XC.
2. **Shorthand:** `theory="blyp"`: embed maps theory alias to `dft` + XC; still fine if `scfType` left default.

HF: `theory="scf"`, `scfType="rhf"` or `"uhf"`.

### Lifecycle (apply vs calculate)

| Step | What happens |
|------|----------------|
| `setPotentialConfig` / RPC `configure` / `setParams` | Sticky on the C++ pot: stores serialized flat Cap'n Proto `NWChemParams` words |
| Each `forceImpl` / `calculate` | Frontend passes the current message bytes to `nwchemc_energy_gradient(...)` at the dlopen boundary |
| Direct C callers | Pass the same unpacked flat `NWChemParams` message bytes to `nwchemc_set_params(...)` or `nwchemc_energy_gradient(...)` |

### Units

Embed / C ABI: energy **Hartree**, gradient **Hartree/Bohr**. Frontend converts to rgpot **eV** / **eV/Angstrom**. Geometry units for `calculate` remain on `ForceInput`, not on `PotentialConfig`.

```cpp
::capnp::MallocMessageBuilder msg;
auto cfg = msg.initRoot<::PotentialConfig>();
auto nw = cfg.initNwchem();
nw.setTheory("dft");
nw.setScfType("blyp");
nw.setEnginePath("/path/to/libnwchemc.so");
rgpot::NWChemPot pot;
pot.setPotentialConfig(cfg.asReader());  // rgpot params, nwchem arm
```

Python: `configure_nwchem` builds `PotentialConfig` with `nwchem` set.

## Direct C++ smoke (no RPC)

Build `libnwchemc.so` from the split `nwchemc` project, put it on the dlopen
path (`NWCHEMC_LIBRARY` / `LD_LIBRARY_PATH`), then drive `rgpot::NWChemPot`
directly (serialize `NWChemParams`, call `setPotentialConfig`, then `force`).
The dlopen boundary itself is covered by the `nwchem` meson test suite
(`NWChemPotMessageAbiTest`, `NWChemDlopenContract`) using a fake engine.

## RPC (optional, separate)

`potserv <port> NWChem` + `configure` with `NWChemParams` is optional plumbing on
top of the same frontend; it is not required for the C ABI or embed path.

## Units

ABI: Hartree, Hartree/Bohr. Frontend: eV, eV/Angstrom (`rgpot::units`).

## MPI hosts

With RPC schema support and MPI enabled, the build provides two explicit MPI
executables. `nwchem_mpi_force_host` evaluates one geometry through the public
NWChem frontend. `potserv_mpi` serves NWChem or CPMD requests with one socket on
rank zero and engine workers on the other ranks. The serial `potserv` and core
library retain their existing MPI-free load path.

Build the executables with Meson:

```sh
meson setup build -Dwith_rpc=true -Dwith_mpi=enabled
meson compile -C build nwchem_mpi_force_host potserv_mpi
```

The force host accepts a built-in H2, water, or benzene geometry. An explicit
engine path selects the shared library on every rank:

```sh
mpirun -np 4 build/CppCore/nwchem_mpi_force_host \
  --system water --basis sto-3g --engine /path/to/libnwchemc.so
```

Its output contains the energy in eV and the maximum absolute force component
in eV/angstrom. `ok=1` means every rank returned finite results. `wall_s` is the
maximum elapsed force-call time across the ranks. A geometry file starts with
the atom count, followed by one `Z x y z` line per atom; pass it with `--geom`.
Rank zero reads that file and sends the same coordinates and parameters to all
workers.

Start an MPI RPC server with the existing backend syntax:

```sh
export RGPOT_NWCHEM_ENGINE=/path/to/libnwchemc.so
mpirun -np 4 build/CppCore/potserv_mpi 12345 NWChem:sto-3g:scf
```

The RPC client connects to port 12345 and uses the same `PotentialConfig` and
`ForceInput` messages as the serial server. `CPMD` selects the CPMD backend.
CMake also builds and installs both executables when RPC and MPI are available.
These hosts use one `MPI_COMM_WORLD` for the engine request stream; calculator
groups continue to use the separate `rgpot::bindCalculators` interface.

A host that initializes MPI registers its exit handler before loading the
engine. Engine exit handlers therefore run while that runtime is live. A caller
that supplies an initialized runtime owns its finalization; borrowed
communicators are never freed by the host helper.

The command channel agrees geometry preparation and receive-allocation errors
before transferring a request. A force or configuration failure returned by a
worker reaches the RPC client, and workers can accept another request. Every
engine callback must return or throw on all participating ranks. Collective
operations inside the engine must follow its own matching-call contract.
