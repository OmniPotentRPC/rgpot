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
| *(later)* `xtb` / `tblite` | … | XTBPot / TBLitePot |

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

### Host-only paths and process teardown

- `NWChemParams.enginePath` / `nwchemRoot` locate the DSO and set env hints in
  **this** process. They are stripped before `nwchemc_set_params` /
  `nwchemc_energy_gradient` (the embed ABI rejects non-empty `enginePath`).
- `libnwchemc` registers `atexit(nwchemc_finalize)`. The engine is opened with
  `RTLD_NODELETE`, and `~NWChemPot` calls `nwchemc_finalize` while the handle is
  still valid, so process exit does not SIGSEGV into an unmapped DSO.

## Layers

| Piece | Built when | Role |
|-------|------------|------|
| `NWChemPot.cc` frontend | always | Serialize `NWChemParams`, `dlopen` engine, units to eV/Angstrom |
| `nwchem_c_abi.h` | header | stable consumer C-ABI contract (`nwchemc_*`) |
| `nwchem_c_abi_stub.c` | always | no-op ABI: `nwchemc_available()==0`, used by the ABI conformance test |

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
# Frontend only: always builds. Resolves libnwchemc.so by dlopen at runtime.
meson setup bbdir -Dwith_rpc=false
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
| `theory` | `scf` | Method: `scf`, `dft`, `blyp`, `b3lyp`, … |
| `scfType` | `rhf` | HF: `rhf`/`uhf`; with DFT: XC functional (`blyp`, …) |
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
