# CPMDPot

`CPMDPot` is the rgpot frontend for CPMD calculations. It is the C++ and RPC
side of the integration, then loads a split `libcpmdc` engine at runtime and
calls the stable C ABI declared in `cpmd_c_abi.h`.

Configuration and geometry stay separate:

| Carrier | Owner | Role |
| --- | --- | --- |
| `PotentialConfig.cpmd` | rgpot RPC clients | selects the CPMD backend arm |
| `CPMDParams` | rgpot clients, forwarded to `cpmdc` | method and backend setup |
| `ForceInput` | every `calculate()` call | coordinates, atomic numbers, box, units |
| `PotentialResult` | `cpmdc` session result ABI, returned by rgpot | energy and forces |

`CPMDPot` does not render CPMD decks itself. It serializes `CPMDParams`, passes
the flat Cap'n Proto bytes into `libcpmdc`, and lets the engine merge those
params with each `ForceInput`.

## Minimal RPC Path

Build `libcpmdc` in the split engine repository:

```bash
git clone https://github.com/OmniPotentRPC/cpmdc.git
cd cpmdc
meson setup build -Dwith_tests=true
meson compile -C build
```

Build rgpot and run the CPMD configure smoke with that engine on the loader
path:

```bash
pixi run -e rpctest meson setup bbdir -Dwith_tests=true -Dwith_rpc=true
pixi run -e rpctest meson compile -C bbdir
CPMDC_LIBRARY=/path/to/cpmdc/build/libcpmdc.so \
  pixi run -e rpctest python tests/rpc_integ.py \
    --server-bin ./bbdir/CppCore/potserv \
    --cpmd-smoke
```

That smoke covers RPC startup and `configure(PotentialConfig.cpmd)`. A real
calculation still sends geometry through `calculate(ForceInput)`; coordinates do
not belong in `CPMDParams`.

## In-process

`rgpot::makePotential` builds a `CPMDPot` from the `cpmd` arm of a
`PotentialConfig`. The call does not start `potserv`.

`try_load_engine` in `CppCore/rgpot/CPMDPot/CPMDPot.cc` loads `libcpmdc`
at run time. It opens the library and resolves `cpmdc_session_create`
when that symbol is present. Set `CPMDC_LIBRARY`, or
`CPMDParams.enginePath`, before the process starts, as in the RPC smoke
above.

```cpp
#include <cstdio>

#include <capnp/message.h>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/PotentialFactory.hpp"

int main() {
  capnp::MallocMessageBuilder msg;
  auto cfg = msg.initRoot<PotentialConfig>();
  auto params = cfg.initCpmd();
  params.setFunctional("BLYP");
  params.setCutOffRy(70.0);
  auto pot = rgpot::makePotential(cfg.asReader());

  const double pos[9] = {0.0, 0.0,      0.11779,
                         0.0, 0.75545,  -0.47116,
                         0.0, -0.75545, -0.47116};
  const int z[3] = {8, 1, 1};
  const double box[9] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  double forces[9] = {};
  rgpot::ForceInput in{3, pos, z, box};
  rgpot::ForceOut out{};
  out.F = forces;
  pot->forceBatch(rgpot::ForceBatch{1, &in, &out});
  std::printf("energy %.6f eV\n", out.energy);
  return 0;
}
```

The geometry is the water example from the CPMD how-to, in a 20 Angstrom
cube, with BLYP and a 70 Ry cutoff. The program needs a build with RPC
on, because the frontend and `makePotential` use the Cap'n Proto schema.
A force with no `libcpmdc` ends in `MPI_Abort` on an MPI build.

## Engine Loading

`CPMDPot` probes engine paths in this order:

| Source | Meaning |
| --- | --- |
| `CPMDParams.enginePath` | explicit path carried in the config message |
| `CPMDC_LIBRARY` | process-level engine path |
| `RGPOT_CPMDC_ENGINE` | rgpot-specific engine path |
| `RGPOT_CPMD_ENGINE` | compatibility engine path |
| `libcpmdc.so`, `./libcpmdc.so`, `libcpmdc.dylib`, `./libcpmdc.dylib`, `cpmdc.dll` | default probe names |

`CPMDParams.cpmdRoot` is applied to `CPMD_ROOT` before the engine is loaded.

The frontend requires the feature-discovery ABI:
`cpmdc_feature_count`, `cpmdc_feature_table`, and `cpmdc_feature_find`. It then
uses the session-result ABI when available:
`cpmdc_session_create`, `cpmdc_potential_result_size_for_force_input`, and
`cpmdc_session_calculate_result`. Older engines can fall back to the one-shot
energy-gradient ABI if they expose it.

The box passed to a force call is the cell CPMD uses. With a cpmdc newer
than 0.3.0 a non-zero box replaces the `CELL` and `SYMMETRY` of the deck
(`SYMMETRY 1` for a cubic deck and box, `SYMMETRY 8` for another
orthorhombic box, `CELL VECTORS` otherwise). An isolated deck
(`SYMMETRY 0`) takes only an orthorhombic box. An all-zero box, the
convention of hosts that pass no cell for a non-periodic system, keeps
the deck's `CELL`.

`CPMDPot::selectOrbitals(key)` applies the key to the live engine session at
call time and re-applies it after session recreation. A caller can therefore
issue `selectOrbitals` calls before the next force without ambiguity. When the
engine exports `cpmdc_session_select_orbitals`, it keeps converged orbitals per
key, so each SCF starts from that key's own previous orbitals. Every rank of a
calculator names the same key before the same force. If the engine refuses the
key while a session is being created, that session is dropped, and the next
`setParams` with the same parameters creates another and applies the key again.

`CPMDPot::copyOrbitals(from, to)` requests an explicit copy from one key to
another through `cpmdc_session_copy_orbitals` and returns false when the engine
does not provide that primitive.

## `CPMDParams` Fields

| Field | Default | Role |
| --- | --- | --- |
| `functional` | `BLYP` | DFT functional default |
| `cutOffRy` | `70.0` | plane-wave cutoff in Ry |
| `charge` | `0` | system charge |
| `multiplicity` | `1` | spin multiplicity |
| `task` | `gradient` | method task hint |
| `title` | empty | optional CPMD title |
| `memoryMb` | `0` | frontend memory hint |
| `scratchDir` | empty | CPMD file placement hint |
| `permanentDir` | empty | preferred CPMD file placement hint |
| `cpmdRoot` | empty | `CPMD_ROOT` hint |
| `enginePath` | empty | explicit `libcpmdc` path |
| `inputBlocks` | empty | raw CPMD input blocks |
| `inputSections` | empty | typed or raw CPMD input sections |

`inputSections` supports `generic`, `system`, `cpmd`, `dft`, `atoms`, `set`,
and `raw` arms. Prefer typed arms where they exist, then use `generic`, `set`,
or `raw` for CPMD input that has no typed rgpot schema yet.

Typed `atoms` sections carry pseudopotential entries keyed by element symbol.
The engine groups `ForceInput` coordinates into `&ATOMS`; every atomic number
in a step must have a matching pseudopotential entry. When `atoms` is omitted,
the engine's built-in BLYP defaults cover H and O only. The same section also
accepts structured `directives` for non-coordinate `&ATOMS` keywords.

## Python Client Helpers

`tests/cpmd_params.py` builds `CPMDParams` and `PotentialConfig.cpmd` messages
for pycapnp clients. It reads `CPMD_ROOT`, `CPMDC_LIBRARY`,
`RGPOT_CPMDC_ENGINE`, and `RGPOT_CPMD_ENGINE` when explicit values are not
provided.

```python
params = make_cpmd_params(
    pot_capnp,
    functional="PBE0",
    cut_off_ry=88.0,
    input_sections=[
        {"kind": "generic", "name": "PIMD",
         "directives": [{"keyword": "TEMP", "args": ["300"]}]},
        {"kind": "system",
         "cell": [8.0, 8.0, 9.0, 90.0, 90.0, 120.0],
         "charge": 1,
         "multiplicity": 3,
         "scale": 0.5},
        {"kind": "cpmd",
         "optimizeGeometry": True,
         "molecularDynamics": True,
         "molecularDynamicsCp": True,
         "molecularDynamicsBo": True,
         "molecularDynamicsEh": True,
         "molecularDynamicsPt": True,
         "molecularDynamicsClassical": True,
         "molecularDynamicsFile": "TRAJECTORY.in",
         "convergenceGeometry": 1.0e-4,
         "maxStep": 8,
         "maxIter": 12,
         "electronMass": 450.0,
         "nose": True,
         "noseIons": True,
         "noseElectrons": True,
         "berendsen": "300 100",
         "langevin": True,
         "annealing": "IONS 300 50",
         "quench": True,
         "rattle": True,
         "shake": True,
         "constraint": "FIX COM",
         "trotter": "8",
         "restart": True,
         "printOptions": "FORCES ON",
         "storeOptions": "WAVEFUNCTION",
         "centerMoleculeOff": True,
         "centerMoleculeOn": True,
         "diis": True,
         "odiis": True,
         "pcg": True,
         "diagonalization": True,
         "freeEnergy": True,
         "interface": True,
         "qmmm": True,
         "bicanonicalEnsemble": True,
         "cdft": True,
         "properties": True,
         "restartWavefunction": True,
         "trajectory": True},
        {"kind": "dft",
         "functional": "PBE0",
         "lsd": True,
         "gcCutoff": 1.0e-8,
         "xcDriver": "LIBXC",
         "libxc": "GGA_X_PBE GGA_C_PBE",
         "lrKernel": "PBE",
         "refunct": "PBE",
         "mtsHighFunc": "PBE0",
         "mtsLowFunc": "PBE",
         "hfx": True,
         "hfxScreening": "0.2",
         "hubbard": "U 1 4.0",
         "alpha": 0.25,
         "beta": 0.75,
         "oldCode": True,
         "newCode": True,
         "correlation": "LYP",
         "exchange": "B88",
         "becke88": True},
        {"kind": "atoms",
         "pseudopotentials": [
             {"element": "Si", "path": "Si_MT_PBE.psp", "lmax": 2},
         ],
         "directives": [{"keyword": "ISOLATED MOLECULE", "args": []}]},
        {"kind": "set", "key": "CPMD.MAXSTEP", "value": "12"},
        {"kind": "raw", "text": "&VDW\n  DISPERSION\n&END"},
    ],
)
```

To configure a running RPC server:

```python
cfg = make_potential_config_cpmd(pot_capnp, functional="BLYP")
result = await pot.configure(cfg)
```

`configure(cpmd)` returns `ok=false` when no usable `libcpmdc` engine can be
loaded or when the engine rejects the params. `configure(none)` is a no-op and
does not prove that CPMD evaluation is available.

There is no session open, close or reset RPC method.
Restart `potserv` to drop the engine session with the process.
`configure` with changed `CPMDParams` forces a new session when the engine
has the session result ABI.
The same serialized params keep the live session.
`configure(none)` does not reset it.

## Local Checks

```bash
pixi run -e rpctest python tests/test_cpmd_params.py
pixi run -e rpctest python tests/test_rpc_integ_cpmd.py
pixi run -e rpctest meson test -C ./bbdir --suite cpmd --print-errorlogs
pixi run -e rpctest python tests/rpc_integ.py \
  --server-bin ./bbdir/CppCore/potserv \
  --cpmd-smoke
```

The `--cpmd-smoke` path exercises RPC `configure()` plumbing. `calculate()`
requires an available `libcpmdc` engine.
