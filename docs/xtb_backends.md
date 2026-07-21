# xTB backends: linked vs dlopen

rgpot exposes two GFN-xTB force loading strategies over the same single-point
semantics (Å / eV host units; Bohr / Hartree inside libxtb).

| Backend | Class | Build | Runtime |
|---------|-------|-------|---------|
| **Linked** | `rgpot::XTBPot` | `-Dwith_xtb=true` (pkg-config `xtb`) | NEEDED `libxtb` in `libxtbpot` / host |
| **dlopen** | `rgpot::XTBDlopen` | **always** (no libxtb at link time); engine needs `with_xtb` to *build* | `dlopen` `libxtb_engine.so`; set path below |

`GFNMethod` / `XTBConfig` / `XTBDlopenConfig` live in `XTBConfig.hpp` (no
`#include <xtb.h>`), so hosts can compile against the dlopen frontend without
pkg-config xtb. Linked `XTBPot.hpp` still includes libxtb and requires
`-Dwith_xtb=true`.

Default method: **GFN2-xTB** (`GFNMethod::GFN2xTB` / `RGPOT_XTB_METHOD_GFN2`).

## ISO_C_BINDING C API (libxtb)

libxtb is implemented in Fortran but the public surface used here is the
**ISO_C_BINDING** C API (`xtb.h`): interoperable types, `bind(c)` procedures,
and opaque C pointers. Hosts (linked `XTBPot` and `libxtb_engine.so`) only call
that C contract.

Contract tests care about:

| Call pattern | C API |
|--------------|--------|
| Create | `xtb_newEnvironment` / `newCalculator` / `newResults` (+ `xtb_releaseOutput`) |
| First geometry | `xtb_newMolecule` + `xtb_loadGFN*xTB` |
| Updates | `xtb_updateMolecule` then `xtb_singlepoint` |
| Destroy | `xtb_del*` in reverse order |
| Units | host Å/eV; API Bohr/Hartree (converted in pot) |

Catch tags: `[xtb][capi]` — multi-handle lifecycle, warm update, method switch,
linked↔dlopen parity through the same ISO C semantics.

## Engine selection (`XTBDlopen`)

Search order for `libxtb_engine.so`:

1. `XTBDlopenConfig::engine_path`
2. env `RGPOT_XTB_ENGINE`
3. env `XTB_ENGINE`
4. bare `libxtb_engine.so` (loader path)
5. `EON_POTENTIALS_PATH` / `RGPOT_ENGINE_PATH` directory lists

Construct throws if no engine is found (same pattern as `MetatomicDlopen`).

## eOn / thin host ship

eOn and other thin hosts build with `-Dwith_xtb=false` and still link
`XTBDlopen`. Provide `libxtb_engine.so` (built once in an xtb-enabled tree or
install) via `RGPOT_XTB_ENGINE` at runtime. No NEEDED `libxtb` on the host.

eOn’s in-tree `client/potentials/XTBPot` with `-Dwith_xtb=true` remains the
**linked ship** comparison target (same xtb C API + GFN2).

## Tests

```bash
pixi run -e xtbbld meson setup bbdir-xtb -Dwith_xtb=true -Dwith_tests=true \
  -Dwith_rpc=false -Dwith_cache=false --buildtype=debug
pixi run -e xtbbld meson compile -C bbdir-xtb
pixi run -e xtbbld meson test -C bbdir-xtb --suite xtb --print-errorlogs
```

Catch tags: `[xtb]`, `[xtb][linked]`, `[xtb][dlopen]`. Dlopen tests set
`RGPOT_XTB_ENGINE` to the built engine full path.

Thin-host compile check (no libxtb):

```bash
meson setup bbdir-thin -Dwith_xtb=false -Dwith_tests=true \
  -Dwith_rpc=false -Dwith_cache=false
meson compile -C bbdir-thin
# XTBDlopen is linked; constructing it without RGPOT_XTB_ENGINE throws.
```

## Timing compare

rgpot microbench (linked + dlopen, warm SCF state):

```bash
export RGPOT_XTB_ENGINE=$PWD/bbdir-xtb/CppCore/libxtb_engine.so
./bbdir-xtb/CppCore/xtb_backend_bench --warmup 5 --iters 50 \
  --json /tmp/rgpot_xtb_internal.json
```

Full head-to-head vs eOn ship (subprocess JSON merge):

```bash
bash scripts/run_xtb_backend_bench.sh --out-dir /path/to/scratch
```

See `scripts/compare_xtb_backends.py` for flags. Protocol: shared water GFN2
geometry, warmup force calls, then timed samples; report mean wall ms and
whether dlopen is as-fast-or-faster than eOn linked ship (5% band).
