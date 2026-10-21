# xTB backends: linked vs dlopen

rgpot exposes two GFN-xTB force loading strategies over the same single-point
semantics (Å / eV host units; Bohr / Hartree inside libxtb).

| Backend | Class | Build | Runtime |
|---------|-------|-------|---------|
| **Linked** | `rgpot::XTBPot` | `-Dwith_xtb=true` (pkg-config `xtb`) | NEEDED `libxtb` in `libxtbpot` / host |
| **dlopen** | `rgpot::XTBDlopen` | same; builds `libxtb_engine.so` | `dlopen` engine; set path below |

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

A non-empty `XTBDlopenConfig::engine_path` is exclusive: that path is the only
candidate. If it does not open, or the library does not export the C ABI,
construction throws. `RGPOT_XTB_ENGINE`, the bare soname, and the directory
lists are not consulted, so a loader path (`LD_LIBRARY_PATH`,
`DYLD_LIBRARY_PATH`, or `PATH`) cannot substitute a different
`libxtb_engine`.

When `engine_path` is empty, search order is:

1. env `RGPOT_XTB_ENGINE`
2. env `XTB_ENGINE`
3. bare soname on the loader path (`libxtb_engine.so` on Linux,
   `libxtb_engine.dylib` then `libxtb_engine.so` on Apple,
   `xtb_engine.dll` then `libxtb_engine.dll` on Windows)
4. `EON_POTENTIALS_PATH` / `RGPOT_ENGINE_PATH` directory lists

## eOn ship baseline

eOn’s in-tree `client/potentials/XTBPot` with `-Dwith_xtb=true` is the
**linked ship** comparison target (same xtb C API + GFN2). It is not removed
or replaced by the rgpot engine plugin.

## Tests

```bash
pixi run -e xtbbld meson setup bbdir-xtb -Dwith_xtb=true -Dwith_tests=true \
  -Dwith_rpc=false -Dwith_cache=false --buildtype=debug
pixi run -e xtbbld meson compile -C bbdir-xtb
pixi run -e xtbbld meson test -C bbdir-xtb --suite xtb --print-errorlogs
```

Catch tags: `[xtb]`, `[xtb][linked]`, `[xtb][dlopen]`. Dlopen tests set
`RGPOT_XTB_ENGINE` to the built engine full path. `XTBDlopenTest` also puts
that engine on the process loader path and requires a missing explicit
`engine_path` to throw anyway.

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
