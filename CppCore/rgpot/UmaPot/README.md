# UmaPot: UMA / OMol AOTInductor frontend

`UmaPot` loads a compiled AOTInductor package (`.pt2`) that
`scripts/export_uma_aoti.py` wrote from a UMA checkpoint.

vesin builds the neighbor list. The graph returns energy and forces.
Charge and spin are per-call tensors. The compiled graph does not
call fairchem at evaluate time.

## Package

One `.pt2` per composition, charge and spin. The exporter embeds the
runtime contract in the package itself
(`aot_inductor.metadata`). `UmaPot` reads it with
`AOTIModelPackageLoader::get_metadata()`; no file next to the
package is read.

| Key | Use at runtime |
| --- | --- |
| `cutoff` | Model interaction range when `model_cutoff` is absent. Vesin radius for a package that was not traced as every ordered pair. |
| `model_cutoff` | Backbone interaction range. Pairs past it do not change the energy. Older packages omit the key. |
| `max_neighbors` | Vesin cap per atom, on a list that is not the full ordered pair list |
| `molecular_box` | Cube side length; positions are centred at zero (0: caller's cell). A positive value with a full-pair edge count (or with no recorded edge count) builds every ordered pair, not a cutoff truncation. |
| `batch_max` | band graph size (0 or 1: single system) |
| `pos_dtype` | `float64` switches the input dtype |
| `task_name`, `charge`, `spin` | must equal `UmaConfig` |
| `z_set`, `natoms`, `counts` | must match the input atoms |
| `label`, `shapes`, `inputs`, `outputs` | recorded |
| `model`, `torch_version`, `fairchem_version`, `export_path` | recorded |

`UmaConfig.cutoff` / `max_neighbors` are defaults. The embedded
values win when present. A molecular package traced at `n(n-1)`
edges keeps that count for every geometry. The envelope, not a
10 angstrom neighbor cutoff, drops pairs past the model range.

## Contract checks

`merge_mole` folds one composition, charge and spin into the graph.
A different charge or spin evaluates another potential energy
surface without any error; a different composition aborts inside
the graph. Every force call therefore checks the config and the
input against the embedded keys and throws `rgpot::UmaContractError`
(a `std::runtime_error`; `field()` names the key) on a mismatch:

```
UmaPot: config charge is -1, the package was exported for 0
UmaPot: input counts is {1: 4, 6: 2}, the package was exported for {1: 2, 6: 2}
```

`natoms` and `counts` exist only in packages from the current
exporter. Older packages get the `z_set` check, which cannot tell
C2H2 from C2H4.

`scripts/export_baker_uma_aoti.py` walks Baker endpoints and
deduplicates by exact composition (atom count per element), charge
and spin.

## Export path

`export_uma_aoti.py` tries `torch.export` (dynamic nonstrict, static
nonstrict, static strict) and falls back to
`make_fx(tracing_mode="real")` plus a nonstrict export of the traced
module. With torch 2.13 and fairchem-core 2.23 every UMA export takes
the fallback, and `export_path` reads `make_fx-nonstrict`. That
package is static in atom count and edge count, so export with
`--molecular-box` (every ordered pair, `n(n-1)` edges at every
geometry). The trace does not widen the neighbor cutoff to `0.4*L`.
Without the molecular box the exporter warns: a geometry that moves
a pair across the model cutoff changes the edge count and the call
fails unless the list is padded back to the compiled count.

## potserv

```
Uma:<model.pt2>
Uma:<model.pt2>:<task>
```

Task default is `omol`. Device default is `cpu`.

## Build

Needs torch and vesin (`-Dwith_uma=true`, pixi env that has both).
`libuma_engine.so` is the dlopen plugin for eonclient: the host
never links torch. Config is a Cap'n Proto `UmaParams` message.

Molecular graphs use a large constant box so the neighbor list
shape is stable. Do not treat that box as a periodic crystal.

Elja / some HPC loaders need the AOTI execstack rewrite in
`aoti_execstack.hpp` (`docs/newsfragments/+uma-elja-execstack.fixed.md`).
