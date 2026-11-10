# UmaPot: universal model for atoms (UMA) / OMol AOTI frontend

`UmaPot` loads an AOTI (AOTInductor, PyTorch ahead-of-time
compiled) package (`.pt2`) that `scripts/export_uma_aoti.py`
wrote from a UMA checkpoint.

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
| `cutoff`, `max_neighbors` | vesin neighbor list |
| `molecular_box` | cube side length; molecular positions are centred at zero (0: caller's cell) |
| `batch_max` | band graph size (0 or 1: single system) |
| `pos_dtype` | input dtype of positions, the cell and cell offsets (`float32` or `float64`) |
| `model_dtype` | weight dtype recorded by the exporter (`float32` or `float64`) |
| `task_name`, `charge`, `spin` | must equal `UmaConfig` |
| `z_set`, `natoms`, `counts` | must match the input atoms |
| `label`, `shapes`, `inputs`, `outputs` | recorded |
| `model`, `torch_version`, `fairchem_version`, `export_path` | recorded |

`UmaConfig.cutoff` / `max_neighbors` are defaults. The embedded
values win when present.

The exporter converts the model with `hydra.double()` before the
AOTI trace for `--dtype float64` and records both dtype keys as
`float64`. `--dtype float32` calls `hydra.float()` and records
`float32`. A package with neither key falls back to float32, and the
load line keeps a missing key as `unset` rather than inventing it.
`pos_dtype` and `model_dtype` must name the same dtype; a disagreement
throws before any force call.

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
UmaPot: edge_index shape is [2, 42], the package was exported for [2, 56]
```

Positions and the edge list are checked against the shape the package
declares before the compiled graph is called. A static package names
the traced sizes (`[8, 3]` and `[2, 56]` for an eight-atom complete
graph). A package that was exported with a symbolic edge count names
that axis and its range. A mismatch throws `rgpot::UmaShapeError`.
An edge index outside the position buffer throws the same error and
is not gathered. `pos_shape` and `edge_index_shape` record the
declaration; when they are absent the atom count and the edge range
are used. A package that records neither `nedges_min` nor
`nedges_max` is refused: there is no range to check, and a static
graph would otherwise return a finite force.

A molecular box recenters the atoms in a cube and is traced on every
ordered pair, `n(n-1)`. Any other edge count throws `UmaShapeError`
before the compiled graph runs. The single path and the band path
use that same check. A band whose edge range is not a single number
still checks each system against the range. Any nonzero cell offset
is a periodic image, so the call throws `UmaContractError` on
`cell_offsets` (`nonzero`).

`natoms` and `counts` exist only in packages from the current
exporter. Older packages get the `z_set` check, which cannot tell
C2H2 from C2H4.

`scripts/export_baker_uma_aoti.py` walks Baker endpoints and
deduplicates by exact composition (atom count per element), charge
and spin.

## Export path

`export_uma_aoti.py` tries a symbolic edge count, then a dynamic
export, then a static one, then `make_fx(tracing_mode="real")`.
Every path that is kept records the traced edge count as both
`nedges_min` and `nedges_max`, plus the molecular box and a
molecular flag. A path does not record 1 to 65536. `symbolic-nedges`,
`dynamic-nonstrict` and `make_fx-nonstrict` without a molecular box
are refused and write no package. A static path with no box records
the traced count, so a later call whose edge list changed size
throws `UmaShapeError` and names both shapes. The atom count stays
the traced size.

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
