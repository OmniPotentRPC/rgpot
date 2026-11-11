# UMA cutoff follows the model

Base: `develop` at `96e910064885a6e81a7b54b5a09d063294823382`.

## What failed

gpridge P-RFO, baker/17_claisen, dimer seed 4, died after 850 force
calls. The edge log is `compiled_edges` 182, `cutoff_edges` 180,
`max_distance_A` 10.197616276524268, pair `[5, 11]`. The AOTI call
raised `index out of bounds: 0 <= tmp15 < 14L`. `rgpot_engine_force`
turned that exception into status 2 (`EngineTable.cc`, the `catch`
that returns 2). The harness files status 2 as `engine_limit`.

182 is `14 * 13`: every ordered pair of a 14-atom system. 180 is that
list with one unordered pair removed in both directions.

## Where the 10 angstrom cutoff is

Confirmed in the tree before the change.

`scripts/export_uma_aoti.py` (`--molecular-box L`) replaces the
backbone cutoff with `0.4 * L` and writes that number into the package
as `cutoff`. For the harness box of 25.0 angstrom, `0.4 * 25 = 10`.
The comment there says the wider cutoff exists only so a static graph
keeps `n(n-1)` edges. The model envelope is what zeroes pairs past the
backbone cutoff.

`UmaPot::ensureLoaded` copies metadata `cutoff` onto the neighbor
list. `forceImpl` and `forceBatchImpl` then call `vesin_fairchem_edges`
with that value. Vendored vesin keeps a pair only when
`distance^2 < cutoff^2` (`third_party/vesin/vesin-single-build.cpp`).
A pair at 10.1976 angstrom is outside 10. Both directions disappear.
`UmaContract::checkEdges` throws when `nedges_min == nedges_max` and
the count differs. A package without those keys reaches
`AOTIModelPackageLoader::run` with the short edge tensor, which is the
index error in the log.

No new cutoff constant belongs in the runtime. The interaction range
is the backbone cutoff, recorded as `model_cutoff`. A molecular
package whose compiled edge count is every ordered pair (or whose
edge count was not recorded) must emit that full list. A static count
that is shorter is padded with image edges past the recorded range,
which the envelope treats as absent.

## Geometry

`claisen-endpoints.xyz` is two 14-atom frames. The nearest contact
(C4–H12) is 0.5576 in the file. Times `BOHR_TO_ANGSTROM`
(0.529177210903) that is 0.295 angstrom. Neither reading is a C–H
bond, and the longest pair in either frame is 3.30 file units, so no
uniform scale can put atoms 5 and 11 at 10.1976 angstrom without
pushing other pairs past 10 angstrom as well. The dimer frame that
died is not in the file.

The regression geometry is frame 0, raw numbers kept as angstrom
(the unit of the harness distance and of the engine cutoff), with
atoms 5 and 11 moved equally along the line that joins them until
that distance is exact. One unordered pair then sits past 10
angstrom, so a 10 angstrom list has 180 directed edges.

## Change

`model_cutoff` is the backbone range, written by the exporter and
read by `UmaPot`. It is not a new constant. A molecular package whose
compiled count is `n(n-1)`, or which did not record a count, emits
every ordered pair. The 10 angstrom metadata value is not applied.
A static count that is not that full list is filled out with an image
of atoms 0 and 1 past the recorded range, or shortened by dropping
edges already past that range. The envelope makes either edit leave
the energy and forces unchanged.

`scripts/export_uma_aoti.py` no longer replaces the backbone cutoff
with `0.4 * L`. `--molecular-box` traces every ordered pair and
records `model_cutoff`.

## Probe

`CppCore/tests/uma_cutoff_probe.cc` builds the geometry, checks the
unit reading, and drives one `UmaPot` from 8 angstrom through the
logged distance to 11 angstrom. The package is
`scripts/export_cutoff_probe.py`: static 182 edges, metadata cutoff
10, envelope cutoff 6. Removing pairs at or past 6 angstrom from the
reference sum matches the engine. A central difference of the energy
at 9.5, 10.0 and the logged distance matches the reported force.

Against `UmaPot.cc` from `96e910064885a6e81a7b54b5a09d063294823382`
the sequence passes at 8, 9.5, 9.9 and 10 angstrom (the constructed
10 angstrom contact is still inside vesin's strict cutoff) and stops
at 10.197616276524268 angstrom with
`UmaPot: input nedges is 180, the package was exported for 182`.
Exit status 1.

After the change the same binary steps through 11 angstrom. Energy
and forces agree with the pair removed to about `1e-14`. The finite
difference stays under `2e-9`. Exit status 0.
