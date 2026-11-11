# Cell-list EAM

`EAMPot` evaluates the embedded-atom energy and forces for aluminium,
palladium and gold. The interaction is short ranged: the widest cutoff in
the element table is the aluminium value, 5.555 Angstrom.

## Neighbour list

The list is the shared neighbour layer (`rgpot::nlist::Neighbours`): a
linked-cell search with a Verlet skin. The skin is one fifth of the widest
cutoff (`kEamSkinOverCutoff`). A later call on the same object reuses the
list while every atom has moved by at most half of that distance and the
cell, the atom count and the species are unchanged. The minimum image is
one half-box correction after the coordinates have been wrapped into the
cell. A streaming evaluation walks the cells and stores no pair list.
Pairs are visited in atom-index order. One thread therefore performs the
same arithmetic, on the same pairs, as the direct double loop, and the
energy and forces match that loop bit for bit.

The stored list is the neighbours inside the cutoff plus the skin. On a
bulk aluminium crystal that is a few dozen neighbours per atom, not one
slot per atom pair.

## Threads

`OMP_NUM_THREADS` is read on each call. When it is unset or empty the call
stays on one thread. A value above 1 splits the atoms across that many
OpenMP threads. Each thread writes only the forces of its own atoms. Density, the
pair potential and that atom's own embedding are one pass. A second
pass adds only the neighbour's embedding, using the scale stored on
the first pass. Partial energies are added in thread-id order, so a
fixed thread count repeats bit for bit.

Several threads are not bit-identical to one thread. Energy and every
force component stay within a relative tolerance of `1e-9`
(`kEamThreadedTolerance`). A component whose one-thread magnitude is below
1 is compared on an absolute scale of the same size. `EAMCellVerletTest`
checks the one-thread bits, the skin rebuild, the linear list, and the
threaded tolerance.

## Timings

FCC aluminium, lattice constant 4.05 Angstrom, on one Intel Xeon socket
with 8 cores and 1 thread per core. g++ 13.3.0, `-O3 -DNDEBUG`. The new
binary is also built with `-fopenmp`. Each quoted time is a median of
seven warm calls that reuse the neighbour list.

Develop still walked every pair and allocated a neighbour table of
`N * (N + 1)` integers, which is about 8 GB at 32 000 atoms and about
93 GB at 108 000.

| atoms | develop (s) | this branch (s) | speedup |
| ---: | ---: | ---: | ---: |
| 108 | 0.000122 | 0.000112 | 1.09 |
| 500 | 0.001270 | 0.000547 | 2.32 |
| 2048 | 0.016273 | 0.002130 | 7.64 |
| 6912 | 0.170350 | 0.007174 | 23.7 |
| 32000 |  | 0.032605 |  |
| 108000 |  | 0.109318 |  |

The first call at 6912 atoms dropped from 2.181 s to 0.0418 s. That call
used to build the full pair table and then ignore it.

Strong scaling of the stored list, 108 000 atoms:

| threads | median (s) | speedup |
| ---: | ---: | ---: |
| 1 | 0.109320 | 1.00 |
| 2 | 0.075928 | 1.44 |
| 4 | 0.038904 | 2.81 |
| 8 | 0.036770 | 2.97 |

At 32 000 atoms the same list reaches 3.87x on 8 threads (0.008380 s
against 0.032441 s). Two threads are 1.66x there. The earlier separate
density and force passes were 1.20x at two threads on 108 000 atoms.
The cross-potential tables, both neighbour modes, and the weak-scaling
rows are in `docs/neighbour_layer.md`.

## Reproduce

```bash
g++ -std=c++20 -O3 -DNDEBUG -fopenmp -I CppCore -I CppCore/rgpot/EAM \
  -DEAM_STANDALONE \
  CppCore/tests/eam_cell_bench.cc CppCore/rgpot/EAM/EAM.cpp \
  -o eam_cell_bench
./eam_cell_bench
```

Meson builds the same program as the `eam_cell_bench` benchmark.
