# Cell-list EAM

`EAMPot` evaluates the embedded-atom energy and forces for aluminium,
palladium and gold. The interaction is short ranged: the widest cutoff in
the element table is the aluminium value, 5.555 Angstrom.

## Neighbour list

The list is a linked-cell search with a Verlet skin. The skin is one fifth
of the widest cutoff (`kEamSkinOverCutoff`). A later call on the same
object reuses the list while every atom has moved by at most half of that
distance and the cell, the atom count and the species are unchanged.
Pairs are visited in atom-index order. One thread therefore performs the
same arithmetic, on the same pairs, as the direct double loop, and the
energy and forces match that loop bit for bit.

The stored list is the neighbours inside the cutoff plus the skin. On a
bulk aluminium crystal that is a few dozen neighbours per atom, not one
slot per atom pair.

## Threads

`OMP_NUM_THREADS` is read on each call. When it is unset or empty the call
stays on one thread. A value above 1 splits the atoms across that many
OpenMP threads. Each thread writes only the forces of its own atoms.
Partial energies are added in thread-id order, so a fixed thread count
repeats bit for bit.

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
seven warm calls that reuse the neighbour list. The scaling rows are the
median of five such medians.

Develop still walked every pair and allocated a neighbour table of
`N * (N + 1)` integers, which is about 8 GB at 32 000 atoms and about
93 GB at 108 000.

| atoms | develop (s) | this branch (s) | speedup |
| ---: | ---: | ---: | ---: |
| 108 | 0.000122 | 0.000109 | 1.12 |
| 500 | 0.001270 | 0.000535 | 2.37 |
| 2048 | 0.016273 | 0.002089 | 7.79 |
| 6912 | 0.170350 | 0.006930 | 24.6 |
| 32000 |  | 0.031521 |  |
| 108000 |  | 0.106643 |  |

The first call at 6912 atoms dropped from 2.181 s to 0.0418 s. That call
used to build the full pair table and then ignore it.

Strong scaling, 108 000 atoms:

| threads | median (s) | speedup |
| ---: | ---: | ---: |
| 1 | 0.106276 | 1.00 |
| 2 | 0.088772 | 1.20 |
| 4 | 0.045008 | 2.36 |
| 8 | 0.023592 | 4.51 |

Weak scaling, 4000 atoms per thread:

| threads | atoms | median (s) |
| ---: | ---: | ---: |
| 1 | 4000 | 0.003986 |
| 2 | 8000 | 0.006511 |
| 4 | 16000 | 0.006674 |
| 8 | 32000 | 0.006821 |

From 2 threads upward the weak-scaling time stays near 6.7 ms. The step
from 1 to 2 threads is larger because the threaded evaluation computes
the density and the force in separate passes.

The same host, before this change, put the other classical potentials in
a different range. At about 7000 to 14 000 atoms a warm call was 4.6 ms
for Lennard-Jones (cutoff 2.5), 12 ms for periodic Sutton-Chen (cutoff
3), 8.3 ms for Stillinger-Weber and 12 ms for Tersoff. At about 100 000
atoms those four were 69 ms, 160 ms, 60 ms and 88 ms. The cell-list EAM
at 108 000 atoms is 107 ms on one thread, in that same band.

## Reproduce

```bash
g++ -std=c++20 -O3 -DNDEBUG -fopenmp -I CppCore -I CppCore/rgpot/EAM \
  -DEAM_STANDALONE \
  CppCore/tests/eam_cell_bench.cc CppCore/rgpot/EAM/EAM.cpp \
  -o eam_cell_bench
./eam_cell_bench
```

Meson builds the same program as the `eam_cell_bench` benchmark.
