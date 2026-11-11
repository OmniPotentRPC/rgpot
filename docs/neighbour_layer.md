# Shared neighbour layer

The public query is `rgpot::nlist::Query`, `Builder`, `Table` and `Walk`.
Classical potentials still use `rgpot::nlist::Neighbours` for a
minimum-image force walk. Metatomic and UMA are adapters: they ask this
query for pairs and then copy into the layout their model expects.

## What was taken, and why

The stored table is a structure of arrays: central index, partner index,
three integer cell shifts, three vector components, one distance, and a
mask. That is the layout a SIMD kernel or a GPU fill can read without
gathering fields out of a struct. LAMMPS Kokkos already stores neighbours
as a Kokkos view of indices plus a per-atom count
([neigh_list_kokkos.h](https://raw.githubusercontent.com/lammps/lammps/develop/src/KOKKOS/neigh_list_kokkos.h)).
Vesin returns parallel arrays of pairs, shifts, distances and vectors, and
records which device owns them (`third_party/vesin/vesin.h`,
`VesinNeighborList`). Freud stores query index, point
index and a flat distance array
([NeighborList.h](https://raw.githubusercontent.com/glotzerlab/freud/main/freud/locality/NeighborList.h)).
An array of pair structs would fight those consumers.

Half and full lists follow vesin's rule, including the self-image test
that drops a negative shift of an atom with itself
(`third_party/vesin/vesin-single-build.cpp`, the `options.full` branch).
A half list is the default in LAMMPS
([neigh_request.h](https://raw.githubusercontent.com/lammps/lammps/develop/src/neigh_request.h)
sets `half` and can derive a half list from a full one) and in HOOMD
([NeighborList.h](https://raw.githubusercontent.com/glotzerlab/hoomd-blue/trunk/hoomd/md/NeighborList.h),
`half` stores `i < j`, `full` stores every neighbour). ASE's `bothways`
flag is the same choice
([neighborlist.py](https://gitlab.com/ase/ase/-/raw/master/ase/neighborlist.py)).
Cluster4 is the extra format. GROMACS does not store one pair per atom:
an i-cluster of atoms faces packed j-clusters, with a bitmask of which
lanes interact
([pairlist.h](https://gitlab.com/gromacs/gromacs/-/raw/main/src/gromacs/nbnxm/pairlist.h),
`nbnxm_cj_t` and `NBNXM_INTERACTION_MASK_ALL`). Cluster4 is that idea at
the smallest width that is still a mask: four consecutive central atoms,
one partner, one shift, and four bits. The vector and the distance belong
to the lowest set bit. A plain half list cannot feed that kernel without
a regroup.

Strict means distance strictly below the cutoff, which is vesin's
`distance2 < cutoff2` and torch-nl's `strict_nl` mask
([neighbor_list.py](https://raw.githubusercontent.com/felixmusil/torch_nl/main/torch_nl/neighbor_list.py)).
Loose means distance below cutoff + skin. The skin is the Verlet buffer.
HOOMD rebuilds when a particle moves more than half the buffer. ASE
rebuilds when the largest squared displacement exceeds `skin**2`, which
is twice that far and can miss a pair. The builder here uses half the
skin, the distance that keeps every pair that was inside the cutoff.

A stream does not allocate the pairs of the whole system. `Walk::pull`
fills a caller buffer for one central atom, or one cluster of four, and
the caller may stop. JAX-MD cannot do this: a jit-compiled list has a
static capacity, `neighbor_fn.allocate` sizes it, and `did_buffer_overflow`
tells the caller to allocate again
([partition.py](https://raw.githubusercontent.com/jax-md/jax-md/main/jax_md/partition.py)).
That is the right answer when the consumer is XLA. It is the wrong answer
when the consumer is a force kernel that can run during the walk. The C
function `rgpot_nlist_build` still reports the full count when the caller
buffer is short, so a fixed buffer can grow the way JAX-MD does, without
making that the only path. `Auto` stores when a bound on the pair count
fits in `MemAvailable` and otherwise leaves the table empty.

Cell shifts are integers along the three cell vectors, and the vector is
`r_j - r_i + S H`, which is the sentence in vesin's `VesinNeighborList`
comment. Partial periodicity leaves the shift on an open axis at zero.
The search bound is the cutoff divided by the cell height perpendicular
to each face, so a small triclinic cell returns every image inside the
sphere, not one minimum image. ArborX's half neighbour list is a
bounding-volume hierarchy query into Kokkos offsets and indices
([ArborX_NeighborList.hpp](https://raw.githubusercontent.com/kokkos/ArborX/master/src/spatial/detail/ArborX_NeighborList.hpp)).
cuNSearch keeps a spatial grid on the device and rebuilds it with
`update_point_sets`
([cuNSearch.h](https://raw.githubusercontent.com/InteractiveComputerGraphics/cuNSearch/master/include/cuNSearch.h)).
Neither is a cell list, and neither speaks triclinic shifts. This build
does not fill a GPU. `Table::device` is 0. The arrays are the contract a
later device fill would write.

Threads, when `OMP_NUM_THREADS` is set, take contiguous atoms and the
results are concatenated in thread order, so the stored order matches one
thread. Vesin merges per-thread lists without promising that order.

Matscipy's `neighbour_list` returns whichever of pairs, distances and
vectors the caller names in a string
([neighbours.py](https://raw.githubusercontent.com/libAtoms/matscipy/master/matscipy/neighbours.py)).
`Query::fields` is that choice with a fixed set of arrays instead of a
format string.

Metatomic is not the layout of this table. Its labels are one row of
`(first_atom, second_atom, cell_shift_a, cell_shift_b, cell_shift_c)` and
its values are a tensor of shape `(pairs, 3, 1)`. `MetatomicPot` reads
`full_list` and `strict` off the model's request, calls `build`, wraps
the vector array with `torch::from_blob`, and copies only the five label
columns. UMA's edge list does the same and keeps its own sort.

## C ABI and Cap'n Proto

`rgpot_nlist.h` is version 1 (`rgpot_nlist_abi_version`). The request and
the pair struct grow only at the end. `rgpot_nlist_build` writes a stored
list. `rgpot_nlist_walk_open` / `rgpot_nlist_walk_pull` / `rgpot_nlist_walk_close`
are the chunked walk.

`Potentials.capnp` adds `NeighborRequest`, `NeighborList`, an optional
`ForceInput.neighbors` at ordinal 9, an optional `PotentialResult.neighborList`
at ordinal 28, and `Potential.neighbours` at ordinal 13. Older fields are
unchanged. A stream chunk sets `NeighborList.more`; the client repeats the
call instead of receiving every pair in one message.

```cpp
rgpot::nlist::Query query;
query.cutoff = 5.0;
query.skin = 0.5;
query.format = rgpot::nlist::Format::Half;
query.cut = rgpot::nlist::Cut::Loose;
query.mode = rgpot::nlist::Mode::Stored;
rgpot::nlist::Builder builder;
builder.update(positions, n, box, query);
rgpot::nlist::SoA pairs = builder.table().view();
// pairs.first[k], pairs.second[k], pairs.shift[3 * k + axis]
```

EAM, Lennard-Jones, Sutton-Chen, Stillinger-Weber, Tersoff, Morse and
ZBL take candidate pairs from `rgpot::nlist::Neighbours`.

The layer bins atoms into cells whose edge is the cutoff plus the skin.
`Round` is the orthorhombic truncation used by the pair kernels. `Half`
is one correction about the half-box, which is what EAM uses after
wrapping coordinates into the cell. `General`, and any sheared cell,
calls `MinimumImage`.

`Mode::Stored` keeps a Verlet list and reuses it while every atom has
moved by at most half the skin. `Mode::Stream` walks the 27 neighbouring
cells and stores no pair list. The bins are reused only when the geometry
is identical. `Mode::Auto` stores when the candidate index array fits in
`MemAvailable` and in a 32-bit offset, and streams otherwise. There is no
cutoff on the atom count.

`OMP_NUM_THREADS` above 1 splits atoms into contiguous slices. The
streaming cell walk itself runs on those slices: each thread reads the
bins for its own atoms. Private forces are added in thread-id order, so
a fixed thread count repeats bit for bit and stays within `1e-9` of one
thread. One thread keeps the serial pair order.

Stillinger-Weber and Tersoff use this layer for `Auto` as well as
`Stored` and `Stream`. Bonds are gathered for one atom at a time, sorted
by neighbour index, then discarded. That order is not the old
`orderedPairs` walk, so one thread is not bit-identical to it. On a
distorted 216-atom diamond the largest absolute difference was
`3.4e-13` for Stillinger-Weber and `3.3e-16` for Tersoff. Morse and ZBL
use the same half-list. The old `PairListCache` and its private cell
walk are gone.

A streaming three-body call no longer reserves 32 bonds on every atom.
The extra memory is the bins, the coordinates, and one scratch row.

## Timings

One Intel Xeon socket, 8 cores, 1 thread per core. g++ 13.3.0,
`-O3 -DNDEBUG -fopenmp`. Each time is the median of seven warm calls
after two discarded calls. The peak resident set below is from a
separate process per potential, one million-atom streaming call:

| potential | atoms | peak RSS (kB) |
| --- | ---: | ---: |
| EAM | 1048576 | 139796 |
| Lennard-Jones | 1048576 | 115204 |
| Sutton-Chen | 1048576 | 917768 |
| Stillinger-Weber | 1000000 | 112596 |
| Tersoff | 1000000 | 119472 |

Lennard-Jones streaming applies the pair term inside the cell walk and
does not store a half-list. Sutton-Chen streaming keeps the accepted
partner indices from the density walk and does not walk the cells again
for the forces; that index array is why its resident set is larger.

### Serial, stored list

| atoms | EAM (s) | LJ (s) | Sutton-Chen (s) | atoms | SW (s) | Tersoff (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 108 | 0.000111 | 0.000038 | 0.000260 | 64 | 0.000030 | 0.000041 |
| 500 | 0.000545 | 0.000171 | 0.002856 | 216 | 0.000100 | 0.000135 |
| 2048 | 0.002130 | 0.000704 | 0.011370 | 1000 | 0.000446 | 0.000621 |
| 6912 | 0.007063 | 0.002394 | 0.037197 | 4096 | 0.001804 | 0.002553 |
| 32000 | 0.032551 | 0.011227 | 0.169652 | 13824 | 0.006158 | 0.008707 |
| 108000 | 0.110193 | 0.038031 | 0.561026 |  |  |  |

### Serial, streaming walk

| atoms | EAM (s) | LJ (s) | Sutton-Chen (s) | atoms | SW (s) | Tersoff (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 108 | 0.000227 | 0.000166 | 0.000302 | 64 | 0.000054 | 0.000097 |
| 500 | 0.003304 | 0.003621 | 0.005873 | 216 | 0.000260 | 0.000287 |
| 2048 | 0.013226 | 0.040806 | 0.085449 | 1000 | 0.001137 | 0.001294 |
| 6912 | 0.038887 | 0.106809 | 0.369985 | 4096 | 0.004749 | 0.005300 |
| 32000 | 0.161625 | 0.408769 | 1.577800 | 13824 | 0.015888 | 0.017356 |
| 108000 | 0.501919 | 1.443270 | 2.516584 |  |  |  |
| 1048576 | 4.919678 | 12.503108 | 20.824924 | 1000000 | 1.081986 | 1.141701 |

Lennard-Jones and Sutton-Chen at one million atoms are an FCC crystal of
64 repeats (1,048,576 atoms). Stillinger-Weber and Tersoff are a diamond
of 50 repeats (1,000,000 atoms).

### Strong scaling

Metals are 32,000 atoms. Diamond potentials are 13,824 atoms. Speedup is
against the one-thread time of that mode.

| potential | mode | 1 | 2 | 4 | 8 |
| --- | --- | ---: | ---: | ---: | ---: |
| EAM | stored | 0.032498 | 0.019711 (1.65x) | 0.010249 (3.17x) | 0.005425 (5.99x) |
| EAM | stream | 0.156389 | 0.081042 (1.93x) | 0.042261 (3.70x) | 0.023984 (6.52x) |
| LJ | stored | 0.011644 | 0.006217 (1.87x) | 0.003582 (3.25x) | 0.002460 (4.73x) |
| LJ | stream | 0.393600 | 0.202438 (1.94x) | 0.104098 (3.78x) | 0.053970 (7.29x) |
| Sutton-Chen | stored | 0.171792 | 0.115669 (1.49x) | 0.058216 (2.95x) | 0.029906 (5.74x) |
| Sutton-Chen | stream | 0.810416 | 0.479725 (1.69x) | 0.281400 (2.88x) | 0.165698 (4.89x) |
| SW | stored | 0.006136 | 0.003248 (1.89x) | 0.001723 (3.56x) | 0.001037 (5.92x) |
| SW | stream | 0.015693 | 0.008133 (1.93x) | 0.004148 (3.78x) | 0.002283 (6.87x) |
| Tersoff | stored | 0.008756 | 0.004402 (1.99x) | 0.005016 (1.75x) | 0.002909 (3.01x) |
| Tersoff | stream | 0.016700 | 0.008453 (1.98x) | 0.005964 (2.80x) | 0.003416 (4.89x) |

Stored EAM at 108,000 atoms: 0.107392 s, 0.074392 s (1.44x), 0.037288 s
(2.88x), 0.020593 s (5.22x). Streaming EAM at the same size: 0.501968 s,
0.260581 s (1.93x), 0.132694 s (3.78x), 0.068205 s (7.36x). Two threads
used to be slower than one because the stream recomputed the density
exponential in a second pass; the threaded path now uses the same one
walk as one thread and keeps neighbour forces in a private buffer.
Tersoff's parallel stress used to land on one shared array. It now uses
the private buffer. At 13,824 atoms the stored list reaches 3.01x on 8
threads and the stream 4.89x. At 1,000,000 atoms the stream median on 8
threads is 0.236637 s, against 1.141701 s on one thread (4.82x).

### Weak scaling

Metal cells are 6x6x6, 6x6x12, 6x12x12 and 12x12x12 conventional FCC
cells (864, 1,728, 3,456 and 6,912 atoms) on 1, 2, 4 and 8 threads.
Diamond repeats are 4, 5, 6 and 8 (512, 1,000, 1,728 and 4,096 atoms).

| potential | mode | 1 thread | 2 | 4 | 8 |
| --- | --- | ---: | ---: | ---: | ---: |
| EAM | stored | 0.000919 | 0.001159 | 0.001191 | 0.001426 |
| EAM | stream | 0.006408 | 0.012207 | 0.011236 | 0.010896 |
| LJ | stored | 0.000296 | 0.000392 | 0.000428 | 0.000481 |
| LJ | stream | 0.008052 | 0.010016 | 0.012411 | 0.015674 |
| Sutton-Chen | stored | 0.004784 | 0.006297 | 0.006353 | 0.006476 |
| Sutton-Chen | stream | 0.016872 | 0.023663 | 0.033095 | 0.050187 |
| SW | stored | 0.000231 | 0.000241 | 0.000221 | 0.000301 |
| SW | stream | 0.000748 | 0.000584 | 0.000528 | 0.001053 |
| Tersoff | stored | 0.000318 | 0.000521 | 0.000834 | 0.001916 |
| Tersoff | stream | 0.000672 | 0.000769 | 0.000833 | 0.002062 |

## Reproduce

`classical_scaling_bench` prints the serial, strong and weak tables.
`classical_scaling_bench rss EAM` (and `LJ`, `SuttonChen`, `SW`,
`Tersoff`) prints the peak resident set of one million-atom streaming
call in that process. Meson builds the same program as the
`classical_scaling_bench` benchmark.
