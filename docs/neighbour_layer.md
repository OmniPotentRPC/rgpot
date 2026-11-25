# Shared neighbour layer

The public query is `rgpot::nlist::Query`, `Builder`, `Table` and `Walk`.
A query is either a pair shell or a cluster of four. `build` stores the
pairs. `Walk` streams them. Classical potentials still use
`rgpot::nlist::Neighbours` for a minimum-image force walk. Metatomic and
UMA are adapters: they ask this query for pairs and then copy into the
layout their model expects. The calls are in `docs/orgmode/tutorials/neighbour-list.org`.

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
gathers one central atom, or one cluster of four, at a time, writes at
most the length of the caller buffer, and the caller may stop. JAX-MD cannot do this: a jit-compiled list has a
static capacity, `neighbor_fn.allocate` sizes it, and `did_buffer_overflow`
tells the caller to allocate again
([partition.py](https://raw.githubusercontent.com/jax-md/jax-md/main/jax_md/partition.py)).
That is the right answer when the consumer is XLA. It is the wrong answer
when the consumer is a force kernel that can run during the walk. The C
function `rgpot_nlist_build` still reports the full count when the caller
buffer is short, so a fixed buffer can grow the way JAX-MD does, without
making that the only path. The public query has no automatic mode: the
caller picks `build`, `Builder` or `Walk`. The automatic choice between
storing and streaming is `Mode::Auto` in `rgpot::nlist::Neighbours`,
described below.

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

`build` uses threads when `OMP_NUM_THREADS` is above 1 and there are at
least 8192 atoms (`rgpot::kTeamMinAtoms` and `teamSize` in
`CppCore/rgpot/Slices.hpp`); a potential call has a threshold of its own.
A pair list splits the cells into
contiguous ranges when the cell is unsheared and at least two search
radii across every periodic axis. A sheared or narrower cell may need
several images of a pair, so the pair list then splits the atoms and
each slice runs the one-thread walk; a cluster list always splits the
atoms. Every thread count stores the same pairs and shifts. The results
are concatenated in slice order, so a cluster list keeps the one-thread
order; a pair list may store its rows in a different order. Vesin merges
per-thread lists without promising an order. Below 512 atoms a pair list
is a direct loop over atom pairs, without cells. The loop includes a
cell that holds one atom, and so does the image stencil on a larger
cell, so that atom's periodic images inside the cutoff are listed. A
cell narrower than twice the search radius lists every image of a pair
inside the cutoff, not only the nearest.

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

`rgpot_nlist.h` is version 4 (`rgpot_nlist_abi_version`). Every
caller-allocated struct starts with `bytes`. `rgpot_nlist_request_init`
fills a half pair list, periodic boundaries and every geometry column.
`rgpot_nlist_build` and `rgpot_nlist_walk_pull` write caller-owned
structure-of-arrays buffers. `rgpot_nlist_strerror` names a code. A
cluster with a half shell is `RGPOT_NLIST_SHAPE`.

The rgpot-local schema diff `CppCore/rgpot/rpc/local/neighbour-chunk.diff`
adds `NeighborRequest`, `NeighborList`, an optional `ForceInput.neighbors` at
ordinal 9, an optional `PotentialResult.neighborList` at ordinal 28, and
`Potential.neighbours` at ordinal 13 to the `Potentials.capnp` that rgpot
compiles and installs; the vendored copy is the pinned upstream file. Older fields are
unchanged. `NeighborRequest.mode = 2` asks for a stream: a chunk sets
`NeighborList.more` and `next`, and the client repeats the call with
`cursor` instead of receiving every pair in one message. `mode = 1` is
one stored chunk. `potserv` and `rgpot_rpc_server_start` answer `Potential.neighbours`
from this neighbour layer, and `potserv` also fills `PotentialResult.neighborList`
when `ForceInput.neighbors` is set.

<!-- rgpot-doc: run link=rgpot -->
```cpp
#include "rgpot/nlist/NeighbourList.hpp"

#include <cstddef>

int main() {
  const double positions[] = {0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.3, 0.0};
  const std::size_t n = 3;
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};

  rgpot::nlist::Query query;
  query.cutoff = 5.0;
  query.skin = 0.5;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Loose;
  rgpot::nlist::Builder builder;
  builder.update(positions, n, box, query);
  rgpot::nlist::View pairs = builder.table().view();
  // pairs.first[k], pairs.second[k], pairs.shift[3 * k + axis]
  return pairs.first.size() == 3 ? 0 : 1;
}
```

EAM, EAMTable, Lennard-Jones, LJCluster, Sutton-Chen, Stillinger-Weber,
Tersoff, Morse and ZBL take candidate pairs from
`rgpot::nlist::Neighbours`.
EMT uses the vendored Asap neighbour list and force kernel, not
`rgpot::nlist::Neighbours`.

The layer bins atoms into cells whose edge is at least the cutoff plus
the skin.
`Round` is the orthorhombic truncation used by the pair kernels. `Half`
is one correction about the half-box, which is what EAM uses after
wrapping coordinates into the cell. `General`, and any sheared cell,
calls `MinimumImage`.
A periodic orthorhombic axis folds each coordinate into the cell before
binning, so an atom outside the box shares a bin with its image inside.
The pair vector remains the minimum image of the coordinates as given.
Sheared cells stay on the fractional grid.

Periodic and partially periodic skewed cells are binned in fractional coordinates
from the internal minimum-image inverse, wrapped on periodic axes and spanned
on open axes, with the bin count along each axis set by the dual-lattice
norms.
A singular or ill-conditioned cell in fractional binning, or a non-finite
fractional coordinate, is an error instead of falling back to Cartesian
binning.
Orthorhombic cells and fully open skewed cells use Cartesian binning.

An update that rejects its input, a cell or a coordinate, or that finds the
generation counter exhausted, throws before anything changes: a built object
keeps its list and its generation, and a default-constructed object stays empty
without a generation.
An allocation that fails part way through a rebuild, or through the rebinning
of a reuse, leaves the object as a default-constructed one with a cleared
generation, keeping `builds()`.
Nothing is then left to reuse, so the next update builds from its own
coordinates, and the new generation is stored only once a rebuild is complete.
Copy assignment copies into a new object first, so a throw leaves the target
and its generation unchanged.
A moved-from object is left as a default-constructed one with a cleared
generation, so its next update builds from its own coordinates.

`Mode::Stored` keeps a Verlet list and reuses it while every atom has moved by at most half the skin.
The list holds partner indices and each walk takes the image from the current coordinates, so the move is measured to the nearest periodic image of the reference position: an atom that the caller wraps back through a face of the cell has moved by its step, not by a cell vector, and keeps the list.
The bins are redone after such a wrap.
A change of the cell, of the atom count or of the spec rebuilds the list.
`Builder` accepts coordinates any whole number of cells outside the box.
It folds every periodic coordinate into the primary cell in fractional space before binning and the image search.
The stored shift of a pair is then reported for the coordinates as given, `S = S' + W_i - W_j` with `W` the whole cells folded off each atom, and, on a fresh search, the vector and distance are `r_j - r_i + S H` from those coordinates.
When the table is kept, only the shifts are rewritten; the vector and distance are stale and describe the geometry of the last search until a full rebuild.
`build`, `Walk`, `Builder`, the batch build and the C ABI all keep that relation, so a consumer that forms vectors from its own positions and the shifts, such as the Fortran bridge, metatomic or UMA, sees the right image.
A coordinate more than 2^29 cells from the origin is rejected with `Code::Box`.
The skin test measures each folded atom from the image of its folded reference nearest it, `r' - J H` with `J` the rounded change of its fractions, so whole-cell drift and a step across a face of the cell count as the step alone.
The shifts of a kept pair table are then rewritten in place for the basis `W + J`.
A cluster table, whose four lanes share one shift, is searched again when that basis changes, which is when an atom's coordinates as given move by whole cells, for instance when the caller wraps it.
The same update reuses the table across a cell change when the largest image displacement, at every sign of the shift stencil, fits in the skin together with the atom moves.
`Mode::Stream` walks the 27 neighbouring cells and stores no pair list.
The bins are reused only when the geometry is identical.
`Mode::Auto` stores when the candidate index array fits in `MemAvailable` and in a 32-bit offset, and streams otherwise.
`Mode::Stored` makes the same test and also falls back to a stream when the index does not fit.
When `MemAvailable` cannot be read, both modes store.
There is no cutoff on the atom count.

From 8192 atoms, with more than one thread requested, the stored index
is gathered on contiguous atom slices when the stencil holds at least
4096 candidates and more candidates than atoms. The slices are joined
in atom order. Each row is sorted by partner index, which is what one
thread stores, so the list does not depend on the slice count. Each
slice reserves its part of the index once. A shorter list, including a
dilute system above 8192 atoms, stays on the calling thread. A reuse
still skips that build.

An open axis whose nominal grid does not fit in the memory the OS
reports is binned on the atom span. A box of a million angstroms
around a cluster then keeps a real cell list. A periodic axis is
unchanged. When the tight grid also does not fit, the build uses one
cell, as before.

A grid with more cells than an `int` can name is one cell as well, whether or
not the OS reports its memory.
The cells are counted in 64-bit arithmetic, so a periodic box of 3e9 Å with a
1 Å search is one cell.

An infinite cutoff or skin has no search radius, in any mode.
The update bins nothing and stores nothing, and `allPairs()` reports it with `streaming()` set.
Each walk takes the pairs straight from the coordinates, `i` ascending and then `j` ascending, which is the order of the one bin a search wider than the system gives, and folds each pair with the image of the spec.
No move takes a pair out of such a search, so the update is reused for any coordinates while the atom count, the cell and the spec stay the same.
A team walk with no bins gives each slice a contiguous range of atoms; with more slices than atoms, some ranges are empty.
Sutton-Chen with `cutoff <= 0` searches this way.
The public `Query` still rejects a cutoff that is not finite.

A potential call on fewer atoms than the `kTeamMinAtoms` of its class
runs on the calling thread, whatever `OMP_NUM_THREADS` says, and keeps
the serial pair order. From there up it opens a team when it has at
least the atoms the `kTeamFrom` of its class asks of the
`OMP_NUM_THREADS` threads, and otherwise runs the same sums on the
calling thread. Both come from the crossovers in
[Per-potential thresholds](#per-potential-thresholds).
`teamSize(n, from, team)` in `CppCore/rgpot/Slices.hpp` gives the slice
count of every call, and `teamsOpened` counts the regions a thread has
opened, so a test can see that a small call opened none.
The one-argument `teamSize(std::size_t)` overload is deprecated and
remains exported as a compatibility wrapper forwarding to
`teamSize(n, rgpot::kTeamMinAtoms, {})`.

From the `kTeamMinAtoms` of its class up, the least of its `kTeamFrom`
counts (24 atoms for Morse, 28 for Tersoff, 32 for SW, 40 for LJ and
EAM, 48 for the LJ cluster, 64 for Sutton-Chen and 192 for ZBL), a
result has the bits of one thread on any `OMP_NUM_THREADS`. Each atom
keeps the energy and strain of its pairs
with higher-index partners, and the atoms are added in index order. The
pair potentials (`nlist::sumPairs` in `CppCore/rgpot/nlist/PairSum.hpp`),
Sutton-Chen and EAM give each slice of a team a contiguous range of
atoms, and `forRowTeam` walks each atom's whole row on the slice that
owns it, so the force on an atom has one writer. One thread walks each
pair once, by `forRowHalf`, and a row takes the atom's partners in the
order in which that walk reaches the atom; a partner below the atom
takes the negated vector of the walk, so both visits of a pair agree
even where two images tie. On a stored list, in an unbounded search and
in a stream without tiles, `forRowHalf` is the half walk, and a row takes
the partners below the atom in index order, then those above it. A
stream on the tiles visits each pair from the atom that comes first by
cell and then by index. A row there takes every atom of the stencil
cells before its own, by cell, then the atoms before it in its cell,
then, in stencil order, the atoms after it in its cell and every atom of
the later cells. On a stored list that holds at least 4096 candidate
upper pairs (`kMinParallelHalfPairs`), once `MorsePot::kTeamFrom` opens
a team for the thread count, from 2048 atoms on three threads and from
24 on four or more but never on two, `MorsePot` evaluates each upper
pair once in a two-phase OpenMP region (`forEachSlicePhase`), batching
`simd::exp_fixed_n` and partitioning atoms by pair count on 8-atom
boundaries before reducing forces in partner index order; sparser
stored lists, and calls that open no team, stay on one thread. Boundary
words of the 1-bit active-pair mask are merged with `std::atomic_ref`
(which on LLVM's C++ library requires `libc++` 19 or newer,
`_LIBCPP_VERSION >= 190000`), falling back to `__atomic_fetch_or` on
older `libc++`: Meson checks `_LIBCPP_VERSION < 190000` and probes
`__atomic_fetch_or`, while CMake probes `std::atomic_ref<std::uint64_t>`
directly and then probes the compiler intrinsic fallback, failing at
configure time if neither is supported. `fastTiles` runs only when
`streaming()` holds, so `forCellTeam` on a stored list walks `walkStored`
instead of rebuilding a 27-cell tile stencil. Stillinger-Weber and Tersoff
(`CppCore/rgpot/CentreTerms.hpp`) sum the terms of each centre into one
force on the centre and on each partner, an energy and a strain, and add
the centres in index order. `forEachSlice`, with every other OpenMP
region, hands every slice to the threads the runtime starts, also when
there are fewer of them, as under `OMP_THREAD_LIMIT` or inside another
OpenMP region, so those teams give the same bits too.

The thread-local transpose index, split vectors and pair scratch of `MorsePot`
(`MorseWorkspace`), and the workspace of `SuttonChenPot`, are keyed on
`Neighbours::generation()`.
That counter is process-unique, non-zero, and changes at construction and on
every rebuild, copy and move; if the 64-bit counter reaches `UINT64_MAX`, it
throws `std::overflow_error` rather than wrapping to `0`.
`LJPot` keys the list it keeps on each thread on the same counter.
The potentials keep different parts of these buffers between calls.
`MorsePot` keeps each buffer's per-thread high-water capacity across calls of
comparable size, and reallocates a buffer down when a later call needs less
than a quarter of a retained capacity above `kShrinkFloor` (4096 elements).
`SuttonChenPot` keeps its half-pair index on each thread, resized in place when
the generation changes, and frees its pair and row scratch at the end of every
threaded stored call, so `kShrinkFloor` does not apply to it.
The list `LJPot` keeps is a `Neighbours`, whose arrays keep their capacity.

Stillinger-Weber and Tersoff use this layer for `Auto` as well as
`Stored` and `Stream`. Bonds are gathered for one atom at a time, sorted
by neighbour index, then discarded. That order is not the old
`orderedPairs` walk, so one thread is not bit-identical to it. On a
distorted 216-atom diamond the largest absolute difference was
`3.4e-13` for Stillinger-Weber and `3.3e-16` for Tersoff. Morse and ZBL
use the same half-list. The old `PairListCache` and its private cell
walk are gone.
EMT now follows the same OpenMP threshold as this layer: below
`rgpot::kTeamMinAtoms` (8192 atoms by default) it stays on one thread,
and from `rgpot::kTeamMinAtoms` up it
evaluates the Asap half list in two phases, one row-owner pass and one
incoming-partner pass, so energies and forces are bit-identical at 1, 2,
4 and 8 threads.

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
| Sutton-Chen | 1048576 | 131268 |
| Stillinger-Weber | 1000000 | 112596 |
| Tersoff | 1000000 | 119472 |

Lennard-Jones streaming applies the pair term inside the cell walk and
does not store a half-list. Sutton-Chen streaming keeps the per-atom
density between two cell walks and does not store partner indices. The
walk emits each unordered pair once.

Three stream layouts were timed in one process each, cutoff 3, FCC with
lattice `sqrt(2)`. The time is the median of seven calls after one
discarded call.

| layout | 108000 atoms (s) | RSS (kB) | 1048576 atoms (s) | RSS (kB) |
| --- | ---: | ---: | ---: | ---: |
| partner indices from the density walk | 0.985956 | 134360 | 9.392517 | 917680 |
| second walk, indices kept for 4096 atoms | 1.131455 | 21868 | 10.056180 | 135956 |
| second walk, density only | 1.024955 | 17320 | 9.365840 | 131268 |

The density-only walk matches the index store's time and stays with the
atom and cell arrays. The 4096-atom tile is a little slower. The index
store is the 918 MB resident set. In the serial stream table, the
Sutton-Chen entries at 108,000 and 1,048,576 atoms are this density-only
remeasurement. The smaller Sutton-Chen stream rows, and the
strong-scaling row for that stream, are the earlier index-store run.

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
| 108000 | 0.501919 | 1.443270 | 1.024955 |  |  |  |
| 1048576 | 4.919678 | 12.503108 | 9.365840 | 1000000 | 1.081986 | 1.141701 |

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
exponential in a second pass. The stream now makes two cell walks, one
for the density and one for the forces, with the pair terms read from
tabulated splines, and each slice keeps its own density and force rows.
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

### Thread count

`thread_count_bench` times the force call of each potential on a free
cluster: the N sites of its lattice nearest one site, with Gaussian noise
of 0.02 on each coordinate, over 16 frames in turn. LJ, with a cutoff of
2.5, Morse, EAM, SW and Tersoff put the cluster in the middle of a cell
with vacuum on every side. The LJ cluster, ZBL and Sutton-Chen take no
cell, and Sutton-Chen has no cutoff, as a global optimisation calls it.
The neighbour mode is `Auto`, which stored the list at every size in the
first table. A time is the median of five runs of at least 0.2 s, in
microseconds per call. Both builds are the Meson release build of the
pixi `devbld` environment with GCC 14.3.0, on the eight cores above under
`taskset -c 0-7`. Before is 47c46f97 and after is 2703f77a, which had
one threshold for every call;
[Per-potential thresholds](#per-potential-thresholds) gives the
thresholds that replaced it.

| potential | atoms | 1 thread, before | 1 thread, after | 8 threads, before | 8 threads, after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sutton-Chen | 38 | 6.9 | 6.3 | 26.7 | 6.3 |
| Sutton-Chen | 55 | 14.1 | 13.0 | 41.9 | 13.0 |
| Sutton-Chen | 80 | 33.3 | 27.1 | 90.8 | 27.1 |
| Sutton-Chen | 500 | 1136 | 1021 | 3236 | 1017 |
| Sutton-Chen | 4000 | 85998 | 82619 | 309289 | 82530 |
| Sutton-Chen | 10000 | 547670 | 541027 | 2113282 | 156037 |
| LJ | 38 | 10.1 | 8.8 | 23.6 | 8.7 |
| LJ | 55 | 20.4 | 17.3 | 39.1 | 17.3 |
| LJ | 80 | 38.5 | 33.7 | 60.9 | 33.7 |
| LJ | 500 | 495 | 463 | 484 | 463 |
| LJ | 4000 | 5054 | 4651 | 6829 | 4651 |
| LJ | 10000 | 13139 | 12581 | 20444 | 3701 |
| LJ cluster | 38 | 4.5 | 4.0 | 5.4 | 4.0 |
| LJ cluster | 55 | 9.5 | 8.5 | 7.8 | 8.5 |
| LJ cluster | 80 | 18.0 | 16.2 | 9.9 | 16.2 |
| LJ cluster | 500 | 220 | 212 | 78.5 | 212 |
| LJ cluster | 4000 | 2301 | 2133 | 554 | 2127 |
| LJ cluster | 10000 | 6040 | 5528 | 1388 | 1639 |
| Morse | 38 | 21.1 | 20.0 | 26.8 | 19.9 |
| Morse | 55 | 42.7 | 40.2 | 51.8 | 40.2 |
| Morse | 80 | 82.0 | 80.3 | 94.4 | 77.8 |
| Morse | 500 | 1241 | 1192 | 1171 | 1190 |
| Morse | 4000 | 13814 | 13154 | 15035 | 13196 |
| Morse | 10000 | 36873 | 36139 | 41316 | 10886 |
| ZBL | 38 | 2.9 | 2.7 | 8.3 | 2.7 |
| ZBL | 55 | 3.9 | 3.7 | 11.0 | 3.7 |
| ZBL | 80 | 6.2 | 5.8 | 16.3 | 5.8 |
| ZBL | 500 | 43.3 | 40.8 | 236 | 40.9 |
| ZBL | 4000 | 386 | 366 | 10937 | 365 |
| ZBL | 10000 | 996 | 1001 | 66237 | 361 |
| EAM | 38 | 17.0 | 16.8 | 9.9 | 16.8 |
| EAM | 55 | 29.8 | 28.9 | 14.8 | 28.8 |
| EAM | 80 | 51.6 | 48.8 | 30.0 | 48.6 |
| EAM | 500 | 459 | 455 | 178 | 455 |
| EAM | 4000 | 4110 | 4042 | 1411 | 4041 |
| EAM | 10000 | 10625 | 10339 | 3439 | 3337 |
| SW | 38 | 11.2 | 11.1 | 6.7 | 11.1 |
| SW | 55 | 16.1 | 16.0 | 8.2 | 16.0 |
| SW | 80 | 25.8 | 25.7 | 10.8 | 25.7 |
| SW | 500 | 213 | 216 | 60.7 | 216 |
| SW | 4000 | 1997 | 2007 | 469 | 2006 |
| SW | 10000 | 5165 | 5375 | 1236 | 1115 |
| Tersoff | 38 | 16.5 | 16.0 | 10.0 | 16.0 |
| Tersoff | 55 | 22.8 | 22.2 | 11.7 | 22.2 |
| Tersoff | 80 | 37.6 | 36.6 | 16.9 | 36.6 |
| Tersoff | 500 | 272 | 266 | 118 | 266 |
| Tersoff | 4000 | 2420 | 2371 | 818 | 2369 |
| Tersoff | 10000 | 6174 | 6284 | 2032 | 1274 |

Below 8192 atoms every call ran on the calling thread, with the bits
the old build gave on one thread, at most 1.2% slower (SW at 500 atoms)
and up to 19% faster (Sutton-Chen at 80 atoms). The old build
started a team on any call. In this run, on 8 threads Sutton-Chen took
2.7 to 3.9 times its one-thread time at every size, the LJ and Morse
teams were slower than one thread at five of the six sizes, and the ZBL
team at all six. Three more runs of Sutton-Chen at 38 to 500 atoms put
the old 8-thread call at 2.7 to 6.3 times one thread, 2.7 to 4.4 times
at 55 and 80 atoms, while the new times stayed within 1.2% of the table.

EAM, the LJ cluster, SW and Tersoff split their atoms into slices, and
on 8 threads those teams beat one thread from 38 or 55 atoms up. Below
8192 atoms they ran on one thread, up to 4.3 times slower than their
old teams: SW at 4000 atoms took 2006 µs per call, where its team took
469 µs. That was the cost of one threshold for every call. The sums do
not depend on it, so a lower threshold for these potentials changes
only their speed.

At 10000 atoms 8 threads were 2.8 to 4.9 times faster than one, and
faster than the old team for every potential but the LJ cluster, 18%
slower: its team evaluates each pair once from each end, where its old
slices evaluated a pair once and added both forces to private buffers.
One thread cost at most 4% more than before, for SW, whose centres sum
their terms before adding them.

On 8 threads the old build moved the energy on 10 to 16 of the 16 frames
for every potential but ZBL, by up to 2.06e-4 epsilon for the reduced
Sutton-Chen at 10000 atoms, and moved the forces of every potential but
Sutton-Chen and ZBL. The new build gives the bits of one thread on every
frame. At 10000 atoms one thread now adds the same per-atom sums as a
team, so its bits are not those of the old build.

At 100000 atoms:

| potential | 1 thread, before | 1 thread, after | 8 threads, before | 8 threads, after |
| --- | ---: | ---: | ---: | ---: |
| LJ | 144577 | 139135 | 192722 | 36995 |
| LJ cluster | 8773537 | 14933502 | 1872959 | 1909984 |
| Morse | 409610 | 402374 | 406363 | 109952 |
| EAM | 113326 | 112897 | 32373 | 32774 |
| SW | 55391 | 57159 | 12427 | 10341 |
| Tersoff | 64455 | 64937 | 24281 | 10983 |

But for the LJ cluster, one thread cost at most 3.2% more than before
(SW), and 8 threads were 3.4 to 5.9 times faster than one: up to 5.2
times faster than the old team, and 1.2% slower for EAM. The LJ cluster
takes no cell, so that measurement binned a box a million angstroms
wide, which fell back to one bin. At that size its candidates did not
fit an index, so both builds streamed that bin and tested every pair
on each call. ZBL takes no cell either and was not timed there. An
open axis whose nominal grid does not fit is now binned on the atom
span, as described above.

In that build a stream walked each atom's whole row on any thread count,
so one thread had the bits of a team, and a two-body potential evaluated
each pair from both ends. At 100000 atoms that made the LJ cluster 1.70
times slower on one thread, and 2% slower on 8. In `Stream` mode at
10000 atoms:

| potential | 1 thread, before | 1 thread, after | 8 threads, before | 8 threads, after |
| --- | ---: | ---: | ---: | ---: |
| LJ | 30466 | 49931 | 18115 | 12004 |
| Morse | 32512 | 50444 | 42515 | 8325 |
| EAM | 36088 | 50280 | 12000 | 11514 |
| SW | 13560 | 13656 | 2168 | 2154 |
| Tersoff | 13873 | 13969 | 2563 | 2215 |

One thread on a stream was 1.39 to 1.64 times slower for EAM, Morse and
LJ, which used to walk half pairs. SW and Tersoff gather each centre's
bonds either way and stayed within 1%. On 8 threads none was slower than
its old team. One thread now walks each pair of a stream once;
[Before and after the thresholds](#before-and-after-the-thresholds)
gives its times.

(per-potential-thresholds)=
### Per-potential thresholds

Each potential class keeps three atom counts as `kTeamFrom`: the atoms
from which a team of two threads opens, a team of three, and a team of
four or more (`rgpot::TeamFrom` in `CppCore/rgpot/Slices.hpp`), where
`rgpot::kNoTeam` opens none. From the least of them, the `kTeamMinAtoms`
of the class, a call adds its sums by rows on any thread count, so the
counts choose whether a team opens and never change the bits.

The counts are crossovers in `thread_count_bench` on the clusters above.
`--team-from none` sums directly on one thread, as a call below
`kTeamMinAtoms` does, and `--team-from 0` sums by rows and opens a team
on any thread count above one. Every potential ran on 1, 2, 3, 4, 6 and
8 threads at every 4 atoms from 16 to 32, every 8 to 64, and at 80, 96,
112, 128, 160, 192, 256, 384, 512, 768 and 1024 atoms. All but
Sutton-Chen also ran at 2048 and 4096 atoms on those six counts and at
1536 to 16384 atoms on 1, 2, 4 and 8 threads, and on 1 to 4 threads LJ,
Morse and EAM went on to 65536 atoms and the LJ cluster and ZBL to
32768. Sutton-Chen, which has no cutoff, ran at 1536 to 8192 atoms on 1
to 4 threads and at 1536 on 8. Each point is three or six passes, and a
pass is the median of five runs of at least 0.2 s. A team wins at a size
when the median of its pass medians is at most nine tenths of that of
one thread and its slowest pass is faster than the fastest pass of one
thread. The crossover is the smallest size from which a team wins at
every larger size, and none means that it did not win at the largest.
All 4266 team results had the bits of one thread with row sums.

| potential | 2 threads | 3 threads | 4 threads | 6 threads | 8 threads | `kTeamFrom` |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Sutton-Chen | none | 2048 | 64 | 56 | 56 | none, 2048, 64 |
| LJ | none | 768 | 32 | 32 | 40 | none, 768, 40 |
| LJ cluster | none | 768 | 48 | 48 | 48 | none, 768, 48 |
| Morse | none | 2048 | 24 | 24 | 24 | none, 2048, 24 |
| ZBL | none | 768 | 192 | 160 | 112 | none, 768, 192 |
| EAM | none | 384 | 40 | 40 | 40 | none, 384, 40 |
| SW | 112 | 56 | 32 | 32 | 32 | 112, 56, 32 |
| Tersoff | 80 | 28 | 28 | 40 | 28 | 80, 28, 28 |

`kTeamFrom` takes the crossover of two threads, that of three, and the
larger of those of four and eight; none is `kNoTeam`. Six threads crossed
at or below that count for every potential but Tersoff, where one pass
of three at 32 atoms was as slow as one thread while the median took
0.68 of it. Teams of more than eight threads were not timed on these
eight cores and take the count of four or more. Around that count, in
microseconds per call, with the lowest and highest pass median:

| potential | atoms | 1 thread | 4 threads | 8 threads |
| --- | ---: | ---: | ---: | ---: |
| Sutton-Chen | 56 | 14.8 (13.6-14.9) | 13.6 (12.9-13.9) | 12.7 (12.3-12.9) |
| Sutton-Chen | 64 | 17.7 (17.6-18.6) | 15.5 (15.5-15.6) | 11.2 (11.2-11.9) |
| LJ | 32 | 6.96 (6.75-7.75) | 5.86 (5.82-5.97) | 7.43 (6.85-7.56) |
| LJ | 40 | 9.83 (9.81-10.5) | 8.09 (7.88-8.69) | 7.05 (6.99-7.30) |
| LJ cluster | 40 | 4.73 (4.72-4.79) | 4.52 (4.27-4.53) | 4.96 (4.67-5.31) |
| LJ cluster | 48 | 6.80 (6.68-6.94) | 5.33 (5.10-5.33) | 5.16 (5.02-6.10) |
| Morse | 20 | 5.60 (5.58-5.66) | 5.52 (5.43-5.71) | 6.93 (6.58-7.03) |
| Morse | 24 | 8.10 (8.03-8.80) | 6.95 (6.92-7.24) | 7.29 (7.04-7.29) |
| ZBL | 160 | 12.4 (12.3-12.4) | 11.5 (11.4-11.7) | 9.75 (8.71-9.87) |
| ZBL | 192 | 14.8 (14.7-16.1) | 13.3 (13.2-13.4) | 9.96 (9.89-12.1) |
| EAM | 32 | 13.2 (13.1-14.0) | 13.1 (12.1-13.7) | 9.36 (8.63-13.7) |
| EAM | 40 | 18.7 (18.6-21.1) | 15.5 (15.5-15.9) | 10.3 (10.2-11.6) |
| SW | 28 | 7.58 (7.52-7.61) | 7.32 (7.14-7.71) | 7.19 (7.10-7.50) |
| SW | 32 | 9.17 (9.13-9.25) | 8.09 (8.04-8.29) | 7.74 (7.36-7.74) |
| Tersoff | 24 | 8.67 (8.62-9.56) | 7.96 (7.93-8.28) | 8.69 (8.45-10.5) |
| Tersoff | 28 | 11.2 (11.2-11.2) | 8.44 (8.11-8.80) | 8.53 (8.40-8.55) |

A team walks each atom's whole row, so on a team the pair potentials,
Sutton-Chen and EAM evaluate each pair from both atoms, where one thread
evaluates it once. Their teams of two won at no size: from 8192 atoms up
a team of two took 1.03 to 1.21 times one thread for LJ, the LJ cluster
and Morse, 1.14 for Sutton-Chen at 8192 atoms, 1.00 to 1.05 for ZBL and
0.93 to 1.02 for EAM. Their teams of three won from 384 to 2048 atoms,
and from 8192 atoms up took 0.65 to 0.69 of one thread for EAM, 0.70 to
0.82 for LJ, the LJ cluster, Morse and ZBL, and 0.76 for Sutton-Chen at
8192 atoms. SW and Tersoff evaluate each term of a centre once on any
team, and their teams of two won from 112 and 80 atoms. Around the counts
of two and three threads:

| potential | threads | atoms | 1 thread | team |
| --- | ---: | ---: | ---: | ---: |
| Sutton-Chen | 3 | 1536 | 9900 (9750-10494) | 9402 (9237-9414) |
| Sutton-Chen | 3 | 2048 | 19571 (19231-20489) | 16841 (16815-16910) |
| LJ | 3 | 512 | 481 (474-488) | 436 (436-440) |
| LJ | 3 | 768 | 769 (761-783) | 690 (690-693) |
| LJ cluster | 3 | 512 | 219 (218-223) | 199 (198-202) |
| LJ cluster | 3 | 768 | 354 (352-357) | 314 (314-316) |
| Morse | 3 | 1024 | 2894 (2881-2937) | 2633 (2628-2633) |
| Morse | 3 | 2048 | 6354 (6318-6836) | 5691 (5688-5734) |
| ZBL | 3 | 512 | 42.2 (42.0-42.7) | 39.9 (39.7-40.1) |
| ZBL | 3 | 768 | 65.6 (65.2-66.1) | 58.7 (58.3-58.7) |
| EAM | 3 | 256 | 214 (213-218) | 191 (191-217) |
| EAM | 3 | 384 | 343 (341-352) | 284 (283-299) |
| SW | 2 | 96 | 31.8 (31.8-32.0) | 29.1 (29.0-29.6) |
| SW | 2 | 112 | 39.2 (39.1-39.8) | 33.3 (33.1-33.3) |
| SW | 3 | 48 | 13.9 (13.8-14.0) | 12.5 (12.1-12.8) |
| SW | 3 | 56 | 16.6 (16.5-16.6) | 12.9 (12.9-14.5) |
| Tersoff | 2 | 64 | 26.7 (26.4-27.8) | 25.3 (24.6-25.4) |
| Tersoff | 2 | 80 | 37.8 (36.9-39.6) | 30.1 (29.9-31.8) |
| Tersoff | 3 | 24 | 8.67 (8.62-9.56) | 8.41 (8.14-8.54) |
| Tersoff | 3 | 28 | 11.2 (11.2-11.2) | 8.61 (8.47-9.37) |

A single count per potential would open teams of two on LJ, the LJ
cluster, Morse, ZBL and Sutton-Chen, which were no faster than one
thread at any size. With `kNoTeam` as the count of two, two threads run
EAM and ZBL on the calling thread too, where from 8192 atoms up a team
of two took 0.92 to 1.00 of one thread with row sums for EAM and 0.95
to 0.98 for ZBL.

From `kTeamMinAtoms` up one thread sums by rows too, where 424629e5
sums directly below 8192 atoms. Against the direct sums of one thread,
at every size scanned from `kTeamMinAtoms` up, that took:

| potential | `kTeamMinAtoms` | least | most |
| --- | ---: | ---: | ---: |
| Sutton-Chen | 64 | 0.980 | 1.051 |
| LJ | 40 | 0.981 | 1.036 |
| LJ cluster | 48 | 0.943 | 0.988 |
| Morse | 24 | 1.006 | 1.047 |
| ZBL | 192 | 1.031 | 1.080 |
| EAM | 40 | 0.985 | 1.033 |
| SW | 32 | 1.033 | 1.092 |
| Tersoff | 28 | 1.017 | 1.055 |

SW and Tersoff sum the terms of each centre before adding them, and the
pair potentials, Sutton-Chen and EAM keep the energy and strain of each
atom apart and add the atoms in index order, as a team does; the direct
sums add each term to the totals as it comes.

The counts hold for the parameters of the bench. `--cutoff` changes the
partners per atom of LJ, the LJ cluster and Morse. The same scan on 1,
3, 4 and 8 threads at 16 to 1024 and at 2048 atoms, in three passes, put
the crossovers at the sizes below, with the pairs within the cutoff of
the cluster without noise at the crossovers of four and eight threads.
All 1260 team results had the bits of one thread with row sums.

| potential | cutoff | 3 threads | 4 threads | 8 threads | pairs, 4 threads | pairs, 8 threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LJ | 1.5 | 1024 | 80 | 48 | 339 | 181 |
| LJ | 2.5 | 768 | 32 | 40 | 320 | 441 |
| LJ | 4.0 | none | 48 | 48 | 1108 | 1108 |
| LJ cluster | 1.5 | 512 | 192 | 192 | 894 | 894 |
| LJ cluster | 2.5 | 768 | 48 | 48 | 568 | 568 |
| LJ cluster | 4.0 | none | 40 | 56 | 780 | 1475 |
| Morse | 4.5 | 512 | 56 | 32 | 311 | 151 |
| Morse | 9.5 | 2048 | 24 | 24 | 276 | 276 |

Within one potential the crossovers of four and eight threads spread by
1.2 to 4.8 times in atoms and by 1.1 to 6.1 times in pairs, so a count
of pairs, or of pairs times a cost per pair, fits them no better than a
count of atoms. A count of atoms is also known before any walk, in a
stream as well, and depends on the system alone, so the order of the
sums does too, where a count of pairs would follow the coordinates of
the last rebuild. At cutoffs short of the default a team can open early:
at 1.5 sigma, a team of eight on the LJ cluster took 1.13 to 2.79 times
one thread at 48 to 112 atoms and a team of four 1.03 to 1.47 times.
Between their counts and their crossovers LJ at 1.5 sigma
and Morse at 4.5 Å stayed within 0.90 to 1.11 times one thread. At 4.0
sigma the teams of four and eight that opened below their crossovers
took 0.67 to 0.77 of one thread in median, and the teams of three on LJ
and the LJ cluster, which did not cross by 2048 atoms, took 0.90 to 0.92
of one thread from their count of 768.

In `Stream` mode the counts of four or more hold for every potential
timed but the LJ cluster. One thread with direct sums against teams of
four and eight, at 16 to 256 atoms in two passes, put the crossovers of
EAM, Morse and Tersoff at 16 atoms, the smallest scanned, of LJ at 32,
of SW at 24 and 32, and of ZBL at 128 and 96. The LJ cluster crossed at
64: its teams of four and eight took 1.06 and 1.02 times one thread at
48 atoms, the size scanned below, and 0.86 and 0.72 at 64. Sutton-Chen
was not timed there. All 280 team results had the bits of one thread
with row sums.

(before-and-after-the-thresholds)=
### Before and after the thresholds

Against 424629e5, on the clusters, neighbour mode, build and cores of
the thread-count table: before is 424629e5, with one threshold of 8192
atoms, and after has the counts above. Each pass ran both builds in
turn, a process per potential and build that runs the sizes in turn and
each size on 1, 2, 4 and 8 threads. A time is the median of three pass
medians, in microseconds per call. On 8 threads, for SW at 500 atoms on
4 threads and for LJ at 10000 atoms on 2, eight more passes ran that
thread count alone, and a time is the median of eleven.

| potential | atoms | 1 thread, before | 1 thread, after | 2 threads, before | 2 threads, after | 4 threads, before | 4 threads, after | 8 threads, before | 8 threads, after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sutton-Chen | 38 | 6.45 | 6.40 | 6.49 | 6.49 | 6.45 | 6.45 | 6.46 | 6.40 |
| Sutton-Chen | 55 | 13.3 | 13.3 | 13.3 | 13.3 | 13.3 | 13.6 | 13.3 | 13.2 |
| Sutton-Chen | 80 | 27.6 | 28.1 | 27.7 | 28.5 | 27.7 | 28.6 | 27.8 | 16.2 |
| Sutton-Chen | 500 | 1048 | 1051 | 1042 | 1054 | 1041 | 721 | 1040 | 393 |
| Sutton-Chen | 4000 | 85812 | 88547 | 84957 | 87286 | 87236 | 50307 | 85598 | 25977 |
| Sutton-Chen | 10000 | 579920 | 566454 | 629843 | 582923 | 314191 | 324716 | 161173 | 161496 |
| LJ | 38 | 9.09 | 9.22 | 9.05 | 9.31 | 9.04 | 9.27 | 8.94 | 9.22 |
| LJ | 55 | 18.1 | 17.8 | 18.2 | 17.8 | 18.2 | 13.7 | 17.8 | 10.8 |
| LJ | 80 | 35.0 | 34.3 | 35.4 | 35.5 | 34.9 | 28.6 | 34.9 | 16.9 |
| LJ | 500 | 491 | 472 | 475 | 483 | 478 | 343 | 472 | 176 |
| LJ | 4000 | 4742 | 4765 | 4760 | 4754 | 4748 | 3000 | 4735 | 1534 |
| LJ | 10000 | 13003 | 12456 | 14674 | 12334 | 7623 | 7442 | 3821 | 3822 |
| LJ cluster | 38 | 4.21 | 4.30 | 4.22 | 4.29 | 4.17 | 4.27 | 4.15 | 4.29 |
| LJ cluster | 55 | 8.93 | 8.47 | 9.04 | 8.60 | 8.88 | 6.89 | 8.81 | 5.90 |
| LJ cluster | 80 | 17.2 | 16.3 | 17.2 | 16.2 | 17.1 | 13.2 | 16.6 | 8.08 |
| LJ cluster | 500 | 220 | 211 | 223 | 215 | 221 | 154 | 219 | 82.4 |
| LJ cluster | 4000 | 2230 | 2193 | 2246 | 2286 | 2232 | 1334 | 2168 | 676 |
| LJ cluster | 10000 | 5728 | 5769 | 6579 | 5868 | 3298 | 3392 | 1751 | 1702 |
| Morse | 38 | 20.7 | 21.4 | 20.8 | 21.1 | 20.7 | 13.7 | 20.4 | 11.1 |
| Morse | 55 | 41.9 | 43.8 | 41.5 | 42.7 | 41.8 | 23.5 | 41.2 | 17.1 |
| Morse | 80 | 80.4 | 81.8 | 79.8 | 81.6 | 79.6 | 48.6 | 79.4 | 27.3 |
| Morse | 500 | 1262 | 1236 | 1206 | 1239 | 1206 | 876 | 1208 | 472 |
| Morse | 4000 | 13395 | 13589 | 13440 | 13560 | 13490 | 8790 | 13363 | 4508 |
| Morse | 10000 | 37168 | 36193 | 45195 | 36380 | 22367 | 22380 | 11195 | 11320 |
| ZBL | 38 | 2.78 | 2.82 | 2.77 | 2.81 | 3.04 | 2.81 | 2.77 | 2.78 |
| ZBL | 55 | 3.76 | 3.73 | 3.76 | 3.73 | 3.75 | 3.74 | 3.76 | 3.81 |
| ZBL | 80 | 6.04 | 6.03 | 6.07 | 5.97 | 5.94 | 6.29 | 5.99 | 6.02 |
| ZBL | 500 | 41.8 | 44.5 | 41.7 | 45.3 | 42.0 | 33.8 | 41.8 | 24.2 |
| ZBL | 4000 | 373 | 394 | 376 | 397 | 372 | 258 | 369 | 163 |
| ZBL | 10000 | 1022 | 1013 | 1008 | 1007 | 584 | 583 | 407 | 372 |
| EAM | 38 | 17.2 | 17.6 | 17.1 | 17.4 | 17.4 | 17.6 | 17.3 | 17.3 |
| EAM | 55 | 30.0 | 30.7 | 30.2 | 30.4 | 29.5 | 25.2 | 29.9 | 16.4 |
| EAM | 80 | 50.8 | 51.1 | 51.5 | 51.0 | 54.4 | 40.8 | 49.8 | 27.1 |
| EAM | 500 | 489 | 472 | 479 | 466 | 475 | 306 | 461 | 170 |
| EAM | 4000 | 4243 | 4140 | 4167 | 4179 | 4152 | 2382 | 4114 | 1249 |
| EAM | 10000 | 10636 | 10544 | 10287 | 10576 | 5500 | 5495 | 3067 | 3045 |
| SW | 38 | 11.3 | 12.6 | 11.3 | 12.3 | 11.4 | 9.31 | 11.4 | 9.72 |
| SW | 55 | 16.4 | 18.3 | 16.3 | 18.9 | 16.3 | 12.6 | 16.4 | 10.8 |
| SW | 80 | 26.3 | 29.6 | 26.8 | 28.6 | 26.2 | 17.4 | 26.4 | 13.8 |
| SW | 500 | 217 | 230 | 217 | 163 | 217 | 99.6 | 218 | 62.8 |
| SW | 4000 | 2033 | 2128 | 2033 | 1330 | 2030 | 759 | 2042 | 431 |
| SW | 10000 | 5513 | 5620 | 3353 | 3521 | 1900 | 2034 | 1102 | 1090 |
| Tersoff | 38 | 16.5 | 17.2 | 16.5 | 17.2 | 16.4 | 10.9 | 16.3 | 9.65 |
| Tersoff | 55 | 22.5 | 23.6 | 22.6 | 24.2 | 22.5 | 16.1 | 22.6 | 12.3 |
| Tersoff | 80 | 37.1 | 38.9 | 37.2 | 30.6 | 37.1 | 21.7 | 37.3 | 15.7 |
| Tersoff | 500 | 270 | 280 | 269 | 200 | 270 | 123 | 269 | 72.6 |
| Tersoff | 4000 | 2465 | 2492 | 2409 | 1515 | 2406 | 853 | 2406 | 482 |
| Tersoff | 10000 | 6396 | 6376 | 3887 | 3844 | 2152 | 2192 | 1178 | 1184 |

On 8 threads every potential now opens a team below 8192 atoms, from its
`kTeamMinAtoms`, and from there to 4000 atoms a call took 0.20 to 0.86
of its time before, the least for Tersoff and SW at 4000 atoms and the
most for SW at 38. SW at 4000 atoms took 431 µs per call, against 2042
µs before and the 469 µs that the thread-count table gave the team of
47c46f97. On 4 threads the same sizes took 0.35 to 0.85 of the time
before, but for Sutton-Chen at 80 atoms. On 2 threads SW opens a team
from 112 atoms and Tersoff from 80, and from there to 4000 atoms they
took 0.63 to 0.82 of the time before. The other potentials open no team
of two: at 10000 atoms, where 424629e5 opens one, LJ, the LJ cluster,
Morse and Sutton-Chen took 0.80 to 0.93 of the time before on 2
threads, and ZBL and EAM 1.000 and 1.028 times, within the passes
before, 992 to 1021 µs and 10142 to 10762 µs.

Sutton-Chen's team of four at 80 atoms took 28.6 µs per call, as long
as one thread, where the scan had it at 0.83 of one thread, and its
team of eight took 15.3 to 17.4 µs in seven of the eleven passes and
24.7 to 30.7 µs in the other four. In processes that ran 38 and 55
atoms on 1, 2, 4 and 8 threads and then one size, the first at which
they opened a team, its team of four took 1.10 to 1.20 times one thread
at 64 atoms, 0.99 to 1.13 times at 80, and 0.68 to 0.84 times at 96,
112, 128 or 160, in three processes each. Its team of eight took 0.58
to 0.74 of one thread at 64 and 80 atoms in five of those six processes
and 1.12 times in the sixth. Builds with a count of 96 and of 112, run
on 38, 55 and 80 atoms and then up from the count, took 0.88 to 0.96
and 0.75 to 0.79 times one thread on four threads at their count. One
count serves every team of four or more threads, and a higher one would
give up the sizes at which the team of eight won, so Sutton-Chen keeps
its count of 64.

On one thread, from its `kTeamMinAtoms` to 4000 atoms, a call sums by
rows, where 424629e5 sums directly, and took 0.95 to 1.12 times its
time before: at most 0.98 times for the LJ cluster, 1.00 for LJ, 1.02
for EAM, 1.03 for Sutton-Chen, 1.05 for Morse and Tersoff, 1.07 for ZBL
and 1.12 for SW. Below `kTeamMinAtoms` both builds sum directly on any
thread count, and after took 0.92 to 1.06 times the time before. At
10000 atoms both sum by rows, on one thread and in the same teams of
four and eight, and of two for SW and Tersoff, and after took 0.92 to
1.07 times the time before.

No result of either build had bits apart from one thread, and no point
had other bits in another pass. The builds gave the same bits below
each `kTeamMinAtoms` and at 10000 atoms, and other bits between, where
after sums by rows. The pass medians of a point spread by 5.8% of its
time at the median point and by 34% at the ninetieth percentile. The
five widest were teams of four and eight after, where the slowest pass
took up to 2.08 times the median: EAM at 55 atoms on 8 threads took
15.9 to 34.1 µs, with a median of 16.4, and SW at 10000 atoms 1020 to
2078 µs, with a median of 1090.

In `Stream` mode at 10000 atoms, with 47c46f97 in the same three passes:

| potential | 1 thread, 47c46f97 | 1 thread, before | 1 thread, after | 8 threads, 47c46f97 | 8 threads, before | 8 threads, after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LJ | 31116 | 50760 | 32602 | 18585 | 12281 | 10784 |
| LJ cluster | 86509 | 150759 | 86697 | 19671 | 19507 | 16116 |
| Morse | 32993 | 51367 | 28889 | 41620 | 8632 | 8938 |
| ZBL | 77515 | 144099 | 70866 | 67876 | 18984 | 15160 |
| EAM | 36804 | 50599 | 35638 | 13019 | 11833 | 11453 |
| SW | 13934 | 13864 | 13753 | 2424 | 2586 | 2241 |
| Tersoff | 14048 | 14257 | 14268 | 2822 | 2458 | 2330 |

One thread took 0.49 to 0.70 of the time before for the two-body
potentials and EAM, which evaluate each pair once on one thread again,
and 0.88 to 1.00 times the time of 47c46f97, but 1.05 times for LJ.
Three more passes timed the after build with `--team-from none`, which
sums directly on the half walk as 47c46f97 did and gave its bits on all
five: LJ took 0.98 of the time of 47c46f97 that way and 1.05 times with
its own row sums, 1.07 times its direct sums. Morse, ZBL and the LJ
cluster took 0.80 to 0.89 of their direct time with row sums, and EAM
0.97. On all five the direct sums gave other bits than the rows, which
8 threads repeat, so those 7% on LJ are what the bits of a team cost
one thread there. SW and Tersoff gather each centre's bonds either way
and stayed within 1% of the time before on one thread.

On 8 threads after took 0.80 to 0.88 of the time before for LJ, the LJ
cluster and ZBL, 0.97 for EAM, and 0.87 to 1.04 for Morse, SW and
Tersoff, whose fastest passes were within 1% of those before. That is
0.21 to 0.92 of the time of 47c46f97. 47c46f97 moved the energy of all
16 frames on 8 threads for every potential but ZBL. LJ and EAM gave
other bits than before, the same on every thread count, and the others
the bits of before.

At 100000 atoms, in three passes, and in two for the LJ cluster, which
ran two frames and three runs a pass:

| potential | 1 thread, before | 1 thread, after | 8 threads, before | 8 threads, after |
| --- | ---: | ---: | ---: | ---: |
| LJ | 143117 | 137251 | 40383 | 40230 |
| LJ cluster | 15063065 | 9353114 | 1970369 | 1769786 |
| Morse | 426946 | 424074 | 117863 | 115859 |
| EAM | 114791 | 116618 | 34207 | 36101 |
| SW | 63278 | 60870 | 11999 | 12491 |
| Tersoff | 72327 | 69518 | 12647 | 12620 |

The LJ cluster streams its one bin at this size, and one thread took
0.62 of the time before, 9.35 s per call against 15.06 s, with the bits
of before, and 8 threads 0.90. The other potentials store their lists
here and walk them the same way in both builds, with the same bits, and
took 0.96 to 1.02 times the time before on one thread and 0.98 to 1.06
times on 8, where the fastest pass after took 0.97 to 1.02 times the
fastest before.

## Reproduce

`benches/neighbour_api_bench.py` times the public query against vesin
0.6.2, ASE 3.29.0 and matscipy 1.3.1. FCC copper uses lattice 3.615 and
a strict cutoff of 5. A 30-cube cell is 108,000 atoms and a 64-cube
cell is 1,048,576. The triclinic cell is a 4×4×4 grid in

<!-- rgpot-doc: skip cell matrix copied from benches/neighbour_api_bench.py, not a program -->
```text
[[6.0, 0.0, 0.0], [1.2, 5.4, 0.0], [0.3, 0.7, 5.8]]
```

Skin is zero. Each library is a fresh process. The time is the median
of seven calls after two discarded calls, and the resident set is
`VmHWM`. vesin is `NeighborList.compute` with `n_threads` set and
`quantities="ijSDd"`. ASE's full list is `neighbor_list("ijSD")`. ASE's
half list is a new `NeighborList(..., bothways=False)` on every call.
matscipy's `neighbour_list("ijSD")` is a full list and has no half
switch. ASE and matscipy take no thread count; both thread columns were
still run.

On the triclinic cell the half list is 5,952 pairs and the full list is
11,904, in rgpot, vesin and ASE. At 108,000 atoms the half list is
2,268,000 pairs and the full list is 4,536,000. At 1,048,576 atoms the
half list is 22,020,096 pairs and the full list is 44,040,192.

| system | list | threads | rgpot (s) | vesin (s) | ASE (s) | matscipy (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| triclinic | half | 1 | 0.000545 | 0.000414 | 0.005154 |  |
| triclinic | half | 8 | 0.000400 | 0.000415 | 0.005157 |  |
| triclinic | full | 1 | 0.001044 | 0.000641 | 0.035869 | 0.001769 |
| triclinic | full | 8 | 0.000625 | 0.000622 | 0.036025 | 0.001730 |
| 108000 | half | 1 |  | 0.145285 | 2.537847 |  |
| 108000 | half | 8 |  | 0.081414 | 2.509724 |  |
| 108000 | full | 1 |  | 0.235651 | 8.686973 | 0.634442 |
| 108000 | full | 8 |  | 0.158852 | 8.548236 | 0.637582 |
| 1048576 | half | 1 |  | 1.487588 | 25.746639 |  |
| 1048576 | half | 8 |  | 0.794806 | 25.911394 |  |
| 1048576 | full | 1 |  | 2.507937 |  | 6.392880 |
| 1048576 | full | 8 |  | 1.463148 |  | 6.618861 |

These rgpot timings are from an earlier version of the public query
that visited every atom pair; `build` now walks the cell list. One
half-list build at 108,000 atoms was still inside its first call when 180 seconds
elapsed, so the larger rgpot cells were not started. ASE's full list
at 1,048,576 atoms was killed by the kernel at 48,137,444 kB and
48,142,196 kB anonymous resident set, once per thread column, before
it printed a time.

Resident set, kB, for the runs that printed one:

| system | list | threads | rgpot | vesin | ASE | matscipy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| triclinic | half | 1 | 5144 | 33408 | 70252 |  |
| triclinic | half | 8 | 6060 | 33460 | 70388 |  |
| triclinic | full | 1 | 6192 | 34220 | 89840 | 31760 |
| triclinic | full | 8 | 7356 | 34180 | 89980 | 31696 |
| 108000 | half | 1 |  | 316020 | 485528 |  |
| 108000 | half | 8 |  | 426784 | 485776 |  |
| 108000 | full | 1 |  | 590764 | 5947312 | 529248 |
| 108000 | full | 8 |  | 802128 | 5947412 | 529368 |
| 1048576 | half | 1 |  | 2786588 | 4144192 |  |
| 1048576 | half | 8 |  | 3524116 | 4144092 |  |
| 1048576 | full | 1 |  | 5452884 |  | 4872420 |
| 1048576 | full | 8 |  | 6641784 |  | 4872272 |

`classical_scaling_bench` prints the serial, strong and weak tables.
`classical_scaling_bench rss EAM` (and `LJ`, `SuttonChen`, `SW`,
`Tersoff`) prints the peak resident set of one million-atom streaming
call in that process. Meson builds the same program as the
`classical_scaling_bench` benchmark.

`thread_count_bench` prints the thread-count tables, a line per
potential, size and thread count, and Meson builds it as the
`thread_count_bench` benchmark. The old build is 47c46f97 with
`CppCore/tests/thread_count_bench.cc` and its Meson target added. The
first table is, on each build,

<!-- rgpot-doc: syntax needs a Meson build and runs for several minutes -->
```bash
for pot in sc lj ljcluster morse zbl eam sw tersoff; do
  taskset -c 0-7 build/CppCore/thread_count_bench --pots "$pot" \
    --sizes 38,55,80,500,4000,10000 --threads 1,8
done
```

`--sizes 100000` gives the second table, and `--mode stream --sizes
10000` the third. Each line also counts, against the first thread count,
the frames whose energy bits differ and the force components whose bits
differ, and hashes the bits of every energy and force, so two builds can
be compared.

`--team-from` sets the threshold of every potential for the crossovers
of [Per-potential thresholds](#per-potential-thresholds): `none` sums
directly on the calling thread, and `0` sums by rows and opens a team on
any thread count above one. `--cutoff` sets the cutoff of LJ, the LJ
cluster and Morse. The first table of
[Before and after the thresholds](#before-and-after-the-thresholds) is
the loop above with `--threads 1,2,4,8`, its stream table takes
`--mode stream --sizes 10000` on 47c46f97 as well, and the LJ cluster at
100000 atoms takes `--frames 2 --runs 3`.

`morse_omp_bench` (`CppCore/tests/morse_omp_bench.cc`, Meson benchmark
`morse_omp_bench`) runs a smoke check for `MorsePot` at 1, 2, 4, 8 and
16 threads on a 6×6×6 Cu vacancy (863 atoms, below the neighbour-list
default of 8192 atoms), a 14×14×14 Cu vacancy (10,975 atoms, stored list)
and the 10,000-atom Pt cluster, checking bit-for-bit agreement against 1
thread. On the 8-core Xeon host above (GCC 13.3.0, release build),
evaluating each stored upper pair once and batching `simd::exp_fixed_n`
moves the 14×14×14 Cu vacancy median from 10,795 / 11,317 / 5,771 µs at
1 / 2 / 4 threads to 5,591 / 3,459 / 1,789 µs (1.62× at 2 threads, 3.12×
at 4 threads), the 10,000-atom Pt cluster from 36,007 / 45,698 / 23,290
µs to 25,026 / 17,672 / 9,185 µs (1.42× at 2 threads, 2.72× at 4 threads),
and the 863-atom Cu vacancy from 805 µs to 457 µs.
