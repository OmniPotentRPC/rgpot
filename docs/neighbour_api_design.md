# Neighbour query, four passes

The public surface is `rgpot::nlist::Query`, `Builder` and `Walk`, the C
ABI in `CppCore/rgpot/nlist/rgpot_nlist.h`, and `NeighborRequest` /
`NeighborList` in `CppCore/rgpot/rpc/Potentials.capnp`. The classical
kernels keep `rgpot::nlist::Neighbours`. This log is the design of the
public surface. Each pass below was written against the API as it stood,
then the API moved.

## Survey

Read in November 2026. Vendored vesin is an older snapshot
(`third_party/vesin/vesin.h`); the notes below use the upstream files
unless they say otherwise.

### vesin

C header: [vesin/include/vesin.h](https://github.com/Luthaf/vesin/blob/master/vesin/include/vesin.h).
Python: [python/vesin/vesin/_neighbors.py](https://github.com/Luthaf/vesin/blob/master/python/vesin/vesin/_neighbors.py).
The repository root
([contents listing](https://github.com/Luthaf/vesin)) has `vesin`,
`python`, `fortran`, `vesin-jax` and `vesin-torch`. It has no Rust
crate. A Rust caller uses the C ABI, which is the same route this
library offers.

What it gets right:

- One options struct and one list struct. Pairs, shifts, distances and
  vectors are parallel arrays. The vector is documented as
  `r_j - r_i + S @ H`, and the box is one lattice vector per row.
- `full` is a bool, not a pile of flags. Upstream Python refuses
  `padded_to` unless `full_list` is true (`_neighbors.py`, the
  constructor). That combination cannot succeed by accident.
- Skin is explicit. A positive skin rebuilds when an atom moves more
  than `skin / 2`. Zero disables the cache. Negative asks the library
  to pick a skin. The comment says so.
- `vesin_neighbors` returns an `int32_t` and a `const char*` message
  the caller does not free. Failure is not a zero length list.
- `pre_allocated_outputs` is the caller-owned path. Otherwise vesin
  allocates and `vesin_free` releases. Ownership is one bool, not a
  hidden cache.
- `n_threads == 0` means the OpenMP default, and the comment says so.

What it gets wrong:

- `VesinOptions` is a C struct with no `sizeof` field. Adding a field
  changes the ABI with no way for an old caller to say how large a
  struct it passed.
- A zeroed `VesinOptions` has `cutoff == 0`, `full == false`, and every
  `return_*` flag false. The obvious C call returns a half list with no
  geometry. Python makes `cutoff` and `full_list` keyword-only, so the
  safe path is not the C path.
- `error_message` points at storage the header does not describe as
  thread-local or immortal. Callers cannot tell whether the next call
  overwrites it.
- There is no chunked walk. The whole list is materialised, or the
  caller guesses a capacity and the call fails.
- The vendored header in this tree has none of `padded_to`,
  `pre_allocated_outputs` or the negative-skin convention. Matching the
  vendored snapshot would freeze those gaps.

### JAX-MD

[jax_md/partition.py](https://github.com/jax-md/jax-md/blob/main/jax_md/partition.py),
`neighbor_list` at the function of that name and `NeighborListFormat`.

What it gets right:

- `Dense`, `Sparse` and `OrderedSparse` are an enum.
  `is_format_valid` rejects anything else instead of silently picking
  dense.
- Overflow is data: `did_buffer_overflow` and `PartitionError`. A jit
  update cannot throw, so the bit is the error channel, and the example
  in the docstring reallocates when it is set.
- `allocate` and `update` are different functions. The capacity is
  chosen at allocate time (`capacity_multiplier`, default 1.25) and the
  update cannot change a shape.
- The rebuild distance is `dr_threshold / 2`, the same half-skin rule
  as vesin. The docstring says the box is upper triangular with lattice
  vectors as columns, the transpose of ASE's cell. The convention is
  written next to the argument.

What it gets wrong:

- The default format is `Dense`, whose width is the busiest atom. A
  sparse consumer who forgets `format` pays that padding and then
  indexes a square matrix as if it were pairs.
- `mask_self` defaults to true, so the self pair is dropped, but a
  custom mask can put it back. The self pair is a policy argument, not
  a consequence of the format.
- `NeighborListFns.__call__` still accepts the old "allocate or update"
  signature and logs a deprecation. Two ways to do the same thing
  remain.
- There is no streaming walk. A list that does not fit the static
  buffer does not exist until the caller allocates again.
- The error code is a bitfield on the list. Ignoring
  `did_buffer_overflow` and reading `idx` is a successful-looking wrong
  answer.

### LAMMPS

[src/neigh_request.h](https://github.com/lammps/lammps/blob/develop/src/neigh_request.h)
and
[src/KOKKOS/neigh_list_kokkos.h](https://github.com/lammps/lammps/blob/develop/src/KOKKOS/neigh_list_kokkos.h).

What it gets right:

- The request is separate from the list. Half is the default (`half`
  starts set, `full` does not). `enable_full` is a method, so a caller
  who wants both directions asks for that.
- Kokkos stores a 2D view of neighbour indices plus `d_numneigh`.
  `get_neighbors(i)` returns a small view of one atom's partners. The
  device kernel does not gather an array of structs.
- A request can name a fixed cutoff (`set_cutoff_fixed`) instead of
  inheriting the maximum pair cutoff. Analysis and pair styles do not
  share a silent default.

What it gets wrong:

- The request is a long list of `int` flags (`half`, `full`, `ghost`,
  `newton`, `occasional`, `bond`, package bits). Nothing in the type
  stops `half` and `full` both being 1. `Neighbor` later morphs the
  flags (`halffull`, `copy`, `trim`). The state a user set and the
  state that runs are different structs sharing one object.
- Defaults live in the constructor, not in the header next to the
  fields. A reader of the struct cannot see them.
- There is no per-call error code. A bad request fails inside the
  neighbour build, often as a LAMMPS error string far from the flag
  that caused it.
- The list is per atom, not a pair table. A consumer that wants
  `(i, j, shift, vector)` has to reconstruct it. That is the right
  layout for a LAMMPS pair style and the wrong one for metatomic.

### ArborX

[src/spatial/detail/ArborX_NeighborList.hpp](https://github.com/kokkos/ArborX/blob/master/src/spatial/detail/ArborX_NeighborList.hpp).

What it gets right:

- Half and full are two functions, `findHalfNeighborList` and
  `findFullNeighborList`. There is no flag that can be both.
- The output is Kokkos views the caller owns: offsets of length
  `n + 1` and a flat index array. The library reallocates those views.
  CSR is a zero-copy layout for a kernel that walks one atom's
  neighbours.
- The execution space is an argument. Which thread (or which GPU) runs
  the query is not a global.

What it gets wrong:

- The experimental header has no shifts and no periodic box. The
  predicate is a sphere of radius `radius` in the coordinates it is
  given. A triclinic cell is the caller's problem.
- There is no status return. A view the execution space cannot see
  fails the `static_assert` on accessibility, which is a compile error
  for that case and nothing for a bad radius.
- Indices only. Distances and vectors are not part of the result, so
  every consumer recomputes them.
- No ABI story. It is a C++ header, and the views follow Kokkos.

### matscipy and ASE

matscipy
[matscipy/neighbours.py](https://github.com/libAtoms/matscipy/blob/master/matscipy/neighbours.py),
`neighbour_list`.
ASE
[ase/neighborlist.py](https://gitlab.com/ase/ase/-/blob/master/ase/neighborlist.py),
`neighbor_list` and `NewPrimitiveNeighborList`.

What they get right:

- The quantity string (`"ijSdD"`) asks for indices, shifts, distances
  and vectors, and unknown letters raise `ValueError`
  (`make_result` in matscipy). The caller does not allocate arrays they
  will not read.
- Both document `D = positions[j] - positions[i] + S.dot(cell)` next to
  the function. The cell is the ASE 3×3 matrix, rows as lattice
  vectors.
- matscipy returns triplets (`ijk`) from the same connectivity as
  pairs. A three-body term does not build a second list with a second
  convention.
- ASE's primitive list takes `bothways` as a bool. Half is the default
  on the class.

What they get wrong:

- ASE's function defaults `self_interaction` to false. The class
  `NewPrimitiveNeighborList` defaults it to true. The same word does
  not mean the same default.
- ASE rebuilds when the largest squared displacement exceeds
  `skin**2` (`update` compares against `self.skin**2`). That is a full
  skin, not half a skin. Two atoms can each move almost `skin` towards
  each other and a pair that belongs inside the cutoff is still
  missing. The skin default of 0.3 hides the comparison.
- The quantity string is untyped. `"ijS"` and `"jiS"` are both legal
  and easy to swap, and the result tuple changes length with the
  string. A C++ signature cannot express that.
- matscipy's default `pbc` is a nonperiodic box. Forgetting it on a
  crystal drops every image. ASE takes periodicity from the `Atoms`
  object, so the same omission does not happen there.
- Neither has a C ABI or a version field.

### freud

[freud/locality/NeighborQuery.h](https://github.com/glotzerlab/freud/blob/main/freud/locality/NeighborQuery.h),
[AABBQuery.h](https://github.com/glotzerlab/freud/blob/main/freud/locality/AABBQuery.h),
[NeighborList.h](https://github.com/glotzerlab/freud/blob/main/freud/locality/NeighborList.h),
[NeighborBond.h](https://github.com/glotzerlab/freud/blob/main/freud/locality/NeighborBond.h).

What it gets right:

- `NeighborQuery` is the object that holds points and the box.
  `query` returns an iterator. The caller can stop. `AABBQuery` is one
  acceleration structure behind that interface, and the header says how
  the tree and the periodic images work.
- `QueryArgs` is a single struct. `validateQueryArgs` infers ball
  versus nearest from which fields were set, then rejects a combination
  that still does not make sense. `QueryType::none` exists so a zeroed
  argument is not silently a ball query.
- A point set of length zero throws `std::invalid_argument`. A 2D box
  with a nonzero z throws. Failure is an exception with a sentence.
- `NeighborList` stores query index and point index in an `(n_bonds, 2)`
  array, plus flat distances and vectors. Bonds are also a small struct
  for the iterator. The stored list and the streamed bond are the same
  fields.
- The query borrows the points pointer. The constructor does not copy
  the coordinates. Lifetime is the caller's, and the type does not
  pretend otherwise.

What it gets wrong:

- `query` throws `std::domain_error` when any axis is nonperiodic
  ("Pair queries in a non-periodic box are not implemented"). Partial
  periodicity is a hard error, not a supported convention.
- `DEFAULT_R_MAX` is `-1`. A forgotten cutoff is a sentinel, and the
  validity check has to know that sentinel. A negative length that is
  not the sentinel is a different bug.
- `NeighborBond` is an array of structs on the iterator path and a
  structure of arrays on `NeighborList`. Callers learn both.
- The points pointer is bare. Nothing in the type records that the
  array must outlive the query. The documentation is the comment, and
  a use after free compiles.
- No C ABI. The Python layer is Cython over this C++.

### torch-nl

[torch_nl/neighbor_list.py](https://github.com/felixmusil/torch_nl/blob/main/torch_nl/neighbor_list.py).

What it gets right:

- The linked-cell candidate list and the strict cutoff are separate.
  `strict_nl` keeps pairs with `d2 < cutoff * cutoff`. The strict
  inequality is the code, not a comment.
- The return is three tensors: `mapping` of shape `(2, n_neighbors)`,
  a batch index, and integer cell shifts. Structure of arrays, and the
  mapping is `(center, neighbor)`.
- `self_interaction` defaults to false, and the docstring says so.
- Positions are documented as already wrapped into the cell.

What it gets wrong:

- The docstring says partial periodic boundaries are not supported, and
  the code still accepts a `pbc` tensor. A mixed `[true, true, false]`
  is a runtime surprise rather than a type that cannot say it.
- There is no skin and no error type. A bad cell fails inside a tensor
  kernel.
- Shifts are indices into the cell, and the vector is recomputed by
  the caller (`pos[j] - pos[i] - cell_shifts` in `strict_nl`, with the
  opposite sign from vesin's `r_j - r_i + S H` depending on how
  `compute_cell_shifts` is defined). Two libraries in this survey
  disagree on the sign unless the caller reads both implementations.
- No version field and no C ABI. TorchScript is the stability boundary.

### What this API should take

From that reading, the public query should:

- Use one layout description that cannot be a half list and a cluster
  at the same time (vesin's `padded_to` check, ArborX's two functions,
  JAX's enum check).
- Default the way the documented example does, in C++ and in C
  (vesin's Python is strict, its C zero-fill is not).
- Return a code and a stable message in C, and throw one exception
  type in C++ (vesin's status pointer, freud's `invalid_argument`).
  Do not stash the message in a global.
- Hand the caller buffers or an owning table of parallel arrays, and
  a walk that fills those arrays in chunks (JAX overflow plus freud's
  iterator). The C structs and the Cap'n Proto structs name the same
  arrays.
- Put `bytes` first on every C struct the caller allocates, so a later
  field can be added without a new major (the gap in `VesinOptions`).
- Document the cell as ASE's row vectors, the vector as
  `r_j - r_i + S H`, the strict test as `r < cutoff`, and the skin
  rebuild as half the skin. Say which calls may run on two threads.

## Baseline, before any pass

A half-list Lennard-Jones energy against the API at `2a195864` looks
like this. The fields that make it a half list are the defaults, and
the fields that make it periodic and stored are not, depending on the
language.

```cpp
rgpot::nlist::Query query;
query.cutoff = 2.5;
query.format = rgpot::nlist::Format::Half;
query.cut = rgpot::nlist::Cut::Strict;
query.mode = rgpot::nlist::Mode::Stored;
rgpot::nlist::Table table;
rgpot::nlist::build(table, positions, n, box, query);
for (std::size_t k = 0; k < table.size(); ++k) {
  const double r = table.distance[k];
  const double sr6 = std::pow(1.0 / r, 6.0);
  energy += 4.0 * (sr6 * sr6 - sr6);
}
```

The same request in C, zero-filled the way a C caller writes a struct:

```c
rgpot_nlist_request request;
memset(&request, 0, sizeof request);
request.abi = RGPOT_NLIST_ABI;
request.cutoff = 2.5;
size_t count = 0, needed = 0;
rgpot_nlist_build(positions, n, box, &request, NULL, 0, &count, &needed);
```

Counted by hand against `rgpot_nlist.h` and `NeighbourList.cc`:

- The C++ call is about fifteen lines before the energy loop and is
  the right physics if `Mode::Auto` does not refuse to store.
  `Mode::Auto` compares a combinatorial pair bound with
  `availableBytes()` and returns an empty table with `stored == false`
  (`build` in `NeighbourList.cc`). The loop above then reports zero
  energy. The call compiled.
- A zeroed C request has `periodic` all false, `want_shifts`,
  `want_vectors` and `want_distances` all false, and `mode` 0, which
  `query_from_c` reads as `Mode::Auto`. The C++ defaults are periodic,
  all three fields, and `Mode::Stored`. The same mental model is not
  the same call.
- `format` other than 1 or 2 becomes a half list. `cut` other than 0
  becomes loose. A typo succeeds.
- `rgpot_nlist_build` returns 1 only when `abi` mismatches. A null
  `positions` with `n > 0` is undefined. There is no message function.
- `rgpot_nlist_walk_pull` copies every chunk through temporary vectors
  into an array of `rgpot_nlist_pair`. Cap'n Proto and C++ are
  structures of arrays. C is an array of structs.
- `Walk::pull` returns 0 and leaves `done` false when `dest.first` is
  null (`NeighbourList.cc`). A caller who forgot a pointer loops.
- `NeighborRequest.mode` defaults to 1 (stored) in the schema, while C
  mode 0 is auto. A stream sets `more` and the client repeats the whole
  call, with no cursor, so a stateless server cannot choose the next
  chunk.
- Cluster4 is a third `Format`, so a half list and a cluster are
  mutually exclusive in the enum. The mask array is still present on a
  half list, filled with `0xff`, and a C caller who sets `format` to 2
  while leaving the rest of a half-list mental model gets every
  directed pair packed into groups of four. Nothing says that cluster
  means full.

The passes below close those gaps without changing the pair walk the
kernels use.
