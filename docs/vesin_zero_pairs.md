# vesin lists with zero pairs

A neighbour list whose system has no pair inside the cutoff (one atom,
atoms farther apart than the cutoff, or no atoms) comes back from vesin
with `length == 0`. Two things in vesin make that list unsafe to read,
and both are present upstream. This note has a reproducer for each and a
proposed patch, [`vesin_zero_pairs.patch`](vesin_zero_pairs.patch),
against Luthaf/vesin 8074d9a (v0.6.2-40). Nothing has been sent
upstream.

rgpot carries the same fixes in its vendored copy
(`third_party/vesin/`), along with two that only concern code that
exists there and not upstream.

## What goes wrong upstream

1. `GrowableNeighborList::reset()` (`vesin/src/cpu/cell_list.cpp:878`)
   allocates the shift, distance and vector arrays a caller asked for,
   but never the pairs array, which only `grow()` creates. A list whose
   builds have all found zero pairs therefore has `pairs == NULL` next to
   three allocated arrays. A debug build also passes that NULL to
   `memset` (`cell_list.cpp:890`), which is undefined even for zero
   bytes; UBSan reports it.
2. The Fortran interface (`fortran/src/vesin.f90:290-301`) associates
   `list%pairs` and the other arrays only when the C pointer is non-NULL.
   With the pairs array missing, `list%pairs` stays disassociated, and
   `size()` of a disassociated pointer is undefined. gfortran reads the
   descriptor anyway: the reproducer below gets `size(list%pairs, 2) = 1`
   and then loads from address zero. In rgpot a stale descriptor reported
   4194209 columns.

An empty system takes the early return at `vesin/src/vesin.cpp:57`. It
sets `length = 0` and leaves the arrays as they were, so a fresh list
also has `pairs == NULL` there. The Fortran change covers that case
too. Because of this early return, the `memcpy` of zero reference points
in `VerletList` (`verlet.cpp:162`) cannot be reached through the C API
upstream. It can in rgpot's vendored copy, which has no early return.

## Reproducer, C++

`vesin.h` uses default member initialisers upstream, so this is
compiled as C++.

```cpp
#include <cstdio>

#include "vesin.h"

static int run(const double (*points)[3], size_t n, const char *name) {
    const double box[3][3] = {{30, 0, 0}, {0, 30, 0}, {0, 0, 30}};
    const bool periodic[3] = {false, false, false};
    VesinOptions options{};
    options.cutoff = 3.0;
    options.full = true;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;
    options.algorithm = VesinCellList;
    VesinNeighborList neighbors{};
    const char *error = nullptr;
    int status = vesin_neighbors(points, n, box, periodic, VesinDevice{VesinCPU, 0}, nullptr,
                                 options, &neighbors, &error);
    std::printf("%-12s status=%d length=%zu pairs=%p shifts=%p distances=%p vectors=%p\n", name,
                status, neighbors.length, (void *)neighbors.pairs, (void *)neighbors.shifts,
                (void *)neighbors.distances, (void *)neighbors.vectors);
    int bad = status != 0 || neighbors.length != 0 || neighbors.pairs == nullptr;
    vesin_free(&neighbors);
    return bad;
}

int main() {
    const double one[1][3] = {{1, 1, 1}};
    const double apart[2][3] = {{1, 1, 1}, {11, 1, 1}};
    int bad = run(one, 1, "one atom") | run(apart, 2, "apart");
    std::printf(bad ? "FAIL: a list of length 0 has pairs == NULL\n" : "OK\n");
    return bad;
}
```

## Reproducer, Fortran

```fortran
program zero_pairs
    use vesin
    use, intrinsic :: iso_c_binding
    implicit none
    type(NeighborList) :: list
    real(c_double) :: points(3, 1), box(3, 3)
    integer :: status
    integer(c_size_t) :: k
    real(c_double) :: total
    logical :: has_pairs

    points(:, 1) = [1d0, 1d0, 1d0]
    box = 0d0
    box(1, 1) = 30d0; box(2, 2) = 30d0; box(3, 3) = 30d0
    list = NeighborList(cutoff=3d0, full=.true., sorted=.false.)
    call list%compute(points, box, [.false., .false., .false.], status)
    has_pairs = associated(list%pairs)
    print *, 'status', status, 'length', list%length, 'pairs associated', has_pairs
    if (.not. has_pairs) then
        print *, 'size(list%pairs, 2) =', size(list%pairs, 2), '(undefined)'
    end if
    total = 0d0
    do k = 1, size(list%pairs, 2, kind=c_size_t)
        total = total + real(list%pairs(1, k), c_double)
    end do
    print *, 'visited', total
    call list%free()
    if (.not. has_pairs) error stop 'FAIL: list%pairs disassociated'
end program zero_pairs
```

## Running them

```sh
cmake -S "$VESIN_SRC" -B build -DCMAKE_BUILD_TYPE=Debug -DVESIN_FORTRAN=ON \
  -DCMAKE_CXX_FLAGS=-fsanitize=address,undefined \
  -DCMAKE_Fortran_FLAGS=-fsanitize=address,undefined
cmake --build build
export ASAN_OPTIONS=detect_leaks=0
g++ -fsanitize=address,undefined -g -I "$VESIN_SRC/vesin/include" zero_pairs.cpp \
  build/vesin/libvesin.a -lpthread -o zero_pairs_cpp && ./zero_pairs_cpp
gfortran -fsanitize=address,undefined -g -I build/fortran/include zero_pairs.f90 \
  build/fortran/libvesin_fortran.a build/vesin/libvesin.a -lstdc++ -lpthread \
  -o zero_pairs_f90 && ./zero_pairs_f90
```

At 8074d9a, both runs report
`cell_list.cpp:890:16: runtime error: null pointer passed as argument 1`.
The C++ program prints `pairs=(nil)` with the other three arrays
allocated, and exits 1. The Fortran program prints
`pairs associated F`, then `load of null pointer` at the loop, and
exits on SIGSEGV.

With the patch, both print `OK` (`pairs associated T`, `visited 0`) and
the sanitizers are silent. vesin's own suites still pass under ASan and
UBSan: `neighbors` (259 assertions), `pre_allocated` (88) and
`fortran-tests`.

## The patch

- `reset()` poisons the pairs array only when it exists. When it is
  missing, `reset()` grows the list to one slot (or to its current
  capacity), so every requested array of a list of length zero is
  non-NULL.
- `vesin_compute_c_double` points each requested Fortran array at a
  module target of extent zero when the list is empty and the C pointer
  is NULL. Loops over `size(list%pairs, 2)` then run zero times. This
  also covers the empty-system early return and any older library.

## Local to rgpot's copy

Two faults were in code that exists only in the vendored copy:

- The brute-force search (an eOn addition) grew a full list with
  `grow()` until two more pairs fit. `grow()` with no size asks for twice
  the stored length, which is one slot for an empty list, so a full list
  with a capacity below two never left the loop. Upstream's CPU path
  counts the pairs first and grows once.
- The vendored copy has no empty-system early return, so
  `VerletList::rebuild` reached `memcpy(ref_points_.data(), points, 0)`
  with NULL pointers.
