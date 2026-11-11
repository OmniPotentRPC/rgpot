# Shared neighbour layer

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
| EAM | 1048576 | 139820 |
| Lennard-Jones | 1048576 | 524968 |
| Sutton-Chen | 1048576 | 131320 |
| Stillinger-Weber | 1000000 | 112724 |
| Tersoff | 1000000 | 119516 |

Lennard-Jones streaming still materialises the half-list the pair kernel
reads, which is why its resident set is larger. The other four stay near
the atom and cell arrays. The previous Tersoff stream shared a process
with the rest of the sweep and reached 5,060,708 kB because every atom
reserved 32 bonds.

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
| 108000 | 0.522220 | 1.504529 | 4.741688 |  |  |  |
| 1048576 | 5.126657 | 13.018659 | 38.846029 | 1000000 | 1.094669 | 1.182631 |

Lennard-Jones and Sutton-Chen at one million atoms are an FCC crystal of
64 repeats (1,048,576 atoms). Stillinger-Weber and Tersoff are a diamond
of 50 repeats (1,000,000 atoms).

### Strong scaling

Metals are 32,000 atoms. Diamond potentials are 13,824 atoms. Speedup is
against the one-thread time of that mode.

| potential | mode | 1 | 2 | 4 | 8 |
| --- | --- | ---: | ---: | ---: | ---: |
| EAM | stored | 0.032498 | 0.019711 (1.65x) | 0.010249 (3.17x) | 0.005425 (5.99x) |
| EAM | stream | 0.161162 | 0.173952 (0.93x) | 0.087949 (1.83x) | 0.044341 (3.63x) |
| LJ | stored | 0.011226 | 0.006290 (1.78x) | 0.004067 (2.76x) | 0.002689 (4.18x) |
| LJ | stream | 0.407436 | 0.210205 (1.94x) | 0.109476 (3.72x) | 0.059021 (6.90x) |
| Sutton-Chen | stored | 0.169374 | 0.113260 (1.50x) | 0.057294 (2.96x) | 0.030835 (5.49x) |
| Sutton-Chen | stream | 1.573356 | 0.803790 (1.96x) | 0.406464 (3.87x) | 0.209057 (7.53x) |
| SW | stored | 0.006087 | 0.003452 (1.76x) | 0.001706 (3.57x) | 0.001026 (5.93x) |
| SW | stream | 0.015788 | 0.008106 (1.95x) | 0.004157 (3.80x) | 0.002254 (7.00x) |
| Tersoff | stored | 0.008718 | 0.006973 (1.25x) | 0.006194 (1.41x) | 0.006576 (1.33x) |
| Tersoff | stream | 0.017359 | 0.009993 (1.74x) | 0.006133 (2.83x) | 0.006370 (2.73x) |

Stored EAM at 108,000 atoms: 0.109018 s, 0.073623 s (1.48x), 0.038092 s
(2.86x), 0.020201 s (5.40x). The Lennard-Jones stream, which was flat at
1.03x, now follows the threaded cell walk. Stillinger-Weber bond building
is inside that same split. Tersoff keeps improving through 4 threads and
levels off at 8 on 13,824 atoms.

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
