# Shared neighbour layer

EAM, Lennard-Jones, Sutton-Chen, Stillinger-Weber and Tersoff take
candidate pairs from `rgpot::nlist::Neighbours`.

The layer bins atoms into cells whose edge is the cutoff plus the skin.
The minimum image is one of three folds. `Round` is the orthorhombic
truncation used by the pair kernels. `Half` is one correction about the
half-box, which is what EAM uses after wrapping coordinates into the
cell. `General`, and any sheared cell, calls `MinimumImage`.

`Mode::Stored` keeps a Verlet list. A later call reuses it while every
atom has moved by at most half the skin and the cell, the count and the
spec are unchanged. Skin 0 reuses the list only when the coordinates and
the cell match. `Mode::Stream` walks the 27 neighbouring cells and stores
no pair list, so the extra memory is the atoms and the bins. Bins are
reused only when the geometry is identical, because a move can change
which cell an atom sits in. `Mode::Auto` stores when the candidate index
array fits in `MemAvailable` and in a 32-bit offset, and streams
otherwise. There is no cutoff on the atom count. A caller can force
either mode with `setNeighbourMode`.

`OMP_NUM_THREADS` is read on each call. Unset, empty, or non-positive
means one thread, and that call keeps the serial pair order. A larger
value splits atoms into contiguous slices. Each thread writes a private
force buffer. Those buffers, and the partial energies, are added in
thread-id order. A fixed thread count therefore repeats bit for bit.
It is not bit-identical to one thread. Energy and every force component
stay within a relative tolerance of `1e-9`.

One thread is the previous evaluation. EAM still runs `accumulate` in
atom order. Lennard-Jones still calls the same pair kernel. Sutton-Chen
still visits each pair from the lower index. Stillinger-Weber and
Tersoff, when the mode is left at `Auto`, still build pairs with
`orderedPairs`, so the published bond order is unchanged. `Stored` and
`Stream` on those two potentials use this layer. Morse and ZBL still
accumulate through `PairListCache`, which is the bitwise FMA path for
those kernels.

## Timings

One Intel Xeon socket, 8 cores, 1 thread per core. g++ 13.3.0,
`-O3 -DNDEBUG -fopenmp`. Each time is the median of seven warm calls
after two discarded calls. FCC metals use a cubic repeat. Diamond silicon
uses the 8-atom basis and lattice constant 5.431 Angstrom. The peak
resident set of the whole run was 5,060,708 kB, reached after the
million-atom Tersoff stream. The high-water mark after the million-atom
EAM stream was 140,044 kB, after Lennard-Jones 524,932 kB, and after
Stillinger-Weber 1,960,804 kB.

### Serial, stored list

| atoms | EAM (s) | LJ (s) | Sutton-Chen (s) | atoms | SW (s) | Tersoff (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 108 | 0.000112 | 0.000038 | 0.000256 | 64 | 0.000025 | 0.000040 |
| 500 | 0.000547 | 0.000172 | 0.002828 | 216 | 0.000083 | 0.000141 |
| 2048 | 0.002130 | 0.000709 | 0.011387 | 1000 | 0.000363 | 0.000601 |
| 6912 | 0.007174 | 0.002383 | 0.037412 | 4096 | 0.001565 | 0.002720 |
| 32000 | 0.032605 | 0.011190 | 0.170623 | 13824 | 0.005709 | 0.009322 |
| 108000 | 0.109318 | 0.037900 | 0.560037 |  |  |  |

### Serial, streaming walk

| atoms | EAM (s) | LJ (s) | Sutton-Chen (s) | atoms | SW (s) | Tersoff (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 108 | 0.000237 | 0.000160 | 0.000290 | 64 | 0.000050 | 0.000085 |
| 500 | 0.003311 | 0.003587 | 0.005853 | 216 | 0.000264 | 0.000292 |
| 2048 | 0.013028 | 0.040405 | 0.084850 | 1000 | 0.001201 | 0.001278 |
| 6912 | 0.038590 | 0.105734 | 0.366937 | 4096 | 0.004853 | 0.005479 |
| 32000 | 0.158288 | 0.402878 | 1.577840 | 13824 | 0.016630 | 0.017924 |
| 108000 | 0.508420 | 1.494474 | 4.734922 |  |  |  |
| 1048576 | 5.012205 | 12.937822 |  | 1000000 | 1.266312 | 1.374535 |

Lennard-Jones at one million atoms is an FCC crystal of 64 repeats
(1,048,576 atoms). Stillinger-Weber and Tersoff at one million atoms are
a diamond of 50 repeats (1,000,000 atoms). Sutton-Chen was not run at
that size. The streaming walk is several times the stored list, and it
is the mode that stays within the atom and cell arrays.

### Strong scaling

Metals are 32,000 atoms. Diamond potentials are 13,824 atoms. Speedup
is against the one-thread time of that mode.

| potential | mode | 1 | 2 | 4 | 8 |
| --- | --- | ---: | ---: | ---: | ---: |
| EAM | stored | 0.032441 | 0.019601 (1.66x) | 0.016078 (2.02x) | 0.008380 (3.87x) |
| EAM | stream | 0.158777 | 0.179846 (0.88x) | 0.093536 (1.70x) | 0.046334 (3.43x) |
| LJ | stored | 0.011193 | 0.006038 (1.85x) | 0.003589 (3.12x) | 0.003066 (3.65x) |
| LJ | stream | 0.407014 | 0.398125 (1.02x) | 0.395795 (1.03x) | 0.395614 (1.03x) |
| Sutton-Chen | stored | 0.176887 | 0.115559 (1.53x) | 0.058263 (3.04x) | 0.029564 (5.98x) |
| Sutton-Chen | stream | 1.577654 | 0.809214 (1.95x) | 0.406435 (3.88x) | 0.211310 (7.47x) |
| SW | stored | 0.005686 | 0.005363 (1.06x) | 0.005209 (1.09x) | 0.005149 (1.10x) |
| SW | stream | 0.016421 | 0.015824 (1.04x) | 0.015784 (1.04x) | 0.015694 (1.05x) |
| Tersoff | stored | 0.009639 | 0.008298 (1.16x) | 0.008945 (1.08x) | 0.008848 (1.09x) |
| Tersoff | stream | 0.017545 | 0.015664 (1.12x) | 0.016538 (1.06x) | 0.016461 (1.07x) |

Stored EAM at 108,000 atoms: 0.109320 s, 0.075928 s (1.44x), 0.038904 s
(2.81x), 0.036770 s (2.97x). Two threads used to be 1.20x when density
and the force were separate passes. The fused stored path is the 1.44x
row. Eight threads at that size stayed near the four-thread time.

The Lennard-Jones stream does not speed up with threads. The cell walk
that builds the pair view runs on one thread, and that walk is the bulk
of the call. The stored list is already built, so the threaded kernel
scales. Stillinger-Weber and Tersoff spend the call building bonds; the
threaded angle loop is a small part of 14,000 atoms.

### Weak scaling

Metal cells are 6x6x6, 6x6x12, 6x12x12 and 12x12x12 conventional FCC
cells (864, 1,728, 3,456 and 6,912 atoms) on 1, 2, 4 and 8 threads.
Diamond repeats are 4, 5, 6 and 8 (512, 1,000, 1,728 and 4,096 atoms).

| potential | mode | 1 thread | 2 | 4 | 8 |
| --- | --- | ---: | ---: | ---: | ---: |
| EAM | stored | 0.000928 | 0.001256 | 0.001286 | 0.001371 |
| EAM | stream | 0.006432 | 0.013056 | 0.011702 | 0.011729 |
| LJ | stored | 0.000297 | 0.000353 | 0.000466 | 0.000526 |
| LJ | stream | 0.007940 | 0.018681 | 0.044433 | 0.117892 |
| Sutton-Chen | stored | 0.004792 | 0.006526 | 0.008030 | 0.006989 |
| Sutton-Chen | stream | 0.017154 | 0.023848 | 0.033540 | 0.057136 |
| SW | stored | 0.000189 | 0.000389 | 0.000624 | 0.001568 |
| SW | stream | 0.000755 | 0.001213 | 0.002152 | 0.005467 |
| Tersoff | stored | 0.000306 | 0.000496 | 0.001012 | 0.002545 |
| Tersoff | stream | 0.000690 | 0.001170 | 0.002243 | 0.005848 |

## Reproduce

```bash
g++ -std=c++20 -O3 -DNDEBUG -fopenmp -I CppCore -I CppCore/rgpot/EAM \
  -DEAM_STANDALONE \
  CppCore/tests/classical_scaling_bench.cc \
  CppCore/rgpot/EAM/EAM.cpp CppCore/rgpot/EAM/EAMPot.cc \
  CppCore/rgpot/LennardJones/LJPot.cc \
  CppCore/rgpot/SuttonChen/SuttonChenPot.cc \
  CppCore/rgpot/SuttonChen/SuttonChenScales.cc \
  CppCore/rgpot/StillingerWeber/SWSi.cc \
  CppCore/rgpot/Tersoff/TersoffSiC.cc \
  CppCore/rgpot/nlist/Neighbours.cc \
  CppCore/rgpot/nlist/HalfShell.cc CppCore/rgpot/nlist/HalfShellWalk.cc \
  CppCore/rgpot/nlist/MinimumImage.cpp \
  CppCore/rgpot/simd/dispatch.cc CppCore/rgpot/simd/scalar_kernels.cc \
  CppCore/rgpot/simd/exp.cc CppCore/rgpot/simd/exact_fma_hw.cc \
  -o classical_scaling_bench
./classical_scaling_bench
```

Meson builds the same program as the `classical_scaling_bench` benchmark.
`NeighbourLayerTest` checks that the two modes agree within `1e-9` and
that a two-thread run repeats bit for bit and stays within that tolerance
of one thread, for each of the five potentials.
