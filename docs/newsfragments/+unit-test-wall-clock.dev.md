Unit tests no longer assert wall-clock bounds, which failed under
`OMP_THREAD_LIMIT`, coverage instrumentation or a loaded runner. The cache
tests check served and computed counts instead of hit timings, and
`thread_count_bench --pots lj --sizes 8788 --cutoff 4.5` times a large
Lennard-Jones cell where a team has enough work to share.
