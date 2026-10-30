The Lennard-Jones cluster kernel no longer crashes at thread exit when RocksDB links jemalloc, as the conda-forge `rocksdb` does with `RGPOT_WITH_CACHE=ON`.
Its aligned scratch buffer comes from `posix_memalign`, which jemalloc replaces along with `free`.
