The comment in `scripts/run_coverage_fortran.sh` now describes what the Fortran
capture does with each lcov version: lcov 2 gets the `inconsistent`,
`mismatch` and `negative` ignore-errors classes for toolchain header noise and
OpenMP arc-counter underflow, and lcov 1.15 gets only the branch coverage rc
setting, whose name differs by major version.
