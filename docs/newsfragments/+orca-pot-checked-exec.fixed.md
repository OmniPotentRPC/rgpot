On Linux, FreeBSD and NetBSD, `ORCAPot` now runs the executable through the
descriptor it checked, with `fexecve`, so a file renamed over the path between
the check and the run is never started; elsewhere it falls back to `execv` and
the check after the run. Each call checks the executable before it removes old
output or writes the input, and a call that fails keeps no orbitals for the
next one. A box whose image bound is not finite or out of range, and a box
around a position that is not finite, are refused with `std::invalid_argument`
before any conversion.
