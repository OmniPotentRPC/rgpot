The aarch64 Linux wheels build again.
The minimum-image fold puts `target("fma")` only on x86-64.
That is the one target where `fma_hw_active()` can select that body.
The half-shell pair walk builds with MSVC.
Under MSVC it uses `__forceinline`, `__declspec(noinline)` and a plain test.
GCC and Clang keep their attributes and `__builtin_expect`.
The pair and Sutton-Chen headers declare their C entry points for export.
MSVC rejected definitions that added the export to a plain declaration.
The Tersoff bench prints the MSVC version, since MSVC has no `__VERSION__`.
The manylinux wheel jobs retry the gfortran and OpenBLAS install.
Each retry reads fresh metadata from the primary AlmaLinux repository.
The mirror list could return the same broken mirror on every attempt.
The jobs stop when the install fails, instead of failing on a missing OpenBLAS.
