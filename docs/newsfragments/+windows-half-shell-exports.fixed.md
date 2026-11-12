The half-shell pair walk builds with the Visual C++ compiler again.
That compiler gets `__forceinline`, `__declspec(noinline)` and a plain test, and GCC and Clang keep their attributes and `__builtin_expect`.
The pair and Sutton-Chen headers declare their C entry points for export, so `cl.exe` accepts the exported definitions.
The Tersoff neighbour-list bench prints `_MSC_FULL_VER` where the compiler has no `__VERSION__`.
