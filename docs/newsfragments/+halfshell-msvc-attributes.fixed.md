The half-shell neighbour walk compiles with MSVC.
Its inline and cold attributes use `__forceinline` and `__declspec(noinline)` there, and its branch hint is a plain condition.
