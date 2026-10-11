int bump_inline(int *p) { return __atomic_fetch_add(p, 1, __ATOMIC_RELAXED); }
