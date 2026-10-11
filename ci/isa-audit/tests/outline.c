int bump(int *p) { return __atomic_fetch_add(p, 1, __ATOMIC_RELAXED); }

int cas_bits(int *p, int expect, int desire) {
  return __atomic_compare_exchange_n(p, &expect, desire, 0, __ATOMIC_RELAXED,
                                     __ATOMIC_RELAXED);
}

int swap_bits(int *p, int v) {
  return __atomic_exchange_n(p, v, __ATOMIC_ACQUIRE);
}

int clear_bits(int *p, int v) {
  return __atomic_fetch_and(p, ~v, __ATOMIC_RELEASE);
}

int eor_bits(int *p, int v) { return __atomic_fetch_xor(p, v, __ATOMIC_RELAXED); }

int set_bits(int *p, int v) { return __atomic_fetch_or(p, v, __ATOMIC_RELAXED); }
