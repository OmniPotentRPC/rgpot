#include <immintrin.h>

/* Same dispatched kernel as ok.c, with no call into libgcc, so a shared
   object can be linked and the ET_DYN symbol addresses checked. */
__attribute__((target("avx512f"))) void
kern_avx512(float *out, const float *in, int n) {
  for (int i = 0; i < n; i += 16) {
    __m512 v = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(out + i, v);
  }
}
