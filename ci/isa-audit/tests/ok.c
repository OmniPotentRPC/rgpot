#include <immintrin.h>

/* Dispatched AVX-512 kernel. The scalar fallback stays on the default ISA.
   The compiler emits vzeroupper (AVX) at the end of the AVX-512 function. */
__attribute__((target("avx512f"))) void
kern_avx512(float *out, const float *in, int n) {
  for (int i = 0; i < n; i += 16) {
    __m512 v = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(out + i, v);
  }
}

void kern_scalar(float *out, const float *in, int n) {
  for (int i = 0; i < n; i++)
    out[i] = in[i];
}

void entry(float *out, const float *in, int n) {
  if (__builtin_cpu_supports("avx512f"))
    kern_avx512(out, in, n);
  else
    kern_scalar(out, in, n);
}
