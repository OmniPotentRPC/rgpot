#include <immintrin.h>

__attribute__((target("avx2"))) void secret_kernel(int *out, const int *in) {
  __m256i v = _mm256_loadu_si256((const __m256i *)in);
  _mm256_storeu_si256((__m256i *)out, _mm256_add_epi32(v, v));
}
