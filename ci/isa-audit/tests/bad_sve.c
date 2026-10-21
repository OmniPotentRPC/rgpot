#include <arm_sve.h>
#include <stdint.h>

__attribute__((target("+sve"))) void other_sve(float *out, const float *in,
                                                unsigned n) {
  for (unsigned i = 0; i < n; i += (unsigned)svcntw()) {
    svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)n);
    svfloat32_t v = svld1(pg, in + i);
    svst1(pg, out + i, svadd_f32_z(pg, v, v));
  }
}
