/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * A C caller. No C++. The request is initialised, then the cutoff is
 * set. The result is the same columns the C++ view uses.
 */

#include "rgpot/nlist/rgpot_nlist.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

struct Seen {
  int n;
  double r;
};

static void on_pair(void *user, int32_t i, int32_t j, int32_t shift_a, int32_t shift_b,
                    int32_t shift_c, double x, double y, double z, double dist) {
  struct Seen *seen = (struct Seen *)user;
  (void)shift_a;
  (void)shift_b;
  (void)shift_c;
  (void)x;
  (void)y;
  (void)z;
  if (i == 0 && j == 1) {
    seen->n += 1;
    seen->r = dist;
  }
}

int main(void) {
  const double r = pow(2.0, 1.0 / 6.0);
  const double pos[] = {0, 0, 0, r, 0, 0};
  const double box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  rgpot_nlist_request request;
  rgpot_nlist_request_init(&request);
  request.cutoff = 2.5;
  request.want_shifts = 0;
  request.want_vectors = 0;
  size_t count = 0;
  size_t needed = 0;
  int32_t rc = rgpot_nlist_build(pos, 2, box, &request, NULL, &count, &needed);
  if (rc != RGPOT_NLIST_OK || needed != 1) {
    fprintf(stderr, "probe: %s needed %zu\n", rgpot_nlist_strerror(rc), needed);
    return 1;
  }
  int32_t first = -1;
  int32_t second = -1;
  double distance = 0;
  rgpot_nlist_soa soa;
  soa.bytes = (uint32_t)sizeof soa;
  soa.first = &first;
  soa.second = &second;
  soa.shift = NULL;
  soa.vector = NULL;
  soa.distance = &distance;
  soa.mask = NULL;
  soa.cap = 1;
  rc = rgpot_nlist_build(pos, 2, box, &request, &soa, &count, &needed);
  if (rc != RGPOT_NLIST_OK || count != 1 || first != 0 || second != 1) {
    fprintf(stderr, "fill: %s\n", rgpot_nlist_strerror(rc));
    return 1;
  }
  if (fabs(distance - r) > 1e-12) {
    fprintf(stderr, "distance %.16g\n", distance);
    return 1;
  }
  struct Seen seen = {0, 0};
  rc = rgpot_nlist_each(pos, 2, box, &request, on_pair, &seen);
  if (rc != RGPOT_NLIST_OK || seen.n != 1 || fabs(seen.r - r) > 1e-12) {
    fprintf(stderr, "each: %s n %d\n", rgpot_nlist_strerror(rc), seen.n);
    return 1;
  }
  return 0;
}
