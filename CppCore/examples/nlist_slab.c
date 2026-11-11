/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * A slab: x and y are periodic, z is open. The neighbour across the x
 * boundary is kept. The pair that would cross z is not.
 */

#include "rgpot/nlist/rgpot_nlist.h"

#include <stdio.h>

struct Tally {
  int wrap_x;
  int wrap_z;
};

static void on_pair(void *user, int32_t thread, int32_t first, int32_t second, int32_t shift_a,
                    int32_t shift_b, int32_t shift_c, double x, double y, double z, double distance) {
  struct Tally *tally = (struct Tally *)user;
  (void)thread;
  (void)first;
  (void)second;
  (void)x;
  (void)y;
  (void)z;
  (void)distance;
  if (shift_c != 0) {
    tally->wrap_z += 1;
  }
  if (shift_a != 0 && shift_b == 0 && shift_c == 0) {
    tally->wrap_x += 1;
  }
}

int main(void) {
  /* 0.2 and 9.6 are 0.6 apart through the x boundary. z = 0.2 and 9.7
   * would be 0.5 apart only if z were periodic. */
  const double pos[] = {0.2, 5, 0.2, 9.6, 5, 0.2, 5, 5, 9.7};
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  rgpot_nlist_request request;
  rgpot_nlist_request_init(&request);
  request.cutoff = 1.0;
  request.periodic[0] = 1;
  request.periodic[1] = 1;
  request.periodic[2] = 0;
  rgpot_nlist_sweep *sweep = NULL;
  if (rgpot_nlist_sweep_open(&sweep) != RGPOT_NLIST_OK) {
    fprintf(stderr, "open failed\n");
    return 1;
  }
  struct Tally tally = {0, 0};
  int32_t reused = 1;
  int32_t rc = rgpot_nlist_sweep_each(sweep, pos, 3, box, &request, on_pair, &tally, 1, 1, &reused);
  if (rc != RGPOT_NLIST_OK || reused != 0 || tally.wrap_x != 2 || tally.wrap_z != 0) {
    fprintf(stderr, "slab: %s x %d z %d reused %d\n", rgpot_nlist_strerror(rc), tally.wrap_x,
            tally.wrap_z, reused);
    rgpot_nlist_sweep_close(sweep);
    return 1;
  }
  tally.wrap_x = 0;
  rc = rgpot_nlist_sweep_each(sweep, pos, 3, box, &request, on_pair, &tally, 1, 1, &reused);
  if (rc != RGPOT_NLIST_OK || reused != 1 || tally.wrap_x != 2) {
    fprintf(stderr, "repeat: %s reused %d x %d\n", rgpot_nlist_strerror(rc), reused, tally.wrap_x);
    rgpot_nlist_sweep_close(sweep);
    return 1;
  }
  rgpot_nlist_sweep_close(sweep);
  return 0;
}
