/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * A Verlet skin in C. The builder keeps the list while the atom moves
 * by at most half the skin, and rebuilds past that. A strict cut with
 * a positive skin is rejected.
 */

#include "rgpot/nlist/rgpot_nlist.h"

#include <stdio.h>
#include <string.h>

static int expect(int rc, int want, const char *what) {
  if (rc != want) {
    fprintf(stderr, "%s: %s\n", what, rgpot_nlist_strerror(rc));
    return 1;
  }
  return 0;
}

int main(void) {
  double pos[6] = {0, 0, 0, 1.2, 0, 0};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  rgpot_nlist_request request;
  rgpot_nlist_request_init(&request);
  request.cutoff = 2.5;
  request.skin = 0.4;
  request.cut = 0;
  request.want_shifts = 0;
  request.want_vectors = 0;
  size_t needed = 0;
  int rc = rgpot_nlist_build(pos, 2, box, &request, NULL, NULL, &needed);
  if (expect(rc, RGPOT_NLIST_SKIN, "strict skin") != 0) {
    return 1;
  }

  request.cut = 1;
  rgpot_nlist_builder *builder = NULL;
  if (expect(rgpot_nlist_builder_open(&builder), RGPOT_NLIST_OK, "open") != 0) {
    return 1;
  }
  int32_t first[4], second[4];
  double distance[4];
  uint8_t mask[4];
  rgpot_nlist_soa soa;
  memset(&soa, 0, sizeof soa);
  soa.bytes = (uint32_t)sizeof soa;
  soa.first = first;
  soa.second = second;
  soa.distance = distance;
  soa.mask = mask;
  soa.cap = 4;
  size_t count = 0;
  int32_t reused = 1;
  rc = rgpot_nlist_builder_update(builder, pos, 2, box, &request, &soa, &count, &needed,
                                  &reused);
  if (expect(rc, RGPOT_NLIST_OK, "first") != 0 || count != 1 || reused != 0) {
    fprintf(stderr, "first count %zu reused %d\n", count, reused);
    rgpot_nlist_builder_close(builder);
    return 1;
  }
  pos[0] = 0.1;
  rc = rgpot_nlist_builder_update(builder, pos, 2, box, &request, &soa, &count, &needed,
                                  &reused);
  if (expect(rc, RGPOT_NLIST_OK, "inside skin") != 0 || reused != 1) {
    fprintf(stderr, "inside reused %d\n", reused);
    rgpot_nlist_builder_close(builder);
    return 1;
  }
  pos[0] = 0.3;
  rc = rgpot_nlist_builder_update(builder, pos, 2, box, &request, &soa, &count, &needed,
                                  &reused);
  if (expect(rc, RGPOT_NLIST_OK, "past skin") != 0 || reused != 0 || count != 1) {
    fprintf(stderr, "past reused %d count %zu\n", reused, count);
    rgpot_nlist_builder_close(builder);
    return 1;
  }
  rgpot_nlist_builder_close(builder);
  puts("verlet-ok");
  return 0;
}
