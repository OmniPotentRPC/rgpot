/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * A pure C pair sum. Half list, distances only.
 */

#include "rgpot/nlist/rgpot_nlist.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  const double pos[] = {0, 0, 0, 1, 0, 0};
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  rgpot_nlist_request request;
  rgpot_nlist_request_pairs(&request, 2.0, 0);
  request.want_shifts = 0;
  request.want_vectors = 0;
  size_t needed = 0;
  if (rgpot_nlist_build(pos, 2, box, &request, NULL, NULL, &needed) != RGPOT_NLIST_OK ||
      needed != 1) {
    fprintf(stderr, "expected one pair\n");
    return 1;
  }
  double *distance = (double *)malloc(needed * sizeof(double));
  int32_t *first = (int32_t *)malloc(needed * sizeof(int32_t));
  int32_t *second = (int32_t *)malloc(needed * sizeof(int32_t));
  rgpot_nlist_soa soa;
  rgpot_nlist_soa_init(&soa);
  soa.first = first;
  soa.second = second;
  soa.distance = distance;
  soa.cap = needed;
  size_t count = 0;
  if (rgpot_nlist_build(pos, 2, box, &request, &soa, &count, &needed) != RGPOT_NLIST_OK) {
    fprintf(stderr, "fill failed\n");
    return 1;
  }
  const double energy = distance[0] * distance[0];
  if (fabs(energy - 1.0) > 1e-12) {
    fprintf(stderr, "energy %.16g\n", energy);
    return 1;
  }
  free(distance);
  free(first);
  free(second);
  return 0;
}
