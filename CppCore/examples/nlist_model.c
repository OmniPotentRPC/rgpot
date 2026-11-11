/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * External model adapter over the C ABI. It asks for a full pair list
 * inside one cutoff, with cell shifts, and copies the five label columns
 * a graph model reads. The model is not named: the columns are the
 * neighbour query's own.
 */

#include "rgpot/nlist/rgpot_nlist.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Labels {
  int32_t *first;
  int32_t *second;
  int32_t *shift;
  size_t count;
};

static int fill_labels(const double *positions, size_t n, const double *box, double cutoff,
                       struct Labels *out) {
  rgpot_nlist_request request;
  rgpot_nlist_request_pairs(&request, cutoff, 1);
  request.want_vectors = 0;
  request.want_distances = 0;
  size_t needed = 0;
  int32_t rc = rgpot_nlist_build(positions, n, box, &request, NULL, NULL, &needed);
  if (rc != RGPOT_NLIST_OK) {
    return rc;
  }
  out->first = (int32_t *)malloc(needed * sizeof(int32_t));
  out->second = (int32_t *)malloc(needed * sizeof(int32_t));
  out->shift = (int32_t *)malloc(needed * 3 * sizeof(int32_t));
  if (out->first == NULL || out->second == NULL || out->shift == NULL) {
    return RGPOT_NLIST_NOMEM;
  }
  rgpot_nlist_soa soa;
  rgpot_nlist_soa_init(&soa);
  soa.first = out->first;
  soa.second = out->second;
  soa.shift = out->shift;
  soa.cap = needed;
  size_t count = 0;
  rc = rgpot_nlist_build(positions, n, box, &request, &soa, &count, &needed);
  out->count = count;
  return rc;
}

int main(void) {
  const double pos[] = {0.2, 0.2, 0.2, 1.4, 0.2, 0.2};
  const double box[9] = {6, 0, 0, 0, 6, 0, 0, 0, 6};
  struct Labels labels = {0, 0, 0, 0};
  int rc = fill_labels(pos, 2, box, 2.0, &labels);
  int saw = 0;
  size_t i;
  if (rc != RGPOT_NLIST_OK) {
    fprintf(stderr, "adapter: %s\n", rgpot_nlist_strerror(rc));
    return 1;
  }
  for (i = 0; i < labels.count; i++) {
    if (labels.first[i] == 0 && labels.second[i] == 1 && labels.shift[3 * i] == 0 &&
        labels.shift[3 * i + 1] == 0 && labels.shift[3 * i + 2] == 0) {
      saw = 1;
    }
  }
  free(labels.first);
  free(labels.second);
  free(labels.shift);
  if (!saw) {
    fprintf(stderr, "missing the direct pair\n");
    return 1;
  }
  return 0;
}
