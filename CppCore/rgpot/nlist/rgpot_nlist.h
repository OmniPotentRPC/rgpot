/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * C ABI of the neighbour query, version 1. Callers pass
 * rgpot_nlist_abi_version() and refuse a mismatch. A later version adds
 * fields at the end of the request and pair structs; it does not reorder
 * the fields below.
 */
#ifndef RGPOT_NLIST_H
#define RGPOT_NLIST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RGPOT_NLIST_ABI 1

/* format: 0 half, 1 full, 2 cluster of four. cut: 0 strict, 1 loose.
 * mode: 0 auto, 1 stored, 2 stream. */
typedef struct rgpot_nlist_request {
  int32_t abi;
  double cutoff;
  double skin;
  int32_t format;
  int32_t cut;
  int32_t periodic[3];
  int32_t want_shifts;
  int32_t want_vectors;
  int32_t want_distances;
  int32_t mode;
} rgpot_nlist_request;

/* mask is 0xff for half and full. For a cluster of four, bit k is atom
 * first+k. vector and distance describe the lowest set bit. */
typedef struct rgpot_nlist_pair {
  int32_t first;
  int32_t second;
  int32_t shift[3];
  double vector[3];
  double distance;
  uint8_t mask;
} rgpot_nlist_pair;

/* Returns RGPOT_NLIST_ABI. */
int32_t rgpot_nlist_abi_version(void);

/* Stored fill. Writes at most cap pairs. *count is the number written.
 * *needed is the full number, which may exceed cap (the caller allocates
 * again). Returns 0, or 1 if the request abi does not match. */
int32_t rgpot_nlist_build(const double *positions, size_t n, const double *box9,
                          const rgpot_nlist_request *request,
                          rgpot_nlist_pair *out, size_t cap, size_t *count,
                          size_t *needed);

/* Opaque chunked walk. It does not store the pairs of the whole system. */
typedef struct rgpot_nlist_walk rgpot_nlist_walk;

/* Returns null if the request abi does not match. */
rgpot_nlist_walk *rgpot_nlist_walk_open(const double *positions, size_t n,
                                        const double *box9,
                                        const rgpot_nlist_request *request);

/* Writes at most cap pairs. *done is 1 when the geometry is exhausted. */
int32_t rgpot_nlist_walk_pull(rgpot_nlist_walk *walk, rgpot_nlist_pair *out,
                              size_t cap, size_t *count, int32_t *done);

void rgpot_nlist_walk_close(rgpot_nlist_walk *walk);

#ifdef __cplusplus
}
#endif

#endif
