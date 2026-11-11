/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * C ABI of the neighbour query. Call rgpot_nlist_request_init before
 * setting cutoff. The first field of every caller-allocated struct is
 * bytes, the size the caller compiled. A later version may append
 * fields. It does not reorder the fields below and it does not change
 * the meaning of a value. A break of that rule bumps RGPOT_NLIST_ABI.
 *
 * Positions and cutoff are one length unit. The library does not
 * convert. box9 is nine doubles, lattice vectors as rows, the ASE cell
 * in row-major order: ax ay az bx by bz cx cy cz. The pair vector is
 * r_j - r_i + S H, with S the integer shift. A pair of an atom with
 * itself at shift zero is never returned.
 *
 * rgpot_nlist_build and rgpot_nlist_walk_pull may be called on different
 * objects at the same time. One walk, and the buffers of one build, are
 * used by one thread. There is no mutable global. rgpot_nlist_strerror
 * returns a literal.
 *
 * The walk borrows positions and box9 until rgpot_nlist_walk_close.
 * The caller keeps those arrays alive and does not write them from
 * another thread during a pull.
 */
#ifndef RGPOT_NLIST_H
#define RGPOT_NLIST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RGPOT_NLIST_ABI 3

enum {
  RGPOT_NLIST_OK = 0,
  RGPOT_NLIST_ERR_ABI = 1,
  RGPOT_NLIST_NULL = 2,
  RGPOT_NLIST_CUTOFF = 3,
  RGPOT_NLIST_BOX = 4,
  RGPOT_NLIST_SHAPE = 5,
  RGPOT_NLIST_SHORT = 6,
  RGPOT_NLIST_NOMEM = 7
};

/* shape: 0 pairs, 1 cluster of four. shell: 0 half, 1 full.
 * A cluster stores every neighbour, so shell must be 1. Half and
 * cluster together return RGPOT_NLIST_SHAPE. cut: 0 strict
 * (distance < cutoff), 1 loose (distance < cutoff + skin, or
 * distance <= cutoff when skin is 0). init() selects pairs, a half
 * shell, a strict cutoff, periodic boundaries and every geometry
 * field. build() stores the list. The walk streams it. There is no
 * mode that stores nothing. */
typedef struct rgpot_nlist_request {
  uint32_t bytes;
  uint32_t abi;
  double cutoff;
  double skin;
  int32_t shape;
  int32_t shell;
  int32_t cut;
  int32_t periodic[3];
  int32_t want_shifts;
  int32_t want_vectors;
  int32_t want_distances;
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

/* Sentence for a code. Unknown codes return "unknown neighbour error".
 * The pointer is a literal and is not freed. */
const char *rgpot_nlist_strerror(int32_t code);

/* Sets bytes, abi, a half pair list, a strict cutoff, periodic
 * boundaries on every axis, and shifts, vectors and distances.
 * cutoff and skin are left at 0. A null request is ignored. */
void rgpot_nlist_request_init(rgpot_nlist_request *request);

/* Stored fill. Writes at most cap pairs. *count is the number written.
 * *needed is the full number, which may exceed cap. A null out with
 * cap 0 reports the size and returns RGPOT_NLIST_OK. A buffer that is
 * shorter than the list returns RGPOT_NLIST_SHORT after writing cap
 * pairs. On any other error *count is 0 when count is not null, and
 * the pair buffer is left untouched. */
int32_t rgpot_nlist_build(const double *positions, size_t n, const double *box9,
                          const rgpot_nlist_request *request,
                          rgpot_nlist_pair *out, size_t cap, size_t *count,
                          size_t *needed);

/* Opaque chunked walk. It does not store the pairs of the whole system. */
typedef struct rgpot_nlist_walk rgpot_nlist_walk;

/* Opens a walk into *walk. On failure *walk is null when walk is not
 * null, and the return is the code from rgpot_nlist_strerror. The walk
 * borrows positions and box9. */
int32_t rgpot_nlist_walk_open(const double *positions, size_t n,
                              const double *box9,
                              const rgpot_nlist_request *request,
                              rgpot_nlist_walk **walk);

/* Writes at most cap pairs. *done is 1 when the geometry is exhausted.
 * A null out with cap greater than 0 returns RGPOT_NLIST_NULL and does
 * not advance. cap 0 writes nothing. */
int32_t rgpot_nlist_walk_pull(rgpot_nlist_walk *walk, rgpot_nlist_pair *out,
                              size_t cap, size_t *count, int32_t *done);

void rgpot_nlist_walk_close(rgpot_nlist_walk *walk);

#ifdef __cplusplus
}
#endif

#endif
