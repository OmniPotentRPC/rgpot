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
 * used by one thread. build may start an OpenMP team when OMP_NUM_THREADS
 * is set and the system has at least 8192 atoms. A smaller system stays
 * on the calling thread. Do not call build from inside an OpenMP region.
 * There is no mutable global. rgpot_nlist_strerror returns a literal.
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

#define RGPOT_NLIST_ABI 4

/* shell and the Cap'n Proto NeighborRequest.format field. */
#define RGPOT_NLIST_FORMAT_HALF 0
#define RGPOT_NLIST_FORMAT_FULL 1
#define RGPOT_NLIST_FORMAT_CLUSTER 2

enum {
  RGPOT_NLIST_OK = 0,
  RGPOT_NLIST_ERR_ABI = 1,
  RGPOT_NLIST_NULL = 2,
  RGPOT_NLIST_CUTOFF = 3,
  RGPOT_NLIST_BOX = 4,
  RGPOT_NLIST_SHAPE = 5,
  RGPOT_NLIST_SHORT = 6,
  RGPOT_NLIST_NOMEM = 7,
  RGPOT_NLIST_SKIN = 8
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
  /* Appended. A caller whose bytes stop at want_distances is a scalar
   * cutoff. types has n entries. pair_cutoff is row-major ntypes*ntypes.
   * ntypes 0 ignores the pointers. A non-positive matrix entry drops
   * that pair type. */
  const int32_t *types;
  const double *pair_cutoff;
  int32_t ntypes;
} rgpot_nlist_request;

/* Caller-owned structure of arrays, the same columns as NeighborList
 * in Potentials.capnp and rgpot::nlist::View. first and second hold
 * cap pairs. shift and vector hold 3 * cap values when the pointers
 * are not null. distance and mask hold cap values. A null pointer
 * skips that column. A cluster list requires mask. bytes is
 * sizeof(rgpot_nlist_soa) as the caller compiled it. */
typedef struct rgpot_nlist_soa {
  uint32_t bytes;
  int32_t *first;
  int32_t *second;
  int32_t *shift;
  double *vector;
  double *distance;
  uint8_t *mask;
  size_t cap;
} rgpot_nlist_soa;

/* Returns RGPOT_NLIST_ABI. */
int32_t rgpot_nlist_abi_version(void);

/* Sentence for a code. Unknown codes return "unknown neighbour error".
 * The pointer is a literal and is not freed. */
const char *rgpot_nlist_strerror(int32_t code);

/* Sets bytes, abi, a half pair list, a strict cutoff, periodic
 * boundaries on every axis, and shifts, vectors and distances.
 * cutoff and skin are left at 0. A null request is ignored. */
void rgpot_nlist_request_init(rgpot_nlist_request *request);

/* init, then a pair list at cutoff. full is 0 for half and 1 for both
 * directions. Shifts, vectors and distances stay on. */
void rgpot_nlist_request_pairs(rgpot_nlist_request *request, double cutoff, int full);

/* Zero a caller-owned column struct and set bytes. */
void rgpot_nlist_soa_init(rgpot_nlist_soa *soa);

/* Stored fill into caller arrays. *count is the number written.
 * *needed is the full number, which may exceed out->cap. A null out,
 * or a buffer with cap 0, reports the size and returns RGPOT_NLIST_OK.
 * A buffer that is shorter than the list returns RGPOT_NLIST_SHORT
 * after writing cap pairs. A requested column whose pointer is null
 * returns RGPOT_NLIST_NULL and writes nothing. On any other error
 * *count is 0 when count is not null. */
int32_t rgpot_nlist_build(const double *positions, size_t n, const double *box9,
                          const rgpot_nlist_request *request, rgpot_nlist_soa *out,
                          size_t *count, size_t *needed);

/* Opaque chunked walk. It does not store the pairs of the whole system. */
typedef struct rgpot_nlist_walk rgpot_nlist_walk;

/* Opens a walk into *walk. On failure *walk is null when walk is not
 * null, and the return is the code from rgpot_nlist_strerror. The walk
 * borrows positions and box9. */
int32_t rgpot_nlist_walk_open(const double *positions, size_t n,
                              const double *box9,
                              const rgpot_nlist_request *request,
                              rgpot_nlist_walk **walk);

/* Writes at most out->cap pairs straight into the caller arrays.
 * *done is 1 when the geometry is exhausted. A null out, or a column
 * the request asked for whose pointer is null while cap > 0, returns
 * RGPOT_NLIST_NULL and does not advance. cap 0 writes nothing. */
int32_t rgpot_nlist_walk_pull(rgpot_nlist_walk *walk, rgpot_nlist_soa *out,
                              size_t *count, int32_t *done);

void rgpot_nlist_walk_close(rgpot_nlist_walk *walk);

/* One pair at a time. The vector is r_j - r_i + S H. The callback runs
 * on the calling thread. It does not store the list. A null fn returns
 * RGPOT_NLIST_NULL. */
typedef void (*rgpot_nlist_each_fn)(void *user, int32_t first, int32_t second, int32_t shift_a,
                                    int32_t shift_b, int32_t shift_c, double x, double y, double z,
                                    double distance);

int32_t rgpot_nlist_each(const double *positions, size_t n, const double *box9,
                         const rgpot_nlist_request *request, rgpot_nlist_each_fn fn, void *user);

/* Reusable geometry callback. shell 0 is half, 1 is full. threads 1
 * runs on the calling thread. threads above 1 may call fn at the same
 * time; the thread argument is the OpenMP thread index. A positive skin
 * needs a loose cut. *reused is 1 when the bins, or the candidate index
 * on a wide orthorhombic cell, were kept. The request's shape is ignored:
 * the shell argument chooses the list. A cluster request returns
 * RGPOT_NLIST_SHAPE. */
typedef struct rgpot_nlist_sweep rgpot_nlist_sweep;

typedef void (*rgpot_nlist_each_team_fn)(void *user, int32_t thread, int32_t first, int32_t second,
                                        int32_t shift_a, int32_t shift_b, int32_t shift_c,
                                        double x, double y, double z, double distance);

int32_t rgpot_nlist_sweep_open(rgpot_nlist_sweep **sweep);

int32_t rgpot_nlist_sweep_each(rgpot_nlist_sweep *sweep, const double *positions, size_t n,
                              const double *box9, const rgpot_nlist_request *request,
                              rgpot_nlist_each_team_fn fn, void *user, int32_t threads,
                              int32_t shell, int32_t *reused);

void rgpot_nlist_sweep_close(rgpot_nlist_sweep *sweep);

/* Several systems that do not interact. positions is the concatenation
 * of counts[s] atoms. box9 holds 9 * nsystems lattice values. first and
 * second in out index the concatenated positions. system, when not
 * null, receives the system id of each written pair and must hold cap
 * ints. A null system pointer skips that column. The size probe and
 * the short-buffer rule match rgpot_nlist_build. */
int32_t rgpot_nlist_build_systems(const double *positions, const int32_t *counts,
                                  size_t nsystems, const double *box9,
                                  const rgpot_nlist_request *request,
                                  rgpot_nlist_soa *out, int32_t *system,
                                  size_t *count, size_t *needed);

/* A builder keeps the last list and skips the search while every atom
 * has moved by at most half the skin. Skin 0 rebuilds unless the
 * coordinates match. open returns a code and writes *builder. update
 * fills out the same way rgpot_nlist_build does. *reused is 1 when the
 * stored list was kept. close frees the builder. */
typedef struct rgpot_nlist_builder rgpot_nlist_builder;

int32_t rgpot_nlist_builder_open(rgpot_nlist_builder **builder);

int32_t rgpot_nlist_builder_update(rgpot_nlist_builder *builder,
                                   const double *positions, size_t n,
                                   const double *box9,
                                   const rgpot_nlist_request *request,
                                   rgpot_nlist_soa *out, size_t *count,
                                   size_t *needed, int32_t *reused);

void rgpot_nlist_builder_close(rgpot_nlist_builder *builder);

/* Fortran gather bridge. One list keeps a builder. The search runs at
 * cutoff + skin and is reused while every atom stays within half the
 * skin. Each call writes the pairs that are strictly inside `cutoff`
 * for the positions passed in, with vectors in Fortran (3, npairs)
 * order: x, y, z of pair 0, then pair 1. Indices are zero-based.
 * A null `first` only sets *count. `periodic` is three ints, 0 or 1. */
typedef struct rgpot_nlist_fortran_list rgpot_nlist_fortran_list;

rgpot_nlist_fortran_list *rgpot_nlist_fortran_open(void);

void rgpot_nlist_fortran_close(rgpot_nlist_fortran_list *list);

int32_t rgpot_nlist_fortran_build(rgpot_nlist_fortran_list *list,
                                  const double *positions, size_t natoms,
                                  const double *box9, double cutoff, double skin,
                                  const int32_t periodic[3], int32_t *first,
                                  int32_t *second, double *vector, double *distance,
                                  size_t cap, size_t *count, int32_t *reused);

#ifdef __cplusplus
}
#endif

#endif
