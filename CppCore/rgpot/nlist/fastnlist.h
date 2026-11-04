#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// C ABI of the linkcell / minimage static library. Orthorhombic and
// general cells both go through that library. Callers do not keep a
// scalar scan beside these entry points.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Unique unordered pairs `(i, j)` with `i < j`. Every pair with an image
/// inside `list_cutoff` is present. On a cell only a few cutoffs wide the
/// buffer is every unordered pair and the caller drops the rest. Self
/// images are never included.
///
/// `cell9` is nine doubles, lattice vectors a, b, c packed as
/// ax, ay, az, bx, by, bz, cx, cy, cz. `periodic` is three ints, nonzero
/// when that axis folds. An open axis is padded inside the call so a
/// periodic image of it cannot fall inside the cutoff.
///
/// `out_ij` holds `cap` pairs (two ints each). A short buffer returns 1,
/// writes the needed pair count to `*out_count`, and leaves `out_ij`
/// untouched. Success returns 0. A hard failure returns a negative code;
/// `rgpot_fast_last_error` then points at a thread-local message.
int rgpot_fast_pairs(const double *xyz, size_t n, const double *cell9,
                     const int32_t *periodic, double list_cutoff,
                     int32_t *out_ij, size_t cap, size_t *out_count);

/// Minimum-image wrap of `n` packed difference vectors. `cell9` is the
/// same layout as `rgpot_fast_pairs`. Every axis folds.
int rgpot_fast_wrap(const double *cell9, const double *diffs, size_t n,
                    double *out);

/// Same wrap, but an axis with `periodic[k] == 0` is left unchanged.
/// The cell must be orthorhombic. Periodic widths must be positive.
int rgpot_fast_wrap_axes(const double *cell9, const int32_t *periodic,
                         const double *diffs, size_t n, double *out);

/// Thread-local message for the most recent negative return on this
/// thread, or NULL when the last call succeeded. Do not free it.
const char *rgpot_fast_last_error(void);

#ifdef __cplusplus
}
#endif
