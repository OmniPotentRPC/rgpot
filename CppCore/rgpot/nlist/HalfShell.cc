// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/HalfShell.hpp"

#include "rgpot/nlist/HalfShellDetail.hpp"

#include <algorithm>
#include <cstring>

namespace rgpot::nlist::detail {
Key legacyKey(int lo, int hi, const Scratch &s, const Grid &g) {
  const int clo = s.cell_of[static_cast<std::size_t>(lo)];
  const int chi = s.cell_of[static_cast<std::size_t>(hi)];
  const int iz = clo % g.nz;
  const int iy = (clo / g.nz) % g.ny;
  const int ix = clo / (g.ny * g.nz);
  const int jz = chi % g.nz;
  const int jy = (chi / g.nz) % g.ny;
  const int jx = chi / (g.ny * g.nz);
  int ox = jx - ix;
  int oy = jy - iy;
  int oz = jz - iz;
  if (ox > g.nx / 2) {
    ox -= g.nx;
  } else if (ox < -(g.nx / 2)) {
    ox += g.nx;
  }
  if (oy > g.ny / 2) {
    oy -= g.ny;
  } else if (oy < -(g.ny / 2)) {
    oy += g.ny;
  }
  if (oz > g.nz / 2) {
    oz -= g.nz;
  } else if (oz < -(g.nz / 2)) {
    oz += g.nz;
  }
  // Loop nest of the 27-walk: cell, then stencil slot, then source atom,
  // then partner. oz is the fastest stencil index.
  Key key;
  key.cell = clo;
  key.slot = (ox + 1) * 9 + (oy + 1) * 3 + (oz + 1);
  key.rank_lo = s.rank[static_cast<std::size_t>(lo)];
  key.rank_hi = s.rank[static_cast<std::size_t>(hi)];
  return key;
}

bool keyLess(const Key &a, const Key &b) {
  if (a.cell != b.cell) {
    return a.cell < b.cell;
  }
  if (a.slot != b.slot) {
    return a.slot < b.slot;
  }
  if (a.rank_lo != b.rank_lo) {
    return a.rank_lo < b.rank_lo;
  }
  return a.rank_hi < b.rank_hi;
}

void sortLegacy(Scratch &s) {
  if (!s.ordered) {
    return;
  }
  const Grid &order = s.order;
  const std::size_t np = s.lo.size();
  s.key.resize(np);
  for (std::size_t p = 0; p < np; ++p) {
    s.key[p] = legacyKey(s.lo[p], s.hi[p], s, order);
  }
  s.ord.resize(np);
  for (std::size_t p = 0; p < np; ++p) {
    s.ord[p] = static_cast<std::uint32_t>(p);
  }
  std::sort(s.ord.begin(), s.ord.end(), [&](std::uint32_t a, std::uint32_t b) {
    return keyLess(s.key[a], s.key[b]);
  });
  s.lo2.resize(np);
  s.hi2.resize(np);
  s.dx2.resize(np);
  s.dy2.resize(np);
  s.dz2.resize(np);
  for (std::size_t p = 0; p < np; ++p) {
    const std::size_t q = s.ord[p];
    s.lo2[p] = s.lo[q];
    s.hi2[p] = s.hi[q];
    s.dx2[p] = s.dx[q];
    s.dy2[p] = s.dy[q];
    s.dz2[p] = s.dz[q];
  }
  s.lo.swap(s.lo2);
  s.hi.swap(s.hi2);
  s.dx.swap(s.dx2);
  s.dy.swap(s.dy2);
  s.dz.swap(s.dz2);
}

} // namespace rgpot::nlist::detail

namespace rgpot::nlist {
namespace {
enum class Hit { Miss, Same, Skin };

Hit reuse(const NeighborList &cache, const double *pos, const double *box,
          std::size_t n, double cutoff) {
  if (!cache.built || cache.n != n || cache.cutoff != cutoff ||
      cache.skin_used != cache.skin) {
    return Hit::Miss;
  }
  // Bitwise, like the positions: a box that differs only in the sign of a
  // zero still gets its own build.
  if (std::memcmp(cache.box, box, sizeof(cache.box)) != 0) {
    return Hit::Miss;
  }
  if (std::memcmp(cache.ref.data(), pos, n * 3 * sizeof(double)) == 0) {
    return Hit::Same;
  }
  if (!(cache.skin > 0.0)) {
    return Hit::Miss;
  }
  const double lim = 0.5 * cache.skin;
  const double lim2 = lim * lim;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = pos[3 * i] - cache.ref[3 * i];
    const double dy = pos[3 * i + 1] - cache.ref[3 * i + 1];
    const double dz = pos[3 * i + 2] - cache.ref[3 * i + 2];
    if (dx * dx + dy * dy + dz * dz > lim2) {
      return Hit::Miss;
    }
  }
  return Hit::Skin;
}
} // namespace

const NeighborList &orderedPairs(const double *positions, const double *box,
                                 std::size_t n, double cutoff,
                                 NeighborList *cache) {
  thread_local NeighborList owned;
  thread_local detail::Scratch scratch;
  NeighborList &dst = cache != nullptr ? *cache : owned;
  if (cache != nullptr) {
    const Hit hit = reuse(*cache, positions, box, n, cutoff);
    if (hit == Hit::Same) {
      dst.use_stored = true;
      return dst;
    }
    if (hit == Hit::Skin) {
      dst.use_stored = false;
      return dst;
    }
  }
  const double skin = cache != nullptr && cache->skin > 0.0 ? cache->skin : 0.0;
  const double search = cutoff + skin;
  detail::fillPairs(positions, box, n, cutoff, search, scratch);
  detail::sortLegacy(scratch);
  dst.lo.swap(scratch.lo);
  dst.hi.swap(scratch.hi);
  dst.dx.swap(scratch.dx);
  dst.dy.swap(scratch.dy);
  dst.dz.swap(scratch.dz);
  dst.use_stored = true;
  dst.n = n;
  dst.cutoff = cutoff;
  dst.skin_used = skin;
  if (cache != nullptr) {
    dst.ref.assign(positions, positions + n * 3);
    std::memcpy(dst.box, box, sizeof(dst.box));
    dst.built = true;
  }
  return dst;
}

} // namespace rgpot::nlist
