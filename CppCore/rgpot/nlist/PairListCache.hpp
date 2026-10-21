#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Verlet-skin cached pair lists for classical pair potentials.
//
// An orthorhombic cell gets its candidates from the fast list. A cell
// wide enough to split into several bins uses linkcell (`pairs_within`,
// half list, self images dropped). A cell only a few cutoffs wide, and a
// small open cluster, is handed back as every unordered pair; the caller
// keeps the in-cutoff subset. Periodic axes fold with minimage
// (`wrap_many`, AVX where that crate built it). An open axis is padded
// inside the pair search so its periodic images stay outside the cutoff,
// then left unwrapped. A non-orthogonal cell, or a periodic axis with a
// non-positive width, stays on the fused scan in vesin_visit.hpp: those
// boxes are outside the cell the fast list accepts.
//
// Design invariants:
// - Slots are immutable after build and handed out as shared_ptr, so
//   readers never race eviction; the pool mutex covers only the
//   proximity match, never the physics.
// - Evaluation always re-derives vectors from the positions of that call. Periodic
//   axes go through one minimage wrap of the cached index pairs, filtered
//   at the true cutoff. Pair content matches a fresh scan while the
//   displacement vectors themselves stay inside a box whose diagonal is
//   shorter than the skin. A rigid translation has a zero diagonal, so
//   the list is kept.
// - The first sighting captures the list and evaluates it. Later calls
//   inside the skin walk the stored pairs.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef RGPOT_HAS_FAST_NLIST
#include "rgpot/nlist/fastnlist.h"
#endif
#include "rgpot/nlist/vesin_visit.hpp"

namespace rgpot {
namespace nlist {

namespace {

struct FastScratch {
  std::vector<int32_t> ij;
  std::vector<double> diff;
  std::vector<double> wrapped;
  int32_t periodic[3]{};
};

inline FastScratch &fast_scratch() {
  thread_local FastScratch scratch;
  return scratch;
}

#ifdef RGPOT_HAS_FAST_NLIST
[[noreturn]] inline void fast_fail(const char *what) {
  const char *msg = rgpot_fast_last_error();
  throw std::runtime_error(std::string(what) + (msg != nullptr ? msg : ""));
}
#endif

} // namespace

class CachedPairList {
public:
  struct Options {
    double cutoff{0.0};
    double skin{1.0};
    std::array<bool, 3> periodic{{true, true, true}};

    bool operator==(const Options &o) const {
      return cutoff == o.cutoff && skin == o.skin && periodic == o.periodic;
    }
  };

  /// True while the cached pair set is valid for positions ``R``: same
  /// atom count, options and box as the build, and the cloud of
  /// displacement vectors has a bounding-box diagonal shorter than the
  /// skin. That is the Verlet condition on relative motion. A shared
  /// translation drops out of the diagonal.
  [[nodiscard]] bool valid(const double *R, std::size_t n, const double *box,
                           const Options &opt) const {
    if (!built_ || n != n_ || !(opt == opt_)) {
      return false;
    }
    for (int k = 0; k < 9; ++k) {
      if (box[k] != boxref_[k]) {
        return false;
      }
    }
    if (complete_ || n == 0) {
      return true; // all pairs are candidates; motion cannot invalidate
    }
    const double *ref = Rref_.data();
    double minx = R[0] - ref[0];
    double maxx = minx;
    double miny = R[1] - ref[1];
    double maxy = miny;
    double minz = R[2] - ref[2];
    double maxz = minz;
    for (std::size_t a = 1; a < n; ++a) {
      const double dx = R[3 * a] - ref[3 * a];
      const double dy = R[3 * a + 1] - ref[3 * a + 1];
      const double dz = R[3 * a + 2] - ref[3 * a + 2];
      minx = std::min(minx, dx);
      maxx = std::max(maxx, dx);
      miny = std::min(miny, dy);
      maxy = std::max(maxy, dy);
      minz = std::min(minz, dz);
      maxz = std::max(maxz, dz);
    }
    const double ex = maxx - minx;
    const double ey = maxy - miny;
    const double ez = maxz - minz;
    const double diag2 = ex * ex + ey * ey + ez * ez;
    const double skin = opt_.skin;
    if (!(skin > 0.0)) {
      return diag2 == 0.0;
    }
    // A non-listed pair sits at or beyond cutoff+skin. It can reach the
    // inclusive cutoff only by shrinking by a full skin. A shorter
    // diagonal cannot do that.
    return diag2 < skin * skin;
  }

  [[nodiscard]] bool isPhantom() const { return phantom_; }

  /// First sighting. An orthorhombic cell captures the linkcell list in
  /// the same call. Anything else runs the fused scan and, when that box
  /// can be cached, a phantom stamp.
  template <typename Fn>
  void visitOnly(const double *R, std::size_t n, const double *box,
                 const Options &opt, Fn &&fn) {
    setup(n, box, opt);
    if (fast_) {
#ifdef RGPOT_HAS_FAST_NLIST
      fillFast(R, n, box, opt, fn);
      if (mic_) {
        finishRebuild(R, n, box, opt);
        phantom_ = false;
      }
#endif
      return;
    }
    double w[3];
    double inv[3];
    fold_params(box, opt, w, inv);
    vesin::cpu::brute_force_visit_only(
        R, n, w, inv, opt.cutoff * opt.cutoff,
        [&](int32_t i, int32_t j, double dx, double dy, double dz,
            double r2) { fn(i, j, -dx, -dy, -dz, r2); });
    if (mic_) {
      pairsIJ_.clear();
      finishRebuild(R, n, box, opt);
      phantom_ = true;
    }
  }

  /// Second sighting of a phantom slot: capture candidates at cutoff+skin.
  template <typename Fn>
  void rebuildFused(const double *R, std::size_t n, const double *box,
                    const Options &opt, Fn &&fn) {
    setup(n, box, opt);
    if (fast_) {
#ifdef RGPOT_HAS_FAST_NLIST
      fillFast(R, n, box, opt, fn);
      finishRebuild(R, n, box, opt);
      phantom_ = false;
#endif
      return;
    }
    double w[3];
    double inv[3];
    fold_params(box, opt, w, inv);
    const double bc = opt.cutoff + opt.skin;
    vesin::cpu::brute_force_visit(
        R, n, w, inv, bc * bc, opt.cutoff * opt.cutoff, pairsIJ_,
        [&](int32_t i, int32_t j, double dx, double dy, double dz,
            double r2) { fn(i, j, -dx, -dy, -dz, r2); });
    finishRebuild(R, n, box, opt);
  }

  /// True when caching applies to this box (decided by the last build).
  [[nodiscard]] bool cacheable() const { return mic_; }

  /// Visit every cached pair within the true cutoff of these positions.
  /// ``fn(i, j, dx, dy, dz, r2)`` uses ``d = r_i - r_j`` with the minimum
  /// image applied.
  template <typename Fn> void forEach(const double *R, Fn &&fn) const {
    const double cutoff2 = opt_.cutoff * opt_.cutoff;
    const bool open = micInv_[0] == 0.0 && micInv_[1] == 0.0 && micInv_[2] == 0.0;
    // A complete open list is every i < j. Walking the coordinates
    // directly keeps that loop as tight as the fused scan; an index
    // vector of the same pairs only adds loads.
    if (open && complete_) {
      for (std::size_t i = 0; i < n_; ++i) {
        const double ix = R[3 * i];
        const double iy = R[3 * i + 1];
        const double iz = R[3 * i + 2];
        for (std::size_t j = i + 1; j < n_; ++j) {
          const double dx = ix - R[3 * j];
          const double dy = iy - R[3 * j + 1];
          const double dz = iz - R[3 * j + 2];
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 <= cutoff2) {
            fn(static_cast<int32_t>(i), static_cast<int32_t>(j), dx, dy, dz, r2);
          }
        }
      }
      return;
    }
    const std::size_t nint = pairsIJ_.size();
    const int32_t *ij = pairsIJ_.data();
    if (open) {
      for (std::size_t p = 0; p < nint; p += 2) {
        const int32_t i = ij[p];
        const int32_t j = ij[p + 1];
        const double dx = R[3 * i] - R[3 * j];
        const double dy = R[3 * i + 1] - R[3 * j + 1];
        const double dz = R[3 * i + 2] - R[3 * j + 2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= cutoff2) {
          fn(i, j, dx, dy, dz, r2);
        }
      }
      return;
    }
    const std::size_t np = nint / 2;
    auto &scratch = fast_scratch();
    scratch.diff.resize(np * 3);
    scratch.wrapped.resize(np * 3);
    for (std::size_t p = 0; p < np; ++p) {
      const int32_t i = ij[2 * p];
      const int32_t j = ij[2 * p + 1];
      scratch.diff[3 * p] = R[3 * i] - R[3 * j];
      scratch.diff[3 * p + 1] = R[3 * i + 1] - R[3 * j + 1];
      scratch.diff[3 * p + 2] = R[3 * i + 2] - R[3 * j + 2];
    }
    periodic_flags(opt_, scratch.periodic);
#ifdef RGPOT_HAS_FAST_NLIST
    if (np > 0 &&
        rgpot_fast_wrap_axes(boxref_.data(), scratch.periodic, scratch.diff.data(),
                             np, scratch.wrapped.data()) != 0) {
      fast_fail("rgpot minimage wrap failed: ");
    }
#else
    const double w0 = boxref_[0];
    const double w1 = boxref_[4];
    const double w2 = boxref_[8];
    const double i0 = micInv_[0];
    const double i1 = micInv_[1];
    const double i2 = micInv_[2];
    for (std::size_t p = 0; p < np; ++p) {
      double dx = scratch.diff[3 * p];
      double dy = scratch.diff[3 * p + 1];
      double dz = scratch.diff[3 * p + 2];
      dx -= w0 * vesin::cpu::visit_round(dx * i0);
      dy -= w1 * vesin::cpu::visit_round(dy * i1);
      dz -= w2 * vesin::cpu::visit_round(dz * i2);
      scratch.wrapped[3 * p] = dx;
      scratch.wrapped[3 * p + 1] = dy;
      scratch.wrapped[3 * p + 2] = dz;
    }
#endif
    for (std::size_t p = 0; p < np; ++p) {
      const double dx = scratch.wrapped[3 * p];
      const double dy = scratch.wrapped[3 * p + 1];
      const double dz = scratch.wrapped[3 * p + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= cutoff2) {
        fn(ij[2 * p], ij[2 * p + 1], dx, dy, dz, r2);
      }
    }
  }

private:
  void setup(std::size_t n, const double *box, const Options &opt) {
#ifndef RGPOT_HAS_FAST_NLIST
    (void)n;
    (void)box;
    (void)opt;
    fast_ = false;
    mic_ = false;
    return;
#else
    const bool orthorhombic = box[1] == 0.0 && box[2] == 0.0 && box[3] == 0.0 &&
                              box[5] == 0.0 && box[6] == 0.0 && box[7] == 0.0;
    fast_ = orthorhombic;
    for (int k = 0; fast_ && k < 3; ++k) {
      if (opt.periodic[static_cast<std::size_t>(k)] && !(box[4 * k] > 0.0)) {
        fast_ = false;
      }
    }
    // Above this count the candidate buffer is left uncached. The search
    // itself still goes through linkcell.
    mic_ = fast_ && n <= 20000;
#endif
  }

  static void periodic_flags(const Options &opt, int32_t out[3]) {
    for (int k = 0; k < 3; ++k) {
      out[k] = opt.periodic[static_cast<std::size_t>(k)] ? 1 : 0;
    }
  }

  static void fold_params(const double *box, const Options &opt, double w[3],
                          double inv[3]) {
    for (int k = 0; k < 3; ++k) {
      w[k] = box[4 * k];
      inv[k] = (opt.periodic[static_cast<std::size_t>(k)] && w[k] != 0.0)
                   ? 1.0 / w[k]
                   : 0.0;
    }
  }

#ifdef RGPOT_HAS_FAST_NLIST
  /// Every unordered pair, wrapped in small batches. Used when linkcell
  /// would not visit fewer images than this walk. The stored subset and
  /// the visited subset use the same inequalities as the fused scan.
  template <typename Fn>
  bool fillNarrow(const double *R, std::size_t n, const double *box,
                  const Options &opt, double list_cut, double search, Fn &&fn) {
    if (n > 4096 || n < 2) {
      return false;
    }
    double mins[3] = {R[0], R[1], R[2]};
    double maxs[3] = {mins[0], mins[1], mins[2]};
    for (std::size_t a = 1; a < n; ++a) {
      for (int k = 0; k < 3; ++k) {
        const double v = R[3 * a + static_cast<std::size_t>(k)];
        mins[k] = std::min(mins[k], v);
        maxs[k] = std::max(maxs[k], v);
      }
    }
    bool helps = true;
    for (int k = 0; k < 3; ++k) {
      if (opt.periodic[static_cast<std::size_t>(k)]) {
        if (box[4 * k] < 3.0 * search) {
          helps = false;
        }
      } else if (maxs[k] - mins[k] <= search) {
        helps = false;
      }
    }
    if (helps) {
      return false;
    }
    auto &scratch = fast_scratch();
    periodic_flags(opt, scratch.periodic);
    constexpr std::size_t batch = 128;
    scratch.ij.resize(batch * 2);
    scratch.diff.resize(batch * 3);
    scratch.wrapped.resize(batch * 3);
    const double list2 = list_cut * list_cut;
    const double cut2 = opt.cutoff * opt.cutoff;
    const bool skin0 = opt.skin == 0.0;
    const bool any_periodic = scratch.periodic[0] != 0 || scratch.periodic[1] != 0 ||
                              scratch.periodic[2] != 0;
    pairsIJ_.clear();
    std::size_t filled = 0;
    auto flush = [&]() {
      if (filled == 0) {
        return;
      }
      const double *vec = scratch.diff.data();
      if (any_periodic) {
        if (rgpot_fast_wrap_axes(box, scratch.periodic, scratch.diff.data(), filled,
                                 scratch.wrapped.data()) != 0) {
          fast_fail("rgpot minimage wrap failed: ");
        }
        vec = scratch.wrapped.data();
      }
      for (std::size_t p = 0; p < filled; ++p) {
        const double dx = vec[3 * p];
        const double dy = vec[3 * p + 1];
        const double dz = vec[3 * p + 2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < list2 || (skin0 && r2 <= cut2)) {
          pairsIJ_.push_back(scratch.ij[2 * p]);
          pairsIJ_.push_back(scratch.ij[2 * p + 1]);
          if (r2 <= cut2) {
            fn(scratch.ij[2 * p], scratch.ij[2 * p + 1], dx, dy, dz, r2);
          }
        }
      }
      filled = 0;
    };
    for (std::size_t i = 0; i < n; ++i) {
      const double ix = R[3 * i];
      const double iy = R[3 * i + 1];
      const double iz = R[3 * i + 2];
      for (std::size_t j = i + 1; j < n; ++j) {
        if (filled == batch) {
          flush();
        }
        scratch.ij[2 * filled] = static_cast<int32_t>(i);
        scratch.ij[2 * filled + 1] = static_cast<int32_t>(j);
        scratch.diff[3 * filled] = ix - R[3 * j];
        scratch.diff[3 * filled + 1] = iy - R[3 * j + 1];
        scratch.diff[3 * filled + 2] = iz - R[3 * j + 2];
        ++filled;
      }
    }
    flush();
    return true;
  }

  /// linkcell candidates at cutoff+skin, minimage vectors, true-cutoff visit.
  template <typename Fn>
  void fillFast(const double *R, std::size_t n, const double *box,
                const Options &opt, Fn &&fn) {
    const double list_cut = opt.cutoff + opt.skin;
    // linkcell keeps dist2 strictly below cutoff^2. The extra sliver
    // keeps a pair that lands on the skin boundary inside the filter
    // below; the filter then applies the same inequalities as the fused
    // scan.
    const double search =
        list_cut + 1e-8 * std::max(1.0, std::abs(list_cut));
    if (fillNarrow(R, n, box, opt, list_cut, search, fn)) {
      return;
    }
    auto &scratch = fast_scratch();
    periodic_flags(opt, scratch.periodic);
    std::size_t cap = n < 2 ? 0 : n * 256;
    const std::size_t all_pairs = n < 2 ? 0 : n * (n - 1) / 2;
    if (n <= 512) {
      cap = all_pairs;
    }
    scratch.ij.resize(std::max<std::size_t>(cap, 1) * 2);
    std::size_t count = 0;
    int rc = rgpot_fast_pairs(R, n, box, scratch.periodic, search,
                              scratch.ij.data(), cap, &count);
    if (rc == 1) {
      scratch.ij.resize(count * 2);
      rc = rgpot_fast_pairs(R, n, box, scratch.periodic, search,
                            scratch.ij.data(), count, &count);
    }
    if (rc != 0) {
      fast_fail("rgpot linkcell pair list failed: ");
    }
    const double list2 = list_cut * list_cut;
    const double cut2 = opt.cutoff * opt.cutoff;
    const bool skin0 = opt.skin == 0.0;
    const bool any_periodic = scratch.periodic[0] != 0 || scratch.periodic[1] != 0 ||
                              scratch.periodic[2] != 0;
    scratch.diff.resize(count * 3);
    scratch.wrapped.resize(count * 3);
    for (std::size_t p = 0; p < count; ++p) {
      const int32_t i = scratch.ij[2 * p];
      const int32_t j = scratch.ij[2 * p + 1];
      scratch.diff[3 * p] = R[3 * i] - R[3 * j];
      scratch.diff[3 * p + 1] = R[3 * i + 1] - R[3 * j + 1];
      scratch.diff[3 * p + 2] = R[3 * i + 2] - R[3 * j + 2];
    }
    const double *vec = scratch.diff.data();
    if (any_periodic && count > 0) {
      if (rgpot_fast_wrap_axes(box, scratch.periodic, scratch.diff.data(), count,
                               scratch.wrapped.data()) != 0) {
        fast_fail("rgpot minimage wrap failed: ");
      }
      vec = scratch.wrapped.data();
    }
    pairsIJ_.clear();
    pairsIJ_.reserve(count * 2);
    for (std::size_t p = 0; p < count; ++p) {
      const double dx = vec[3 * p];
      const double dy = vec[3 * p + 1];
      const double dz = vec[3 * p + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < list2 || (skin0 && r2 <= cut2)) {
        pairsIJ_.push_back(scratch.ij[2 * p]);
        pairsIJ_.push_back(scratch.ij[2 * p + 1]);
        if (r2 <= cut2) {
          fn(scratch.ij[2 * p], scratch.ij[2 * p + 1], dx, dy, dz, r2);
        }
      }
    }
  }
#endif

  void finishRebuild(const double *R, std::size_t n, const double *box,
                     const Options &opt) {
    for (int k = 0; k < 3; ++k) {
      micInv_[static_cast<std::size_t>(k)] =
          (mic_ && opt.periodic[static_cast<std::size_t>(k)]) ? 1.0 / box[4 * k]
                                                              : 0.0;
    }
    Rref_.assign(R, R + 3 * n);
    for (int k = 0; k < 9; ++k) {
      boxref_[k] = box[k];
    }
    opt_ = opt;
    n_ = n;
    complete_ = mic_ && pairsIJ_.size() / 2 == n * (n - 1) / 2;
    built_ = true;
  }

  std::vector<int32_t> pairsIJ_;
  std::vector<double> Rref_;
  std::array<double, 9> boxref_{};
  std::array<double, 3> micInv_{};
  Options opt_{};
  std::size_t n_{0};
  bool mic_{false};
  bool fast_{false};
  bool phantom_{false};
  bool complete_{false};
  bool built_{false};
};

/// Process-wide pool of CachedPairList slots, matched by geometry
/// proximity; safe for shared potential instances evaluated from
/// short-lived worker threads.
class PairListCache {
public:
  static constexpr std::size_t kMaxSlots = 8;

  /// Single-pass evaluation. Orthorhombic cells capture a linkcell list
  /// on the first sighting and reuse it inside the skin.
  template <typename Fn>
  void evaluate(const double *R, std::size_t n, const double *box,
                const CachedPairList::Options &opt, Fn &&fn) {
    std::shared_ptr<const CachedPairList> hit;
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (std::size_t s = 0; s < slots_.size(); ++s) {
        if (slots_[s]->valid(R, n, box, opt)) {
          if (s != 0) {
            auto slot = std::move(slots_[s]);
            slots_.erase(slots_.begin() + static_cast<std::ptrdiff_t>(s));
            slots_.insert(slots_.begin(), std::move(slot));
          }
          hit = slots_.front();
          break;
        }
      }
    }

    if (hit && !hit->isPhantom()) {
      hit->forEach(R, std::forward<Fn>(fn));
      return;
    }

    auto fresh = std::make_shared<CachedPairList>();
    if (hit) {
      fresh->rebuildFused(R, n, box, opt, fn);
    } else {
      fresh->visitOnly(R, n, box, opt, fn);
      if (!fresh->cacheable()) {
        return;
      }
    }

    std::lock_guard<std::mutex> lock(mu_);
    if (hit) {
      for (std::size_t s = 0; s < slots_.size(); ++s) {
        if (slots_[s] == hit) {
          slots_.erase(slots_.begin() + static_cast<std::ptrdiff_t>(s));
          break;
        }
      }
    }
    slots_.insert(slots_.begin(), fresh);
    if (slots_.size() > kMaxSlots) {
      slots_.pop_back();
    }
  }

  static PairListCache &global() {
    static PairListCache cache;
    return cache;
  }

private:
  std::mutex mu_;
  std::vector<std::shared_ptr<CachedPairList>> slots_;
};

} // namespace nlist
} // namespace rgpot
