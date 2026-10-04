#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Verlet-skin cached pair lists for classical pair potentials, ported from
// eOn's eonc::PairListCache (TheochemUI/eOn, fix/vesin-neighbor-perf).
//
// Header-only and dependency-free: the MIC regime (orthorhombic box, true
// cutoff within half the smallest periodic width) searches pairs with the
// linked-cell scan from cell_visit.hpp once the box holds enough cells
// (O(n) per search), and with the fused brute-force scan from
// vesin_visit.hpp below that (O(n^2), cheaper for a handful of cells).
// Both report the same pair set. Outside the cached regime every call
// performs a brute-force scan. Nonorthogonal cells use the full lattice
// minimum image; orthorhombic cells retain the componentwise fold.
//
// Design invariants (see the eOn failure analysis for the derivation):
// - Slots are immutable after build and handed out as shared_ptr, so
//   readers never race eviction; the pool mutex covers only the
//   proximity match, never the physics.
// - Evaluation always re-derives vectors from current positions, folded
//   by a per-call MIC rounding or by the periodic image recorded at build
//   (when cutoff + skin < w/2 makes the two agree), filtered at the true
//   cutoff: the pair content matches a fresh scan while every atom stays
//   within skin/2 of the build positions (Verlet guarantee).
// - The cached list is a CSR half list (partners j > i per atom, sorted),
//   so the force loop keeps atom i's force in registers across its row.
// - Lazy capture: the first sighting of a geometry family runs the fused
//   eval-only scan and records a phantom reference stamp; a second
//   sighting within skin/2 proves reuse and captures the list. One-shot
//   evaluations never pay list capture.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "rgpot/nlist/MinimumImage.hpp"
#include "rgpot/nlist/cell_visit.hpp"
#include "rgpot/nlist/vesin_visit.hpp"

namespace rgpot {
namespace nlist {

/// One pair's contribution, as a radial kernel returns it: the energy and
/// ``fscale = -V'(r) / r``, so the force on atom i is ``fscale * (r_i - r_j)``.
struct PairTerm {
  double energy;
  double fscale;
};

struct IgnorePairContribution {
  void operator()(double, double, double, double) const noexcept {}
};

class CachedPairList {
public:
  /// How evaluation turns r_i - r_j into the minimum image: no fold (free
  /// boundaries), a per-call rounding fold, or a shift recorded at build.
  enum class FoldMode : uint8_t { None, Round, Coded, General };

  struct Options {
    double cutoff{0.0};
    double skin{1.0};
    std::array<bool, 3> periodic{{true, true, true}};

    bool operator==(const Options &o) const {
      return cutoff == o.cutoff && skin == o.skin && periodic == o.periodic;
    }
  };

  /// True while the cached pair set is valid for positions ``R``: same
  /// atom count, options and box as the build, and max displacement
  /// below skin/2.
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
    if (complete_) {
      return true; // all pairs are candidates; motion cannot invalidate
    }
    const double thr2 = 0.25 * opt_.skin * opt_.skin;
    const double *ref = Rref_.data();
    for (std::size_t a = 0; a < n; ++a) {
      const double dx = R[3 * a] - ref[3 * a];
      const double dy = R[3 * a + 1] - ref[3 * a + 1];
      const double dz = R[3 * a + 2] - ref[3 * a + 2];
      if (dx * dx + dy * dy + dz * dz > thr2) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool isPhantom() const { return phantom_; }

  /// First sighting: fused eval-only scan plus a phantom stamp (MIC
  /// regime), or just the scan when caching is unsafe for this box.
  template <typename Fn>
  void visitOnly(const double *R, std::size_t n, const double *box,
                 const Options &opt, Fn &&fn) {
    setup(n, box, opt);
    double w[3];
    double inv[3];
    fold_params(box, opt, w, inv);
    auto flip = [&](int32_t i, int32_t j, double dx, double dy, double dz,
                    double r2) { fn(i, j, -dx, -dy, -dz, r2); };
    CellGrid grid;
    if (general_) {
      generalVisit(R, n, -1.0, opt.cutoff * opt.cutoff, flip);
    } else if (mic_ && grid.build(R, n, w, inv, opt.cutoff)) {
      cell_visit<false>(grid, R, w, inv, HUGE_VAL, opt.cutoff * opt.cutoff,
                        pairsIJ_, flip);
    } else {
      vesin::cpu::brute_force_visit_only(R, n, w, inv, opt.cutoff * opt.cutoff,
                                         flip);
    }
    if (mic_) {
      pairsIJ_.clear();
      finishRebuild(R, n, box, opt, true, false);
      phantom_ = true;
    }
  }

  /// Second sighting: capture the candidate list at cutoff+skin while
  /// evaluating ``fn`` in the same scan.
  template <typename Fn>
  void rebuildFused(const double *R, std::size_t n, const double *box,
                    const Options &opt, Fn &&fn) {
    setup(n, box, opt);
    double w[3];
    double inv[3];
    fold_params(box, opt, w, inv);
    const double bc = opt.cutoff + opt.skin;
    auto flip = [&](int32_t i, int32_t j, double dx, double dy, double dz,
                    double r2) { fn(i, j, -dx, -dy, -dz, r2); };
    CellGrid grid;
    const bool cells = mic_ && grid.build(R, n, w, inv, bc);
    if (general_) {
      generalVisit(R, n, bc * bc, opt.cutoff * opt.cutoff, flip);
    } else if (cells) {
      cell_visit<true>(grid, R, w, inv, bc * bc, opt.cutoff * opt.cutoff,
                       pairsIJ_, flip);
    } else {
      vesin::cpu::brute_force_visit(R, n, w, inv, bc * bc,
                                    opt.cutoff * opt.cutoff, pairsIJ_, flip);
    }
    finishRebuild(R, n, box, opt, !cells, true);
  }

  /// Build the candidate list at cutoff + skin for positions ``R``, the
  /// list a later call reuses while the Verlet guarantee holds.
  void rebuild(const double *R, std::size_t n, const double *box,
               const Options &opt) {
    rebuildFused(R, n, box, opt,
                 [](int32_t, int32_t, double, double, double, double) {});
  }

  /// Build a list holding exactly the pairs within the cutoff of ``R``,
  /// for one evaluation through accumulate. When the box is cacheable the
  /// slot then becomes a phantom (dropPairs) that only stamps the
  /// geometry, as visitOnly leaves it.
  void buildForEval(const double *R, std::size_t n, const double *box,
                    const Options &opt) {
    setup(n, box, opt);
    double w[3];
    double inv[3];
    fold_params(box, opt, w, inv);
    const double c2 = opt.cutoff * opt.cutoff;
    auto none = [](int32_t, int32_t, double, double, double, double) {};
    CellGrid grid;
    const bool cells = mic_ && grid.build(R, n, w, inv, opt.cutoff);
    if (general_) {
      generalVisit(R, n, std::nextafter(c2, HUGE_VAL), -1.0, none);
    } else if (cells) {
      cell_visit<true>(grid, R, w, inv, std::nextafter(c2, HUGE_VAL), -1.0,
                       pairsIJ_, none);
    } else {
      vesin::cpu::brute_force_visit(R, n, w, inv, std::nextafter(c2, HUGE_VAL),
                                    -1.0, pairsIJ_, none);
    }
    // One evaluation: the rounding fold costs what recording the images
    // would, and gives the same vectors bit for bit.
    finishRebuild(R, n, box, opt, !cells, false);
  }

  /// Turn an evaluated slot into a phantom: keep the stamp, drop the pairs.
  void dropPairs() {
    nbr_.clear();
    nbr_.shrink_to_fit();
    code_.clear();
    code_.shrink_to_fit();
    std::fill(rows_.begin(), rows_.end(), 0);
    phantom_ = true;
  }

  /// True when caching applies to this box (decided by the last build).
  [[nodiscard]] bool cacheable() const { return mic_; }

  /// Fold the evaluation loops use for this slot.
  [[nodiscard]] FoldMode foldMode() const { return fold_; }

  /// Candidate pairs held (within cutoff + skin at the build).
  [[nodiscard]] std::size_t size() const { return nbr_.size(); }

  /// Visit every cached pair within the true cutoff of the current
  /// positions; ``fn(i, j, dx, dy, dz, r2)`` uses ``d = r_i - r_j`` with
  /// the minimum image applied.
  template <typename Fn> void forEach(const double *R, Fn &&fn) const {
    switch (fold_) {
    case FoldMode::None:
      forEachImpl<FoldMode::None>(R, fn);
      break;
    case FoldMode::Round:
      forEachImpl<FoldMode::Round>(R, fn);
      break;
    case FoldMode::Coded:
      forEachImpl<FoldMode::Coded>(R, fn);
      break;
    case FoldMode::General:
      forEachImpl<FoldMode::General>(R, fn);
      break;
    }
  }

  /// Fused force loop over the cached pairs: ``kernel(i, j, r2)`` returns
  /// the pair energy and ``fscale = -V'(r) / r``; the loop adds
  /// ``fscale * d`` to atom i and subtracts it from atom j (``d = r_i -
  /// r_j``, minimum image applied) and returns the summed energy. Atom i's
  /// force stays in registers across its row. The optional observer receives
  /// the same force scale and displacement once per accepted pair.
  template <typename Kernel, typename Observer = IgnorePairContribution>
  [[nodiscard]] double accumulate(const double *R, double *F, Kernel &&kernel,
                                  Observer &&observe = {}) const {
    switch (fold_) {
    case FoldMode::None:
      return accumulateImpl<FoldMode::None>(R, F, kernel, observe);
    case FoldMode::Round:
      return accumulateImpl<FoldMode::Round>(R, F, kernel, observe);
    case FoldMode::Coded:
      return accumulateImpl<FoldMode::Coded>(R, F, kernel, observe);
    case FoldMode::General:
      return accumulateImpl<FoldMode::General>(R, F, kernel, observe);
    }
    return 0.0;
  }

private:
  template <FoldMode M, typename Fn>
  void forEachImpl(const double *R, Fn &fn) const {
    const double cutoff2 = opt_.cutoff * opt_.cutoff;
    const Fold fold = makeFold();
    for (std::size_t i = 0; i < n_; ++i) {
      const int32_t p0 = rows_[i];
      const int32_t p1 = rows_[i + 1];
      for (int32_t p = p0; p < p1; ++p) {
        const auto up = static_cast<std::size_t>(p);
        const int32_t j = nbr_[up];
        double dx, dy, dz;
        fold.template apply<M>(R, i, static_cast<std::size_t>(j), up, dx, dy,
                               dz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= cutoff2) {
          fn(static_cast<int32_t>(i), j, dx, dy, dz, r2);
        }
      }
    }
  }

  template <FoldMode M, typename Kernel, typename Observer>
  [[nodiscard]] double accumulateImpl(const double *R, double *F,
                                      Kernel &kernel, Observer &observe) const {
    const double cutoff2 = opt_.cutoff * opt_.cutoff;
    const Fold fold = makeFold();
    double energy = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
      const int32_t p0 = rows_[i];
      const int32_t p1 = rows_[i + 1];
      double fxi = 0.0, fyi = 0.0, fzi = 0.0;
      for (int32_t p = p0; p < p1; ++p) {
        const auto up = static_cast<std::size_t>(p);
        const auto j = static_cast<std::size_t>(nbr_[up]);
        double dx, dy, dz;
        fold.template apply<M>(R, i, j, up, dx, dy, dz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        // A branch, not a mask: masking evaluates the kernel (an exp for
        // Morse) for every candidate past the cutoff, and measures slower
        // for Morse at every size and no faster for LJ.
        if (r2 <= cutoff2) {
          const PairTerm t =
              kernel(static_cast<int32_t>(i), static_cast<int32_t>(j), r2);
          energy += t.energy;
          const double fx = t.fscale * dx;
          const double fy = t.fscale * dy;
          const double fz = t.fscale * dz;
          fxi += fx;
          fyi += fy;
          fzi += fz;
          F[3 * j] -= fx;
          F[3 * j + 1] -= fy;
          F[3 * j + 2] -= fz;
          observe(t.fscale, dx, dy, dz);
        }
      }
      F[3 * i] += fxi;
      F[3 * i + 1] += fyi;
      F[3 * i + 2] += fzi;
    }
    return energy;
  }

  template <typename Fn>
  void generalVisit(const double *R, std::size_t n, double listCutoff2,
                    double visitCutoff2, Fn &fn) {
    pairsIJ_.clear();
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        double dx = R[3 * j] - R[3 * i];
        double dy = R[3 * j + 1] - R[3 * i + 1];
        double dz = R[3 * j + 2] - R[3 * i + 2];
        generalImage_.fold(dx, dy, dz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < listCutoff2) {
          pairsIJ_.push_back(static_cast<int32_t>(i));
          pairsIJ_.push_back(static_cast<int32_t>(j));
        }
        if (r2 <= visitCutoff2)
          fn(static_cast<int32_t>(i), static_cast<int32_t>(j), dx, dy, dz, r2);
      }
    }
  }

  void setup(std::size_t n, const double *box, const Options &opt) {
    const bool orthorhombic = box[1] == 0.0 && box[2] == 0.0 && box[3] == 0.0 &&
                              box[5] == 0.0 && box[6] == 0.0 && box[7] == 0.0;
    general_ = !orthorhombic &&
               (opt.periodic[0] || opt.periodic[1] || opt.periodic[2]);
    if (general_)
      generalImage_ = MinimumImage(box, opt.periodic);
    // The atom cap bounds the brute-force list build; a fully periodic box
    // that the linked-cell scan can serve needs no cap.
    mic_ = orthorhombic && (n <= 20000 || cellGridFits(n, box, opt));
    for (int k = 0; mic_ && k < 3; ++k) {
      if (opt.periodic[static_cast<std::size_t>(k)] &&
          opt.cutoff > 0.5 * box[4 * k]) {
        mic_ = false;
      }
    }
  }

  /// True when every axis is periodic and wide enough that CellGrid::build
  /// accepts the box at the list cutoff for any atom positions.
  static bool cellGridFits(std::size_t n, const double *box,
                           const Options &opt) {
    const double rl = opt.cutoff + opt.skin;
    std::size_t ncells = 1;
    for (int k = 0; k < 3; ++k) {
      if (!opt.periodic[static_cast<std::size_t>(k)] || !(rl > 0.0)) {
        return false;
      }
      const double fit = std::floor(box[4 * k] / rl);
      if (fit < 3.0) {
        return false;
      }
      ncells *= static_cast<std::size_t>(std::min(fit, 1024.0));
    }
    return ncells >= kMinCellsForGrid && ncells <= 8 * n + 64;
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

  /// ``rowsSorted``: the scan already emitted each row in ascending
  /// partner order (the brute-force scan does). ``codes``: record image
  /// codes for a list later calls reuse.
  void finishRebuild(const double *R, std::size_t n, const double *box,
                     const Options &opt, bool rowsSorted, bool codes) {
    // The fold of the historical per-call loop: diagonal widths along the
    // periodic axes, also for a box the pool will not cache.
    {
      double w[3];
      double inv[3];
      fold_params(box, opt, w, inv);
      for (int k = 0; k < 3; ++k) {
        micInv_[static_cast<std::size_t>(k)] = inv[k];
      }
    }
    Rref_.assign(R, R + 3 * n);
    for (int k = 0; k < 9; ++k) {
      boxref_[k] = box[k];
    }
    opt_ = opt;
    n_ = n;
    const std::size_t np = pairsIJ_.size() / 2;
    complete_ = mic_ && np == n * (n - 1) / 2;

    // Row-major (CSR) half list: atom i owns its partners j > i. The scans
    // report each pair as (a, b) with a < b, so a counting sort on a is
    // all it takes.
    rows_.assign(n + 1, 0);
    for (std::size_t p = 0; p < np; ++p) {
      ++rows_[static_cast<std::size_t>(pairsIJ_[2 * p]) + 1];
    }
    for (std::size_t i = 0; i < n; ++i) {
      rows_[i + 1] += rows_[i];
    }
    nbr_.resize(np);
    {
      std::vector<int32_t> fill(rows_.begin(), rows_.end() - 1);
      for (std::size_t p = 0; p < np; ++p) {
        const auto i = static_cast<std::size_t>(pairsIJ_[2 * p]);
        nbr_[static_cast<std::size_t>(fill[i]++)] = pairsIJ_[2 * p + 1];
      }
    }
    // Rows from the cell scan arrive in cell order; sorting each row keeps
    // the partner order, and so the summation order, independent of the
    // scan that built the list.
    if (!rowsSorted) {
      for (std::size_t i = 0; i < n; ++i) {
        std::sort(nbr_.begin() + rows_[i], nbr_.begin() + rows_[i + 1]);
      }
    }
    pairsIJ_.clear();
    pairsIJ_.shrink_to_fit();
    if (general_) {
      code_.clear();
      fold_ = FoldMode::General;
    } else if (codes) {
      buildImageCodes();
    } else {
      code_.clear();
      const bool fold =
          micInv_[0] != 0.0 || micInv_[1] != 0.0 || micInv_[2] != 0.0;
      fold_ = fold ? FoldMode::Round : FoldMode::None;
    }
    built_ = true;
  }

  /// Record each candidate's periodic image at build time, so evaluation
  /// subtracts a stored shift instead of rounding three quotients per pair.
  ///
  /// Sound only when cutoff + skin < w / 2 along every periodic axis: a
  /// pair inside the cutoff now moved less than one skin since the build,
  /// so its build-time separation already sat within w / 2 of the image
  /// it has now, and that image is the one the build folded to. With a
  /// complete list motion never invalidates the slot, so the image can
  /// change and the fold stays per call. Partners more than one box width
  /// apart (unwrapped coordinates) fall back to the fold too.
  void buildImageCodes() {
    code_.clear();
    fold_ = FoldMode::None;
    if (micInv_[0] == 0.0 && micInv_[1] == 0.0 && micInv_[2] == 0.0) {
      return;
    }
    fold_ = FoldMode::Round;
    if (complete_) {
      return;
    }
    const double reach = opt_.cutoff + opt_.skin;
    for (int k = 0; k < 3; ++k) {
      if (micInv_[static_cast<std::size_t>(k)] != 0.0 &&
          !(reach < 0.5 * boxref_[4 * k])) {
        return;
      }
    }
    for (int c = 0; c < 27; ++c) {
      const int kk[3] = {c % 3 - 1, (c / 3) % 3 - 1, c / 9 - 1};
      for (int k = 0; k < 3; ++k) {
        shift_[static_cast<std::size_t>(3 * c + k)] =
            static_cast<double>(kk[k]) * boxref_[4 * k];
      }
    }
    code_.resize(nbr_.size());
    const double *ref = Rref_.data();
    for (std::size_t i = 0; i < n_; ++i) {
      for (int32_t p = rows_[i]; p < rows_[i + 1]; ++p) {
        const auto up = static_cast<std::size_t>(p);
        const auto j = static_cast<std::size_t>(nbr_[up]);
        int c = 0;
        int stride = 1;
        for (int k = 0; k < 3; ++k) {
          const auto uk = static_cast<std::size_t>(k);
          const double d = ref[3 * i + uk] - ref[3 * j + uk];
          const double m = vesin::cpu::visit_round(d * micInv_[uk]);
          if (m < -1.0 || m > 1.0) {
            code_.clear();
            return;
          }
          c += (static_cast<int>(m) + 1) * stride;
          stride *= 3;
        }
        code_[up] = static_cast<uint8_t>(c);
      }
    }
    fold_ = FoldMode::Coded;
  }

  /// How the evaluation loops turn r_i - r_j into the minimum image.
  struct Fold {
    FoldMode mode;
    double w[3];
    double inv[3];
    const uint8_t *code;
    const double *shift;
    const MinimumImage *general;

    template <FoldMode M>
    void apply(const double *R, std::size_t i, std::size_t j, std::size_t p,
               double &dx, double &dy, double &dz) const {
      dx = R[3 * i] - R[3 * j];
      dy = R[3 * i + 1] - R[3 * j + 1];
      dz = R[3 * i + 2] - R[3 * j + 2];
      if constexpr (M == FoldMode::General) {
        general->fold(dx, dy, dz);
      } else if constexpr (M == FoldMode::Coded) {
        const double *sh = shift + 3 * static_cast<std::size_t>(code[p]);
        dx -= sh[0];
        dy -= sh[1];
        dz -= sh[2];
      } else if constexpr (M == FoldMode::Round) {
        dx -= w[0] * vesin::cpu::visit_round(dx * inv[0]);
        dy -= w[1] * vesin::cpu::visit_round(dy * inv[1]);
        dz -= w[2] * vesin::cpu::visit_round(dz * inv[2]);
      }
    }
  };

  [[nodiscard]] Fold makeFold() const {
    return Fold{fold_,
                {boxref_[0], boxref_[4], boxref_[8]},
                {micInv_[0], micInv_[1], micInv_[2]},
                code_.data(),
                shift_.data(),
                &generalImage_};
  }

  std::vector<int32_t> pairsIJ_;   //!< Scan output, (a, b) flat; build only.
  std::vector<int32_t> rows_;      //!< CSR row starts, n + 1 entries.
  std::vector<int32_t> nbr_;       //!< Partner j > i of each candidate.
  std::vector<uint8_t> code_;      //!< Image code per candidate (Coded fold).
  std::array<double, 81> shift_{}; //!< 27 image shifts, three doubles each.
  std::vector<double> Rref_;
  std::array<double, 9> boxref_{};
  std::array<double, 3> micInv_{};
  Options opt_{};
  std::size_t n_{0};
  FoldMode fold_{FoldMode::None};
  MinimumImage generalImage_;
  bool general_{false};
  bool mic_{false};
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

  /// Single-pass evaluation with lazy list capture (see file header).
  /// ``fn(i, j, dx, dy, dz, r2)`` sees every pair within the cutoff once,
  /// with ``d = r_i - r_j`` (minimum image applied).
  template <typename Fn>
  void evaluate(const double *R, std::size_t n, const double *box,
                const CachedPairList::Options &opt, Fn &&fn) {
    run(R, n, box, opt, fn,
        [&](const CachedPairList &list) { list.forEach(R, fn); });
  }

  /// Energy and forces of a radial pair potential. ``kernel(i, j, r2)``
  /// returns the pair's PairTerm; the forces are added into ``F`` (3 n
  /// doubles, interleaved) and the total energy is returned.
  ///
  /// Every call, cached or not, runs CachedPairList::accumulate over a
  /// sorted CSR list: a fresh geometry first gets a list (at the cutoff for
  /// a first sighting, at cutoff + skin when it is captured). The pairs
  /// inside the cutoff and their summation order are then the same whichever
  /// slot or scan served the call, so a result does not depend on the pool's
  /// history or on which thread reached it first.
  template <typename Kernel, typename Observer = IgnorePairContribution>
  [[nodiscard]] double accumulate(const double *R, std::size_t n,
                                  const double *box,
                                  const CachedPairList::Options &opt, double *F,
                                  Kernel &&kernel, Observer &&observe = {}) {
    std::shared_ptr<const CachedPairList> hit = lookup(R, n, box, opt);
    if (hit && !hit->isPhantom()) {
      return hit->accumulate(R, F, kernel, observe);
    }
    auto fresh = std::make_shared<CachedPairList>();
    double energy;
    if (hit) {
      fresh->rebuild(R, n, box, opt);
      energy = fresh->accumulate(R, F, kernel, observe);
    } else {
      fresh->buildForEval(R, n, box, opt);
      energy = fresh->accumulate(R, F, kernel, observe);
      if (!fresh->cacheable()) {
        return energy;
      }
      fresh->dropPairs();
    }
    store(fresh, hit);
    return energy;
  }

  static PairListCache &global() {
    static PairListCache cache;
    return cache;
  }

private:
  /// Most recently used slot valid for ``R``, moved to the front.
  std::shared_ptr<const CachedPairList>
  lookup(const double *R, std::size_t n, const double *box,
         const CachedPairList::Options &opt) {
    std::lock_guard<std::mutex> lock(mu_);
    for (std::size_t s = 0; s < slots_.size(); ++s) {
      if (slots_[s]->valid(R, n, box, opt)) {
        if (s != 0) {
          auto slot = std::move(slots_[s]);
          slots_.erase(slots_.begin() + static_cast<std::ptrdiff_t>(s));
          slots_.insert(slots_.begin(), std::move(slot));
        }
        return slots_.front();
      }
    }
    return nullptr;
  }

  /// Put ``fresh`` at the front, replacing the phantom ``hit`` it grew
  /// from, and evict the least recently used slot past kMaxSlots.
  void store(std::shared_ptr<CachedPairList> fresh,
             const std::shared_ptr<const CachedPairList> &hit) {
    std::lock_guard<std::mutex> lock(mu_);
    if (hit) {
      for (std::size_t s = 0; s < slots_.size(); ++s) {
        if (slots_[s] == hit) {
          slots_.erase(slots_.begin() + static_cast<std::ptrdiff_t>(s));
          break;
        }
      }
    }
    slots_.insert(slots_.begin(), std::move(fresh));
    if (slots_.size() > kMaxSlots) {
      slots_.pop_back();
    }
  }

  /// Pool lookup for evaluate: ``onHit`` runs on a captured slot, ``scan``
  /// is the visitor for a fresh search.
  template <typename Scan, typename OnHit>
  void run(const double *R, std::size_t n, const double *box,
           const CachedPairList::Options &opt, Scan &scan, OnHit &&onHit) {
    std::shared_ptr<const CachedPairList> hit = lookup(R, n, box, opt);
    if (hit && !hit->isPhantom()) {
      onHit(*hit);
      return;
    }
    auto fresh = std::make_shared<CachedPairList>();
    if (hit) {
      fresh->rebuildFused(R, n, box, opt, scan);
    } else {
      fresh->visitOnly(R, n, box, opt, scan);
      if (!fresh->cacheable()) {
        return; // box unsafe for caching; behave exactly like the old loop
      }
    }
    store(std::move(fresh), hit);
  }

  std::mutex mu_;
  std::vector<std::shared_ptr<CachedPairList>> slots_;
};

} // namespace nlist
} // namespace rgpot
