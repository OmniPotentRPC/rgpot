#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace rgpot::nlist {

/// Nearest periodic image for row-major lattice vectors: d - n H.
/// The dual-vector bound includes every image shorter than the initial
/// rounded fractional image. No lattice reduction or fixed image radius
/// is assumed. Singular or ill-conditioned cells and unrepresentable
/// search bounds throw.
///
/// The search runs in IEEE double. On x86-64, long double is x87
/// arithmetic, which the compiler cannot issue as vector instructions
/// and which is slower than double. A cell whose infinity-norm condition
/// estimate consumes more than half a double mantissa, but still passes
/// the long-double guard below, is searched in double-double instead.
/// A displacement whose fractional coordinates would lose the outward
/// one-plane margin, or whose cancellation would move the result by
/// more than 1e-9, is escalated the same way. The folded vector matches
/// the long-double search to within 1e-9. Cells the guard rejects still
/// throw.
class MinimumImage {
public:
  MinimumImage() = default;
  MinimumImage(const double *cell, const std::array<bool, 3> &periodic)
      : periodic_(periodic) {
    const GuardInverse guard = invertAndGuard(cell);
    for (int k = 0; k < 9; ++k)
      cell_[k] = cell[k];
    inv_norm_ = static_cast<double>(guard.inverse_norm);
    bool fits = std::isfinite(inv_norm_);
    for (int k = 0; k < 9; ++k) {
      inverse_[k] = static_cast<double>(guard.inverse[k]);
      const long double rem =
          guard.inverse[k] - static_cast<long double>(inverse_[k]);
      inverse_lo_[k] = static_cast<double>(rem);
      if (!std::isfinite(inverse_[k]) || !std::isfinite(inverse_lo_[k]))
        fits = false;
    }
    for (int k = 0; k < 3; ++k) {
      dual_[k] = static_cast<double>(guard.dual[k]);
      const long double rem =
          guard.dual[k] - static_cast<long double>(dual_[k]);
      dual_lo_[k] = static_cast<double>(rem);
      if (!std::isfinite(dual_[k]) || !std::isfinite(dual_lo_[k]))
        fits = false;
    }
    if (!fits) {
      range_ld_ = true;
      return;
    }
    const double eps = std::numeric_limits<double>::epsilon();
    const double cond = static_cast<double>(guard.cond);
    // Same half-precision test as the guard, applied to double. Cells
    // that fail it still passed the long-double guard above.
    precise_ = !(cond * eps <= std::sqrt(eps));
  }

  void fold(double &dx, double &dy, double &dz) const {
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz))
      throw std::invalid_argument("pair displacement must be finite");
    double x = dx;
    double y = dy;
    double z = dz;
    bool done = false;
    if (range_ld_) {
      foldLongDouble(x, y, z);
      done = true;
    } else if (!precise_ && !needsPrecise(dx, dy, dz)) {
      done = foldFast(x, y, z);
    }
    if (!done) {
      x = dx;
      y = dy;
      z = dz;
      foldPrecise(x, y, z);
    }
    dx = x;
    dy = y;
    dz = z;
  }

private:
  struct GuardInverse {
    std::array<long double, 9> inverse{};
    std::array<long double, 3> dual{};
    long double inverse_norm{0};
    long double cond{0};
  };

  // Dekker double-double: the unevaluated sum hi + lo, |lo| <= ulp(hi)/2
  // after renormalization. Used only on the escalated path.
  struct DD {
    double hi;
    double lo;
  };

  static DD dd_two_sum(double a, double b) {
    const double s = a + b;
    const double v = s - a;
    const double e = (a - (s - v)) + (b - v);
    return {s, e};
  }

  static DD dd_quick_two_sum(double a, double b) {
    const double s = a + b;
    const double e = b - (s - a);
    return {s, e};
  }

  static DD dd_renorm(double hi, double lo) {
    const DD s = dd_two_sum(hi, lo);
    return {s.hi, s.lo};
  }

  static DD dd_add(DD a, DD b) {
    const DD s = dd_two_sum(a.hi, b.hi);
    const DD t = dd_two_sum(a.lo, b.lo);
    DD r = dd_quick_two_sum(s.hi, s.lo + t.hi);
    r = dd_quick_two_sum(r.hi, r.lo + t.lo);
    return dd_renorm(r.hi, r.lo);
  }

  static DD dd_sub(DD a, DD b) { return dd_add(a, {-b.hi, -b.lo}); }

  static DD dd_mul(DD a, DD b) {
    const double p = a.hi * b.hi;
    double e = std::fma(a.hi, b.hi, -p);
    e += a.hi * b.lo + a.lo * b.hi;
    const DD s = dd_quick_two_sum(p, e);
    return dd_renorm(s.hi, s.lo);
  }

  static DD dd_div(DD a, DD b) {
    const double q1 = a.hi / b.hi;
    DD r = dd_sub(a, dd_mul({q1, 0.0}, b));
    const double q2 = r.hi / b.hi;
    r = dd_sub(r, dd_mul({q2, 0.0}, b));
    const double q3 = r.hi / b.hi;
    DD s = dd_quick_two_sum(q1, q2);
    s = dd_quick_two_sum(s.hi, s.lo + q3);
    return dd_renorm(s.hi, s.lo);
  }

  static DD dd_sqrt(DD a) {
    if (!(a.hi > 0.0))
      return {0.0, 0.0};
    const DD x{std::sqrt(a.hi), 0.0};
    // One Newton step doubles a double-accurate seed to double-double.
    return dd_mul(dd_add(x, dd_div(a, x)), {0.5, 0.0});
  }

  static DD dd_floor(DD a) {
    const double hi = std::floor(a.hi);
    if (hi != a.hi)
      return {hi, 0.0};
    const DD s = dd_quick_two_sum(hi, std::floor(a.lo));
    return dd_renorm(s.hi, s.lo);
  }

  static DD dd_ceil(DD a) {
    const DD n = dd_floor({-a.hi, -a.lo});
    return {-n.hi, -n.lo};
  }

  // std::round: nearest integer, halfway cases away from zero.
  static DD dd_round(DD a) {
    if (a.hi >= 0.0)
      return dd_floor(dd_add(a, {0.5, 0.0}));
    const DD n = dd_floor(dd_add({-a.hi, -a.lo}, {0.5, 0.0}));
    return {-n.hi, -n.lo};
  }

  // Exact. Splits on 2^32 so neither half needs more than a double.
  static DD dd_from_index(std::int64_t n) {
    constexpr std::int64_t base = 1LL << 32;
    if (n >= -(1LL << 53) && n <= (1LL << 53))
      return {static_cast<double>(n), 0.0};
    const std::int64_t hi_part = n / base;
    const std::int64_t lo_part = n - hi_part * base;
    return dd_renorm(static_cast<double>(hi_part) * 4294967296.0,
                     static_cast<double>(lo_part));
  }

  static bool dd_less(DD a, DD b) {
    if (a.hi < b.hi)
      return true;
    if (a.hi > b.hi)
      return false;
    return a.lo < b.lo;
  }

  struct DoubleArith {
    using T = double;
    static T load(double hi, double) { return hi; }
    static double store(T v) { return v; }
    static T add(T a, T b) { return a + b; }
    static T sub(T a, T b) { return a - b; }
    static T mul(T a, T b) { return a * b; }
    static T mul_index(T a, std::int64_t n) {
      return a * static_cast<double>(n);
    }
    static T square_sum(T x, T y, T z) { return x * x + y * y + z * z; }
    static T sqrt(T v) { return std::sqrt(v); }
    // Truncate toward zero. Out-of-range values are left unchanged so
    // checkedIndex throws instead of converting a non-representable index.
    static T trunc0(T v) {
      if (!(v > static_cast<T>(INT64_MIN) && v < 0x1p63))
        return v;
      return static_cast<T>(static_cast<std::int64_t>(v));
    }
    // Inline rounding. libm floor/ceil/round do not vectorize and dominate
    // a search whose image box is only a few planes wide.
    static T floor(T v) {
      const T t = trunc0(v);
      return t > v ? t - 1.0 : t;
    }
    static T ceil(T v) {
      const T t = trunc0(v);
      return t < v ? t + 1.0 : t;
    }
    static T round(T v) { return trunc0(v + std::copysign(0.5, v)); }
    static bool less(T a, T b) { return a < b; }
    static std::int64_t index(T value) { return checkedIndex(value); }
  };

  struct PreciseArith {
    using T = DD;
    static T load(double hi, double lo) { return {hi, lo}; }
    static double store(T v) { return dd_renorm(v.hi, v.lo).hi; }
    static T add(T a, T b) { return dd_add(a, b); }
    static T sub(T a, T b) { return dd_sub(a, b); }
    static T mul(T a, T b) { return dd_mul(a, b); }
    static T mul_index(T a, std::int64_t n) {
      return dd_mul(a, dd_from_index(n));
    }
    static T square_sum(T x, T y, T z) {
      return dd_add(dd_add(dd_mul(x, x), dd_mul(y, y)), dd_mul(z, z));
    }
    static T sqrt(T v) { return dd_sqrt(v); }
    static T round(T v) { return dd_round(v); }
    static T floor(T v) { return dd_floor(v); }
    static T ceil(T v) { return dd_ceil(v); }
    static bool less(T a, T b) { return dd_less(a, b); }
    static std::int64_t index(T value) { return checkedIndexDD(value); }
  };

  // Long-double inverse and the historical condition guard. The
  // accept/reject decision is unchanged from the long-double search.
  static GuardInverse invertAndGuard(const double *cell) {
    GuardInverse out;
    std::array<long double, 9> a{};
    for (int k = 0; k < 9; ++k) {
      if (!std::isfinite(cell[k]))
        throw std::invalid_argument("periodic cell must be finite");
      a[k] = cell[k];
    }
    const long double det = a[0] * (a[4] * a[8] - a[5] * a[7]) -
                            a[1] * (a[3] * a[8] - a[5] * a[6]) +
                            a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (!std::isfinite(det) || det == 0.0L)
      throw std::invalid_argument("periodic cell must be nonsingular");
    out.inverse = {
        (a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det,
        (a[1] * a[5] - a[2] * a[4]) / det, (a[5] * a[6] - a[3] * a[8]) / det,
        (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
        (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det,
        (a[0] * a[4] - a[1] * a[3]) / det};
    long double cell_norm = 0.0L;
    long double inverse_norm = 0.0L;
    for (int row = 0; row < 3; ++row) {
      long double cell_sum = 0.0L;
      long double inverse_sum = 0.0L;
      for (int col = 0; col < 3; ++col) {
        cell_sum += std::abs(a[3 * row + col]);
        inverse_sum += std::abs(out.inverse[3 * row + col]);
      }
      cell_norm = std::max(cell_norm, cell_sum);
      inverse_norm = std::max(inverse_norm, inverse_sum);
    }
    const long double eps = std::numeric_limits<long double>::epsilon();
    out.cond = cell_norm * inverse_norm;
    out.inverse_norm = inverse_norm;
    // Refuse cells whose condition estimate consumes over half the
    // working precision; finite image bounds alone do not protect the inverse.
    if (!(out.cond * eps <= std::sqrt(eps)))
      throw std::invalid_argument(
          "periodic cell is too ill-conditioned for image bounds");
    for (int k = 0; k < 3; ++k) {
      out.dual[k] =
          std::hypot(out.inverse[k], out.inverse[3 + k], out.inverse[6 + k]);
      if (!std::isfinite(out.dual[k]))
        throw std::overflow_error("periodic cell inverse is not finite");
    }
    return out;
  }

  static std::int64_t checkedIndexLD(long double value) {
    const long double limit = std::ldexp(1.0L, 63);
    if (!std::isfinite(value) || value <= -limit + 1.0L ||
        value >= limit - 1.0L)
      throw std::overflow_error("periodic image index is out of range");
    return static_cast<std::int64_t>(value);
  }

  static std::int64_t checkedIndex(double value) {
    // Every double the long-double guard would reject is outside
    // [-2^63, 2^63). Integers in between that double can represent are
    // accepted by both.
    if (!std::isfinite(value) || value <= -0x1p63 || value >= 0x1p63)
      throw std::overflow_error("periodic image index is out of range");
    return static_cast<std::int64_t>(value);
  }

  static std::int64_t checkedIndexDD(DD value) {
    if (!std::isfinite(value.hi) || !std::isfinite(value.lo) ||
        value.hi >= 0x1p63 || value.hi < -0x1p63)
      throw std::overflow_error("periodic image index is out of range");
    std::int64_t merged;
    if (std::abs(value.hi) <= 0x1p52) {
      const double v = value.hi + value.lo;
      if (!std::isfinite(v) || v <= static_cast<double>(INT64_MIN) ||
          v >= 0x1p63)
        throw std::overflow_error("periodic image index is out of range");
      merged = static_cast<std::int64_t>(v);
    } else {
      const auto ihi = static_cast<std::int64_t>(value.hi);
      const double lo = std::round(value.lo);
      if (!std::isfinite(lo) || lo <= static_cast<double>(INT64_MIN) ||
          lo >= 0x1p63)
        throw std::overflow_error("periodic image index is out of range");
      const auto ilo = static_cast<std::int64_t>(lo);
      if ((ilo > 0 && ihi > INT64_MAX - ilo) ||
          (ilo < 0 && ihi < INT64_MIN - ilo))
        throw std::overflow_error("periodic image index is out of range");
      merged = ihi + ilo;
    }
    // Match the long-double bounds: reject <= -2^63+1 and >= 2^63-1.
    if (merged <= INT64_MIN + 1 || merged >= INT64_MAX)
      throw std::overflow_error("periodic image index is out of range");
    return merged;
  }

  // Escalate when a double fractional coordinate could miss the
  // one-plane margin, or when cancelling n H from a large displacement
  // would move the returned vector by more than 1e-9. Ordinary
  // separations stay on the double path.
  bool needsPrecise(double dx, double dy, double dz) const {
    const double r1 = std::abs(dx) + std::abs(dy) + std::abs(dz);
    const double eps = std::numeric_limits<double>::epsilon();
    const double scale = std::max(1.0, inv_norm_);
    const double df = 8.0 * eps * r1 * inv_norm_;
    const double dpos = 8.0 * eps * r1 * scale;
    return !(df <= 0.25 && dpos <= 1e-9);
  }

  template <class Arith>
  bool search(double &dx, double &dy, double &dz) const {
    using T = typename Arith::T;
    T reduced[3] = {Arith::load(dx, 0.0), Arith::load(dy, 0.0),
                    Arith::load(dz, 0.0)};
    T initial[3] = {};
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k]) {
        T f = Arith::load(0.0, 0.0);
        for (int row = 0; row < 3; ++row) {
          const T g = Arith::load(inverse_[3 * row + k],
                                  inverse_lo_[3 * row + k]);
          f = Arith::add(f, Arith::mul(reduced[row], g));
        }
        initial[k] = Arith::round(f);
      }
      Arith::index(initial[k]);
    }
    for (int c = 0; c < 3; ++c) {
      for (int k = 0; k < 3; ++k) {
        const T shift =
            Arith::mul_index(Arith::load(cell_[3 * k + c], 0.0),
                             Arith::index(initial[k]));
        reduced[c] = Arith::sub(reduced[c], shift);
      }
    }
    T best[3] = {reduced[0], reduced[1], reduced[2]};
    T best2 = Arith::square_sum(best[0], best[1], best[2]);
    const T radius = Arith::sqrt(best2);
    std::int64_t low[3] = {};
    std::int64_t high[3] = {};
    std::uint64_t count = 1;
    std::int64_t index_abs = 0;
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k]) {
        T f = Arith::load(0.0, 0.0);
        for (int row = 0; row < 3; ++row) {
          const T g = Arith::load(inverse_[3 * row + k],
                                  inverse_lo_[3 * row + k]);
          f = Arith::add(f, Arith::mul(reduced[row], g));
        }
        const T dual = Arith::load(dual_[k], dual_lo_[k]);
        const T extent = Arith::mul(radius, dual);
        // Outward integer rounding adds one lattice plane of numerical
        // margin around the real-arithmetic image bounds.
        const T one = Arith::load(1.0, 0.0);
        const T below = Arith::sub(Arith::sub(f, extent), one);
        const T above = Arith::add(Arith::add(f, extent), one);
        low[k] = Arith::index(Arith::floor(below));
        high[k] = Arith::index(Arith::ceil(above));
      }
      index_abs = std::max(index_abs, std::abs(low[k]));
      index_abs = std::max(index_abs, std::abs(high[k]));
      const auto width = static_cast<std::uint64_t>(high[k]) -
                         static_cast<std::uint64_t>(low[k]) + 1;
      if (width == 0 || count > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::int64_t>::max()) /
                                    width)
        throw std::overflow_error("periodic image search count is too large");
      count *= width;
    }

    const bool check_gap = !std::is_same<Arith, PreciseArith>::value;
    double second2 = std::numeric_limits<double>::infinity();
    double best2_d = check_gap ? Arith::store(best2) : 0.0;
    for (auto i = low[0]; i <= high[0]; ++i) {
      for (auto j = low[1]; j <= high[1]; ++j) {
        for (auto k = low[2]; k <= high[2]; ++k) {
          T candidate[3];
          for (int c = 0; c < 3; ++c) {
            T v = reduced[c];
            v = Arith::sub(v, Arith::mul_index(Arith::load(cell_[c], 0.0), i));
            v = Arith::sub(
                v, Arith::mul_index(Arith::load(cell_[3 + c], 0.0), j));
            v = Arith::sub(
                v, Arith::mul_index(Arith::load(cell_[6 + c], 0.0), k));
            candidate[c] = v;
          }
          const T r2 =
              Arith::square_sum(candidate[0], candidate[1], candidate[2]);
          if (check_gap) {
            const double r2_d = Arith::store(r2);
            if (r2_d < best2_d) {
              second2 = best2_d;
              best2_d = r2_d;
            } else if (r2_d > best2_d && r2_d < second2) {
              second2 = r2_d;
            }
          }
          if (Arith::less(r2, best2)) {
            best[0] = candidate[0];
            best[1] = candidate[1];
            best[2] = candidate[2];
            best2 = r2;
          }
        }
      }
    }
    if (check_gap) {
      double mag = 0.0;
      for (int c = 0; c < 3; ++c)
        mag += std::abs(Arith::store(reduced[c]));
      double cell_sum = 0.0;
      for (double entry : cell_)
        cell_sum += std::abs(entry);
      mag += static_cast<double>(index_abs) * cell_sum;
      const double err = 256.0 * std::numeric_limits<double>::epsilon() * mag *
                         mag;
      if (!(second2 - best2_d > err))
        return false;
    }
    dx = Arith::store(best[0]);
    dy = Arith::store(best[1]);
    dz = Arith::store(best[2]);
    return true;
  }

  bool foldFast(double &dx, double &dy, double &dz) const {
    return search<DoubleArith>(dx, dy, dz);
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((noinline, cold))
#endif
  void foldPrecise(double &dx, double &dy, double &dz) const {
    search<PreciseArith>(dx, dy, dz);
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((noinline, cold))
#endif
  void foldLongDouble(double &dx, double &dy, double &dz) const {
    const GuardInverse guard = invertAndGuard(cell_.data());
    const auto &inverse = guard.inverse;
    const auto &dual = guard.dual;
    std::array<long double, 9> cell{};
    for (int k = 0; k < 9; ++k)
      cell[k] = cell_[k];
    std::array<long double, 3> reduced{dx, dy, dz};
    std::array<long double, 3> initial{};
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k])
        initial[k] =
            std::round(reduced[0] * inverse[k] + reduced[1] * inverse[3 + k] +
                       reduced[2] * inverse[6 + k]);
      checkedIndexLD(initial[k]);
    }
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k)
        reduced[c] -= initial[k] * cell[3 * k + c];
    auto best = reduced;
    long double best2 =
        best[0] * best[0] + best[1] * best[1] + best[2] * best[2];
    const long double radius = std::sqrt(best2);
    std::array<std::int64_t, 3> low{}, high{};
    std::uint64_t count = 1;
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k]) {
        const long double f = reduced[0] * inverse[k] +
                              reduced[1] * inverse[3 + k] +
                              reduced[2] * inverse[6 + k];
        const long double extent = radius * dual[k];
        low[k] = checkedIndexLD(std::floor(f - extent) - 1.0L);
        high[k] = checkedIndexLD(std::ceil(f + extent) + 1.0L);
      }
      const auto width = static_cast<std::uint64_t>(high[k]) -
                         static_cast<std::uint64_t>(low[k]) + 1;
      if (width == 0 || count > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::int64_t>::max()) /
                                    width)
        throw std::overflow_error("periodic image search count is too large");
      count *= width;
    }
    for (auto i = low[0]; i <= high[0]; ++i)
      for (auto j = low[1]; j <= high[1]; ++j)
        for (auto k = low[2]; k <= high[2]; ++k) {
          std::array<long double, 3> candidate{};
          for (int c = 0; c < 3; ++c)
            candidate[c] = reduced[c] - static_cast<long double>(i) * cell[c] -
                           static_cast<long double>(j) * cell[3 + c] -
                           static_cast<long double>(k) * cell[6 + c];
          const long double r2 = candidate[0] * candidate[0] +
                                 candidate[1] * candidate[1] +
                                 candidate[2] * candidate[2];
          if (r2 < best2) {
            best = candidate;
            best2 = r2;
          }
        }
    dx = static_cast<double>(best[0]);
    dy = static_cast<double>(best[1]);
    dz = static_cast<double>(best[2]);
  }

  std::array<double, 9> cell_{};
  std::array<double, 9> inverse_{};
  std::array<double, 3> dual_{};
  std::array<bool, 3> periodic_{};
  double inv_norm_{0};
  std::array<double, 9> inverse_lo_{};
  std::array<double, 3> dual_lo_{};
  bool precise_{false};
  bool range_ld_{false};
};

} // namespace rgpot::nlist
