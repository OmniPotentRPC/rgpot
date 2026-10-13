// Header-only fused MIC pair visitation for vesin's CPU brute-force
// algorithm (eOn extension, upstream candidate).
//
// The C API's vesin_neighbors_visit pays an indirect call per in-cutoff
// pair, which forces register spills inside the scan loop. This template
// lets the compiler inline the visitor into the single O(n^2) scan, and
// collects the candidate pair list (as compact int32 index pairs) in the
// same pass. Orthorhombic / free-boundary boxes only; exactly one pair per
// (i, j) — the nearest periodic image, folded with the widths in `w`.
#ifndef VESIN_VISIT_HPP
#define VESIN_VISIT_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vesin {
namespace cpu {
namespace detail {

/// Split interleaved xyz into three unit-stride arrays for the pair scan.
struct Soa3 {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
};

inline Soa3 split_xyz(const double *points, std::size_t n) {
  Soa3 s;
  s.x.resize(n);
  s.y.resize(n);
  s.z.resize(n);
  for (std::size_t a = 0; a < n; ++a) {
    s.x[a] = points[3 * a];
    s.y[a] = points[3 * a + 1];
    s.z[a] = points[3 * a + 2];
  }
  return s;
}

} // namespace detail
} // namespace cpu
} // namespace vesin

namespace vesin {
namespace cpu {

/// Round to nearest via truncate-cast: no libm floor call on -march
/// targets without roundsd. Half-integer inputs round away from zero.
inline double visit_round(double t) {
    return static_cast<double>(static_cast<long long>(t + std::copysign(0.5, t)));
}

/// Half-list length from the number density times the sphere of radius
/// sqrt(list_cutoff2). Each unordered pair is stored once. A zero box
/// volume uses a small default. The pad covers a shell on the sphere.
inline std::size_t half_list_reserve(std::size_t n, const double w[3],
                                    double list_cutoff2) {
    constexpr double kPi = 3.14159265358979323846;
    const double rc2 = list_cutoff2 > 0.0 ? list_cutoff2 : 0.0;
    const double rc = std::sqrt(rc2);
    const double sphere = (4.0 / 3.0) * kPi * rc * rc * rc;
    const double vol = w[0] * w[1] * w[2];
    double neigh = 16.0;
    if (n > 0 && vol > 0.0 && sphere > 0.0) {
        neigh = (static_cast<double>(n) / vol) * sphere;
    }
    if (!(neigh >= 1.0)) {
        neigh = 1.0;
    }
    const double all = static_cast<double>(n) * static_cast<double>(n > 0 ? n - 1 : 0) / 2.0;
    double pairs = 0.5 * static_cast<double>(n) * neigh * 1.5 + static_cast<double>(n);
    if (pairs > all) {
        pairs = all;
    }
    if (pairs < 0.0) {
        pairs = 0.0;
    }
    return static_cast<std::size_t>(pairs);
}

/// Scan all i < j pairs of `points` (interleaved x,y,z, `n` atoms), folding
/// each dimension by width `w[k]` when `inv[k] = 1/w[k]` is nonzero
/// (`inv[k] == 0` disables the fold for that dimension).
///
/// Pairs with `r2 < list_cutoff2` append (i, j) to `pairs` (flat int32,
/// cleared first); pairs with `r2 <= visit_cutoff2` additionally invoke
/// `visit(i, j, dx, dy, dz, r2)` with the folded `r_j - r_i` vector. Pass a
/// negative `visit_cutoff2` for a collect-only scan.
template <typename Visitor>
inline void brute_force_visit(
    const double* points,
    std::size_t n,
    const double w[3],
    const double inv[3],
    double list_cutoff2,
    double visit_cutoff2,
    std::vector<int32_t>& pairs,
    Visitor&& visit
) {
    pairs.clear();
    const std::size_t ints = half_list_reserve(n, w, list_cutoff2) * 2;
    if (pairs.capacity() < ints) {
        pairs.reserve(ints);
    }
    const auto soa = detail::split_xyz(points, n);
    const double *x = soa.x.data();
    const double *y = soa.y.data();
    const double *z = soa.z.data();
    const double w0 = w[0], w1 = w[1], w2 = w[2];
    const double i0 = inv[0], i1 = inv[1], i2 = inv[2];
    for (std::size_t i = 0; i < n; i++) {
        const double xi = x[i];
        const double yi = y[i];
        const double zi = z[i];
        // Two partners per step. Each pair is independent, and j stays
        // ascending so each row is already sorted.
        std::size_t j = i + 1;
        auto one = [&](std::size_t jj) {
            double dx = x[jj] - xi;
            double dy = y[jj] - yi;
            double dz = z[jj] - zi;
            dx -= w0 * visit_round(dx * i0);
            dy -= w1 * visit_round(dy * i1);
            dz -= w2 * visit_round(dz * i2);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < list_cutoff2) {
                pairs.push_back(static_cast<int32_t>(i));
                pairs.push_back(static_cast<int32_t>(jj));
                if (r2 <= visit_cutoff2) {
                    visit(static_cast<int32_t>(i), static_cast<int32_t>(jj),
                          dx, dy, dz, r2);
                }
            }
        };
        for (; j + 1 < n; j += 2) {
            one(j);
            one(j + 1);
        }
        if (j < n) {
            one(j);
        }
    }
}

/// Evaluation-only variant: same scan and fold, no pair collection. Serves
/// first-sighting evaluations where list capture would be paid for nothing.
template <typename Visitor>
inline void brute_force_visit_only(
    const double* points,
    std::size_t n,
    const double w[3],
    const double inv[3],
    double visit_cutoff2,
    Visitor&& visit
) {
    const auto soa = detail::split_xyz(points, n);
    const double *x = soa.x.data();
    const double *y = soa.y.data();
    const double *z = soa.z.data();
    const double w0 = w[0], w1 = w[1], w2 = w[2];
    const double i0 = inv[0], i1 = inv[1], i2 = inv[2];
    for (std::size_t i = 0; i < n; i++) {
        const double xi = x[i];
        const double yi = y[i];
        const double zi = z[i];
        std::size_t j = i + 1;
        auto one = [&](std::size_t jj) {
            double dx = x[jj] - xi;
            double dy = y[jj] - yi;
            double dz = z[jj] - zi;
            dx -= w0 * visit_round(dx * i0);
            dy -= w1 * visit_round(dy * i1);
            dz -= w2 * visit_round(dz * i2);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= visit_cutoff2) {
                visit(static_cast<int32_t>(i), static_cast<int32_t>(jj),
                      dx, dy, dz, r2);
            }
        };
        for (; j + 1 < n; j += 2) {
            one(j);
            one(j + 1);
        }
        if (j < n) {
            one(j);
        }
    }
}

} // namespace cpu
} // namespace vesin

#endif
