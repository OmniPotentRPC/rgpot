#include "rgpot/StillingerWeber/SWInternal.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {
namespace swi {
namespace {

struct PairTerm {
  bool live;
  double f2;
  double dedr;
  double gexp;
  double dgdr;
};

PairTerm pairTerm(double r) {
  const double u = r / StillingerWeberSi::kSigma;
  const double gap = u - StillingerWeberSi::kCut;
  if (!(gap < 0.0) || gap > -1.0 / 700.0) {
    return {};
  }
  const double s = 1.0 / gap;
  const double inv_u = 1.0 / u;
  const double u2 = inv_u * inv_u;
  const double u_m4 = u2 * u2;
  const double g = StillingerWeberSi::kB * u_m4 - 1.0;
  const double es = std::exp(s);
  const double f2 = StillingerWeberSi::kA * g * es;
  const double dgdu = -StillingerWeberSi::kP * StillingerWeberSi::kB * u_m4 * inv_u;
  const double dfdu = StillingerWeberSi::kA * es * (dgdu - g * s * s);
  const double dedr = StillingerWeberSi::kEpsilon * dfdu / StillingerWeberSi::kSigma;
  const double gexp = std::exp(StillingerWeberSi::kGamma * s);
  const double dgdr = gexp * (-StillingerWeberSi::kGamma * s * s) / StillingerWeberSi::kSigma;
  return {true, f2, dedr, gexp, dgdr};
}

int storeBonds(nlist::Neighbours &cells, int i, double cut2, Bond *dest, int room) {
  int m = 0;
  auto take = [&](std::int32_t, std::int32_t j, double dx, double dy, double dz, double r2) {
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
      throw std::invalid_argument("StillingerWeberSi: non-finite position");
    }
    if (!(r2 > 0.0) || r2 >= cut2) {
      return;
    }
    const double r = std::sqrt(r2);
    const PairTerm t = pairTerm(r);
    if (!t.live) {
      return;
    }
    if (m >= room) {
      return;
    }
    dest[m] = Bond{static_cast<int>(j), -dx, -dy, -dz, r, t.gexp, t.dgdr};
    ++m;
  };
  if (cells.streaming()) {
    cells.forAtom(static_cast<std::int32_t>(i), take);
  } else {
    nlist::visitDirected(cells, i, static_cast<std::int64_t>(i) + 1, take);
  }
  std::sort(dest, dest + m, [](const Bond &a, const Bond &b) { return a.j < b.j; });
  return m;
}

} // namespace

void prepareWide(nlist::Neighbours &cells, int n, int nt, double cut2, std::vector<Bond> &bonds,
                 std::vector<int> &bond_at, std::vector<int> &bond_n, std::vector<int> &energy_at,
                 std::vector<int> &energy_n, std::vector<int> &energy_used,
                 std::vector<int> &force_at, std::vector<int> &force_n,
                 std::vector<int> &force_used) {
  const auto &row = cells.offsets();
  bond_at.resize(static_cast<std::size_t>(n) + 1);
  bond_n.assign(static_cast<std::size_t>(n), 0);
  bond_at[0] = 0;
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
  {
#if defined(_OPENMP)
    const int t = omp_get_thread_num();
#else
    const int t = 0;
#endif
    const auto range = nlist::partition(n, t, nt);
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      bond_n[static_cast<std::size_t>(i)] =
          row[static_cast<std::size_t>(i) + 1] - row[static_cast<std::size_t>(i)];
    }
  }
  for (int i = 0; i < n; ++i) {
    bond_at[static_cast<std::size_t>(i) + 1] =
        bond_at[static_cast<std::size_t>(i)] + bond_n[static_cast<std::size_t>(i)];
  }
  if (static_cast<int>(bonds.size()) < bond_at[static_cast<std::size_t>(n)]) {
    bonds.resize(static_cast<std::size_t>(bond_at[static_cast<std::size_t>(n)]));
  }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
  {
#if defined(_OPENMP)
    const int t = omp_get_thread_num();
#else
    const int t = 0;
#endif
    const auto range = nlist::partition(n, t, nt);
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const int room =
          bond_at[static_cast<std::size_t>(i) + 1] - bond_at[static_cast<std::size_t>(i)];
      Bond *dest = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      bond_n[static_cast<std::size_t>(i)] = storeBonds(cells, i, cut2, dest, room);
    }
  }
  energy_at.resize(static_cast<std::size_t>(n) + 1);
  force_at.resize(static_cast<std::size_t>(n) + 1);
  energy_n.assign(static_cast<std::size_t>(n), 0);
  energy_used.assign(static_cast<std::size_t>(n), 0);
  force_n.assign(static_cast<std::size_t>(n), 0);
  force_used.assign(static_cast<std::size_t>(n), 0);
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
  {
#if defined(_OPENMP)
    const int t = omp_get_thread_num();
#else
    const int t = 0;
#endif
    const auto range = nlist::partition(n, t, nt);
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const Bond *nb = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      const int m = bond_n[static_cast<std::size_t>(i)];
      int pairs = 0;
      for (int a = 0; a < m; ++a) {
        if (nb[a].j > i) {
          ++pairs;
        }
      }
      const int trips = m * (m - 1) / 2;
      energy_n[static_cast<std::size_t>(i)] = pairs + trips;
      force_n[static_cast<std::size_t>(i)] = 2 * pairs + 3 * trips;
    }
  }
  energy_at[0] = 0;
  force_at[0] = 0;
  for (int i = 0; i < n; ++i) {
    energy_at[static_cast<std::size_t>(i) + 1] =
        energy_at[static_cast<std::size_t>(i)] + energy_n[static_cast<std::size_t>(i)];
    force_at[static_cast<std::size_t>(i) + 1] =
        force_at[static_cast<std::size_t>(i)] + force_n[static_cast<std::size_t>(i)];
  }
}

} // namespace swi
} // namespace rgpot
