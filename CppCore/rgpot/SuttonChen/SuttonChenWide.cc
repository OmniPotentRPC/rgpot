#include "rgpot/SuttonChen/SuttonChenPot.hpp"

#include "rgpot/nlist/OrderedWalk.hpp"
#include "rgpot/stress.hpp"

#include <cmath>
#include <vector>

namespace rgpot {
namespace {

double ipow(double base, int exp) {
  double result = 1.0;
  while (exp > 0) {
    if ((exp & 1) != 0) {
      result *= base;
    }
    base *= base;
    exp >>= 1;
  }
  return result;
}

double pow_ar(double ar2, int exp) {
  double p = ipow(ar2, exp / 2);
  if ((exp % 2) != 0) {
    p *= std::sqrt(ar2);
  }
  return p;
}

void pow_ar_both(double ar2, int n_exp, int m_exp, double &rep, double &phi) {
  if (n_exp == 10 && m_exp == 8) {
    const double s2 = ar2 * ar2;
    const double s4 = s2 * s2;
    phi = 1.0 * s4;
    rep = ar2 * s4;
    return;
  }
  rep = pow_ar(ar2, n_exp);
  phi = pow_ar(ar2, m_exp);
}

} // namespace

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline, visibility("hidden")))
#endif
int suttonChenThreads(int asked, long n_atoms, long pairs, bool listed) {
  int nt = asked;
  if (nt > 1 && nt < 4) {
    if (!listed || n_atoms < 4096 || pairs >= 50000) {
      nt = 1;
    }
  } else if (nt >= 8) {
    const long dense = n_atoms < 1024 ? 30000 : n_atoms * 400;
    const long sparse = n_atoms < 1024 ? 6000 : (n_atoms < 4096 ? 20000 : 0);
    if (!listed || n_atoms <= 128 || pairs < sparse || pairs >= dense) {
      nt = 1;
    }
  } else if (nt > 1) {
    if (!listed || n_atoms < 1024 || pairs >= n_atoms * 400 ||
        (n_atoms < 4096 && pairs < 50000)) {
      nt = 1;
    }
  }
  return nt;
}

void SuttonChenPot::gatherWide(long n_atoms, double *F, double *strain, double cutoff2,
                               double epsilon, double c_coef, double lattice, int n_exp,
                               int m_exp, int nt, double &energy) const {
  const auto n_size = static_cast<std::size_t>(n_atoms);
  std::vector<double> rho(n_size, 0.0);
  double epair = 0.0;
  std::vector<nlist::HalfPart> parts;
  nlist::gather_half(
      neighbours_, nt, n_atoms, parts,
      [&](std::int32_t, std::int32_t, double, double, double, double r2, nlist::HalfTerm &term) {
        if (!(r2 > 0.0) || r2 > cutoff2) {
          return false;
        }
        double phi = 0.0;
        double rep = 0.0;
        pow_ar_both((lattice * lattice) / r2, n_exp, m_exp, rep, phi);
        term.e = epsilon * rep;
        term.fscale = phi;
        return true;
      });
  for (const nlist::HalfPart &part : parts) {
    for (const nlist::HalfTerm &term : part.terms) {
      rho[static_cast<std::size_t>(term.i)] += term.fscale;
      rho[static_cast<std::size_t>(term.j)] += term.fscale;
      epair += term.e;
    }
  }
  std::vector<double> weight(n_size, 0.0);
  double eemb = 0.0;
  for (std::size_t i = 0; i < n_size; ++i) {
    if (!(rho[i] > 0.0)) {
      continue;
    }
    const double root = std::sqrt(rho[i]);
    eemb -= epsilon * c_coef * root;
    weight[i] = epsilon * c_coef / (2.0 * root);
  }
  parts.clear();
  nlist::gather_half(
      neighbours_, nt, n_atoms, parts,
      [&](std::int32_t i, std::int32_t j, double, double, double, double r2, nlist::HalfTerm &term) {
        if (!(r2 > 0.0) || r2 > cutoff2) {
          return false;
        }
        const double inv_r2 = 1.0 / r2;
        const double ar2 = (lattice * lattice) * inv_r2;
        double phi = 0.0;
        double rep = 0.0;
        pow_ar_both(ar2, n_exp, m_exp, rep, phi);
        const double wi = weight[static_cast<std::size_t>(i)];
        const double wj = weight[static_cast<std::size_t>(j)];
        term.e = 0;
        term.fscale = (epsilon * static_cast<double>(n_exp) * rep -
                       static_cast<double>(m_exp) * phi * (wi + wj)) *
                      inv_r2;
        return true;
      });
  double ignored = 0;
  nlist::replay_half(parts, F, ignored, [&](const nlist::HalfTerm &term) {
    if (strain != nullptr) {
      accumulatePairStrain(strain, term.fscale, term.dx, term.dy, term.dz);
    }
  });
  energy = epair + eemb;
}

} // namespace rgpot
