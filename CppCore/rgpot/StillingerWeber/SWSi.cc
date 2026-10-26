// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file SWSi.cc
 * @author rgpot Developers
 * @date 2026-10-26
 * @brief Stillinger-Weber silicon kernel, 1985 parameters.
 */

#include "rgpot/StillingerWeber/SWSi.hpp"

#include "rgpot/ParamHash.hpp"
#include "rgpot/nlist/HalfShell.hpp"
#include "rgpot/nlist/MinimumImage.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace rgpot {
namespace {

struct Bond {
  int j;
  double x, y, z, r, gexp, dgdr;
};

struct Scratch {
  std::vector<std::vector<Bond>> rows;
};

Scratch &scratch() {
  thread_local Scratch s;
  return s;
}

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
  // The factor exp(1/(u-a)) underflows to zero before u reaches a.
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
  const double dgdu =
      -StillingerWeberSi::kP * StillingerWeberSi::kB * u_m4 * inv_u;
  const double dfdu = StillingerWeberSi::kA * es * (dgdu - g * s * s);
  const double dedr =
      StillingerWeberSi::kEpsilon * dfdu / StillingerWeberSi::kSigma;
  const double gexp = std::exp(StillingerWeberSi::kGamma * s);
  const double dgdr =
      gexp * (-StillingerWeberSi::kGamma * s * s) / StillingerWeberSi::kSigma;
  return {true, f2, dedr, gexp, dgdr};
}

bool orthogonal(const double *box, double &lx, double &ly, double &lz) {
  if (box[1] != 0.0 || box[2] != 0.0 || box[3] != 0.0 || box[5] != 0.0 ||
      box[6] != 0.0 || box[7] != 0.0) {
    return false;
  }
  lx = box[0];
  ly = box[4];
  lz = box[8];
  return lx > 0.0 && ly > 0.0 && lz > 0.0 && std::isfinite(lx) &&
         std::isfinite(ly) && std::isfinite(lz);
}

void foldOrtho(double &dx, double &dy, double &dz, double lx, double ly,
               double lz) {
  // std::round is the half-away tie of MinimumImage's Babai step, so an
  // orthogonal cell takes the same image as the general fold.
  dx -= lx * std::round(dx / lx);
  dy -= ly * std::round(dy / ly);
  dz -= lz * std::round(dz / lz);
}

void resetRows(Scratch &s, std::size_t n) {
  if (s.rows.size() < n) {
    s.rows.resize(n);
  }
  for (std::size_t i = 0; i < n; ++i) {
    s.rows[i].clear();
    if (s.rows[i].capacity() < 32) {
      s.rows[i].reserve(32);
    }
  }
}

} // namespace

StillingerWeberSi::StillingerWeberSi() : Potential(PotType::SWSi1985) {
  Fnv1a fp;
  fp.u64(1);
  fp.f64(kEpsilon);
  fp.f64(kSigma);
  fp.f64(kA);
  fp.f64(kB);
  fp.f64(kP);
  fp.f64(kCut);
  fp.f64(kLambda);
  fp.f64(kGamma);
  m_paramsKey = fp.h;
}

void StillingerWeberSi::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (out == nullptr) {
    throw std::invalid_argument("StillingerWeberSi: null ForceOut");
  }
  out->variance = 0.0;
  out->has_stress = 0;
  for (double &s : out->stress) {
    s = 0.0;
  }
  if (in.nAtoms > 0 &&
      (in.pos == nullptr || in.box == nullptr || out->F == nullptr)) {
    throw std::invalid_argument("StillingerWeberSi: null force buffer");
  }
  evaluate(in.pos, in.box, in.nAtoms, &out->energy, out->F, 1.0, nullptr);
}

void StillingerWeberSi::energyAndGradient(
    const double *positions, const int *atomic_numbers, const double *box,
    std::size_t n_atoms, double *energy, double *gradient,
    nlist::NeighborList *neighbors) const {
  (void)atomic_numbers;
  if (energy == nullptr ||
      (n_atoms > 0 &&
       (positions == nullptr || box == nullptr || gradient == nullptr))) {
    throw std::invalid_argument("StillingerWeberSi: null energy buffer");
  }
  // Gradient is dE/dx, the negation of the force written by forceImpl.
  evaluate(positions, box, n_atoms, energy, gradient, -1.0, neighbors);
}

void StillingerWeberSi::energyAndGradientBatch(
    std::size_t n_config, const double *positions, const int *atomic_numbers,
    const double *box, std::size_t n_atoms, double *energies,
    double *gradients) const {
  if (n_config == 0) {
    return;
  }
  if (energies == nullptr ||
      (n_atoms > 0 &&
       (positions == nullptr || box == nullptr || gradients == nullptr))) {
    throw std::invalid_argument("StillingerWeberSi: null batch buffer");
  }
  const std::size_t stride = n_atoms * 3;
  for (std::size_t c = 0; c < n_config; ++c) {
    energyAndGradient(positions + c * stride, atomic_numbers, box, n_atoms,
                      energies + c, gradients + c * stride);
  }
}

void StillingerWeberSi::evaluate(const double *positions, const double *box,
                                 std::size_t n_atoms, double *energy,
                                 double *acc, double scale,
                                 nlist::NeighborList *neighbors) const {
  *energy = 0.0;
  const std::size_t n3 = n_atoms * 3;
  for (std::size_t i = 0; i < n3; ++i) {
    acc[i] = 0.0;
  }
  if (n_atoms < 2) {
    return;
  }

  double lx = 0.0;
  double ly = 0.0;
  double lz = 0.0;
  const bool ortho = orthogonal(box, lx, ly, lz);
  nlist::MinimumImage image;
  if (!ortho) {
    image = nlist::MinimumImage(box, {true, true, true});
  }

  Scratch &sc = scratch();
  resetRows(sc, n_atoms);

  const double cut = kCut * kSigma;
  const double cut2 = cut * cut;

  auto consider = [&](int i, int j, double dx, double dy, double dz) {
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
      throw std::invalid_argument("StillingerWeberSi: non-finite position");
    }
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (!(r2 > 0.0) || r2 >= cut2) {
      return;
    }
    const double r = std::sqrt(r2);
    const PairTerm t = pairTerm(r);
    if (!t.live) {
      return;
    }
    *energy += kEpsilon * t.f2;
    const double inv_r = 1.0 / r;
    const double sx = scale * t.dedr * dx * inv_r;
    const double sy = scale * t.dedr * dy * inv_r;
    const double sz = scale * t.dedr * dz * inv_r;
    acc[3 * i] += sx;
    acc[3 * i + 1] += sy;
    acc[3 * i + 2] += sz;
    acc[3 * j] -= sx;
    acc[3 * j + 1] -= sy;
    acc[3 * j + 2] -= sz;
    sc.rows[static_cast<std::size_t>(i)].push_back(
        Bond{j, dx, dy, dz, r, t.gexp, t.dgdr});
    sc.rows[static_cast<std::size_t>(j)].push_back(
        Bond{i, -dx, -dy, -dz, r, t.gexp, t.dgdr});
  };

  const nlist::NeighborList &pairs =
      nlist::orderedPairs(positions, box, n_atoms, cut, neighbors);
  if (pairs.use_stored) {
    for (std::size_t p = 0; p < pairs.lo.size(); ++p) {
      consider(pairs.lo[p], pairs.hi[p], pairs.dx[p], pairs.dy[p], pairs.dz[p]);
    }
  } else {
    for (std::size_t p = 0; p < pairs.lo.size(); ++p) {
      const int i = pairs.lo[p];
      const int j = pairs.hi[p];
      double dx = positions[3 * j] - positions[3 * i];
      double dy = positions[3 * j + 1] - positions[3 * i + 1];
      double dz = positions[3 * j + 2] - positions[3 * i + 2];
      if (ortho) {
        foldOrtho(dx, dy, dz, lx, ly, lz);
      } else {
        image.fold(dx, dy, dz);
      }
      consider(i, j, dx, dy, dz);
    }
  }

}

} // namespace rgpot
