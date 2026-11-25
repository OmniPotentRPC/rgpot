// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the Lennard-Jones potential methods.
 *
 * This file contains the implementation of the force and energy
 * calculation for the Lennard-Jones potential, including periodic
 * boundary condition handling.
 */

// clang-format off
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>
// clang-format on

#include "rgpot/LennardJones/LJPair.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Dual.hpp"
#include "rgpot/KeptPotential.hpp"
#include "rgpot/LennardJones/LJPairEnergy.hpp"
#include "rgpot/PairHessian.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/PairSum.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {
namespace {

struct LjNeighbours {
  nlist::Neighbours cells;
};

using KeptLjNeighbours = Kept<LjNeighbours, 4>;

// One angstrom of skin. A stored list keeps pairs out to cutoff + skin and
// the kernels drop those past the cutoff. Stream asks for no skin: the bins
// are not reused unless the coordinates match.
void updateScratchCells(LjNeighbours &scratch, const ForceInput &in,
                        double cutoff, nlist::Mode mode) {
  const double skin = mode == nlist::Mode::Stream ? 0.0 : 1.0;
  const std::array<bool, 3> periodic{{true, true, true}};
  nlist::updateCells(scratch.cells, in.pos, static_cast<std::size_t>(in.nAtoms),
                     in.box, cutoff, skin, nlist::Image::Round, mode, periodic);
}

// A destructor releases only its own thread's slots. Keyed on the
// address, a potential built where another died would find that one's
// list on every other thread.
KeptKey ljNeighbourKey(std::uint64_t generation, double cutoff,
                       nlist::Mode mode) noexcept {
  KeptKey key;
  key.i64(static_cast<std::int64_t>(generation))
      .f64(cutoff)
      .i64(static_cast<std::int64_t>(mode));
  return key;
}

/// Each half-list pair of @a in once, from the calling thread's kept list,
/// as @c visit(i, j, dx, dy, dz, r2) with @c i < j.
template <typename Visit>
void visitLjPairs(std::uint64_t generation, double cutoff, nlist::Mode mode,
                  const ForceInput &in, Visit &&visit) {
  KeptLjNeighbours::with(
      ljNeighbourKey(generation, cutoff, mode),
      [] { return std::make_unique<LjNeighbours>(); },
      [&](LjNeighbours &scratch) {
        updateScratchCells(scratch, in, cutoff, mode);
        nlist::visitHalf(scratch.cells, 0, static_cast<long>(in.nAtoms), visit);
      });
}

struct LjRadial {
  double d1; //!< phi'(r).
  double d2; //!< phi''(r).
};

/// Radial derivatives of the shifted pair term, or of the switched one when
/// @a sw is set.
LjRadial ljRadial(double r2, double r, double inv_r, double psi2, double fourU0,
                  const QuinticSwitch *sw) noexcept {
  const double sr2 = psi2 / r2;
  const double a = sr2 * sr2 * sr2;
  double d1 = fourU0 * inv_r * (6.0 * a - 12.0 * a * a);
  double d2 = fourU0 * inv_r * inv_r * (156.0 * a * a - 42.0 * a);
  if (sw != nullptr) {
    const double bare = fourU0 * a * (a - 1.0);
    const auto s = (*sw)(r);
    const double s2 = sw->d2sdr2(r);
    d2 = d2 * s.s + 2.0 * d1 * s.dsdr + bare * s2;
    d1 = d1 * s.s + bare * s.dsdr;
  }
  return {d1, d2};
}

/// The pair block from dual numbers: g_s and g_ss of the pair energy at r2.
pair_hessian::Block ljDualBlock(double r2, double dx, double dy, double dz,
                                const simd::LjParams &params) {
  const ad::Dual2 g = lj::pairEnergy(ad::seed2(r2), params);
  return pair_hessian::squared(g.v.d, g.d.d, dx, dy, dz);
}

bool isFullyPeriodicCell(const double *box, double volume) noexcept {
  if (box == nullptr || !(volume > 0.0) || !std::isfinite(volume)) {
    return false;
  }
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(box[k])) {
      return false;
    }
  }
  const bool sheared = box[1] != 0.0 || box[2] != 0.0 || box[3] != 0.0 ||
                       box[5] != 0.0 || box[6] != 0.0 || box[7] != 0.0;
  if (sheared) {
    return true;
  }
  return box[0] > 0.0 && box[4] > 0.0 && box[8] > 0.0;
}

// Relative tolerance for adaptive 7/15-point Gauss-Kronrod quadrature of the
// switch-window tail integral.
constexpr long double kSwitchQuadRelTol = 1e-13L;
constexpr int kSwitchQuadMaxDepth = 32;

// 7/15-point Gauss-Kronrod rule on [a, b], returning {I_15, |I_15 - I_7|}.
template <typename F>
std::pair<long double, long double> gk15Panel(F &&fn, long double a,
                                              long double b) noexcept {
  // Positive abscissae on [-1, 1]; indices 1, 3, 5 are shared with the 7-point
  // Gauss rule.
  constexpr long double kX[7] = {0.9914553711208126392068546975263285L,
                                 0.9491079123427585245261896840478513L,
                                 0.8648644233597690727897127886409262L,
                                 0.7415311855993944398638647732807884L,
                                 0.5860872354676911302941448382587296L,
                                 0.4058451513773971669066064120769615L,
                                 0.2077849550078984676006894037732449L};
  constexpr long double kW15[7] = {0.0229353220105292249637320080589700L,
                                   0.0630920926299785532907006631892043L,
                                   0.1047900103222501838398763225415180L,
                                   0.1406532597155259187451895905102379L,
                                   0.1690047266392679028265834265985503L,
                                   0.1903505780647854099132564024210137L,
                                   0.2044329400752988924141619992346491L};
  constexpr long double kW15_0 = 0.2094821410847278280129991748917143L;
  constexpr long double kW7[3] = {0.1294849661688696932706114326790820L,
                                  0.2797053914892766679014677714237796L,
                                  0.3818300505051189449503697754889751L};
  constexpr long double kW7_0 = 0.4179591836734693877551020408163265L;

  const long double mid = 0.5L * (a + b);
  const long double half = 0.5L * (b - a);
  const long double f0 = fn(mid);
  long double sum15 = kW15_0 * f0;
  long double sum7 = kW7_0 * f0;
  for (int k = 0; k < 7; ++k) {
    const long double dx = half * kX[k];
    const long double fpair = fn(mid - dx) + fn(mid + dx);
    sum15 += kW15[k] * fpair;
    if ((k & 1) == 1) {
      sum7 += kW7[k / 2] * fpair;
    }
  }
  const long double i15 = half * sum15;
  const long double i7 = half * sum7;
  return {i15, std::abs(i15 - i7)};
}

template <typename F>
long double gk15Adapt(F &&fn, long double a, long double b, long double i15,
                      long double err, long double tol, int depth) noexcept {
  if (err <= tol || depth <= 0) {
    return i15;
  }
  const long double mid = 0.5L * (a + b);
  const auto [l_val, l_err] = gk15Panel(fn, a, mid);
  const auto [r_val, r_err] = gk15Panel(fn, mid, b);
  if (l_err + r_err <= tol) {
    return l_val + r_val;
  }
  return gk15Adapt(fn, a, mid, l_val, l_err, 0.5L * tol, depth - 1) +
         gk15Adapt(fn, mid, b, r_val, r_err, 0.5L * tol, depth - 1);
}

// Integral of r^n (1 - S(r)) over the switch window [r_on, r_on + width],
// with 1 - S(u) = u^3 (10 - 15 u + 6 u^2) and u = (r - r_on) / width in [0, 1].
// For wide switches (r_on << width), (1 + (width / r_on) u)^n has a sharp
// boundary layer near u = 0. The Möbius map u = s / (s + beta (1 - s)) with
// beta = (r_on + width) / r_on = r_c / r_on maps [0, 1] onto [0, 1] and
// turns the integrand into s^3 (s^2 + 5 s y + 10 y^2) D(s)^(-n - 7) with
// y = beta (1 - s) and D(s) = s + y. For n = -10 the transformed integrand is
// a degree-8 polynomial in s (exact in one 15-point Gauss-Kronrod panel for
// any width < r_c), and adaptive G7-K15 quadrature to relative tolerance
// kSwitchQuadRelTol resolves n = -4 across both narrow and wide switches.
double switchWindowIntegral(int n, double r_on, double width) noexcept {
  const auto a = static_cast<long double>(r_on);
  const auto w = static_cast<long double>(width);
  const long double rc = a + w;
  const long double beta = rc / a;
  const int p = -n - 7;
  auto eval = [&](long double s) noexcept {
    const long double y = beta * (1.0L - s);
    const long double d = s + y;
    const long double poly = s * s * s * (s * s + 5.0L * s * y + 10.0L * y * y);
    if (p == 3) {
      return poly * (d * d * d);
    }
    if (p == -3) {
      return poly / (d * d * d);
    }
    return poly * std::pow(d, static_cast<long double>(p));
  };
  const auto [i15, err] = gk15Panel(eval, 0.0L, 1.0L);
  const long double tol = kSwitchQuadRelTol * std::abs(i15);
  const long double integral =
      gk15Adapt(eval, 0.0L, 1.0L, i15, err, tol, kSwitchQuadMaxDepth);
  const long double pre =
      (w / a) * std::pow(rc, static_cast<long double>(n + 1));
  return static_cast<double>(pre * integral);
}

// Radial moments of the part of the bare 12-6 pair term that the truncated
// sum does not carry, taken over a uniform fluid (g = 1).
// With dphi = V - phi_used (phi_used = the pair term the kernel evaluates):
//   energy  = integral of r^2 dphi dr,
//   virial  = -1/3 integral of r^3 dphi' dr over the pair force the kernel
//             does not apply.
// An unswitched cutoff leaves dphi = V beyond rc. The kernel applies no
// force at the jump, so the virial keeps the surface term rc^3 V(rc) / 3.
// A switch takes phi_used to zero continuously, so dphi is continuous and
// integration by parts makes the two moments equal.
struct LjTailMoments {
  double energy;
  double virial;
};

double computeSwitchWindowMoment(const LJConfig &c) {
  if (!c.tail_correction || c.switch_width == 0.0 || !(c.cutoff > 0.0) ||
      !(c.psi > 0.0)) {
    return 0.0;
  }
  const double sig2 = c.psi * c.psi;
  const double sig6 = sig2 * sig2 * sig2;
  const double r_on = c.cutoff - c.switch_width;
  return 4.0 * c.u0 *
         (sig6 * sig6 * switchWindowIntegral(-10, r_on, c.switch_width) -
          sig6 * switchWindowIntegral(-4, r_on, c.switch_width));
}

LjTailMoments ljTailMoments(const LJConfig &c, double cached_window) noexcept {
  const double sig2 = c.psi * c.psi;
  const double sig6 = sig2 * sig2 * sig2;
  const double rc3 = c.cutoff * c.cutoff * c.cutoff;
  const double rc6 = rc3 * rc3;
  const double rc9 = rc3 * rc6;
  const double pre = (4.0 * c.u0 * sig6) / (9.0 * rc9);
  const double beyond = pre * (sig6 - 3.0 * rc6);
  if (c.switch_width == 0.0) {
    return {beyond, 2.0 * pre * (2.0 * sig6 - 3.0 * rc6)};
  }
  return {beyond + cached_window, beyond + cached_window};
}

// Adds the continuum correction for N atoms in one cell. Every pair shares
// one (u0, psi), so the correction depends on N alone and not on atmnrs.
void applyLjTail(const ForceInput &in, const LJConfig &c, double cached_window,
                 double volume, double &energy, double strain[6],
                 const PerAtomOut *atoms) {
  if (in.nAtoms == 0 || !(c.cutoff > 0.0) || !(c.psi > 0.0) ||
      !isFullyPeriodicCell(in.box, volume)) {
    return;
  }
  const double n = static_cast<double>(in.nAtoms);
  const LjTailMoments moments = ljTailMoments(c, cached_window);
  const double scale = 2.0 * std::numbers::pi * n * n / volume;
  const double e_tail = scale * moments.energy;
  const double w_tail = scale * moments.virial;
  energy += e_tail;
  strain[0] -= w_tail;
  strain[1] -= w_tail;
  strain[2] -= w_tail;
  if (atoms != nullptr) {
    const double inv_n = 1.0 / n;
    if (atoms->energy != nullptr) {
      for (std::size_t i = 0; i < in.nAtoms; ++i) {
        atoms->energy[i] += e_tail * inv_n;
      }
    }
    if (atoms->virial != nullptr) {
      for (std::size_t i = 0; i < in.nAtoms; ++i) {
        atoms->virial[6 * i] += w_tail * inv_n;
        atoms->virial[6 * i + 1] += w_tail * inv_n;
        atoms->virial[6 * i + 2] += w_tail * inv_n;
      }
    }
  }
}

} // namespace

LJPot::LJPot(const LJConfig &c)
    : Potential(PotType::LJ), u0{c.u0}, cuttOffR{c.cutoff}, psi{c.psi},
      m_config{c} {
  if (c.switch_width != 0.0) {
    m_switch = QuinticSwitch::endingAt(c.cutoff, c.switch_width);
    if (c.tail_correction && !(c.switch_width < c.cutoff)) {
      throw std::invalid_argument("LJConfig: tail_correction with a switch "
                                  "needs switch_width < cutoff");
    }
  }
  // When tail_correction is off, cuttOffU is the pair shift V(cuttOffR) so
  // U(cuttOffR) = 0. When tail_correction is on, the pair shift is zero and
  // cuttOffU caches the precomputed switch-window tail moment.
  if (c.tail_correction) {
    cuttOffU = computeSwitchWindowMoment(c);
  } else {
    const double a = std::pow(psi / cuttOffR, 6.0);
    cuttOffU = 4.0 * u0 * a * (a - 1.0);
  }
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.f64(c.u0);
  fp.f64(c.cutoff);
  fp.f64(c.psi);
  // Hashed only when set, so unswitched configs keep their cache keys.
  if (c.switch_width != 0.0) {
    fp.f64(c.switch_width);
  }
  if (c.tail_correction) {
    fp.u64(1);
  }
  m_paramsKey = fp.h;
}

LJPot::~LJPot() {
  if (!neighbours_.hasGeneration()) {
    return;
  }
  const std::uint64_t gen = neighbours_.generation();
  for (const nlist::Mode mode :
       {nlist::Mode::Auto, nlist::Mode::Stored, nlist::Mode::Stream}) {
    KeptLjNeighbours::release(ljNeighbourKey(gen, cuttOffR, mode));
  }
}

bool LJPot::listReused() const noexcept {
  if (!neighbours_.hasGeneration()) {
    return false;
  }
  const LjNeighbours *kept =
      KeptLjNeighbours::peek(ljNeighbourKey(neighbours_.generation(), cuttOffR, mode_));
  return kept != nullptr && kept->cells.reused();
}

std::size_t LJPot::retainedPairs() const noexcept {
  if (!neighbours_.hasGeneration()) {
    return 0;
  }
  const LjNeighbours *kept =
      KeptLjNeighbours::peek(ljNeighbourKey(neighbours_.generation(), cuttOffR, mode_));
  if (kept == nullptr || kept->cells.streaming()) {
    return 0;
  }
  return kept->cells.slots();
}

/**
 * @class LJPot
 * @details
 *
 * Pairwise interactions within the cutoff radius, minimum image convention
 * on an orthogonal or sheared box. Pairs come from the shared neighbour
 * layer: a cell list, a Verlet skin, and either a stored list or a
 * streaming walk. The pair term runs during that walk. A streaming call
 * does not copy a half-list. A stored call keeps candidates out to the
 * cutoff plus the skin and drops the rest from the current distances, so
 * a later call inside half the skin reuses the candidates. nlist::sumPairs
 * adds the pairs, with the same bits on any number of threads. A nonzero
 * switch width uses the quintic switch. Cauchy stress is published from
 * the kernel's strain accumulator.
 *
 * @warning The box is assumed to be orthogonal.
 *
 */
void LJPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  compute(in, out, nullptr);
}

void LJPot::forcePerAtom(const ForceInput &in, ForceOut *out,
                         const PerAtomOut &atoms) const {
  guardForceCall(in, out, kernelNeedsCell(*this), false);
  clearPerAtom(atoms, in.nAtoms);
  compute(in, out, &atoms);
}

void LJPot::compute(const ForceInput &in, ForceOut *out,
                    const PerAtomOut *atoms) const {
  const long N = in.nAtoms;
  const double *box = in.box;
  double *F = out->F;
  double *U = &out->energy;
  double strain[6] = {};
  const double volume = cellVolume(box);
  *U = 0;
  for (long i = 0; i < N; i++) {
    F[3 * i] = 0;
    F[3 * i + 1] = 0;
    F[3 * i + 2] = 0;
  }
  if (N < 2 || cuttOffR <= 0.0) {
    if (neighbours_.hasGeneration()) {
      KeptLjNeighbours::release(ljNeighbourKey(neighbours_.generation(), cuttOffR, mode_));
    }
    if (m_config.tail_correction && N == 1 && cuttOffR > 0.0) {
      applyLjTail(in, m_config, cuttOffU, volume, *U, strain, atoms);
    }
    publishCauchyStress(out, strain, volume);
    return;
  }

  const double pair_shift = m_config.tail_correction ? 0.0 : cuttOffU;
  const simd::LjParams params =
      makeLjParams(u0, psi, pair_shift, m_config.switch_width, m_switch);
  const auto n_atoms = static_cast<std::size_t>(N);
  const double cutoff2 = cuttOffR * cuttOffR;
  // Two different atoms on the same point make the energy NaN. Skipping
  // the pair would report a finite energy of zero and hide the singularity.
  KeptLjNeighbours::with(
      ljNeighbourKey(neighbours_.generation(), cuttOffR, mode_),
      [] { return std::make_unique<LjNeighbours>(); },
      [&](LjNeighbours &scratch) {
        updateScratchCells(scratch, in, cuttOffR, mode_);
        nlist::sumPairs(
            scratch.cells, n_atoms, teamThreshold(kTeamMinAtoms), kTeamFrom, cutoff2,
            nlist::Overlap::Poison,
            [&params](std::int32_t, std::int32_t, double r2, double &pair_e,
                      double &fscale) { evalLjPair(r2, params, pair_e, fscale); },
            F, *U, strain, atoms);
      });
  if (std::isnan(*U)) {
    throw std::domain_error("LJPot force is undefined for coincident atoms");
  }
  if (m_config.tail_correction) {
    applyLjTail(in, m_config, cuttOffU, volume, *U, strain, atoms);
  }
  publishCauchyStress(out, strain, volume);
}

/**
 * @details
 *
 * With @f$a = (\psi/r)^6@f$ the bare pair term @f$V = 4 u_0 a (a - 1)@f$
 * has @f$V' = (4 u_0 / r)(6 a - 12 a^2)@f$ and
 * @f$V'' = (4 u_0 / r^2)(156 a^2 - 42 a)@f$. The shift is a constant and
 * drops out. The switch multiplies the bare term by @f$S@f$, so
 * @f$\phi' = V' S + V S'@f$ and @f$\phi'' = V'' S + 2 V' S' + V S''@f$.
 * @a v and @a hv must not overlap (`hv` is zeroed up front).
 */
void LJPot::hessianVectorImpl(const ForceInput &in, const double *v,
                              double *hv) const {
  const auto N = static_cast<long>(in.nAtoms);
  std::fill_n(hv, 3 * in.nAtoms, 0.0);
  if (N < 2 || cuttOffR <= 0.0) {
    return;
  }
  const double cutoff2 = cuttOffR * cuttOffR;
  const double psi2 = psi * psi;
  const double fourU0 = 4.0 * u0;
  const QuinticSwitch *sw = m_config.switch_width != 0.0 ? &m_switch : nullptr;
  const auto addPair = [&](std::int32_t i, std::int32_t j, double dx, double dy,
                           double dz, double r2) {
    if (!(i < j) || r2 > cutoff2) {
      return;
    }
    if (!(r2 > 0.0)) {
      throw std::domain_error(
          "LJPot Hessian-vector product is undefined for coincident atoms");
    }
    const double r = std::sqrt(r2);
    const double inv_r = 1.0 / r;
    const auto [d1, d2] = ljRadial(r2, r, inv_r, psi2, fourU0, sw);
    const double ux = dx * inv_r;
    const double uy = dy * inv_r;
    const double uz = dz * inv_r;
    const double *vi = v + 3 * static_cast<std::size_t>(i);
    const double *vj = v + 3 * static_cast<std::size_t>(j);
    const double wx = vi[0] - vj[0];
    const double wy = vi[1] - vj[1];
    const double wz = vi[2] - vj[2];
    const double radial = d1 * inv_r;
    const double along = (d2 - radial) * (ux * wx + uy * wy + uz * wz);
    const double kx = along * ux + radial * wx;
    const double ky = along * uy + radial * wy;
    const double kz = along * uz + radial * wz;
    double *hi = hv + 3 * static_cast<std::size_t>(i);
    double *hj = hv + 3 * static_cast<std::size_t>(j);
    hi[0] += kx;
    hi[1] += ky;
    hi[2] += kz;
    hj[0] -= kx;
    hj[1] -= ky;
    hj[2] -= kz;
  };
  visitLjPairs(neighbours_.generation(), cuttOffR, mode_, in, addPair);
}

/**
 * @details
 *
 * The blocks are @c pair_hessian::radial of the @f$\phi'@f$ and
 * @f$\phi''@f$ that @c hessianVectorImpl contracts with the direction.
 */
void LJPot::hessianImpl(const ForceInput &in, double *h) const {
  const std::size_t dim = 3 * in.nAtoms;
  std::fill_n(h, dim * dim, 0.0);
  if (in.nAtoms < 2 || cuttOffR <= 0.0) {
    return;
  }
  const double cutoff2 = cuttOffR * cuttOffR;
  const double psi2 = psi * psi;
  const double fourU0 = 4.0 * u0;
  const QuinticSwitch *sw = m_config.switch_width != 0.0 ? &m_switch : nullptr;
  visitLjPairs(neighbours_.generation(), cuttOffR, mode_, in,
               [&](std::int32_t i, std::int32_t j, double dx, double dy,
                   double dz, double r2) {
                 if (!(i < j) || r2 > cutoff2) {
                   return;
                 }
                 if (!(r2 > 0.0)) {
                   throw std::domain_error(
                       "LJPot Hessian is undefined for coincident atoms");
                 }
                 const double r = std::sqrt(r2);
                 const double inv_r = 1.0 / r;
                 const auto [d1, d2] = ljRadial(r2, r, inv_r, psi2, fourU0, sw);
                 pair_hessian::scatter(
                     pair_hessian::radial(d1, d2, dx, dy, dz, inv_r),
                     static_cast<std::size_t>(i), static_cast<std::size_t>(j),
                     dim, h);
               });
}

void LJPot::hessianAdImpl(const ForceInput &in, double *h) const {
  const std::size_t dim = 3 * in.nAtoms;
  std::fill_n(h, dim * dim, 0.0);
  if (in.nAtoms < 2 || cuttOffR <= 0.0) {
    return;
  }
  const double pair_shift = m_config.tail_correction ? 0.0 : cuttOffU;
  const simd::LjParams params =
      makeLjParams(u0, psi, pair_shift, m_config.switch_width, m_switch);
  const double cutoff2 = cuttOffR * cuttOffR;
  visitLjPairs(neighbours_.generation(), cuttOffR, mode_, in,
               [&](std::int32_t i, std::int32_t j, double dx, double dy,
                   double dz, double r2) {
                 if (!(i < j) || r2 > cutoff2) {
                   return;
                 }
                 if (!(r2 > 0.0)) {
                   throw std::domain_error(
                       "LJPot Hessian is undefined for coincident atoms");
                 }
                 pair_hessian::scatter(ljDualBlock(r2, dx, dy, dz, params),
                                       static_cast<std::size_t>(i),
                                       static_cast<std::size_t>(j), dim, h);
               });
}

void LJPot::hessianVectorAdImpl(const ForceInput &in, const double *v,
                                double *hv) const {
  std::fill_n(hv, 3 * in.nAtoms, 0.0);
  if (in.nAtoms < 2 || cuttOffR <= 0.0) {
    return;
  }
  const double pair_shift = m_config.tail_correction ? 0.0 : cuttOffU;
  const simd::LjParams params =
      makeLjParams(u0, psi, pair_shift, m_config.switch_width, m_switch);
  const double cutoff2 = cuttOffR * cuttOffR;
  visitLjPairs(neighbours_.generation(), cuttOffR, mode_, in,
               [&](std::int32_t i, std::int32_t j, double dx, double dy,
                   double dz, double r2) {
                 if (!(i < j) || r2 > cutoff2) {
                   return;
                 }
                 if (!(r2 > 0.0)) {
                   throw std::domain_error("LJPot Hessian-vector product is "
                                           "undefined for coincident atoms");
                 }
                 pair_hessian::apply(ljDualBlock(r2, dx, dy, dz, params),
                                     static_cast<std::size_t>(i),
                                     static_cast<std::size_t>(j), v, hv);
               });
}

void LJPot::densityStage(const HalfList &, double *) const {
  if (m_config.tail_correction) {
    throw std::invalid_argument(
        "LJPot: staged evaluation does not support tail_correction");
  }
}

void LJPot::forceStage(const HalfList &list, const double *, double *forces,
                       StageTally *tally) const {
  if (m_config.tail_correction) {
    throw std::invalid_argument(
        "LJPot: staged evaluation does not support tail_correction");
  }
  if (cuttOffR <= 0.0) {
    return;
  }
  const simd::LjParams params =
      makeLjParams(u0, psi, cuttOffU, m_config.switch_width, m_switch);
  const double cutoff2 = cuttOffR * cuttOffR;
  staged::withTally(tally, [&](auto tallied) {
    staged::forEachPair(
        list, [&](int i, int j, double dx, double dy, double dz, double r2, bool both) {
          if (r2 > cutoff2) {
            return;
          }
          if (!(r2 > 0.0)) {
            throw std::domain_error("LJPot force is undefined for coincident atoms");
          }
          double pair_e = 0.0;
          double fscale = 0.0;
          evalLjPair(r2, params, pair_e, fscale);
          staged::addPair(tallied, forces, tally, i, j, both, pair_e, fscale, dx, dy, dz);
        });
  });
}

} // namespace rgpot
