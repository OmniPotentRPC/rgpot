// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file QSCPot.cc
 * @author rgpot Developers
 * @date 2026-11-19
 * @brief Quantum Sutton-Chen multi-element alloy potential implementation.
 */

// clang-format off
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
// clang-format on

#include "rgpot/QSC/QSCPot.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define RGPOT_QSC_INLINE inline __attribute__((always_inline))
#define RGPOT_QSC_LAMBDA_INLINE __attribute__((always_inline))
#else
#define RGPOT_QSC_INLINE inline
#define RGPOT_QSC_LAMBDA_INLINE
#endif

namespace {

constexpr double kFreeBoxLength = 1000000.0;

struct ElementSpec {
  int z = 0;
  double n = 0.0;
  double m = 0.0;
  double epsilon = 0.0;
  double c = 0.0;
  double a = 0.0;
};

struct PairSpec {
  double n = 0.0;
  double m = 0.0;
  double epsilon = 0.0;
  double a = 0.0;
};

// Default Quantum Sutton-Chen parameters. Ni and Cu rows are Table I of Qi,
// Cagin, Kimura and Goddard, Phys. Rev. B 59, 3527 (1999),
// doi:10.1103/PhysRevB.59.3527. The other rows follow the eight-metal table of
// Ozdemir Kart, Tomak, Uludogan and Cagin, J. Non-Cryst. Solids 337, 101
// (2004), doi:10.1016/j.jnoncrysol.2004.03.121, Table 1, as tabulated in eOn
// client/potentials/QSC/Parameters.h. eOn lists a = 3.5200 for Cu; Table I
// gives 3.6030, which is used here. Table I prints a different Ag row
// (4.0072 meV, c = 94.948); the table below keeps 3.9450 meV and c = 96.524.
constexpr std::array<ElementSpec, 8> kDefaultElements = {{
    {28, 10.0, 5.0, 0.0073767, 84.745, 3.5157},  // Ni
    {29, 10.0, 5.0, 0.0057921, 84.843, 3.6030},  // Cu
    {45, 13.0, 5.0, 0.0024612, 305.499, 3.7984}, // Rh
    {46, 12.0, 6.0, 0.0032864, 148.205, 3.8813}, // Pd
    {47, 11.0, 6.0, 0.0039450, 96.524, 4.0691},  // Ag
    {77, 13.0, 6.0, 0.0037674, 224.815, 3.8344}, // Ir
    {78, 11.0, 7.0, 0.0097894, 71.336, 3.9163},  // Pt
    {79, 11.0, 8.0, 0.0078052, 53.581, 4.0651},  // Au
}};

struct CompiledTable {
  std::vector<ElementSpec> elements;
  std::vector<int> z_to_slot;
  std::vector<PairSpec> pairs;
  uint64_t params_key = 0;

  [[nodiscard]] int slotOf(int z) const noexcept {
    if (z < 0 || static_cast<std::size_t>(z) >= z_to_slot.size()) {
      return -1;
    }
    return z_to_slot[static_cast<std::size_t>(z)];
  }
};

double ipow(double base, int exp) noexcept {
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

double pow_ar(double ar2, int exp) noexcept {
  double p = ipow(ar2, exp / 2);
  if ((exp % 2) != 0) {
    p *= std::sqrt(ar2);
  }
  return p;
}

/// Evaluates (a / r)^exp. Integer exponents use binary exponentiation on
/// (a/r)^2 so integer-exponent single-species calls match SuttonChenPot bit
/// for bit; fractional exponents (such as 10.5 on Cu-Ag) use std::pow.
RGPOT_QSC_INLINE double pow_qsc(double a, double r2, double inv_r2, double exp,
                                bool force_side) noexcept {
  const double flo = std::floor(exp);
  if (exp == flo && exp >= 1.0 && exp <= 1024.0) {
    const int k = static_cast<int>(exp);
    const double a2 = a * a;
    if (a2 == 1.0) {
      return pow_ar(inv_r2, k);
    }
    const double ar2 = a2 / r2;
    if (!force_side) {
      return pow_ar(ar2, k);
    }
    const double ar2_force = a2 * inv_r2;
    return ar2_force == ar2 ? pow_ar(ar2, k) : pow_ar(ar2_force, k);
  }
  return std::pow(a / std::sqrt(r2), exp);
}

double searchCutoff(const rgpot::QSCConfig &cfg) noexcept {
  return cfg.cutoff > 0.0 ? cfg.cutoff
                          : std::numeric_limits<double>::infinity();
}

struct PeriodicPair {
  int32_t i = 0;
  int32_t j = 0;
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
  double r2 = 0.0;
};

struct CellGeometry {
  double reciprocal[9]{};
  double height[3]{};
};

CellGeometry cellGeometry(const double *box) {
  double scaled[9];
  double axis_scale[3]{};
  for (int axis = 0; axis < 3; ++axis) {
    for (int k = 0; k < 3; ++k) {
      axis_scale[axis] =
          std::max(axis_scale[axis], std::abs(box[3 * axis + k]));
    }
    if (!(axis_scale[axis] > 0.0) || !std::isfinite(axis_scale[axis])) {
      throw std::invalid_argument("QSCPot: periodic cell axis is invalid");
    }
    for (int k = 0; k < 3; ++k) {
      scaled[3 * axis + k] = box[3 * axis + k] / axis_scale[axis];
    }
  }
  const double ax = scaled[0], ay = scaled[1], az = scaled[2];
  const double bx = scaled[3], by = scaled[4], bz = scaled[5];
  const double cx = scaled[6], cy = scaled[7], cz = scaled[8];
  const double bxc[3] = {by * cz - bz * cy, bz * cx - bx * cz,
                         bx * cy - by * cx};
  const double cxa[3] = {cy * az - cz * ay, cz * ax - cx * az,
                         cx * ay - cy * ax};
  const double axb[3] = {ay * bz - az * by, az * bx - ax * bz,
                         ax * by - ay * bx};
  const double det = ax * bxc[0] + ay * bxc[1] + az * bxc[2];
  if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
    throw std::invalid_argument("QSCPot: periodic cell is singular");
  }
  CellGeometry cell;
  for (int k = 0; k < 3; ++k) {
    cell.reciprocal[k] = (bxc[k] / det) / axis_scale[0];
    cell.reciprocal[3 + k] = (cxa[k] / det) / axis_scale[1];
    cell.reciprocal[6 + k] = (axb[k] / det) / axis_scale[2];
  }
  const auto norm = [](const double *v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  };
  cell.height[0] = axis_scale[0] * (std::abs(det) / norm(bxc));
  cell.height[1] = axis_scale[1] * (std::abs(det) / norm(cxa));
  cell.height[2] = axis_scale[2] * (std::abs(det) / norm(axb));
  for (double value : cell.reciprocal) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "QSCPot: periodic reciprocal cell is not representable");
    }
  }
  for (double height : cell.height) {
    if (!(height > 0.0) || std::isnan(height)) {
      throw std::invalid_argument("QSCPot: periodic cell height is invalid");
    }
  }
  return cell;
}

bool needsPeriodicImages(const rgpot::QSCConfig &cfg, const double *box,
                         double cutoff) {
  if (!cfg.periodic) {
    return false;
  }
  const CellGeometry cell = cellGeometry(box);
  return 2.0 * cutoff >=
         std::min({cell.height[0], cell.height[1], cell.height[2]});
}

std::vector<PeriodicPair> periodicPairs(const double *pos, std::size_t n,
                                        const double *box, double cutoff) {
  constexpr long long kMaxImageExtent = 1024;
  constexpr long long kMaxImageCount = 64LL * 64LL * 64LL;
  const CellGeometry cell = cellGeometry(box);
  std::array<long long, 3> reach{};
  for (int axis = 0; axis < 3; ++axis) {
    const double span = std::ceil(cutoff / cell.height[axis]);
    if (!(span >= 0.0) || !std::isfinite(span) ||
        span > static_cast<double>(kMaxImageExtent)) {
      throw std::invalid_argument("QSCPot: periodic image extent exceeds cap");
    }
    reach[axis] = static_cast<long long>(span);
  }
  long long image_count = 1;
  for (long long extent : reach) {
    const long long span = 2 * extent + 1;
    if (image_count > kMaxImageCount / span) {
      throw std::invalid_argument("QSCPot: periodic image count exceeds cap");
    }
    image_count *= span;
  }

  const double cutoff2 = cutoff * cutoff;
  std::vector<PeriodicPair> pairs;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      const double d0[3] = {pos[3 * i] - pos[3 * j],
                            pos[3 * i + 1] - pos[3 * j + 1],
                            pos[3 * i + 2] - pos[3 * j + 2]};
      std::array<long long, 3> centre{};
      for (int axis = 0; axis < 3; ++axis) {
        const double f = d0[0] * cell.reciprocal[3 * axis] +
                         d0[1] * cell.reciprocal[3 * axis + 1] +
                         d0[2] * cell.reciprocal[3 * axis + 2];
        const double lower = static_cast<double>(
            std::numeric_limits<long long>::min() + reach[axis]);
        const double upper = static_cast<double>(
            std::numeric_limits<long long>::max() - reach[axis]);
        if (!(f > lower && f < upper)) {
          throw std::invalid_argument(
              "QSCPot: periodic fractional displacement is too large");
        }
        centre[axis] = std::llround(f);
      }
      for (long long n0 = -reach[0]; n0 <= reach[0]; ++n0) {
        for (long long n1 = -reach[1]; n1 <= reach[1]; ++n1) {
          for (long long n2 = -reach[2]; n2 <= reach[2]; ++n2) {
            const long long s0 = centre[0] + n0;
            const long long s1 = centre[1] + n1;
            const long long s2 = centre[2] + n2;
            if (i == j && !(s0 > 0 || (s0 == 0 && s1 > 0) ||
                            (s0 == 0 && s1 == 0 && s2 > 0))) {
              continue;
            }
            const double dx = d0[0] - static_cast<double>(s0) * box[0] -
                              static_cast<double>(s1) * box[3] -
                              static_cast<double>(s2) * box[6];
            const double dy = d0[1] - static_cast<double>(s0) * box[1] -
                              static_cast<double>(s1) * box[4] -
                              static_cast<double>(s2) * box[7];
            const double dz = d0[2] - static_cast<double>(s0) * box[2] -
                              static_cast<double>(s1) * box[5] -
                              static_cast<double>(s2) * box[8];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > 0.0 && r2 <= cutoff2) {
              pairs.push_back({static_cast<int32_t>(i), static_cast<int32_t>(j),
                               dx, dy, dz, r2});
            }
          }
        }
      }
    }
  }
  return pairs;
}

bool allFinite(const double *arr, std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(arr[i])) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<const CompiledTable>
compileTable(const std::vector<ElementSpec> &elements,
             const rgpot::QSCConfig &cfg, uint64_t kernel_version) {
  auto out = std::make_shared<CompiledTable>();
  out->elements = elements;
  int max_z = 0;
  for (const auto &el : elements) {
    if (el.z > max_z) {
      max_z = el.z;
    }
  }
  out->z_to_slot.assign(static_cast<std::size_t>(max_z) + 1, -1);
  for (std::size_t i = 0; i < elements.size(); ++i) {
    out->z_to_slot[static_cast<std::size_t>(elements[i].z)] =
        static_cast<int>(i);
  }
  const std::size_t m = elements.size();
  out->pairs.resize(m * m);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      const ElementSpec &pa = elements[i];
      const ElementSpec &pb = elements[j];
      PairSpec p;
      if (i == j) {
        p.epsilon = pa.epsilon;
        p.a = pa.a;
        p.n = pa.n;
        p.m = pa.m;
      } else {
        p.epsilon = std::sqrt(pa.epsilon * pb.epsilon);
        p.a = 0.5 * (pa.a + pb.a);
        p.n = 0.5 * (pa.n + pb.n);
        p.m = 0.5 * (pa.m + pb.m);
      }
      out->pairs[i * m + j] = p;
    }
  }

  rgpot::Fnv1a fp;
  fp.u64(kernel_version);
  fp.f64(cfg.cutoff);
  fp.f64(cfg.verlet_skin);
  fp.u64(cfg.periodic ? 1 : 0);
  if (cfg.stress) {
    fp.u64(1);
  }
  for (const auto &el : elements) {
    fp.i64(el.z);
    fp.f64(el.n);
    fp.f64(el.m);
    fp.f64(el.epsilon);
    fp.f64(el.c);
    fp.f64(el.a);
  }
  out->params_key = fp.h;
  return out;
}

std::vector<std::size_t> mapSlots(const CompiledTable &table, const int *atmnrs,
                                  std::size_t n_atoms) {
  std::vector<std::size_t> slots(n_atoms);
  for (std::size_t i = 0; i < n_atoms; ++i) {
    const int s = table.slotOf(atmnrs[i]);
    if (s < 0) {
      throw std::invalid_argument("QSCPot: no parameters for atomic number " +
                                  std::to_string(atmnrs[i]));
    }
    slots[i] = static_cast<std::size_t>(s);
  }
  return slots;
}

const double *validateBox(const rgpot::QSCConfig &cfg, const double *box) {
  static constexpr double free_box[9] = {kFreeBoxLength, 0.0, 0.0, 0.0,
                                         kFreeBoxLength, 0.0, 0.0, 0.0,
                                         kFreeBoxLength};
  const bool finite_box = box != nullptr && allFinite(box, 9);
  if (cfg.periodic) {
    if (!finite_box) {
      throw std::invalid_argument(
          "QSCPot: periodic evaluation needs a real cell");
    }
    (void)cellGeometry(box);
    return box;
  }
  const bool box_ok = finite_box && rgpot::cellVolume(box) > 0.0;
  return box_ok ? box : free_box;
}

} // namespace

namespace rgpot {

struct QSCPot::TableState {
  mutable std::mutex mtx;
  std::vector<ElementSpec> elements;
  std::shared_ptr<const CompiledTable> compiled;

  explicit TableState(const QSCConfig &cfg)
      : elements(kDefaultElements.begin(), kDefaultElements.end()),
        compiled(compileTable(elements, cfg, kKernelVersion)) {}

  std::shared_ptr<const CompiledTable> snapshot() const {
    std::lock_guard<std::mutex> lock(mtx);
    return compiled;
  }

  TableState(const TableState &other, std::lock_guard<std::mutex> &)
      : elements(other.elements), compiled(other.compiled) {}

  std::shared_ptr<TableState> clone() const {
    std::lock_guard<std::mutex> lock(mtx);
    return std::make_shared<TableState>(*this, lock);
  }

  void update(int z, double n, double m, double epsilon, double c, double a,
              const QSCConfig &cfg) {
    std::lock_guard<std::mutex> lock(mtx);
    bool found = false;
    for (auto &el : elements) {
      if (el.z == z) {
        el = ElementSpec{z, n, m, epsilon, c, a};
        found = true;
        break;
      }
    }
    if (!found) {
      elements.push_back(ElementSpec{z, n, m, epsilon, c, a});
    }
    compiled = compileTable(elements, cfg, kKernelVersion);
  }
};

QSCPot::QSCPot(const QSCConfig &c) : Potential(PotType::QSC), m_config{c} {
  if (!std::isfinite(c.cutoff) || !std::isfinite(c.verlet_skin) ||
      !(c.verlet_skin >= 0.0)) {
    throw std::invalid_argument(
        "QSCPot: cutoff and verlet_skin must be finite and verlet_skin >= 0");
  }
  if (c.periodic && !(c.cutoff > 0.0)) {
    throw std::invalid_argument(
        "QSCPot: periodic evaluation needs a positive cutoff");
  }
  m_state = std::make_shared<TableState>(m_config);
}

QSCPot::QSCPot(const QSCPot &other)
    : Potential(other), m_config{other.m_config},
      m_state{other.m_state ? other.m_state->clone() : nullptr},
      mode_{other.mode_}, neighbours_{other.neighbours_},
      direct_images_{other.direct_images_} {}

QSCPot::QSCPot(const QSCPot &other, SharedTable)
    : Potential(other), m_config{other.m_config}, m_state{other.m_state},
      mode_{other.mode_}, neighbours_{other.neighbours_},
      direct_images_{other.direct_images_} {}

QSCPot &QSCPot::operator=(const QSCPot &other) {
  if (this != &other) {
    m_type = other.m_type;
#ifdef RGPOT_HAS_CACHE
    setCache(nullptr);
#endif
    m_config = other.m_config;
    m_state = other.m_state ? other.m_state->clone() : nullptr;
    mode_ = other.mode_;
    neighbours_ = other.neighbours_;
    direct_images_ = other.direct_images_;
  }
  return *this;
}

QSCPot QSCPot::sharedReplica() { return QSCPot(*this, SharedTable{}); }

void QSCPot::setElementParameter(int z, double n, double m, double epsilon,
                                 double c, double a) {
  if (z < 1 || !std::isfinite(n) || !std::isfinite(m) ||
      !std::isfinite(epsilon) || !std::isfinite(c) || !std::isfinite(a) ||
      !(n >= 1.0) || !(m >= 1.0) || !(epsilon > 0.0) || !(c > 0.0) ||
      !(a > 0.0)) {
    throw std::invalid_argument(
        "QSCPot: element parameters require z >= 1, n >= 1, m >= 1 and finite "
        "positive epsilon, c and a");
  }
  m_state->update(z, n, m, epsilon, c, a, m_config);
}

uint64_t QSCPot::paramsKey() const noexcept {
  return m_state->snapshot()->params_key;
}

void QSCPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  compute(in, out, nullptr);
}

void QSCPot::forcePerAtom(const ForceInput &in, ForceOut *out,
                          const PerAtomOut &atoms) const {
  clearPerAtom(atoms, in.nAtoms);
  compute(in, out, &atoms);
}

void QSCPot::compute(const ForceInput &in, ForceOut *out,
                     const PerAtomOut *atoms) const {
  if (out == nullptr) {
    throw std::invalid_argument("QSCPot: null ForceOut");
  }
  const std::size_t n_size = in.nAtoms;
  if (n_size > 0 &&
      (in.pos == nullptr || in.atmnrs == nullptr || out->F == nullptr)) {
    throw std::invalid_argument("QSCPot: null force buffer");
  }
  if (n_size > 0 && !allFinite(in.pos, 3 * n_size)) {
    throw std::invalid_argument("QSCPot: non-finite position");
  }

  double *F = out->F;
  out->energy = 0.0;
  out->variance = 0.0;
  const QSCConfig cfg = m_config;
  double strain[6] = {};
  const auto table = m_state->snapshot();
  const std::vector<std::size_t> el = mapSlots(*table, in.atmnrs, n_size);
  const double *box_use = validateBox(cfg, in.box);

  for (std::size_t k = 0; k < 3 * n_size; ++k) {
    F[k] = 0.0;
  }
  const double cutoff = searchCutoff(cfg);
  const bool all_images = needsPeriodicImages(cfg, box_use, cutoff);
  direct_images_ = all_images;
  if (n_size < 2 && !all_images) {
    if (cfg.stress) {
      publishCauchyStress(out, strain, cellVolume(in.box));
    }
    return;
  }

  const double cutoff2 = cutoff * cutoff;
  const std::array<bool, 3> periodic =
      cfg.periodic ? std::array<bool, 3>{true, true, true}
                   : std::array<bool, 3>{false, false, false};
  const double skin = (mode_ == nlist::Mode::Stream || !(cfg.cutoff > 0.0))
                          ? 0.0
                          : cfg.verlet_skin;
  std::vector<PeriodicPair> image_pairs;
  if (all_images) {
    image_pairs = periodicPairs(in.pos, n_size, box_use, cutoff);
  } else {
    nlist::updateCells(neighbours_, in.pos, n_size, box_use, cutoff, skin,
                       nlist::Image::Round, mode_, periodic);
  }
  auto visit_pairs = [&](auto &&fn) {
    if (all_images) {
      for (const PeriodicPair &pair : image_pairs) {
        fn(pair.i, pair.j, pair.dx, pair.dy, pair.dz, pair.r2);
      }
    } else {
      nlist::visitHalf(neighbours_, 0, static_cast<long>(n_size), fn);
    }
  };

  const bool stream = !all_images && neighbours_.streaming();
  const bool rows = !all_images && n_size >= kTeamMinAtoms;
  const int nt = nlist::teamSize(n_size);
  const bool by_row = rows && (nt > 1 || (stream && !neighbours_.allPairs()));

  const std::size_t n_species = table->elements.size();
  const ElementSpec *elems = table->elements.data();
  const PairSpec *pairs = table->pairs.data();
  const std::size_t *slots = el.data();
  double *atom_e = atoms != nullptr ? atoms->energy : nullptr;

  std::vector<double> rho(n_size, 0.0);
  double *rho_ptr = rho.data();
  double epair = 0.0;
  std::vector<double> row_e(rows ? n_size : 0, 0.0);
  std::vector<double> row_s(rows && cfg.stress ? 6 * n_size : 0, 0.0);
  double *e_sum = rows ? row_e.data() : &epair;
  double *s_sum = !cfg.stress ? nullptr : (rows ? row_s.data() : strain);

  bool coincident = false;

  if (!by_row) {
    auto walk_density = [&](auto stride) {
      visit_pairs([=, &coincident](int32_t i, int32_t j, double, double, double,
                                   double r2) RGPOT_QSC_LAMBDA_INLINE {
        if (r2 > cutoff2) {
          return;
        }
        if (!(r2 > 0.0)) {
          coincident = true;
          return;
        }
        const auto ui = static_cast<std::size_t>(i);
        const auto uj = static_cast<std::size_t>(j);
        const PairSpec &pij = pairs[slots[ui] * n_species + slots[uj]];
        const double inv_r2 = 1.0 / r2;
        const double phi = pow_qsc(pij.a, r2, inv_r2, pij.m, false);
        const double rep =
            pij.epsilon * pow_qsc(pij.a, r2, inv_r2, pij.n, false);
        rho_ptr[ui] += phi;
        rho_ptr[uj] += phi;
        e_sum[ui * decltype(stride)::value] += rep;
        if (atom_e != nullptr) {
          atom_e[ui] += 0.5 * rep;
          atom_e[uj] += 0.5 * rep;
        }
      });
    };
    if (rows) {
      walk_density(std::integral_constant<std::size_t, 1>{});
    } else {
      walk_density(std::integral_constant<std::size_t, 0>{});
    }
  } else {
    std::vector<unsigned char> slice_coincident(static_cast<std::size_t>(nt),
                                                0);
    unsigned char *coinc_flags = slice_coincident.data();
    neighbours_.forRowTeam(nt, [=](int t, int32_t i, int32_t j, double, double,
                                   double, double r2) RGPOT_QSC_LAMBDA_INLINE {
      if (r2 > cutoff2) {
        return;
      }
      if (!(r2 > 0.0)) {
        coinc_flags[t] = 1;
        return;
      }
      const auto ui = static_cast<std::size_t>(i);
      const auto uj = static_cast<std::size_t>(j);
      const PairSpec &pij = pairs[slots[ui] * n_species + slots[uj]];
      const double inv_r2 = 1.0 / r2;
      const double phi = pow_qsc(pij.a, r2, inv_r2, pij.m, false);
      const double rep = pij.epsilon * pow_qsc(pij.a, r2, inv_r2, pij.n, false);
      rho_ptr[ui] += phi;
      if (j > i) {
        e_sum[ui] += rep;
      }
      if (atom_e != nullptr) {
        atom_e[ui] += 0.5 * rep;
      }
    });
    for (unsigned char flag : slice_coincident) {
      if (flag != 0) {
        coincident = true;
      }
    }
  }

  if (coincident) {
    throw std::domain_error("QSCPot: coincident atoms");
  }

  std::vector<double> weight(n_size, 0.0);
  double *w_ptr = weight.data();
  double eemb = 0.0;
  for (std::size_t i = 0; i < n_size; ++i) {
    if (!(rho_ptr[i] > 0.0)) {
      continue;
    }
    const ElementSpec &ei = elems[slots[i]];
    const double ec = ei.epsilon * ei.c;
    const double root = std::sqrt(rho_ptr[i]);
    const double emb = ec * root;
    eemb -= emb;
    w_ptr[i] = ec / (2.0 * root);
    if (atom_e != nullptr) {
      atom_e[i] -= emb;
    }
  }

  auto apply_pair = [=](auto stride, int32_t i, int32_t j, double dx, double dy,
                        double dz, double r2, double *force, double *virial,
                        bool both, bool tally) RGPOT_QSC_LAMBDA_INLINE {
    if (!(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    const auto lo = static_cast<std::size_t>(std::min(i, j));
    const auto hi = static_cast<std::size_t>(std::max(i, j));
    const PairSpec &pij = pairs[slots[lo] * n_species + slots[hi]];
    const double inv_r2 = 1.0 / r2;
    const double rep_term =
        pij.n * pij.epsilon * pow_qsc(pij.a, r2, inv_r2, pij.n, true);
    const double dphi = pij.m * pow_qsc(pij.a, r2, inv_r2, pij.m, true);
    const double emb_term = dphi * (w_ptr[lo] + w_ptr[hi]);
    const double fscale = (rep_term - emb_term) * inv_r2;
    const double fx = fscale * dx;
    const double fy = fscale * dy;
    const double fz = fscale * dz;
    force[3 * i] += fx;
    force[3 * i + 1] += fy;
    force[3 * i + 2] += fz;
    if (both) {
      force[3 * j] -= fx;
      force[3 * j + 1] -= fy;
      force[3 * j + 2] -= fz;
    }
    if (virial != nullptr && tally) {
      accumulatePairStrain(virial + 6 * static_cast<std::size_t>(i) *
                                        decltype(stride)::value,
                           fscale, dx, dy, dz);
    }
    if (atoms != nullptr) {
      tallyAtomVirial(*atoms, static_cast<std::size_t>(i), 0.5, fscale, dx, dy,
                      dz);
      if (both) {
        tallyAtomVirial(*atoms, static_cast<std::size_t>(j), 0.5, fscale, dx,
                        dy, dz);
      }
    }
  };

  if (!by_row) {
    auto walk_force = [&](auto stride) {
      visit_pairs([=](int32_t i, int32_t j, double dx, double dy, double dz,
                      double r2) RGPOT_QSC_LAMBDA_INLINE {
        apply_pair(stride, i, j, dx, dy, dz, r2, F, s_sum, true, true);
      });
    };
    if (rows) {
      walk_force(std::integral_constant<std::size_t, 1>{});
    } else {
      walk_force(std::integral_constant<std::size_t, 0>{});
    }
  } else {
    neighbours_.forRowTeam(nt, [=](int, int32_t i, int32_t j, double dx,
                                   double dy, double dz,
                                   double r2) RGPOT_QSC_LAMBDA_INLINE {
      apply_pair(std::integral_constant<std::size_t, 1>{}, i, j, dx, dy, dz, r2,
                 F, s_sum, false, j > i);
    });
  }

  if (rows) {
    for (std::size_t i = 0; i < n_size; ++i) {
      epair += row_e[i];
    }
    if (cfg.stress) {
      for (std::size_t i = 0; i < n_size; ++i) {
        for (std::size_t k = 0; k < 6; ++k) {
          strain[k] += row_s[6 * i + k];
        }
      }
    }
  }

  out->energy = epair + eemb;
  if (cfg.stress) {
    publishCauchyStress(out, strain, cellVolume(in.box));
  }
}

void QSCPot::hessianVectorImpl(const ForceInput &in, const double *v,
                               double *hv) const {
  const std::size_t n_size = in.nAtoms;
  if (n_size > 0 && (in.pos == nullptr || in.atmnrs == nullptr ||
                     v == nullptr || hv == nullptr)) {
    throw std::invalid_argument("QSCPot: null HVP buffer");
  }
  if (n_size > 0 && v == hv) {
    throw std::invalid_argument("QSCPot: HVP direction and output overlap");
  }
  if (n_size > 0 &&
      (!allFinite(in.pos, 3 * n_size) || !allFinite(v, 3 * n_size))) {
    throw std::invalid_argument("QSCPot: non-finite HVP input");
  }

  const QSCConfig cfg = m_config;
  const auto table = m_state->snapshot();
  const std::vector<std::size_t> el = mapSlots(*table, in.atmnrs, n_size);
  const double *box_use = validateBox(cfg, in.box);

  for (std::size_t k = 0; k < 3 * n_size; ++k) {
    hv[k] = 0.0;
  }
  const double cutoff = searchCutoff(cfg);
  const bool all_images = needsPeriodicImages(cfg, box_use, cutoff);
  direct_images_ = all_images;
  if (n_size < 2 && !all_images) {
    return;
  }

  const double cutoff2 = cutoff * cutoff;
  const std::array<bool, 3> periodic =
      cfg.periodic ? std::array<bool, 3>{true, true, true}
                   : std::array<bool, 3>{false, false, false};
  const double skin = (mode_ == nlist::Mode::Stream || !(cfg.cutoff > 0.0))
                          ? 0.0
                          : cfg.verlet_skin;
  std::vector<PeriodicPair> image_pairs;
  if (all_images) {
    image_pairs = periodicPairs(in.pos, n_size, box_use, cutoff);
  } else {
    nlist::updateCells(neighbours_, in.pos, n_size, box_use, cutoff, skin,
                       nlist::Image::Round, mode_, periodic);
  }
  auto visit_pairs = [&](auto &&fn) {
    if (all_images) {
      for (const PeriodicPair &pair : image_pairs) {
        fn(pair.i, pair.j, pair.dx, pair.dy, pair.dz, pair.r2);
      }
    } else {
      nlist::visitHalf(neighbours_, 0, static_cast<long>(n_size), fn);
    }
  };

  const std::size_t n_species = table->elements.size();
  const ElementSpec *elems = table->elements.data();
  const PairSpec *pairs = table->pairs.data();
  const std::size_t *slots = el.data();

  std::vector<double> rho(n_size, 0.0);
  std::vector<double> drho(n_size, 0.0);
  bool coincident = false;

  visit_pairs(
      [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
        if (r2 > cutoff2) {
          return;
        }
        if (!(r2 > 0.0)) {
          coincident = true;
          return;
        }
        const auto ui = static_cast<std::size_t>(i);
        const auto uj = static_cast<std::size_t>(j);
        const PairSpec &pij = pairs[slots[ui] * n_species + slots[uj]];
        const double inv_r2 = 1.0 / r2;
        const double phi = pow_qsc(pij.a, r2, inv_r2, pij.m, false);
        rho[ui] += phi;
        rho[uj] += phi;
        const double dvx = v[3 * ui] - v[3 * uj];
        const double dvy = v[3 * ui + 1] - v[3 * uj + 1];
        const double dvz = v[3 * ui + 2] - v[3 * uj + 2];
        const double ddv = dx * dvx + dy * dvy + dz * dvz;
        const double drho_pair = (-pij.m * phi * inv_r2) * ddv;
        drho[ui] += drho_pair;
        drho[uj] += drho_pair;
      });

  if (coincident) {
    throw std::domain_error("QSCPot: coincident atoms");
  }

  std::vector<double> w(n_size, 0.0);
  std::vector<double> dw(n_size, 0.0);
  for (std::size_t i = 0; i < n_size; ++i) {
    if (rho[i] > 0.0) {
      const ElementSpec &ei = elems[slots[i]];
      const double ec = ei.epsilon * ei.c;
      w[i] = ec / (2.0 * std::sqrt(rho[i]));
      dw[i] = -w[i] * drho[i] / (2.0 * rho[i]);
    }
  }

  visit_pairs([&](int32_t i, int32_t j, double dx, double dy, double dz,
                  double r2) {
    if (!(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    const auto ui = static_cast<std::size_t>(i);
    const auto uj = static_cast<std::size_t>(j);
    const PairSpec &pij = pairs[slots[ui] * n_species + slots[uj]];
    const double inv_r2 = 1.0 / r2;
    const double vij = pij.epsilon * pow_qsc(pij.a, r2, inv_r2, pij.n, true);
    const double phi = pow_qsc(pij.a, r2, inv_r2, pij.m, true);
    const double w_sum = w[ui] + w[uj];
    const double psi_p_over_r = (-pij.n * vij + w_sum * pij.m * phi) * inv_r2;
    const double psi_pp =
        (pij.n * (pij.n + 1.0) * vij - w_sum * pij.m * (pij.m + 1.0) * phi) *
        inv_r2;
    const double dvx = v[3 * ui] - v[3 * uj];
    const double dvy = v[3 * ui + 1] - v[3 * uj + 1];
    const double dvz = v[3 * ui + 2] - v[3 * uj + 2];
    const double ddv = dx * dvx + dy * dvy + dz * dvz;
    const double dw_term = ((dw[ui] + dw[uj]) * pij.m * phi) * inv_r2;
    const double radial_coeff =
        (psi_pp - psi_p_over_r) * (ddv * inv_r2) + dw_term;
    const double hvx = psi_p_over_r * dvx + radial_coeff * dx;
    const double hvy = psi_p_over_r * dvy + radial_coeff * dy;
    const double hvz = psi_p_over_r * dvz + radial_coeff * dz;
    hv[3 * ui] += hvx;
    hv[3 * ui + 1] += hvy;
    hv[3 * ui + 2] += hvz;
    hv[3 * uj] -= hvx;
    hv[3 * uj + 1] -= hvy;
    hv[3 * uj + 2] -= hvz;
  });
}

void QSCPot::densityStage(const HalfList &list, double *densities) const {
  if (list.nAtoms > 0 && list.atmnrs == nullptr) {
    throw std::invalid_argument("QSCPot: null atomic numbers in HalfList");
  }
  const auto table = m_state->snapshot();
  const std::size_t n_species = table->elements.size();
  const PairSpec *pairs = table->pairs.data();
  const double cutoff = searchCutoff(m_config);
  const double cutoff2 = cutoff * cutoff;
  staged::forEachPair(list, [&](int i, int j, double, double, double, double r2,
                                bool both) {
    if (!(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    const int si = table->slotOf(list.atmnrs[i]);
    const int sj = table->slotOf(list.atmnrs[j]);
    if (si < 0 || sj < 0) {
      const int bad_z = si < 0 ? list.atmnrs[i] : list.atmnrs[j];
      throw std::invalid_argument("QSCPot: no parameters for atomic number " +
                                  std::to_string(bad_z));
    }
    const PairSpec &pij = pairs[static_cast<std::size_t>(si) * n_species +
                                static_cast<std::size_t>(sj)];
    const double inv_r2 = 1.0 / r2;
    const double phi = pow_qsc(pij.a, r2, inv_r2, pij.m, false);
    densities[i] += phi;
    if (both) {
      densities[j] += phi;
    }
  });
}

void QSCPot::forceStage(const HalfList &list, const double *densities,
                        double *forces, StageTally *tally) const {
  if (list.nAtoms > 0 && list.atmnrs == nullptr) {
    throw std::invalid_argument("QSCPot: null atomic numbers in HalfList");
  }
  const auto table = m_state->snapshot();
  const std::size_t n_species = table->elements.size();
  const ElementSpec *elems = table->elements.data();
  const PairSpec *pairs = table->pairs.data();
  const double cutoff = searchCutoff(m_config);
  const double cutoff2 = cutoff * cutoff;

  std::vector<double> weight(list.nAtoms, 0.0);
  for (std::size_t i = 0; i < list.nAtoms; ++i) {
    const int si = table->slotOf(list.atmnrs[i]);
    if (si >= 0 && densities[i] > 0.0) {
      const ElementSpec &ei = elems[static_cast<std::size_t>(si)];
      weight[i] = (ei.epsilon * ei.c) / (2.0 * std::sqrt(densities[i]));
    }
  }
  if (tally != nullptr) {
    for (std::size_t k = 0; k < list.nRows; ++k) {
      const int i = list.rows[k];
      const int si = table->slotOf(list.atmnrs[i]);
      if (si < 0) {
        throw std::invalid_argument("QSCPot: no parameters for atomic number " +
                                    std::to_string(list.atmnrs[i]));
      }
      if (densities[i] > 0.0) {
        const ElementSpec &ei = elems[static_cast<std::size_t>(si)];
        staged::addSite(*tally, i,
                        -(ei.epsilon * ei.c * std::sqrt(densities[i])));
      }
    }
  }
  staged::withTally(tally, [&](auto tallied) {
    staged::forEachPair(list, [&](int i, int j, double dx, double dy, double dz,
                                  double r2, bool both) {
      if (!(r2 > 0.0) || r2 > cutoff2) {
        return;
      }
      const int si_raw = table->slotOf(list.atmnrs[i]);
      const int sj_raw = table->slotOf(list.atmnrs[j]);
      if (si_raw < 0 || sj_raw < 0) {
        const int bad_z = si_raw < 0 ? list.atmnrs[i] : list.atmnrs[j];
        throw std::invalid_argument("QSCPot: no parameters for atomic number " +
                                    std::to_string(bad_z));
      }
      const auto si = static_cast<std::size_t>(si_raw);
      const auto sj = static_cast<std::size_t>(sj_raw);
      const PairSpec &pij = pairs[si * n_species + sj];
      const double inv_r2 = 1.0 / r2;
      const double vij = pij.epsilon * pow_qsc(pij.a, r2, inv_r2, pij.n, true);
      const double dphi = pij.m * pow_qsc(pij.a, r2, inv_r2, pij.m, true);
      const double wi = weight[static_cast<std::size_t>(i)];
      const double wj = weight[static_cast<std::size_t>(j)];
      const double fscale = (pij.n * vij - (wi + wj) * dphi) * inv_r2;
      staged::addPair(tallied, forces, tally, i, j, both, vij, fscale, dx, dy,
                      dz);
    });
  });
}

} // namespace rgpot

#undef RGPOT_QSC_LAMBDA_INLINE
#undef RGPOT_QSC_INLINE
