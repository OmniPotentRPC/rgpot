#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Displaced FCC cells, and the five force models that the thread, the
// finite-difference and the fold tests run on them.
//
// A perfect lattice repeats bond lengths, angles and coordination sums, so
// a kernel that reuses them there can look correct and fast. Every cell
// here is the FCC lattice with each coordinate moved by a fixed uniform
// draw of half-width 0.05 angstrom. The three components are independent,
// so the root-mean-square displacement of an atom is 0.05 angstrom.
//
// The draws are the raw 32-bit words of std::mt19937, which the standard
// specifies exactly, scaled by powers of two. std::uniform_real_distribution
// is not specified to give the same values in every standard library, so it
// is not used: every library builds the same cell from the same seed.

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace displaced {

using rgpot::types::AtomMatrix;
using Box = std::array<std::array<double, 3>, 3>;

constexpr std::uint32_t kSeed = 2844147;
constexpr double kHalfWidth = 0.05;
constexpr double kLattice = 4.05;

struct Cell {
  AtomMatrix pos;
  Box box{};
  std::array<int, 3> reps{};
};

// A uniform draw on [-1, 1) from one engine word. Both the division and the
// doubling are exact in binary floating point.
template <typename Engine> double unit(Engine &gen) {
  return 2.0 * (static_cast<double>(gen()) / 4294967296.0) - 1.0;
}

// Unit cells along each axis for n atoms: the most cubic split of n/4
// cells, with the longest axis first.
inline std::array<int, 3> repeats(int n) {
  const int cells = (n + 3) / 4;
  int nz = 1;
  for (int k = 1; k * k * k <= cells; ++k) {
    if (cells % k == 0) {
      nz = k;
    }
  }
  const int rest = cells / nz;
  int ny = 1;
  for (int k = 1; k * k <= rest; ++k) {
    if (rest % k == 0) {
      ny = k;
    }
  }
  return {rest / ny, ny, nz};
}

// The perfect FCC lattice cut to the first n sites, in cell order.
inline Cell lattice(int n) {
  static const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  Cell cell;
  cell.reps = repeats(n);
  cell.pos = AtomMatrix(static_cast<std::size_t>(n), 3);
  int filled = 0;
  for (int ix = 0; ix < cell.reps[0]; ++ix) {
    for (int iy = 0; iy < cell.reps[1]; ++iy) {
      for (int iz = 0; iz < cell.reps[2]; ++iz) {
        for (int b = 0; b < 4 && filled < n; ++b, ++filled) {
          const auto row = static_cast<std::size_t>(filled);
          cell.pos(row, 0) = (ix + basis[b][0]) * kLattice;
          cell.pos(row, 1) = (iy + basis[b][1]) * kLattice;
          cell.pos(row, 2) = (iz + basis[b][2]) * kLattice;
        }
      }
    }
  }
  for (std::size_t axis = 0; axis < 3; ++axis) {
    cell.box[axis][axis] = cell.reps[axis] * kLattice;
  }
  return cell;
}

// The same lattice, every coordinate moved by Uniform[-0.05, 0.05).
inline Cell fcc(int n, std::uint32_t seed = kSeed) {
  Cell cell = lattice(n);
  std::mt19937 gen(seed);
  for (std::size_t i = 0; i < cell.pos.rows(); ++i) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      cell.pos(i, axis) += kHalfWidth * unit(gen);
    }
  }
  return cell;
}

// Root-mean-square length of the displacement from the perfect lattice.
inline double rmsDisplacement(const Cell &cell) {
  const Cell perfect = lattice(static_cast<int>(cell.pos.rows()));
  double sum = 0.0;
  for (std::size_t i = 0; i < cell.pos.rows(); ++i) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double shift = cell.pos(i, axis) - perfect.pos(i, axis);
      sum += shift * shift;
    }
  }
  return std::sqrt(sum / static_cast<double>(cell.pos.rows()));
}

// The same atoms and lattice written with sheared rows. Row 1 gains a times
// row 0, and row 2 gains b times row 0 and c times row 1. The multipliers
// are integers, so the set of lattice vectors, and the periodic system, are
// unchanged.
inline Cell sheared(const Cell &cell, int a, int b, int c) {
  Cell out = cell;
  for (std::size_t k = 0; k < 3; ++k) {
    out.box[1][k] = cell.box[1][k] + a * cell.box[0][k];
    out.box[2][k] = cell.box[2][k] + b * cell.box[0][k] + c * cell.box[1][k];
  }
  return out;
}

// Every atom moved by a lattice vector of the cell. The integer multiples
// of the three rows depend only on the atom index, and some are zero.
inline AtomMatrix shiftedByLattice(const Cell &cell) {
  AtomMatrix out = cell.pos;
  for (std::size_t i = 0; i < out.rows(); ++i) {
    const double m[3] = {static_cast<double>(static_cast<int>(i % 5) - 2),
                         static_cast<double>(static_cast<int>(i / 5 % 3) - 1),
                         i % 2 == 0 ? 1.0 : -1.0};
    for (std::size_t k = 0; k < 3; ++k) {
      out(i, k) +=
          m[0] * cell.box[0][k] + m[1] * cell.box[1][k] + m[2] * cell.box[2][k];
    }
  }
  return out;
}

// The pair, inside `reach`, whose separation along one axis of an
// orthorhombic cell lies closest to half the box. Where a pair crosses that
// distance its image changes, so the energy is not smooth there. Only pairs
// whose two images are both inside `reach` can change the energy.
struct Tie {
  double margin = std::numeric_limits<double>::infinity();
  std::size_t first = 0;
  std::size_t second = 0;
  std::size_t axis = 0;
};

inline Tie nearestTie(const Cell &cell, double reach) {
  Tie best;
  const double reach2 = reach * reach;
  const std::size_t n = cell.pos.rows();
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double v[3];
      for (std::size_t k = 0; k < 3; ++k) {
        const double length = cell.box[k][k];
        const double d = cell.pos(j, k) - cell.pos(i, k);
        v[k] = d - length * std::floor(d / length + 0.5);
      }
      if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] >= reach2) {
        continue;
      }
      for (std::size_t k = 0; k < 3; ++k) {
        const double length = cell.box[k][k];
        double w[3] = {v[0], v[1], v[2]};
        w[k] = v[k] - std::copysign(length, v[k]);
        if (w[0] * w[0] + w[1] * w[1] + w[2] * w[2] >= reach2) {
          continue;
        }
        const double margin = std::abs(std::abs(v[k]) - 0.5 * length);
        if (margin < best.margin) {
          best = {margin, i, j, k};
        }
      }
    }
  }
  return best;
}

enum class Model {
  TersoffSi,
  TersoffSiC,
  StillingerWeber,
  EamStored,
  EamStream
};

inline constexpr std::array<Model, 5> kModels = {
    Model::TersoffSi, Model::TersoffSiC, Model::StillingerWeber,
    Model::EamStored, Model::EamStream};

inline const char *name(Model model) {
  switch (model) {
  case Model::TersoffSi:
    return "Tersoff Si";
  case Model::TersoffSiC:
    return "Tersoff Si-C";
  case Model::StillingerWeber:
    return "Stillinger-Weber Si";
  case Model::EamStored:
    return "EAM Al, stored list";
  case Model::EamStream:
    return "EAM Al, streamed";
  }
  return "";
}

// Silicon everywhere, silicon on even sites and carbon on odd sites,
// or aluminium, as the model needs. SW ignores the species.
inline std::vector<int> species(Model model, std::size_t n) {
  switch (model) {
  case Model::TersoffSiC: {
    std::vector<int> z(n, 14);
    for (std::size_t i = 1; i < n; i += 2) {
      z[i] = 6;
    }
    return z;
  }
  case Model::EamStored:
  case Model::EamStream:
    return std::vector<int>(n, 13);
  default:
    return std::vector<int>(n, 14);
  }
}

struct Result {
  double energy = 0.0;
  double variance = 0.0;
  std::vector<double> force;
};

// One potential object. It keeps its neighbour list between calls, so a
// later call on a nearby geometry takes the reuse path. The neighbour list
// is stored, except for Model::EamStream and when `streamed` asks for it.
class Engine {
public:
  explicit Engine(Model model) : Engine(model, model == Model::EamStream) {}

  Engine(Model model, bool streamed) : model_(model) {
    const auto mode =
        streamed ? rgpot::nlist::Mode::Stream : rgpot::nlist::Mode::Stored;
    switch (model) {
    case Model::TersoffSi:
    case Model::TersoffSiC:
      bind(std::make_shared<rgpot::TersoffSiC>(), mode);
      break;
    case Model::StillingerWeber:
      bind(std::make_shared<rgpot::StillingerWeberSi>(), mode);
      break;
    case Model::EamStored:
    case Model::EamStream: {
      auto pot = std::make_shared<rgpot::EAMPot>();
      bind(pot, mode);
      used_ = [pot] { return pot->threadsUsed(); };
      builds_ = [pot] { return pot->listBuilds(); };
      break;
    }
    }
  }

  Result operator()(const AtomMatrix &pos, const Box &box) const {
    return run_(pos, species(model_, pos.rows()), box);
  }

  // Threads the last call used, and neighbour-list builds so far. Only the
  // EAM object reports them; both are zero for the other models.
  int threadsUsed() const { return used_ ? used_() : 0; }
  std::uint64_t listBuilds() const { return builds_ ? builds_() : 0; }

private:
  template <typename Pot>
  void bind(std::shared_ptr<Pot> pot, rgpot::nlist::Mode mode) {
    pot->setNeighbourMode(mode);
    run_ = [pot](const AtomMatrix &pos, const std::vector<int> &z,
                 const Box &box) {
      auto [energy, force, variance] = (*pot)(pos, z, box);
      Result out;
      out.energy = energy;
      out.variance = variance;
      out.force.assign(force.data(), force.data() + force.size());
      return out;
    };
  }

  Model model_;
  std::function<Result(const AtomMatrix &, const std::vector<int> &,
                       const Box &)>
      run_;
  std::function<int()> used_;
  std::function<std::uint64_t()> builds_;
};

inline std::uint64_t bits(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

// Empty when the two results agree bit for bit, else the first place they
// differ.
inline std::string difference(const Result &a, const Result &b) {
  char text[200];
  if (bits(a.energy) != bits(b.energy)) {
    std::snprintf(text, sizeof text,
                  "energy %.17g (%016" PRIx64 ") against %.17g (%016" PRIx64
                  ")",
                  a.energy, bits(a.energy), b.energy, bits(b.energy));
    return text;
  }
  if (bits(a.variance) != bits(b.variance)) {
    std::snprintf(text, sizeof text, "variance %.17g against %.17g", a.variance,
                  b.variance);
    return text;
  }
  if (a.force.size() != b.force.size()) {
    std::snprintf(text, sizeof text, "%zu force components against %zu",
                  a.force.size(), b.force.size());
    return text;
  }
  for (std::size_t k = 0; k < a.force.size(); ++k) {
    if (bits(a.force[k]) != bits(b.force[k])) {
      std::snprintf(
          text, sizeof text,
          "force[%zu] %.17g (%016" PRIx64 ") against %.17g (%016" PRIx64 ")", k,
          a.force[k], bits(a.force[k]), b.force[k], bits(b.force[k]));
      return text;
    }
  }
  return "";
}

inline void setThreadEnvironment(const std::string &value, bool unset) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", unset ? "" : value.c_str());
#else
  if (unset) {
    unsetenv("OMP_NUM_THREADS");
  } else {
    setenv("OMP_NUM_THREADS", value.c_str(), 1);
  }
#endif
}

// The kernels read the requested thread count from OMP_NUM_THREADS. This
// sets it and the OpenMP default together, and puts both back on exit.
class Threads {
public:
  Threads() : had_(std::getenv("OMP_NUM_THREADS") != nullptr) {
    if (had_) {
      saved_ = std::getenv("OMP_NUM_THREADS");
    }
#if defined(_OPENMP)
    max_ = omp_get_max_threads();
    dynamic_ = omp_get_dynamic();
    omp_set_dynamic(0);
#endif
  }
  Threads(const Threads &) = delete;
  Threads &operator=(const Threads &) = delete;
  ~Threads() {
    setThreadEnvironment(saved_, !had_);
#if defined(_OPENMP)
    omp_set_num_threads(max_);
    omp_set_dynamic(dynamic_);
#endif
  }

  void use(int n) const {
    setThreadEnvironment(std::to_string(n), false);
#if defined(_OPENMP)
    omp_set_num_threads(n);
#endif
  }

  // The size of the team the runtime gives a region that asks for n threads.
  static int granted(int n) {
    int got = 1;
#if defined(_OPENMP)
#pragma omp parallel num_threads(n)
    {
#pragma omp single
      got = omp_get_num_threads();
    }
#else
    (void)n;
#endif
    return got;
  }

  // The team sizes above one that the runtime gives for requests of 2, 4 and
  // 8. A request it cuts down is replaced by the size it grants, so a runtime
  // capped at 6 threads is run at 2, 4 and 6, and one capped at 1 gives none.
  static std::vector<int> multiThreadCounts() {
    std::vector<int> counts;
    for (int n : {2, 4, 8}) {
      const int got = granted(n);
      if (got > 1 &&
          std::find(counts.begin(), counts.end(), got) == counts.end()) {
        counts.push_back(got);
      }
    }
    return counts;
  }

private:
  bool had_;
#if defined(_OPENMP)
  int max_ = 1;
  int dynamic_ = 0;
#endif
  std::string saved_;
};

} // namespace displaced
