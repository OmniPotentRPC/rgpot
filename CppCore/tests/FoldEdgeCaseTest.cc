// MIT License
// Copyright 2023--present rgpot developers

// Minimum-image fold edge cases, seen through the Tersoff,
// Stillinger-Weber and EAM kernels: atoms on a cell face, pairs exactly half
// a box apart, and a cell written with sheared rows.
//
// Two descriptions of one periodic system agree to rounding and not bit for
// bit, because a different image moves a coordinate by a multiple of the
// box. A shift of a few tens of angstrom rounds a coordinate by about
// 1e-14 A, which with a bond stiffness near 1e2 eV/A^2 moves a force by
// 1e-12 eV/A. The tolerances are 1e-9 eV/A on a force and 1e-12 relative on
// the energy: a factor 1e3 above that, and far below the 1e-2 eV/A by which
// one wrong image changes a force.
//
// EAMPot reads only the diagonal of the cell, so the sheared cases cover the
// Tersoff and SW kernels.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_all.hpp>

#include "displaced_cell.hpp"
#include "rgpot/nlist/Neighbours.hpp"

using namespace displaced;

namespace {

constexpr double kEnergyTolerance = 1.0e-12;
constexpr double kForceTolerance = 1.0e-9;

struct Gap {
  double energy = 0.0;
  double force = 0.0;
};

Gap gap(const Result &a, const Result &b) {
  Gap out;
  out.energy =
      std::abs(a.energy - b.energy) / std::max(1.0, std::abs(b.energy));
  for (std::size_t k = 0; k < a.force.size(); ++k) {
    out.force = std::max(out.force, std::abs(a.force[k] - b.force[k]));
  }
  return out;
}

void requireClose(const Result &got, const Result &reference) {
  REQUIRE(got.force.size() == reference.force.size());
  const Gap g = gap(got, reference);
  INFO("energy gap " << g.energy << ", force gap " << g.force);
  REQUIRE(g.energy <= kEnergyTolerance);
  REQUIRE(g.force <= kForceTolerance);
}

bool finite(const Result &r) {
  bool all = std::isfinite(r.energy);
  for (double f : r.force) {
    all = all && std::isfinite(f);
  }
  return all;
}

// The largest sum of the forces along one axis. It is zero for any energy
// that depends only on separations.
double netForce(const Result &r) {
  double worst = 0.0;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    double sum = 0.0;
    for (std::size_t k = axis; k < r.force.size(); k += 3) {
      sum += r.force[k];
    }
    worst = std::max(worst, std::abs(sum));
  }
  return worst;
}

// The bits at every count the runtime grants equal the bits at one thread.
void requireThreadBits(Model model, const AtomMatrix &pos, const Box &box,
                       const Threads &threads) {
  threads.use(1);
  const Result serial = Engine(model)(pos, box);
  for (int count : Threads::multiThreadCounts()) {
    CAPTURE(count);
    threads.use(count);
    REQUIRE(difference(Engine(model)(pos, box), serial) == "");
  }
  threads.use(1);
}

// A 2 x 2 x 2 grid on the lattice vectors halved: every separation along a
// row of the cell is 0 or exactly half of it, a fractional coordinate of
// exactly 0.5.
Cell halfLatticeCell(const Box &box) {
  Cell cell;
  cell.box = box;
  cell.pos = AtomMatrix(8, 3);
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      cell.pos(i, k) = 0.5 * (static_cast<double>((i >> 2) & 1) * box[0][k] +
                              static_cast<double>((i >> 1) & 1) * box[1][k] +
                              static_cast<double>(i & 1) * box[2][k]);
    }
  }
  return cell;
}

// Even sites reverse among themselves and odd sites among themselves, so an
// atom keeps its parity and, in the Si-C cell, its species.
std::size_t slot(std::size_t i, std::size_t n) {
  return i % 2 == 0 ? n - 2 - i : n - i;
}

// The wide shear leaves the rows far from a reduced basis, so the image
// search visits many candidates per pair and costs seconds at 256 atoms in a
// debug build. The mild shear runs at both sizes, the wide one at 64 atoms.
struct Shearing {
  int atoms;
  std::array<int, 3> rows;
};

} // namespace

TEST_CASE("a coordinate on a cell face gives the same answer one box over",
          "[nlist][fold][displaced]") {
  const Model model = GENERATE(from_range(kModels));
  const int n = GENERATE(64, 256);
  CAPTURE(name(model), n);
  const Threads threads;
  threads.use(1);

  // The lattice sites next to a lower face of the box lie within 0.05 A of
  // it. Those coordinates are put exactly on the face.
  Cell cell = fcc(n);
  std::vector<std::size_t> face;
  for (std::size_t k = 0; k < cell.pos.size(); ++k) {
    if (std::abs(cell.pos.data()[k]) < 0.1) {
      face.push_back(k);
    }
  }
  REQUIRE(face.size() >= static_cast<std::size_t>(n) / 8);
  for (std::size_t k : face) {
    cell.pos.data()[k] = 0.0;
  }
  const Result reference = Engine(model)(cell.pos, cell.box);
  REQUIRE(finite(reference));
  const bool check_stream =
      model != Model::EamStored && model != Model::EamStream;
  if (check_stream) {
    requireClose(Engine(model, true)(cell.pos, cell.box), reference);
  }

  // The face coordinates on the other side of the cell, one to three boxes
  // away, and at the two signed zeros and the smallest steps either side of
  // them. The last pass gives each coordinate a different one.
  const double tiny = std::numeric_limits<double>::denorm_min();
  const int kinds = 7;
  auto place = [&](int kind, std::size_t axis) {
    const double length = cell.box[axis][axis];
    switch (kind) {
    case 0:
      return -0.0;
    case 1:
      return length;
    case 2:
      return -length;
    case 3:
      return 2.0 * length;
    case 4:
      return -3.0 * length;
    case 5:
      return tiny;
    default:
      return -tiny;
    }
  };
  AtomMatrix mixed = cell.pos;
  for (int pass = 0; pass <= kinds; ++pass) {
    CAPTURE(pass);
    AtomMatrix moved = cell.pos;
    for (std::size_t r = 0; r < face.size(); ++r) {
      const int kind = pass < kinds ? pass : static_cast<int>(r % kinds);
      moved.data()[face[r]] = place(kind, face[r] % 3);
    }
    requireClose(Engine(model)(moved, cell.box), reference);
    if (check_stream && (pass == 0 || pass == 4 || pass == kinds)) {
      requireClose(Engine(model, true)(moved, cell.box), reference);
    }
    mixed = moved;
  }
  requireThreadBits(model, mixed, cell.box, threads);
}

TEST_CASE("pairs exactly half a box apart fold the same either way round",
          "[nlist][fold][displaced]") {
  const Model model = GENERATE(from_range(kModels));
  CAPTURE(name(model));
  const Threads threads;
  threads.use(1);

  // Every pair is at 0 or +-0.5 of the box along each axis. Each orthorhombic
  // fold is odd in the separation: the rounding fold takes +-0.5 to -+0.5, the
  // half-box fold leaves it, and both give a reversed pair the opposite
  // vector. A pair then takes the same image whichever of its two atoms comes
  // first, and the order of the atoms cannot change the energy or the forces.
  const Cell cell =
      halfLatticeCell({{{5.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 5.0}}});
  const std::size_t n = cell.pos.rows();
  const Result result = Engine(model)(cell.pos, cell.box);
  REQUIRE(finite(result));
  REQUIRE(netForce(result) <= kForceTolerance);

  AtomMatrix reordered(n, 3);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      reordered(slot(i, n), k) = cell.pos(i, k);
    }
  }
  const Result other = Engine(model)(reordered, cell.box);
  Result restored = other;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      restored.force[3 * i + k] = other.force[3 * slot(i, n) + k];
    }
  }
  requireClose(restored, result);
  requireThreadBits(model, cell.pos, cell.box, threads);

  if (model != Model::EamStored && model != Model::EamStream) {
    const Result stream_res = Engine(model, true)(cell.pos, cell.box);
    REQUIRE(finite(stream_res));
    REQUIRE(netForce(stream_res) <= kForceTolerance);
    const Result stream_other = Engine(model, true)(reordered, cell.box);
    Result stream_restored = stream_other;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t k = 0; k < 3; ++k) {
        stream_restored.force[3 * i + k] =
            stream_other.force[3 * slot(i, n) + k];
      }
    }
    requireClose(stream_restored, stream_res);
  }
}

TEST_CASE("Neighbours half-box tie rules hold on every axis",
          "[nlist][fold][displaced]") {
  // EAMPot reads only r^2 from Image::Half stored lists and recomputes its
  // own pair vectors, while stored Tersoff/SW stay on cached shifts. Querying
  // Neighbours directly exercises both Image::Half (forDirected) and
  // Image::Round (forAtom) at exact half-box ties on each axis.
  constexpr double L = 10.0;
  constexpr double half = 0.5 * L;
  constexpr double eps = 1.0e-6;
  const double box9[9] = {L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L};

  for (int axis = 0; axis < 3; ++axis) {
    CAPTURE(axis);
    // Exact half-box: pair (0, 1) has r_1 - r_0 = +half, so r_0 - r_1 = -half.
    double pos_tie[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    pos_tie[3 + axis] = half;

    {
      rgpot::nlist::Neighbours cells;
      rgpot::nlist::Spec spec;
      spec.cutoff = 6.0;
      spec.image = rgpot::nlist::Image::Half;
      spec.mode = rgpot::nlist::Mode::Stored;
      cells.update(pos_tie, 2, box9, spec);
      int seen = 0;
      cells.forDirected(0, 2, [&](std::int32_t i, std::int32_t j, double dx,
                                  double dy, double dz, double) {
        const double v[3] = {dx, dy, dz};
        const double expect = (i == 0 && j == 1) ? -half : half;
        REQUIRE(v[axis] == expect);
        ++seen;
      });
      REQUIRE(seen == 2);
    }

    // One ULP-scale step past +half must wrap across the boundary.
    double pos_past[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    pos_past[3 + axis] = half + eps;
    {
      rgpot::nlist::Neighbours cells;
      rgpot::nlist::Spec spec;
      spec.cutoff = 6.0;
      spec.image = rgpot::nlist::Image::Half;
      spec.mode = rgpot::nlist::Mode::Stored;
      cells.update(pos_past, 2, box9, spec);
      int seen = 0;
      cells.forDirected(0, 2, [&](std::int32_t i, std::int32_t j, double dx,
                                  double dy, double dz, double) {
        const double v[3] = {dx, dy, dz};
        const double expect =
            (i == 0 && j == 1) ? (half - eps) : (-half + eps);
        REQUIRE(std::abs(v[axis] - expect) <= 1.0e-12);
        ++seen;
      });
      REQUIRE(seen == 2);
    }

    // Image::Round via forAtom hits Neighbours::fold directly at +-half and +-1.5*L.
    double pos_round[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    pos_round[3 + axis] = 1.5 * L;
    {
      rgpot::nlist::Neighbours cells;
      rgpot::nlist::Spec spec;
      spec.cutoff = 6.0;
      spec.image = rgpot::nlist::Image::Round;
      spec.mode = rgpot::nlist::Mode::Stream;
      cells.update(pos_round, 2, box9, spec);
      int seen = 0;
      cells.forAtom(0, [&](std::int32_t, std::int32_t, double dx, double dy,
                           double dz, double) {
        const double v[3] = {dx, dy, dz};
        // dx = r_0 - r_1 = -1.5 * L -> trunc(-1.5 - 0.5) = -2 -> +0.5 * L
        REQUIRE(v[axis] == half);
        ++seen;
      });
      cells.forAtom(1, [&](std::int32_t, std::int32_t, double dx, double dy,
                           double dz, double) {
        const double v[3] = {dx, dy, dz};
        // dx = r_1 - r_0 = +1.5 * L -> trunc(+1.5 + 0.5) = +2 -> -0.5 * L
        REQUIRE(v[axis] == -half);
        ++seen;
      });
      REQUIRE(seen == 2);
    }
  }
}

TEST_CASE("tile bounds select the half-box pair", "[nlist][fold][displaced]") {
  // Two bins on a side. Twelve atoms sit on one side of the cut and twelve
  // on the other, so the stencil has a tile pair whose bounds reach half a
  // box and tile pairs that do not. The cross-cut pair is exactly half a
  // box and has to keep that sign; the rounded image would flip it.
  constexpr double L = 12.0;
  constexpr double half = 0.5 * L;
  constexpr int n = 24;
  double pos[n * 3];
  for (int i = 0; i < n; ++i) {
    pos[3 * i] = i < n / 2 ? 0.0 : half;
    pos[3 * i + 1] = 0.0;
    pos[3 * i + 2] = 0.0;
  }
  pos[3 * 1] = 1.0;
  const double box9[9] = {L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L};
  rgpot::nlist::Neighbours cells;
  rgpot::nlist::Spec spec;
  spec.cutoff = 6.0;
  spec.skin = 0.0;
  spec.image = rgpot::nlist::Image::Half;
  spec.mode = rgpot::nlist::Mode::Stream;
  cells.update(pos, static_cast<std::size_t>(n), box9, spec);

  int tie = 0;
  int inside = 0;
  cells.forDirected(0, n, [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                              double) {
    if (i == 0 && j == n / 2) {
      REQUIRE(dx == -half);
      REQUIRE(dy == 0.0);
      REQUIRE(dz == 0.0);
      ++tie;
    }
    if (i == 0 && j == 1) {
      REQUIRE(dx == -1.0);
      REQUIRE(dy == 0.0);
      REQUIRE(dz == 0.0);
      ++inside;
    }
  });
  REQUIRE(tie == 1);
  REQUIRE(inside == 1);
}

TEST_CASE("a sheared description of the cell gives the same answer",
          "[nlist][fold][displaced]") {
  const Model model =
      GENERATE(Model::TersoffSi, Model::TersoffSiC, Model::StillingerWeber);
  const Shearing shearing =
      GENERATE(Shearing{64, {1, 0, 1}}, Shearing{64, {3, -2, 5}},
               Shearing{256, {1, 0, 1}});
  CAPTURE(name(model), shearing.atoms, shearing.rows);
  const Threads threads;
  threads.use(1);

  // The same atoms in a cell written with sheared rows are the same periodic
  // system. A second pass also moves each atom by lattice vectors of the
  // sheared cell, which the fold has to undo.
  const Cell plain = fcc(shearing.atoms);
  const Result reference = Engine(model)(plain.pos, plain.box);
  const Cell skew =
      sheared(plain, shearing.rows[0], shearing.rows[1], shearing.rows[2]);
  requireClose(Engine(model)(skew.pos, skew.box), reference);

  const AtomMatrix moved = shiftedByLattice(skew);
  requireClose(Engine(model)(moved, skew.box), reference);
  requireThreadBits(model, moved, skew.box, threads);
}

TEST_CASE("pairs exactly half a lattice vector apart in a sheared cell",
          "[nlist][fold][displaced]") {
  const Model model =
      GENERATE(Model::TersoffSi, Model::TersoffSiC, Model::StillingerWeber);
  const Box box =
      GENERATE(Box{{{5.0, 0.0, 0.0}, {2.5, 5.0, 0.0}, {0.0, 2.5, 5.0}}},
               Box{{{5.0, 0.0, 0.0}, {5.0, 5.0, 0.0}, {-5.0, 5.0, 5.0}}});
  CAPTURE(name(model), box[1][0], box[2][0]);
  const Threads threads;
  threads.use(1);

  // Two images of a pair can be equally close. The search keeps the first
  // it meets, so which one a pair takes can depend on the order of its two
  // atoms, and with it the force. Only what holds for either choice is
  // checked: finite values, no net force, and the same bits at every count.
  const Cell cell = halfLatticeCell(box);
  const Result result = Engine(model)(cell.pos, cell.box);
  REQUIRE(finite(result));
  REQUIRE(netForce(result) <= kForceTolerance);
  requireThreadBits(model, cell.pos, cell.box, threads);
}

TEST_CASE("the EAM energy is continuous where a pair changes image",
          "[nlist][fold][displaced]") {
  const Model model = GENERATE(Model::EamStored, Model::EamStream);
  CAPTURE(name(model));
  const Threads threads;
  threads.use(1);

  // The 64 atom cell is shorter than twice the cutoff, so some pair inside
  // the cutoff sits close to half a box. Its two images are at the same
  // distance there, so the energy has no step across the switch, while the
  // force on the pair turns round. The two cells differ by 2e-9 A of one
  // coordinate, either side of the switch.
  const Cell cell = fcc(64);
  const Tie tie = nearestTie(cell, 6.0);
  REQUIRE(tie.margin < 0.1);
  const double delta = 1.0e-9;
  REQUIRE(tie.margin > 10.0 * delta);
  const double length = cell.box[tie.axis][tie.axis];
  double apart = cell.pos(tie.second, tie.axis) - cell.pos(tie.first, tie.axis);
  apart -= length * std::floor(apart / length + 0.5);
  const double direction = std::copysign(1.0, apart);

  AtomMatrix before = cell.pos;
  AtomMatrix after = cell.pos;
  before(tie.second, tie.axis) += direction * (tie.margin - delta);
  after(tie.second, tie.axis) += direction * (tie.margin + delta);
  const Result a = Engine(model)(before, cell.box);
  const Result b = Engine(model)(after, cell.box);
  INFO("energies " << a.energy << " and " << b.energy);
  REQUIRE(std::abs(a.energy - b.energy) <= 1.0e-6);

  const std::size_t hit = 3 * tie.second + tie.axis;
  INFO("force on the moved atom " << a.force[hit] << " and " << b.force[hit]);
  REQUIRE(std::abs(a.force[hit] - b.force[hit]) > 1.0e-4);
}
