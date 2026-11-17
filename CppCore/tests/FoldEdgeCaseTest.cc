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
#include <limits>
#include <vector>

#include <catch2/catch_all.hpp>

#include "displaced_cell.hpp"

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
