// MIT License
// Copyright 2023--present rgpot developers
//
// Cells thinner than twice the cutoff. Each kernel must count a neighbour
// at every periodic image inside the cutoff, so a primitive cell gives the
// per-atom energy, forces and stress of an explicit supercell thick enough
// for the minimum image.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/fortran/FortranPots.hpp"

using Catch::Matchers::WithinAbs;

namespace {

using Box = std::array<double, 9>;

struct Cell {
  std::vector<double> pos;
  std::vector<int> types;
  Box box{};
  [[nodiscard]] std::size_t size() const { return types.size(); }
};

struct Result {
  double energy = 0.0;
  std::vector<double> forces;
  std::array<double, 9> stress{};
  int has_stress = 0;
};

template <class Pot>
Result evaluate(const Pot &pot, const Cell &cell) {
  Result r;
  r.forces.assign(3 * cell.size(), 0.0);
  rgpot::ForceOut out{r.forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl(rgpot::ForceInput{cell.size(), cell.pos.data(), cell.types.data(),
                                  cell.box.data()},
                &out);
  r.energy = out.energy;
  r.has_stress = out.has_stress;
  for (int k = 0; k < 9; ++k) {
    r.stress[static_cast<std::size_t>(k)] = out.stress[k];
  }
  return r;
}

// Copies are laid out copy-major, so atom i of the first copy is row i.
Cell repeat(const Cell &cell, int na, int nb, int nc) {
  Cell big;
  const int reps[3] = {na, nb, nc};
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < 3; ++k) {
      big.box[static_cast<std::size_t>(3 * r + k)] =
          reps[r] * cell.box[static_cast<std::size_t>(3 * r + k)];
    }
  }
  for (int a = 0; a < na; ++a) {
    for (int b = 0; b < nb; ++b) {
      for (int c = 0; c < nc; ++c) {
        for (std::size_t i = 0; i < cell.size(); ++i) {
          for (int k = 0; k < 3; ++k) {
            big.pos.push_back(cell.pos[3 * i + static_cast<std::size_t>(k)] +
                              a * cell.box[static_cast<std::size_t>(k)] +
                              b * cell.box[static_cast<std::size_t>(3 + k)] +
                              c * cell.box[static_cast<std::size_t>(6 + k)]);
          }
          big.types.push_back(cell.types[i]);
        }
      }
    }
  }
  return big;
}

double height(const Box &box, int axis) {
  const double *a = &box[static_cast<std::size_t>(3 * ((axis + 1) % 3))];
  const double *b = &box[static_cast<std::size_t>(3 * ((axis + 2) % 3))];
  const double *c = &box[static_cast<std::size_t>(3 * axis)];
  const double n[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                       a[0] * b[1] - a[1] * b[0]};
  const double area = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  return std::fabs(n[0] * c[0] + n[1] * c[1] + n[2] * c[2]) / area;
}

// Primitive fcc vectors in lower-triangular form for cubic constant `a`.
Box triclinicFcc(double a) {
  const double s2 = std::sqrt(2.0);
  const double s3 = std::sqrt(3.0);
  const double s6 = std::sqrt(6.0);
  return {a / s2, 0.0, 0.0, a / (2.0 * s2), a * s3 / (2.0 * s2), 0.0,
          a / (2.0 * s2), a / (2.0 * s6), a / s3};
}

// Two atoms in a cell doubled along the first primitive vector, the second
// pushed off its lattice site so forces and stress are not zero.
Cell rattledPair(const Box &primitive, int type) {
  Cell cell;
  cell.box = primitive;
  for (int k = 0; k < 3; ++k) {
    cell.box[static_cast<std::size_t>(k)] *= 2.0;
  }
  const double push[3] = {0.031, -0.022, 0.043};
  cell.pos = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) {
    cell.pos.push_back(primitive[static_cast<std::size_t>(k)] + push[k]);
  }
  cell.types = {type, type};
  return cell;
}

// The small cell must be thinner than twice `search` and the supercell at
// least that thick, or the check proves nothing.
void requireThinAndThick(const Cell &small, const Cell &big, double search) {
  bool thin = false;
  for (int axis = 0; axis < 3; ++axis) {
    thin = thin || height(small.box, axis) < 2.0 * search;
    REQUIRE(height(big.box, axis) >= 2.0 * search);
  }
  REQUIRE(thin);
}

template <class Pot>
void requireMatchesRepeat(const Pot &pot, const Cell &small, int na, int nb, int nc,
                          double search, bool stress, double tol) {
  const Cell big = repeat(small, na, nb, nc);
  requireThinAndThick(small, big, search);
  const Result a = evaluate(pot, small);
  const Result b = evaluate(pot, big);
  const int copies = na * nb * nc;
  REQUIRE_THAT(a.energy, WithinAbs(b.energy / copies, tol * std::fabs(a.energy) + tol));
  for (std::size_t k = 0; k < a.forces.size(); ++k) {
    REQUIRE_THAT(a.forces[k], WithinAbs(b.forces[k], tol));
  }
  if (stress) {
    REQUIRE(a.has_stress == 1);
    REQUIRE(b.has_stress == 1);
    for (std::size_t k = 0; k < 9; ++k) {
      REQUIRE_THAT(a.stress[k], WithinAbs(b.stress[k], tol));
    }
  }
}

} // namespace

TEST_CASE("LJ one-atom fcc primitive cell sums every image", "[images][lj]") {
  // Cubic constant 1.6, u0 = psi = 1, cutoff 2.5: -7.2294117627 per atom.
  const rgpot::LJPot pot{rgpot::LJConfig{.u0 = 1.0, .cutoff = 2.5, .psi = 1.0}};
  const double a = 1.6;
  Cell primitive;
  primitive.box = {0.0, a / 2, a / 2, a / 2, 0.0, a / 2, a / 2, a / 2, 0.0};
  primitive.pos = {0.0, 0.0, 0.0};
  primitive.types = {1};
  const Result one = evaluate(pot, primitive);
  REQUIRE_THAT(one.energy, WithinAbs(-7.2294117627, 1e-9));
  for (double f : one.forces) {
    REQUIRE_THAT(f, WithinAbs(0.0, 1e-12));
  }

  Cell cubic;
  cubic.box = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
  cubic.pos = {0.0, 0.0, 0.0, 0.0, a / 2, a / 2, a / 2, 0.0, a / 2, a / 2, a / 2, 0.0};
  cubic.types = {1, 1, 1, 1};
  const Cell big = repeat(cubic, 5, 5, 5);
  requireThinAndThick(primitive, big, 3.0);
  const Result many = evaluate(pot, big);
  REQUIRE_THAT(one.energy, WithinAbs(many.energy / static_cast<double>(big.size()), 1e-11));
  for (std::size_t k = 0; k < 9; ++k) {
    REQUIRE_THAT(one.stress[k], WithinAbs(many.stress[k], 1e-11));
  }
}

TEST_CASE("Triclinic primitive cells match their supercells", "[images][triclinic]") {
  const double a = 1.6;
  const Box tri = triclinicFcc(a);

  SECTION("LJ one atom in the lower-triangular fcc cell") {
    const rgpot::LJPot pot{rgpot::LJConfig{.u0 = 1.0, .cutoff = 2.5, .psi = 1.0}};
    Cell one;
    one.box = tri;
    one.pos = {0.0, 0.0, 0.0};
    one.types = {1};
    REQUIRE_THAT(evaluate(pot, one).energy, WithinAbs(-7.2294117627, 1e-9));
    requireMatchesRepeat(pot, one, 7, 7, 7, 3.0, true, 1e-10);
  }

  SECTION("LJ rattled pair") {
    const rgpot::LJPot pot{rgpot::LJConfig{.u0 = 1.0, .cutoff = 2.5, .psi = 1.0}};
    requireMatchesRepeat(pot, rattledPair(tri, 1), 4, 7, 7, 3.0, true, 1e-10);
  }

  SECTION("Morse rattled pair") {
    const rgpot::MorsePot pot{
        rgpot::MorseConfig{.De = 0.7, .a = 1.6, .re = 1.13, .cutoff = 2.5}};
    requireMatchesRepeat(pot, rattledPair(tri, 1), 4, 7, 7, 3.0, true, 1e-10);
  }
}

TEST_CASE("Sutton-Chen primitive cell matches a supercell", "[images][eam]") {
  // Nearest neighbour 1 in reduced units. The cutoff sits between the
  // fifth and sixth fcc shells, clear of the rattled distances.
  rgpot::SuttonChenConfig cfg;
  cfg.cutoff = 2.3;
  cfg.periodic = true;
  cfg.stress = true;
  const rgpot::SuttonChenPot pot{cfg};
  const Box tri = triclinicFcc(std::sqrt(2.0));
  Cell one;
  one.box = tri;
  one.pos = {0.0, 0.0, 0.0};
  one.types = {78};
  requireMatchesRepeat(pot, one, 8, 8, 8, 2.3 * 1.2, true, 1e-10);
  requireMatchesRepeat(pot, rattledPair(tri, 78), 4, 8, 8, 2.3 * 1.2, true, 1e-10);
}

TEST_CASE("EAM-Al primitive cell matches a supercell", "[images][eam]") {
  // Neither EAM-Al kernel writes a stress, so energy and forces only.
  const Box tri = triclinicFcc(4.05);
  const rgpot::fortranpots::EAMAlPot fortran;
  Cell one;
  one.box = tri;
  one.pos = {0.0, 0.0, 0.0};
  one.types = {13};
  requireMatchesRepeat(fortran, one, 8, 8, 8, 7.1 * 1.2, false, 1e-9);
  requireMatchesRepeat(fortran, rattledPair(tri, 13), 4, 8, 8, 7.1 * 1.2, false, 1e-9);

  const rgpot::EAMPot cell;
  const double a = 4.05;
  Cell cubic;
  cubic.box = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
  cubic.pos = {0.0, 0.0, 0.0, 0.03, a / 2 - 0.02, a / 2 + 0.04, a / 2, 0.0, a / 2,
               a / 2, a / 2, 0.0};
  cubic.types = {13, 13, 13, 13};
  requireMatchesRepeat(cell, cubic, 4, 4, 4, 5.555 * 1.2, false, 1e-9);
}
