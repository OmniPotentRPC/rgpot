// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <capnp/message.h>

#include <array>
#include <vector>

#include "cpmd_stress_oracle.hpp"
#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("CPMDPot passes serialized CPMDParams to cpmdc engine",
          "[cpmd][abi]") {
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CPMDParams>();
  p.setFunctional("BLYP");
  p.setCutOffRy(70.0);
  p.setCharge(0);
  p.setMultiplicity(1);

  rgpot::CPMDPot pot(p.asReader());
  REQUIRE(pot.available());

  rgpot::types::AtomMatrix positions(1, 3);
  positions(0, 0) = 0.0;
  positions(0, 1) = 0.0;
  positions(0, 2) = 0.0;
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 21.0, 0.0}, {0.0, 0.0, 23.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  const double cell_zz = 23.0;
  const double hartree = cpmd_stress_oracle::sessionHartree(cell_zz);
  REQUIRE_THAT(energy, WithinAbs(0.773, 1e-12));
  REQUIRE_THAT(energy, WithinAbs(hartree * rgpot::units::HARTREE_TO_EV, 1e-12));
  REQUIRE_THAT(forces(0, 0), WithinAbs(0.011, 1e-12));
  REQUIRE_THAT(forces(0, 1), WithinAbs(0.012, 1e-12));
  REQUIRE_THAT(forces(0, 2), WithinAbs(0.013, 1e-12));

  const double pos[3] = {0.0, 0.0, 0.0};
  const int atm = 8;
  const double flat_box[9] = {20.0, 0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 0.0, cell_zz};
  std::array<double, 3> raw_forces{};
  rgpot::ForceOut raw{};
  raw.F = raw_forces.data();
  const rgpot::ForceInput in{
      .nAtoms = 1, .pos = pos, .atmnrs = &atm, .box = flat_box};
  pot.forceImpl(in, &raw);
  REQUIRE_THAT(raw.energy, WithinAbs(hartree * rgpot::units::HARTREE_TO_EV, 1e-12));
  REQUIRE(raw.has_stress == 1);
  for (int i = 0; i < 9; ++i) {
    REQUIRE_THAT(raw.stress[i],
                 WithinAbs(cpmd_stress_oracle::stressEvPerAngstrom3(i), 1e-9));
  }
}
