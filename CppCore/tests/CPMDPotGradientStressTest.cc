// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <capnp/message.h>

#include <array>

#include "cpmd_stress_oracle.hpp"
#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/units.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("CPMDPot gradient fallback keeps nine stress components",
          "[cpmd][stress]") {
  ::capnp::MallocMessageBuilder msg;
  auto params = msg.initRoot<::CPMDParams>();
  params.setFunctional("BLYP");
  params.setCutOffRy(70.0);

  rgpot::CPMDPot pot(params.asReader());
  REQUIRE(pot.available());

  const double pos[3] = {0.0, 0.0, 0.0};
  const int atm = 8;
  const double box[9] = {20.0, 0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 0.0, 23.0};
  std::array<double, 3> forces{};
  rgpot::ForceOut out{};
  out.F = forces.data();
  const rgpot::ForceInput in{
      .nAtoms = 1, .pos = pos, .atmnrs = &atm, .box = box};
  pot.forceImpl(in, &out);

  const double hartree = cpmd_stress_oracle::kGradientHartree;
  REQUIRE_THAT(out.energy,
               WithinAbs(hartree * rgpot::units::HARTREE_TO_EV, 1e-12));
  REQUIRE(out.has_stress == 1);
  for (int i = 0; i < 9; ++i) {
    REQUIRE_THAT(out.stress[i],
                 WithinAbs(cpmd_stress_oracle::stressEvPerAngstrom3(i), 1e-9));
  }
  REQUIRE_THAT(forces[0],
               WithinAbs(0.02 * rgpot::units::NEG_GRAD_TO_FORCE, 1e-9));
  REQUIRE_THAT(forces[1],
               WithinAbs(0.021 * rgpot::units::NEG_GRAD_TO_FORCE, 1e-9));
  REQUIRE_THAT(forces[2],
               WithinAbs(0.022 * rgpot::units::NEG_GRAD_TO_FORCE, 1e-9));
}
