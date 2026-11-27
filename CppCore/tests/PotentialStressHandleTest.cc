// MIT License
// Copyright 2023--present rgpot developers
//
// The stress path of PotentialHandle::from_impl: a force-only call never asks
// the potential for stress and never carries one, a stress request does, and
// a potential that does not advertise stress refuses the request.

#include <array>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/potential.hpp"

using rgpot::PotentialHandle;

namespace {

// Writes a stress whenever it is asked to, whether or not the caller wanted
// one, and records what each call requested.
struct EagerStress {
  mutable int calls = 0;
  mutable int stress_requests = 0;
  bool advertises = true;

  rgpot::PotCaps caps() const {
    rgpot::PotCaps c;
    c.stress = advertises;
    return c;
  }

  void forceImpl(const rgpot::ForceInput &in, rgpot::ForceOut *out) const {
    ++calls;
    if (in.compute_stress) {
      ++stress_requests;
    }
    for (std::size_t i = 0; i < 3 * in.nAtoms; ++i) {
      out->F[i] = 0.25 * static_cast<double>(i);
    }
    out->energy = 1.5;
    for (int k = 0; k < 9; ++k) {
      out->stress[k] = 10.0 + k;
    }
    out->has_stress = 1;
  }
};

struct Dimer {
  std::vector<double> pos{0.0, 0.0, 0.0, 1.2, 0.1, -0.2};
  std::vector<int> z{18, 18};
  std::array<double, 9> box{10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0};

  rgpot::InputSpec spec() {
    return rgpot::InputSpec(pos, z, box.data());
  }
};

} // namespace

TEST_CASE("a force-only call asks for no stress and returns none",
          "[PotentialHandle][stress]") {
  EagerStress impl;
  auto handle = PotentialHandle::from_impl(impl);
  REQUIRE(handle.supports_stress());
  Dimer dimer;
  auto input = dimer.spec();
  const auto result = handle.calculate(input);
  CHECK(impl.calls == 1);
  CHECK(impl.stress_requests == 0);
  CHECK(result.energy() == 1.5);
  CHECK_FALSE(result.has_stress());
  for (int k = 0; k < 9; ++k) {
    CHECK(result.stress_data()[k] == 0.0);
  }
}

TEST_CASE("a stress request reaches a potential that advertises stress",
          "[PotentialHandle][stress]") {
  EagerStress impl;
  auto handle = PotentialHandle::from_impl(impl);
  Dimer dimer;
  auto input = dimer.spec();
  input.with_stress();
  const auto result = handle.calculate(input);
  CHECK(impl.stress_requests == 1);
  CHECK(result.has_stress());
  for (int k = 0; k < 9; ++k) {
    CHECK(result.stress_data()[k] == 10.0 + k);
  }
  const auto forces = result.forces_vec();
  REQUIRE(forces.size() == 6);
  CHECK(forces[5] == 1.25);
}

TEST_CASE("a stress request to a potential without stress is refused",
          "[PotentialHandle][stress]") {
  EagerStress impl;
  impl.advertises = false;
  auto handle = PotentialHandle::from_impl(impl);
  CHECK_FALSE(handle.supports_stress());
  Dimer dimer;
  auto input = dimer.spec();
  input.with_stress();
  try {
    (void)handle.calculate(input);
    FAIL("the request was served");
  } catch (const rgpot::Error &e) {
    CHECK(e.status() == RGPOT_INVALID_PARAMETER);
    CHECK(std::string(e.what()).find("stress") != std::string::npos);
  }
  CHECK(impl.calls == 0);

  auto force_only = dimer.spec();
  const auto result = handle.calculate(force_only);
  CHECK(impl.calls == 1);
  CHECK(impl.stress_requests == 0);
  CHECK_FALSE(result.has_stress());
}

TEST_CASE("a Lennard-Jones handle gives the stress of a direct call",
          "[PotentialHandle][stress]") {
  rgpot::LJPot lj(rgpot::LJConfig{.cutoff = 4.0});
  auto handle = PotentialHandle::from_impl(lj);
  REQUIRE(handle.supports_stress());
  Dimer dimer;

  auto plain_input = dimer.spec();
  const auto plain = handle.calculate(plain_input);
  CHECK_FALSE(plain.has_stress());

  auto stressed_input = dimer.spec();
  stressed_input.with_stress();
  const auto stressed = handle.calculate(stressed_input);
  REQUIRE(stressed.has_stress());
  CHECK(stressed.energy() == plain.energy());
  CHECK(stressed.forces_vec() == plain.forces_vec());

  std::vector<double> forces(6, 0.0);
  rgpot::ForceOut direct{forces.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceInput in{2, dimer.pos.data(), dimer.z.data(), dimer.box.data()};
  in.compute_stress = true;
  lj.forceImpl(in, &direct);
  REQUIRE(direct.has_stress == 1);
  for (int k = 0; k < 9; ++k) {
    CHECK(stressed.stress_data()[k] == direct.stress[k]);
  }
}
