// MIT License
// Copyright 2023--present rgpot developers
//
// The virial and the stress survive a Cap'n Proto round trip with the
// same conversion the RPC server applies.

#include <cmath>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/rpc/write_mechanical.hpp"
#include "rgpot/stress.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("PotentialResult round-trips the virial", "[rpc][virial]") {
  double pos[6] = {0.0, 0.0, 0.0, 1.3, 0.25, -0.1};
  int types[2] = {1, 1};
  double box[9] = {10, 0, 0, 0.2, 9, 0, 0, 0.1, 8};
  double forces[6] = {};
  rgpot::LJPot pot;
  REQUIRE(pot.caps().virial);
  rgpot::ForceOut out{forces, 0.0, 0.0, {}, 0};
  pot.forceImpl({2, pos, types, box}, &out);
  REQUIRE(out.has_virial == 1);
  REQUIRE(out.has_stress == 1);

  const double ev_to_caller = 2.5;
  const double len_to_angstrom = 1.0;
  ::capnp::MallocMessageBuilder msg;
  auto built = msg.initRoot<::PotentialResult>();
  rgpot::rpc::writeMechanical(built, out.stress, out.has_stress, out.virial,
                              out.has_virial, ev_to_caller, len_to_angstrom);
  auto words = ::capnp::messageToFlatArray(msg);
  ::capnp::FlatArrayMessageReader reader(words);
  auto got = reader.getRoot<::PotentialResult>();
  REQUIRE(got.getHasVirial());
  REQUIRE(got.getHasStress());
  REQUIRE(got.getVirial().size() == 9);
  REQUIRE(got.getStress().size() == 9);
  const double volume = rgpot::cellVolume(box);
  for (int i = 0; i < 9; ++i) {
    const double w = got.getVirial()[static_cast<unsigned int>(i)];
    const double s = got.getStress()[static_cast<unsigned int>(i)];
    REQUIRE_THAT(w, WithinAbs(out.virial[i] * ev_to_caller, 0.0));
    REQUIRE_THAT(s, WithinAbs(out.stress[i] * ev_to_caller, 0.0));
    REQUIRE_THAT(s, WithinAbs(-w / volume, 1e-12));
  }
}

TEST_CASE("An absent virial stays absent on the wire", "[rpc][virial]") {
  ::capnp::MallocMessageBuilder msg;
  auto built = msg.initRoot<::PotentialResult>();
  const double zeros[9] = {};
  rgpot::rpc::writeMechanical(built, zeros, 0, nullptr, 0, 1.0, 1.0);
  auto words = ::capnp::messageToFlatArray(msg);
  ::capnp::FlatArrayMessageReader reader(words);
  auto got = reader.getRoot<::PotentialResult>();
  REQUIRE_FALSE(got.getHasVirial());
  REQUIRE_FALSE(got.getHasStress());
  REQUIRE(got.getVirial().size() == 0);
  REQUIRE(got.getStress().size() == 0);
}
