// MIT License — SkalaPot is NWChem DFT with dft.xc = skala-1.1

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <capnp/message.h>
#include <catch2/catch_all.hpp>

#include "rgpot/SkalaPot/SkalaPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

TEST_CASE("SkalaPot empty xc throws", "[SkalaPot]") {
  rgpot::SkalaConfig cfg;
  cfg.xc.clear();
  REQUIRE_THROWS_WITH(rgpot::SkalaPot(cfg),
                      Catch::Matchers::ContainsSubstring("xc"));
}

TEST_CASE("SkalaPot empty basis throws", "[SkalaPot]") {
  rgpot::SkalaConfig cfg;
  cfg.basis.clear();
  REQUIRE_THROWS_WITH(rgpot::SkalaPot(cfg),
                      Catch::Matchers::ContainsSubstring("basis"));
}

TEST_CASE("SkalaPot writes skala XC into NWChemParams", "[SkalaPot]") {
  rgpot::SkalaConfig cfg;
  cfg.xc = "skala-1.1";
  cfg.basis = "6-31g";
  cfg.charge = 0;
  cfg.multiplicity = 1;
  rgpot::SkalaPot pot(cfg);

  ::capnp::MallocMessageBuilder msg;
  auto out = msg.initRoot<::NWChemParams>();
  pot.getParams(out);
  REQUIRE(std::string(out.getTheory().cStr()) == "dft");
  REQUIRE(std::string(out.getBasis().cStr()) == "6-31g");
  REQUIRE(out.getCharge() == 0);
  REQUIRE(out.getMultiplicity() == 1);
  REQUIRE(out.getInputStanzas().size() >= 1);
  const auto stanza = out.getInputStanzas()[0];
  REQUIRE(stanza.getKind() == ::NWChemInputStanza::Kind::DFT);
  REQUIRE(std::string(stanza.getDft().getXc().cStr()) == "skala-1.1");
}

TEST_CASE("SkalaPot paramsKey changes with xc and charge", "[SkalaPot]") {
  rgpot::SkalaConfig cfg;
  rgpot::SkalaPot a(cfg);
  const uint64_t k0 = a.paramsKey();
  a.setChargeMultiplicity(-1, 1);
  REQUIRE(a.paramsKey() != k0);
  cfg.xc = "skala-1.0";
  rgpot::SkalaPot b(cfg);
  REQUIRE(b.paramsKey() != k0);
}

TEST_CASE("SkalaPot energy comes from nwchemc", "[SkalaPot]") {
  rgpot::SkalaPot pot;
  if (!pot.available()) {
    AtomMatrix positions{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
    const std::vector<int> z{1, 1};
    const std::array<std::array<double, 3>, 3> box{
        {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
    REQUIRE_THROWS(pot(positions, z, box));
    return;
  }
  AtomMatrix positions{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  const std::vector<int> z{1, 1};
  const std::array<std::array<double, 3>, 3> box{
      {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
  auto [energy, forces, variance] = pot(positions, z, box);
  (void)variance;
  REQUIRE(std::isfinite(energy));
  REQUIRE(forces.rows() == 2);
}
