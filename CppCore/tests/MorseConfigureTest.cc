// MIT License
// Copyright 2023--present rgpot developers
//
// PotentialConfig.morse: the wire arm round-trips, its schema defaults are
// MorseConfig's, and configure on a Morse backend rebuilds the pot. Every
// other arm, including one this build's schema does not know, is refused
// by name.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <capnp/any.h>
#include <capnp/dynamic.h>
#include <capnp/message.h>
#include <capnp/schema.h>
#include <capnp/serialize.h>
#include <catch2/catch_all.hpp>

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/rpc/MorseConfigure.hpp"
#include "rgpot/rpc/MullerBrownConfigure.hpp"
#include "rgpot/rpc/OrcaConfigure.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;
using rgpot::types::AtomMatrix;

namespace {

double pairEnergyAtRe(rgpot::MorsePot &pot) {
  const double re = pot.config().re;
  const AtomMatrix positions{{10.0, 10.0, 10.0}, {10.0 + re, 10.0, 10.0}};
  const std::vector<int> atmtypes{78, 78};
  const std::array<std::array<double, 3>, 3> box{
      {{40.0, 0.0, 0.0}, {0.0, 40.0, 0.0}, {0.0, 0.0, 40.0}}};
  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)forces;
  (void)variance;
  return energy;
}

} // namespace

TEST_CASE("PotentialConfig.morse round-trips at ordinal 10", "[morse][rpc]") {
  const auto field = capnp::Schema::from<::PotentialConfig>().getFieldByName("morse");
  REQUIRE(field.getProto().getOrdinal().getExplicit() == 10);

  capnp::MallocMessageBuilder builder;
  auto morse = builder.initRoot<::PotentialConfig>().initMorse();
  morse.setDe(1.25);
  morse.setA(1.5);
  morse.setRe(2.75);
  morse.setCutoff(8.0);
  const auto words = capnp::messageToFlatArray(builder);

  capnp::FlatArrayMessageReader reader(words.asPtr());
  const auto cfg = reader.getRoot<::PotentialConfig>();
  REQUIRE(cfg.which() == ::PotentialConfig::MORSE);
  const auto back = rgpot::rpc::morseConfigFrom(cfg.getMorse());
  REQUIRE(back.De == 1.25);
  REQUIRE(back.a == 1.5);
  REQUIRE(back.re == 2.75);
  REQUIRE(back.cutoff == 8.0);
}

TEST_CASE("MorseParams defaults are the MorseConfig defaults", "[morse][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  cfg.initMorse();
  const auto wire = rgpot::rpc::morseConfigFrom(cfg.asReader().getMorse());
  const rgpot::MorseConfig native{};
  REQUIRE(wire.De == native.De);
  REQUIRE(wire.a == native.a);
  REQUIRE(wire.re == native.re);
  REQUIRE(wire.cutoff == native.cutoff);
}

TEST_CASE("configure with morse on the Morse backend rebuilds the pot", "[morse][rpc]") {
  rgpot::MorsePot current;
  const double before = pairEnergyAtRe(current);

  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  cfg.initMorse().setDe(2.0 * current.config().De);
  auto applied = rgpot::rpc::configureMorse(cfg.asReader(), current);

  REQUIRE(applied.ok);
  REQUIRE(applied.replacement != nullptr);
  const auto &next = applied.replacement->config();
  REQUIRE(next.De == 2.0 * current.config().De);
  REQUIRE(next.a == current.config().a);
  REQUIRE(next.re == current.config().re);
  REQUIRE(next.cutoff == current.config().cutoff);
  REQUIRE(applied.replacement->paramsKey() != current.paramsKey());

  // At r = re the shifted pair energy is -De d^2 with d = 1 - exp(-a (rc - re)),
  // linear in De.
  const double after = pairEnergyAtRe(*applied.replacement);
  const double d = 1.0 - std::exp(-next.a * (next.cutoff - next.re));
  REQUIRE_THAT(after, WithinRel(-next.De * d * d, 1e-12));
  REQUIRE_THAT(after, WithinRel(2.0 * before, 1e-12));
}

TEST_CASE("configure on the Morse backend keeps the switch width", "[morse][rpc]") {
  const rgpot::MorsePot current{rgpot::MorseConfig{.switch_width = 1.0}};
  capnp::MallocMessageBuilder builder;
  builder.initRoot<::PotentialConfig>().initMorse().setCutoff(7.0);
  auto applied =
      rgpot::rpc::configureMorse(builder.getRoot<::PotentialConfig>().asReader(), current);
  REQUIRE(applied.ok);
  REQUIRE(applied.replacement->config().switch_width == 1.0);
  REQUIRE(applied.replacement->config().cutoff == 7.0);
}

TEST_CASE("configure on the Morse backend refuses a cutoff inside the switch",
          "[morse][rpc]") {
  const rgpot::MorsePot current{rgpot::MorseConfig{.switch_width = 1.0}};
  capnp::MallocMessageBuilder builder;
  builder.initRoot<::PotentialConfig>().initMorse().setCutoff(0.5);
  auto applied =
      rgpot::rpc::configureMorse(builder.getRoot<::PotentialConfig>().asReader(), current);
  REQUIRE_FALSE(applied.ok);
  REQUIRE(applied.replacement == nullptr);
  REQUIRE_THAT(applied.message, ContainsSubstring("switch width"));
}

TEST_CASE("configure on the Morse backend refuses other arms", "[morse][rpc]") {
  const rgpot::MorsePot current;

  capnp::MallocMessageBuilder none_msg;
  none_msg.initRoot<::PotentialConfig>().setNone();
  auto none = rgpot::rpc::configureMorse(none_msg.getRoot<::PotentialConfig>().asReader(),
                                         current);
  REQUIRE(none.ok);
  REQUIRE(none.replacement == nullptr);

  capnp::MallocMessageBuilder bad_msg;
  bad_msg.initRoot<::PotentialConfig>().initMorse().setDe(
      std::numeric_limits<double>::quiet_NaN());
  auto bad =
      rgpot::rpc::configureMorse(bad_msg.getRoot<::PotentialConfig>().asReader(), current);
  REQUIRE_FALSE(bad.ok);
  REQUIRE(bad.replacement == nullptr);

  // Every arm the schema declares, so an arm added by a repin is covered
  // without editing this test.
  std::size_t refused = 0;
  for (auto field : capnp::Schema::from<::PotentialConfig>().getUnionFields()) {
    const std::string name = field.getProto().getName().cStr();
    if (name == "none" || name == "morse") {
      continue;
    }
    DYNAMIC_SECTION("arm " << name) {
      capnp::MallocMessageBuilder message;
      capnp::DynamicStruct::Builder dynamic =
          capnp::toDynamic(message.initRoot<::PotentialConfig>());
      if (field.getType().isVoid()) {
        dynamic.set(field, capnp::VOID);
      } else {
        dynamic.init(field);
      }
      const auto cfg = message.getRoot<::PotentialConfig>().asReader();
      REQUIRE(rgpot::rpc::potentialConfigArm(cfg) == name);
      auto applied = rgpot::rpc::configureMorse(cfg, current);
      REQUIRE_FALSE(applied.ok);
      REQUIRE(applied.replacement == nullptr);
      REQUIRE_THAT(applied.message, ContainsSubstring("arm " + name + " not handled"));
    }
    ++refused;
  }
  // nwchem, cpmd, metatomic, lammps, xtb, tblite, uma, lj, zbl, d3, d4,
  // mopac, skala, expr and cp2k at the 190fee6 pin.
  REQUIRE(refused >= 15);
}

TEST_CASE("an arm from a newer schema is refused by its discriminant", "[morse][rpc]") {
  const auto schema = capnp::Schema::from<::PotentialConfig>();
  const std::uint16_t unknown = static_cast<std::uint16_t>(schema.getUnionFields().size() + 7);
  const auto offset = schema.getProto().getStruct().getDiscriminantOffset();

  capnp::MallocMessageBuilder message;
  message.initRoot<::PotentialConfig>();
  auto data = message.getRoot<capnp::AnyStruct>().getDataSection();
  REQUIRE(data.size() >= (offset + 1) * 2);
  data[offset * 2] = static_cast<kj::byte>(unknown & 0xff);
  data[offset * 2 + 1] = static_cast<kj::byte>(unknown >> 8);

  const auto cfg = message.getRoot<::PotentialConfig>().asReader();
  REQUIRE(static_cast<std::uint16_t>(cfg.which()) == unknown);
  const std::string arm = rgpot::rpc::potentialConfigArm(cfg);
  REQUIRE_THAT(arm, ContainsSubstring("discriminant " + std::to_string(unknown)));

  auto applied = rgpot::rpc::configureMorse(cfg, rgpot::MorsePot{});
  REQUIRE_FALSE(applied.ok);
  REQUIRE(applied.replacement == nullptr);
  REQUIRE_THAT(applied.message, ContainsSubstring(arm));
}

TEST_CASE("MullerBrownParams defaults and overrides rebuild the surface",
          "[muller-brown][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto params = builder.initRoot<::PotentialConfig>().initMullerBrown();
  auto amplitude = params.initAmplitude(4);
  amplitude.set(0, -123.0);
  amplitude.set(1, -100.0);
  amplitude.set(2, -170.0);
  amplitude.set(3, 15.0);
  const rgpot::MullerBrownConfig defaults;
  auto applied = rgpot::rpc::configureMullerBrown(
      builder.getRoot<::PotentialConfig>().asReader());
  REQUIRE(applied.ok);
  REQUIRE(applied.replacement != nullptr);
  REQUIRE(applied.replacement->config().amplitude[0] == -123.0);
  REQUIRE(applied.replacement->config().a == defaults.a);
}

TEST_CASE("MullerBrownParams empty lists select the published defaults",
          "[muller-brown][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto params = builder.initRoot<::PotentialConfig>().initMullerBrown();
  params.initAmplitude(0);
  params.initA(0);
  params.initB(0);
  params.initC(0);
  params.initX0(0);
  params.initY0(0);
  auto applied = rgpot::rpc::configureMullerBrown(
      builder.getRoot<::PotentialConfig>().asReader());
  REQUIRE(applied.ok);
  REQUIRE(applied.replacement != nullptr);
  const rgpot::MullerBrownConfig defaults;
  const auto &got = applied.replacement->config();
  REQUIRE(got.amplitude == defaults.amplitude);
  REQUIRE(got.a == defaults.a);
  REQUIRE(got.b == defaults.b);
  REQUIRE(got.c == defaults.c);
  REQUIRE(got.x0 == defaults.x0);
  REQUIRE(got.y0 == defaults.y0);
}

TEST_CASE("MullerBrownParams with no lists set selects the published defaults",
          "[muller-brown][rpc]") {
  capnp::MallocMessageBuilder builder;
  builder.initRoot<::PotentialConfig>().initMullerBrown();
  auto applied = rgpot::rpc::configureMullerBrown(
      builder.getRoot<::PotentialConfig>().asReader());
  REQUIRE(applied.ok);
  REQUIRE(applied.replacement != nullptr);
  const rgpot::MullerBrownConfig defaults;
  REQUIRE(applied.replacement->config().amplitude == defaults.amplitude);
  REQUIRE(applied.replacement->config().y0 == defaults.y0);
}

TEST_CASE("MullerBrownParams overrides only the lists that are not empty",
          "[muller-brown][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto params = builder.initRoot<::PotentialConfig>().initMullerBrown();
  params.initAmplitude(0);
  auto x0 = params.initX0(4);
  for (unsigned i = 0; i < 4; ++i)
    x0.set(i, 0.5 * i);
  auto applied = rgpot::rpc::configureMullerBrown(
      builder.getRoot<::PotentialConfig>().asReader());
  REQUIRE(applied.ok);
  const rgpot::MullerBrownConfig defaults;
  const auto &got = applied.replacement->config();
  REQUIRE(got.amplitude == defaults.amplitude);
  REQUIRE(got.x0 == std::array<double, 4>{0.0, 0.5, 1.0, 1.5});
}

TEST_CASE("MullerBrownParams rejects parameter lists of the wrong length",
          "[muller-brown][rpc]") {
  capnp::MallocMessageBuilder builder;
  builder.initRoot<::PotentialConfig>().initMullerBrown().initA(3);
  auto applied = rgpot::rpc::configureMullerBrown(
      builder.getRoot<::PotentialConfig>().asReader());
  REQUIRE_FALSE(applied.ok);
  REQUIRE(applied.replacement == nullptr);
  REQUIRE_THAT(applied.message, ContainsSubstring("four values"));
}

namespace {

/// The RPC server's step for a mullerBrown configure: an accepted arm swaps the
/// replacement in, a refused one leaves the pot as it was.
bool configureAndSwap(std::unique_ptr<rgpot::MullerBrownPot> &pot,
                      capnp::MallocMessageBuilder &builder) {
  auto applied = rgpot::rpc::configureMullerBrown(
      builder.getRoot<::PotentialConfig>().asReader());
  if (applied.replacement)
    pot = std::move(applied.replacement);
  return applied.ok;
}

void setList(capnp::List<double>::Builder list,
             const std::array<double, 4> &values) {
  for (unsigned i = 0; i < values.size(); ++i)
    list.set(i, values[i]);
}

double energyAt(const rgpot::MullerBrownPot &pot) {
  const std::array<double, 2> coordinates{0.2, 0.3};
  const std::array<double, 2> direction{1.0, 0.0};
  return pot.evaluateCoordinates(coordinates, direction).energy;
}

} // namespace

TEST_CASE(
    "repeated MullerBrownParams configure calls keep nothing from earlier "
    "calls",
    "[muller-brown][rpc]") {
  const rgpot::MullerBrownConfig defaults;
  const rgpot::MullerBrownPot reference;
  auto pot = std::make_unique<rgpot::MullerBrownPot>();

  const std::array<double, 4> amplitude{-400.0, -200.0, -340.0, 30.0};
  const std::array<double, 4> x0{2.0, 1.0, 0.0, -1.0};
  {
    capnp::MallocMessageBuilder builder;
    auto params = builder.initRoot<::PotentialConfig>().initMullerBrown();
    setList(params.initAmplitude(4), amplitude);
    setList(params.initX0(4), x0);
    REQUIRE(configureAndSwap(pot, builder));
  }
  REQUIRE(pot->config().amplitude == amplitude);
  REQUIRE(pot->config().x0 == x0);
  REQUIRE(pot->config().a == defaults.a);
  REQUIRE(energyAt(*pot) != energyAt(reference));

  {
    capnp::MallocMessageBuilder builder;
    builder.initRoot<::PotentialConfig>().initMullerBrown();
    REQUIRE(configureAndSwap(pot, builder));
  }
  REQUIRE(pot->config().amplitude == defaults.amplitude);
  REQUIRE(pot->config().x0 == defaults.x0);
  REQUIRE(energyAt(*pot) == energyAt(reference));

  const std::array<double, 4> a{-2.0, -2.0, -13.0, 1.4};
  {
    capnp::MallocMessageBuilder builder;
    auto params = builder.initRoot<::PotentialConfig>().initMullerBrown();
    params.initAmplitude(0);
    setList(params.initA(4), a);
    REQUIRE(configureAndSwap(pot, builder));
  }
  REQUIRE(pot->config().a == a);
  REQUIRE(pot->config().amplitude == defaults.amplitude);
  REQUIRE(pot->config().x0 == defaults.x0);

  const double partial = energyAt(*pot);
  {
    capnp::MallocMessageBuilder builder;
    builder.initRoot<::PotentialConfig>().initMullerBrown().initB(2);
    REQUIRE_FALSE(configureAndSwap(pot, builder));
  }
  REQUIRE(pot->config().a == a);
  REQUIRE(energyAt(*pot) == partial);

  {
    capnp::MallocMessageBuilder builder;
    builder.initRoot<::PotentialConfig>().initMullerBrown();
    REQUIRE(configureAndSwap(pot, builder));
  }
  REQUIRE(pot->config().a == defaults.a);
  REQUIRE(energyAt(*pot) == energyAt(reference));
}

TEST_CASE("PotentialConfig.orca round-trips at ordinal 22", "[orca][rpc]") {
  const auto field =
      capnp::Schema::from<::PotentialConfig>().getFieldByName("orca");
  REQUIRE(field.getProto().getOrdinal().getExplicit() == 22);

  capnp::MallocMessageBuilder builder;
  auto orca = builder.initRoot<::PotentialConfig>().initOrca();
  orca.setMethod("B3LYP");
  orca.setBasis("def2-SVP");
  orca.setKeywords("TightSCF D3BJ");
  orca.setBlocks("%scf\n  MaxIter 200\nend");
  orca.setCharge(-1);
  orca.setMultiplicity(2);
  orca.setNprocs(4);
  orca.setMaxcoreMb(2000);
  orca.setExecutable("/opt/orca/orca");
  orca.setLibrary("/opt/orcac/lib/liborcac.so");
  orca.setGuessPolicy(::ORCAParams::GuessPolicy::MOREAD);
  orca.setTimeoutS(600.0);
  orca.setScratchDir("/scratch");
  orca.setKeepFiles(true);
  orca.setFreshRetry(false);
  const auto words = capnp::messageToFlatArray(builder);

  capnp::FlatArrayMessageReader reader(words.asPtr());
  const auto cfg = reader.getRoot<::PotentialConfig>();
  REQUIRE(cfg.which() == ::PotentialConfig::ORCA);
  REQUIRE(rgpot::rpc::potentialConfigArm(cfg) == "orca");
  const auto back = rgpot::rpc::orcaConfigFrom(cfg.getOrca());
  REQUIRE(back.method == "B3LYP");
  REQUIRE(back.basis == "def2-SVP");
  REQUIRE(back.keywords == "TightSCF D3BJ");
  REQUIRE(back.blocks == "%scf\n  MaxIter 200\nend");
  REQUIRE(back.charge == -1);
  REQUIRE(back.multiplicity == 2);
  REQUIRE(back.nprocs == 4);
  REQUIRE(back.maxcore_mb == 2000);
  REQUIRE(back.executable == "/opt/orca/orca");
  REQUIRE(back.library == "/opt/orcac/lib/liborcac.so");
  REQUIRE(back.guess_policy == rgpot::ORCAConfig::GuessPolicy::MoRead);
  REQUIRE(back.timeout_s == 600.0);
  REQUIRE(back.scratch_dir == "/scratch");
  REQUIRE(back.keep_files);
  REQUIRE_FALSE(back.fresh_retry);
}

TEST_CASE("ORCAParams defaults are the ORCAConfig defaults", "[orca][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  cfg.initOrca();
  const auto wire = rgpot::rpc::orcaConfigFrom(cfg.asReader().getOrca());
  const rgpot::ORCAConfig native{};
  REQUIRE(wire.method == native.method);
  REQUIRE(wire.basis == native.basis);
  REQUIRE(wire.keywords == native.keywords);
  REQUIRE(wire.blocks == native.blocks);
  REQUIRE(wire.charge == native.charge);
  REQUIRE(wire.multiplicity == native.multiplicity);
  REQUIRE(wire.nprocs == native.nprocs);
  REQUIRE(wire.maxcore_mb == native.maxcore_mb);
  REQUIRE(wire.executable == native.executable);
  REQUIRE(wire.library == native.library);
  REQUIRE(wire.guess_policy == native.guess_policy);
  REQUIRE(wire.timeout_s == native.timeout_s);
  REQUIRE(wire.scratch_dir == native.scratch_dir);
  REQUIRE(wire.keep_files == native.keep_files);
  REQUIRE(wire.fresh_retry == native.fresh_retry);
}

TEST_CASE("CommonMethodSpec lowers into ORCA keywords and blocks",
          "[orca][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  auto common = cfg.initCommon();
  auto xc = common.initXcFunctionals(2);
  xc.set(0, "GGA_X_PBE");
  xc.set(1, "GGA_C_PBE");
  common.setBasisSet("def2-TZVP");
  common.setCharge(1);
  common.setSpinMultiplicity(2);
  common.setScfEnergyToleranceEv(rgpot::units::HARTREE_TO_EV * 1e-8);
  common.setScfMaxIterations(150);
  common.setVanDerWaalsMethod("DFT-D3(BJ)");
  common.setVanDerWaalsS6(0.75);
  common.setRelativityMethod("ZORA");
  cfg.initOrca().setKeywords("TightSCF");

  const auto lowered = rgpot::rpc::orcaConfigFrom(cfg.asReader());
  REQUIRE(lowered.method == "PBE");
  REQUIRE(lowered.basis == "def2-TZVP");
  REQUIRE(lowered.charge == 1);
  REQUIRE(lowered.multiplicity == 2);
  REQUIRE(lowered.keywords == "D3BJ ZORA TightSCF");
  const std::string tole = "%scf\n  TolE ";
  const auto at = lowered.blocks.find(tole);
  REQUIRE(at != std::string::npos);
  REQUIRE(std::stod(lowered.blocks.substr(at + tole.size())) ==
          Catch::Approx(1e-8));
  REQUIRE_THAT(lowered.blocks, ContainsSubstring("\n  MaxIter 150\nend"));
  REQUIRE_THAT(lowered.blocks, ContainsSubstring("%method\n  D3S6 0.75\nend"));
}

TEST_CASE("the orca arm wins over CommonMethodSpec", "[orca][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  auto common = cfg.initCommon();
  common.initXcFunctionals(1).set(0, "B3LYP");
  common.setBasisSet("def2-SVP");
  common.setSpinMultiplicity(3);
  auto orca = cfg.initOrca();
  orca.setMethod("PBE0");
  orca.setMultiplicity(1);
  orca.setBasis("def2-TZVP");
  const auto lowered = rgpot::rpc::orcaConfigFrom(cfg.asReader());
  REQUIRE(lowered.method == "PBE0");
  REQUIRE(lowered.basis == "def2-TZVP");
  // The arm's multiplicity equals its default, so the overlay's stands.
  REQUIRE(lowered.multiplicity == 3);
}

TEST_CASE("CommonMethodSpec refuses what ORCA cannot take", "[orca][rpc]") {
  auto refused = [](auto &&fill) {
    capnp::MallocMessageBuilder builder;
    auto cfg = builder.initRoot<::PotentialConfig>();
    fill(cfg.initCommon());
    cfg.initOrca().setMethod("PBE");
    REQUIRE_THROWS_AS(rgpot::rpc::orcaConfigFrom(cfg.asReader()),
                      std::invalid_argument);
  };
  refused([](::CommonMethodSpec::Builder c) { c.setBasisSet("planewave"); });
  refused([](::CommonMethodSpec::Builder c) { c.setPlanewaveCutoffEv(400.0); });
  refused([](::CommonMethodSpec::Builder c) {
    auto k = c.initKMesh(3);
    k.set(0, 2);
    k.set(1, 2);
    k.set(2, 2);
  });
  refused([](::CommonMethodSpec::Builder c) {
    c.getSmearing().setKind(::CommonMethodSpec::Smearing::Kind::GAUSSIAN);
  });
  refused([](::CommonMethodSpec::Builder c) {
    auto xc = c.initXcFunctionals(2);
    xc.set(0, "GGA_X_RPBE");
    xc.set(1, "GGA_C_PBE");
  });
  refused([](::CommonMethodSpec::Builder c) { c.setVanDerWaalsMethod("MBD"); });
  refused([](::CommonMethodSpec::Builder c) { c.setRelativityMethod("RECP"); });

  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  cfg.initCommon();
  cfg.initOrca().setMethod("PBE");
  REQUIRE_NOTHROW(rgpot::rpc::orcaConfigFrom(cfg.asReader()));
}

TEST_CASE("configure with orca keeps the pot when liborcac is absent",
          "[orca][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto orca = builder.initRoot<::PotentialConfig>().initOrca();
  orca.setMethod("B3LYP");
  orca.setLibrary("no-such-dir/liborcac.so");
  auto applied = rgpot::rpc::configureOrca(
      builder.getRoot<::PotentialConfig>().asReader());
  REQUIRE_FALSE(applied.ok);
  REQUIRE(applied.replacement == nullptr);
  REQUIRE_THAT(applied.message, ContainsSubstring("liborcac not loaded"));
}

TEST_CASE("configure with orca rejects a configuration that cannot run",
          "[orca][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto orca = builder.initRoot<::PotentialConfig>().initOrca();
  orca.setMethod("B3LYP");
  orca.setMultiplicity(0);
  auto applied = rgpot::rpc::configureOrca(
      builder.getRoot<::PotentialConfig>().asReader());
  REQUIRE_FALSE(applied.ok);
  REQUIRE_THAT(applied.message, ContainsSubstring("rejected"));
  REQUIRE_THAT(applied.message, ContainsSubstring("multiplicity"));
}

TEST_CASE("configure on the ORCA backend refuses other arms", "[orca][rpc]") {
  capnp::MallocMessageBuilder builder;
  builder.initRoot<::PotentialConfig>().initMorse();
  auto applied = rgpot::rpc::configureOrca(
      builder.getRoot<::PotentialConfig>().asReader());
  REQUIRE_FALSE(applied.ok);
  REQUIRE_THAT(applied.message, ContainsSubstring("morse"));

  capnp::MallocMessageBuilder none;
  none.initRoot<::PotentialConfig>();
  REQUIRE(rgpot::rpc::configureOrca(
              none.getRoot<::PotentialConfig>().asReader())
              .ok);

  capnp::MallocMessageBuilder common_only;
  common_only.initRoot<::PotentialConfig>().initCommon();
  REQUIRE_FALSE(rgpot::rpc::configureOrca(
                    common_only.getRoot<::PotentialConfig>().asReader())
                    .ok);
}
