// MIT License
// Copyright 2023--present rgpot developers

// rgpot::makePotential: the in-process arms against the potentials built
// directly, the plugin arm over the classical engine, the orca arm over a
// stand-in liborcac, and the arms this build cannot construct.
// RGPOT_FACTORY_CLASSICAL_ENGINE and RGPOT_FACTORY_ORCAC_FAKE name the two
// libraries.

#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <catch2/catch_all.hpp>
#include <dlfcn.h>

#include "rgpot/EnginePot/EnginePot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/ORCACPot/ORCACPot.hpp"
#include "rgpot/PotentialFactory.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using rgpot::types::AtomMatrix;

namespace {

const AtomMatrix kPositions{{0.0, 0.0, 0.0},
                            {1.1, 0.0, 0.0},
                            {0.0, 1.2, 0.0},
                            {0.3, 0.4, 1.3}};
const std::vector<int> kZ{18, 18, 18, 18};
const std::array<std::array<double, 3>, 3> kBox{
    {{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};

std::vector<capnp::word> flat(capnp::MessageBuilder &message) {
  const auto words = capnp::messageToFlatArray(message);
  return {words.begin(), words.end()};
}

void requireSame(rgpot::PotentialBase &got, rgpot::PotentialBase &want) {
  auto [e1, f1, v1] = got(kPositions, kZ, kBox);
  auto [e2, f2, v2] = want(kPositions, kZ, kBox);
  (void)v1;
  (void)v2;
  REQUIRE(e2 != 0.0);
  REQUIRE_THAT(e1, WithinRel(e2, 1e-14));
  for (std::size_t i = 0; i < static_cast<std::size_t>(f2.size()); ++i) {
    REQUIRE_THAT(f1.data()[i], WithinAbs(f2.data()[i], 1e-12));
  }
}

const char *requiredEnv(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    FAIL(std::string(name) + " is not set");
  }
  return value;
}

/// Sets one environment variable for the life of the object.
class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : name_(name) {
    if (const char *old = std::getenv(name)) {
      had_ = true;
      old_ = old;
    }
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (had_) {
      setenv(name_.c_str(), old_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }
  ScopedEnv(const ScopedEnv &) = delete;
  ScopedEnv &operator=(const ScopedEnv &) = delete;

private:
  std::string name_;
  std::string old_;
  bool had_ = false;
};

/// BuiltinParams naming @p name, as the classical engine reads it.
std::vector<capnp::word> builtinMessage(const char *name) {
  capnp::MallocMessageBuilder message;
  message.initRoot<::BuiltinParams>().setName(name);
  return flat(message);
}

} // namespace

TEST_CASE("the lj arm builds the LJPot its parameters name", "[factory]") {
  capnp::MallocMessageBuilder message;
  auto lj = message.initRoot<::PotentialConfig>().initLj();
  lj.setU0(0.0104);
  lj.setPsi(1.1);
  lj.setCutoff(6.5);
  lj.setSwitchWidth(0.75);
  const auto bytes = flat(message);

  rgpot::LJPot direct(
      rgpot::LJConfig{.u0 = 0.0104, .cutoff = 6.5, .psi = 1.1, .switch_width = 0.75});

  auto fromReader = rgpot::makePotential(message.getRoot<::PotentialConfig>().asReader());
  REQUIRE(fromReader->getType() == rgpot::PotType::LJ);
  requireSame(*fromReader, direct);

  auto fromBytes = rgpot::makePotential(bytes.data(), bytes.size() * sizeof(capnp::word));
  REQUIRE(fromBytes->getType() == rgpot::PotType::LJ);
  requireSame(*fromBytes, direct);
}

TEST_CASE("the morse arm builds the MorsePot its parameters name", "[factory]") {
  capnp::MallocMessageBuilder message;
  auto morse = message.initRoot<::PotentialConfig>().initMorse();
  morse.setDe(0.5);
  morse.setA(1.3);
  morse.setRe(1.15);
  morse.setCutoff(7.0);
  morse.setSwitchWidth(0.5);
  const auto bytes = flat(message);

  rgpot::MorsePot direct(rgpot::MorseConfig{
      .De = 0.5, .a = 1.3, .re = 1.15, .cutoff = 7.0, .switch_width = 0.5});

  auto fromReader = rgpot::makePotential(message.getRoot<::PotentialConfig>().asReader());
  REQUIRE(fromReader->getType() == rgpot::PotType::Morse);
  requireSame(*fromReader, direct);

  auto fromBytes = rgpot::makePotential(bytes.data(), bytes.size() * sizeof(capnp::word));
  requireSame(*fromBytes, direct);
}

TEST_CASE("the plugin arm loads an engine library with the opaque payload",
          "[factory][plugin]") {
  const std::string classical = requiredEnv("RGPOT_FACTORY_CLASSICAL_ENGINE");
  const auto payload = builtinMessage("lj");
  rgpot::LJPot direct;

  SECTION("from libraryPath") {
    capnp::MallocMessageBuilder message;
    auto plugin = message.initRoot<::PotentialConfig>().initPlugin();
    plugin.setPluginName("classical");
    plugin.setLibraryPath(classical);
    plugin.setOpaquePayload(kj::arrayPtr(reinterpret_cast<const kj::byte *>(payload.data()),
                                         payload.size() * sizeof(capnp::word)));
    auto pot = rgpot::makePotential(message.getRoot<::PotentialConfig>().asReader());
    auto *engine = dynamic_cast<rgpot::EnginePot *>(pot.get());
    REQUIRE(engine != nullptr);
    REQUIRE(engine->libraryPath() == classical);
    REQUIRE(pot->getType() == rgpot::PotType::Engine);
    requireSame(*pot, direct);
  }

  SECTION("from pluginName through RGPOT_<NAME>_ENGINE") {
    ScopedEnv env("RGPOT_CLASSICAL_ENGINE", classical.c_str());
    capnp::MallocMessageBuilder message;
    auto plugin = message.initRoot<::PotentialConfig>().initPlugin();
    plugin.setPluginName("classical");
    plugin.setOpaquePayload(kj::arrayPtr(reinterpret_cast<const kj::byte *>(payload.data()),
                                         payload.size() * sizeof(capnp::word)));
    const auto bytes = flat(message);
    auto pot = rgpot::makePotential(bytes.data(), bytes.size() * sizeof(capnp::word));
    auto *engine = dynamic_cast<rgpot::EnginePot *>(pot.get());
    REQUIRE(engine != nullptr);
    REQUIRE(engine->libraryPath() == classical);
    REQUIRE(engine->label() == "plugin classical");
    requireSame(*pot, direct);
  }

  SECTION("an engine that refuses the payload is reported") {
    const auto wrong = builtinMessage("no_such_potential");
    capnp::MallocMessageBuilder message;
    auto plugin = message.initRoot<::PotentialConfig>().initPlugin();
    plugin.setLibraryPath(classical);
    plugin.setOpaquePayload(kj::arrayPtr(reinterpret_cast<const kj::byte *>(wrong.data()),
                                         wrong.size() * sizeof(capnp::word)));
    REQUIRE_THROWS_WITH(rgpot::makePotential(message.getRoot<::PotentialConfig>().asReader()),
                        ContainsSubstring("rgpot_engine_create failed"));
  }
}

TEST_CASE("the orca arm builds an ORCACPot over liborcac", "[factory][orca]") {
  const std::string fake = requiredEnv("RGPOT_FACTORY_ORCAC_FAKE");

  capnp::MallocMessageBuilder message;
  auto config = message.initRoot<::PotentialConfig>();
  auto orca = config.initOrca();
  orca.setMethod("PBE0");
  orca.setBasis("def2-SVP");
  orca.setKeywords("TightSCF D4");
  orca.setCharge(-1);
  orca.setMultiplicity(2);
  orca.setNprocs(4);
  orca.setTimeoutS(120.0);
  config.initCommon().setBasisSet("def2-TZVP");
  // liborcac takes the whole PotentialConfig and lowers common itself.
  const auto expected = flat(message);
  const std::size_t expected_bytes = expected.size() * sizeof(capnp::word);

  SECTION("ORCAC_LIBRARY names the library") {
    ScopedEnv env("ORCAC_LIBRARY", fake.c_str());
    ScopedEnv other("RGPOT_ORCAC_ENGINE", nullptr);
    auto pot = rgpot::makePotential(config.asReader());
    auto *orcac = dynamic_cast<rgpot::ORCACPot *>(pot.get());
    REQUIRE(orcac != nullptr);
    REQUIRE(pot->getType() == rgpot::PotType::ORCAC);
    REQUIRE(orcac->available());
    REQUIRE(orcac->loadError().empty());
    REQUIRE(orcac->libraryPath() == fake);
    REQUIRE(orcac->runDirectory() == "/fake/orcac");

    // The fake's energy in Hartree is the message length, and its forces are
    // 0.1 * (i + 1) Hartree/Bohr.
    auto [e, f, v] = (*pot)(kPositions, kZ, kBox);
    (void)v;
    REQUIRE_THAT(e, WithinRel(static_cast<double>(expected_bytes) *
                                  rgpot::units::HARTREE_TO_EV,
                              1e-14));
    for (std::size_t i = 0; i < static_cast<std::size_t>(f.size()); ++i) {
      REQUIRE_THAT(f.data()[i], WithinRel(0.1 * static_cast<double>(i + 1) *
                                              rgpot::units::HARTREE_BOHR_TO_EV_ANGSTROM,
                                          1e-14));
    }

    void *handle = dlopen(fake.c_str(), RTLD_NOW | RTLD_NOLOAD);
    REQUIRE(handle != nullptr);
    using LastFn = std::size_t (*)(void *, std::size_t, int *);
    auto last = reinterpret_cast<LastFn>(dlsym(handle, "orcac_fake_last"));
    REQUIRE(last != nullptr);
    std::vector<capnp::word> seen(expected.size());
    int from_config = -1;
    REQUIRE(last(seen.data(), expected_bytes, &from_config) == expected_bytes);
    REQUIRE(from_config == 1);
    REQUIRE(std::memcmp(seen.data(), expected.data(), expected_bytes) == 0);
    dlclose(handle);

    capnp::FlatArrayMessageReader reader(kj::arrayPtr(seen.data(), seen.size()));
    const auto params = reader.getRoot<::PotentialConfig>().getOrca();
    REQUIRE(std::string(params.getMethod().cStr()) == "PBE0");
    REQUIRE(params.getCharge() == -1);
    REQUIRE(params.getNprocs() == 4);
  }

  SECTION("RGPOT_ORCAC_ENGINE names the library") {
    ScopedEnv env("ORCAC_LIBRARY", nullptr);
    ScopedEnv other("RGPOT_ORCAC_ENGINE", fake.c_str());
    auto pot = rgpot::makePotential(config.asReader());
    auto *orcac = dynamic_cast<rgpot::ORCACPot *>(pot.get());
    REQUIRE(orcac != nullptr);
    REQUIRE(orcac->libraryPath() == fake);
  }

  SECTION("ORCACPot takes ORCAParams alone") {
    capnp::MallocMessageBuilder alone;
    alone.setRoot(orca.asReader());
    const auto words = flat(alone);
    rgpot::ORCACConfig c;
    const auto *bytes = reinterpret_cast<const unsigned char *>(words.data());
    c.message.assign(bytes, bytes + words.size() * sizeof(capnp::word));
    c.engine_path = fake;
    rgpot::ORCACPot pot(std::move(c));
    REQUIRE(pot.available());
    void *handle = dlopen(fake.c_str(), RTLD_NOW | RTLD_NOLOAD);
    REQUIRE(handle != nullptr);
    using LastFn = std::size_t (*)(void *, std::size_t, int *);
    auto last = reinterpret_cast<LastFn>(dlsym(handle, "orcac_fake_last"));
    std::vector<capnp::word> seen(words.size());
    int from_config = -1;
    REQUIRE(last(seen.data(), seen.size() * sizeof(capnp::word), &from_config) ==
            words.size() * sizeof(capnp::word));
    REQUIRE(from_config == 0);
    dlclose(handle);
  }

  SECTION("without liborcac the error names what was tried") {
    ScopedEnv env("ORCAC_LIBRARY", "/nonexistent/liborcac.so");
    ScopedEnv other("RGPOT_ORCAC_ENGINE", nullptr);
    auto pot = rgpot::makePotential(config.asReader());
    auto *orcac = dynamic_cast<rgpot::ORCACPot *>(pot.get());
    REQUIRE(orcac != nullptr);
    REQUIRE_FALSE(orcac->available());
    REQUIRE_THAT(orcac->loadError(), ContainsSubstring("liborcac not loaded") &&
                                         ContainsSubstring("/nonexistent/liborcac.so") &&
                                         ContainsSubstring("liborcac.so.1"));
    REQUIRE_THROWS_WITH((*pot)(kPositions, kZ, kBox),
                        ContainsSubstring("ORCACPot: liborcac not loaded"));
  }
}

TEST_CASE("arms this build cannot construct are refused by name", "[factory]") {
  capnp::MallocMessageBuilder message;
  auto config = message.initRoot<::PotentialConfig>();

  SECTION("none") {
    config.setNone();
    REQUIRE_THROWS_MATCHES(rgpot::makePotential(config.asReader()), std::invalid_argument,
                           Catch::Matchers::MessageMatches(ContainsSubstring("(none)")));
  }
  SECTION("gpr") {
    config.initGpr();
    REQUIRE_THROWS_MATCHES(rgpot::makePotential(config.asReader()), std::invalid_argument,
                           Catch::Matchers::MessageMatches(ContainsSubstring("gpr arm")));
  }
  SECTION("external") {
    config.initExternal();
    REQUIRE_THROWS_AS(rgpot::makePotential(config.asReader()), std::invalid_argument);
  }
#ifndef RGPOT_HAS_CP2K
  SECTION("cp2k without -Dwith_cp2k") {
    config.initCp2k();
    REQUIRE_THROWS_MATCHES(
        rgpot::makePotential(config.asReader()), std::invalid_argument,
        Catch::Matchers::MessageMatches(ContainsSubstring("cp2k arm") &&
                                        ContainsSubstring("-Dwith_cp2k=true")));
  }
#endif
#ifndef RGPOT_HAS_EXPR
  SECTION("expr without -Dwith_expr") {
    config.initExpr();
    REQUIRE_THROWS_MATCHES(
        rgpot::makePotential(config.asReader()), std::invalid_argument,
        Catch::Matchers::MessageMatches(ContainsSubstring("-Dwith_expr=true")));
  }
#endif
  SECTION("a mopac field MOPACPot cannot take") {
    config.initMopac().setSolver(1);
    REQUIRE_THROWS_MATCHES(rgpot::makePotential(config.asReader()), std::invalid_argument,
                           Catch::Matchers::MessageMatches(ContainsSubstring("solver")));
  }
  SECTION("bytes that are not a message") {
    const std::array<unsigned char, 5> junk{1, 2, 3, 4, 5};
    REQUIRE_THROWS_AS(rgpot::makePotential(junk.data(), junk.size()), std::invalid_argument);
  }
}
