// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "rgpot/abi/ProfileLoader.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

using Catch::Matchers::WithinAbs;

namespace {

std::string fake_engine_path() {
  const char *path = std::getenv("RGPOT_CPMD_ENGINE");
  REQUIRE(path != nullptr);
  return path;
}

std::vector<unsigned char> flat_bytes(::capnp::MallocMessageBuilder &msg) {
  auto words = ::capnp::messageToFlatArray(msg);
  const auto bytes = words.asBytes();
  return std::vector<unsigned char>(bytes.begin(), bytes.end());
}

std::vector<unsigned char> make_config() {
  ::capnp::MallocMessageBuilder msg;
  auto config = msg.initRoot<::PotentialConfig>();
  auto cpmd = config.initCpmd();
  cpmd.setFunctional("BLYP");
  cpmd.setCutOffRy(70.0);
  return flat_bytes(msg);
}

std::vector<unsigned char> make_force_input(double cell_zz) {
  ::capnp::MallocMessageBuilder msg;
  auto input = msg.initRoot<::ForceInput>();
  auto pos = input.initPos(9);
  const double coords[9] = {0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0};
  for (unsigned int i = 0; i < 9; ++i)
    pos.set(i, coords[i]);
  auto atmnrs = input.initAtmnrs(3);
  atmnrs.set(0, 8);
  atmnrs.set(1, 1);
  atmnrs.set(2, 1);
  auto box = input.initBox(9);
  for (unsigned int i = 0; i < 9; ++i)
    box.set(i, 0.0);
  box.set(0, 10.0);
  box.set(4, 10.0);
  box.set(8, cell_zz);
  input.setLengthUnit("angstrom");
  input.setEnergyUnit("eV");
  return flat_bytes(msg);
}

} // namespace

TEST_CASE("ProfileLoader resolves the full minimum profile from one prefix",
          "[abi][profile]") {
  rgpot::abi::ProfileLoader loader;
  loader.load("cpmdc", fake_engine_path());
  REQUIRE(loader.loaded());
  REQUIRE(loader.prefix() == "cpmdc");
  REQUIRE(loader.abi_version() == 0);
  REQUIRE(loader.available() == 1);
  REQUIRE(std::string(loader.version()).find("fake") != std::string::npos);
  REQUIRE(std::string(loader.last_error()).empty());
}

TEST_CASE("ProfileLoader capabilities round-trips a Capabilities message",
          "[abi][profile]") {
  rgpot::abi::ProfileLoader loader;
  loader.load("cpmdc", fake_engine_path());

  const auto caps_bytes = loader.capabilities();
  REQUIRE(caps_bytes.size() % sizeof(::capnp::word) == 0);
  auto words = kj::arrayPtr(
      reinterpret_cast<const ::capnp::word *>(caps_bytes.data()),
      caps_bytes.size() / sizeof(::capnp::word));
  ::capnp::FlatArrayMessageReader reader(words);
  auto caps = reader.getRoot<::Capabilities>();

  REQUIRE(std::string(caps.getBackendName().cStr()) == "cpmdc");
  REQUIRE(caps.getAvailable());
  auto ops = caps.getOperations();
  REQUIRE(ops.size() == 3);
  REQUIRE(ops[0] == ::Capabilities::Operation::ENERGY);
  REQUIRE(ops[1] == ::Capabilities::Operation::FORCES);
  REQUIRE(ops[2] == ::Capabilities::Operation::GRADIENT);
  auto kinds = caps.getConfigKinds();
  REQUIRE(kinds.size() == 1);
  REQUIRE(std::string(kinds[0].cStr()) == "cpmd");
  REQUIRE(std::string(RGPOT_BUILD_VERSION) != "");
  REQUIRE(std::string(caps.getBuildVersion().cStr()) == RGPOT_BUILD_VERSION);
  REQUIRE(std::string(caps.getBuildRevision().cStr()) == RGPOT_BUILD_REVISION);
  REQUIRE(std::string(caps.getProtocolFamily().cStr()) == "rgpot.potentials");
}

TEST_CASE("ProfileLoader drives a config -> session -> step evaluation",
          "[abi][profile]") {
  rgpot::abi::ProfileLoader loader;
  loader.load("cpmdc", fake_engine_path());

  const auto config = make_config();
  const double cell_zz = 12.5;
  const auto step = make_force_input(cell_zz);

  REQUIRE(loader.configure(config.data(), config.size()) == 0);

  void *session =
      loader.session_create_from_config(config.data(), config.size());
  REQUIRE(session != nullptr);
  REQUIRE(loader.session_configure(session, config.data(), config.size()) ==
          0);

  const size_t need =
      loader.potential_result_size_for_force_input(step.data(), step.size());
  REQUIRE(need > 0);
  std::vector<unsigned char> out(need);
  size_t wrote = 0;
  auto result = loader.session_calculate_result(session, step.data(),
                                                step.size(), out.data(),
                                                out.size(), &wrote);
  REQUIRE(result.ok == 1);
  REQUIRE(wrote > 0);
  loader.session_destroy(session);

  auto words = kj::arrayPtr(
      reinterpret_cast<const ::capnp::word *>(out.data()),
      wrote / sizeof(::capnp::word));
  ::capnp::FlatArrayMessageReader reader(words);
  auto decoded = reader.getRoot<::PotentialResult>();
  REQUIRE_THAT(decoded.getEnergy(), WithinAbs(0.75 + 0.001 * cell_zz, 1e-12));
  REQUIRE(decoded.getForces().size() == 9);

  std::vector<unsigned char> out_one_shot(need);
  size_t wrote_one_shot = 0;
  auto one_shot = loader.calculate_result_from_config(
      config.data(), config.size(), step.data(), step.size(),
      out_one_shot.data(), out_one_shot.size(), &wrote_one_shot);
  REQUIRE(one_shot.ok == 1);
  REQUIRE(wrote_one_shot == wrote);
  loader.finalize();
}

TEST_CASE("ProfileLoader rejects a library missing the profile symbols",
          "[abi][profile]") {
  rgpot::abi::ProfileLoader loader;
  REQUIRE_THROWS_AS(loader.load("nwchemc", fake_engine_path()),
                    std::runtime_error);
}

namespace {

::Capabilities::Builder filled_capabilities(::capnp::MallocMessageBuilder &msg) {
  auto caps = msg.initRoot<::Capabilities>();
  rgpot::abi::fill_compatibility(caps);
  auto ops = caps.initOperations(2);
  ops.set(0, ::Capabilities::Operation::ENERGY);
  ops.set(1, ::Capabilities::Operation::FORCES);
  return caps;
}

} // namespace

TEST_CASE("capability handshake names the field, required value and received value",
          "[abi][profile]") {
  using rgpot::abi::Expectation;
  auto refuse = [](auto edit, const Expectation &want, const char *exact) {
    ::capnp::MallocMessageBuilder msg;
    auto caps = filled_capabilities(msg);
    edit(caps);
    REQUIRE(rgpot::abi::check_capabilities(caps.asReader(), want) == exact);
  };
  auto allow = [](auto edit, const Expectation &want) {
    ::capnp::MallocMessageBuilder msg;
    auto caps = filled_capabilities(msg);
    edit(caps);
    REQUIRE(rgpot::abi::check_capabilities(caps.asReader(), want).empty());
  };

  Expectation want;
  Expectation minor = want;
  minor.protocol_minor_min = 1;
  Expectation bridge_minor = want;
  bridge_minor.bridge_abi_minor_max = 2;
  Expectation dlpack_minor = want;
  dlpack_minor.dlpack_minor_max = 2;

  allow([](auto) {}, want);
  refuse([](auto caps) { caps.setProtocolFamily("old.family"); }, want,
         "protocolFamily: required rgpot.potentials, received old.family");
  refuse([](auto caps) { caps.setProtocolFamily("new.family"); }, want,
         "protocolFamily: required rgpot.potentials, received new.family");
  refuse([](auto caps) { caps.setProtocolFamily(""); }, want,
         "protocolFamily: required rgpot.potentials, received ");
  refuse([](auto caps) { caps.setProtocolMajor(0); }, want,
         "protocolMajor: required 1, received 0");
  refuse([](auto caps) { caps.setProtocolMajor(2); }, want,
         "protocolMajor: required 1, received 2");
  allow([](auto caps) { caps.setProtocolMinor(1); }, minor);
  refuse([](auto caps) { caps.setProtocolMinor(0); }, minor,
         "protocolMinor: required >= 1, received 0");
  allow([](auto caps) { caps.setProtocolMinor(3); }, minor);
  refuse([](auto) {}, minor, "protocolMinor: required >= 1, received 0");
  refuse([](auto caps) { caps.setSchemaId(""); }, want,
         "schemaId: required 0xbd1f89fa17369103, received ");
  refuse([](auto caps) { caps.setSchemaId("0x0000000000000001"); }, want,
         "schemaId: required 0xbd1f89fa17369103, received 0x0000000000000001");
  refuse([](auto caps) { caps.setSchemaId("0xffffffffffffffff"); }, want,
         "schemaId: required 0xbd1f89fa17369103, received 0xffffffffffffffff");
  refuse([](auto caps) { caps.setBridgeAbiMajor(0); }, want,
         "bridgeAbiMajor: required 1, received 0");
  refuse([](auto caps) { caps.setBridgeAbiMajor(2); }, want,
         "bridgeAbiMajor: required 1, received 2");
  allow([](auto caps) { caps.setBridgeAbiMinor(2); }, bridge_minor);
  allow([](auto caps) { caps.setBridgeAbiMinor(1); }, bridge_minor);
  refuse([](auto caps) { caps.setBridgeAbiMinor(3); }, bridge_minor,
         "bridgeAbiMinor: required <= 2, received 3");
  allow([](auto caps) { caps.setBridgeAbiMinor(0); }, bridge_minor);
  refuse([](auto caps) { caps.setBridgeLayout(0); }, want,
         "bridgeLayout: required 1, received 0");
  refuse([](auto caps) { caps.setBridgeLayout(2); }, want,
         "bridgeLayout: required 1, received 2");
  allow([](auto caps) { caps.setBridgeFeatures(0x1); }, want);
  refuse([](auto caps) { caps.setBridgeFeatures(0x7); }, want,
         "bridgeFeatures: required subset of 0x3, received 0x7");
  allow([](auto caps) { caps.setBridgeFeatures(0); }, want);
  refuse([](auto caps) { caps.setDlpackMajor(0); }, want,
         "dlpackMajor: required 1, received 0");
  refuse([](auto caps) { caps.setDlpackMajor(2); }, want,
         "dlpackMajor: required 1, received 2");
  allow([](auto caps) { caps.setDlpackMinor(2); }, dlpack_minor);
  allow([](auto caps) { caps.setDlpackMinor(1); }, dlpack_minor);
  refuse([](auto caps) { caps.setDlpackMinor(3); }, dlpack_minor,
         "dlpackMinor: required <= 2, received 3");
  allow([](auto caps) { caps.setDlpackMinor(0); }, dlpack_minor);
  refuse(
      [](auto caps) {
        auto ops = caps.initOperations(1);
        ops.set(0, ::Capabilities::Operation::ENERGY);
      },
      want, "operations: required forces, received energy");
  allow(
      [](auto caps) {
        auto ops = caps.initOperations(3);
        ops.set(0, ::Capabilities::Operation::ENERGY);
        ops.set(1, ::Capabilities::Operation::FORCES);
        ops.set(2, ::Capabilities::Operation::HESSIAN);
      },
      want);
  refuse([](auto caps) { caps.initOperations(0); }, want,
         "operations: required energy, received none");

  ::capnp::MallocMessageBuilder cleared;
  auto caps = filled_capabilities(cleared);
  caps.setBuildVersion("");
  caps.setBuildRevision("");
  REQUIRE(rgpot::abi::check_capabilities(caps.asReader()).empty());
}
