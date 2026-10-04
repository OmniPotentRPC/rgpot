// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "rgpot/plugin/PluginBridge.hpp"
#include "rgpot/plugin/PluginRegistry.hpp"
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Catch::Matchers::WithinAbs;
using rgpot::types::AtomMatrix;
struct State { bool fail; };
int created = 0;
int destroyed = 0;
int optional_calls = 0;
rgpot_plugin_status_t create(const char *config, rgpot_plugin_instance **output) {
  ++created;
  *output = reinterpret_cast<rgpot_plugin_instance *>(new State{
      std::strcmp(config, "{\"fail\":true}") == 0});
  if (std::strcmp(config, "{\"partial\":true}") == 0)
    return RGPOT_PLUGIN_ERROR;
  if (std::strcmp(config, "{\"throw\":true}") == 0)
    throw std::runtime_error("fixture constructor threw");
  return RGPOT_PLUGIN_OK;
}
void destroy(rgpot_plugin_instance *instance) {
  ++destroyed;
  delete reinterpret_cast<State *>(instance);
}
rgpot_plugin_status_t calculate(rgpot_plugin_instance *instance,
                                const rgpot_plugin_input_t *input,
                                rgpot_plugin_output_t *output) {
  if (reinterpret_cast<State *>(instance)->fail)
    return RGPOT_PLUGIN_ERROR;
  output->energy = 0.125 * input->cell[0];
  output->variance = 0.25;
  for (size_t i = 0; i < 3 * input->n_atoms; ++i) {
    output->energy += 0.5 * input->positions[i] * input->positions[i];
    output->forces[i] = -input->positions[i];
  }
  return RGPOT_PLUGIN_OK;
}
uint32_t unsupportedCaps(const rgpot_plugin_instance *) {
  ++optional_calls;
  return RGPOT_CAP_ENERGY;
}
const char *optionalError(const rgpot_plugin_instance *) {
  ++optional_calls;
  return "optional error callback must be inaccessible";
}
rgpot_plugin_descriptor_t descriptor() {
  return {sizeof(rgpot_plugin_descriptor_t), RGPOT_PLUGIN_API_VERSION,
          "contract-harmonic", "1", "bohr", "hartree",
          create, destroy, calculate, nullptr, nullptr};
}
const std::array<std::array<double,3>,3> box{{{20,0,0},{0,20,0},{0,0,20}}};
AtomMatrix coordinates() {
  AtomMatrix result(1,3);
  result(0,0)=0.7; result(0,1)=-0.2; result(0,2)=0.1;
  return result;
}
} // namespace

TEST_CASE("Plugin unit conversion preserves the Cartesian energy derivative", "[plugin][contract]") {
  auto desc=descriptor();
  auto pot=rgpot::plugin::create_from_plugin(&desc);
  auto positions=coordinates();
  const auto [energy,forces,variance]=(*pot)(positions,{1},box);
  constexpr double hartree=4.359744722206048e-18/1.602176634e-19;
  constexpr double bohr=5.2917721054482e-11/1e-10;
  const double expected=hartree*(0.125*20.0/bohr+0.5*(0.49+0.04+0.01)/(bohr*bohr));
  REQUIRE_THAT(energy,WithinAbs(expected,2e-12));
  REQUIRE_THAT(variance,WithinAbs(0.25*hartree*hartree,2e-12));
  for (size_t axis=0;axis<3;++axis) {
    REQUIRE_THAT(forces(0,axis),WithinAbs(-hartree*positions(0,axis)/(bohr*bohr),2e-12));
    auto plus=positions;
    auto minus=positions;
    constexpr double step=1e-5;
    plus(0,axis)+=step; minus(0,axis)-=step;
    const double derivative=(std::get<0>((*pot)(plus,{1},box))-std::get<0>((*pot)(minus,{1},box)))/(2*step);
    REQUIRE_THAT(forces(0,axis),WithinAbs(-derivative,2e-8));
  }
  REQUIRE(positions(0,0)==0.7);
  REQUIRE(positions(0,1)==-0.2);
  REQUIRE(positions(0,2)==0.1);
}

TEST_CASE("Plugin short descriptors omit optional callbacks", "[plugin][contract]") {
  auto desc=descriptor();
  desc.descriptor_size=offsetof(rgpot_plugin_descriptor_t,capabilities);
  desc.capabilities=unsupportedCaps;
  desc.last_error=optionalError;
  optional_calls=0;
  auto pot=rgpot::plugin::create_from_plugin(&desc,"{\"fail\":true}");
  REQUIRE_THROWS_WITH((*pot)(coordinates(),{1},box),"plugin calculate failed");
  REQUIRE(optional_calls==0);
}

TEST_CASE("Plugin invalid descriptors fail before creating an instance", "[plugin][contract]") {
  auto desc=descriptor();
  created=0;
  SECTION("version") { ++desc.api_version; }
  SECTION("size") { desc.descriptor_size=offsetof(rgpot_plugin_descriptor_t,calculate); }
  SECTION("name") { desc.name=nullptr; }
  SECTION("callback") { desc.destroy=nullptr; }
  REQUIRE_THROWS_AS(rgpot::plugin::create_from_plugin(&desc),std::invalid_argument);
  REQUIRE(created==0);
}

TEST_CASE("Plugin construction failure releases a returned instance", "[plugin][contract]") {
  auto desc=descriptor();
  created=destroyed=0;
  SECTION("error status") {
    REQUIRE_THROWS_AS(rgpot::plugin::create_from_plugin(&desc,"{\"partial\":true}"),std::runtime_error);
  }
  SECTION("exception") {
    REQUIRE_THROWS_WITH(rgpot::plugin::create_from_plugin(&desc,"{\"throw\":true}"),"fixture constructor threw");
  }
  SECTION("unsupported forces") {
    desc.capabilities=unsupportedCaps;
    REQUIRE_THROWS_AS(rgpot::plugin::create_from_plugin(&desc),std::invalid_argument);
  }
  REQUIRE(created==1);
  REQUIRE(destroyed==1);
}

TEST_CASE("Plugin units are validated before calling the constructor", "[plugin][contract]") {
  auto desc=descriptor();
  desc.length_unit="not_a_length";
  created=destroyed=0;
  REQUIRE_THROWS(rgpot::plugin::create_from_plugin(&desc));
  REQUIRE(created==0);
  REQUIRE(destroyed==0);
}

TEST_CASE("Plugin cache identity includes plugin parameters", "[plugin][contract]") {
  auto desc=descriptor();
  auto first=rgpot::plugin::create_from_plugin(&desc,"{}");
  auto identical=rgpot::plugin::create_from_plugin(&desc,"{}");
  auto changed=rgpot::plugin::create_from_plugin(&desc,"{\"k\":2}");
  REQUIRE(first->paramsKey()==identical->paramsKey());
  REQUIRE(first->paramsKey()!=changed->paramsKey());
  REQUIRE(first->caps().reentrancy==rgpot::Reentrancy::ProcessSerial);
  REQUIRE_FALSE(first->caps().periodic);
}

TEST_CASE("Plugin registry entries survive registration growth", "[plugin][contract]") {
  auto &registry=rgpot::plugin::PluginRegistry::instance();
  const auto *original=registry.find("lennard-jones");
  REQUIRE(original!=nullptr);
  static std::deque<std::string> names;
  static std::deque<rgpot_plugin_descriptor_t> descriptors;
  for (unsigned i=0;i<257;++i) {
    names.push_back("registry-growth-"+std::to_string(i));
    descriptors.push_back(descriptor());
    descriptors.back().name=names.back().c_str();
    registry.register_builtin(&descriptors.back());
  }
  REQUIRE(registry.find("lennard-jones")==original);
  auto pot=rgpot::plugin::create_from_plugin(original->desc);
  REQUIRE(pot->caps().reentrancy==rgpot::Reentrancy::PerInstance);
  REQUIRE(pot->caps().periodic);
}
