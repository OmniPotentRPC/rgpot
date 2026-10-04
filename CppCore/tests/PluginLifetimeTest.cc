// MIT License
#include "rgpot/plugin/PluginBridge.hpp"
#include "rgpot/plugin/PluginRegistry.hpp"
#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace {
struct FinalObserver {
  ~FinalObserver() { std::puts("owner destructors complete"); }
} observer;
// This owner outlives function-local static objects initialized in main.
std::unique_ptr<rgpot::PotentialBase> potential;
}
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("expected fixture library path");
    auto &registry = rgpot::plugin::PluginRegistry::instance();
    if (!registry.load_plugin(argv[1]))
      throw std::runtime_error("fixture failed to load");
    const auto *plugin = registry.find("lifetime-contract");
    if (!plugin)
      throw std::runtime_error("fixture is absent from registry");
    potential = rgpot::plugin::create_from_plugin(plugin->desc);
    rgpot::types::AtomMatrix positions(1, 3);
    positions(0, 0) = positions(0, 1) = positions(0, 2) = 0.0;
    const std::array<std::array<double, 3>, 3> box{{{20,0,0},{0,20,0},{0,0,20}}};
    const auto [energy, force, variance] = (*potential)(positions, {1}, box);
    if (energy != 37.0 || variance != 0.0 || force(0, 0) != 0.0 ||
        force(0, 1) != 0.0 || force(0, 2) != 0.0)
      throw std::runtime_error("fixture result changed");
    std::puts("lifetime fixture evaluated");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
