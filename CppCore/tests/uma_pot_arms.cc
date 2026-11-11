// Copyright 2023--present rgpot developers
//
// Takes the UmaPot arms a CPU host can reach: empty path, a non-package
// suffix, a zero atom count, a package that builds no edges, and the
// dtype names.

#include "rgpot/UmaPot/UmaPot.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string env_or_die(const char *name) {
  const char *path = std::getenv(name);
  if (path == nullptr || path[0] == '\0') {
    std::cerr << name << " is unset\n";
    std::exit(2);
  }
  return path;
}

void expect_throw(const char *label, const auto &fn) {
  try {
    fn();
    std::cerr << label << " did not throw\n";
    std::exit(1);
  } catch (const std::exception &) {
  }
}

} // namespace

int main() {
  expect_throw("empty path", [] {
    rgpot::UmaConfig cfg;
    rgpot::UmaPot pot(cfg);
  });
  expect_throw("plain pt", [] {
    rgpot::UmaConfig cfg;
    cfg.model_path = "model.pt";
    rgpot::UmaPot pot(cfg);
  });
  rgpot::UmaConfig cfg;
  cfg.model_path = env_or_die("UMA_FLOAT64_PT2");
  cfg.cutoff = std::numeric_limits<double>::quiet_NaN();
  cfg.device = "cpu";
  rgpot::UmaPot pot(cfg);
  (void)pot.caps();
  (void)pot.config().model_path;
  (void)pot.paramsKey();
  pot.setChargeSpin(0, 1);
  (void)pot.posDtypeName();
  (void)pot.modelDtypeName();
  expect_throw("zero atoms", [&] {
    double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    int z = 1;
    double pos = 0;
    rgpot::ForceOut out{&pos, 0, 0};
    const rgpot::ForceInput in{0, &pos, &z, box};
    pot.forceImpl(in, &out);
  });
  rgpot::UmaConfig none;
  none.model_path = env_or_die("UMA_NO_EDGES_PT2");
  rgpot::UmaPot empty(none);
  expect_throw("no edges", [&] {
    const double pos[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    const int z[3] = {1, 1, 1};
    const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
    double force[9] = {};
    rgpot::ForceOut out{force, 0, 0};
    const rgpot::ForceInput in{3, pos, z, box};
    empty.forceImpl(in, &out);
  });
  std::cout << "uma pot arms ok\n";
  return 0;
}
