// MIT License
// Copyright 2023--present rgpot developers
//
// UMA through the directly compiled UmaPot and through libuma_engine.so on
// the same inputs: writes energies and forces, and the median wall time of
// repeated calls. One mode per process so torch and the UMA code are loaded
// once.
//
//   uma_engine_bench MODE MODEL.pt2 GEOM.xyz [ENGINE.so] [NCALLS] [OUT.bin]
//
// MODE is one of
//   direct-force    UmaPot::forceImpl on ForceInput/ForceOut
//   direct-operator UmaPot::operator() on AtomMatrix (the path gpr_optim's
//                   PotentialWrapper takes)
//   engine          rgpot_engine_force from ENGINE.so (dlopen)
//
// Geometry is an xyz file placed in a 25 Angstrom cubic cell, charge 0,
// spin 1, task omol, cpu.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>

#include "rgpot/UmaPot/UmaPot.hpp"
#include "rgpot/engine_c_abi.h"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"

namespace {

int atomic_number(const std::string &symbol) {
  static const std::map<std::string, int> table = {
      {"H", 1},  {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5},  {"C", 6},
      {"N", 7},  {"O", 8},  {"F", 9},  {"Ne", 10}, {"Na", 11}, {"Mg", 12},
      {"Al", 13}, {"Si", 14}, {"P", 15}, {"S", 16}, {"Cl", 17}, {"Br", 35}};
  const auto it = table.find(symbol);
  if (it == table.end()) {
    std::fprintf(stderr, "unknown element %s\n", symbol.c_str());
    std::exit(2);
  }
  return it->second;
}

struct Geometry {
  std::vector<double> pos; // 3 * n, Angstrom
  std::vector<int> z;
};

Geometry read_xyz(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(2);
  }
  size_t n = 0;
  std::string line;
  std::getline(in, line);
  n = static_cast<size_t>(std::stoul(line));
  std::getline(in, line); // comment
  Geometry g;
  for (size_t i = 0; i < n; ++i) {
    std::getline(in, line);
    std::istringstream ls(line);
    std::string sym;
    double x, y, zc;
    ls >> sym >> x >> y >> zc;
    g.z.push_back(atomic_number(sym));
    // Centre in the 25 A cell the packages are exported for.
    g.pos.insert(g.pos.end(), {x + 12.5, y + 12.5, zc + 12.5});
  }
  return g;
}

double median_us(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

using Clock = std::chrono::steady_clock;

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s MODE MODEL.pt2 GEOM.xyz [ENGINE.so] [NCALLS] "
                 "[OUT.bin]\n",
                 argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  const std::string model = argv[2];
  const Geometry g = read_xyz(argv[3]);
  const std::string engine_path = argc > 4 ? argv[4] : "";
  const int ncalls = argc > 5 ? std::atoi(argv[5]) : 200;
  const char *out_path = argc > 6 ? argv[6] : nullptr;
  const long n = static_cast<long>(g.z.size());
  const std::array<double, 9> box = {25, 0, 0, 0, 25, 0, 0, 0, 25};

  std::vector<double> forces(3 * n, 0.0);
  double energy = 0.0;
  std::vector<double> times_us;
  const int warmup = 5;

  if (mode == "direct-force" || mode == "direct-operator") {
    rgpot::UmaConfig cfg;
    cfg.model_path = model;
    cfg.device = "cpu";
    cfg.task_name = "omol";
    cfg.charge = 0;
    cfg.spin = 1;
    rgpot::UmaPot pot(cfg);
    rgpot::types::AtomMatrix positions(n, 3);
    for (long i = 0; i < n; ++i) {
      for (int d = 0; d < 3; ++d) {
        positions(i, d) = g.pos[3 * i + d];
      }
    }
    const std::array<std::array<double, 3>, 3> boxm = {
        {{25, 0, 0}, {0, 25, 0}, {0, 0, 25}}};
    for (int it = 0; it < warmup + ncalls; ++it) {
      const auto t0 = Clock::now();
      if (mode == "direct-force") {
        rgpot::ForceInput in{static_cast<size_t>(n), g.pos.data(), g.z.data(),
                             box.data()};
        rgpot::ForceOut out{forces.data(), 0.0, 0.0};
        pot.forceImpl(in, &out);
        energy = out.energy;
      } else {
        auto [e, f, v] = pot(positions, g.z, boxm);
        (void)v;
        energy = e;
        for (long i = 0; i < n; ++i) {
          for (int d = 0; d < 3; ++d) {
            forces[3 * i + d] = f(i, d);
          }
        }
      }
      const auto t1 = Clock::now();
      if (it >= warmup) {
        times_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
  } else if (mode == "engine") {
    void *handle = dlopen(engine_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      std::fprintf(stderr, "dlopen: %s\n", dlerror());
      return 2;
    }
    using CreateFn = RgpotEnginePot *(*)(const void *, size_t, char *, size_t);
    using DestroyFn = void (*)(RgpotEnginePot *);
    using ForceFn = int (*)(RgpotEnginePot *, long, const double *, const int *,
                            double *, double *, double *, const double *,
                            rgpot_engine_coord_transform, void *);
    using VersionFn = int (*)();
    auto version = reinterpret_cast<VersionFn>(
        dlsym(handle, "rgpot_engine_abi_version"));
    auto create = reinterpret_cast<CreateFn>(dlsym(handle, "rgpot_engine_create"));
    auto destroy =
        reinterpret_cast<DestroyFn>(dlsym(handle, "rgpot_engine_destroy"));
    auto force = reinterpret_cast<ForceFn>(dlsym(handle, "rgpot_engine_force"));
    if (!version || !create || !destroy || !force ||
        version() != RGPOT_ENGINE_ABI_VERSION) {
      std::fprintf(stderr, "engine symbols or version mismatch\n");
      return 2;
    }
    capnp::MallocMessageBuilder msg;
    auto params = msg.initRoot<::UmaParams>();
    params.setModelPath(model);
    params.setTaskName("omol");
    params.setDevice("cpu");
    params.setCharge(0);
    params.setSpin(1);
    auto words = capnp::messageToFlatArray(msg);
    auto bytes = words.asBytes();
    char err[512] = {0};
    RgpotEnginePot *pot = create(bytes.begin(), bytes.size(), err, sizeof(err));
    if (!pot) {
      std::fprintf(stderr, "create: %s\n", err);
      return 2;
    }
    for (int it = 0; it < warmup + ncalls; ++it) {
      const auto t0 = Clock::now();
      const int rc = force(pot, n, g.pos.data(), g.z.data(), forces.data(),
                           &energy, nullptr, box.data(), nullptr, nullptr);
      const auto t1 = Clock::now();
      if (rc != 0) {
        std::fprintf(stderr, "force rc=%d\n", rc);
        return 2;
      }
      if (it >= warmup) {
        times_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    destroy(pot);
  } else {
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
  }

  std::printf("%s natoms=%ld energy=%a median_us=%.1f min_us=%.1f\n",
              mode.c_str(), n, energy, median_us(times_us),
              *std::min_element(times_us.begin(), times_us.end()));
  if (out_path) {
    std::ofstream out(out_path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&energy), sizeof(energy));
    out.write(reinterpret_cast<const char *>(forces.data()),
              static_cast<std::streamsize>(forces.size() * sizeof(double)));
  }
  return 0;
}
