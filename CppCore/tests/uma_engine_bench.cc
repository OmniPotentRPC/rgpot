// MIT License
// Copyright 2023--present rgpot developers
//
// UMA through the directly compiled UmaPot and through libuma_engine.so on
// the same inputs: writes energies and forces, and the median wall time of
// repeated calls. One mode per process so torch and the UMA code are loaded
// once.
//
//   uma_engine_bench MODE MODEL.pt2 GEOM.xyz [ENGINE.so] [NCALLS] [OUT.bin]
//   uma_engine_bench dtype-cost FLOAT32.pt2 FLOAT64.pt2 [NCALLS]
//
// MODE is one of
//   dtype-cost      per-call cost of the two packages at 10, 50, 200 and
//                   1000 atoms. Median, min, max and interquartile range
//                   over at least 20 calls after warmup, then the raw
//                   samples. Timed on CPU. cuda=unavailable when this
//                   process has no CUDA device; a CPU package is not
//                   retimed on a device. The packages must accept those
//                   sizes.
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
#include <thread>
#include <vector>

#include <torch/torch.h>

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

double quantile_us(const std::vector<double> &sorted, double q) {
  if (sorted.empty())
    return 0.0;
  const double idx = q * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  const double w = idx - static_cast<double>(lo);
  return sorted[lo] * (1.0 - w) + sorted[hi] * w;
}

double median_us(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return quantile_us(v, 0.5);
}

using Clock = std::chrono::steady_clock;

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("model name", 0) != 0)
      continue;
    const auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    std::string name = line.substr(colon + 1);
    while (!name.empty() && name.front() == ' ')
      name.erase(name.begin());
    return name;
  }
  return "unknown";
}

// Cubic lattice, 2 Å spacing, sitting inside a 50 Å cell. The cost
// packages are exported with a 6 Å cutoff, so each atom has neighbours
// and the edge count stays inside the dynamic range.
Geometry lattice(int n) {
  int side = 1;
  while (side * side * side < n)
    ++side;
  Geometry g;
  g.z.assign(static_cast<size_t>(n), 6);
  int filled = 0;
  for (int z = 0; z < side && filled < n; ++z) {
    for (int y = 0; y < side && filled < n; ++y) {
      for (int x = 0; x < side && filled < n; ++x, ++filled) {
        g.pos.push_back(15.0 + 2.0 * x);
        g.pos.push_back(15.0 + 2.0 * y);
        g.pos.push_back(15.0 + 2.0 * z);
      }
    }
  }
  return g;
}

int dtype_cost(const std::string &f32_path, const std::string &f64_path,
               int ncalls) {
  if (ncalls < 20) {
    std::fprintf(stderr, "dtype-cost needs at least 20 timed calls\n");
    return 2;
  }
  const int warmup = 5;
  const int counts[] = {10, 50, 200, 1000};
  const std::array<double, 9> box = {50, 0, 0, 0, 50, 0, 0, 0, 50};
  const bool cuda = torch::cuda::is_available();
  std::printf("cpu=%s\n", cpu_model().c_str());
  std::printf("cpu_threads=%d\n",
              static_cast<int>(std::thread::hardware_concurrency()));
  std::printf("cuda=%s\n", cuda ? "available" : "unavailable");
  const char *devices[] = {"cpu"};
  const int ndevices = 1;
  (void)ndevices;
  if (cuda) {
    std::fprintf(stderr,
                 "dtype-cost: a CUDA device is visible, but these packages "
                 "were exported for CPU\n");
  }

  struct Spec {
    const char *label;
    const std::string *path;
  };
  const Spec specs[] = {{"float32", &f32_path}, {"float64", &f64_path}};
  for (const Spec &spec : specs) {
    rgpot::UmaConfig cfg;
    cfg.model_path = *spec.path;
    cfg.device = "cpu";
    cfg.task_name = "omol";
    cfg.charge = 0;
    cfg.spin = 1;
    rgpot::UmaPot pot(cfg);
    std::printf("package=%s pos_dtype=%s model_dtype=%s\n", spec.label,
                pot.posDtypeName(), pot.modelDtypeName());
    for (int n : counts) {
      const Geometry g = lattice(n);
      std::vector<double> forces(static_cast<size_t>(n) * 3, 0.0);
      double energy = 0.0;
      std::vector<double> times_us;
      times_us.reserve(static_cast<size_t>(ncalls));
      for (int it = 0; it < warmup + ncalls; ++it) {
        const auto t0 = Clock::now();
        rgpot::ForceInput in{static_cast<size_t>(n), g.pos.data(), g.z.data(),
                             box.data()};
        rgpot::ForceOut out{forces.data(), 0.0, 0.0};
        pot.forceImpl(in, &out);
        energy = out.energy;
        const auto t1 = Clock::now();
        if (it >= warmup) {
          times_us.push_back(
              std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
      }
      std::vector<double> sorted = times_us;
      std::sort(sorted.begin(), sorted.end());
      const double q1 = quantile_us(sorted, 0.25);
      const double med = quantile_us(sorted, 0.5);
      const double q3 = quantile_us(sorted, 0.75);
      std::printf("dtype=%s natoms=%d device=cpu energy=%.17g "
                  "median_us=%.3f min_us=%.3f max_us=%.3f iqr_us=%.3f n=%zu\n",
                  spec.label, n, energy, med, sorted.front(), sorted.back(),
                  q3 - q1, times_us.size());
      std::printf("raw_us=");
      for (size_t i = 0; i < times_us.size(); ++i) {
        if (i)
          std::printf(",");
        std::printf("%.3f", times_us[i]);
      }
      std::printf("\n");
    }
  }
  if (!cuda)
    std::printf("cuda_timings=not_run\n");
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc >= 2 && std::string(argv[1]) == "dtype-cost") {
    if (argc < 4) {
      std::fprintf(stderr,
                   "usage: %s dtype-cost FLOAT32.pt2 FLOAT64.pt2 [NCALLS]\n",
                   argv[0]);
      return 2;
    }
    const int ncalls = argc > 4 ? std::atoi(argv[4]) : 20;
    return dtype_cost(argv[2], argv[3], ncalls);
  }
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
  std::string precision;

  if (mode == "dlsym-cost") {
    // Cost of the three RTLD_DEFAULT lookups the engine makes per call to
    // see whether a Python host holds the GIL, with torch loaded.
    for (int it = 0; it < warmup + ncalls; ++it) {
      const auto t0 = Clock::now();
      volatile void *a = dlsym(RTLD_DEFAULT, "PyGILState_Check");
      volatile void *b = dlsym(RTLD_DEFAULT, "PyEval_SaveThread");
      volatile void *c = dlsym(RTLD_DEFAULT, "PyEval_RestoreThread");
      (void)a;
      (void)b;
      (void)c;
      const auto t1 = Clock::now();
      if (it >= warmup) {
        times_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
  } else if (mode == "direct-force" || mode == "direct-operator") {
    rgpot::UmaConfig cfg;
    cfg.model_path = model;
    cfg.device = "cpu";
    cfg.task_name = "omol";
    cfg.charge = 0;
    cfg.spin = 1;
    rgpot::UmaPot pot(cfg);
    precision = std::string("pos_dtype=") + pot.posDtypeName() +
                " model_dtype=" + pot.modelDtypeName();
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

  const std::string head = precision.empty() ? std::string() : precision + " ";
  std::printf("%s%s natoms=%ld energy=%a median_us=%.1f min_us=%.1f\n",
              head.c_str(), mode.c_str(), n, energy, median_us(times_us),
              *std::min_element(times_us.begin(), times_us.end()));
  if (out_path) {
    std::ofstream out(out_path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&energy), sizeof(energy));
    out.write(reinterpret_cast<const char *>(forces.data()),
              static_cast<std::streamsize>(forces.size() * sizeof(double)));
  }
  return 0;
}
