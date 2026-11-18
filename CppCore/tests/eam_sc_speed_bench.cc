// MIT License
// Copyright 2023--present rgpot developers
//
// Reproducible EAMTable and Sutton-Chen call benchmark for speed work.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgpot/EAMTable/EAMTablePot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"

namespace {

struct Stats {
  double median = 0.0;
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double stdev = 0.0;
};

std::uint64_t fnv(std::uint64_t h, const void *data, std::size_t len) {
  const auto *b = static_cast<const unsigned char *>(data);
  for (std::size_t i = 0; i < len; ++i) {
    h ^= b[i];
    h *= 1099511628211ULL;
  }
  return h;
}

struct Opts {
  std::string task = "both";
  int repeats = 8;
  int calls = 32;
  double sigma = 0.02;
  std::uint64_t seed = 7;
  std::string pt38 = "CppCore/tests/data/sc108/pt38";
  std::string pt75 = "CppCore/tests/data/sc108/pt75";
};

Opts parse(int argc, char **argv) {
  Opts o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const std::string v = argv[i + 1];
    if (k == "--task") {
      o.task = v;
    } else if (k == "--repeats") {
      o.repeats = std::atoi(v.c_str());
    } else if (k == "--calls") {
      o.calls = std::atoi(v.c_str());
    } else if (k == "--sigma") {
      o.sigma = std::atof(v.c_str());
    } else if (k == "--seed") {
      o.seed =
          static_cast<std::uint64_t>(std::strtoull(v.c_str(), nullptr, 10));
    } else if (k == "--pt38") {
      o.pt38 = v;
    } else if (k == "--pt75") {
      o.pt75 = v;
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      std::exit(2);
    }
  }
  return o;
}

Stats computeStats(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  Stats s{};
  const std::size_t n = samples.size();
  s.min = samples.front();
  s.max = samples.back();
  s.median = (n % 2 == 1) ? samples[n / 2]
                          : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
  s.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
           static_cast<double>(samples.size());
  double var = 0.0;
  for (double x : samples) {
    const double d = x - s.mean;
    var += d * d;
  }
  var /= static_cast<double>(samples.size());
  s.stdev = std::sqrt(var);
  return s;
}

std::vector<double> fcc(int nx, int ny, int nz, double a, double *box) {
  const double basis[4][3] = {
      {0.0, 0.0, 0.0},
      {0.0, 0.5, 0.5},
      {0.5, 0.0, 0.5},
      {0.5, 0.5, 0.0},
  };
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(nx * ny * nz * 12));
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * a);
          pos.push_back((iy + b[1]) * a);
          pos.push_back((iz + b[2]) * a);
        }
      }
    }
  }
  std::fill(box, box + 9, 0.0);
  box[0] = nx * a;
  box[4] = ny * a;
  box[8] = nz * a;
  return pos;
}

std::vector<double> pointFile(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  double head[5];
  for (double &h : head) {
    in >> h;
  }
  std::vector<double> pos;
  double x = 0.0;
  while (in >> x) {
    pos.push_back(x);
  }
  return pos;
}

std::vector<std::vector<double>> frames(const std::vector<double> &base, double sigma,
                                        std::uint64_t seed, int nframes) {
  std::vector<std::vector<double>> out(static_cast<std::size_t>(nframes), base);
  std::uint64_t state = seed ? seed : 1;
  auto next_u01 = [&state]() {
    state = state * 6364136223846793005ULL + 1ULL;
    const std::uint64_t bits = (state >> 11) | 0x3ff0000000000000ULL;
    const double v = *reinterpret_cast<const double *>(&bits);
    return v - 1.0;
  };
  for (auto &f : out) {
    for (double &x : f) {
      // Box-Muller from two simple uniform draws.
      const double u1 = std::max(next_u01(), 1e-12);
      const double u2 = next_u01();
      const double mag = sigma * std::sqrt(-2.0 * std::log(u1));
      x += mag * std::cos(6.283185307179586 * u2);
    }
  }
  return out;
}

Stats benchEAM(const Opts &o) {
  double box[9] = {};
  const std::vector<double> pos = fcc(10, 10, 10, 3.615, box);
  const int n = static_cast<int>(pos.size() / 3);
  std::vector<int> z(static_cast<std::size_t>(n), 29);
  rgpot::EAMTablePot pot({.path = "CppCore/tests/data/eam_table/Cu_u3.eam",
                          .format = rgpot::EAMTableFormat::Funcfl,
                          .periodic = true,
                          .stress = false,
                          .skin = 1.0});
  std::vector<double> f(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = f.data();
  const rgpot::ForceInput in{static_cast<std::size_t>(n), pos.data(), z.data(), box};
  pot.forceImpl(in, &out);
  pot.forceImpl(in, &out);

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(o.repeats));
  std::uint64_t reuse = 0;
  std::uint64_t bits = 1469598103934665603ULL;
  for (int r = 0; r < o.repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < o.calls; ++k) {
      pot.forceImpl(in, &out);
      if (pot.listReused()) {
        ++reuse;
      }
      bits = fnv(bits, &out.energy, sizeof(out.energy));
      bits = fnv(bits, f.data(), f.size() * sizeof(double));
    }
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() /
                      static_cast<double>(o.calls));
  }
  const Stats s = computeStats(samples);
  std::printf("bench=eam atoms=%d repeats=%d calls=%d reused_calls=%llu "
              "builds=%llu energy=%.15g bits=%016llx\n",
              n, o.repeats, o.calls, static_cast<unsigned long long>(reuse),
              static_cast<unsigned long long>(pot.listBuilds()), out.energy,
              static_cast<unsigned long long>(bits));
  return s;
}

double run_sc_case(const rgpot_sc_config &cfg, const std::vector<std::vector<double>> &fr,
                   std::vector<double> &f, int calls) {
  double e = 0.0;
  const int32_t n = static_cast<int32_t>(fr.front().size() / 3);
  for (int k = 0; k < calls; ++k) {
    const auto &x = fr[static_cast<std::size_t>(k % static_cast<int>(fr.size()))];
    if (rgpot_sutton_chen_energy_forces(&cfg, n, x.data(), nullptr, &e, f.data()) != 0) {
      throw std::runtime_error("rgpot_sutton_chen_energy_forces failed");
    }
  }
  return e;
}

void benchSC(const Opts &o) {
  const std::vector<double> pt38 = pointFile(o.pt38);
  const std::vector<double> pt75 = pointFile(o.pt75);
  const double pt_a = rgpot::sutton_chen_platinum().a;
  const double au_a = rgpot::sutton_chen_gold().a;
  std::vector<double> au38 = pt38;
  std::vector<double> au75 = pt75;
  const double scale = au_a / pt_a;
  for (double &x : au38) {
    x *= scale;
  }
  for (double &x : au75) {
    x *= scale;
  }

  struct Case {
    const char *name;
    std::vector<double> pos;
    rgpot_sc_config cfg;
  };
  std::vector<Case> cases{
      {"pt38", pt38, {1.98330e-2, 34.408, 3.92, 10, 8, 40.0, 0}},
      {"au38", au38, {1.27930e-2, 34.408, 4.08, 10, 8, 40.0, 0}},
      {"au75", au75, {1.27930e-2, 34.408, 4.08, 10, 8, 40.0, 0}},
  };
  for (const Case &c : cases) {
    const auto fr = frames(c.pos, o.sigma, o.seed, 16);
    std::vector<double> force(fr.front().size(), 0.0);
    run_sc_case(c.cfg, fr, force, 8);
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(o.repeats));
    double energy = 0.0;
    std::uint64_t bits = 1469598103934665603ULL;
    for (int r = 0; r < o.repeats; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      energy = run_sc_case(c.cfg, fr, force, o.calls);
      bits = fnv(bits, &energy, sizeof(energy));
      bits = fnv(bits, force.data(), force.size() * sizeof(double));
      const auto t1 = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                        static_cast<double>(o.calls));
    }
    const Stats s = computeStats(samples);
    std::printf("bench=sc case=%s atoms=%zu repeats=%d calls=%d "
                "median_us=%.6f min_us=%.6f max_us=%.6f stdev_us=%.6f energy=%.15g bits=%016llx\n",
                c.name, c.pos.size() / 3, o.repeats, o.calls, s.median, s.min, s.max,
                s.stdev, energy, static_cast<unsigned long long>(bits));
  }
}

} // namespace

int main(int argc, char **argv) {
  const Opts o = parse(argc, argv);
  std::printf("task=%s repeats=%d calls=%d sigma=%g seed=%llu\n", o.task.c_str(), o.repeats,
              o.calls, o.sigma, static_cast<unsigned long long>(o.seed));
  if (o.task == "eam" || o.task == "both") {
    const Stats s = benchEAM(o);
    std::printf("bench=eam median_ms=%.6f min_ms=%.6f max_ms=%.6f stdev_ms=%.6f\n", s.median,
                s.min, s.max, s.stdev);
  }
  if (o.task == "sc" || o.task == "both") {
    benchSC(o);
  }
  return 0;
}
