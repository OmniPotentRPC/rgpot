// MIT License
// Copyright 2023--present rgpot developers
//
// Wall time (median, minimum and maximum over the repetitions) of the XcKernel
// Fock (xck_gga_r_o1) and fxc (xck_gga_r_o2) contractions on the committed
// H2O/STO-3G operands and on a random-grid case generated here. Usage, from
// the repository root:
//   xckernel_bench [--nbf N] [--npts N] [--reps N]
// Thread count is the BLAS library's own setting, for example
// OPENBLAS_NUM_THREADS.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "npy_io.hpp"
#include "rgpot/XcKernel/XcKernel.hpp"

using rgpot::XcGrid;
using rgpot::XcKernel;
using rgpot::testio::load_npz;
using Clock = std::chrono::steady_clock;

namespace {

struct Timing {
  double median;
  double min;
  double max;
};

Timing time_contract(const XcKernel &k, const XcGrid &g,
                     const std::map<std::string, const double *> &scal,
                     std::int64_t nbf, int reps) {
  std::vector<double> out(static_cast<std::size_t>(nbf * nbf), 0.0);
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(std::max(reps, 0)));
  for (int r = 0; r < reps; ++r) {
    std::fill(out.begin(), out.end(), 0.0);
    const auto t0 = Clock::now();
    if (k.contract(g, scal, out.data()) != 0) {
      std::fprintf(stderr, "contract failed: %s\n", k.name().c_str());
      std::exit(3);
    }
    const std::chrono::duration<double, std::milli> dt = Clock::now() - t0;
    samples.push_back(dt.count());
  }
  if (samples.empty()) {
    std::fprintf(stderr, "xckernel_bench: --reps must be at least 1\n");
    std::exit(2);
  }
  std::sort(samples.begin(), samples.end());
  // Upper median: for an odd count this is the middle sample.
  return {samples[samples.size() / 2], samples.front(), samples.back()};
}

} // namespace

int main(int argc, char **argv) {
  std::int64_t nbf_big = 200;
  std::int64_t npts_big = 100000;
  int reps = 3;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--nbf")) {
      nbf_big = std::atoll(argv[i + 1]);
    } else if (!std::strcmp(argv[i], "--npts")) {
      npts_big = std::atoll(argv[i + 1]);
    } else if (!std::strcmp(argv[i], "--reps")) {
      reps = std::atoi(argv[i + 1]);
    }
  }
  const char *kernels[] = {"xck_gga_r_o1", "xck_gga_r_o2"};

  {
    auto op = load_npz("CppCore/tests/data/xckernel/"
                       "mol_h2o_sto3g_lvl3_operands.npz");
    const auto nbf = static_cast<std::int64_t>(op.at("chi").shape[0]);
    const auto npts = static_cast<std::int64_t>(op.at("chi").shape[1]);
    XcGrid g;
    g.nbf = nbf;
    g.npts = npts;
    g.chi = op.at("chi").data.data();
    g.dchi = op.at("dchi").data.data();
    for (const char *name : kernels) {
      XcKernel k(name);
      std::map<std::string, const double *> scal;
      for (const auto &n : k.scalNames()) {
        scal[n] = op.at(n).data.data();
      }
      const Timing t = time_contract(k, g, scal, nbf, 20 * reps);
      std::printf("h2o-sto3g nbf=%lld npts=%lld %s median %.3f min %.3f max "
                  "%.3f ms (%d reps)\n",
                  static_cast<long long>(nbf), static_cast<long long>(npts),
                  name, t.median, t.min, t.max, 20 * reps);
    }
  }

  {
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto fill = [&](std::size_t n) {
      std::vector<double> v(n);
      for (auto &x : v) {
        x = dist(rng);
      }
      return v;
    };
    const auto nb = static_cast<std::size_t>(nbf_big);
    const auto np = static_cast<std::size_t>(npts_big);
    auto chi = fill(nb * np);
    auto dchi = fill(3 * nb * np);
    XcGrid g;
    g.nbf = nbf_big;
    g.npts = npts_big;
    g.chi = chi.data();
    g.dchi = dchi.data();
    for (const char *name : kernels) {
      XcKernel k(name);
      std::map<std::string, std::vector<double>> store;
      std::map<std::string, const double *> scal;
      for (const auto &n : k.scalNames()) {
        store[n] = fill(np);
        scal[n] = store[n].data();
      }
      const Timing t = time_contract(k, g, scal, nbf_big, reps);
      std::printf("random nbf=%lld npts=%lld %s median %.1f min %.1f max %.1f "
                  "ms (%d reps)\n",
                  static_cast<long long>(nbf_big),
                  static_cast<long long>(npts_big), name, t.median, t.min,
                  t.max, reps);
    }
  }
  return 0;
}
