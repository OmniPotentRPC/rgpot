// MIT License
// Copyright 2023--present rgpot developers
//
// ns/atom of repeated forceImpl calls on silicon diamond: TersoffSiC with
// a rebuild per call (default), TersoffSiC with a held skinned list, and
// the Fortran gather behind TersoffPot when it is built.
//
// jitter: each call is the lattice plus a fresh uniform offset of at most
// --amp per coordinate, cycling over 16 frames. walk: each call adds a
// uniform step of at most --step per coordinate to the last, so the held
// list rebuilds now and then. Every mode replays the same frames.
//
// tersoff_held_bench [--nrep 4,8] [--runs 7] [--calls 200] [--warm 20]
//                    [--skin 0.3] [--amp 0.01] [--step 0.005] [--seed 1]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "rgpot/Tersoff/TersoffSiC.hpp"
#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif

namespace {

constexpr double kA0 = 5.432;

struct Opts {
  std::vector<int> nrep{4, 8};
  int runs = 7;
  int calls = 200;
  int warm = 20;
  double skin = 0.3;
  double amp = 0.01;
  double step = 0.005;
  std::uint32_t seed = 1;
};

Opts parse(int argc, char **argv) {
  Opts o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const char *v = argv[i + 1];
    if (k == "--nrep") {
      o.nrep.clear();
      std::stringstream ss(v);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        o.nrep.push_back(std::atoi(tok.c_str()));
      }
    } else if (k == "--runs") {
      o.runs = std::atoi(v);
    } else if (k == "--calls") {
      o.calls = std::atoi(v);
    } else if (k == "--warm") {
      o.warm = std::atoi(v);
    } else if (k == "--skin") {
      o.skin = std::atof(v);
    } else if (k == "--amp") {
      o.amp = std::atof(v);
    } else if (k == "--step") {
      o.step = std::atof(v);
    } else if (k == "--seed") {
      o.seed = static_cast<std::uint32_t>(std::atol(v));
    } else {
      std::cerr << "unknown option " << k << "\n";
      std::exit(2);
    }
  }
  return o;
}

std::vector<double> diamond(int nrep) {
  const double basis[8][3] = {{0, 0, 0},          {0.5, 0.5, 0},
                              {0.5, 0, 0.5},      {0, 0.5, 0.5},
                              {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                              {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  std::vector<double> pos;
  for (int ix = 0; ix < nrep; ++ix) {
    for (int iy = 0; iy < nrep; ++iy) {
      for (int iz = 0; iz < nrep; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * kA0);
          pos.push_back((iy + b[1]) * kA0);
          pos.push_back((iz + b[2]) * kA0);
        }
      }
    }
  }
  return pos;
}

double uniform(std::mt19937 &gen, double amp) {
  return amp * (2.0 * (static_cast<double>(gen()) / 4294967295.0) - 1.0);
}

/// Frame f of a scenario. jitter cycles 16 frames around the lattice.
std::vector<std::vector<double>> frames(const std::vector<double> &base,
                                        const std::string &scenario,
                                        int count, const Opts &o) {
  std::mt19937 gen(o.seed);
  std::vector<std::vector<double>> out;
  if (scenario == "jitter") {
    for (int f = 0; f < 16; ++f) {
      out.push_back(base);
      for (double &x : out.back()) {
        x += uniform(gen, o.amp);
      }
    }
    return out;
  }
  std::vector<double> cur = base;
  for (int f = 0; f < count; ++f) {
    for (double &x : cur) {
      x += uniform(gen, o.step);
    }
    out.push_back(cur);
  }
  return out;
}

struct Call {
  double energy = 0.0;
  std::vector<double> force;
};

template <typename Pot>
void evalInto(const Pot &pot, const std::vector<double> &pos,
              const std::vector<int> &types, const double *box, Call &c) {
  c.force.resize(pos.size());
  rgpot::ForceInput in{.nAtoms = types.size(),
                       .pos = pos.data(),
                       .atmnrs = types.data(),
                       .box = box};
  rgpot::ForceOut out{.F = c.force.data(),
                      .energy = 0.0,
                      .variance = 0.0,
                      .stress = {},
                      .has_stress = 0};
  pot.forceImpl(in, &out);
  c.energy = out.energy;
}

template <typename Make>
std::vector<double> timeMode(Make make,
                             const std::vector<std::vector<double>> &fr,
                             const std::vector<int> &types, const double *box,
                             const Opts &o) {
  std::vector<double> ns_per_atom;
  Call c;
  const auto frame = [&](int k) -> const std::vector<double> & {
    return fr[static_cast<std::size_t>(k) % fr.size()];
  };
  for (int run = 0; run < o.runs; ++run) {
    auto pot = make();
    for (int k = 0; k < o.warm; ++k) {
      evalInto(*pot, frame(k), types, box, c);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = o.warm; k < o.warm + o.calls; ++k) {
      evalInto(*pot, frame(k), types, box, c);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    ns_per_atom.push_back(ns / (static_cast<double>(o.calls) *
                                static_cast<double>(types.size())));
  }
  return ns_per_atom;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

void report(const char *mode, int n, const std::string &scenario,
            const std::vector<double> &t, const Opts &o, long rebuilds) {
  std::cout << "result n_atoms=" << n << " scenario=" << scenario
            << " mode=" << mode << " runs=" << o.runs << " calls=" << o.calls
            << " median_ns_per_atom=" << median(t)
            << " min=" << *std::min_element(t.begin(), t.end())
            << " max=" << *std::max_element(t.begin(), t.end());
  if (rebuilds >= 0) {
    std::cout << " rebuilds=" << rebuilds;
  }
  std::cout << " all=";
  for (std::size_t i = 0; i < t.size(); ++i) {
    std::cout << (i ? "," : "") << t[i];
  }
  std::cout << "\n";
}

} // namespace

int main(int argc, char **argv) {
  const Opts o = parse(argc, argv);
  std::cout.precision(6);
#if defined(__VERSION__)
  std::cout << "compiler: " << __VERSION__ << "\n";
#elif defined(_MSC_FULL_VER)
  std::cout << "compiler: MSVC " << _MSC_FULL_VER << "\n";
#endif
  std::cout << "skin: " << o.skin << " amp: " << o.amp << " step: " << o.step
            << " seed: " << o.seed << " warm: " << o.warm << "\n";
  for (int nrep : o.nrep) {
    const std::vector<double> base = diamond(nrep);
    const std::vector<int> types(base.size() / 3, 14);
    const int n = static_cast<int>(types.size());
    const double len = nrep * kA0;
    const double box[9] = {len, 0, 0, 0, len, 0, 0, 0, len};
    for (const std::string scenario : {"jitter", "walk"}) {
      const auto fr = frames(base, scenario, o.warm + o.calls, o);
      const auto frame = [&](int k) -> const std::vector<double> & {
        return fr[static_cast<std::size_t>(k) % fr.size()];
      };

      // Untimed replay: rebuild count and held against default.
      rgpot::TersoffSiC plain;
      rgpot::TersoffSiC held;
      held.setHeldSkin(o.skin);
      long rebuilds = 0;
      long same = 0;
      double max_de = 0.0;
      double max_rel = 0.0;
      double max_df = 0.0;
      Call a;
      Call b;
      std::vector<double> last_ref;
      for (int k = 0; k < o.warm + o.calls; ++k) {
        evalInto(held, frame(k), types, box, a);
        evalInto(plain, frame(k), types, box, b);
        if (held.heldList().ref != last_ref) {
          ++rebuilds;
          last_ref = held.heldList().ref;
        }
        bool bits = std::memcmp(&a.energy, &b.energy, sizeof(double)) == 0 &&
                    std::memcmp(a.force.data(), b.force.data(),
                                a.force.size() * sizeof(double)) == 0;
        same += bits ? 1 : 0;
        max_de = std::max(max_de, std::fabs(a.energy - b.energy));
        max_rel =
            std::max(max_rel, std::fabs(a.energy - b.energy) / std::fabs(b.energy));
        for (std::size_t i = 0; i < a.force.size(); ++i) {
          max_df = std::max(max_df, std::fabs(a.force[i] - b.force[i]));
        }
      }
      std::cout << "check n_atoms=" << n << " scenario=" << scenario
                << " held_vs_default calls=" << o.warm + o.calls
                << " bit_identical=" << same << " max_abs_dE=" << max_de
                << " max_rel_dE=" << max_rel << " max_abs_dF=" << max_df
                << " rebuilds=" << rebuilds << "\n";
#ifdef RGPOT_HAS_FORTRAN_POTS
      {
        rgpot::fortranpots::TersoffPot gather;
        evalInto(gather, frame(0), types, box, a);
        evalInto(plain, frame(0), types, box, b);
        double df = 0.0;
        for (std::size_t i = 0; i < a.force.size(); ++i) {
          df = std::max(df, std::fabs(a.force[i] - b.force[i]));
        }
        std::cout << "check n_atoms=" << n << " scenario=" << scenario
                  << " fortran_vs_default frame=0 abs_dE="
                  << std::fabs(a.energy - b.energy) << " max_abs_dF=" << df
                  << "\n";
      }
#endif

      report("default", n, scenario,
             timeMode([] { return std::make_unique<rgpot::TersoffSiC>(); }, fr,
                      types, box, o),
             o, -1);
      report("held", n, scenario,
             timeMode(
                 [&] {
                   auto p = std::make_unique<rgpot::TersoffSiC>();
                   p->setHeldSkin(o.skin);
                   return p;
                 },
                 fr, types, box, o),
             o, rebuilds);
#ifdef RGPOT_HAS_FORTRAN_POTS
      report("fortran", n, scenario,
             timeMode(
                 [] {
                   return std::make_unique<rgpot::fortranpots::TersoffPot>();
                 },
                 fr, types, box, o),
             o, -1);
#endif
    }
  }
  return 0;
}
