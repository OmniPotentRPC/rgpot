// MIT License
// Copyright 2023--present rgpot developers
//
// Wall time of the all-images pair sums against the minimum image. Not a
// test. One thread. Medians of seven warm calls after two discarded
// calls. The plain columns keep positions fixed, so both modes reuse their
// lists. The moved columns rebuild the lists on every call.
//
// "bulk" is a cubic fcc crystal at least twice the cutoff across, where
// both modes are exact and give the same numbers. The small cells are
// narrower than twice the cutoff: all-images on the cell itself, and the
// minimum image on the replicated supercell that gives the same energy
// per cell.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifdef RGPOT_HAS_EAM_TABLE
#include "rgpot/EAMTable/EAMTablePot.hpp"
#endif
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"

using rgpot::nlist::Images;

namespace {

struct Cell {
  std::vector<double> pos;
  std::vector<int> z;
  std::array<double, 9> box{};
};

std::string cpuModel() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("model name", 0) == 0) {
      const auto colon = line.find(':');
      if (colon != std::string::npos) {
        return line.substr(line.find_first_not_of(" \t", colon + 1));
      }
    }
  }
  return "unknown";
}

Cell cube(int n, double a, int z) {
  Cell c;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        for (const auto &b : basis) {
          c.pos.insert(c.pos.end(), {(i + b[0]) * a, (j + b[1]) * a, (k + b[2]) * a});
          c.z.push_back(z);
        }
      }
    }
  }
  c.box = {n * a, 0, 0, 0, n * a, 0, 0, 0, n * a};
  return c;
}

Cell primitive(double a, int z) {
  Cell c;
  c.box = {0.5 * a, 0.5 * a, 0, 0, 0.5 * a, 0.5 * a, 0.5 * a, 0, 0.5 * a};
  c.pos = {0.05 * a, -0.02 * a, 0.01 * a};
  c.z = {z};
  return c;
}

Cell triclinic(double a, int z) {
  Cell c;
  c.box = {a, 0, 0, 0.35 * a, 0.9 * a, 0, -0.2 * a, 0.3 * a, 1.1 * a};
  const double frac[3][3] = {{0.1, 0.2, 0.3}, {0.6, 0.55, 0.4}, {0.35, 0.8, 0.85}};
  for (const auto &f : frac) {
    for (int k = 0; k < 3; ++k) {
      c.pos.push_back(f[0] * c.box[static_cast<std::size_t>(k)] +
                      f[1] * c.box[static_cast<std::size_t>(3 + k)] +
                      f[2] * c.box[static_cast<std::size_t>(6 + k)]);
    }
    c.z.push_back(z);
  }
  return c;
}

Cell slab(double a, int z, double cutoff) {
  Cell c;
  const double side = a / std::sqrt(2.0);
  for (int k = 0; k < 3; ++k) {
    const double off = (k % 2 == 1) ? 0.5 * side : 0.0;
    c.pos.insert(c.pos.end(), {off, off, 0.5 * a * k});
    c.z.push_back(z);
  }
  c.box = {side, 0, 0, 0, side, 0, 0, 0, a + 2.0 * cutoff + 1.0};
  return c;
}

Cell replicate(const Cell &c, double cutoff) {
  double width[3];
  rgpot::nlist::perpendicularWidths(c.box.data(), width);
  int n[3];
  for (int k = 0; k < 3; ++k) {
    n[k] = std::max(1, static_cast<int>(std::ceil((2.0 * cutoff + 0.5) / width[k])));
  }
  Cell out;
  for (int a = 0; a < n[0]; ++a) {
    for (int b = 0; b < n[1]; ++b) {
      for (int d = 0; d < n[2]; ++d) {
        for (std::size_t i = 0; i < c.z.size(); ++i) {
          for (std::size_t k = 0; k < 3; ++k) {
            out.pos.push_back(c.pos[3 * i + k] + a * c.box[k] + b * c.box[3 + k] +
                              d * c.box[6 + k]);
          }
          out.z.push_back(c.z[i]);
        }
      }
    }
  }
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t j = 0; j < 3; ++j) {
      out.box[3 * k + j] = n[k] * c.box[3 * k + j];
    }
  }
  return out;
}

// With moved, every call alternates between the cell and a copy whose atoms
// sit 1.0 apart from it, past half of every skin here, so each call
// rebuilds its list.
template <class Pot>
double medianCall(const Pot &pot, const Cell &c, bool moved = false) {
  std::vector<double> f(c.pos.size());
  rgpot::ForceOut out{};
  out.F = f.data();
  std::vector<double> shifted = c.pos;
  for (std::size_t i = 0; i < c.z.size(); ++i) {
    const double s = static_cast<double>(i);
    shifted[3 * i] += std::sin(s);
    shifted[3 * i + 1] += std::cos(s);
  }
  const rgpot::ForceInput in[2] = {
      {c.z.size(), c.pos.data(), c.z.data(), c.box.data()},
      {c.z.size(), shifted.data(), c.z.data(), c.box.data()}};
  pot.forceImpl(in[0], &out);
  pot.forceImpl(in[moved ? 1 : 0], &out);
  std::vector<double> t;
  for (int r = 0; r < 7; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    pot.forceImpl(in[moved ? r % 2 : 0], &out);
    const auto t1 = std::chrono::steady_clock::now();
    t.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

template <class Pot>
void rows(const char *name, const Pot &minimum, const Pot &all, double a, int z,
          double cutoff) {
  const int narrowest = static_cast<int>(std::ceil((2.0 * cutoff + 0.5) / a));
  for (int n : {narrowest, 10, 20}) {
    if (n != narrowest && n <= narrowest) {
      continue;
    }
    const Cell bulk = cube(n, a, z);
    std::printf("%-10s %-16s %7zu %12.3e %12.3e %12.3e %12.3e\n", name, "bulk",
                bulk.z.size(), medianCall(minimum, bulk), medianCall(all, bulk),
                medianCall(minimum, bulk, true), medianCall(all, bulk, true));
  }
  const Cell small[3] = {primitive(a, z), triclinic(a, z), slab(a, z, cutoff)};
  const char *label[3] = {"fcc primitive", "triclinic", "slab"};
  for (int k = 0; k < 3; ++k) {
    const Cell big = replicate(small[k], cutoff);
    std::printf("%-10s %-16s %7zu %12.3e %12.3e %12.3e %12.3e  (minimum image on %zu)\n",
                name, label[k], small[k].z.size(), medianCall(minimum, big),
                medianCall(all, small[k]), medianCall(minimum, big, true),
                medianCall(all, small[k], true), big.z.size());
  }
}

} // namespace

int main() {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", "1");
#else
  setenv("OMP_NUM_THREADS", "1", 1);
#endif
  std::printf("cpu: %s\n", cpuModel().c_str());
  std::printf("one thread, median of 7 warm calls, seconds per call\n");
  std::printf("%-10s %-16s %7s %12s %12s %12s %12s\n", "kernel", "cell", "atoms",
              "minimum_s", "all_s", "min_moved_s", "all_moved_s");
  {
    const rgpot::LJConfig cfg{.u0 = 1.0, .cutoff = 2.5, .psi = 1.0};
    rows("LJ", rgpot::LJPot(cfg), rgpot::LJPot(cfg, Images::All), 1.56, 18, cfg.cutoff);
  }
  {
    const rgpot::MorseConfig cfg{.cutoff = 6.0};
    rows("Morse", rgpot::MorsePot(cfg), rgpot::MorsePot(cfg, Images::All), 3.92, 78,
         cfg.cutoff);
  }
  {
    rgpot::SuttonChenConfig cfg{.cutoff = 3.0, .periodic = true};
    const rgpot::SuttonChenPot minimum(cfg);
    cfg.images = Images::All;
    rows("SuttonChen", minimum, rgpot::SuttonChenPot(cfg), 1.43, 78, 3.0);
  }
#ifdef RGPOT_HAS_EAM_TABLE
  {
    const std::string path = "CppCore/tests/data/eam_table/Cu_u3.eam";
    const rgpot::EAMTablePot minimum(
        {.path = path, .format = rgpot::EAMTableFormat::Funcfl, .periodic = true});
    const rgpot::EAMTablePot all({.path = path,
                                  .format = rgpot::EAMTableFormat::Funcfl,
                                  .periodic = true,
                                  .images = Images::All});
    rows("EAMTable", minimum, all, 3.615, 29, 4.95);
  }
#endif
  return 0;
}
