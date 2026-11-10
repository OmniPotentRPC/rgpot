// MIT License
// Copyright 2023--present rgpot developers
//
// Wall time of one warm force call. Kinds:
//   ljcluster   free LJClusterPot, cutoff from --cutoff (default 2.5)
//   ljpot       LJPot on the same blob, in a periodic cell large enough
//               that no image lies inside the cutoff
//   sutton      free SuttonChenPot, cutoff from --cutoff (default 3)
//   ljpot_asv   LJPot, coordinates in a 10-cube, cell 40, cutoff 15
//   linkcell    5x5x5 lattice, cell 25, LJPot cutoff 6
//   ljfile      LJClusterPot on a coordinate file (--file), cutoff 15
//   truncate    energies of Pt38, a 147-atom Mackay icosahedron and a
//               1000-atom blob at cutoff 0, 3 and 40
//
// The pair kernels are serial. --threads is recorded and does not change
// the arithmetic. Prints one line per timed call.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/nlist/PairListCache.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"

namespace {

using clock_type = std::chrono::steady_clock;

std::vector<double> blob(std::size_t n, double spacing, std::uint64_t seed) {
  const auto m =
      static_cast<int>(std::ceil(std::cbrt(static_cast<double>(n))));
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> jitter(-0.02 * spacing, 0.02 * spacing);
  std::vector<double> R;
  R.reserve(3 * n);
  for (int iz = 0; iz < m && R.size() < 3 * n; ++iz) {
    for (int iy = 0; iy < m && R.size() < 3 * n; ++iy) {
      for (int ix = 0; ix < m && R.size() < 3 * n; ++ix) {
        R.push_back(static_cast<double>(ix) * spacing + jitter(rng));
        R.push_back(static_cast<double>(iy) * spacing + jitter(rng));
        R.push_back(static_cast<double>(iz) * spacing + jitter(rng));
      }
    }
  }
  return R;
}

std::vector<double> unitCube(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 10.0);
  std::vector<double> R(3 * n);
  for (double &x : R) {
    x = u(rng);
  }
  return R;
}

std::vector<double> loadCoords(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open coordinates\n");
    std::exit(2);
  }
  std::vector<double> R;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  std::string all((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
  const auto mark = all.find("Coordinates of Component");
  if (mark != std::string::npos) {
    // eOn .con: the atom count is the 8th number of the header, and the
    // coordinates follow the component line.
    const auto head = all.substr(0, mark);
    std::istringstream header(head);
    std::string word;
    std::vector<std::string> nums;
    while (header >> word) {
      nums.push_back(word);
    }
    if (nums.size() < 8) {
      std::fprintf(stderr, "coordinate header is short\n");
      std::exit(2);
    }
    const auto nat = static_cast<std::size_t>(std::atof(nums[7].c_str()));
    std::istringstream body(all.substr(mark));
    std::string skip;
    std::getline(body, skip);
    std::string line;
    while (R.size() / 3 < nat && std::getline(body, line)) {
      if (std::sscanf(line.c_str(), "%lf %lf %lf", &x, &y, &z) == 3) {
        R.push_back(x);
        R.push_back(y);
        R.push_back(z);
      }
    }
    return R;
  }
  std::istringstream in2(all);
  std::string line;
  while (std::getline(in2, line)) {
    // Three coordinates, or a .con row whose fourth field is an integer
    // flag. The Sutton-Chen header has five floating-point fields and is
    // left out.
    std::istringstream row(line);
    std::vector<std::string> tok;
    std::string word;
    while (row >> word) {
      tok.push_back(word);
    }
    if (tok.size() != 3 && tok.size() < 5) {
      continue;
    }
    if (tok.size() >= 5) {
      const bool integral =
          !tok[3].empty() &&
          tok[3].find_first_not_of("+-0123456789") == std::string::npos;
      if (!integral) {
        continue;
      }
    }
    if (std::sscanf(line.c_str(), "%lf %lf %lf", &x, &y, &z) == 3) {
      R.push_back(x);
      R.push_back(y);
      R.push_back(z);
    }
  }
  if (R.size() < 6 || R.size() % 3 != 0) {
    std::fprintf(stderr, "coordinate file is empty\n");
    std::exit(2);
  }
  return R;
}

std::array<double, 9> paddedCell(const std::vector<double> &R, double pad,
                                 std::vector<double> &shifted) {
  double minx = R[0], miny = R[1], minz = R[2];
  double maxx = R[0], maxy = R[1], maxz = R[2];
  for (std::size_t a = 1; a < R.size() / 3; ++a) {
    minx = std::min(minx, R[3 * a]);
    maxx = std::max(maxx, R[3 * a]);
    miny = std::min(miny, R[3 * a + 1]);
    maxy = std::max(maxy, R[3 * a + 1]);
    minz = std::min(minz, R[3 * a + 2]);
    maxz = std::max(maxz, R[3 * a + 2]);
  }
  shifted.resize(R.size());
  for (std::size_t a = 0; a < R.size() / 3; ++a) {
    shifted[3 * a] = R[3 * a] - minx + pad;
    shifted[3 * a + 1] = R[3 * a + 1] - miny + pad;
    shifted[3 * a + 2] = R[3 * a + 2] - minz + pad;
  }
  const double sx = (maxx - minx) + 2.0 * pad;
  const double sy = (maxy - miny) + 2.0 * pad;
  const double sz = (maxz - minz) + 2.0 * pad;
  return {sx, 0, 0, 0, sy, 0, 0, 0, sz};
}

template <typename Pot>
double once(const Pot &pot, const std::vector<double> &R,
            const std::vector<int> &types, const double *box,
            std::vector<double> &F) {
  rgpot::ForceInput fi{.nAtoms = R.size() / 3,
                       .pos = R.data(),
                       .atmnrs = types.data(),
                       .box = box};
  rgpot::ForceOut fo{.F = F.data(),
                     .energy = 0.0,
                     .variance = 0.0,
                     .stress = {},
                     .has_stress = 0};
  pot.forceImpl(fi, &fo);
  return fo.energy;
}

void timeCalls(const char *kind, const char *phase, std::size_t n, int threads,
               double cutoff, int warmup, int reps,
               const std::function<void(int)> &prepare,
               const std::function<double()> &call) {
  for (int i = 0; i < warmup; ++i) {
    prepare(-1);
    (void)call();
  }
  for (int i = 0; i < reps; ++i) {
    prepare(i);
    const auto t0 = clock_type::now();
    const double energy = call();
    const auto t1 = clock_type::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::printf(
        "kind=%s phase=%s n=%zu threads=%d cutoff=%.8g rep=%d seconds=%.9f "
        "energy=%.16e\n",
        kind, phase, n, threads, cutoff, i, seconds, energy);
    std::fflush(stdout);
  }
}

std::vector<double> mackay147() {
  const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
  const double v[12][3] = {
      {0, 1, phi},     {0, -1, phi},  {0, 1, -phi},    {0, -1, -phi},
      {1, phi, 0},     {-1, phi, 0},  {1, -phi, 0},    {-1, -phi, 0},
      {phi, 0, 1},     {phi, 0, -1},  {-phi, 0, 1},    {-phi, 0, -1}};
  auto dist2 = [&](int i, int j) {
    double s = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double d = v[i][k] - v[j][k];
      s += d * d;
    }
    return s;
  };
  std::vector<std::array<int, 2>> edges;
  std::vector<std::array<int, 3>> faces;
  for (int i = 0; i < 12; ++i) {
    for (int j = i + 1; j < 12; ++j) {
      if (std::abs(dist2(i, j) - 4.0) < 1e-8) {
        edges.push_back({i, j});
      }
    }
  }
  for (int i = 0; i < 12; ++i) {
    for (int j = i + 1; j < 12; ++j) {
      for (int k = j + 1; k < 12; ++k) {
        if (std::abs(dist2(i, j) - 4.0) < 1e-8 &&
            std::abs(dist2(j, k) - 4.0) < 1e-8 &&
            std::abs(dist2(k, i) - 4.0) < 1e-8) {
          faces.push_back({i, j, k});
        }
      }
    }
  }
  if (edges.size() != 30 || faces.size() != 20) {
    std::fprintf(stderr, "icosahedron has %zu edges and %zu faces\n",
                 edges.size(), faces.size());
    std::exit(2);
  }
  std::vector<double> R;
  auto add = [&](double x, double y, double z) {
    R.push_back(x);
    R.push_back(y);
    R.push_back(z);
  };
  add(0, 0, 0);
  constexpr int nshell = 3;
  for (int s = 1; s <= nshell; ++s) {
    const double scale = 0.5 * static_cast<double>(s);
    for (const auto &p : v) {
      add(scale * p[0], scale * p[1], scale * p[2]);
    }
    for (const auto &e : edges) {
      for (int t = 1; t < s; ++t) {
        const double a = static_cast<double>(s - t) / static_cast<double>(s);
        const double b = static_cast<double>(t) / static_cast<double>(s);
        add(scale * (a * v[e[0]][0] + b * v[e[1]][0]),
            scale * (a * v[e[0]][1] + b * v[e[1]][1]),
            scale * (a * v[e[0]][2] + b * v[e[1]][2]));
      }
    }
    for (const auto &f : faces) {
      for (int i = 1; i < s; ++i) {
        for (int j = 1; i + j < s; ++j) {
          const int k = s - i - j;
          if (k < 1) {
            continue;
          }
          const double ii = static_cast<double>(i) / static_cast<double>(s);
          const double jj = static_cast<double>(j) / static_cast<double>(s);
          const double kk = static_cast<double>(k) / static_cast<double>(s);
          add(scale * (ii * v[f[0]][0] + jj * v[f[1]][0] + kk * v[f[2]][0]),
              scale * (ii * v[f[0]][1] + jj * v[f[1]][1] + kk * v[f[2]][1]),
              scale * (ii * v[f[0]][2] + jj * v[f[1]][2] + kk * v[f[2]][2]));
        }
      }
    }
  }
  if (R.size() != 3 * 147) {
    std::fprintf(stderr, "Mackay cluster has %zu atoms\n", R.size() / 3);
    std::exit(2);
  }
  return R;
}

void printTruncation(const char *name, const std::vector<double> &R) {
  const std::size_t n = R.size() / 3;
  const std::vector<int> types(n, 78);
  std::vector<double> F(3 * n, 0.0);
  const double cuts[] = {0.0, 3.0, 40.0};
  double energy[3] = {};
  std::vector<double> forces[3] = {F, F, F};
  for (int c = 0; c < 3; ++c) {
    rgpot::SuttonChenConfig cfg;
    cfg.cutoff = cuts[c];
    const rgpot::SuttonChenPot pot(cfg);
    forces[c].assign(3 * n, 0.0);
    energy[c] = once(pot, R, types, nullptr, forces[c]);
  }
  std::printf("truncate name=%s n=%zu", name, n);
  for (int c = 0; c < 3; ++c) {
    const double de = energy[c] - energy[0];
    const double rel = std::abs(energy[0]) > 0.0 ? de / energy[0] : 0.0;
    double df = 0.0;
    for (std::size_t i = 0; i < forces[c].size(); ++i) {
      df = std::max(df, std::abs(forces[c][i] - forces[0][i]));
    }
    std::printf(
        " cutoff=%.1f energy=%.16e rel_to_full=%.16e max_abs_df=%.16e",
        cuts[c], energy[c], rel, df);
  }
  std::printf("\n");
}

const char *arg(int argc, char **argv, const char *key, const char *fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], key) == 0) {
      return argv[i + 1];
    }
  }
  return fallback;
}

} // namespace

int main(int argc, char **argv) {
  const std::string kind = arg(argc, argv, "--kind", "ljcluster");
  const auto n = static_cast<std::size_t>(std::strtoull(arg(argc, argv, "--n", "38"), nullptr, 10));
  const int threads = std::atoi(arg(argc, argv, "--threads", "1"));
  const double cutoff = std::atof(arg(argc, argv, "--cutoff", "2.5"));
  const int warmup = std::atoi(arg(argc, argv, "--warmup", "2"));
  const int reps = std::atoi(arg(argc, argv, "--reps", "1"));
  const char *phase = arg(argc, argv, "--phase", "warm");
  const char *file = arg(argc, argv, "--file", "");

  if (kind == "truncate") {
    printTruncation("pt38", loadCoords("CppCore/tests/data/sc108/pt38"));
    printTruncation("mackay147", mackay147());
    printTruncation("blob1000", blob(1000, 1.1, 1000));
    return 0;
  }

  std::vector<double> R;
  std::array<double, 9> box{1.0e6, 0, 0, 0, 1.0e6, 0, 0, 0, 1.0e6};
  const double *box_ptr = nullptr;
  double used_cutoff = cutoff;
  if (kind == "ljcluster" || kind == "ljpot" || kind == "sutton" ||
      kind == "admit") {
    R = blob(n, 1.1, 42);
  } else if (kind == "ljpot_asv") {
    R = unitCube(n, 1);
    box = {40, 0, 0, 0, 40, 0, 0, 0, 40};
    box_ptr = box.data();
    used_cutoff = 15.0;
  } else if (kind == "linkcell") {
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
        for (int k = 0; k < 5; ++k) {
          R.push_back((i + 0.5) * 5.0);
          R.push_back((j + 0.5) * 5.0);
          R.push_back((k + 0.5) * 5.0);
        }
      }
    }
    box = {25, 0, 0, 0, 25, 0, 0, 0, 25};
    box_ptr = box.data();
    used_cutoff = 6.0;
  } else if (kind == "ljfile" || kind == "morse") {
    R = loadCoords(file);
    box = {100, 0, 0, 0, 100, 0, 0, 0, 100};
    box_ptr = box.data();
    used_cutoff = cutoff > 0.0 ? cutoff : 15.0;
  } else {
    std::fprintf(stderr, "unknown kind %s\n", kind.c_str());
    return 2;
  }

  const std::size_t nat = R.size() / 3;
  const std::vector<int> types(nat, kind == "sutton" ? 78 : 1);
  std::vector<double> F(3 * nat, 0.0);
  std::vector<double> shifted;
  const int warm_n = std::strcmp(phase, "cold") == 0 ? 0 : warmup;
  auto prepare = [&](int rep) {
    if (rep < 0) {
      return;
    }
    if (std::strcmp(phase, "cold") == 0 && rep > 0) {
      for (std::size_t a = 0; a < R.size(); a += 3) {
        R[a] += 10000.0;
      }
    } else if (std::strcmp(phase, "rebuild") == 0) {
      for (std::size_t a = 0; a < R.size(); a += 3) {
        R[a] += 2.0;
      }
    }
  };

  if (kind == "admit") {
    rgpot::nlist::CachedPairList::Options opt;
    opt.cutoff = used_cutoff;
    opt.periodic = {{false, false, false}};
    const double wide[9] = {1.0e6, 0, 0, 0, 1.0e6, 0, 0, 0, 1.0e6};
    rgpot::nlist::CachedPairList list;
    list.rebuild(R.data(), nat, wide, opt);
    std::printf("kind=admit n=%zu cacheable=%d\n", nat,
                list.cacheable() ? 1 : 0);
    return 0;
  }

  if (kind == "ljpot") {
    box = paddedCell(R, used_cutoff, shifted);
    box_ptr = box.data();
    R.swap(shifted);
    const rgpot::LJPot pot(
        rgpot::LJConfig{.u0 = 1.0, .cutoff = used_cutoff, .psi = 1.0});
    timeCalls(kind.c_str(), phase, nat, threads, used_cutoff, warm_n, reps,
              prepare, [&] {
                std::fill(F.begin(), F.end(), 0.0);
                return once(pot, R, types, box_ptr, F);
              });
    return 0;
  }
  if (kind == "ljpot_asv" || kind == "linkcell") {
    const rgpot::LJPot pot(
        rgpot::LJConfig{.u0 = 1.0, .cutoff = used_cutoff, .psi = 1.0});
    timeCalls(kind.c_str(), phase, nat, threads, used_cutoff, warm_n, reps,
              prepare, [&] {
                std::fill(F.begin(), F.end(), 0.0);
                return once(pot, R, types, box_ptr, F);
              });
    return 0;
  }
  if (kind == "morse") {
    const rgpot::MorsePot pot;
    std::vector<int> z(nat, 78);
    std::array<double, 9> cell{100, 0, 0, 0, 100, 0, 0, 0, 100};
    std::ifstream hin(file);
    std::string skip, line;
    std::getline(hin, skip);
    std::getline(hin, line);
    double bx = 0, by = 0, bz = 0;
    if (std::sscanf(line.c_str(), "%lf %lf %lf", &bx, &by, &bz) == 3 &&
        bx > 0.0 && by > 0.0 && bz > 0.0) {
      cell = {bx, 0, 0, 0, by, 0, 0, 0, bz};
    }
    timeCalls(kind.c_str(), phase, nat, threads, pot.config().cutoff, warm_n,
              reps, prepare, [&] {
                std::fill(F.begin(), F.end(), 0.0);
                return once(pot, R, z, cell.data(), F);
              });
    return 0;
  }
  if (kind == "sutton") {
    rgpot::SuttonChenConfig cfg;
    cfg.cutoff = used_cutoff > 0.0 ? used_cutoff : 3.0;
    const rgpot::SuttonChenPot pot(cfg);
    timeCalls(kind.c_str(), phase, nat, threads, cfg.cutoff, warm_n, reps,
              prepare, [&] {
                std::fill(F.begin(), F.end(), 0.0);
                return once(pot, R, types, nullptr, F);
              });
    return 0;
  }
  const rgpot::LJClusterPot pot(
      rgpot::LJClusterConfig{.u0 = 1.0, .cutoff = used_cutoff, .psi = 1.0});
  if (kind == "ljfile") {
    box_ptr = box.data();
  }
  timeCalls(kind.c_str(), phase, nat, threads, used_cutoff, warm_n, reps,
            prepare, [&] {
              std::fill(F.begin(), F.end(), 0.0);
              return once(pot, R, types, box_ptr, F);
            });
  return 0;
}
