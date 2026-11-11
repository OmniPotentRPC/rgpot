// MIT License
// Copyright 2023--present rgpot developers
//
// Serial, strong and weak scaling for the classical potentials in stored
// and streaming neighbour modes. Medians of seven warm calls. Not a test.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace {

void setThreads(const char *value) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value == nullptr ? "" : value);
#else
  if (value == nullptr || value[0] == '\0') {
    unsetenv("OMP_NUM_THREADS");
  } else {
    setenv("OMP_NUM_THREADS", value, 1);
  }
#endif
}

long peakRssKb() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.compare(0, 6, "VmHWM:") == 0) {
      return std::strtol(line.c_str() + 6, nullptr, 10);
    }
  }
  return 0;
}

std::vector<double> fcc(int nx, int ny, int nz, double a, double &length_x,
                        double &length_y, double &length_z) {
  const double b[4][3] = {{0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0}};
  std::vector<double> pos;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &p : b) {
          pos.push_back((ix + p[0]) * a);
          pos.push_back((iy + p[1]) * a);
          pos.push_back((iz + p[2]) * a);
        }
      }
    }
  }
  length_x = nx * a;
  length_y = ny * a;
  length_z = nz * a;
  return pos;
}

std::vector<double> diamond(int n, double a, double &length) {
  const double b[8][3] = {{0, 0, 0},          {0.5, 0.5, 0},
                          {0.5, 0, 0.5},      {0, 0.5, 0.5},
                          {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                          {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  std::vector<double> pos;
  for (int ix = 0; ix < n; ++ix) {
    for (int iy = 0; iy < n; ++iy) {
      for (int iz = 0; iz < n; ++iz) {
        for (const auto &p : b) {
          pos.push_back((ix + p[0]) * a);
          pos.push_back((iy + p[1]) * a);
          pos.push_back((iz + p[2]) * a);
        }
      }
    }
  }
  length = n * a;
  return pos;
}

template <typename Pot>
double medianCall(Pot &pot, const std::vector<double> &pos,
                  const std::vector<int> &z, double lx, double ly, double lz,
                  int repeats, long &rss) {
  rgpot::types::AtomMatrix xyz(z.size(), 3);
  for (std::size_t i = 0; i < z.size(); ++i) {
    xyz(i, 0) = pos[3 * i];
    xyz(i, 1) = pos[3 * i + 1];
    xyz(i, 2) = pos[3 * i + 2];
  }
  const std::array<std::array<double, 3>, 3> box{{
      {lx, 0, 0},
      {0, ly, 0},
      {0, 0, lz},
  }};
  pot(xyz, z, box);
  pot(xyz, z, box);
  std::vector<double> samples;
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    pot(xyz, z, box);
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  rss = std::max(rss, peakRssKb());
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc > 1 && std::string(argv[1]) == "rss") {
    const std::string which = argc > 2 ? argv[2] : "EAM";
    long rss = 0;
    auto one = [&](auto &&pot, const std::vector<double> &pos,
                   const std::vector<int> &z, double lx, double ly, double lz) {
      pot.setNeighbourMode(rgpot::nlist::Mode::Stream);
      (void)medianCall(pot, pos, z, lx, ly, lz, 1, rss);
      std::printf("%s n %zu rss_kb %ld\n", which.c_str(), z.size(), rss);
    };
    if (which == "EAM" || which == "LJ" || which == "SuttonChen") {
      double lx = 0, ly = 0, lz = 0;
      const double lattice = which == "LJ" ? std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0)
                             : which == "SuttonChen" ? std::sqrt(2.0)
                                                     : 4.05;
      auto pos = fcc(64, 64, 64, lattice, lx, ly, lz);
      std::vector<int> z(pos.size() / 3, which == "LJ" ? 18 : which == "SuttonChen" ? 78 : 13);
      if (which == "EAM") {
        rgpot::EAMPot pot;
        one(pot, pos, z, lx, ly, lz);
      } else if (which == "LJ") {
        rgpot::LJConfig cfg;
        cfg.cutoff = 2.5;
        cfg.psi = 1.0;
        rgpot::LJPot pot(cfg);
        one(pot, pos, z, lx, ly, lz);
      } else {
        rgpot::SuttonChenConfig cfg;
        cfg.cutoff = 3.0;
        cfg.periodic = true;
        rgpot::SuttonChenPot pot(cfg);
        one(pot, pos, z, lx, ly, lz);
      }
    } else {
      double length = 0;
      auto pos = diamond(50, 5.431, length);
      std::vector<int> z(pos.size() / 3, 14);
      if (which == "SW") {
        rgpot::StillingerWeberSi pot;
        one(pot, pos, z, length, length, length);
      } else {
        rgpot::TersoffSiC pot;
        one(pot, pos, z, length, length, length);
      }
    }
    return 0;
  }
  constexpr int kRepeats = 7;
  long rss = 0;
  std::printf("repeats %d\n", kRepeats);
  std::fflush(stdout);
  const int cubic[] = {3, 5, 8, 12, 20, 30};
  const char *modes[] = {"stored", "stream"};
  const rgpot::nlist::Mode modev[] = {rgpot::nlist::Mode::Stored,
                                      rgpot::nlist::Mode::Stream};

  auto report = [&](const char *name, const char *mode, int atoms, double sec) {
    std::printf("%-12s %-7s %8d %10.6f rss_kb %ld\n", name, mode, atoms, sec,
                rss);
    std::fflush(stdout);
  };

  setThreads("");
  std::printf("\nserial\n");
  for (int m = 0; m < 2; ++m) {
    for (int nrep : cubic) {
      double lx = 0, ly = 0, lz = 0;
      auto pos = fcc(nrep, nrep, nrep, 4.05, lx, ly, lz);
      std::vector<int> z(pos.size() / 3, 13);
      rgpot::EAMPot pot;
      pot.setNeighbourMode(modev[m]);
      const double sec = medianCall(pot, pos, z, lx, ly, lz, kRepeats, rss);
      report("EAM", modes[m], static_cast<int>(z.size()), sec);
    }
    if (m == 1) {
      double lx = 0, ly = 0, lz = 0;
      auto pos = fcc(64, 64, 64, 4.05, lx, ly, lz);
      std::vector<int> z(pos.size() / 3, 13);
      rgpot::EAMPot pot;
      pot.setNeighbourMode(rgpot::nlist::Mode::Stream);
      const double sec = medianCall(pot, pos, z, lx, ly, lz, kRepeats, rss);
      report("EAM", "stream", static_cast<int>(z.size()), sec);
    }
  }

  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  rgpot::LJConfig ljcfg;
  ljcfg.cutoff = 2.5;
  ljcfg.psi = 1.0;
  for (int m = 0; m < 2; ++m) {
    for (int nrep : cubic) {
      double lx = 0, ly = 0, lz = 0;
      auto pos = fcc(nrep, nrep, nrep, a_lj, lx, ly, lz);
      std::vector<int> z(pos.size() / 3, 18);
      rgpot::LJPot pot(ljcfg);
      pot.setNeighbourMode(modev[m]);
      const double sec = medianCall(pot, pos, z, lx, ly, lz, kRepeats, rss);
      report("LJ", modes[m], static_cast<int>(z.size()), sec);
    }
    if (m == 1) {
      double lx = 0, ly = 0, lz = 0;
      auto pos = fcc(64, 64, 64, a_lj, lx, ly, lz);
      std::vector<int> z(pos.size() / 3, 18);
      rgpot::LJPot pot(ljcfg);
      pot.setNeighbourMode(rgpot::nlist::Mode::Stream);
      const double sec = medianCall(pot, pos, z, lx, ly, lz, kRepeats, rss);
      report("LJ", "stream", static_cast<int>(z.size()), sec);
    }
  }

  rgpot::SuttonChenConfig sc;
  sc.cutoff = 3.0;
  sc.periodic = true;
  const double a_sc = std::sqrt(2.0);
  for (int m = 0; m < 2; ++m) {
    for (int nrep : cubic) {
      double lx = 0, ly = 0, lz = 0;
      auto pos = fcc(nrep, nrep, nrep, a_sc, lx, ly, lz);
      std::vector<int> z(pos.size() / 3, 78);
      rgpot::SuttonChenPot pot(sc);
      pot.setNeighbourMode(modev[m]);
      const double sec = medianCall(pot, pos, z, lx, ly, lz, kRepeats, rss);
      report("SuttonChen", modes[m], static_cast<int>(z.size()), sec);
    }
    if (m == 1) {
      double lx = 0, ly = 0, lz = 0;
      auto pos = fcc(64, 64, 64, a_sc, lx, ly, lz);
      std::vector<int> z(pos.size() / 3, 78);
      rgpot::SuttonChenPot pot(sc);
      pot.setNeighbourMode(rgpot::nlist::Mode::Stream);
      const double sec = medianCall(pot, pos, z, lx, ly, lz, kRepeats, rss);
      report("SuttonChen", "stream", static_cast<int>(z.size()), sec);
    }
  }

  const int dia[] = {2, 3, 5, 8, 12};
  for (int m = 0; m < 2; ++m) {
    for (int nrep : dia) {
      double length = 0;
      auto pos = diamond(nrep, 5.431, length);
      std::vector<int> z(pos.size() / 3, 14);
      rgpot::StillingerWeberSi pot;
      pot.setNeighbourMode(modev[m]);
      const double sec =
          medianCall(pot, pos, z, length, length, length, kRepeats, rss);
      report("SW", modes[m], static_cast<int>(z.size()), sec);
    }
    for (int nrep : dia) {
      double length = 0;
      auto pos = diamond(nrep, 5.431, length);
      std::vector<int> z(pos.size() / 3, 14);
      rgpot::TersoffSiC pot;
      pot.setNeighbourMode(modev[m]);
      const double sec =
          medianCall(pot, pos, z, length, length, length, kRepeats, rss);
      report("Tersoff", modes[m], static_cast<int>(z.size()), sec);
    }
  }
  // One million-atom streaming diamond is 8*50^3 = 1,000,000.
  {
    double length = 0;
    auto pos = diamond(50, 5.431, length);
    std::vector<int> z(pos.size() / 3, 14);
    rgpot::StillingerWeberSi sw;
    sw.setNeighbourMode(rgpot::nlist::Mode::Stream);
    report("SW", "stream", static_cast<int>(z.size()),
           medianCall(sw, pos, z, length, length, length, kRepeats, rss));
    rgpot::TersoffSiC te;
    te.setNeighbourMode(rgpot::nlist::Mode::Stream);
    report("Tersoff", "stream", static_cast<int>(z.size()),
           medianCall(te, pos, z, length, length, length, kRepeats, rss));
  }

  std::printf("\nstrong n fcc 20^3 and diamond 12^3\n");
  std::fflush(stdout);
  const char *ts[] = {"1", "2", "4", "8"};
  for (int m = 0; m < 2; ++m) {
    double lx = 0, ly = 0, lz = 0;
      auto pos = fcc(20, 20, 20, 4.05, lx, ly, lz);
    std::vector<int> z(pos.size() / 3, 13);
    for (const char *th : ts) {
      setThreads(th);
      rgpot::EAMPot pot;
      pot.setNeighbourMode(modev[m]);
      const double sec = medianCall(pot, pos, z, lx, ly, lz, kRepeats, rss);
      std::printf("EAM %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m], th,
                  z.size(), sec, rss);
    }
    auto lpos = fcc(20, 20, 20, a_lj, lx, ly, lz);
    std::vector<int> lzatoms(lpos.size() / 3, 18);
    for (const char *th : ts) {
      setThreads(th);
      rgpot::LJPot pot(ljcfg);
      pot.setNeighbourMode(modev[m]);
      const double sec = medianCall(pot, lpos, lzatoms, lx, ly, lz, kRepeats, rss);
      std::printf("LJ %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m], th,
                  lzatoms.size(), sec, rss);
    }
    auto spos = fcc(20, 20, 20, a_sc, lx, ly, lz);
    std::vector<int> satoms(spos.size() / 3, 78);
    for (const char *th : ts) {
      setThreads(th);
      rgpot::SuttonChenPot pot(sc);
      pot.setNeighbourMode(modev[m]);
      const double sec = medianCall(pot, spos, satoms, lx, ly, lz, kRepeats, rss);
      std::printf("SuttonChen %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m], th,
                  satoms.size(), sec, rss);
    }
    double length = 0;
    auto dpos = diamond(12, 5.431, length);
    std::vector<int> datoms(dpos.size() / 3, 14);
    for (const char *th : ts) {
      setThreads(th);
      rgpot::StillingerWeberSi pot;
      pot.setNeighbourMode(modev[m]);
      const double sec =
          medianCall(pot, dpos, datoms, length, length, length, kRepeats, rss);
      std::printf("SW %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m], th,
                  datoms.size(), sec, rss);
    }
    for (const char *th : ts) {
      setThreads(th);
      rgpot::TersoffSiC pot;
      pot.setNeighbourMode(modev[m]);
      const double sec =
          medianCall(pot, dpos, datoms, length, length, length, kRepeats, rss);
      std::printf("Tersoff %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m], th,
                  datoms.size(), sec, rss);
    }
  }
  std::printf("\nstrong EAM fcc 30^3\n");
  std::fflush(stdout);
  for (int m = 0; m < 2; ++m) {
    double lx = 0, ly = 0, lz = 0;
    auto pos = fcc(30, 30, 30, 4.05, lx, ly, lz);
    std::vector<int> z(pos.size() / 3, 13);
    for (const char *th : ts) {
      setThreads(th);
      rgpot::EAMPot pot;
      pot.setNeighbourMode(modev[m]);
      const double sec = medianCall(pot, pos, z, lx, ly, lz, kRepeats, rss);
      std::printf("EAM %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m], th,
                  z.size(), sec, rss);
      std::fflush(stdout);
    }
  }
  std::printf("\nweak\n");
  std::fflush(stdout);
  const int weak_shape[][3] = {{6, 6, 6}, {6, 6, 12}, {6, 12, 12}, {12, 12, 12}};
  const int weak_dia[] = {4, 5, 6, 8};
  for (int m = 0; m < 2; ++m) {
    for (int i = 0; i < 4; ++i) {
      setThreads(ts[i]);
      double lx = 0, ly = 0, lz = 0;
      auto pos = fcc(weak_shape[i][0], weak_shape[i][1], weak_shape[i][2], 4.05,
                     lx, ly, lz);
      std::vector<int> z(pos.size() / 3, 13);
      rgpot::EAMPot pot;
      pot.setNeighbourMode(modev[m]);
      const double sec = medianCall(pot, pos, z, lx, ly, lz, kRepeats, rss);
      std::printf("EAM %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m], ts[i],
                  z.size(), sec, rss);
      std::fflush(stdout);
      auto lpos = fcc(weak_shape[i][0], weak_shape[i][1], weak_shape[i][2],
                      a_lj, lx, ly, lz);
      std::vector<int> lzatoms(lpos.size() / 3, 18);
      rgpot::LJPot lj(ljcfg);
      lj.setNeighbourMode(modev[m]);
      const double lsec =
          medianCall(lj, lpos, lzatoms, lx, ly, lz, kRepeats, rss);
      std::printf("LJ %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m], ts[i],
                  lzatoms.size(), lsec, rss);
      std::fflush(stdout);
      auto spos = fcc(weak_shape[i][0], weak_shape[i][1], weak_shape[i][2],
                      a_sc, lx, ly, lz);
      std::vector<int> satoms(spos.size() / 3, 78);
      rgpot::SuttonChenPot sutton(sc);
      sutton.setNeighbourMode(modev[m]);
      const double ssec =
          medianCall(sutton, spos, satoms, lx, ly, lz, kRepeats, rss);
      std::printf("SuttonChen %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m],
                  ts[i], satoms.size(), ssec, rss);
      std::fflush(stdout);
      double length = 0;
      auto dpos = diamond(weak_dia[i], 5.431, length);
      std::vector<int> datoms(dpos.size() / 3, 14);
      rgpot::StillingerWeberSi sw;
      sw.setNeighbourMode(modev[m]);
      const double wsec = medianCall(sw, dpos, datoms, length, length, length,
                                     kRepeats, rss);
      std::printf("SW %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m], ts[i],
                  datoms.size(), wsec, rss);
      std::fflush(stdout);
      rgpot::TersoffSiC te;
      te.setNeighbourMode(modev[m]);
      const double tsec = medianCall(te, dpos, datoms, length, length, length,
                                     kRepeats, rss);
      std::printf("Tersoff %-7s T=%s n=%zu %.6f rss_kb %ld\n", modes[m], ts[i],
                  datoms.size(), tsec, rss);
      std::fflush(stdout);
    }
  }
  setThreads("");
  std::printf("peak_rss_kb %ld\n", rss);
  return 0;
}
