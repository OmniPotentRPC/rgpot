// MIT License
// Copyright 2023--present rgpot developers
//
// List against stream for the potentials that joined the shared layer
// here. One child process per row. Times are medians of warm calls.
// Peak resident set is that child alone.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/ReducedPair/ReducedPair.hpp"

#if defined(RGPOT_HAS_FORTRAN_POTS)
#include "rgpot/fortran/FortranPots.hpp"
#endif

namespace {

void setThreads(const char *value) { setenv("OMP_NUM_THREADS", value, 1); }

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

std::vector<double> fcc(int nx, int ny, int nz, double a, double &lx, double &ly, double &lz) {
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
  lx = nx * a;
  ly = ny * a;
  lz = nz * a;
  return pos;
}

std::vector<double> diamond(int n, double a, double &lx, double &ly, double &lz) {
  const double b[8][3] = {{0, 0, 0},         {0.5, 0.5, 0},        {0.5, 0, 0.5},
                          {0, 0.5, 0.5},     {0.25, 0.25, 0.25},   {0.75, 0.75, 0.25},
                          {0.75, 0.25, 0.75},{0.25, 0.75, 0.75}};
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
  lx = n * a;
  ly = n * a;
  lz = n * a;
  return pos;
}

struct Job {
  const char *name;
  const char *mode;
  int n;
  int repeats;
};

template <typename Fn>
void timeCall(Fn &&fn, int repeats, double &median, long &rss) {
  fn();
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  median = samples[samples.size() / 2];
  rss = peakRssKb();
}

int runJob(const Job &job) {
  setThreads("1");
  const int mode = std::string(job.mode) == "stream" ? 2 : 1;
  double median = 0;
  long rss = 0;
  std::size_t atoms = 0;
  double cutoff = 0;
  if (std::string(job.name) == "ReducedLJ" || std::string(job.name) == "ReducedMorse") {
    double lx = 0;
    double ly = 0;
    double lz = 0;
    const double a = std::string(job.name) == "ReducedLJ"
                         ? std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0)
                         : 1.2;
    auto pos = fcc(job.n, job.n, job.n, a, lx, ly, lz);
    atoms = pos.size() / 3;
    cutoff = std::string(job.name) == "ReducedLJ" ? 2.5 : 2.5;
    std::vector<double> g(pos.size());
    double energy = 0;
    const auto natoms = static_cast<int32_t>(atoms);
    rgpot_pair_set_neighbour_mode(mode);
    auto call = [&] {
      if (std::string(job.name) == "ReducedLJ") {
        rgpot_lj_gradient(1.0, 1.0, cutoff, natoms, pos.data(), &energy, g.data());
      } else {
        rgpot_morse_gradient(1.0, 6.0, 1.0, cutoff, natoms, pos.data(), &energy, g.data());
      }
    };
    timeCall(call, job.repeats, median, rss);
  } else if (std::string(job.name) == "EMT") {
    double lx = 0;
    double ly = 0;
    double lz = 0;
    auto pos = fcc(job.n, job.n, job.n, 3.615, lx, ly, lz);
    atoms = pos.size() / 3;
    cutoff = 0;
    rgpot::types::AtomMatrix xyz(atoms, 3);
    std::vector<int> z(atoms, 29);
    for (std::size_t i = 0; i < atoms; ++i) {
      xyz(i, 0) = pos[3 * i];
      xyz(i, 1) = pos[3 * i + 1];
      xyz(i, 2) = pos[3 * i + 2];
    }
    const std::array<std::array<double, 3>, 3> box{{{lx, 0, 0}, {0, ly, 0}, {0, 0, lz}}};
    rgpot::EMTPot pot;
    pot.setNeighbourMode(mode == 2 ? rgpot::nlist::Mode::Stream : rgpot::nlist::Mode::Stored);
    auto call = [&] { pot(xyz, z, box); };
    timeCall(call, job.repeats, median, rss);
  }
#if defined(RGPOT_HAS_FORTRAN_POTS)
  else if (std::string(job.name) == "FortranSW" || std::string(job.name) == "FortranEAMAl") {
    double lx = 0;
    double ly = 0;
    double lz = 0;
    std::vector<double> pos;
    std::vector<int> z;
    if (std::string(job.name) == "FortranSW") {
      pos = diamond(job.n, 5.431, lx, ly, lz);
      z.assign(pos.size() / 3, 14);
      cutoff = 3.77;
    } else {
      pos = fcc(job.n, job.n, job.n, 4.05, lx, ly, lz);
      z.assign(pos.size() / 3, 13);
      cutoff = 5.558;
    }
    atoms = pos.size() / 3;
    rgpot::types::AtomMatrix xyz(atoms, 3);
    for (std::size_t i = 0; i < atoms; ++i) {
      xyz(i, 0) = pos[3 * i];
      xyz(i, 1) = pos[3 * i + 1];
      xyz(i, 2) = pos[3 * i + 2];
    }
    const std::array<std::array<double, 3>, 3> box{{{lx, 0, 0}, {0, ly, 0}, {0, 0, lz}}};
    const auto which = mode == 2 ? rgpot::nlist::Mode::Stream : rgpot::nlist::Mode::Stored;
    if (std::string(job.name) == "FortranSW") {
      rgpot::fortranpots::SWPot pot;
      pot.setNeighbourMode(which);
      auto call = [&] { pot(xyz, z, box); };
      timeCall(call, job.repeats, median, rss);
    } else {
      rgpot::fortranpots::EAMAlPot pot;
      pot.setNeighbourMode(which);
      auto call = [&] { pot(xyz, z, box); };
      timeCall(call, job.repeats, median, rss);
    }
  }
#endif
  else {
    std::fprintf(stderr, "unknown potential %s\n", job.name);
    return 1;
  }
  std::printf("%s,%s,%zu,1,%d,%.6f,%ld,%.3f\n", job.name, job.mode, atoms, job.repeats, median,
              rss, cutoff);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc >= 6 && std::string(argv[1]) == "--job") {
    Job job;
    job.name = argv[2];
    job.mode = argv[3];
    job.n = std::atoi(argv[4]);
    job.repeats = std::atoi(argv[5]);
    return runJob(job);
  }
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.csv\n", argv[0]);
    return 1;
  }
  char self[4096];
  const ssize_t nread = ::readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (nread < 0) {
    std::perror("readlink");
    return 1;
  }
  self[nread] = '\0';
  std::ofstream out(argv[1]);
  if (!out) {
    std::perror(argv[1]);
    return 1;
  }
  out << "potential,mode,atoms,threads,repeats,median_s,peak_rss_kb,cutoff_A\n";
  const char *pots[] = {"ReducedLJ", "ReducedMorse", "EMT", "FortranSW", "FortranEAMAl"};
  const char *modes[] = {"stored", "stream"};
  // fcc n^3*4 and diamond n^3*8 land near 10k, 100k and 1M.
  const int fcc_n[] = {14, 30, 64};
  const int dia_n[] = {11, 24, 50};
  constexpr int kRepeats = 3;
  for (const char *pot : pots) {
    const bool diamond_pot = std::string(pot) == "FortranSW";
    const int *shape = diamond_pot ? dia_n : fcc_n;
    for (const char *mode : modes) {
      for (int s = 0; s < 3; ++s) {
        std::ostringstream cmd;
        cmd << '"' << self << '"' << " --job " << pot << ' ' << mode << ' ' << shape[s] << ' '
            << kRepeats;
        std::fprintf(stderr, "running %s\n", cmd.str().c_str());
        FILE *pipe = popen(cmd.str().c_str(), "r");
        if (pipe == nullptr) {
          std::perror("popen");
          return 1;
        }
        char line[512];
        if (std::fgets(line, sizeof(line), pipe) == nullptr) {
          const int rc = pclose(pipe);
          std::fprintf(stderr, "job produced no row, status %d\n", rc);
          return 1;
        }
        const int raw = pclose(pipe);
        const int rc = WIFEXITED(raw) ? WEXITSTATUS(raw) : raw;
        if (rc != 0) {
          std::fprintf(stderr, "job failed, status %d\n", rc);
          return 1;
        }
        out << line;
        out.flush();
        std::fputs(line, stderr);
      }
    }
  }
  return 0;
}
