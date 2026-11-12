// MIT License
// Copyright 2023--present rgpot developers
//
// List against stream for the classical potentials. One child process per
// row so the peak resident set is that row alone. Times are medians of
// warm calls. The child sets OMP_NUM_THREADS.

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

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"

namespace {

void setThreads(const char *value) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value == nullptr ? "" : value);
#else
  setenv("OMP_NUM_THREADS", value, 1);
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

std::vector<double> fcc(int nx, int ny, int nz, double a, double &lx, double &ly,
                        double &lz) {
  const double b[4][3] = {{0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0}};
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
              static_cast<std::size_t>(nz) * 12);
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

std::vector<double> diamond(int nx, int ny, int nz, double a, double &lx, double &ly,
                            double &lz) {
  const double b[8][3] = {{0, 0, 0},          {0.5, 0.5, 0},     {0.5, 0, 0.5},
                          {0, 0.5, 0.5},      {0.25, 0.25, 0.25},{0.75, 0.75, 0.25},
                          {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
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

struct Job {
  const char *name;
  const char *mode;
  int nx;
  int ny;
  int nz;
  int kind; // 0 fcc, 1 diamond
  int repeats;
  int threads;
};

template <typename Pot>
void timePot(Pot &pot, const std::vector<double> &pos, const std::vector<int> &z, double lx,
             double ly, double lz, int repeats, double &median, long &rss) {
  rgpot::types::AtomMatrix xyz(z.size(), 3);
  for (std::size_t i = 0; i < z.size(); ++i) {
    xyz(i, 0) = pos[3 * i];
    xyz(i, 1) = pos[3 * i + 1];
    xyz(i, 2) = pos[3 * i + 2];
  }
  const std::array<std::array<double, 3>, 3> box{{{lx, 0, 0}, {0, ly, 0}, {0, 0, lz}}};
  pot(xyz, z, box);
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    pot(xyz, z, box);
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  median = samples[samples.size() / 2];
  rss = peakRssKb();
}

int runJob(const Job &job) {
  setThreads(std::to_string(job.threads).c_str());
  const std::string which = job.mode;
  const auto mode = which == "stream"  ? rgpot::nlist::Mode::Stream
                    : which == "auto" ? rgpot::nlist::Mode::Auto
                                      : rgpot::nlist::Mode::Stored;
  double lx = 0;
  double ly = 0;
  double lz = 0;
  double median = 0;
  long rss = 0;
  double cutoff = 0;
  std::string name = job.name;
  if (name == "LJ") {
    const double a = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
    auto pos = fcc(job.nx, job.ny, job.nz, a, lx, ly, lz);
    std::vector<int> z(pos.size() / 3, 18);
    rgpot::LJConfig cfg;
    cfg.cutoff = 2.5;
    cfg.psi = 1.0;
    cutoff = cfg.cutoff;
    rgpot::LJPot pot(cfg);
    pot.setNeighbourMode(mode);
    timePot(pot, pos, z, lx, ly, lz, job.repeats, median, rss);
    std::printf("%s,%s,%zu,%d,%d,%.6f,%ld,%.3f\n", name.c_str(), job.mode, z.size(),
                job.threads, job.repeats, median, rss, cutoff);
    return 0;
  }
  if (name == "Morse") {
    const double a = 4.05;
    auto pos = fcc(job.nx, job.ny, job.nz, a, lx, ly, lz);
    std::vector<int> z(pos.size() / 3, 78);
    rgpot::MorseConfig cfg;
    cfg.cutoff = 6.0;
    cutoff = cfg.cutoff;
    rgpot::MorsePot pot(cfg);
    pot.setNeighbourMode(mode);
    timePot(pot, pos, z, lx, ly, lz, job.repeats, median, rss);
    std::printf("%s,%s,%zu,%d,%d,%.6f,%ld,%.3f\n", name.c_str(), job.mode, z.size(),
                job.threads, job.repeats, median, rss, cutoff);
    return 0;
  }
  if (name == "EAM") {
    auto pos = fcc(job.nx, job.ny, job.nz, 4.05, lx, ly, lz);
    std::vector<int> z(pos.size() / 3, 13);
    cutoff = 5.555;
    rgpot::EAMPot pot;
    pot.setNeighbourMode(mode);
    timePot(pot, pos, z, lx, ly, lz, job.repeats, median, rss);
    std::printf("%s,%s,%zu,%d,%d,%.6f,%ld,%.3f\n", name.c_str(), job.mode, z.size(),
                job.threads, job.repeats, median, rss, cutoff);
    return 0;
  }
  if (name == "SuttonChen") {
    const double a = std::sqrt(2.0);
    auto pos = fcc(job.nx, job.ny, job.nz, a, lx, ly, lz);
    std::vector<int> z(pos.size() / 3, 78);
    rgpot::SuttonChenConfig cfg;
    cfg.cutoff = 3.0;
    cfg.periodic = true;
    cutoff = cfg.cutoff;
    rgpot::SuttonChenPot pot(cfg);
    pot.setNeighbourMode(mode);
    timePot(pot, pos, z, lx, ly, lz, job.repeats, median, rss);
    std::printf("%s,%s,%zu,%d,%d,%.6f,%ld,%.3f\n", name.c_str(), job.mode, z.size(),
                job.threads, job.repeats, median, rss, cutoff);
    return 0;
  }
  if (name == "ZBL") {
    auto pos = fcc(job.nx, job.ny, job.nz, 4.05, lx, ly, lz);
    std::vector<int> z(pos.size() / 3, 79);
    cutoff = 2.5;
    rgpot::ZBLPot pot;
    pot.setNeighbourMode(mode);
    timePot(pot, pos, z, lx, ly, lz, job.repeats, median, rss);
    std::printf("%s,%s,%zu,%d,%d,%.6f,%ld,%.3f\n", name.c_str(), job.mode, z.size(),
                job.threads, job.repeats, median, rss, cutoff);
    return 0;
  }
  if (name == "SW" || name == "Tersoff") {
    auto pos = diamond(job.nx, job.ny, job.nz, 5.431, lx, ly, lz);
    std::vector<int> z(pos.size() / 3, 14);
    cutoff = name == "SW" ? 1.80 * 2.0951 : 3.0;
    if (name == "SW") {
      rgpot::StillingerWeberSi pot;
      pot.setNeighbourMode(mode);
      timePot(pot, pos, z, lx, ly, lz, job.repeats, median, rss);
    } else {
      rgpot::TersoffSiC pot;
      pot.setNeighbourMode(mode);
      timePot(pot, pos, z, lx, ly, lz, job.repeats, median, rss);
    }
    std::printf("%s,%s,%zu,%d,%d,%.6f,%ld,%.3f\n", name.c_str(), job.mode, z.size(),
                job.threads, job.repeats, median, rss, cutoff);
    return 0;
  }
  std::fprintf(stderr, "unknown potential %s\n", job.name);
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc >= 8 && std::string(argv[1]) == "--job") {
    Job job;
    job.name = argv[2];
    job.mode = argv[3];
    job.nx = std::atoi(argv[4]);
    job.ny = std::atoi(argv[5]);
    job.nz = std::atoi(argv[6]);
    job.kind = 0;
    job.repeats = std::atoi(argv[7]);
    job.threads = argc >= 9 ? std::atoi(argv[8]) : 8;
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
  const char *pots[] = {"LJ", "Morse", "EAM", "SuttonChen", "ZBL", "SW", "Tersoff"};
  const char *modes[] = {"stored", "stream", "auto"};
  struct Shape {
    int nx, ny, nz;
  };
  const Shape fcc_shapes[] = {{25, 10, 10}, {30, 30, 30}, {64, 64, 64}};
  const Shape dia_shapes[] = {{25, 5, 10}, {30, 15, 30}, {50, 50, 50}};
  constexpr int kRepeats = 5;
  constexpr int kThreads = 8;
  for (const char *pot : pots) {
    const bool diamond_pot = std::string(pot) == "SW" || std::string(pot) == "Tersoff";
    const Shape *shapes = diamond_pot ? dia_shapes : fcc_shapes;
    for (const char *mode : modes) {
      for (int s = 0; s < 3; ++s) {
        std::ostringstream cmd;
        cmd << '"' << self << '"' << " --job " << pot << ' ' << mode << ' ' << shapes[s].nx
            << ' ' << shapes[s].ny << ' ' << shapes[s].nz << ' ' << kRepeats << ' ' << kThreads;
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
