// MIT License
// Copyright 2023--present rgpot developers

// Report-only hashes of classical energies, forces and stress.
// The kernels are not modified. Thread count comes from OMP_NUM_THREADS,
// which vesin uses to size its pool. MPI rank, when the process was
// launched under mpirun, is read from the environment so each rank writes
// its own dump. A difference across those dumps is a result, not a failure.

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/engine/Sha256.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace {

const char *rank_label() {
  if (const char *r = std::getenv("OMPI_COMM_WORLD_RANK")) {
    return r;
  }
  if (const char *r = std::getenv("PMI_RANK")) {
    return r;
  }
  return "0";
}

void box_of(std::size_t side, double spacing, double box[9]) {
  std::memset(box, 0, 9 * sizeof(double));
  const double L = static_cast<double>(side) * spacing;
  box[0] = L;
  box[4] = L;
  box[8] = L;
}

void fill_lattice(std::size_t n, std::vector<double> &pos, double box[9]) {
  const double spacing = 3.2;
  std::size_t side = 1;
  while (side * side * side < n) {
    ++side;
  }
  box_of(side, spacing, box);
  pos.assign(3 * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t x = i % side;
    const std::size_t y = (i / side) % side;
    const std::size_t z = i / (side * side);
    pos[3 * i] = (static_cast<double>(x) + 0.5) * spacing;
    pos[3 * i + 1] = (static_cast<double>(y) + 0.5) * spacing;
    pos[3 * i + 2] = (static_cast<double>(z) + 0.5) * spacing;
  }
  // Signed zeroes on coordinates that do not land on another atom.
  pos[0] = -0.0;
  if (n > 1) {
    pos[5] = 0.0;
  }
}

void fill_edge(std::vector<double> &pos, double box[9]) {
  box_of(1, 10.0, box);
  pos.assign(24, 0.0);
  const double d = std::numeric_limits<double>::denorm_min();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // -0, +0, a subnormal, a half-box tie against atom 0, one NaN.
  const double raw[8][3] = {
      {-0.0, 1.0, 1.0}, {0.0, 2.0, 1.0}, {d, 1.0, 2.0},   {5.0, 1.0, 3.0},
      {nan, 2.0, 2.0},  {1.0, 3.0, 1.0}, {2.0, 3.0, 2.0}, {3.0, 3.0, 3.0},
  };
  for (int i = 0; i < 8; ++i) {
    pos[3 * i] = raw[i][0];
    pos[3 * i + 1] = raw[i][1];
    pos[3 * i + 2] = raw[i][2];
  }
}

struct Dump {
  double energy;
  int has_stress;
  double stress[9];
  std::vector<double> force;
  double wall_s;
  bool threw;
  std::string what;
};

template <typename Pot>
Dump evaluate(Pot &pot, const std::vector<double> &pos,
              const std::vector<int> &z, const double box[9]) {
  Dump d;
  d.threw = false;
  d.has_stress = 0;
  d.energy = 0;
  std::memset(d.stress, 0, sizeof(d.stress));
  d.force.assign(pos.size(), 0.0);
  rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  rgpot::ForceOut out{};
  out.F = d.force.data();
  const auto t0 = std::chrono::steady_clock::now();
  try {
    pot.forceImpl(in, &out);
    d.energy = out.energy;
    d.has_stress = out.has_stress;
    std::memcpy(d.stress, out.stress, sizeof(d.stress));
  } catch (const std::exception &ex) {
    d.threw = true;
    d.what = ex.what();
  } catch (...) {
    d.threw = true;
    d.what = "nonstandard exception";
  }
  const auto t1 = std::chrono::steady_clock::now();
  d.wall_s = std::chrono::duration<double>(t1 - t0).count();
  return d;
}

std::string hex_of(const void *data, std::size_t n) {
  rgpot::engine::Sha256 h;
  h.update(data, n);
  return h.hex_digest();
}

void write_dump(const std::string &dir, const char *name, const Dump &d) {
  const std::string path =
      dir + "/" + name + ".r" + rank_label() + (d.threw ? ".err" : ".bin");
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::perror(path.c_str());
    std::exit(1);
  }
  if (d.threw) {
    std::fwrite(d.what.data(), 1, d.what.size(), f);
    std::fclose(f);
    std::printf("threw %s %s\n", name, d.what.c_str());
    return;
  }
  const std::uint32_t magic = 0x52504749u;
  const std::uint32_t n = static_cast<std::uint32_t>(d.force.size() / 3);
  const std::uint32_t has = static_cast<std::uint32_t>(d.has_stress);
  const std::uint32_t pad = 0;
  std::fwrite(&magic, 4, 1, f);
  std::fwrite(&n, 4, 1, f);
  std::fwrite(&has, 4, 1, f);
  std::fwrite(&pad, 4, 1, f);
  std::fwrite(&d.energy, sizeof(double), 1, f);
  std::fwrite(d.stress, sizeof(double), 9, f);
  std::fwrite(d.force.data(), sizeof(double), d.force.size(), f);
  std::fclose(f);
  const std::string eh = hex_of(&d.energy, sizeof(double));
  const std::string fh =
      hex_of(d.force.data(), d.force.size() * sizeof(double));
  const std::string sh = hex_of(d.stress, sizeof(d.stress));
  std::printf(
      "hash %s n %u stress %d energy %s force %s stress_sha %s wall_s %.6f\n",
      name, n, d.has_stress, eh.c_str(), fh.c_str(), sh.c_str(), d.wall_s);
  std::fflush(stdout);
}

struct Sys {
  std::vector<double> pos;
  std::vector<int> z;
  double box[9];
};

Sys make_sys(std::size_t n, bool edge) {
  Sys s;
  if (edge) {
    fill_edge(s.pos, s.box);
  } else {
    fill_lattice(n, s.pos, s.box);
  }
  s.z.assign(s.pos.size() / 3, 13);
  return s;
}

} // namespace

int main(int argc, char **argv) {
  std::size_t n = 4000;
  bool edge = false;
  bool bench = false;
  bool want_lj = true;
  bool want_morse = true;
  bool want_eam = true;
  const char *dir = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--n" && i + 1 < argc) {
      n = static_cast<std::size_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (a == "--dump-dir" && i + 1 < argc) {
      dir = argv[++i];
    } else if (a == "--pots" && i + 1 < argc) {
      const std::string pots = argv[++i];
      want_lj = pots.find("lj") != std::string::npos;
      want_morse = pots.find("morse") != std::string::npos;
      want_eam = pots.find("eam") != std::string::npos;
    } else if (a == "--edge") {
      edge = true;
    } else if (a == "--bench") {
      bench = true;
    } else {
      std::fprintf(stderr, "usage: hash_classical --n N --dump-dir DIR [--pots "
                           "lj,morse,eam] [--edge] [--bench]\n");
      return 2;
    }
  }
  const Sys sys = make_sys(edge ? 8 : n, edge);
  rgpot::LJPot lj{rgpot::LJConfig{.u0 = 1.0, .cutoff = 4.5, .psi = 1.0}};
  rgpot::MorsePot morse{rgpot::MorseConfig{.cutoff = 6.0}};
  rgpot::EAMPot eam;

  if (want_lj) {
    const Dump lj_d = evaluate(lj, sys.pos, sys.z, sys.box);
    if (bench) {
      std::printf("bench lj %.9f\n", lj_d.wall_s);
      std::fflush(stdout);
    }
    if (dir != nullptr) {
      write_dump(dir, "lj", lj_d);
    }
  }
  if (want_morse) {
    const Dump mo_d = evaluate(morse, sys.pos, sys.z, sys.box);
    if (bench) {
      std::printf("bench morse %.9f\n", mo_d.wall_s);
      std::fflush(stdout);
    }
    if (dir != nullptr) {
      write_dump(dir, "morse", mo_d);
    }
  }
  if (want_eam) {
    const Dump ea_d = evaluate(eam, sys.pos, sys.z, sys.box);
    if (bench) {
      std::printf("bench eam %.9f\n", ea_d.wall_s);
      std::fflush(stdout);
    }
    if (dir != nullptr) {
      write_dump(dir, "eam", ea_d);
    }
  }
  if (dir == nullptr && !bench) {
    std::fprintf(stderr, "pass --dump-dir or --bench\n");
    return 2;
  }
  return 0;
}
