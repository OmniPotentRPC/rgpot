// MIT License
// Copyright 2023--present rgpot developers
//
// One force evaluation per rgpot_asv_force call. asv times that call.
// rgpot_asv_prepare loads the geometry and warms the cache, and asv does
// not time it.
//
// LJ38, LJ75 and LJ150 are the optbench cluster coordinates. The Pt
// island is the optbench reactant. The periodic lattice is built here so
// the linked-cell search has a cell several cutoffs wide.

#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"

extern "C" int rgpot_water_h_force(int natoms, const double *positions,
                                   const int *atomic_numbers, const double *cell,
                                   double *forces, double *energy);

namespace {

enum Kind {
  kLj38 = 0,
  kLj75 = 1,
  kLj150 = 2,
  kPtWarm = 3,
  kPtCold = 4,
  kWater = 5,
  kLjLinkcell = 6,
};

struct Cloud {
  std::vector<double> base;
  std::vector<double> pos;
  std::vector<double> forces;
  std::vector<int> types;
  double box[9]{};
  bool loaded{false};
};

std::string g_error;
Cloud g_lj38;
Cloud g_lj75;
Cloud g_lj150;
Cloud g_pt;
Cloud g_water;
Cloud g_lattice;
int g_cold_step = 0;

#ifndef RGPOT_ASV_DATA
#define RGPOT_ASV_DATA "benchmarks/data"
#endif

int fail(const std::string &msg) {
  g_error = msg;
  return 1;
}

bool load_con(const std::string &path, Cloud &cloud) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::string line;
  std::getline(in, line);
  double lx = 0, ly = 0, lz = 0, a = 0, b = 0, c = 0;
  in >> lx >> ly >> lz >> a >> b >> c;
  int ntypes = 0, nat = 0;
  in >> ntypes >> nat;
  if (nat < 1) {
    return false;
  }
  std::getline(in, line);
  while (std::getline(in, line)) {
    if (line.find("Coordinates of Component") != std::string::npos) {
      break;
    }
  }
  cloud.base.assign(static_cast<std::size_t>(nat) * 3, 0.0);
  for (int i = 0; i < nat; ++i) {
    int fixed = 0, idx = 0;
    if (!(in >> cloud.base[3 * i] >> cloud.base[3 * i + 1] >> cloud.base[3 * i + 2] >> fixed >>
          idx)) {
      return false;
    }
  }
  cloud.box[0] = lx;
  cloud.box[4] = ly;
  cloud.box[8] = lz;
  cloud.pos = cloud.base;
  cloud.forces.assign(cloud.base.size(), 0.0);
  cloud.types.assign(cloud.base.size() / 3, 1);
  cloud.loaded = true;
  return true;
}

bool load_xyz(const std::string &path, Cloud &cloud) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  cloud.base.clear();
  double x = 0, y = 0, z = 0;
  while (in >> x >> y >> z) {
    cloud.base.push_back(x);
    cloud.base.push_back(y);
    cloud.base.push_back(z);
  }
  if (cloud.base.empty() || cloud.base.size() % 3 != 0) {
    return false;
  }
  cloud.box[0] = 100;
  cloud.box[4] = 100;
  cloud.box[8] = 100;
  cloud.pos = cloud.base;
  cloud.forces.assign(cloud.base.size(), 0.0);
  cloud.types.assign(cloud.base.size() / 3, 1);
  cloud.loaded = true;
  return true;
}

void build_lattice(Cloud &cloud) {
  cloud.base.clear();
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        cloud.base.push_back((i + 0.5) * 5.0);
        cloud.base.push_back((j + 0.5) * 5.0);
        cloud.base.push_back((k + 0.5) * 5.0);
      }
    }
  }
  cloud.box[0] = 25;
  cloud.box[4] = 25;
  cloud.box[8] = 25;
  cloud.pos = cloud.base;
  cloud.forces.assign(cloud.base.size(), 0.0);
  cloud.types.assign(cloud.base.size() / 3, 1);
  cloud.loaded = true;
}

void build_water(Cloud &cloud) {
  constexpr int nmol = 10;
  constexpr int nat = 3 * nmol + 1;
  cloud.base.assign(3 * nat, 0.0);
  cloud.types.assign(nat, 1);
  const double roh = 0.9572;
  const double half = 0.5 * 104.52 * std::acos(-1.0) / 180.0;
  const double armx = roh * std::sin(half);
  const double army = roh * std::cos(half);
  for (int m = 0; m < nmol; ++m) {
    const double ox = 3.0 + (m % 5) * 3.2;
    const double oy = 3.0 + (m / 5) * 4.0;
    const int h1 = 2 * m;
    const int h2 = 2 * m + 1;
    const int o = 2 * nmol + m;
    cloud.base[3 * h1] = ox + armx;
    cloud.base[3 * h1 + 1] = oy + army;
    cloud.base[3 * h1 + 2] = 3.0;
    cloud.base[3 * h2] = ox - armx;
    cloud.base[3 * h2 + 1] = oy + army;
    cloud.base[3 * h2 + 2] = 3.0;
    cloud.base[3 * o] = ox;
    cloud.base[3 * o + 1] = oy;
    cloud.base[3 * o + 2] = 3.0;
    cloud.types[o] = 8;
  }
  cloud.base[3 * (nat - 1)] = 8.0;
  cloud.base[3 * (nat - 1) + 1] = 8.0;
  cloud.base[3 * (nat - 1) + 2] = 4.2;
  cloud.box[0] = 24;
  cloud.box[4] = 24;
  cloud.box[8] = 24;
  cloud.pos = cloud.base;
  cloud.forces.assign(cloud.base.size(), 0.0);
  cloud.loaded = true;
}

int ensure_loaded() {
  if (g_lj38.loaded) {
    return 0;
  }
  const std::string root = RGPOT_ASV_DATA;
  if (!load_con(root + "/lj38.con", g_lj38)) {
    return fail("failed to read " + root + "/lj38.con");
  }
  if (!load_xyz(root + "/lj75.xyz", g_lj75)) {
    return fail("failed to read " + root + "/lj75.xyz");
  }
  if (!load_xyz(root + "/lj150.xyz", g_lj150)) {
    return fail("failed to read " + root + "/lj150.xyz");
  }
  if (!load_con(root + "/pt-island.con", g_pt)) {
    return fail("failed to read " + root + "/pt-island.con");
  }
  for (int &z : g_pt.types) {
    z = 78;
  }
  build_water(g_water);
  build_lattice(g_lattice);
  return 0;
}

void wobble(Cloud &cloud, double phase) {
  const double amp = 0.02;
  for (std::size_t k = 0; k < cloud.base.size(); k += 3) {
    const double idx = static_cast<double>(k / 3);
    cloud.pos[k] = cloud.base[k] + amp * std::sin(0.37 * phase + idx);
    cloud.pos[k + 1] = cloud.base[k + 1] + amp * std::sin(0.41 * phase + 1.3 * idx);
    cloud.pos[k + 2] = cloud.base[k + 2] + amp * std::sin(0.53 * phase + 2.1 * idx);
  }
}

template <typename Pot> double eval_pot(Pot &pot, Cloud &cloud) {
  rgpot::ForceInput in{.nAtoms = cloud.pos.size() / 3,
                       .pos = cloud.pos.data(),
                       .atmnrs = cloud.types.data(),
                       .box = cloud.box};
  rgpot::ForceOut out{.F = cloud.forces.data(), .energy = 0.0, .variance = 0.0};
  pot.forceImpl(in, &out);
  return out.energy;
}

rgpot::LJClusterPot &lj_cluster() {
  static rgpot::LJClusterPot pot(rgpot::LJClusterConfig{.cutoff = 15.0});
  return pot;
}

rgpot::LJPot &lj_periodic() {
  static rgpot::LJPot pot(rgpot::LJConfig{.cutoff = 6.0});
  return pot;
}

rgpot::MorsePot &morse() {
  static rgpot::MorsePot pot;
  return pot;
}

double eval_water(Cloud &cloud) {
  double energy = 0.0;
  const int st = rgpot_water_h_force(static_cast<int>(cloud.pos.size() / 3), cloud.pos.data(),
                                     cloud.types.data(), cloud.box, cloud.forces.data(), &energy);
  if (st != 0) {
    g_error = "rgpot_water_h_force status " + std::to_string(st);
    return std::nan("");
  }
  return energy;
}

Cloud *cloud_for(int kind) {
  switch (kind) {
  case kLj38:
    return &g_lj38;
  case kLj75:
    return &g_lj75;
  case kLj150:
    return &g_lj150;
  case kPtWarm:
  case kPtCold:
    return &g_pt;
  case kWater:
    return &g_water;
  case kLjLinkcell:
    return &g_lattice;
  default:
    return nullptr;
  }
}

double eval_kind(int kind, Cloud &cloud) {
  switch (kind) {
  case kLj38:
  case kLj75:
  case kLj150:
    return eval_pot(lj_cluster(), cloud);
  case kPtWarm:
  case kPtCold:
    return eval_pot(morse(), cloud);
  case kWater:
    return eval_water(cloud);
  case kLjLinkcell:
    return eval_pot(lj_periodic(), cloud);
  default:
    return std::nan("");
  }
}

void place_cold(Cloud &cloud, int step) {
  for (std::size_t k = 0; k < cloud.base.size(); k += 3) {
    cloud.pos[k] = cloud.base[k] + static_cast<double>(step);
    cloud.pos[k + 1] = cloud.base[k + 1];
    cloud.pos[k + 2] = cloud.base[k + 2];
  }
}

} // namespace

extern "C" const char *rgpot_asv_last_error(void) {
  return g_error.empty() ? nullptr : g_error.c_str();
}

extern "C" int rgpot_asv_prepare(int kind) {
  if (ensure_loaded() != 0) {
    return 1;
  }
  Cloud *cloud = cloud_for(kind);
  if (cloud == nullptr) {
    return fail("unknown asv kind");
  }
  if (kind == kPtCold) {
    g_cold_step = 0;
    place_cold(*cloud, 0);
    eval_kind(kind, *cloud);
    return 0;
  }
  cloud->pos = cloud->base;
  eval_kind(kind, *cloud);
  wobble(*cloud, 1.0);
  eval_kind(kind, *cloud);
  return 0;
}

extern "C" double rgpot_asv_force(int kind) {
  Cloud *cloud = cloud_for(kind);
  if (cloud == nullptr || !cloud->loaded) {
    g_error = "rgpot_asv_force called before a successful prepare";
    return std::nan("");
  }
  if (kind == kPtCold) {
    g_cold_step += 1;
    place_cold(*cloud, g_cold_step);
    return eval_kind(kind, *cloud);
  }
  wobble(*cloud, 1.0);
  return eval_kind(kind, *cloud);
}
