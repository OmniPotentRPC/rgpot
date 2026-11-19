// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "rgpot/EAM/EAMPot.hpp"
#ifdef RGPOT_HAS_EAM_TABLE
#include "rgpot/EAMTable/EAMTablePot.hpp"
#endif
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace {

struct Args {
  std::string potential;
  std::size_t atoms = 0;
  int threads = 0;
  int samples = 1;
  double sample_seconds = 0.03;
  std::optional<double> cutoff;
  std::optional<std::size_t> term_limit;
  std::string eam_table_path = "CppCore/tests/data/eam_table/Cu_u3.eam";
  std::optional<std::string> dump_force;
  bool verify_small_cell = false;
};

[[noreturn]] void usage(const char *argv0) {
  std::cerr << "usage: " << argv0
            << " --potential NAME --atoms N --threads N --samples N"
               " --sample-seconds SECONDS [--cutoff VAL] [--term-limit N]"
               " [--eam-table-path PATH] [--dump-force FILE]"
               " [--verify-small-cell]\n";
  std::exit(2);
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--verify-small-cell") {
      args.verify_small_cell = true;
      continue;
    }
    if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
      usage(argv[0]);
    }
    const std::string val = argv[++i];
    if (key == "--potential") {
      args.potential = val;
    } else if (key == "--atoms") {
      args.atoms = static_cast<std::size_t>(std::stoull(val));
    } else if (key == "--threads") {
      args.threads = std::stoi(val);
    } else if (key == "--samples") {
      args.samples = std::stoi(val);
    } else if (key == "--sample-seconds") {
      args.sample_seconds = std::stod(val);
    } else if (key == "--cutoff") {
      args.cutoff = std::stod(val);
    } else if (key == "--term-limit") {
      args.term_limit = static_cast<std::size_t>(std::stoull(val));
    } else if (key == "--eam-table-path") {
      args.eam_table_path = val;
    } else if (key == "--dump-force") {
      args.dump_force = val;
    } else {
      usage(argv[0]);
    }
  }
  if (args.potential.empty() || args.atoms == 0 || args.threads < 1 ||
      args.samples < 1 || !(args.sample_seconds > 0.0)) {
    usage(argv[0]);
  }
  return args;
}

struct LatticeData {
  std::vector<double> positions;
  std::vector<int> atomic_numbers;
  std::array<double, 9> box{};
};

std::tuple<int, int, int> factor_triplet(std::size_t cells) {
  if (cells == 0) {
    throw std::invalid_argument("factor_triplet requires cells > 0");
  }
  int best_x = 1;
  int best_y = 1;
  int best_z = static_cast<int>(cells);
  double best_score = std::numeric_limits<double>::infinity();
  const int limit = static_cast<int>(std::cbrt(static_cast<double>(cells))) + 2;
  for (int x = 1; x <= limit; ++x) {
    if (cells % static_cast<std::size_t>(x) != 0) {
      continue;
    }
    const std::size_t yz = cells / static_cast<std::size_t>(x);
    const int y_limit = static_cast<int>(std::sqrt(static_cast<double>(yz))) + 2;
    for (int y = x; y <= y_limit; ++y) {
      if (yz % static_cast<std::size_t>(y) != 0) {
        continue;
      }
      const int z = static_cast<int>(yz / static_cast<std::size_t>(y));
      if (z < y) {
        continue;
      }
      const double ratio = static_cast<double>(z) / static_cast<double>(x);
      const double score = ratio + static_cast<double>(z - x);
      if (score < best_score) {
        best_score = score;
        best_x = x;
        best_y = y;
        best_z = z;
      }
    }
  }
  return {best_x, best_y, best_z};
}

LatticeData build_fcc(std::size_t atoms, double lattice_a, int z) {
  if (atoms % 4 != 0) {
    throw std::invalid_argument("fcc requires atom count divisible by 4");
  }
  const auto [nx, ny, nz] = factor_triplet(atoms / 4);
  const std::array<std::array<double, 3>, 4> basis{{
      {0.0, 0.0, 0.0},
      {0.0, 0.5, 0.5},
      {0.5, 0.0, 0.5},
      {0.5, 0.5, 0.0},
  }};
  LatticeData out;
  out.positions.reserve(atoms * 3);
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &site : basis) {
          out.positions.push_back((static_cast<double>(ix) + site[0]) * lattice_a);
          out.positions.push_back((static_cast<double>(iy) + site[1]) * lattice_a);
          out.positions.push_back((static_cast<double>(iz) + site[2]) * lattice_a);
        }
      }
    }
  }
  out.atomic_numbers.assign(atoms, z);
  out.box = {
      static_cast<double>(nx) * lattice_a, 0.0, 0.0,
      0.0, static_cast<double>(ny) * lattice_a, 0.0,
      0.0, 0.0, static_cast<double>(nz) * lattice_a};
  return out;
}

LatticeData build_eam_tri3() {
  LatticeData out;
  out.positions = {0.2, 0.3, 0.4, 1.8, 1.4, 1.2, 0.5, 2.2, 1.9};
  out.atomic_numbers = {29, 29, 29};
  out.box = {4.2, 0.0, 0.0, 1.5, 4.0, 0.0, 1.1, 0.8, 3.8};
  return out;
}

std::uint64_t bits_of(double value) {
  std::uint64_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

std::uint64_t hash_forces(const std::vector<double> &forces) {
  std::uint64_t h = 14695981039346656037ULL;
  for (double value : forces) {
    const std::uint64_t bits = bits_of(value);
    for (int shift = 0; shift < 64; shift += 8) {
      h ^= (bits >> shift) & 0xffU;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("force vector size mismatch");
  }
  double best = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::abs(a[i] - b[i]);
    if (d > best) {
      best = d;
    }
  }
  return best;
}

std::string affinity_string() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
    return "unavailable";
  }
  std::ostringstream out;
  bool first = true;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &mask)) {
      continue;
    }
    if (!first) {
      out << ",";
    }
    out << cpu;
    first = false;
  }
  return out.str();
}

int parallel_region_threads() {
#if defined(_OPENMP)
  int observed = 1;
#pragma omp parallel
  {
#pragma omp master
    { observed = omp_get_num_threads(); }
  }
  return observed;
#else
  return 1;
#endif
}

std::string parallel_region_affinity() {
#if defined(_OPENMP)
  int observed = 1;
  std::vector<std::string> per_thread;
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const std::string aff = affinity_string();
#pragma omp single
    {
      observed = omp_get_num_threads();
      per_thread.resize(static_cast<std::size_t>(observed));
    }
#pragma omp barrier
    per_thread[static_cast<std::size_t>(tid)] = aff;
  }
  std::ostringstream out;
  for (std::size_t i = 0; i < per_thread.size(); ++i) {
    if (i != 0) {
      out << ";";
    }
    out << per_thread[i];
  }
  return out.str();
#else
  return affinity_string();
#endif
}

void configure_threads(int threads) {
#if defined(_OPENMP)
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
#else
  (void)threads;
#endif
}

double face_spacing(const double *box, int axis) {
  const double *u = box + 3 * ((axis + 1) % 3);
  const double *v = box + 3 * ((axis + 2) % 3);
  const double cx = u[1] * v[2] - u[2] * v[1];
  const double cy = u[2] * v[0] - u[0] * v[2];
  const double cz = u[0] * v[1] - u[1] * v[0];
  const double area = std::sqrt(cx * cx + cy * cy + cz * cz);
  const double vol = std::abs(
      box[0] * (box[4] * box[8] - box[5] * box[7]) -
      box[1] * (box[3] * box[8] - box[5] * box[6]) +
      box[2] * (box[3] * box[7] - box[4] * box[6]));
  return (area > 0.0 && vol > 0.0) ? (vol / area) : 0.0;
}

double min_face_spacing(const double *box) {
  return std::min({face_spacing(box, 0), face_spacing(box, 1), face_spacing(box, 2)});
}

template <typename Pot>
concept HasSetNeighbourMode = requires(Pot pot) {
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
};

template <typename Pot>
concept HasSetPairTermLimit = requires(Pot pot, std::size_t limit) {
  pot.setPairTermLimit(limit);
};

template <typename Pot>
concept HasListReused = requires(const Pot pot) {
  { pot.listReused() } -> std::convertible_to<bool>;
};

class Runner {
public:
  virtual ~Runner() = default;
  virtual void set_stored_mode() = 0;
  virtual void set_term_limit(std::size_t limit) = 0;
  virtual bool list_reused() const = 0;
  virtual void evaluate(const rgpot::ForceInput &in, rgpot::ForceOut *out) = 0;
};

template <typename Pot>
class PotRunnerT final : public Runner {
public:
  template <typename... BuildArgs>
  explicit PotRunnerT(BuildArgs &&...args)
      : pot_(std::forward<BuildArgs>(args)...) {}

  void set_stored_mode() override {
    if constexpr (HasSetNeighbourMode<Pot>) {
      pot_.setNeighbourMode(rgpot::nlist::Mode::Stored);
    }
  }

  void set_term_limit(std::size_t limit) override {
    if constexpr (HasSetPairTermLimit<Pot>) {
      pot_.setPairTermLimit(limit);
    } else {
      (void)limit;
    }
  }

  bool list_reused() const override {
    if constexpr (HasListReused<Pot>) {
      return pot_.listReused();
    } else {
      return false;
    }
  }

  void evaluate(const rgpot::ForceInput &in, rgpot::ForceOut *out) override {
    pot_.forceImpl(in, out);
  }

private:
  Pot pot_;
};

struct CorrectnessInfo {
  bool checked = false;
  std::string reference_kind = "none";
  double max_abs_energy_diff = 0.0;
  double max_abs_force_diff = 0.0;
  std::size_t got_pairs = 0;
  std::size_t ref_pairs = 0;
  std::string note;
};

void brute_force_lj_minimage(const LatticeData &lat, double cutoff,
                             double psi, double u0, double &energy,
                             std::vector<double> &forces) {
  const std::size_t n = lat.atomic_numbers.size();
  forces.assign(3 * n, 0.0);
  energy = 0.0;
  const double cut2 = cutoff * cutoff;
  const double psi2 = psi * psi;
  const double sr2_c = psi2 / cut2;
  const double sr6_c = sr2_c * sr2_c * sr2_c;
  const double shift = 4.0 * u0 * sr6_c * (sr6_c - 1.0);
  const double lx = lat.box[0];
  const double ly = lat.box[4];
  const double lz = lat.box[8];
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double dx = lat.positions[3 * i] - lat.positions[3 * j];
      double dy = lat.positions[3 * i + 1] - lat.positions[3 * j + 1];
      double dz = lat.positions[3 * i + 2] - lat.positions[3 * j + 2];
      dx -= lx * std::round(dx / lx);
      dy -= ly * std::round(dy / ly);
      dz -= lz * std::round(dz / lz);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0) || r2 > cut2) {
        continue;
      }
      const double inv_r2 = 1.0 / r2;
      const double sr2 = psi2 * inv_r2;
      const double sr6 = sr2 * sr2 * sr2;
      const double b = 4.0 * u0 * sr6;
      const double v = b * (sr6 - 1.0) - shift;
      const double fscale = (6.0 * b * inv_r2) * (2.0 * sr6 - 1.0);
      energy += v;
      forces[3 * i] += fscale * dx;
      forces[3 * i + 1] += fscale * dy;
      forces[3 * i + 2] += fscale * dz;
      forces[3 * j] -= fscale * dx;
      forces[3 * j + 1] -= fscale * dy;
      forces[3 * j + 2] -= fscale * dz;
    }
  }
}

void brute_force_morse_minimage(const LatticeData &lat, const rgpot::MorseConfig &cfg,
                                double &energy, std::vector<double> &forces) {
  const std::size_t n = lat.atomic_numbers.size();
  forces.assign(3 * n, 0.0);
  energy = 0.0;
  const double cut2 = cfg.cutoff * cfg.cutoff;
  const double lx = lat.box[0];
  const double ly = lat.box[4];
  const double lz = lat.box[8];
  const double d_cut = 1.0 - std::exp(-cfg.a * (cfg.cutoff - cfg.re));
  const double e_cut = cfg.De * d_cut * d_cut - cfg.De;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double dx = lat.positions[3 * i] - lat.positions[3 * j];
      double dy = lat.positions[3 * i + 1] - lat.positions[3 * j + 1];
      double dz = lat.positions[3 * i + 2] - lat.positions[3 * j + 2];
      dx -= lx * std::round(dx / lx);
      dy -= ly * std::round(dy / ly);
      dz -= lz * std::round(dz / lz);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0) || r2 > cut2) {
        continue;
      }
      const double r = std::sqrt(r2);
      const double ex = std::exp(-cfg.a * (r - cfg.re));
      const double d = 1.0 - ex;
      const double pair_e = (cfg.De * d * d - cfg.De) - e_cut;
      const double dphi_dr = 2.0 * cfg.De * cfg.a * d * ex;
      const double fscale = -dphi_dr / r;
      energy += pair_e;
      forces[3 * i] += fscale * dx;
      forces[3 * i + 1] += fscale * dy;
      forces[3 * i + 2] += fscale * dz;
      forces[3 * j] -= fscale * dx;
      forces[3 * j + 1] -= fscale * dy;
      forces[3 * j + 2] -= fscale * dz;
    }
  }
}

std::size_t brute_force_multi_image_pair_count(const LatticeData &lat, double cutoff) {
  const std::size_t n = lat.atomic_numbers.size();
  const double cut2 = cutoff * cutoff;
  const int sx = static_cast<int>(std::ceil(cutoff / face_spacing(lat.box.data(), 0))) + 1;
  const int sy = static_cast<int>(std::ceil(cutoff / face_spacing(lat.box.data(), 1))) + 1;
  const int sz = static_cast<int>(std::ceil(cutoff / face_spacing(lat.box.data(), 2))) + 1;
  std::size_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      const double dx0 = lat.positions[3 * j] - lat.positions[3 * i];
      const double dy0 = lat.positions[3 * j + 1] - lat.positions[3 * i + 1];
      const double dz0 = lat.positions[3 * j + 2] - lat.positions[3 * i + 2];
      for (int h = -sx; h <= sx; ++h) {
        for (int k = -sy; k <= sy; ++k) {
          for (int l = -sz; l <= sz; ++l) {
            if (i == j) {
              const int sum = h + k + l;
              if (h == 0 && k == 0 && l == 0) {
                continue;
              }
              if (sum < 0 || (sum == 0 && (l < 0 || (l == 0 && k < 0)))) {
                continue;
              }
            }
            const double dx = dx0 + h * lat.box[0] + k * lat.box[3] + l * lat.box[6];
            const double dy = dy0 + h * lat.box[1] + k * lat.box[4] + l * lat.box[7];
            const double dz = dz0 + h * lat.box[2] + k * lat.box[5] + l * lat.box[8];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > 0.0 && r2 < cut2) {
              ++count;
            }
          }
        }
      }
    }
  }
  return count;
}

#ifdef RGPOT_HAS_EAM_TABLE
CorrectnessInfo verify_eam_table_supercell(const rgpot::EAMTablePot &pot,
                                          const LatticeData &lat, double cutoff,
                                          double got_energy,
                                          const std::vector<double> &got_forces) {
  CorrectnessInfo info;
  info.checked = true;
  info.reference_kind = "replicated_supercell_minimum_image";
  std::array<int, 3> rep{1, 1, 1};
  for (int axis = 0; axis < 3; ++axis) {
    const double sp = face_spacing(lat.box.data(), axis);
    while (!(2.0 * cutoff < static_cast<double>(rep[axis]) * sp)) {
      ++rep[axis];
    }
  }
  const std::size_t n = lat.atomic_numbers.size();
  const std::size_t nrep = static_cast<std::size_t>(rep[0] * rep[1] * rep[2]);
  std::array<double, 9> sbox = {
      rep[0] * lat.box[0], rep[0] * lat.box[1], rep[0] * lat.box[2],
      rep[1] * lat.box[3], rep[1] * lat.box[4], rep[1] * lat.box[5],
      rep[2] * lat.box[6], rep[2] * lat.box[7], rep[2] * lat.box[8]};
  std::vector<double> spos;
  std::vector<int> sz;
  spos.reserve(3 * n * nrep);
  sz.reserve(n * nrep);
  for (int h = 0; h < rep[0]; ++h) {
    for (int k = 0; k < rep[1]; ++k) {
      for (int l = 0; l < rep[2]; ++l) {
        const double sx = h * lat.box[0] + k * lat.box[3] + l * lat.box[6];
        const double sy = h * lat.box[1] + k * lat.box[4] + l * lat.box[7];
        const double szz = h * lat.box[2] + k * lat.box[5] + l * lat.box[8];
        for (std::size_t i = 0; i < n; ++i) {
          spos.push_back(lat.positions[3 * i] + sx);
          spos.push_back(lat.positions[3 * i + 1] + sy);
          spos.push_back(lat.positions[3 * i + 2] + szz);
          sz.push_back(lat.atomic_numbers[i]);
        }
      }
    }
  }
  std::vector<double> sf(3 * n * nrep, 0.0);
  rgpot::ForceOut sout{sf.data(), 0.0, 0.0};
  pot.forceImpl(rgpot::ForceInput{sz.size(), spos.data(), sz.data(), sbox.data()}, &sout);
  const double inv = 1.0 / static_cast<double>(nrep);
  const double ref_e = sout.energy * inv;
  std::vector<double> ref_f(3 * n, 0.0);
  for (std::size_t t = 0; t < nrep; ++t) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t src = t * n + i;
      ref_f[3 * i] += sf[3 * src] * inv;
      ref_f[3 * i + 1] += sf[3 * src + 1] * inv;
      ref_f[3 * i + 2] += sf[3 * src + 2] * inv;
    }
  }
  info.max_abs_energy_diff = std::abs(got_energy - ref_e);
  info.max_abs_force_diff = max_abs_diff(got_forces, ref_f);
  std::ostringstream note;
  note << "rep=" << rep[0] << "x" << rep[1] << "x" << rep[2];
  info.note = note.str();
  return info;
}
#endif

void print_double_array(const std::vector<double> &values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << std::setprecision(17) << values[i];
  }
  std::cout << "]";
}

void print_size_array(const std::vector<std::size_t> &values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_int_array(const std::vector<int> &values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_string_array(const std::vector<std::string> &values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "\"";
    for (char c : values[i]) {
      if (c == '"' || c == '\\') {
        std::cout << "\\";
      }
      std::cout << c;
    }
    std::cout << "\"";
  }
  std::cout << "]";
}

void print_json_str(const std::string &s) {
  std::cout << "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') {
      std::cout << "\\";
    } else if (c == '\n') {
      std::cout << "\\n";
      continue;
    }
    std::cout << c;
  }
  std::cout << "\"";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);
    configure_threads(args.threads);

    LatticeData lattice;
    double cutoff = 0.0;
    double skin = 0.0;
    std::unique_ptr<Runner> runner;
    bool is_nlist_build = false;

    rgpot::LJConfig lj_cfg;
    lj_cfg.cutoff = args.cutoff.value_or(2.0);
    lj_cfg.psi = 1.0;
    lj_cfg.u0 = 1.0;

    rgpot::MorseConfig morse_cfg;
    morse_cfg.cutoff = args.cutoff.value_or(6.0);

    rgpot::SuttonChenConfig sc_cfg = rgpot::sutton_chen_platinum();
    sc_cfg.periodic = true;
    sc_cfg.cutoff = args.cutoff.value_or(3.0);

#ifdef RGPOT_HAS_EAM_TABLE
    std::unique_ptr<rgpot::EAMTablePot> eam_table_verify_pot;
#endif

    if (args.potential == "nlist_build") {
      is_nlist_build = true;
      cutoff = args.cutoff.value_or(5.0);
      skin = 1.0;
      lattice = build_fcc(args.atoms, 3.615, 29);
    } else if (args.potential == "eam_table" || args.potential == "eam_table_tri3") {
#ifdef RGPOT_HAS_EAM_TABLE
      rgpot::EAMTableConfig cfg;
      cfg.path = args.eam_table_path;
      cfg.format = rgpot::EAMTableFormat::Funcfl;
      cfg.periodic = true;
      cfg.stress = false;
      eam_table_verify_pot = std::make_unique<rgpot::EAMTablePot>(cfg);
      cutoff = eam_table_verify_pot->locality().cutoff;
      skin = 1.0;
      if (args.potential == "eam_table_tri3") {
        if (args.atoms != 3) {
          throw std::invalid_argument("eam_table_tri3 requires --atoms 3");
        }
        lattice = build_eam_tri3();
      } else {
        lattice = build_fcc(args.atoms, 3.615, 29);
      }
      runner = std::make_unique<PotRunnerT<rgpot::EAMTablePot>>(cfg);
#else
      throw std::runtime_error("built without RGPOT_HAS_EAM_TABLE");
#endif
    } else if (args.potential == "eam") {
      cutoff = 7.1;
      skin = 7.1 * 0.2;
      lattice = build_fcc(args.atoms, 4.05, 13);
      runner = std::make_unique<PotRunnerT<rgpot::EAMPot>>();
    } else if (args.potential == "morse") {
      cutoff = morse_cfg.cutoff;
      skin = rgpot::MorsePot::neighbourSkin(morse_cfg.cutoff);
      lattice = build_fcc(args.atoms, 4.05, 78);
      runner = std::make_unique<PotRunnerT<rgpot::MorsePot>>(morse_cfg);
    } else if (args.potential == "lj") {
      cutoff = lj_cfg.cutoff;
      skin = 1.0;
      const double a = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
      lattice = build_fcc(args.atoms, a, 18);
      runner = std::make_unique<PotRunnerT<rgpot::LJPot>>(lj_cfg);
    } else if (args.potential == "sutton-chen") {
      cutoff = sc_cfg.cutoff;
      skin = sc_cfg.cutoff / 5.0;
      const rgpot::SuttonChenScales sc_scales = rgpot::sutton_chen_scales(sc_cfg);
      lattice = build_fcc(args.atoms, sc_scales.nearest_neighbor * std::sqrt(2.0), 78);
      runner = std::make_unique<PotRunnerT<rgpot::SuttonChenPot>>(sc_cfg);
    } else {
      throw std::invalid_argument("unknown potential: " + args.potential);
    }

    const std::size_t n_atoms = lattice.atomic_numbers.size();
    const double min_face = min_face_spacing(lattice.box.data());
    const bool small_cell_multi_image_regime = min_face <= 2.0 * cutoff;

    std::vector<double> total_ns_per_eval;
    std::vector<std::size_t> evals_per_sample;
    std::vector<int> omp_max_threads_per_sample;
    std::vector<int> omp_parallel_threads_per_sample;
    std::vector<std::string> affinity_per_sample;
    std::vector<std::string> omp_parallel_affinity_per_sample;

    double measured_energy = 0.0;
    std::vector<double> measured_forces(3 * n_atoms, 0.0);
    std::size_t last_pair_count = 0;
    bool list_reused_flag = false;

    bool repeat_e_bit_ok = true;
    bool repeat_f_bit_ok = true;
    double repeat_max_f_diff = 0.0;

    rgpot::nlist::Table table;
    rgpot::nlist::Query query;
    query.cutoff = cutoff;
    query.skin = skin;
    query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
    query.cut = rgpot::nlist::Cut::Loose;
    query.periodic = {true, true, true};
    query.fields = {false, false, false};

    rgpot::ForceInput in{n_atoms, lattice.positions.data(),
                         lattice.atomic_numbers.data(), lattice.box.data()};

    if (is_nlist_build) {
      for (int w = 0; w < 3; ++w) {
        rgpot::nlist::build(table, lattice.positions.data(), n_atoms,
                            lattice.box.data(), query);
      }
      last_pair_count = table.size();
    } else {
      runner->set_stored_mode();
      if (args.term_limit.has_value()) {
        runner->set_term_limit(*args.term_limit);
      }
      for (int w = 0; w < 3; ++w) {
        std::fill(measured_forces.begin(), measured_forces.end(), 0.0);
        rgpot::ForceOut out{measured_forces.data(), 0.0, 0.0};
        runner->evaluate(in, &out);
        if (!std::isfinite(out.energy)) {
          throw std::runtime_error("non-finite energy in warmup");
        }
        measured_energy = out.energy;
      }
      list_reused_flag = runner->list_reused();
    }

    for (int sample = 0; sample < args.samples; ++sample) {
      int omp_max = 1;
#if defined(_OPENMP)
      omp_max = omp_get_max_threads();
#endif
      const int team_threads = parallel_region_threads();
      const std::string team_aff = parallel_region_affinity();
      omp_max_threads_per_sample.push_back(omp_max);
      omp_parallel_threads_per_sample.push_back(team_threads);
      affinity_per_sample.push_back(affinity_string());
      omp_parallel_affinity_per_sample.push_back(team_aff);

      std::size_t evals = 0;
      const auto t0 = std::chrono::steady_clock::now();
      while (true) {
        if (is_nlist_build) {
          rgpot::nlist::build(table, lattice.positions.data(), n_atoms,
                              lattice.box.data(), query);
          last_pair_count = table.size();
        } else {
          std::fill(measured_forces.begin(), measured_forces.end(), 0.0);
          rgpot::ForceOut out{measured_forces.data(), 0.0, 0.0};
          runner->evaluate(in, &out);
          if (!std::isfinite(out.energy)) {
            throw std::runtime_error("non-finite energy");
          }
          measured_energy = out.energy;
        }
        ++evals;
        const double elapsed_s = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - t0)
                                     .count();
        if (elapsed_s >= args.sample_seconds && evals >= 1) {
          const double elapsed_ns = std::chrono::duration<double, std::nano>(
                                        std::chrono::steady_clock::now() - t0)
                                        .count();
          total_ns_per_eval.push_back(elapsed_ns / static_cast<double>(evals));
          evals_per_sample.push_back(evals);
          break;
        }
      }
    }

    if (!is_nlist_build) {
      std::vector<double> repeat_forces(3 * n_atoms, 0.0);
      rgpot::ForceOut rout{repeat_forces.data(), 0.0, 0.0};
      runner->evaluate(in, &rout);
      repeat_e_bit_ok = (bits_of(rout.energy) == bits_of(measured_energy));
      repeat_max_f_diff = max_abs_diff(repeat_forces, measured_forces);
      repeat_f_bit_ok =
          (std::memcmp(repeat_forces.data(), measured_forces.data(),
                       repeat_forces.size() * sizeof(double)) == 0);
    } else {
      rgpot::nlist::Table table2;
      rgpot::nlist::build(table2, lattice.positions.data(), n_atoms,
                          lattice.box.data(), query);
      repeat_f_bit_ok = (table2.size() == last_pair_count);
      repeat_e_bit_ok = repeat_f_bit_ok;
    }

    CorrectnessInfo corr;
    if (args.verify_small_cell) {
      if (args.potential == "nlist_build") {
        rgpot::nlist::Query strict_q = query;
        strict_q.skin = 0.0;
        strict_q.cut = rgpot::nlist::Cut::Strict;
        rgpot::nlist::Table strict_tbl;
        rgpot::nlist::build(strict_tbl, lattice.positions.data(), n_atoms,
                            lattice.box.data(), strict_q);
        corr.checked = true;
        corr.reference_kind = "brute_force_multi_image_pairs";
        corr.got_pairs = strict_tbl.size();
        corr.ref_pairs = brute_force_multi_image_pair_count(lattice, cutoff);
        corr.max_abs_energy_diff =
            static_cast<double>(corr.got_pairs > corr.ref_pairs
                                    ? (corr.got_pairs - corr.ref_pairs)
                                    : (corr.ref_pairs - corr.got_pairs));
        corr.note = "strict half-list pair count vs brute-force multi-image";
      } else if (args.potential == "lj") {
        double ref_e = 0.0;
        std::vector<double> ref_f;
        brute_force_lj_minimage(lattice, lj_cfg.cutoff, lj_cfg.psi, lj_cfg.u0,
                                ref_e, ref_f);
        corr.checked = true;
        corr.reference_kind = "brute_force_minimum_image";
        corr.max_abs_energy_diff = std::abs(measured_energy - ref_e);
        corr.max_abs_force_diff = max_abs_diff(measured_forces, ref_f);
        corr.note = "minimum-image pair potential";
      } else if (args.potential == "morse") {
        double ref_e = 0.0;
        std::vector<double> ref_f;
        brute_force_morse_minimage(lattice, morse_cfg, ref_e, ref_f);
        corr.checked = true;
        corr.reference_kind = "brute_force_minimum_image";
        corr.max_abs_energy_diff = std::abs(measured_energy - ref_e);
        corr.max_abs_force_diff = max_abs_diff(measured_forces, ref_f);
        corr.note = "minimum-image pair potential";
      } else if (args.potential == "eam_table" ||
                 args.potential == "eam_table_tri3") {
#ifdef RGPOT_HAS_EAM_TABLE
        corr = verify_eam_table_supercell(*eam_table_verify_pot, lattice, cutoff,
                                          measured_energy, measured_forces);
#endif
      }
    }

    if (args.dump_force.has_value()) {
      std::ofstream out_file(*args.dump_force, std::ios::binary);
      if (!out_file) {
        throw std::runtime_error("cannot open dump file");
      }
      const std::uint64_t n = static_cast<std::uint64_t>(n_atoms);
      out_file.write(reinterpret_cast<const char *>(&n), sizeof(n));
      out_file.write(reinterpret_cast<const char *>(&measured_energy),
                     sizeof(double));
      out_file.write(reinterpret_cast<const char *>(measured_forces.data()),
                     static_cast<std::streamsize>(measured_forces.size() *
                                                  sizeof(double)));
    }

    std::cout << "{";
    std::cout << "\"potential\":";
    print_json_str(args.potential);
    std::cout << ",\"atoms\":" << n_atoms;
    std::cout << ",\"threads\":" << args.threads;
    std::cout << ",\"cutoff\":" << std::setprecision(17) << cutoff;
    std::cout << ",\"min_face_spacing\":" << std::setprecision(17) << min_face;
    std::cout << ",\"small_cell_multi_image_regime\":"
              << (small_cell_multi_image_regime ? "true" : "false");
    std::cout << ",\"list_reused\":" << (list_reused_flag ? "true" : "false");
    std::cout << ",\"last_pair_count\":" << last_pair_count;
    std::cout << ",\"total_ns_per_eval\":";
    print_double_array(total_ns_per_eval);
    std::cout << ",\"evals_per_sample\":";
    print_size_array(evals_per_sample);
    std::cout << ",\"omp_max_threads_per_sample\":";
    print_int_array(omp_max_threads_per_sample);
    std::cout << ",\"omp_parallel_threads_per_sample\":";
    print_int_array(omp_parallel_threads_per_sample);
    std::cout << ",\"affinity_per_sample\":";
    print_string_array(affinity_per_sample);
    std::cout << ",\"omp_parallel_affinity_per_sample\":";
    print_string_array(omp_parallel_affinity_per_sample);
    std::cout << ",\"energy\":" << std::setprecision(17) << measured_energy;
    std::cout << ",\"energy_bits\":\"0x" << std::hex << bits_of(measured_energy)
              << std::dec << "\"";
    std::cout << ",\"force_hash\":\"0x" << std::hex
              << hash_forces(measured_forces) << std::dec << "\"";
    std::cout << ",\"repeat_energy_bit_identical\":"
              << (repeat_e_bit_ok ? "true" : "false");
    std::cout << ",\"repeat_force_bit_identical\":"
              << (repeat_f_bit_ok ? "true" : "false");
    std::cout << ",\"repeat_max_abs_force_diff\":" << std::setprecision(17)
              << repeat_max_f_diff;
    std::cout << ",\"correctness\":{";
    std::cout << "\"checked\":" << (corr.checked ? "true" : "false");
    std::cout << ",\"reference_kind\":";
    print_json_str(corr.reference_kind);
    std::cout << ",\"max_abs_energy_diff\":" << std::setprecision(17)
              << corr.max_abs_energy_diff;
    std::cout << ",\"max_abs_force_diff\":" << std::setprecision(17)
              << corr.max_abs_force_diff;
    std::cout << ",\"got_pairs\":" << corr.got_pairs;
    std::cout << ",\"ref_pairs\":" << corr.ref_pairs;
    std::cout << ",\"note\":";
    print_json_str(corr.note);
    std::cout << "}";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception &err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "error: unknown exception\n";
    return 1;
  }
}
