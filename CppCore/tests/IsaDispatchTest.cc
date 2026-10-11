// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/simd/api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Sha256 {
  uint32_t s[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  uint64_t bits = 0;
  unsigned char buf[64] = {};
  int n = 0;

  static uint32_t rotr(uint32_t x, int c) {
    return (x >> c) | (x << (32 - c));
  }

  void block(const unsigned char *p) {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
             (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5],
             g = s[6], h = s[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = h + S1 + ch + k[i] + w[i];
      const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    s[0] += a;
    s[1] += b;
    s[2] += c;
    s[3] += d;
    s[4] += e;
    s[5] += f;
    s[6] += g;
    s[7] += h;
  }

  void update(const void *data, std::size_t len) {
    const auto *p = static_cast<const unsigned char *>(data);
    bits += uint64_t(len) * 8;
    while (len > 0) {
      const int take = std::min<int>(64 - n, static_cast<int>(len));
      std::memcpy(buf + n, p, static_cast<std::size_t>(take));
      n += take;
      p += take;
      len -= static_cast<std::size_t>(take);
      if (n == 64) {
        block(buf);
        n = 0;
      }
    }
  }

  void f64(double x) {
    std::uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) {
      b[i] = static_cast<unsigned char>(u >> (8 * i));
    }
    update(b, 8);
  }

  std::string hex() {
    unsigned char tail[128];
    std::memcpy(tail, buf, static_cast<std::size_t>(n));
    tail[n] = 0x80;
    int end = n + 1;
    if (n >= 56) {
      while (end < 64) {
        tail[end++] = 0;
      }
      block(tail);
      end = 0;
    }
    while (end < 56) {
      tail[end++] = 0;
    }
    for (int i = 7; i >= 0; --i) {
      tail[end++] = static_cast<unsigned char>(bits >> (8 * i));
    }
    block(tail);
    std::string out(64, '0');
    for (int i = 0; i < 8; ++i) {
      for (int b = 0; b < 4; ++b) {
        const unsigned v = (s[i] >> (24 - 8 * b)) & 0xffu;
        static const char *digits = "0123456789abcdef";
        out[static_cast<std::size_t>(i * 8 + b * 2)] = digits[v >> 4];
        out[static_cast<std::size_t>(i * 8 + b * 2 + 1)] = digits[v & 0xfu];
      }
    }
    return out;
  }
};

bool runnable_has(const char *name) {
  const int n = rgpot::simd::runnable_target_count();
  for (int i = 0; i < n; ++i) {
    if (std::strcmp(rgpot::simd::runnable_target_name(i), name) == 0) {
      return true;
    }
  }
  return false;
}

std::string lj_hash() {
  const double box[9] = {18, 0, 0, 0, 18, 0, 0, 0, 18};
  const double ulp = std::nextafter(2.5, 3.0);
  const double sub = std::numeric_limits<double>::denorm_min();
  const double neg0 = std::copysign(0.0, -1.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double pos[4][12] = {
      {1.0, 2.0, 3.0, 4.2, 2.0, 3.0, 0, 0, 0, 0, 0, 0},
      {neg0, 1.0, 1.0, 3.1, 1.0, sub, 6.0, 1.0, 1.0, 0, 0, 0},
      {1.0, 1.0, 1.0, ulp, 4.0, 1.0, 1.0, 1.0, 7.0, 9.0, 1.0, 1.0},
      {nan, 1.0, 1.0, 8.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0},
  };
  const int natoms[4] = {2, 3, 4, 2};
  rgpot::LJPot pot{rgpot::LJConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.4}};
  const int types[4] = {18, 18, 18, 18};
  Sha256 sha;
  for (int case_i = 0; case_i < 4; ++case_i) {
    double F[12] = {};
    rgpot::ForceOut out{F, 0.0, 0.0, {}, 0};
    pot.forceImpl(rgpot::ForceInput{static_cast<std::size_t>(natoms[case_i]),
                                    pos[case_i], types, box},
                  &out);
    sha.f64(out.energy);
    for (int k = 0; k < 3 * natoms[case_i]; ++k) {
      sha.f64(F[k]);
    }
    for (int k = 0; k < 6; ++k) {
      sha.f64(out.stress[k]);
    }
  }
  return sha.hex();
}

std::string morse_hash() {
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  const double neg0 = std::copysign(0.0, -1.0);
  const double sub = std::numeric_limits<double>::denorm_min();
  const double ulp = std::nextafter(2.9, 4.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double pos[3][9] = {
      {1, 1, 1, 1 + 2.897, 1, 1, 0, 0, 0},
      {neg0, sub, 1, ulp, 1, 1, 5, 1, 1},
      {nan, 1, 1, 6, 1, 1, 0, 0, 0},
  };
  const int natoms[3] = {2, 3, 2};
  rgpot::MorsePot pot;
  const int types[3] = {78, 78, 78};
  Sha256 sha;
  for (int case_i = 0; case_i < 3; ++case_i) {
    double F[9] = {};
    rgpot::ForceOut out{F, 0.0, 0.0, {}, 0};
    pot.forceImpl(rgpot::ForceInput{static_cast<std::size_t>(natoms[case_i]),
                                    pos[case_i], types, box},
                  &out);
    sha.f64(out.energy);
    for (int k = 0; k < 3 * natoms[case_i]; ++k) {
      sha.f64(F[k]);
    }
    for (int k = 0; k < 6; ++k) {
      sha.f64(out.stress[k]);
    }
  }
  return sha.hex();
}

std::string eam_hash() {
  // Direct kernel. EAMPot's cell-list prelude is not the dispatched loop,
  // and it casts coordinates to an integer cell.
  const double neg0 = std::copysign(0.0, -1.0);
  const double sub = std::numeric_limits<double>::denorm_min();
  const double ulp = std::nextafter(2.5, 3.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double cases[][9] = {
      {0.2, 0.2, 0.2, 2.7, 0.3, 0.4, 0, 0, 0},
      {neg0, sub, 0.5, ulp, 0.4, 0.5, 4.0, 0.2, 0.2},
      {nan, 0.2, 0.2, 3.0, 0.2, 0.2, 0, 0, 0},
  };
  const int natoms[3] = {2, 3, 2};
  const double lengths[3] = {12, 12, 12};
  Sha256 sha;
  {
    const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
    const double pos[6] = {0, 0, 0, 2.5, 0, 0};
    const int types[2] = {13, 13};
    double F[6] = {};
    rgpot::ForceOut out{F, 0.0, 0.0, {}, 0};
    rgpot::EAMPot pot;
    pot.forceImpl(rgpot::ForceInput{2, pos, types, box}, &out);
    sha.f64(out.energy);
    for (double v : F) {
      sha.f64(v);
    }
  }
  for (int case_i = 0; case_i < 3; ++case_i) {
    rgpot::simd::EamAtom atoms[3] = {};
    for (int a = 0; a < natoms[case_i]; ++a) {
      atoms[a].beta1 = 1.5;
      atoms[a].beta2 = 0.5;
      atoms[a].alpha_m = 1.2;
      atoms[a].r_m = 2.5;
      atoms[a].d_m = 0.3;
      atoms[a].cutoff2 = 36;
      atoms[a].coeff[0] = -1.0;
      atoms[a].coeff[1] = 0.1;
      atoms[a].coeff[2] = 0.01;
    }
    double F[9] = {};
    double U = 0;
    rgpot::simd::eam_force(natoms[case_i], cases[case_i], atoms, lengths, F,
                           &U);
    sha.f64(U);
    for (int k = 0; k < 3 * natoms[case_i]; ++k) {
      sha.f64(F[k]);
    }
  }
  return sha.hex();
}

void fill_fcc(int ncell, double a, std::vector<double> &R, double box[9]) {
  static const double basis[4][3] = {
      {0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  R.clear();
  for (int i = 0; i < ncell; ++i) {
    for (int j = 0; j < ncell; ++j) {
      for (int k = 0; k < ncell; ++k) {
        for (int b = 0; b < 4; ++b) {
          R.push_back(a * (static_cast<double>(i) + basis[b][0]));
          R.push_back(a * (static_cast<double>(j) + basis[b][1]));
          R.push_back(a * (static_cast<double>(k) + basis[b][2]));
        }
      }
    }
  }
  for (int i = 0; i < 9; ++i) {
    box[i] = 0;
  }
  const double L = a * static_cast<double>(ncell);
  box[0] = box[4] = box[8] = L;
}

template <class Pot>
std::string fcc_hash(Pot &pot, int ncell, double a, int atom_type) {
  std::vector<double> R;
  double box[9];
  fill_fcc(ncell, a, R, box);
  const std::size_t n = R.size() / 3;
  std::vector<int> types(n, atom_type);
  std::vector<double> F(3 * n, 0.0);
  rgpot::ForceOut out{F.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl(rgpot::ForceInput{n, R.data(), types.data(), box}, &out);
  Sha256 sha;
  sha.f64(out.energy);
  for (double v : F) {
    sha.f64(v);
  }
  for (int k = 0; k < 6; ++k) {
    sha.f64(out.stress[k]);
  }
  return sha.hex();
}

// FCC shells past one width-8 round. Morse and EAM run two AVX2 vectors
// or four SSE vectors, and a sorted neighbour row mixes a near-equilibrium
// pair (the tiny exp path) with a nearest-neighbour pair (the large
// reduction), so one chunk takes the all-lanes path and another takes both.
std::string lj_fcc_hash() {
  rgpot::LJPot pot{rgpot::LJConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.4}};
  return fcc_hash(pot, 6, 5.26, 18);
}

std::string morse_fcc_hash() {
  rgpot::MorsePot pot;
  return fcc_hash(pot, 4, 2.9, 78);
}

std::string eam_fcc_hash() {
  rgpot::EAMPot pot;
  return fcc_hash(pot, 4, 4.05, 13);
}

std::string fold_hash() {
  const double neg0 = std::copysign(0.0, -1.0);
  const double sub = std::numeric_limits<double>::denorm_min();
  const double ulp = std::nextafter(3.0, 4.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  double x[8] = {0.2, 6.0, -4.0, neg0, sub, ulp, 25.0, nan};
  double y[8] = {0.2, -7.0, 3.0, 1.0, neg0, 11.0, sub, 1.0};
  double z[8] = {0.2, 1.0, 9.5, ulp, 1.0, neg0, -30.0, nan};
  rgpot::simd::fold_orthorhombic(x, y, z, 8, 10.0, 10.0, 10.0, 0.1, 0.1, 0.0);
  Sha256 sha;
  for (int i = 0; i < 8; ++i) {
    sha.f64(x[i]);
    sha.f64(y[i]);
    sha.f64(z[i]);
  }
  return sha.hex();
}

std::string golden_value(const std::string &key) {
  std::ifstream in(RGPOT_ISA_GOLDEN);
  REQUIRE(in.good());
  std::string line;
  const std::string needle = key + " = \"";
  while (std::getline(in, line)) {
    const auto at = line.find(needle);
    if (at == std::string::npos) {
      continue;
    }
    const auto start = at + needle.size();
    const auto end = line.find('"', start);
    REQUIRE(end != std::string::npos);
    return line.substr(start, end - start);
  }
  FAIL("missing golden key " << key);
  return {};
}

} // namespace

TEST_CASE("SIMD targets match the scalar kernel bit for bit", "[simd][isa]") {
  std::cout << "omp_num_threads="
            << (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS")
                                               : "unset")
            << "\n";
  std::ifstream cpu("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpu, line)) {
    if (line.rfind("flags", 0) == 0) {
      std::cout << "cpu_flags: " << line << "\n";
      break;
    }
  }

  const char *known[] = {"scalar", "sse2",  "sse4", "avx2", "avx3",
                         "avx3_zen4", "neon", "sve",  "sve2", "sve_256",
                         "rvv", "emu128"};
  std::cout << "exercised:";
  const int nrun = rgpot::simd::runnable_target_count();
  REQUIRE(nrun >= 1);
  REQUIRE(std::strcmp(rgpot::simd::runnable_target_name(0), "scalar") == 0);
  for (int i = 0; i < nrun; ++i) {
    std::cout << " " << rgpot::simd::runnable_target_name(i);
  }
  std::cout << "\nnot_exercised:";
  for (const char *name : known) {
    if (!runnable_has(name)) {
      std::cout << " " << name;
    }
  }
  std::cout << "\n";

#if defined(__x86_64__) || defined(_M_X64)
  // Highway is compiled in: an x86-64 host stays off the portable kernel.
  // A build without Highway has only that kernel, so the check does not apply.
  if (std::getenv("RGPOT_SIMD") == nullptr && runnable_has("sse2")) {
    REQUIRE(std::strcmp(rgpot::simd::chosen_target_name(), "scalar") != 0);
  }
#endif

  rgpot::simd::set_target_for_test("scalar");
  const std::string lj0 = lj_hash();
  const std::string mo0 = morse_hash();
  const std::string ea0 = eam_hash();
  const std::string fo0 = fold_hash();
  const std::string lj_fcc0 = lj_fcc_hash();
  const std::string mo_fcc0 = morse_fcc_hash();
  const std::string ea_fcc0 = eam_fcc_hash();
  std::cout << "golden lj=" << lj0 << "\n";
  std::cout << "golden morse=" << mo0 << "\n";
  std::cout << "golden eam=" << ea0 << "\n";
  std::cout << "golden fold=" << fo0 << "\n";
  std::cout << "golden lj_fcc=" << lj_fcc0 << "\n";
  std::cout << "golden morse_fcc=" << mo_fcc0 << "\n";
  std::cout << "golden eam_fcc=" << ea_fcc0 << "\n";
  if (std::getenv("RGPOT_ISA_SKIP_GOLDEN") == nullptr) {
    REQUIRE(lj0 == golden_value("lj"));
    REQUIRE(mo0 == golden_value("morse"));
    REQUIRE(ea0 == golden_value("eam"));
    REQUIRE(fo0 == golden_value("fold"));
    REQUIRE(lj_fcc0 == golden_value("lj_fcc"));
    REQUIRE(mo_fcc0 == golden_value("morse_fcc"));
    REQUIRE(ea_fcc0 == golden_value("eam_fcc"));
  }

  for (int i = 0; i < nrun; ++i) {
    const char *name = rgpot::simd::runnable_target_name(i);
    rgpot::simd::set_target_for_test(name);
    REQUIRE(std::strcmp(rgpot::simd::chosen_target_name(), name) == 0);
    REQUIRE(lj_hash() == lj0);
    REQUIRE(morse_hash() == mo0);
    REQUIRE(eam_hash() == ea0);
    REQUIRE(fold_hash() == fo0);
    REQUIRE(lj_fcc_hash() == lj_fcc0);
    REQUIRE(morse_fcc_hash() == mo_fcc0);
    REQUIRE(eam_fcc_hash() == ea_fcc0);
  }

  const char *before = rgpot::simd::chosen_target_name();
  std::string saved = before;
  rgpot::simd::set_target_for_test("not_a_target");
  REQUIRE(saved == rgpot::simd::chosen_target_name());

  rgpot::simd::set_target_for_test("rvv");
  if (runnable_has("rvv")) {
    REQUIRE(std::strcmp(rgpot::simd::chosen_target_name(), "rvv") == 0);
  } else if (nrun > 1) {
    const char *fell = rgpot::simd::chosen_target_name();
    REQUIRE(std::strcmp(fell, "scalar") != 0);
    REQUIRE(runnable_has(fell));
  } else {
    REQUIRE(std::strcmp(rgpot::simd::chosen_target_name(), "scalar") == 0);
  }
}
