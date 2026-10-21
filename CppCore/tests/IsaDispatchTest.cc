// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_all.hpp>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/simd/api.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

std::uint64_t bits_of(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

void require_bits(const char *what, const double *got, const double *ref,
                  std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    INFO(what << " i=" << i);
    REQUIRE(bits_of(got[i]) == bits_of(ref[i]));
  }
}

bool runnable_has(const char *name) {
  const int n = rgpot::simd::runnable_target_count();
  for (int i = 0; i < n; ++i) {
    if (std::strcmp(rgpot::simd::runnable_target_name(i), name) == 0) {
      return true;
    }
  }
  return false;
}

struct Built {
  std::vector<double> R;
  std::vector<std::int32_t> rows;
  std::vector<std::int32_t> nbr;
  rgpot::simd::PairView view{};
};

Built half_list(std::vector<double> R, double cutoff, int fold, double w0,
                double w1, double w2) {
  Built b;
  b.R = std::move(R);
  const std::size_t n = b.R.size() / 3;
  b.rows.assign(n + 1, 0);
  for (std::size_t i = 0; i < n; ++i) {
    b.rows[i] = static_cast<std::int32_t>(b.nbr.size());
    for (std::size_t j = i + 1; j < n; ++j) {
      b.nbr.push_back(static_cast<std::int32_t>(j));
    }
    b.rows[i + 1] = static_cast<std::int32_t>(b.nbr.size());
  }
  b.view.R = b.R.data();
  b.view.rows = b.rows.data();
  b.view.nbr = b.nbr.data();
  b.view.n = n;
  b.view.cutoff2 = cutoff * cutoff;
  b.view.fold = fold;
  b.view.w[0] = w0;
  b.view.w[1] = w1;
  b.view.w[2] = w2;
  b.view.inv[0] = w0 != 0.0 ? 1.0 / w0 : 0.0;
  b.view.inv[1] = w1 != 0.0 ? 1.0 / w1 : 0.0;
  b.view.inv[2] = w2 != 0.0 ? 1.0 / w2 : 0.0;
  return b;
}

struct LjOut {
  double energy;
  double F[12];
  double strain[6];
};

LjOut run_lj(const rgpot::simd::PairView &view) {
  LjOut o{};
  rgpot::simd::LjParams params{};
  params.psi2 = 3.4 * 3.4;
  params.four_u0 = 4.0 * 0.0104;
  params.shift = 0.0;
  o.energy = rgpot::simd::lj_force(view, params, o.F, o.strain);
  return o;
}

struct MorseOut {
  double energy;
  double F[12];
  double strain[6];
};

MorseOut run_morse(const rgpot::simd::PairView &view) {
  MorseOut o{};
  rgpot::simd::MorseParams params{};
  params.depth = 0.7102;
  params.range = 1.6047;
  params.r_eq = 2.8970;
  params.two_de_a = 2.0 * 0.7102 * 1.6047;
  o.energy = rgpot::simd::morse_force(view, params, o.F, o.strain);
  return o;
}

struct EamOut {
  double energy;
  double F[9];
};

EamOut run_eam(const double *R, int n) {
  EamOut o{};
  rgpot::simd::EamAtom atoms[3] = {};
  for (int a = 0; a < n; ++a) {
    atoms[a].beta1 = 1.5;
    atoms[a].beta2 = 0.5;
    atoms[a].alpha_m = 1.2;
    atoms[a].r_m = 2.5;
    atoms[a].d_m = 0.3;
    atoms[a].cutoff2 = 36.0;
    atoms[a].coeff[0] = -1.0;
    atoms[a].coeff[1] = 0.1;
    atoms[a].coeff[2] = 0.01;
  }
  const double lengths[3] = {12.0, 12.0, 12.0};
  rgpot::simd::eam_force(n, R, atoms, lengths, o.F, &o.energy);
  return o;
}

struct PotOut {
  double energy;
  std::vector<double> F;
};

PotOut run_lj_pot(const double *R, std::size_t n, const double *box) {
  PotOut o;
  o.F.assign(3 * n, 0.0);
  rgpot::LJPot pot{rgpot::LJConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.4}};
  std::vector<int> types(n, 18);
  rgpot::ForceOut out{o.F.data(), 0.0, 0.0};
  pot.forceImpl(rgpot::ForceInput{n, R, types.data(), box}, &out);
  o.energy = out.energy;
  return o;
}

PotOut run_morse_pot(const double *R, std::size_t n, const double *box) {
  PotOut o;
  o.F.assign(3 * n, 0.0);
  rgpot::MorsePot pot;
  std::vector<int> types(n, 78);
  rgpot::ForceOut out{o.F.data(), 0.0, 0.0};
  pot.forceImpl(rgpot::ForceInput{n, R, types.data(), box}, &out);
  o.energy = out.energy;
  return o;
}

} // namespace

TEST_CASE("SIMD targets match the scalar kernel bit for bit", "[simd][isa]") {
  const char *known[] = {"scalar", "sse2",     "sse4", "avx2",    "avx3",
                         "avx3_zen4", "neon",  "sve",  "sve2",    "sve_256",
                         "rvv",       "emu128"};
  const int nrun = rgpot::simd::runnable_target_count();
  REQUIRE(nrun >= 1);
  REQUIRE(std::strcmp(rgpot::simd::runnable_target_name(0), "scalar") == 0);
  std::string exercised;
  for (int i = 0; i < nrun; ++i) {
    const char *name = rgpot::simd::runnable_target_name(i);
    bool known_name = false;
    for (const char *k : known) {
      if (std::strcmp(name, k) == 0) {
        known_name = true;
      }
    }
    REQUIRE(known_name);
    exercised += name;
    exercised += ' ';
  }
  INFO("exercised " << exercised);

#if defined(__x86_64__) || defined(_M_X64)
  if (std::getenv("RGPOT_SIMD") == nullptr && runnable_has("sse2")) {
    REQUIRE(std::strcmp(rgpot::simd::chosen_target_name(), "scalar") != 0);
  }
#endif

  const double lj_pos[] = {0.2, 0.3, 0.4, 2.8, 0.2, 0.5, 9.6, 0.4, 0.3,
                           4.0, 5.0, 1.0};
  const double morse_pos[] = {1.0, 1.0, 1.0, 1.0 + 2.897, 1.0, 1.0,
                              5.0, 1.2, 1.1};
  const double eam_pos[] = {0.2, 0.2, 0.2, 2.7, 0.3, 0.4, 4.0, 0.2, 0.2};
  const double box[9] = {12, 0, 0, 0, 12, 0, 0, 0, 12};
  Built lj = half_list(std::vector<double>(lj_pos, lj_pos + 12), 8.5, 1, 12, 12,
                       12);
  Built morse =
      half_list(std::vector<double>(morse_pos, morse_pos + 9), 9.5, 1, 20, 20,
                20);

  auto capture = [&](const char *name) {
    rgpot::simd::set_target_for_test(name);
    REQUIRE(std::strcmp(rgpot::simd::chosen_target_name(), name) == 0);
    struct Pack {
      LjOut lj;
      MorseOut morse;
      EamOut eam;
      PotOut lj_pot;
      PotOut morse_pot;
      double fold[24];
    };
    Pack p{run_lj(lj.view),
           run_morse(morse.view),
           run_eam(eam_pos, 3),
           run_lj_pot(lj_pos, 4, box),
           run_morse_pot(morse_pos, 3, box),
           {}};
    const double neg0 = std::copysign(0.0, -1.0);
    const double sub = std::numeric_limits<double>::denorm_min();
    double x[8] = {0.2, 6.0, -4.0, neg0, sub, 3.0, 25.0, 1.0};
    double y[8] = {0.2, -7.0, 3.0, 1.0, neg0, 11.0, sub, 1.0};
    double z[8] = {0.2, 1.0, 9.5, 3.0, 1.0, neg0, -30.0, 1.0};
    rgpot::simd::fold_orthorhombic(x, y, z, 8, 10.0, 10.0, 10.0, 0.1, 0.1, 0.0);
    for (int i = 0; i < 8; ++i) {
      p.fold[3 * i] = x[i];
      p.fold[3 * i + 1] = y[i];
      p.fold[3 * i + 2] = z[i];
    }
    return p;
  };

  rgpot::simd::set_target_for_test("scalar");
  const auto scalar = capture("scalar");

  auto check = [&](const char *name, const auto &got) {
    INFO(name);
    REQUIRE(bits_of(got.lj.energy) == bits_of(scalar.lj.energy));
    require_bits("lj F", got.lj.F, scalar.lj.F, 12);
    require_bits("lj strain", got.lj.strain, scalar.lj.strain, 6);
    REQUIRE(bits_of(got.morse.energy) == bits_of(scalar.morse.energy));
    require_bits("morse F", got.morse.F, scalar.morse.F, 12);
    require_bits("morse strain", got.morse.strain, scalar.morse.strain, 6);
    REQUIRE(bits_of(got.eam.energy) == bits_of(scalar.eam.energy));
    require_bits("eam F", got.eam.F, scalar.eam.F, 9);
    REQUIRE(bits_of(got.lj_pot.energy) == bits_of(scalar.lj_pot.energy));
    require_bits("lj pot F", got.lj_pot.F.data(), scalar.lj_pot.F.data(),
                 got.lj_pot.F.size());
    REQUIRE(bits_of(got.morse_pot.energy) == bits_of(scalar.morse_pot.energy));
    require_bits("morse pot F", got.morse_pot.F.data(),
                 scalar.morse_pot.F.data(), got.morse_pot.F.size());
    require_bits("fold", got.fold, scalar.fold, 24);
  };

  for (int i = 0; i < nrun; ++i) {
    const char *name = rgpot::simd::runnable_target_name(i);
    check(name, capture(name));
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
