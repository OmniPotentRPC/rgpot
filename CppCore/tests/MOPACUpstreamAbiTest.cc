// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MOPACPot/MOPACPot.hpp"
#include "rgpot/MOPACPot/mopac_c_abi.h"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using rgpot::types::AtomMatrix;

namespace {

constexpr std::uint64_t kCanaryBefore = 0xA5A5A5A5A5A5A5A5ULL;
constexpr std::uint64_t kCanaryAfter = 0x5A5A5A5A5A5A5A5AULL;
constexpr std::size_t kCanaryWords = 16;

#if defined(__GNUC__) || defined(__clang__)
#define RGPOT_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define RGPOT_NOINLINE __declspec(noinline)
#else
#define RGPOT_NOINLINE
#endif

using EnergyGradientFn = MopacCResult (*)(int, const double *, const int *,
                                          const void *, std::size_t, double *);

struct GuardedMopacResult {
  volatile std::uint64_t before[kCanaryWords];
  MopacCResult result;
  volatile std::uint64_t after[kCanaryWords];
};

void fill_canaries(volatile std::uint64_t *buf, std::size_t count,
                   std::uint64_t value) {
  for (std::size_t i = 0; i < count; ++i) {
    buf[i] = value;
  }
}

void verify_canaries(const volatile std::uint64_t *buf, std::size_t count,
                     std::uint64_t expected) {
  for (std::size_t i = 0; i < count; ++i) {
    REQUIRE(buf[i] == expected);
  }
}

RGPOT_NOINLINE void call_with_canaries(EnergyGradientFn fn, int n_atoms,
                                       const double *pos, const int *atmnrs,
                                       const MopacCParams *params,
                                       double *grad,
                                       GuardedMopacResult *out) {
  volatile std::uint64_t stack_before[kCanaryWords];
  fill_canaries(stack_before, kCanaryWords, kCanaryBefore);
  fill_canaries(out->before, kCanaryWords, kCanaryBefore);
  fill_canaries(out->after, kCanaryWords, kCanaryAfter);
  volatile std::uint64_t stack_after[kCanaryWords];
  fill_canaries(stack_after, kCanaryWords, kCanaryAfter);

  out->result =
      fn(n_atoms, pos, atmnrs, params, sizeof(*params), grad);

  verify_canaries(stack_before, kCanaryWords, kCanaryBefore);
  verify_canaries(stack_after, kCanaryWords, kCanaryAfter);
  verify_canaries(out->before, kCanaryWords, kCanaryBefore);
  verify_canaries(out->after, kCanaryWords, kCanaryAfter);
}

constexpr std::array<double, 9> kBox = {100.0, 0.0, 0.0, 0.0, 100.0,
                                        0.0,   0.0, 0.0, 100.0};

RGPOT_NOINLINE void call_pot_with_canaries(const rgpot::MOPACPot &pot,
                                           const AtomMatrix &pos,
                                           const std::vector<int> &z,
                                           double *energy_out,
                                           AtomMatrix *forces_out) {
  volatile std::uint64_t stack_before[kCanaryWords];
  fill_canaries(stack_before, kCanaryWords, kCanaryBefore);
  volatile std::uint64_t stack_after[kCanaryWords];
  fill_canaries(stack_after, kCanaryWords, kCanaryAfter);

  *forces_out = AtomMatrix::Zero(z.size(), 3);
  const rgpot::ForceInput in{z.size(), pos.data(), z.data(), kBox.data()};
  rgpot::ForceOut out{};
  out.F = forces_out->data();
  pot.forceImpl(in, &out);
  *energy_out = out.energy;

  verify_canaries(stack_before, kCanaryWords, kCanaryBefore);
  verify_canaries(stack_after, kCanaryWords, kCanaryAfter);
}

RGPOT_NOINLINE void call_pot_expect_throw_with_canaries(
    const rgpot::MOPACPot &pot, const AtomMatrix &pos,
    const std::vector<int> &z) {
  volatile std::uint64_t stack_before[kCanaryWords];
  fill_canaries(stack_before, kCanaryWords, kCanaryBefore);
  volatile std::uint64_t stack_after[kCanaryWords];
  fill_canaries(stack_after, kCanaryWords, kCanaryAfter);

  std::vector<double> forces(z.size() * 3u, 0.0);
  const rgpot::ForceInput in{z.size(), pos.data(), z.data(), kBox.data()};
  rgpot::ForceOut out{};
  out.F = forces.data();
  bool threw = false;
  try {
    pot.forceImpl(in, &out);
  } catch (const std::runtime_error &e) {
    threw = true;
    const std::string what = e.what();
    REQUIRE(what.find("upstream-err:") != std::string::npos);
    REQUIRE(what.find("EEEE") != std::string::npos);
  }
  REQUIRE(threw);

  verify_canaries(stack_before, kCanaryWords, kCanaryBefore);
  verify_canaries(stack_after, kCanaryWords, kCanaryAfter);
}

AtomMatrix threeAtoms() {
  return AtomMatrix{{0.0, 0.0, 0.0}, {1.15, 0.0, 0.0}, {-1.06, 0.0, 0.0}};
}

} // namespace

TEST_CASE("Vendored mopacc ABI layout matches upstream", "[mopac][abi]") {
  constexpr std::size_t expected_params_size =
      (alignof(double) == 8) ? 48u : 36u;
  constexpr std::size_t expected_result_size =
      (alignof(double) == 8) ? 600u : 596u;
  constexpr std::size_t expected_dipole_off =
      (alignof(double) == 8) ? 16u : 12u;
  constexpr std::size_t expected_stress_off =
      (alignof(double) == 8) ? 40u : 36u;
  constexpr std::size_t expected_msg_off = (alignof(double) == 8) ? 88u : 84u;

  static_assert(sizeof(MopacCParams) == expected_params_size);
  static_assert(sizeof(MopacCResult) == expected_result_size);
  static_assert(offsetof(MopacCResult, dipole_debye) == expected_dipole_off);
  static_assert(offsetof(MopacCResult, stress_gpa) == expected_stress_off);
  static_assert(offsetof(MopacCResult, message) == expected_msg_off);

  REQUIRE(RGPOT_MOPACC_C_ABI_VERSION == 2);
  REQUIRE(MOPACC_ABI_VERSION == 2);
  REQUIRE(sizeof(MopacCParams) == expected_params_size);
  REQUIRE(sizeof(MopacCResult) == expected_result_size);
  REQUIRE(offsetof(MopacCResult, dipole_debye) == expected_dipole_off);
  REQUIRE(offsetof(MopacCResult, stress_gpa) == expected_stress_off);
  REQUIRE(offsetof(MopacCResult, message) == expected_msg_off);
}

TEST_CASE("Upstream-layout stub return preserves stack canaries",
          "[mopac][abi]") {
  const char *engine_path = std::getenv("RGPOT_MOPACC_ENGINE");
  REQUIRE(engine_path != nullptr);
  REQUIRE(*engine_path != '\0');

  rgpot::DynLib lib(engine_path);
  REQUIRE(lib.valid());
  auto fn = lib.sym_optional<EnergyGradientFn>("mopacc_energy_gradient");
  REQUIRE(fn != nullptr);

  MopacCParams params{};
  params.charge = 1;
  params.spin = 0;
  params.model = MOPACC_MODEL_AM1;
  params.tolerance = 1.0;
  params.max_time = 3600;
  params.epsilon = 1.0;
  params.solver = MOPACC_SOLVER_MOPAC;

  const std::array<double, 9> pos = {0.0, 0.0, 0.0, 1.15, 0.0,
                                     0.0, -1.06, 0.0, 0.0};
  const std::array<int, 3> ok_z = {1, 6, 7};
  std::array<double, 9> grad{};

  GuardedMopacResult guarded{};
  call_with_canaries(fn, 3, pos.data(), ok_z.data(), &params, grad.data(),
                     &guarded);

  REQUIRE(guarded.result.ok == 1);
  REQUIRE(guarded.result.energy_h == Approx(-1.5 + 0.01 + 0.001));
  REQUIRE(guarded.result.dipole_debye[0] == Approx(1.25));
  REQUIRE(guarded.result.dipole_debye[1] == Approx(-2.50));
  REQUIRE(guarded.result.dipole_debye[2] == Approx(3.75));
  REQUIRE(guarded.result.stress_gpa[0] == Approx(0.1));
  REQUIRE(guarded.result.stress_gpa[5] == Approx(0.6));
  REQUIRE(std::strlen(guarded.result.message) == 511);
  REQUIRE(std::string(guarded.result.message).rfind("upstream-ok:", 0) == 0);

  const std::array<int, 3> bad_z = {0, 6, 7};
  call_with_canaries(fn, 3, pos.data(), bad_z.data(), &params, grad.data(),
                     &guarded);
  REQUIRE(guarded.result.ok == 0);
  REQUIRE(std::strlen(guarded.result.message) == 511);
  REQUIRE(std::string(guarded.result.message).rfind("upstream-err:", 0) == 0);
}

TEST_CASE("MOPACPot succeeds and fails safely against upstream-layout stub",
          "[mopac][abi]") {
  const char *engine_path = std::getenv("RGPOT_MOPACC_ENGINE");
  REQUIRE(engine_path != nullptr);
  REQUIRE(*engine_path != '\0');

  rgpot::MOPACPot::Config cfg;
  cfg.charge = 2;
  cfg.model = MOPACC_MODEL_AM1;
  cfg.engine_path = engine_path;

  rgpot::MOPACPot pot(cfg);
  REQUIRE(pot.available());

  double energy = 0.0;
  AtomMatrix forces(3, 3);
  call_pot_with_canaries(pot, threeAtoms(), {1, 6, 7}, &energy, &forces);

  REQUIRE(energy ==
          Approx((-1.5 + 0.02 + 0.001) * rgpot::units::HARTREE_TO_EV));
  REQUIRE(forces(0, 0) == Approx(0.002 * rgpot::units::NEG_GRAD_TO_FORCE));

  call_pot_expect_throw_with_canaries(pot, threeAtoms(), {0, 6, 7});
}
