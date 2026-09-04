// MIT License
// Copyright 2023--present rgpot developers
//
// Wall-time LJPot operator() (and optional ExprPot("lj")) on the
// ExprPotTest two-atom Ar fixture. Prints ns/call. Not a meson test.
// Does not time XcKernel.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#ifdef RGPOT_HAS_EXPR
#include "rgpot/ExprPot/ExprPot.hpp"
#endif

using rgpot::types::AtomMatrix;
using clock_type = std::chrono::steady_clock;

namespace {

// ExprPotTest identity fixture (Angstrom, eV).
AtomMatrix twoAtomPositions() {
  return AtomMatrix{{0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}};
}

std::vector<int> twoAtomTypes() { return {18, 18}; }

std::array<std::array<double, 3>, 3> wideCell() {
  return {{{40.0, 0.0, 0.0}, {0.0, 40.0, 0.0}, {0.0, 0.0, 40.0}}};
}

std::int64_t parse_positive(const char *s, const char *name) {
  char *end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || v <= 0) {
    std::fprintf(stderr, "invalid %s: %s\n", name, s);
    std::exit(2);
  }
  return static_cast<std::int64_t>(v);
}

double ns_per_call(clock_type::duration d, std::int64_t n) {
  const double ns =
      std::chrono::duration<double, std::nano>(d).count();
  return ns / static_cast<double>(n);
}

} // namespace

int main(int argc, char **argv) {
  std::int64_t n_calls = 100000;
  std::int64_t warmup = 1000;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      n_calls = parse_positive(argv[++i], "--n");
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = parse_positive(argv[++i], "--warmup");
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::fprintf(stdout,
                   "usage: time_lj_force [--n N] [--warmup W]\n"
                   "ExprPotTest two-atom Ar fixture, default LJConfig.\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      return 2;
    }
  }

  const auto pos = twoAtomPositions();
  const auto types = twoAtomTypes();
  const auto box = wideCell();

  rgpot::LJPot lj;
  double energy = 0.0;
  for (std::int64_t i = 0; i < warmup; ++i) {
    auto [e, f, v] = lj(pos, types, box);
    energy = e;
    (void)f;
    (void)v;
  }
  const auto t0 = clock_type::now();
  for (std::int64_t i = 0; i < n_calls; ++i) {
    auto [e, f, v] = lj(pos, types, box);
    energy = e;
    (void)f;
    (void)v;
  }
  const auto t1 = clock_type::now();
  const double a_ns = ns_per_call(t1 - t0, n_calls);

  std::printf("fixture=exprpot_two_atom\n");
  std::printf("n_atoms=2\n");
  std::printf("n_calls=%lld\n", static_cast<long long>(n_calls));
  std::printf("warmup=%lld\n", static_cast<long long>(warmup));
  std::printf("A_operator_ns_per_call=%.4f\n", a_ns);
  std::printf("A_energy_eV=%.16g\n", energy);

  std::printf(
      "A_handle_note=operator() is the timed A path; from_impl trampolines "
      "the same forceImpl\n");

#ifdef RGPOT_HAS_EXPR
  {
    std::vector<rgpot::ExprPot::Term> terms;
    terms.emplace_back("lj", std::make_unique<rgpot::LJPot>());
    rgpot::ExprPot expr("lj", std::move(terms));
    double c_energy = 0.0;
    for (std::int64_t i = 0; i < warmup; ++i) {
      auto [e, f, v] = expr(pos, types, box);
      c_energy = e;
      (void)f;
      (void)v;
    }
    const auto c0 = clock_type::now();
    for (std::int64_t i = 0; i < n_calls; ++i) {
      auto [e, f, v] = expr(pos, types, box);
      c_energy = e;
      (void)f;
      (void)v;
    }
    const auto c1 = clock_type::now();
    const double c_ns = ns_per_call(c1 - c0, n_calls);
    std::printf("C_expr_ns_per_call=%.4f\n", c_ns);
    std::printf("C_expr_energy_eV=%.16g\n", c_energy);
    if (a_ns > 0.0) {
      std::printf("C_over_A=%.4f\n", c_ns / a_ns);
    }
  }
#else
  std::printf("C_expr_ns_per_call=\n");
  std::printf("C_expr_note=not_built (needs -Dwith_expr=true)\n");
#endif

  return 0;
}
