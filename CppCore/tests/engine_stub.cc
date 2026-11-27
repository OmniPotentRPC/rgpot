// MIT License
// Copyright 2023--present rgpot developers
//
// A stand-in backend for the generic engine table: E = 0.5 * sum(p^2) and
// F = -p. The first byte of the configuration picks the reentrancy the
// backend reports (0 shared, 1 per instance, 2 process serial), the second
// byte clears the optional charge/spin, thread, orbital-selection and
// orbital-copy hooks, and the third byte names the model file in
// RGPOT_ENGINE_STUB_MODEL. Byte 3 is zero when the analytic Hessian-vector
// product is off. Byte 4 is nonzero when byte 5 is an explicit spin: 0 is
// a real value there, and it is not the same as leaving byte 4 clear. A
// nonzero byte 6 selects an anharmonic backend instead, with bit 0 for its
// dense Hessian and bit 1 for its Hessian-vector product; 4 gives neither.
// The library exports how many calls were inside the backend at once and how
// often each curvature hook ran.

#include "rgpot/GradientJet.hpp"
#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/engine_c_abi.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace {
std::atomic<int> g_inside{0};
std::atomic<int> g_max{0};
std::atomic<int> g_charge{0};
std::atomic<int> g_spin{0};
std::atomic<long long> g_orbital_key{0};
std::atomic<long long> g_hessian_calls{0};
std::atomic<long long> g_hvp_calls{0};

/// E = sum_k w_k p_k^2 / 2 + lambda s^2 / 4 with s = sum_k p_k^2 and
/// w_k = 1 + k / 4, so the Hessian
/// H_kl = delta_kl (w_k + lambda s) + 2 lambda p_k p_l
/// couples every coordinate and changes with the geometry.
class CurvedPot final : public rgpot::PotentialBase {
public:
  static constexpr double kLambda = 0.3;

  CurvedPot(bool dense, bool hvp)
      : PotentialBase(rgpot::PotType::UNKNOWN), dense_(dense), hvp_(hvp) {}

  [[nodiscard]] rgpot::PotCaps caps() const noexcept override {
    return {.reentrancy = rgpot::Reentrancy::PerInstance,
            .analyticHvp = hvp_,
            .analyticHessian = dense_};
  }

  std::tuple<double, rgpot::types::AtomMatrix, double>
  operator()(const rgpot::types::AtomMatrix &positions,
             const std::vector<int> &,
             const std::array<std::array<double, 3>, 3> &) override {
    return {0.0, rgpot::types::AtomMatrix::Zero(positions.rows(), 3), 0.0};
  }

  void forceBatch(const rgpot::ForceBatch &batch) override {
    for (size_t s = 0; s < batch.nSystems; ++s) {
      const rgpot::ForceInput &in = batch.in[s];
      const size_t dim = 3 * in.nAtoms;
      const double sq = squares(in.pos, dim);
      double e = 0.25 * kLambda * sq * sq;
      for (size_t k = 0; k < dim; ++k) {
        e += 0.5 * weight(k) * in.pos[k] * in.pos[k];
        batch.out[s].F[k] = -(weight(k) + kLambda * sq) * in.pos[k];
      }
      batch.out[s].energy = e;
      batch.out[s].variance = 0.0;
    }
  }

  void hessianImpl(const rgpot::ForceInput &in, double *h) const override {
    ++g_hessian_calls;
    const size_t dim = 3 * in.nAtoms;
    const double sq = squares(in.pos, dim);
    for (size_t k = 0; k < dim; ++k) {
      for (size_t l = 0; l < dim; ++l) {
        h[k * dim + l] = 2.0 * kLambda * in.pos[k] * in.pos[l];
      }
      h[k * dim + k] += weight(k) + kLambda * sq;
    }
  }

  void hessianVectorImpl(const rgpot::ForceInput &in, const double *v,
                         double *hv) const override {
    ++g_hvp_calls;
    const size_t dim = 3 * in.nAtoms;
    const double sq = squares(in.pos, dim);
    double pv = 0.0;
    for (size_t k = 0; k < dim; ++k) {
      pv += in.pos[k] * v[k];
    }
    for (size_t k = 0; k < dim; ++k) {
      hv[k] = (weight(k) + kLambda * sq) * v[k] + 2.0 * kLambda * in.pos[k] * pv;
    }
  }

private:
  static double weight(size_t k) { return 1.0 + 0.25 * static_cast<double>(k); }
  static double squares(const double *p, size_t n) {
    double s = 0.0;
    for (size_t k = 0; k < n; ++k) {
      s += p[k] * p[k];
    }
    return s;
  }

  bool dense_;
  bool hvp_;
};

class StubPot : public rgpot::PotentialBase {
public:
  StubPot(rgpot::Reentrancy r, bool analytic_hvp)
      : PotentialBase(rgpot::PotType::UNKNOWN), reentrancy_(r),
        analytic_hvp_(analytic_hvp) {}

  [[nodiscard]] rgpot::PotCaps caps() const noexcept override {
    return {.reentrancy = reentrancy_,
            .perImageInstances = true,
            .analyticHvp = analytic_hvp_};
  }

  std::tuple<double, rgpot::types::AtomMatrix, double>
  operator()(const rgpot::types::AtomMatrix &positions,
             const std::vector<int> &,
             const std::array<std::array<double, 3>, 3> &) override {
    return {0.0, rgpot::types::AtomMatrix::Zero(positions.rows(), 3), 0.0};
  }

  void forceBatch(const rgpot::ForceBatch &batch) override {
    const int now = ++g_inside;
    int seen = g_max.load();
    while (now > seen && !g_max.compare_exchange_weak(seen, now)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    for (size_t s = 0; s < batch.nSystems; ++s) {
      const rgpot::ForceInput &in = batch.in[s];
      double e = 0.0;
      for (size_t k = 0; k < 3 * in.nAtoms; ++k) {
        e += 0.5 * in.pos[k] * in.pos[k];
        batch.out[s].F[k] = -in.pos[k];
      }
      batch.out[s].energy = e;
      batch.out[s].variance = 0.25;
    }
    --g_inside;
  }

  void hessianVectorImpl(const rgpot::ForceInput &in, const double *direction,
                         double *product) const override {
    std::copy_n(direction, 3 * in.nAtoms, product);
  }

private:
  rgpot::Reentrancy reentrancy_;
  bool analytic_hvp_;
};

/// Writes the whole jet and then throws, so a table that forwards the
/// caller's buffer would leave that buffer partly updated.
class PartialJet final : public rgpot::PotentialBase, public rgpot::GradientJet {
public:
  PartialJet() : PotentialBase(rgpot::PotType::UNKNOWN) {}

  [[nodiscard]] rgpot::PotCaps caps() const noexcept override {
    return {.reentrancy = rgpot::Reentrancy::PerInstance};
  }

  std::tuple<double, rgpot::types::AtomMatrix, double>
  operator()(const rgpot::types::AtomMatrix &positions,
             const std::vector<int> &,
             const std::array<std::array<double, 3>, 3> &) override {
    return {0.0, rgpot::types::AtomMatrix::Zero(positions.rows(), 3), 0.0};
  }

  [[nodiscard]] int maxJetOrder() const noexcept override { return 3; }

  rgpot::DerivativeReport
  gradientJet(std::size_t systems, std::size_t nAtoms, const double *,
              const int *, const double *, const double *, int order,
              double *jet) const override {
    const std::size_t n =
        systems * nAtoms * 3 * static_cast<std::size_t>(order);
    for (std::size_t i = 0; i < n; ++i) {
      jet[i] = 42.0;
    }
    throw std::runtime_error("jet failed after writing the buffer");
  }
};
} // namespace

namespace rgpot::engine {

const char *backend_name() { return "stub"; }

Created create(const void *config, std::size_t config_len) {
  const auto *bytes = static_cast<const unsigned char *>(config);
  // An explicit spin is part of the configuration, visible before any
  // rgpot_engine_set_charge_spin. A clear flag leaves the recorded spin
  // alone, which is 0 after stub_reset and is not a configured value.
  if (bytes != nullptr && config_len >= 6 && bytes[4] != 0) {
    g_spin = bytes[5];
  }
  Created out;
  auto backend = std::make_unique<Backend>();
  if (config_len >= 7 && bytes[6] != 0) {
    backend->pot = std::make_unique<CurvedPot>((bytes[6] & 1) != 0,
                                               (bytes[6] & 2) != 0);
  } else if (bytes[4] != 0) {
    backend->pot = std::make_unique<PartialJet>();
  } else {
    backend->pot = std::make_unique<StubPot>(static_cast<Reentrancy>(bytes[0]),
                                             bytes[3] != 0);
  }
  if (bytes[1] == 0) {
    backend->set_charge_spin = [](int charge, int spin) {
      if (charge == 99) {
        throw rgpot::engine::Unsupported("charge 99 refused");
      }
      if (spin == 101) {
        throw rgpot::engine::Unsupported("spin 101 refused");
      }
      g_charge = charge;
      g_spin = spin;
    };
    backend->set_num_threads = [](int intra_op, int inter_op) {
      if (inter_op == 99) {
        throw std::runtime_error("inter-op threads refused");
      }
      (void)intra_op;
    };
    backend->select_orbitals = [](long long key) {
      if (key == -1)
        throw rgpot::engine::Unsupported("orbital key refused");
      if (key == -2)
        throw std::runtime_error("orbital selection failed");
      g_orbital_key = key;
    };
    backend->copy_orbitals = [](long long from_key, long long to_key) {
      if (from_key == -1)
        throw rgpot::engine::Unsupported("orbital copy refused");
      if (to_key == -2)
        throw std::runtime_error("orbital copy failed");
    };
  }
  if (bytes[2] != 0) {
    if (const char *path = std::getenv("RGPOT_ENGINE_STUB_MODEL")) {
      backend->model_path = path;
    }
  }
  out.backend = std::move(backend);
  return out;
}

} // namespace rgpot::engine

extern "C" {
RGPOT_ENGINE_API int stub_max_concurrent(void) { return g_max.load(); }
RGPOT_ENGINE_API void stub_reset(void) {
  g_max = 0;
  g_inside = 0;
  g_charge = 0;
  g_spin = 0;
  g_orbital_key = 0;
  g_hessian_calls = 0;
  g_hvp_calls = 0;
}
RGPOT_ENGINE_API long long stub_hessian_calls(void) {
  return g_hessian_calls.load();
}
RGPOT_ENGINE_API long long stub_hvp_calls(void) { return g_hvp_calls.load(); }
RGPOT_ENGINE_API int stub_charge(void) { return g_charge.load(); }
RGPOT_ENGINE_API int stub_spin(void) { return g_spin.load(); }
RGPOT_ENGINE_API long long stub_orbital_key(void) { return g_orbital_key.load(); }
}
