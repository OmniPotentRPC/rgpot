#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/Potential.hpp"
#include "rgpot/engine/EngineError.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace rgpot {

enum class DxtbMethod { GFN1, GFN2 };

struct DxtbConfig {
  DxtbMethod method = DxtbMethod::GFN2;
  double accuracy = 1.0;
  double electronic_temperature = 300.0;
  int max_iterations = 250;
  double charge = 0.0;
  int uhf = 0;
  std::string python;
};

class DxtbError : public engine::BackendError {
public:
  DxtbError(int code, const std::string &message)
      : engine::BackendError(code, message) {}
};

class DxtbPot final : public Potential<DxtbPot> {
public:
  explicit DxtbPot(const DxtbConfig &config);
  ~DxtbPot() override;

  DxtbPot(const DxtbPot &) = delete;
  DxtbPot &operator=(const DxtbPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  void hessianVectorImpl(const ForceInput &in, const double *direction,
                         double *product) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .periodic = false,
            .analyticHvp = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return params_key_;
  }

  void setChargeSpin(int charge, int multiplicity);
  void setNumThreads(int intra_op, int inter_op);

  /// Whether the Python a worker without DxtbConfig::python would run imports
  /// Torch, dxtb 0.4.0 and a supported tad-libcint. Cached per executable.
  static bool runtimeAvailable();

private:
  static constexpr uint64_t kKernelVersion = 1;

  class Worker;
  DxtbConfig config_;
  uint64_t params_key_;
  std::unique_ptr<Worker> worker_;
};

} // namespace rgpot
