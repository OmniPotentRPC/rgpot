#pragma once
// MIT License — UMA / OMol via vesin neighbors + AOTInductor .pt2

#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/UmaPot/UmaConfig.hpp"

#include <cstdint>
#include <memory>

namespace rgpot {

/**
 * @brief UMA / OMol calculator: vesin neighbor list + AOTI package.
 *
 * Loads an AOTInductor ``.pt2`` exported by ``scripts/export_uma_aoti.py``.
 * Each force call builds a fairchem-convention edge list with vesin and
 * runs the compiled graph. Charge and spin are per-call tensor inputs.
 */
class UmaPot : public Potential<UmaPot> {
public:
  explicit UmaPot(const UmaConfig &config);
  ~UmaPot() override;

  UmaPot(const UmaPot &) = delete;
  UmaPot &operator=(const UmaPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// One AOTI call per chunk of up to the package's batch_max
  /// same-composition systems (a NEB band). Band packages carry only
  /// the batched graph, so a lone system is padded to two.
  void forceBatchImpl(const ForceBatch &batch) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::SharedInstance,
            .perImageInstances = true};
  }

  [[nodiscard]] const UmaConfig &config() const noexcept { return m_config; }

  void setChargeSpin(int charge, int spin);

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

private:
  static constexpr uint64_t kKernelVersion = 2;

  struct Impl;

  UmaConfig m_config;
  uint64_t m_paramsKey{0};
  std::unique_ptr<Impl> m_impl;

  void recomputeParamsKey();
  void ensureLoaded() const;
};

} // namespace rgpot
