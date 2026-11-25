#pragma once
// MIT License — thin Metatomic frontend: dlopen libmetatomic_engine.so
#include "rgpot/ForceStructs.hpp"
#include "rgpot/abi/EngineThreadCount.hpp"
#include "rgpot/MetatomicPot/MetatomicConfig.hpp"
#include "rgpot/MetatomicPot/metatomic_c_abi.h"

#include <string>

namespace rgpot {

/**
 * Slow path: no torch headers; loads libmetatomic_engine.so at runtime.
 * Search: config.engine_path, RGPOT_METATOMIC_ENGINE, METATOMIC_ENGINE,
 * bare libmetatomic_engine.so on LD_LIBRARY_PATH / EON_POTENTIALS_PATH.
 */
class MetatomicDlopen {
public:
  explicit MetatomicDlopen(const MetatomicConfig &config);
  ~MetatomicDlopen();

  MetatomicDlopen(const MetatomicDlopen &) = delete;
  MetatomicDlopen &operator=(const MetatomicDlopen &) = delete;

  [[nodiscard]] bool available() const noexcept { return m_pot != nullptr; }

  void forceImpl(const ForceInput &in, ForceOut *out) const;

  /// Records a thread count the way @c PotentialBase::setThreadCount does
  /// and throws std::invalid_argument when @p threads is below 1. The
  /// engine ABI has no thread setter, so the count is not passed on and
  /// evaluation does not change: the engine keeps the torch and OpenMP
  /// threads of its own runtime. See @c abi/EngineThreadCount.hpp.
  void setThreadCount(int threads) { m_threads.set(threads); }
  void clearThreadCount() noexcept { m_threads.clear(); }
  /// Zero when none was recorded.
  [[nodiscard]] int threadCount() const noexcept { return m_threads.get(); }

private:
  abi::EngineThreadCount m_threads;
  void *m_lib{nullptr};
  RgpotMtaPot *m_pot{nullptr};
  using create_fn = RgpotMtaPot *(*)(const RgpotMtaConfig *, char *, size_t);
  using destroy_fn = void (*)(RgpotMtaPot *);
  using force_fn = int (*)(RgpotMtaPot *, long, const double *, const int *,
                           double *, double *, double *, const double *);
  create_fn m_create{nullptr};
  destroy_fn m_destroy{nullptr};
  force_fn m_force{nullptr};
};

} // namespace rgpot
