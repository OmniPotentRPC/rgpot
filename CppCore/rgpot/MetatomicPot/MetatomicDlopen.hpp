#pragma once
// MIT License — thin Metatomic frontend: dlopen libmetatomic_engine.so
#include "rgpot/ForceStructs.hpp"
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

  /// Thread count the engine handle runs with, forwarded on every call.
  /// Throws std::invalid_argument below 1. A @c rgpot::ThreadCount on the
  /// calling thread wins over it.
  void setThreadCount(int threads);
  void clearThreadCount();
  /// Zero when no count is set.
  [[nodiscard]] int threadCount() const noexcept;

  void forceImpl(const ForceInput &in, ForceOut *out) const;

private:
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
