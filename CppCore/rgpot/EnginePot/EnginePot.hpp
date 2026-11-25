#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief A potential served by an engine library over the generic engine C
 * ABI of `rgpot/engine_c_abi.h`.
 *
 * The library is the first candidate path that opens and exports
 * `rgpot_engine_create`, `rgpot_engine_destroy` and `rgpot_engine_force`.
 * One engine instance is created from the configuration message the caller
 * passes (for the in-tree engines a word-aligned Cap'n Proto flat array
 * whose root is the backend's params struct) and destroyed with the
 * potential. The library stays mapped for the life of the process, as the
 * engine loaders of rgpot-core keep theirs: a backend may own threads or
 * runtimes that outlive its last instance.
 *
 * Calls on one instance are serialised; engines that allow concurrent calls
 * on a handle are still called one at a time through this class.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rgpot/Potential.hpp"

namespace rgpot {

class EnginePot : public Potential<EnginePot> {
public:
  /// Opens the first of @p candidates that loads and exports the engine
  /// symbols and creates an instance from @p config (@p config_len bytes).
  /// @p label names the backend in error messages ("orcac", "plugin uma").
  /// Throws std::runtime_error when no candidate opens, the engine reports
  /// itself unavailable or another ABI major, or create fails; the message
  /// names every path tried.
  EnginePot(const std::vector<std::string> &candidates, const void *config,
            std::size_t config_len, std::string label);
  ~EnginePot() override;

  EnginePot(const EnginePot &) = delete;
  EnginePot &operator=(const EnginePot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override;

  /// The library path and the configuration bytes.
  [[nodiscard]] uint64_t paramsKey() const noexcept override;

  /// The candidate the library was opened from.
  [[nodiscard]] const std::string &libraryPath() const noexcept;

  /// The label given at construction.
  [[nodiscard]] const std::string &label() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rgpot
