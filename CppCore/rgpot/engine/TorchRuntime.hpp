#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file TorchRuntime.hpp
 * @brief Thread and determinism settings of the torch runtime an engine links.
 *
 * An engine plugin is opened with dlopen and links its own libtorch, so a host
 * that does not link torch sets these through the engine's parameters.
 */

#include <ATen/Context.h>
#include <ATen/Parallel.h>

#include <stdexcept>
#include <string>

namespace rgpot::engine {

/// A value below 1 leaves that setting alone. The inter-op count can be set
/// once per process before parallel work starts; asking for the value already
/// in force is accepted.
inline void set_torch_threads(int intra_op, int inter_op) {
  if (intra_op >= 1) {
    at::set_num_threads(intra_op);
  }
  if (inter_op >= 1 && at::get_num_interop_threads() != inter_op) {
    try {
      at::set_num_interop_threads(inter_op);
    } catch (const std::exception &e) {
      throw std::runtime_error(
          "cannot set " + std::to_string(inter_op) +
          " inter-op threads (the runtime has " +
          std::to_string(at::get_num_interop_threads()) + "): " + e.what());
    }
  }
}

/// Deterministic algorithms: an operation without a deterministic
/// implementation throws instead of running.
inline void set_torch_deterministic(bool on) {
  if (on) {
    at::globalContext().setDeterministicAlgorithms(true, /*warn_only=*/false);
  }
}

} // namespace rgpot::engine
