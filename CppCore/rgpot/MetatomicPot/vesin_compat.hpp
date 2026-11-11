#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Source-level compatibility for vesin neighbor options across:
//   0.3.x  — enum VesinDevice, no Options.algorithm
//   0.5.x  — struct VesinDevice {type, device_id}, Options.algorithm
//   0.6.x  — Options.skin + Options.n_threads inserted before return_* flags
//
// Detection is pure C++ type traits — no parent-build macros required.
//
// IMPORTANT: VesinOptions is passed by value into vesin_neighbors. The
// compile-time header layout MUST match the runtime libvesin.so. Building
// against 0.5 and loading 0.6 (or the reverse) shifts return_shifts /
// return_vectors into the skin/n_threads slots and fails with a bare
// "vesin_neighbors failed" (no error_message). Pin build and runtime vesin
// to the same major line (see pyproject / pixi.toml).

#include <cstdint>
#include <type_traits>
#include <utility>

#include "vesin.h"

namespace rgpot {
namespace vesin_compat {

// True when VesinDevice is the 0.5+ struct with type + device_id members.
template <typename Device, typename = void>
struct is_device_struct : std::false_type {};

template <typename Device>
struct is_device_struct<
    Device, std::void_t<decltype(std::declval<Device>().type),
                        decltype(std::declval<Device>().device_id)>>
    : std::true_type {};

// CPU device handle for the installed vesin headers.
// enum VesinDevice  -> VesinCPU
// struct VesinDevice -> {VesinCPU, 0}
template <typename Device = VesinDevice>
constexpr Device make_cpu_device() {
  if constexpr (is_device_struct<Device>::value) {
    return Device{VesinCPU, /*device_id=*/0};
  } else {
    return static_cast<Device>(VesinCPU);
  }
}

// True when VesinOptions has an algorithm member (vesin 0.5+).
template <typename Options, typename = void>
struct has_algorithm_member : std::false_type {};

template <typename Options>
struct has_algorithm_member<
    Options, std::void_t<decltype(std::declval<Options>().algorithm)>>
    : std::true_type {};

// vesin 0.6+: Verlet skin (0 disables caching).
template <typename Options, typename = void>
struct has_skin_member : std::false_type {};

template <typename Options>
struct has_skin_member<Options,
                       std::void_t<decltype(std::declval<Options>().skin)>>
    : std::true_type {};

// vesin 0.6+: explicit thread count (0 = OMP_NUM_THREADS / hardware).
template <typename Options, typename = void>
struct has_n_threads_member : std::false_type {};

template <typename Options>
struct has_n_threads_member<
    Options, std::void_t<decltype(std::declval<Options>().n_threads)>>
    : std::true_type {};

// When algorithm exists, assign its zero/default value without naming
// VesinAutoAlgorithm (absent from older headers). No-op otherwise.
template <typename Options>
void set_algorithm_default(Options &options) noexcept {
  if constexpr (has_algorithm_member<Options>::value) {
    options.algorithm = {};
  }
}

template <typename Options>
void set_skin(Options &options, double skin) noexcept {
  if constexpr (has_skin_member<Options>::value) {
    options.skin = skin;
  }
}

template <typename Options>
void set_n_threads(Options &options, std::int32_t n_threads) noexcept {
  if constexpr (has_n_threads_member<Options>::value) {
    options.n_threads = n_threads;
  }
}

// Fill shared VesinOptions fields for a neighbor request. Zero-init the
// struct first so unknown/newer members stay at safe defaults.
template <typename Options>
void fill_neighbor_options(Options &options, double cutoff, bool full_list) noexcept {
  options = Options{};
  options.cutoff = cutoff;
  options.full = full_list;
  options.sorted = false;
  set_algorithm_default(options);
  // 0.6+: no Verlet cache; auto thread count.
  set_skin(options, 0.0);
  set_n_threads(options, 0);
  options.return_shifts = true;
  options.return_distances = false;
  options.return_vectors = true;
}

} // namespace vesin_compat
} // namespace rgpot
