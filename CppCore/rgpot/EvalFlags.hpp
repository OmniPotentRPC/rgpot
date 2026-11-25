#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file EvalFlags.hpp
 * @brief Bits that qualify one successful evaluation.
 *
 * An evaluation that returns success may still have needed help to get
 * there. A potential that can tell records the bits below for each call;
 * zero is a plain evaluation. The values are those of
 * @c RGPOT_ENGINE_EVAL_* in engine_c_abi.h and @c RGPOT_XTB_EVAL_* in
 * xtb_c_abi.h, so the bits pass through either ABI unchanged.
 *
 * @ingroup rgpot_cpp
 */

#include <cstdint>

namespace rgpot::eval_flags {

/// The SCC did not converge at the configured settings. A smeared
/// occupation at a higher electronic temperature converged it, and the
/// answer is a single point at the configured temperature started from
/// that density. Such a state can lie far from the one a plain SCC
/// reaches.
inline constexpr std::uint64_t SccRescued = 1u << 0;

/// The first SCC attempt did not converge and a second attempt at the
/// configured settings, from the engine's own guess, did.
inline constexpr std::uint64_t SccRetried = 1u << 1;

/// The answer came from a persistent result cache, so this call evaluated
/// nothing and the bits of the evaluation that stored it are unknown.
inline constexpr std::uint64_t Cached = 1u << 2;

/// Every bit defined so far.
inline constexpr std::uint64_t Known = SccRescued | SccRetried | Cached;

} // namespace rgpot::eval_flags
