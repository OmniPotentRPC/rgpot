/**
 * @brief Aggregator for standard library and common external headers.
 *
 * This header provides a centralized location for foundational STL  includes
 * and utility headers required by the core library. It also  manages
 * conditional inclusion for external formatting libraries.
 */

#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// clang-format off
#include <cxxabi.h>
// clang-format on

#include <algorithm>
#include <any>
#include <cctype>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>

/**
 * @note The @c NOT_PURE_LIB flag enables integration with the @c fmt
 * library, providing enhanced string formatting and logging capabilities.
 * This is typically disabled for header-only or minimalist builds.
 */
#ifdef NOT_PURE_LIB
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#endif
