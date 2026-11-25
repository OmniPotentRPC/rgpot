// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/EAMTable/EAMTablePot.hpp"

#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

enum : int32_t { kSuccess = 0, kInvalidParameter = 1, kInternalError = 2 };
enum : int32_t { kAuto = 0, kFuncfl = 1, kSetfl = 2, kFinnisSinclair = 3 };

thread_local std::string g_last_error;
std::int64_t g_drop_count = 0;
std::mutex g_drop_count_mutex;

void set_error(const std::string &text) { g_last_error = text; }

rgpot::EAMTableFormat parse_format(int32_t format, const std::string &path) {
  if (format == kFuncfl) {
    return rgpot::EAMTableFormat::Funcfl;
  }
  if (format == kSetfl) {
    return rgpot::EAMTableFormat::Setfl;
  }
  if (format == kFinnisSinclair) {
    return rgpot::EAMTableFormat::FinnisSinclair;
  }
  if (format != kAuto) {
    throw std::invalid_argument("EAM table: unknown format code");
  }
  if (path.ends_with(".eam")) {
    return rgpot::EAMTableFormat::Funcfl;
  }
  if (path.ends_with(".eam.alloy") || path.ends_with(".setfl")) {
    return rgpot::EAMTableFormat::Setfl;
  }
  if (path.ends_with(".eam.fs")) {
    return rgpot::EAMTableFormat::FinnisSinclair;
  }
  throw std::invalid_argument(
      "EAM table: cannot tell table format from file suffix");
}

int32_t to_status(const std::exception &e) {
  set_error(e.what());
  return kInvalidParameter;
}

} // namespace

extern "C" {

const char *eam_bridge_cpp_last_error() { return g_last_error.c_str(); }

void *eam_bridge_cpp_new_from_file(const char *path, int32_t format,
                                   bool periodic, double skin) {
  try {
    if (path == nullptr) {
      throw std::invalid_argument("EAM table: path is NULL");
    }
    std::ifstream in(path);
    if (!in) {
      throw std::invalid_argument(std::string("EAM table: cannot open ") +
                                  path);
    }
    const auto parsed = parse_format(format, path);
    const auto table = rgpot::read_eam_table(in, parsed);
    return new rgpot::EAMTablePot(table, periodic, false, skin);
  } catch (const std::exception &e) {
    set_error(e.what());
    return nullptr;
  } catch (...) {
    set_error("EAM table: unknown exception while constructing EAMTablePot");
    return nullptr;
  }
}

void *eam_bridge_cpp_new_from_text(const char *text, int32_t format,
                                   bool periodic, double skin) {
  try {
    if (text == nullptr) {
      throw std::invalid_argument("EAM table: text is NULL");
    }
    if (format == kAuto) {
      throw std::invalid_argument(
          "EAM table: AUTO format is invalid for in-memory text");
    }
    std::istringstream in(text);
    const auto table =
        rgpot::read_eam_table(in, parse_format(format, "table.eam.fs"));
    return new rgpot::EAMTablePot(table, periodic, false, skin);
  } catch (const std::exception &e) {
    set_error(e.what());
    return nullptr;
  } catch (...) {
    set_error("EAM table: unknown exception while constructing EAMTablePot");
    return nullptr;
  }
}

void eam_bridge_cpp_free(void *handle) {
  if (handle != nullptr) {
    delete static_cast<rgpot::EAMTablePot *>(handle);
    std::lock_guard<std::mutex> hold(g_drop_count_mutex);
    ++g_drop_count;
  }
}

int32_t eam_bridge_cpp_force(void *handle, std::size_t n_atoms,
                             const double *positions,
                             const int32_t *atomic_numbers, const double *cell,
                             double *forces, double *energy) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument("EAM table: null potential handle");
    }
    if (positions == nullptr || atomic_numbers == nullptr || cell == nullptr ||
        forces == nullptr || energy == nullptr) {
      throw std::invalid_argument("EAM table: null force argument");
    }
    rgpot::ForceInput in{n_atoms, positions,
                         reinterpret_cast<const int *>(atomic_numbers), cell};
    rgpot::ForceOut out{
        forces, 0.0, 0.0, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 0};
    static_cast<rgpot::EAMTablePot *>(handle)->forceImpl(in, &out);
    *energy = out.energy;
    return kSuccess;
  } catch (const std::exception &e) {
    return to_status(e);
  } catch (...) {
    set_error("EAM table: unknown exception while evaluating EAMTablePot");
    return kInternalError;
  }
}

int64_t eam_bridge_cpp_drop_count() {
  std::lock_guard<std::mutex> hold(g_drop_count_mutex);
  return g_drop_count;
}

void eam_bridge_cpp_reset_drop_count() {
  std::lock_guard<std::mutex> hold(g_drop_count_mutex);
  g_drop_count = 0;
}

} // extern "C"
