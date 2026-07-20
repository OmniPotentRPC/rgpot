#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Portable dynamic library loader (dlopen / LoadLibrary family).
 *
 * Used by NWChemPot to resolve libnwchemc at runtime without a
 * build-time link dependency on NWChem.
 */

#include <stdexcept>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace rgpot {

class DynLib {
public:
  DynLib() = default;

  explicit DynLib(const std::string &path) { open(path); }

  DynLib(const DynLib &) = delete;
  DynLib &operator=(const DynLib &) = delete;

  DynLib(DynLib &&other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  DynLib &operator=(DynLib &&other) noexcept {
    if (this != &other) {
      close();
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  ~DynLib() { close(); }

  bool valid() const { return handle_ != nullptr; }

  void open(const std::string &path) {
    close();
#if defined(_WIN32) || defined(_WIN64)
    handle_ = LoadLibraryA(path.c_str());
    if (!handle_) {
      throw std::runtime_error("LoadLibrary failed: " + path);
    }
#else
    // Prefer LOCAL so the engine's BLAS/LAPACK (often linked into
    // libnwchemc) does not become the process-global dpotf2/dgemm used by
    // ScaLAPACK / OpenBLAS after first load. DEEPBIND keeps the engine's
    // own unresolved symbols inside its DSO first. GLOBAL alone made
    // pdpotrf in ScaLAPACK lazy-bind into NWChem LAPACK and SIGSEGV during
    // GP hyperparameter Cholesky. Fall back without DEEPBIND when the
    // platform lacks it (non-glibc).
    int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    flags |= RTLD_DEEPBIND;
#endif
    handle_ = dlopen(path.c_str(), flags);
    if (!handle_) {
      const char *err = dlerror();
      throw std::runtime_error(std::string("dlopen failed: ") + path + ": " +
                               (err ? err : "unknown"));
    }
#endif
  }

  void close() {
    if (!handle_)
      return;
#if defined(_WIN32) || defined(_WIN64)
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
    handle_ = nullptr;
  }

  template <typename T> T sym(const char *name) const {
    void *p = sym_raw(name);
    if (!p) {
      throw std::runtime_error(std::string("symbol not found: ") + name);
    }
    return reinterpret_cast<T>(p);
  }

  template <typename T> T sym_optional(const char *name) const {
    void *p = sym_raw(name);
    return p ? reinterpret_cast<T>(p) : nullptr;
  }

private:
  void *sym_raw(const char *name) const {
    if (!handle_)
      return nullptr;
#if defined(_WIN32) || defined(_WIN64)
    return reinterpret_cast<void *>(
        GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    dlerror(); // clear
    return dlsym(handle_, name);
#endif
  }

  void *handle_ = nullptr;
};

} // namespace rgpot
