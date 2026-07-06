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
    // RTLD_GLOBAL: export NWChem/engine symbols for dependent .so resolution.
    // RTLD_NODELETE: libnwchemc registers atexit(nwchemc_finalize). If DynLib
    // unmaps the DSO before process atexit handlers run (typical when
    // NWChemPot is destroyed at end of main), the atexit trampoline points
    // into freed text and SIGSEGV on Quill/libc teardown threads.
#ifndef RTLD_NODELETE
#define RTLD_NODELETE 0
#endif
    handle_ =
        dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
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
    // With RTLD_NODELETE the mapping stays for atexit; still drop our handle.
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
