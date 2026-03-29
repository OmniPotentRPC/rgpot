#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Cross-platform dynamic library loading abstraction.
// Ported from eOn (TheochemUI/eOn, BSD-3-Clause).

#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace rgpot::plugin::dynlib {

#ifdef _WIN32
using Handle = HMODULE;

inline Handle open(const char *name) noexcept { return LoadLibraryA(name); }

inline void *sym(Handle h, const char *name) noexcept {
  return reinterpret_cast<void *>(GetProcAddress(h, name));
}

inline void close(Handle h) noexcept {
  if (h)
    FreeLibrary(h);
}

inline std::string error() {
  DWORD err = GetLastError();
  if (err == 0)
    return {};
  LPSTR buf = nullptr;
  FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                     FORMAT_MESSAGE_IGNORE_INSERTS,
                 nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                 reinterpret_cast<LPSTR>(&buf), 0, nullptr);
  std::string msg(buf ? buf : "unknown error");
  if (buf)
    LocalFree(buf);
  return msg;
}

inline const char *so_extension() { return ".dll"; }

#else // POSIX
using Handle = void *;

inline Handle open(const char *name) noexcept {
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}

inline void *sym(Handle h, const char *name) noexcept { return dlsym(h, name); }

inline void close(Handle h) noexcept {
  if (h)
    dlclose(h);
}

inline std::string error() {
  const char *msg = dlerror();
  return msg ? std::string(msg) : std::string{};
}

#ifdef __APPLE__
inline const char *so_extension() { return ".dylib"; }
#else
inline const char *so_extension() { return ".so"; }
#endif

#endif

template <typename Fn> Fn loadSym(Handle h, const char *name) noexcept {
  return reinterpret_cast<Fn>(sym(h, name));
}

} // namespace rgpot::plugin::dynlib
