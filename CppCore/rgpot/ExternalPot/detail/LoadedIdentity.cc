// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ExternalPot/detail/LoadedIdentity.hpp"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#endif

namespace rgpot {

namespace detail {

namespace {

#if defined(_WIN32)
std::string narrow_path(const std::wstring &wide) {
  if (wide.empty()) {
    return {};
  }
  const int size = WideCharToMultiByte(CP_UTF8, 0, wide.data(),
                                       static_cast<int>(wide.size()), nullptr,
                                       0, nullptr, nullptr);
  if (size <= 0) {
    return {};
  }
  std::string out(static_cast<size_t>(size), '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                      out.data(), size, nullptr, nullptr);
  return out;
}

std::string win_error(DWORD code) {
  return "Windows error " + std::to_string(static_cast<unsigned long>(code));
}
#endif

} // namespace

FileIdentity identify_loaded(const void *anchor, const std::string &path) {
  const std::string prefix = path_prefix(path);
#if defined(_WIN32)
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(anchor), &module) == 0 ||
      module == nullptr) {
    throw std::runtime_error(prefix +
                             "the loader did not report the module it "
                             "loaded, so the engine cannot be identified: " +
                             win_error(GetLastError()));
  }
  std::wstring loaded(MAX_PATH, L'\0');
  for (;;) {
    const DWORD n = GetModuleFileNameW(module, loaded.data(),
                                       static_cast<DWORD>(loaded.size()));
    if (n == 0) {
      throw std::runtime_error(prefix +
                               "the loader did not report the file it "
                               "loaded, so the engine cannot be identified: " +
                               win_error(GetLastError()));
    }
    if (n < loaded.size()) {
      loaded.resize(n);
      break;
    }
    loaded.resize(loaded.size() * 2);
  }
  // The final path resolves links and short names, as realpath does on POSIX,
  // so two requests that load one file share an identity.
  const HANDLE file = CreateFileW(
      loaded.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    throw std::runtime_error(prefix + "cannot identify the loaded file " +
                             narrow_path(loaded) + ": " +
                             win_error(GetLastError()));
  }
  std::wstring canonical(MAX_PATH, L'\0');
  DWORD n = 0;
  for (;;) {
    n = GetFinalPathNameByHandleW(file, canonical.data(),
                                  static_cast<DWORD>(canonical.size()),
                                  FILE_NAME_NORMALIZED);
    if (n == 0 || n < canonical.size()) {
      break;
    }
    canonical.resize(n);
  }
  const DWORD failure = n == 0 ? GetLastError() : 0;
  CloseHandle(file);
  if (n == 0) {
    throw std::runtime_error(prefix + "cannot identify the loaded file " +
                             narrow_path(loaded) + ": " + win_error(failure));
  }
  canonical.resize(n);
  FileIdentity id;
  id.canonical = narrow_path(canonical);
  if (id.canonical.empty()) {
    throw std::runtime_error(prefix +
                             "the loaded file name is not valid UTF-16, so "
                             "the engine cannot be identified");
  }
  return id;
#else
  Dl_info info{};
  if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr ||
      info.dli_fname[0] == '\0') {
    throw std::runtime_error(prefix +
                             "the dynamic linker did not report the file "
                             "it loaded, so the engine cannot be identified");
  }
  char canonical[PATH_MAX];
  struct stat st {};
  if (realpath(info.dli_fname, canonical) == nullptr ||
      stat(canonical, &st) != 0) {
    throw std::runtime_error(prefix + "cannot identify the loaded file " +
                             info.dli_fname + ": " + std::strerror(errno));
  }
  FileIdentity id;
  id.canonical = canonical;
  id.dev = st.st_dev;
  id.ino = st.st_ino;
  return id;
#endif
}

} // namespace detail

} // namespace rgpot
