// MIT License
// Copyright 2023--present rgpot developers
//
// A non-empty XTBDlopenConfig::engine_path is the only candidate, and a
// slash-less name is rejected. A missing file must throw even when the
// process loader would resolve the bare soname (LD_LIBRARY_PATH /
// DYLD_LIBRARY_PATH / PATH, captured at process start) and even when
// RGPOT_XTB_ENGINE names a real engine. An empty engine_path still
// discovers via those routes.
//
// setenv("LD_LIBRARY_PATH") after startup does not affect glibc dlopen, so
// the meson test environment and the binary's build rpath publish the engine
// directory before this process starts.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgpot/XTBPot/XTBDlopen.hpp"

#ifndef _WIN32
#include <dlfcn.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

constexpr const char *kMissingEngine =
    "/nonexistent/rgpot-test/libxtb_engine.so";

constexpr const char *kDiscoveryVars[] = {
    "RGPOT_XTB_ENGINE",
    "XTB_ENGINE",
    "EON_POTENTIALS_PATH",
    "RGPOT_ENGINE_PATH",
};

void set_env(const char *name, const char *value) {
#ifdef _WIN32
  _putenv_s(name, value != nullptr ? value : "");
#else
  if (value != nullptr)
    setenv(name, value, 1);
  else
    unsetenv(name);
#endif
}

void unset_env(const char *name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

class DiscoveryEnvRestore {
public:
  DiscoveryEnvRestore() {
    saved_.reserve(std::size(kDiscoveryVars));
    for (const char *name : kDiscoveryVars) {
      const char *cur = std::getenv(name);
      saved_.push_back({name, cur != nullptr, cur != nullptr ? cur : ""});
    }
  }

  ~DiscoveryEnvRestore() {
    for (const auto &slot : saved_) {
      if (slot.present)
        set_env(slot.name, slot.value.c_str());
      else
        unset_env(slot.name);
    }
  }

  DiscoveryEnvRestore(const DiscoveryEnvRestore &) = delete;
  DiscoveryEnvRestore &operator=(const DiscoveryEnvRestore &) = delete;

private:
  struct Slot {
    const char *name;
    bool present;
    std::string value;
  };
  std::vector<Slot> saved_;
};

void clear_discovery_env() {
  for (const char *name : kDiscoveryVars)
    unset_env(name);
}

void *open_bare(const char *name) {
#ifdef _WIN32
  return reinterpret_cast<void *>(LoadLibraryA(name));
#else
  // NODELETE: probing must not unload libxtb. A later XTBDlopen of the same
  // soname shares this mapping; unloading Fortran runtimes mid-process is
  // not reliable.
  return dlopen(name, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
#endif
}

void close_bare(void *handle) {
  if (handle == nullptr)
    return;
#ifdef _WIN32
  // Pin the module for the rest of the process. FreeLibrary would unload
  // libxtb between cases; the Unix probe uses RTLD_NODELETE instead.
  (void)handle;
#else
  dlclose(handle);
#endif
}

// True when this process can already dlopen the platform bare soname.
// That is the EESSI module-load setup: LD_LIBRARY_PATH names a directory
// that contains libxtb_engine.so.
bool bare_soname_visible() {
#ifdef _WIN32
  const char *names[] = {"xtb_engine.dll", "libxtb_engine.dll"};
#elif defined(__APPLE__)
  const char *names[] = {"libxtb_engine.dylib", "libxtb_engine.so"};
#else
  const char *names[] = {"libxtb_engine.so"};
#endif
  for (const char *name : names) {
    void *handle = open_bare(name);
    if (handle != nullptr) {
      close_bare(handle);
      return true;
    }
  }
  return false;
}

void require_missing_engine(const rgpot::XTBDlopenConfig &cfg) {
  try {
    rgpot::XTBDlopen pot(cfg);
    (void)pot;
    FAIL("XTBDlopen opened an engine that should be missing");
  } catch (const std::runtime_error &ex) {
    const std::string msg = ex.what();
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring(
                          "XTBDlopen: libxtb_engine.so not found"));
    if (!cfg.engine_path.empty()) {
      REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring(cfg.engine_path));
      REQUIRE_THAT(msg,
                   !Catch::Matchers::ContainsSubstring("RGPOT_XTB_ENGINE"));
      REQUIRE_THAT(msg,
                   !Catch::Matchers::ContainsSubstring("EON_POTENTIALS_PATH"));
#ifndef _WIN32
      REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("dlerror:"));
#endif
    }
  }
}

bool expect_bare_soname() {
  const char *flag = std::getenv("RGPOT_XTB_EXPECT_BARE_SONAME");
  return flag != nullptr && *flag != '\0' && std::string(flag) != "0";
}

} // namespace

TEST_CASE("XTBDlopen explicit engine_path does not fall back",
          "[xtb][dlopen]") {
  rgpot::XTBDlopenConfig cfg;
  cfg.engine_path = kMissingEngine;

  // Env candidates must not rescue a failed explicit path.
  require_missing_engine(cfg);

  // Meson publishes the engine directory on the loader path and sets this
  // flag. A hand run without it still checks the throw, but cannot prove the
  // bare-soname fallback is rejected.
  if (expect_bare_soname())
    REQUIRE(bare_soname_visible());
  DiscoveryEnvRestore restore;
  clear_discovery_env();
  // Directory lists and the bare soname must not rescue it either.
  require_missing_engine(cfg);
}

TEST_CASE("XTBDlopen slash-less engine_path is rejected", "[xtb][dlopen]") {
  // The meson environment puts the real engine on the loader path. A bare
  // name must still throw, otherwise dlopen substitutes that library.
  if (expect_bare_soname())
    REQUIRE(bare_soname_visible());

  rgpot::XTBDlopenConfig cfg;
  cfg.engine_path = "libxtb_engine.so";
  try {
    rgpot::XTBDlopen pot(cfg);
    (void)pot;
    FAIL("slash-less engine_path must not be opened through the loader path");
  } catch (const std::runtime_error &ex) {
    const std::string msg = ex.what();
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("libxtb_engine.so"));
    REQUIRE_THAT(msg,
                 Catch::Matchers::ContainsSubstring("directory separator"));
    REQUIRE_THAT(msg, !Catch::Matchers::ContainsSubstring("RGPOT_XTB_ENGINE"));
    REQUIRE_THAT(msg,
                 !Catch::Matchers::ContainsSubstring("EON_POTENTIALS_PATH"));
  }
}

TEST_CASE("XTBDlopen empty engine_path discovers RGPOT_XTB_ENGINE",
          "[xtb][dlopen]") {
  const char *eng = std::getenv("RGPOT_XTB_ENGINE");
  if (eng == nullptr || *eng == '\0') {
    WARN("RGPOT_XTB_ENGINE unset; empty-path env discovery not exercised");
    return;
  }
  rgpot::XTBDlopenConfig cfg;
  rgpot::XTBDlopen pot(cfg);
  REQUIRE(pot.available());
}

TEST_CASE("XTBDlopen empty engine_path discovers the bare soname",
          "[xtb][dlopen]") {
  if (!bare_soname_visible()) {
    if (expect_bare_soname()) {
      FAIL("bare libxtb_engine is not on the process loader path");
    }
    WARN("bare libxtb_engine is not on the process loader path");
    return;
  }
  DiscoveryEnvRestore restore;
  clear_discovery_env();
  rgpot::XTBDlopenConfig cfg;
  rgpot::XTBDlopen pot(cfg);
  REQUIRE(pot.available());
}
