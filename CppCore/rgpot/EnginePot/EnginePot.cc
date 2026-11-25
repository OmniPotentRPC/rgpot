// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/EnginePot/EnginePot.hpp"

#include "rgpot/ParamHash.hpp"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/engine_c_abi.h"

#include <stdexcept>
#include <utility>

namespace rgpot {
namespace {

using AbiVersionFn = int (*)();
using AvailableFn = int (*)();
using CreateFn = RgpotEnginePot *(*)(const void *, size_t, char *, size_t);
using DestroyFn = void (*)(RgpotEnginePot *);
using ForceFn = int (*)(RgpotEnginePot *, long, const double *, const int *,
                        double *, double *, double *, const double *,
                        rgpot_engine_coord_transform, void *);
using LastErrorFn = size_t (*)(const RgpotEnginePot *, char *, size_t);

struct Symbols {
  CreateFn create = nullptr;
  DestroyFn destroy = nullptr;
  ForceFn force = nullptr;
  LastErrorFn last_error = nullptr;
  AbiVersionFn abi_version = nullptr;
  AvailableFn available = nullptr;
};

} // namespace

struct EnginePot::Impl {
  std::string label;
  std::string path;
  std::vector<unsigned char> config;
  Symbols sym;
  RgpotEnginePot *pot = nullptr;
  mutable std::mutex mutex;
};

EnginePot::EnginePot(const std::vector<std::string> &candidates,
                     const void *config, std::size_t config_len,
                     std::string label)
    : Potential(PotType::Engine), impl_(std::make_unique<Impl>()) {
  impl_->label = std::move(label);
  if (candidates.empty()) {
    throw std::runtime_error(impl_->label + ": no engine library to open");
  }
  std::string tried;
  DynLib lib;
  for (const std::string &path : candidates) {
    try {
      lib.open(path, /*global=*/false);
    } catch (const std::exception &e) {
      tried += "; " + std::string(e.what());
      continue;
    }
    Symbols s;
    s.create = lib.sym_optional<CreateFn>("rgpot_engine_create");
    s.destroy = lib.sym_optional<DestroyFn>("rgpot_engine_destroy");
    s.force = lib.sym_optional<ForceFn>("rgpot_engine_force");
    s.last_error = lib.sym_optional<LastErrorFn>("rgpot_engine_last_error");
    s.abi_version = lib.sym_optional<AbiVersionFn>("rgpot_engine_abi_version");
    s.available = lib.sym_optional<AvailableFn>("rgpot_engine_available");
    if (s.create == nullptr || s.destroy == nullptr || s.force == nullptr) {
      tried += "; " + path +
               ": missing rgpot_engine_create, rgpot_engine_destroy or "
               "rgpot_engine_force";
      lib.close();
      continue;
    }
    impl_->sym = s;
    impl_->path = path;
    break;
  }
  if (impl_->path.empty()) {
    throw std::runtime_error(impl_->label + ": engine library not loaded" +
                             tried);
  }
  // The mapping stays for the life of the process (see the class comment).
  lib.release();

  const std::string where = impl_->label + " (" + impl_->path + ")";
  if (impl_->sym.abi_version != nullptr &&
      impl_->sym.abi_version() != RGPOT_ENGINE_ABI_VERSION) {
    throw std::runtime_error(where + ": engine ABI " +
                             std::to_string(impl_->sym.abi_version()) +
                             ", this rgpot speaks " +
                             std::to_string(RGPOT_ENGINE_ABI_VERSION));
  }
  if (impl_->sym.available != nullptr && impl_->sym.available() == 0) {
    throw std::runtime_error(where + ": the engine reports its runtime unavailable");
  }
  const auto *bytes = static_cast<const unsigned char *>(config);
  impl_->config.assign(bytes, bytes + (bytes != nullptr ? config_len : 0));
  char err[1024] = {};
  impl_->pot = impl_->sym.create(impl_->config.data(), impl_->config.size(),
                                 err, sizeof(err));
  if (impl_->pot == nullptr) {
    throw std::runtime_error(where + ": rgpot_engine_create failed: " +
                             (err[0] != '\0' ? err : "no reason given"));
  }
}

EnginePot::~EnginePot() {
  if (impl_ && impl_->pot != nullptr) {
    impl_->sym.destroy(impl_->pot);
  }
}

void EnginePot::forceImpl(const ForceInput &in, ForceOut *out) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  double energy = 0.0;
  double variance = 0.0;
  const int rc = impl_->sym.force(impl_->pot, static_cast<long>(in.nAtoms),
                                  in.pos, in.atmnrs, out->F, &energy,
                                  &variance, in.box, nullptr, nullptr);
  if (rc != RGPOT_ENGINE_OK) {
    std::string why;
    if (impl_->sym.last_error != nullptr) {
      char buf[1024] = {};
      impl_->sym.last_error(impl_->pot, buf, sizeof(buf));
      why = buf;
    }
    throw std::runtime_error(impl_->label + ": rgpot_engine_force returned " +
                             std::to_string(rc) +
                             (why.empty() ? std::string() : ": " + why));
  }
  out->energy = energy;
  out->variance = variance;
}

PotCaps EnginePot::caps() const noexcept {
  return {.reentrancy = Reentrancy::PerInstance};
}

uint64_t EnginePot::paramsKey() const noexcept {
  Fnv1a fp;
  fp.str("engine");
  fp.str(impl_->path);
  fp.u64(impl_->config.size());
  fp.bytes(impl_->config.data(), impl_->config.size());
  return fp.h;
}

const std::string &EnginePot::libraryPath() const noexcept {
  return impl_->path;
}

const std::string &EnginePot::label() const noexcept { return impl_->label; }

} // namespace rgpot
