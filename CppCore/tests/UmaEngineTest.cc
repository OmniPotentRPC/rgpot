// MIT License
// Copyright 2023--present rgpot developers
//
// libuma_engine.so through the generic engine C ABI (engine_c_abi.h): the
// symbols a host probes, the ABI identity, argument validation, error
// reporting, and, with a UMA fixture, evaluation against the directly
// compiled UmaPot. The engine is opened the way a host opens it: dlopen of the
// path in RGPOT_UMA_ENGINE and dlsym of each symbol.

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <catch2/catch_all.hpp>

#include "rgpot/UmaPot/UmaPot.hpp"
#include "rgpot/engine_c_abi.h"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"

namespace {

struct Engine {
  void *handle = nullptr;
  int (*abi_version)() = nullptr;
  int (*available)() = nullptr;
  RgpotEnginePot *(*create)(const void *, size_t, char *, size_t) = nullptr;
  void (*destroy)(RgpotEnginePot *) = nullptr;
  int (*force)(RgpotEnginePot *, long, const double *, const int *, double *,
               double *, double *, const double *,
               rgpot_engine_coord_transform, void *) = nullptr;
  int (*force_batch)(RgpotEnginePot *, long, long, const double *, const int *,
                     double *, double *, double *, const double *,
                     rgpot_engine_coord_transform, void *) = nullptr;
  RgpotEngineAbiStamp (*stamp)() = nullptr;
  int (*compatible)(const RgpotEngineAbiStamp *) = nullptr;
  int (*caps)(const RgpotEnginePot *, RgpotEngineCaps *) = nullptr;
  size_t (*last_error)(const RgpotEnginePot *, char *, size_t) = nullptr;
  int (*set_num_threads)(RgpotEnginePot *, int, int) = nullptr;
  int (*set_charge_spin)(RgpotEnginePot *, int, int) = nullptr;
};

template <typename Fn> Fn sym(void *handle, const char *name) {
  return reinterpret_cast<Fn>(dlsym(handle, name));
}

Engine &engine() {
  static Engine e = [] {
    Engine out;
    const char *path = std::getenv("RGPOT_UMA_ENGINE");
    REQUIRE(path != nullptr);
    out.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    REQUIRE(out.handle != nullptr);
    out.abi_version = sym<int (*)()>(out.handle, "rgpot_engine_abi_version");
    out.available = sym<int (*)()>(out.handle, "rgpot_engine_available");
    out.create = sym<decltype(out.create)>(out.handle, "rgpot_engine_create");
    out.destroy = sym<decltype(out.destroy)>(out.handle, "rgpot_engine_destroy");
    out.force = sym<decltype(out.force)>(out.handle, "rgpot_engine_force");
    out.force_batch =
        sym<decltype(out.force_batch)>(out.handle, "rgpot_engine_force_batch");
    out.stamp = sym<decltype(out.stamp)>(out.handle, "rgpot_engine_abi_stamp");
    out.compatible =
        sym<decltype(out.compatible)>(out.handle, "rgpot_engine_abi_compatible");
    out.caps = sym<decltype(out.caps)>(out.handle, "rgpot_engine_caps");
    out.last_error =
        sym<decltype(out.last_error)>(out.handle, "rgpot_engine_last_error");
    out.set_num_threads = sym<decltype(out.set_num_threads)>(
        out.handle, "rgpot_engine_set_num_threads");
    out.set_charge_spin = sym<decltype(out.set_charge_spin)>(
        out.handle, "rgpot_engine_set_charge_spin");
    return out;
  }();
  return e;
}

std::string last_error(const Engine &e, const RgpotEnginePot *pot) {
  const size_t n = e.last_error(pot, nullptr, 0);
  std::string out(n + 1, '\0');
  e.last_error(pot, out.data(), out.size());
  out.resize(n);
  return out;
}

/// capnp flat array of UmaParams.
kj::Array<capnp::word> uma_params(const std::string &model, int charge = 0,
                                  int spin = 1) {
  capnp::MallocMessageBuilder msg;
  auto params = msg.initRoot<::UmaParams>();
  params.setModelPath(model);
  params.setTaskName("omol");
  params.setDevice("cpu");
  params.setCharge(charge);
  params.setSpin(spin);
  return capnp::messageToFlatArray(msg);
}

/// Path of a fixture, or empty when absent and not required; a required but
/// missing fixture fails.
std::string fixture(const char *variable) {
  const char *path = std::getenv(variable);
  if (path != nullptr && *path != '\0') {
    return path;
  }
  const char *require = std::getenv("RGPOT_UMA_REQUIRE_FIXTURES");
  if (require != nullptr && *require != '\0') {
    FAIL(std::string(variable) + " is required and not set");
  }
  return {};
}

// Baker 01_hcn reactant, Angstrom, 25 A cubic cell.
const std::vector<double> kHcnPositions = {
    12.49734736216627162, 12.49892801474515913, 12.54059929828148512,
    12.50115413363106498, 12.50036504272228832, 11.38209979880783251,
    12.50149850420264563, 12.50069809648255514, 13.61514544631068446};
const std::vector<int> kHcnZ = {6, 7, 1};
const std::array<double, 9> kBox = {25, 0, 0, 0, 25, 0, 0, 0, 25};

} // namespace

TEST_CASE("the engine exports the required and the optional symbols",
          "[uma][engine]") {
  const Engine &e = engine();
  REQUIRE(e.abi_version != nullptr);
  REQUIRE(e.available != nullptr);
  REQUIRE(e.create != nullptr);
  REQUIRE(e.destroy != nullptr);
  REQUIRE(e.force != nullptr);
  REQUIRE(e.force_batch != nullptr);
  REQUIRE(e.stamp != nullptr);
  REQUIRE(e.compatible != nullptr);
  REQUIRE(e.caps != nullptr);
  REQUIRE(e.last_error != nullptr);
  REQUIRE(e.set_num_threads != nullptr);
  REQUIRE(e.set_charge_spin != nullptr);
  REQUIRE(e.available() == 1);
}

TEST_CASE("the engine reports its ABI identity", "[uma][engine]") {
  const Engine &e = engine();
  REQUIRE(e.abi_version() == RGPOT_ENGINE_ABI_VERSION);
  const RgpotEngineAbiStamp stamp = e.stamp();
  REQUIRE(stamp.abi_major == RGPOT_ENGINE_ABI_VERSION);
  REQUIRE(stamp.abi_minor == RGPOT_ENGINE_ABI_MINOR);
  REQUIRE(stamp.layout_revision == RGPOT_ENGINE_ABI_LAYOUT_REVISION);

  RgpotEngineAbiStamp host = stamp;
  REQUIRE(e.compatible(&host) == 1);
  host.abi_minor = static_cast<unsigned short>(stamp.abi_minor - 1);
  REQUIRE(e.compatible(&host) == 1); // an older host minor is served
  host = stamp;
  host.abi_minor += 1;
  REQUIRE(e.compatible(&host) == 0);
  host = stamp;
  host.abi_major += 1;
  REQUIRE(e.compatible(&host) == 0);
  host = stamp;
  host.layout_revision += 1;
  REQUIRE(e.compatible(&host) == 0);
  REQUIRE(e.compatible(nullptr) == 0);
}

TEST_CASE("create refuses a missing or malformed configuration",
          "[uma][engine]") {
  const Engine &e = engine();
  char err[256] = {0};
  REQUIRE(e.create(nullptr, 0, err, sizeof(err)) == nullptr);
  REQUIRE(std::string(err).find("UmaParams") != std::string::npos);

  std::memset(err, 0, sizeof(err));
  auto empty = uma_params("");
  auto bytes = empty.asBytes();
  REQUIRE(e.create(bytes.begin(), bytes.size(), err, sizeof(err)) == nullptr);
  REQUIRE(std::string(err).find("modelPath") != std::string::npos);
}

TEST_CASE("an instance reports capabilities and rejects bad calls",
          "[uma][engine]") {
  const Engine &e = engine();
  // The model loads on the first evaluation, so an instance exists without a
  // package.
  auto params = uma_params("/nonexistent/model.pt2");
  auto bytes = params.asBytes();
  char err[256] = {0};
  RgpotEnginePot *pot = e.create(bytes.begin(), bytes.size(), err, sizeof(err));
  REQUIRE(pot != nullptr);

  RgpotEngineCaps caps;
  std::memset(&caps, 0xff, sizeof(caps));
  caps.size = sizeof(caps);
  REQUIRE(e.caps(pot, &caps) == 0);
  REQUIRE(caps.size == sizeof(caps));
  REQUIRE(caps.reentrancy == 0); // shared instance
  REQUIRE(caps.per_image_instances == 1);
  REQUIRE(caps.periodic == 1);
  REQUIRE(caps.stress == 0);
  REQUIRE(caps.group_collective == 0);

  // A host built against a smaller struct receives only what fits.
  unsigned small[2] = {sizeof(small), 0xdeadbeefu};
  REQUIRE(e.caps(pot, reinterpret_cast<RgpotEngineCaps *>(small)) == 0);
  REQUIRE(small[0] == sizeof(small));
  REQUIRE(small[1] == 0u); // reentrancy: shared instance
  REQUIRE(e.caps(pot, nullptr) == 1);

  REQUIRE(last_error(e, pot).empty());
  double forces[9] = {0};
  double energy = 0.0;
  REQUIRE(e.force(pot, 3, nullptr, kHcnZ.data(), forces, &energy, nullptr,
                  kBox.data(), nullptr, nullptr) == 1);
  REQUIRE(last_error(e, pot).find("invalid argument") != std::string::npos);

  auto reject = [](void *, long, double *, double *) { return 7; };
  REQUIRE(e.force(pot, 3, kHcnPositions.data(), kHcnZ.data(), forces, &energy,
                  nullptr, kBox.data(), reject, nullptr) == 7);
  REQUIRE(last_error(e, pot).find("7") != std::string::npos);

  // A package that cannot load is a recoverable failure with a reason.
  REQUIRE(e.force(pot, 3, kHcnPositions.data(), kHcnZ.data(), forces, &energy,
                  nullptr, kBox.data(), nullptr, nullptr) == 2);
  REQUIRE(!last_error(e, pot).empty());

  char small_buf[8];
  const size_t full = e.last_error(pot, small_buf, sizeof(small_buf));
  REQUIRE(full > sizeof(small_buf) - 1);
  REQUIRE(std::strlen(small_buf) == sizeof(small_buf) - 1);

  REQUIRE(e.set_charge_spin(pot, 0, 0) == 1); // spin multiplicity is >= 1
  REQUIRE(e.set_charge_spin(pot, 1, 2) == 0);
  e.destroy(pot);
}

TEST_CASE("the engine evaluates like the directly compiled UmaPot",
          "[uma][engine][fixture]") {
  const std::string model = fixture("RGPOT_UMA_OMOL_PT2");
  if (model.empty()) {
    SKIP("RGPOT_UMA_OMOL_PT2 names no fixture");
  }
  const Engine &e = engine();
  auto params = uma_params(model);
  auto bytes = params.asBytes();
  char err[512] = {0};
  RgpotEnginePot *pot = e.create(bytes.begin(), bytes.size(), err, sizeof(err));
  INFO(err);
  REQUIRE(pot != nullptr);
  REQUIRE(e.set_num_threads(pot, 1, -1) == 0);

  std::vector<double> forces(9, 0.0);
  double energy = 0.0;
  REQUIRE(e.force(pot, 3, kHcnPositions.data(), kHcnZ.data(), forces.data(),
                  &energy, nullptr, kBox.data(), nullptr, nullptr) == 0);
  REQUIRE(last_error(e, pot).empty());

  rgpot::UmaConfig cfg;
  cfg.model_path = model;
  rgpot::UmaPot direct(cfg);
  rgpot::ForceInput in{3, kHcnPositions.data(), kHcnZ.data(), kBox.data()};
  std::vector<double> ref(9, 0.0);
  rgpot::ForceOut out{ref.data(), 0.0, 0.0};
  direct.forceImpl(in, &out);

  // The compiled model is not bit-reproducible between processes (about 1e-7
  // eV); the engine and the direct path agree to well inside 1e-5.
  REQUIRE_THAT(energy, Catch::Matchers::WithinAbs(out.energy, 1e-5));
  for (size_t k = 0; k < 9; ++k) {
    REQUIRE_THAT(forces[k], Catch::Matchers::WithinAbs(ref[k], 1e-5));
  }

  // A charge the package was not exported for is refused with a reason, and
  // restoring it clears the failure.
  REQUIRE(e.set_charge_spin(pot, 1, 2) == 0);
  REQUIRE(e.force(pot, 3, kHcnPositions.data(), kHcnZ.data(), forces.data(),
                  &energy, nullptr, kBox.data(), nullptr, nullptr) == 2);
  REQUIRE(last_error(e, pot).find("charge") != std::string::npos);
  REQUIRE(e.set_charge_spin(pot, 0, 1) == 0);
  REQUIRE(e.force(pot, 3, kHcnPositions.data(), kHcnZ.data(), forces.data(),
                  &energy, nullptr, kBox.data(), nullptr, nullptr) == 0);
  REQUIRE(last_error(e, pot).empty());
  e.destroy(pot);
}

TEST_CASE("a band package serves a batch through the engine",
          "[uma][engine][fixture]") {
  const std::string model = fixture("RGPOT_UMA_OMOL_BAND_PT2");
  if (model.empty()) {
    SKIP("RGPOT_UMA_OMOL_BAND_PT2 names no fixture");
  }
  const Engine &e = engine();
  auto params = uma_params(model);
  auto bytes = params.asBytes();
  char err[512] = {0};
  RgpotEnginePot *pot = e.create(bytes.begin(), bytes.size(), err, sizeof(err));
  INFO(err);
  REQUIRE(pot != nullptr);

  constexpr long B = 3;
  std::vector<double> R, boxes, F(9 * B, 0.0), U(B, 0.0);
  for (long s = 0; s < B; ++s) {
    for (double v : kHcnPositions) {
      R.push_back(v);
    }
    boxes.insert(boxes.end(), kBox.begin(), kBox.end());
  }
  REQUIRE(e.force_batch(pot, B, 3, R.data(), kHcnZ.data(), F.data(), U.data(),
                        nullptr, boxes.data(), nullptr, nullptr) == 0);
  std::vector<double> f1(9, 0.0);
  double e1 = 0.0;
  REQUIRE(e.force(pot, 3, kHcnPositions.data(), kHcnZ.data(), f1.data(), &e1,
                  nullptr, kBox.data(), nullptr, nullptr) == 0);
  for (long s = 0; s < B; ++s) {
    REQUIRE_THAT(U[s], Catch::Matchers::WithinAbs(e1, 1e-5));
  }
  e.destroy(pot);
}
