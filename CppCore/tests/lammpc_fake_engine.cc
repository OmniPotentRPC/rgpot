// MIT License
// Copyright 2023--present rgpot developers
//
// A liblammpc stand-in exporting the minimum potential ABI profile. A session
// reads lj/cut "1 1 eps sigma" from PotentialConfig.lammps and evaluates the
// unshifted 12-6 pair over all atom pairs, open boundaries. It declares the
// stress operation and every result carries kStress, so a host can check it
// is passed through. lammpc_fake_init_calls and lammpc_fake_finalize_calls
// count how often its backend-global state was set up and finalized.
//
// Two variants are built from this file. LAMMPC_FAKE_NO_STRESS declares no
// stress operation and returns no stress. LAMMPC_FAKE_BRIDGE_ABI_MAJOR
// reports that eindir bridge ABI major in its Capabilities, which a host
// with another major must refuse.

#include "rgpot/abi/Compat.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>
#include <kj/exception.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#define LAMMPC_FAKE_EXPORT __declspec(dllexport)
#else
#define LAMMPC_FAKE_EXPORT
#endif

struct LammpcFakeResult {
  int ok;
  double energy_h;
  char message[512];
};

struct LammpcFakeSession {
  double eps = 0.0;
  double sigma = 0.0;
};

namespace {

#ifndef LAMMPC_FAKE_NO_STRESS
constexpr double kStress[9] = {0.1, 0.2, 0.3, 0.2, 0.4, 0.5, 0.3, 0.5, 0.6};
#endif

std::string g_last_error;

// Stands in for the process-global state a real lammpc sets up on its first
// session, before it reads the config, and releases in lammpc_finalize.
bool g_initialized = false;
int g_init_calls = 0;
int g_finalize_calls = 0;

bool has_flat_message(const void *msg, size_t size) {
  return msg != nullptr && size >= sizeof(::capnp::word) &&
         (size % sizeof(::capnp::word)) == 0;
}

kj::ArrayPtr<const ::capnp::word> words_of(const void *msg, size_t size) {
  return kj::arrayPtr(static_cast<const ::capnp::word *>(msg),
                      size / sizeof(::capnp::word));
}

LammpcFakeResult make_result(int ok, const std::string &message) {
  LammpcFakeResult r;
  r.ok = ok;
  r.energy_h = 0.0;
  std::snprintf(r.message, sizeof(r.message), "%s", message.c_str());
  return r;
}

bool read_session(const void *config, size_t size, LammpcFakeSession *out) {
  if (!has_flat_message(config, size)) {
    g_last_error = "config is not a flat message";
    return false;
  }
  try {
    ::capnp::FlatArrayMessageReader reader(words_of(config, size));
    const auto root = reader.getRoot<::PotentialConfig>();
    if (root.which() != ::PotentialConfig::LAMMPS) {
      g_last_error = "config has no lammps arm";
      return false;
    }
    const auto lammps = root.getLammps();
    const std::string pair_style = lammps.getPairStyle().cStr();
    if (pair_style.rfind("lj/cut", 0) != 0 ||
        lammps.getPairCoeffs().size() != 1) {
      g_last_error =
          "fake serves one lj/cut pair_coeff, got '" + pair_style + "'";
      return false;
    }
    std::istringstream coeff(lammps.getPairCoeffs()[0].cStr());
    int ti = 0;
    int tj = 0;
    if (!(coeff >> ti >> tj >> out->eps >> out->sigma)) {
      g_last_error = "unreadable pair_coeff";
      return false;
    }
  } catch (const kj::Exception &ex) {
    g_last_error = ex.getDescription().cStr();
    return false;
  }
  g_last_error.clear();
  return true;
}

// PotentialResult for one ForceInput, or an empty vector with g_last_error.
std::vector<unsigned char> evaluate(const LammpcFakeSession &session,
                                    const void *input, size_t size) {
  if (!has_flat_message(input, size)) {
    g_last_error = "ForceInput is not a flat message";
    return {};
  }
  try {
    ::capnp::FlatArrayMessageReader reader(words_of(input, size));
    const auto in = reader.getRoot<::ForceInput>();
    const auto pos = in.getPos();
    if (pos.size() == 0 || pos.size() % 3 != 0 ||
        in.getAtmnrs().size() != pos.size() / 3) {
      g_last_error = "ForceInput geometry is inconsistent";
      return {};
    }
    if (std::string(in.getLengthUnit().cStr()) != "angstrom" ||
        std::string(in.getEnergyUnit().cStr()) != "eV") {
      g_last_error = "fake takes angstrom and eV";
      return {};
    }
    const size_t n = pos.size() / 3;
    std::vector<double> forces(pos.size(), 0.0);
    double energy = 0.0;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        double d[3];
        double r2 = 0.0;
        for (size_t k = 0; k < 3; ++k) {
          d[k] = pos[3 * j + k] - pos[3 * i + k];
          r2 += d[k] * d[k];
        }
        const double sr2 = session.sigma * session.sigma / r2;
        const double sr6 = sr2 * sr2 * sr2;
        energy += 4.0 * session.eps * (sr6 * sr6 - sr6);
        const double f_over_r =
            24.0 * session.eps * (2.0 * sr6 * sr6 - sr6) / r2;
        for (size_t k = 0; k < 3; ++k) {
          forces[3 * j + k] += f_over_r * d[k];
          forces[3 * i + k] -= f_over_r * d[k];
        }
      }
    }
    ::capnp::MallocMessageBuilder msg;
    auto result = msg.initRoot<::PotentialResult>();
    result.setEnergy(energy);
    auto out_forces =
        result.initForces(static_cast<unsigned int>(forces.size()));
    for (unsigned int i = 0; i < out_forces.size(); ++i)
      out_forces.set(i, forces[i]);
#ifndef LAMMPC_FAKE_NO_STRESS
    auto stress = result.initStress(9);
    for (unsigned int i = 0; i < 9; ++i)
      stress.set(i, kStress[i]);
#endif
    auto words = ::capnp::messageToFlatArray(msg);
    const auto bytes = words.asBytes();
    g_last_error.clear();
    return std::vector<unsigned char>(bytes.begin(), bytes.end());
  } catch (const kj::Exception &ex) {
    g_last_error = ex.getDescription().cStr();
    return {};
  }
}

} // namespace

extern "C" {

LAMMPC_FAKE_EXPORT int lammpc_abi_version(void) { return 0; }

LAMMPC_FAKE_EXPORT const char *lammpc_version(void) {
  return "lammpc-fake/0.1";
}

LAMMPC_FAKE_EXPORT const char *lammpc_last_error(void) {
  return g_last_error.c_str();
}

LAMMPC_FAKE_EXPORT int lammpc_available(void) { return 1; }

LAMMPC_FAKE_EXPORT void lammpc_finalize(void) {
  ++g_finalize_calls;
  g_initialized = false;
}

LAMMPC_FAKE_EXPORT int lammpc_fake_init_calls(void) { return g_init_calls; }

LAMMPC_FAKE_EXPORT int lammpc_fake_finalize_calls(void) {
  return g_finalize_calls;
}

LAMMPC_FAKE_EXPORT int lammpc_configure(const void *config, size_t size) {
  LammpcFakeSession probe;
  return read_session(config, size, &probe) ? 0 : -1;
}

LAMMPC_FAKE_EXPORT LammpcFakeSession *
lammpc_session_create_from_config(const void *config, size_t size) {
  if (!g_initialized) {
    g_initialized = true;
    ++g_init_calls;
  }
  LammpcFakeSession session;
  if (!read_session(config, size, &session))
    return nullptr;
  return new LammpcFakeSession(session);
}

LAMMPC_FAKE_EXPORT int lammpc_session_configure(LammpcFakeSession *session,
                                                const void *config,
                                                size_t size) {
  if (session == nullptr)
    return -1;
  LammpcFakeSession next;
  if (!read_session(config, size, &next))
    return -1;
  *session = next;
  return 0;
}

LAMMPC_FAKE_EXPORT void lammpc_session_destroy(LammpcFakeSession *session) {
  delete session;
}

LAMMPC_FAKE_EXPORT size_t
lammpc_potential_result_size_for_force_input(const void *input, size_t size) {
  LammpcFakeSession unit;
  unit.eps = 1.0;
  unit.sigma = 1.0;
  return evaluate(unit, input, size).size();
}

LAMMPC_FAKE_EXPORT LammpcFakeResult lammpc_session_calculate_result(
    LammpcFakeSession *session, const void *input, size_t input_size, void *out,
    size_t capacity, size_t *written) {
  if (session == nullptr || written == nullptr)
    return make_result(0, "invalid fake session arguments");
  const auto bytes = evaluate(*session, input, input_size);
  if (bytes.empty())
    return make_result(0, g_last_error);
  *written = bytes.size();
  if (out == nullptr || capacity < bytes.size())
    return make_result(0, "result buffer too small");
  std::memcpy(out, bytes.data(), bytes.size());
  return make_result(1, "fake result ok");
}

LAMMPC_FAKE_EXPORT LammpcFakeResult lammpc_calculate_result_from_config(
    const void *config, size_t config_size, const void *input,
    size_t input_size, void *out, size_t capacity, size_t *written) {
  LammpcFakeSession session;
  if (!read_session(config, config_size, &session))
    return make_result(0, g_last_error);
  return lammpc_session_calculate_result(&session, input, input_size, out,
                                         capacity, written);
}

LAMMPC_FAKE_EXPORT int lammpc_capabilities_result(void *out, size_t capacity,
                                                  size_t *written) {
  if (written == nullptr)
    return -1;
  try {
    ::capnp::MallocMessageBuilder msg;
    auto caps = msg.initRoot<::Capabilities>();
    caps.setBackendName("lammpc");
    caps.setBackendVersion(lammpc_version());
    caps.setAbiVersion(lammpc_abi_version());
    caps.setAvailable(true);
#ifdef LAMMPC_FAKE_NO_STRESS
    auto ops = caps.initOperations(2);
#else
    auto ops = caps.initOperations(3);
    ops.set(2, ::Capabilities::Operation::STRESS);
#endif
    ops.set(0, ::Capabilities::Operation::ENERGY);
    ops.set(1, ::Capabilities::Operation::FORCES);
    auto kinds = caps.initConfigKinds(1);
    kinds.set(0, "lammps");
    caps.setSchemaVersion("fake");
    rgpot::abi::fill_compatibility(caps);
#ifdef LAMMPC_FAKE_BRIDGE_ABI_MAJOR
    caps.setBridgeAbiMajor(LAMMPC_FAKE_BRIDGE_ABI_MAJOR);
#endif
    auto words = ::capnp::messageToFlatArray(msg);
    const auto bytes = words.asBytes();
    *written = bytes.size();
    if (out == nullptr || capacity < bytes.size())
      return -1;
    std::memcpy(out, bytes.begin(), bytes.size());
    return 0;
  } catch (const kj::Exception &) {
    *written = 0;
    return -1;
  }
}

} // extern "C"
