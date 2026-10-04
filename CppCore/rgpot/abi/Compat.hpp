#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Capability handshake for a peer `Capabilities` message.
 *
 * The refusal names the field, the required value and the received value.
 * A numeric field left at zero is absent. Empty buildVersion and
 * buildRevision are unknown, not a mismatch. The bridge numbers are the
 * eindir-core objective stamp this build links (major 1, minor 0, layout 1,
 * DLPack 1.0, features 0x3).
 */

#include "rgpot/rpc/Potentials.capnp.h"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace rgpot {
namespace abi {

#ifndef RGPOT_BUILD_VERSION
#define RGPOT_BUILD_VERSION ""
#endif
#ifndef RGPOT_BUILD_REVISION
#define RGPOT_BUILD_REVISION ""
#endif

inline constexpr const char *kProtocolFamily = "rgpot.potentials";
inline constexpr uint16_t kProtocolMajor = 1;
inline constexpr uint16_t kProtocolMinor = 0;
inline constexpr const char *kSchemaId = "0xbd1f89fa17369103";

/** What this build requires of a peer before a force evaluation. */
struct Expectation {
  std::string family = kProtocolFamily;
  uint16_t protocol_major = kProtocolMajor;
  uint16_t protocol_minor_min = 0;
  std::string schema_id = kSchemaId;
  uint16_t bridge_abi_major = 1;
  uint16_t bridge_abi_minor_max = 0;
  uint32_t bridge_layout = 1;
  uint16_t dlpack_major = 1;
  uint16_t dlpack_minor_max = 0;
  uint64_t bridge_features = 0x3;
  std::vector<::Capabilities::Operation> required_operations = {
      ::Capabilities::Operation::ENERGY, ::Capabilities::Operation::FORCES};
};

inline std::string mismatch(const std::string &field, const std::string &required,
                            const std::string &received) {
  return field + ": required " + required + ", received " + received;
}

inline const char *operation_name(::Capabilities::Operation op) {
  switch (op) {
  case ::Capabilities::Operation::ENERGY:
    return "energy";
  case ::Capabilities::Operation::FORCES:
    return "forces";
  case ::Capabilities::Operation::GRADIENT:
    return "gradient";
  case ::Capabilities::Operation::HESSIAN:
    return "hessian";
  case ::Capabilities::Operation::DIPOLE:
    return "dipole";
  case ::Capabilities::Operation::POLARIZABILITY:
    return "polarizability";
  case ::Capabilities::Operation::QUADRUPOLE:
    return "quadrupole";
  case ::Capabilities::Operation::STRESS:
    return "stress";
  case ::Capabilities::Operation::OPTIMIZE:
    return "optimize";
  case ::Capabilities::Operation::FREQUENCIES:
    return "frequencies";
  }
  return "unknown";
}

inline std::string hex_u64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << value;
  return out.str();
}

/** Empty when @p caps is acceptable, else the first incompatibility. */
inline std::string check_capabilities(::Capabilities::Reader caps,
                                      const Expectation &want = {}) {
  const std::string family = caps.getProtocolFamily().cStr();
  if (family != want.family)
    return mismatch("protocolFamily", want.family, family);
  if (caps.getProtocolMajor() != want.protocol_major)
    return mismatch("protocolMajor", std::to_string(want.protocol_major),
                    std::to_string(caps.getProtocolMajor()));
  if (caps.getProtocolMinor() < want.protocol_minor_min)
    return mismatch("protocolMinor",
                    ">= " + std::to_string(want.protocol_minor_min),
                    std::to_string(caps.getProtocolMinor()));
  const std::string schema_id = caps.getSchemaId().cStr();
  if (schema_id != want.schema_id)
    return mismatch("schemaId", want.schema_id, schema_id);
  if (caps.getBridgeAbiMajor() != want.bridge_abi_major)
    return mismatch("bridgeAbiMajor", std::to_string(want.bridge_abi_major),
                    std::to_string(caps.getBridgeAbiMajor()));
  if (caps.getBridgeAbiMinor() > want.bridge_abi_minor_max)
    return mismatch("bridgeAbiMinor",
                    "<= " + std::to_string(want.bridge_abi_minor_max),
                    std::to_string(caps.getBridgeAbiMinor()));
  if (caps.getBridgeLayout() != want.bridge_layout)
    return mismatch("bridgeLayout", std::to_string(want.bridge_layout),
                    std::to_string(caps.getBridgeLayout()));
  if (caps.getDlpackMajor() != want.dlpack_major)
    return mismatch("dlpackMajor", std::to_string(want.dlpack_major),
                    std::to_string(caps.getDlpackMajor()));
  if (caps.getDlpackMinor() > want.dlpack_minor_max)
    return mismatch("dlpackMinor",
                    "<= " + std::to_string(want.dlpack_minor_max),
                    std::to_string(caps.getDlpackMinor()));
  if ((caps.getBridgeFeatures() & ~want.bridge_features) != 0)
    return mismatch("bridgeFeatures",
                    "subset of " + hex_u64(want.bridge_features),
                    hex_u64(caps.getBridgeFeatures()));
  std::string served;
  for (const auto op : caps.getOperations()) {
    if (!served.empty())
      served += ", ";
    served += operation_name(op);
  }
  if (served.empty())
    served = "none";
  for (const auto required : want.required_operations) {
    bool found = false;
    for (const auto op : caps.getOperations())
      found = found || op == required;
    if (!found)
      return mismatch("operations", operation_name(required), served);
  }
  return {};
}

/** Decode a flat message and apply @ref check_capabilities. */
inline std::string
check_capabilities_bytes(const std::vector<unsigned char> &bytes,
                         const Expectation &want = {}) {
  if (bytes.empty() || bytes.size() % sizeof(::capnp::word) != 0)
    return mismatch("capabilities", "a word-aligned Capabilities message",
                    std::to_string(bytes.size()) + " bytes");
  std::vector<::capnp::word> storage(bytes.size() / sizeof(::capnp::word));
  std::memcpy(storage.data(), bytes.data(), bytes.size());
  try {
    ::capnp::FlatArrayMessageReader reader(
        kj::arrayPtr(storage.data(), storage.size()));
    return check_capabilities(reader.getRoot<::Capabilities>(), want);
  } catch (const kj::Exception &ex) {
    return mismatch("capabilities", "a Capabilities message",
                    ex.getDescription().cStr());
  }
}

/** Fill compatibility and build-identity fields. Other fields stay as set. */
inline void fill_compatibility(::Capabilities::Builder caps) {
  const Expectation want;
  caps.setProtocolFamily(want.family);
  caps.setProtocolMajor(want.protocol_major);
  caps.setProtocolMinor(kProtocolMinor);
  caps.setSchemaId(want.schema_id);
  caps.setBridgeAbiMajor(want.bridge_abi_major);
  caps.setBridgeAbiMinor(want.bridge_abi_minor_max);
  caps.setBridgeLayout(want.bridge_layout);
  caps.setDlpackMajor(want.dlpack_major);
  caps.setDlpackMinor(want.dlpack_minor_max);
  caps.setBridgeFeatures(want.bridge_features);
  caps.setBuildVersion(RGPOT_BUILD_VERSION);
  caps.setBuildRevision(RGPOT_BUILD_REVISION);
}

} // namespace abi
} // namespace rgpot
