#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Capability handshake for backends loaded through the potential
 * profile: decide from a `Capabilities` message whether this build can drive
 * the backend before any work is dispatched, and fill the same message from
 * the producing side.
 *
 * Metadata a backend leaves unset (empty family, zero bridge major) is not a
 * declaration and is not checked. The constants mirror the eindir-core
 * objective ABI stamp that rgpot-core compares against.
 */

#include "rgpot/abi/ProfileLoader.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

namespace rgpot {
namespace abi {

inline constexpr const char *kProtocolFamily = "rgpot.potentials";
inline constexpr uint16_t kProtocolMajor = 1;
inline constexpr uint16_t kProtocolMinor = 0;
inline constexpr const char *kSchemaId = "0xbd1f89fa17369103";

/** What the host requires of a backend. */
struct Expectation {
  std::string family = kProtocolFamily;
  uint16_t protocol_major = kProtocolMajor;
  uint16_t protocol_minor_min = 0; //!< Lowest protocol minor accepted.
  std::string schema_id = kSchemaId;
  uint16_t bridge_abi_major = 1;
  uint16_t bridge_abi_minor_max = 0; //!< Highest bridge minor consumable.
  uint32_t bridge_layout = 1;
  uint16_t dlpack_major = 1;
  uint16_t dlpack_minor_max = 0;     //!< Highest DLPack minor consumable.
  uint64_t bridge_features = 0x3;    //!< gradient | batch.
  std::vector<::Capabilities::Operation> required_operations = {
      ::Capabilities::Operation::ENERGY, ::Capabilities::Operation::FORCES};
};

/** Empty when @p caps is acceptable, else the first incompatibility. */
inline std::string check_capabilities(::Capabilities::Reader caps,
                                      const Expectation &want = {}) {
  const std::string family = caps.getProtocolFamily().cStr();
  if (!family.empty()) {
    if (family != want.family)
      return "protocol family \"" + family + "\" is not \"" + want.family + "\"";
    if (caps.getProtocolMajor() != want.protocol_major)
      return "protocol major " + std::to_string(caps.getProtocolMajor()) +
             " is not " + std::to_string(want.protocol_major);
    if (caps.getProtocolMinor() < want.protocol_minor_min)
      return "protocol minor " + std::to_string(caps.getProtocolMinor()) +
             " is below the required " +
             std::to_string(want.protocol_minor_min);
  }
  const std::string schema_id = caps.getSchemaId().cStr();
  if (!schema_id.empty() && schema_id != want.schema_id)
    return "schema id " + schema_id + " is not " + want.schema_id;
  if (caps.getBridgeAbiMajor() != 0) {
    if (caps.getBridgeAbiMajor() != want.bridge_abi_major)
      return "eindir bridge ABI major " +
             std::to_string(caps.getBridgeAbiMajor()) + " is not " +
             std::to_string(want.bridge_abi_major);
    if (caps.getBridgeAbiMinor() > want.bridge_abi_minor_max)
      return "eindir bridge ABI minor " +
             std::to_string(caps.getBridgeAbiMinor()) +
             " exceeds the supported " +
             std::to_string(want.bridge_abi_minor_max);
    if (caps.getBridgeLayout() != want.bridge_layout)
      return "eindir objective layout " +
             std::to_string(caps.getBridgeLayout()) + " is not " +
             std::to_string(want.bridge_layout);
    if ((caps.getBridgeFeatures() & ~want.bridge_features) != 0)
      return "eindir bridge features include bits outside the supported set";
  }
  if (caps.getDlpackMajor() != 0) {
    if (caps.getDlpackMajor() != want.dlpack_major)
      return "DLPack major " + std::to_string(caps.getDlpackMajor()) +
             " is not " + std::to_string(want.dlpack_major);
    if (caps.getDlpackMinor() > want.dlpack_minor_max)
      return "DLPack minor " + std::to_string(caps.getDlpackMinor()) +
             " exceeds the supported " + std::to_string(want.dlpack_minor_max);
  }
  for (const auto required : want.required_operations) {
    bool served = false;
    for (const auto op : caps.getOperations())
      served = served || op == required;
    if (!served)
      return "a required operation is not served";
  }
  return {};
}

/** Fill the compatibility fields of @p caps with what this build speaks. */
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
}

/**
 * Load a profile backend and refuse it before dispatch when its
 * `Capabilities` are incompatible. On refusal the loader is left unloaded
 * and std::runtime_error names the first mismatch.
 */
inline void checked_load(ProfileLoader &loader, const std::string &prefix,
                         const std::string &explicit_path = "",
                         const Expectation &want = {}) {
  loader.load(prefix, explicit_path);
  const auto bytes = loader.capabilities();
  if (bytes.size() % sizeof(::capnp::word) != 0) {
    loader.capabilities_result = nullptr;
    throw std::runtime_error(prefix + " capabilities have a ragged size");
  }
  std::vector<::capnp::word> storage(bytes.size() / sizeof(::capnp::word));
  std::memcpy(storage.data(), bytes.data(), bytes.size());
  std::string why;
  try {
    ::capnp::FlatArrayMessageReader reader(
        kj::arrayPtr(storage.data(), storage.size()));
    why = check_capabilities(reader.getRoot<::Capabilities>(), want);
  } catch (const kj::Exception &ex) {
    why = std::string("unreadable: ") + ex.getDescription().cStr();
  }
  if (!why.empty()) {
    loader.capabilities_result = nullptr;
    throw std::runtime_error(prefix + " backend refused: " + why);
  }
}

} // namespace abi
} // namespace rgpot
