#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Capability handshake applied while loading a profile backend; the
 * expectation, check and producer helpers live in Handshake.hpp.
 */

#include "rgpot/abi/Handshake.hpp"
#include "rgpot/abi/ProfileLoader.hpp"

namespace rgpot {
namespace abi {

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
