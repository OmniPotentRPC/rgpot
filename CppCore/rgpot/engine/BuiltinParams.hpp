#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file BuiltinParams.hpp
 * @brief Reads the `BuiltinParams` arm (a name and an optional library path)
 * that selects a backend with no parameters of its own.
 */

#include <algorithm>
#include <cctype>
#include <string>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::engine {

struct Builtin {
  std::string name;    ///< Lowercased.
  std::string library; ///< Optional explicit engine library path.
};

inline Builtin read_builtin(const void *config, std::size_t config_len) {
  const kj::ArrayPtr<const capnp::word> words(
      reinterpret_cast<const capnp::word *>(config),
      config_len / sizeof(capnp::word));
  capnp::FlatArrayMessageReader reader(words);
  const auto params = reader.getRoot<::BuiltinParams>();
  Builtin out;
  out.name = params.getName().cStr();
  std::transform(out.name.begin(), out.name.end(), out.name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  out.library = params.getLibrary().cStr();
  return out;
}

} // namespace rgpot::engine
