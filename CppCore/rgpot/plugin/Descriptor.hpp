#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <rgpot/plugin.h>
#include <cstring>
#include <stdexcept>

namespace rgpot::plugin::detail {

inline rgpot_plugin_descriptor_t checkedDescriptor(
    const rgpot_plugin_descriptor_t *source) {
  constexpr auto required = offsetof(rgpot_plugin_descriptor_t, capabilities);
  if (!source || source->descriptor_size < required)
    throw std::invalid_argument("plugin descriptor is missing required fields");
  rgpot_plugin_descriptor_t result{};
  std::memcpy(&result, source, required);
  if (result.api_version != RGPOT_PLUGIN_API_VERSION)
    throw std::invalid_argument("plugin API version mismatch");
  if (!result.name || !result.name[0] || !result.create || !result.destroy ||
      !result.calculate)
    throw std::invalid_argument("plugin descriptor has invalid required fields");
  if (source->descriptor_size >=
      offsetof(rgpot_plugin_descriptor_t, capabilities) + sizeof(result.capabilities))
    result.capabilities = source->capabilities;
  if (source->descriptor_size >=
      offsetof(rgpot_plugin_descriptor_t, last_error) + sizeof(result.last_error))
    result.last_error = source->last_error;
  return result;
}

} // namespace rgpot::plugin::detail
