#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <rgpot/plugin.h>

#include <string>

#include "rgpot/Potential.hpp"

namespace rgpot::plugin {

/// Create a PotentialBase-compatible wrapper from a loaded plugin.
/// The returned object owns the plugin instance and handles unit conversion.
///
/// @param desc   Plugin descriptor (must outlive the returned object).
/// @param config JSON config string passed to desc->create().
/// @return A unique_ptr to a PotentialBase.
std::unique_ptr<rgpot::PotentialBase>
create_from_plugin(const rgpot_plugin_descriptor_t *desc,
                   const std::string &config = "{}");

} // namespace rgpot::plugin
