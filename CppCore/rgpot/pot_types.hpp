/**
 * @brief Definitions for supported potential energy surface types.
 *
 * This file defines the central @c PotType enumeration used throughout  the
 * library to identify and instantiate specific potentials.
 *
 * @note The implementations here are intentionally limited for demonstration
 * only. [eOn](https://eondocs.org) has a server component with a much larger
 * set of supported potentials.
 */

#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

#include "rgpot/base_types.hpp"

namespace rgpot {

/**
 * @brief Supported potential energy surface types.
 *
 * This enumeration is used by the factory and registry systems to dispatch
 * calculation requests to the appropriate implementation.
 */
enum class PotType {
  UNKNOWN = 0, //!<  The type is not defined or is invalid.
  CuH2,        //!<  Copper-Hydrogen EAM potential.
  LJ           //!<  Standard 12-6 Lennard-Jones pairwise potential.
};

} // namespace rgpot
