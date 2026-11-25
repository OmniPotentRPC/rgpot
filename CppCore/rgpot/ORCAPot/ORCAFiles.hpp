#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file ORCAFiles.hpp
 * @brief ORCA input text and .engrad parsing, with no process handling.
 *
 * Pure text functions so the input ORCAPot hands ORCA, and the reading of
 * what ORCA wrote back, can be checked without an ORCA installation.
 */

#include <cstddef>
#include <string>
#include <vector>

namespace rgpot::orca {

/// What a .engrad file holds, in ORCA's own units.
struct Engrad {
  double energy_hartree = 0.0;
  /// Gradient dE/dR, Hartree/Bohr, 3 * nAtoms entries, atom-major.
  std::vector<double> gradient_hartree_per_bohr;
  std::vector<int> atomic_numbers;
};

/// Everything the input of one single point depends on.
struct InputSpec {
  std::string method;
  std::string basis;
  std::string keywords;
  int charge = 0;
  int multiplicity = 1;
  int nprocs = 1;
  int maxcore_mb = 0;
  /// Name of the .gbw ORCA reads its starting orbitals from, relative to
  /// the run directory. Empty starts from ORCA's own guess.
  std::string moinp;
};

/// Element symbol of atomic number @p z (1..118). Throws
/// std::invalid_argument outside that range.
std::string elementSymbol(int z);

/// The ORCA input for one energy and gradient. Positions are Angstrom,
/// which is also ORCA's unit for a `* xyz` block, so no factor is applied.
/// Throws std::invalid_argument for a keyword field holding a line break.
std::string renderInput(const InputSpec &spec, std::size_t n_atoms,
                        const double *pos_angstrom, const int *atomic_numbers);

/// Parse the text of a .engrad file. Throws std::runtime_error naming the
/// first thing that does not fit, including a count that is not
/// @p n_atoms. Does not check that the numbers are finite.
Engrad parseEngrad(const std::string &text, std::size_t n_atoms);

/// The last @p max_bytes bytes of @p path, with a line break cut at a line
/// start where one exists. A file that cannot be read gives a note saying
/// so; this never throws.
std::string fileTail(const std::string &path, std::size_t max_bytes);

} // namespace rgpot::orca
