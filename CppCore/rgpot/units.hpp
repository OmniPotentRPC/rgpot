#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Physical constants, unit conversion factors, and runtime unit parser.
 *
 * Compile-time constants use CODATA 2018 recommended values.
 * rgpot's native unit system: energy = eV, length = Angstrom, force = eV/Angstrom.
 *
 * The runtime expression parser (unit_conversion_factor) is derived from
 * metatomic-torch (https://github.com/metatensor/metatomic), specifically
 * PR #173 by Rohit Goswami. Original code: BSD-3-Clause licensed, copyright
 * metatensor developers. The parser uses the Shunting-Yard algorithm for
 * compound unit expressions like "kJ/mol/A^2" or "(eV*u)^(1/2)" with
 * SI-based dimensional analysis.
 */

#include <string>

namespace rgpot::units {

// ---- Compile-time constants (CODATA 2018) ----

// Length
inline constexpr double BOHR_TO_ANGSTROM = 0.529177210903;
inline constexpr double ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM;

// Energy
inline constexpr double HARTREE_TO_EV = 27.211386245988;
inline constexpr double EV_TO_HARTREE = 1.0 / HARTREE_TO_EV;

// Force / gradient conversion (Hartree/Bohr <-> eV/Angstrom)
inline constexpr double HARTREE_BOHR_TO_EV_ANGSTROM =
    HARTREE_TO_EV / BOHR_TO_ANGSTROM;
// Fused: negate gradient and convert to forces in eV/Angstrom
inline constexpr double NEG_GRAD_TO_FORCE = -HARTREE_BOHR_TO_EV_ANGSTROM;

// Temperature / Boltzmann
inline constexpr double KB_HARTREE = 3.1668115634556e-6; // k_B in Hartree/K
inline constexpr double KB_EV = 8.617333262e-5;          // k_B in eV/K

// ---- Runtime unit expression parser ----

/// Get the multiplicative conversion factor from @p from_unit to @p to_unit.
///
/// Both arguments are parsed as compound unit expressions. Operators:
/// `*` (multiply), `/` (divide), `^` (power), `()` (grouping).
/// Whitespace is ignored, lookup is case-insensitive.
///
/// Examples:
/// - `unit_conversion_factor("angstrom", "bohr")` -- length
/// - `unit_conversion_factor("eV/A", "hartree/bohr")` -- force
/// - `unit_conversion_factor("kJ/mol", "eV")` -- energy per particle
/// - `unit_conversion_factor("(eV*u)^(1/2)", "u*A/fs")` -- momentum
///
/// Supported base units:
/// - Length: angstrom (A), bohr, nanometer (nm), meter (m), cm, mm, um
/// - Energy: eV, meV, Hartree, Ry, Joule (J), kcal, kJ
/// - Time: second (s), ms, us, ns, ps, femtosecond (fs)
/// - Mass: dalton (u), kilogram (kg), gram (g), electron_mass (m_e)
/// - Charge: e, coulomb (c)
/// - Dimensionless: mol
/// - Derived: hbar
///
/// Throws std::invalid_argument on dimension mismatch or unknown unit.
///
/// @note Derived from metatomic-torch (metatensor/metatomic PR #173).
double unit_conversion_factor(const std::string &from_unit,
                              const std::string &to_unit);

/// Check that @p unit is a valid expression with dimensions matching @p quantity.
/// Known quantities: "length", "energy", "force", "pressure", "momentum",
/// "mass", "velocity", "charge".
///
/// Throws std::invalid_argument on mismatch or parse error.
void validate_unit(const std::string &quantity, const std::string &unit);

} // namespace rgpot::units
