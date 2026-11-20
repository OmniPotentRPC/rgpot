#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Physical constants, unit conversion factors, and runtime unit parser.
 *
 * Compile-time constants use CODATA 2018 recommended values.
 * rgpot's native unit system: energy = eV, length = Angstrom, force =
 * eV/Angstrom.
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

// Exact SI defining constants and CODATA 2018 derived constants
inline constexpr double BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23;
inline constexpr double ELEMENTARY_CHARGE_C = 1.602176634e-19;
inline constexpr double HARTREE_ENERGY_J = 4.3597447222071e-18;
inline constexpr double REDUCED_PLANCK_CONSTANT_J_S = 1.0545718176461565e-34;

// Length
inline constexpr double BOHR_TO_ANGSTROM = 0.529177210903;
inline constexpr double ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM;
// Bohr radius in meters (CODATA 2018): a0 = 5.29177210903e-11 m.
inline constexpr double BOHR_RADIUS_M = 5.29177210903e-11;

// Energy
inline constexpr double HARTREE_TO_EV = 27.211386245988;
inline constexpr double EV_TO_HARTREE = 1.0 / HARTREE_TO_EV;

// Force / gradient conversion (Hartree/Bohr <-> eV/Angstrom)
inline constexpr double HARTREE_BOHR_TO_EV_ANGSTROM =
    HARTREE_TO_EV / BOHR_TO_ANGSTROM;
// Fused: negate gradient and convert to forces in eV/Angstrom
inline constexpr double NEG_GRAD_TO_FORCE = -HARTREE_BOHR_TO_EV_ANGSTROM;
// Stress: Hartree/Bohr^3 to eV/Angstrom^3. Matches a ForceInput that
// asks for eV and Angstrom (energy factor over length scale cubed).
inline constexpr double HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3 =
    HARTREE_TO_EV / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

// Temperature / Boltzmann
inline constexpr double KB_HARTREE = 3.1668115634556e-6; // k_B in Hartree/K
inline constexpr double KB_EV =
    BOLTZMANN_CONSTANT_J_PER_K / ELEMENTARY_CHARGE_C; // k_B in eV/K

// Mass (electron mass in u: 5.48579909065e-4 u)
inline constexpr double ELECTRON_MASS_IN_DALTON = 5.48579909065e-4;
inline constexpr double DALTON_IN_ELECTRON_MASSES =
    1.0 / ELECTRON_MASS_IN_DALTON;
// Unified atomic mass unit in kilograms (CODATA 2018): 1 u = 1.66053906660e-27 kg.
inline constexpr double DALTON_MASS_KG = 1.66053906660e-27;
// Electron rest mass in kilograms (CODATA 2018): m_e = 9.1093837015e-31 kg.
inline constexpr double ELECTRON_MASS_KG = 9.1093837015e-31;

// Amount of substance
// Avogadro constant, exact SI 2019 defining value.
inline constexpr double AVOGADRO_CONSTANT_PER_MOL = 6.02214076e23;

// Time (hbar / E_h in seconds)
inline constexpr double ATOMIC_UNIT_OF_TIME_S =
    REDUCED_PLANCK_CONSTANT_J_S / HARTREE_ENERGY_J;

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

/// Check that @p unit is a valid expression with dimensions matching @p
/// quantity. Known quantities: "length", "energy", "force", "pressure",
/// "momentum", "mass", "velocity", "charge".
///
/// Throws std::invalid_argument on mismatch or parse error.
void validate_unit(const std::string &quantity, const std::string &unit);

} // namespace rgpot::units
