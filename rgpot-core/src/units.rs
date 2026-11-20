// MIT License
// Copyright 2023--present rgpot developers

#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

//! CODATA 2018 physical constants and unit conversions shared with
//! `CppCore/rgpot/units.hpp`.
//!
//! The integration tests compare every value bit for bit with the C++ header.
//! Defining the constants inside this crate also keeps the published `units`
//! feature independent of files outside the crate package.
//!
//! ```
//! use rgpot_core::units;
//!
//! assert_eq!(units::ANGSTROM_TO_BOHR, 1.0 / units::BOHR_TO_ANGSTROM);
//! assert_eq!(units::EV_TO_HARTREE, 1.0 / units::HARTREE_TO_EV);
//! ```

/// Exact Boltzmann constant in joules per kelvin.
pub const BOLTZMANN_CONSTANT_J_PER_K: f64 = 1.380_649e-23;
/// Exact elementary charge in coulombs.
pub const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;
/// Hartree energy in joules (CODATA 2018).
pub const HARTREE_ENERGY_J: f64 = 4.359_744_722_207_1e-18;
/// Reduced Planck constant in joule seconds (CODATA 2018).
pub const REDUCED_PLANCK_CONSTANT_J_S: f64 = 1.054_571_817_646_156_5e-34;

/// Bohr radius in angstroms (`a_0`, CODATA 2018).
pub const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_903;
/// One angstrom in bohrs.
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_TO_ANGSTROM;
/// Hartree energy in electronvolts (`E_h / e`, CODATA 2018).
pub const HARTREE_TO_EV: f64 = 27.211_386_245_988;
/// One electronvolt in hartrees.
pub const EV_TO_HARTREE: f64 = 1.0 / HARTREE_TO_EV;
/// Boltzmann constant in hartrees per kelvin (`k_B / E_h`, CODATA 2018).
pub const KB_HARTREE: f64 = 3.166_811_563_455_6e-6;
/// Boltzmann constant in electronvolts per kelvin (`k_B / e`).
pub const KB_EV: f64 = BOLTZMANN_CONSTANT_J_PER_K / ELEMENTARY_CHARGE_C;
/// Electron mass in daltons (`m_e / u`, CODATA 2018).
pub const ELECTRON_MASS_IN_DALTON: f64 = 5.485_799_090_65e-4;
/// Unified atomic mass unit (dalton) in electron masses.
pub const DALTON_IN_ELECTRON_MASSES: f64 = 1.0 / ELECTRON_MASS_IN_DALTON;
/// Atomic unit of time in seconds (`hbar / E_h`).
pub const ATOMIC_UNIT_OF_TIME_S: f64 = REDUCED_PLANCK_CONSTANT_J_S / HARTREE_ENERGY_J;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_constants_match_their_definitions() {
        assert!(KB_HARTREE > 0.0 && KB_HARTREE.is_finite());
        assert!(KB_EV > 0.0 && KB_EV.is_finite());
        assert!(ELECTRON_MASS_IN_DALTON > 0.0 && ELECTRON_MASS_IN_DALTON.is_finite());
        assert!(BOHR_TO_ANGSTROM > 0.0 && BOHR_TO_ANGSTROM.is_finite());
        assert!(HARTREE_TO_EV > 0.0 && HARTREE_TO_EV.is_finite());
        assert!(DALTON_IN_ELECTRON_MASSES > 0.0 && DALTON_IN_ELECTRON_MASSES.is_finite());
        assert!(ATOMIC_UNIT_OF_TIME_S > 0.0 && ATOMIC_UNIT_OF_TIME_S.is_finite());
        assert_eq!(ANGSTROM_TO_BOHR, 1.0 / BOHR_TO_ANGSTROM);
        assert_eq!(EV_TO_HARTREE, 1.0 / HARTREE_TO_EV);
        assert_eq!(
            KB_EV.to_bits(),
            (BOLTZMANN_CONSTANT_J_PER_K / ELEMENTARY_CHARGE_C).to_bits()
        );
        assert_eq!(
            ATOMIC_UNIT_OF_TIME_S.to_bits(),
            (REDUCED_PLANCK_CONSTANT_J_S / HARTREE_ENERGY_J).to_bits()
        );
        assert_eq!(
            DALTON_IN_ELECTRON_MASSES.to_bits(),
            (1.0 / ELECTRON_MASS_IN_DALTON).to_bits()
        );
    }
}
