// MIT License
// Copyright 2023--present rgpot developers

//! Physical constants from `CppCore/rgpot/units.hpp`.
//!
//! `build.rs` compiles that header and writes these values. Boltzmann's
//! constant, the bohr and hartree conversions are the compile-time
//! constants. The dalton in electron masses is
//! `unit_conversion_factor("dalton", "electron_mass")`, and the atomic
//! unit of time is `unit_conversion_factor("hbar", "hartree*s")`.

include!(concat!(env!("OUT_DIR"), "/units_generated.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_constants_match_their_definitions() {
        assert!(KB_HARTREE > 0.0 && KB_HARTREE.is_finite());
        assert!(KB_EV > 0.0 && KB_EV.is_finite());
        assert!(BOHR_TO_ANGSTROM > 0.0 && BOHR_TO_ANGSTROM.is_finite());
        assert!(HARTREE_TO_EV > 0.0 && HARTREE_TO_EV.is_finite());
        assert!(DALTON_IN_ELECTRON_MASSES > 0.0 && DALTON_IN_ELECTRON_MASSES.is_finite());
        assert!(ATOMIC_UNIT_OF_TIME_S > 0.0 && ATOMIC_UNIT_OF_TIME_S.is_finite());
        assert_eq!(ANGSTROM_TO_BOHR, 1.0 / BOHR_TO_ANGSTROM);
        assert_eq!(EV_TO_HARTREE, 1.0 / HARTREE_TO_EV);
    }
}
