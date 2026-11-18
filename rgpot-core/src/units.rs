// MIT License
// Copyright 2023--present rgpot developers

#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

//! CODATA 2018 physical constants and unit conversions from
//! `CppCore/rgpot/units.hpp`.
//!
//! `build.rs` compiles `CppCore/rgpot/units.hpp` together with
//! `CppCore/rgpot/units.cc` and writes the exact IEEE-754 `f64` bit patterns
//! of the header constants into this module, keeping `units.hpp` as the
//! single table shared by C++ and Rust consumers.
//!
//! ```
//! use rgpot_core::units;
//!
//! assert_eq!(units::ANGSTROM_TO_BOHR, 1.0 / units::BOHR_TO_ANGSTROM);
//! assert_eq!(units::EV_TO_HARTREE, 1.0 / units::HARTREE_TO_EV);
//! ```

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
