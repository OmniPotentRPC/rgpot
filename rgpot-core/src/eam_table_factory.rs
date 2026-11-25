// MIT License
// Copyright 2023--present rgpot developers

//! Safe Rust factory constructors for owned EAM table potentials.

use std::path::Path;

use crate::eam_table::{EamTable, EamTableFormat};
use crate::eam_table_cpp::{cpp_potential_from_data, cpp_potential_from_file};
use crate::error::RgpotError;
use crate::potential::PotentialImpl;

/// Options of the owned EAM potential factory.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EamTableFactoryOptions {
    /// Keep the nearest image of each pair when evaluating.
    pub periodic: bool,
    /// Verlet skin for neighbour-list rebuilds.
    pub skin: f64,
}

impl Default for EamTableFactoryOptions {
    fn default() -> Self {
        Self {
            periodic: true,
            skin: 1.0,
        }
    }
}

impl EamTableFactoryOptions {
    /// Set whether periodic boundary conditions are applied during evaluation.
    #[must_use]
    pub fn with_periodic(mut self, periodic: bool) -> Self {
        self.periodic = periodic;
        self
    }

    /// Set the Verlet neighbour-list skin distance in Angstrom.
    #[must_use]
    pub fn with_skin(mut self, skin: f64) -> Self {
        self.skin = skin;
        self
    }
}

fn validate(options: EamTableFactoryOptions) -> Result<(), RgpotError> {
    if !options.skin.is_finite() || options.skin < 0.0 {
        return Err(RgpotError::Unsupported(
            "EAM table factory: skin must be finite and non-negative".into(),
        ));
    }
    Ok(())
}

/// Build an owned EAM potential handle from a table file.
pub fn eam_table_from_file(
    path: impl AsRef<Path>,
    format: EamTableFormat,
    periodic: bool,
    skin: f64,
) -> Result<PotentialImpl, RgpotError> {
    eam_table_from_file_with(
        path,
        format,
        EamTableFactoryOptions::default()
            .with_periodic(periodic)
            .with_skin(skin),
    )
}

/// Build an owned EAM potential handle from a table file with options.
pub fn eam_table_from_file_with(
    path: impl AsRef<Path>,
    format: EamTableFormat,
    options: EamTableFactoryOptions,
) -> Result<PotentialImpl, RgpotError> {
    validate(options)?;
    cpp_potential_from_file(path.as_ref(), format, options.periodic, options.skin)
}

/// Build an owned EAM potential handle from an in-memory table.
pub fn eam_table_from_data(
    table: EamTable,
    periodic: bool,
    skin: f64,
) -> Result<PotentialImpl, RgpotError> {
    eam_table_from_data_with(
        table,
        EamTableFactoryOptions::default()
            .with_periodic(periodic)
            .with_skin(skin),
    )
}

/// Build an owned EAM potential handle from an in-memory table with options.
pub fn eam_table_from_data_with(
    table: EamTable,
    options: EamTableFactoryOptions,
) -> Result<PotentialImpl, RgpotError> {
    validate(options)?;
    cpp_potential_from_data(table, options.periodic, options.skin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eam_table_cpp::{cpp_drop_count, cpp_reset_drop_count, CPP_TEST_LOCK};
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../CppCore/tests/data/eam_table")
    }

    #[test]
    fn drop_frees_cpp_potential_exactly_once() {
        let _lock = CPP_TEST_LOCK.lock().unwrap();
        cpp_reset_drop_count();
        {
            let _potential = eam_table_from_file(
                data_dir().join("Cu_u3.eam"),
                EamTableFormat::Funcfl,
                true,
                1.0,
            )
            .expect("construct C++ EAM table potential");
            assert_eq!(cpp_drop_count(), 0);
        }
        assert_eq!(cpp_drop_count(), 1);
    }
}
