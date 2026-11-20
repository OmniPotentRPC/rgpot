// MIT License
// Copyright 2023--present rgpot developers

//! Crate-level errors for safe Rust entry points.

use std::fmt;

use crate::eam_table::EamTableError;

/// Errors returned by high-level Rust constructors.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RgpotError {
    /// The EAM table input could not be parsed or evaluated.
    EamTable(EamTableError),
    /// An argument is currently unsupported by this constructor.
    Unsupported(String),
}

impl fmt::Display for RgpotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EamTable(why) => write!(f, "{why}"),
            Self::Unsupported(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for RgpotError {}

impl From<EamTableError> for RgpotError {
    fn from(value: EamTableError) -> Self {
        Self::EamTable(value)
    }
}
