// MIT License
// Copyright 2023--present rgpot developers

//! Configuration for the 12-6 Lennard-Jones potential and its
//! `PotentialConfig.lj` wire representation.

use std::fmt;

/// Parameters for the 12-6 Lennard-Jones potential (`LJPot`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct LjConfig {
    /// Well depth `u0` in eV.
    pub u0: f64,
    /// Zero-crossing distance `psi` in Angstrom.
    pub psi: f64,
    /// Interaction cutoff distance in Angstrom.
    pub cutoff: f64,
    /// C2 quintic switching window width in Angstrom (`0.0` disables switching).
    pub switch_width: f64,
    /// Add the isotropic 3D continuum long-range energy and pressure tail
    /// correction and leave the pair kernel unshifted.
    pub tail_correction: bool,
}

impl Default for LjConfig {
    fn default() -> Self {
        Self {
            u0: 1.0,
            psi: 1.0,
            cutoff: 15.0,
            switch_width: 0.0,
            tail_correction: false,
        }
    }
}

/// Refusal when an [`LjConfig`] or wire `PotentialConfig` is invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LjConfigError {
    /// `u0`, `psi`, or `cutoff` is not finite and positive.
    InvalidParameters,
    /// `switch_width` is not in `[0, cutoff]`.
    InvalidSwitchWidth,
    /// Tail correction with a switch requires a positive switch-on radius.
    SwitchCoversWholeCutoff,
    /// The current wire schema has no field for a nonzero switching width.
    WireSwitchWidthUnsupported,
    /// A `PotentialConfig` message carried an arm other than `lj`.
    WrongArm,
}

impl fmt::Display for LjConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameters => {
                write!(f, "LJ u0, psi and cutoff must be finite and positive")
            }
            Self::InvalidSwitchWidth => {
                write!(f, "LJ switch_width must be finite and in [0, cutoff]")
            }
            Self::SwitchCoversWholeCutoff => write!(
                f,
                "LJ tail_correction requires switch_width < cutoff so r_on > 0"
            ),
            Self::WireSwitchWidthUnsupported => {
                write!(f, "LJ switch_width is not representable on the wire")
            }
            Self::WrongArm => write!(f, "PotentialConfig does not select the lj arm"),
        }
    }
}

impl std::error::Error for LjConfigError {}

impl LjConfig {
    #[must_use]
    pub fn with_u0(mut self, u0: f64) -> Self {
        self.u0 = u0;
        self
    }

    #[must_use]
    pub fn with_psi(mut self, psi: f64) -> Self {
        self.psi = psi;
        self
    }

    #[must_use]
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff;
        self
    }

    #[must_use]
    pub fn with_switch_width(mut self, switch_width: f64) -> Self {
        self.switch_width = switch_width;
        self
    }

    #[must_use]
    pub fn with_tail_correction(mut self, tail_correction: bool) -> Self {
        self.tail_correction = tail_correction;
        self
    }

    /// Check that all parameters satisfy `LJPot` invariants.
    pub fn validate(&self) -> Result<(), LjConfigError> {
        if !self.u0.is_finite()
            || !self.psi.is_finite()
            || !self.cutoff.is_finite()
            || self.u0 <= 0.0
            || self.psi <= 0.0
            || self.cutoff <= 0.0
        {
            return Err(LjConfigError::InvalidParameters);
        }
        if !self.switch_width.is_finite()
            || self.switch_width < 0.0
            || self.switch_width > self.cutoff
        {
            return Err(LjConfigError::InvalidSwitchWidth);
        }
        if self.tail_correction && self.switch_width > 0.0 && self.switch_width >= self.cutoff {
            return Err(LjConfigError::SwitchCoversWholeCutoff);
        }
        Ok(())
    }

    /// Populate a Cap'n Proto `LennardJonesParams` builder after validating.
    #[cfg(feature = "schema")]
    pub fn write_params(
        &self,
        mut builder: crate::Potentials_capnp::lennard_jones_params::Builder<'_>,
    ) -> Result<(), LjConfigError> {
        self.validate()?;
        if self.switch_width != 0.0 {
            return Err(LjConfigError::WireSwitchWidthUnsupported);
        }
        builder.set_u0(self.u0);
        builder.set_psi(self.psi);
        builder.set_cutoff(self.cutoff);
        builder.set_tail_correction(self.tail_correction);
        Ok(())
    }

    /// Populate the `lj` arm of a Cap'n Proto `PotentialConfig` builder after validating.
    #[cfg(feature = "schema")]
    pub fn write_potential_config(
        &self,
        builder: crate::Potentials_capnp::potential_config::Builder<'_>,
    ) -> Result<(), LjConfigError> {
        self.validate()?;
        if self.switch_width != 0.0 {
            return Err(LjConfigError::WireSwitchWidthUnsupported);
        }
        let lj = builder.init_lj();
        self.write_params(lj)
    }

    /// Decode and validate a Cap'n Proto `LennardJonesParams` reader.
    #[cfg(feature = "schema")]
    pub fn from_params(
        reader: crate::Potentials_capnp::lennard_jones_params::Reader<'_>,
    ) -> Result<Self, LjConfigError> {
        let cfg = Self::default()
            .with_u0(reader.get_u0())
            .with_psi(reader.get_psi())
            .with_cutoff(reader.get_cutoff())
            .with_tail_correction(reader.get_tail_correction());
        cfg.validate()?;
        Ok(cfg)
    }

    /// Decode and validate the `lj` arm of a Cap'n Proto `PotentialConfig` reader.
    #[cfg(feature = "schema")]
    pub fn from_potential_config(
        reader: crate::Potentials_capnp::potential_config::Reader<'_>,
    ) -> Result<Self, LjConfigError> {
        use crate::Potentials_capnp::potential_config::Which;
        match reader.which() {
            Ok(Which::Lj(Ok(lj))) => Self::from_params(lj),
            _ => Err(LjConfigError::WrongArm),
        }
    }
}
