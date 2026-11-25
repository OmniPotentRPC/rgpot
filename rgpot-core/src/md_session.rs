// MIT License
// Copyright 2023--present rgpot developers

//! Engine-neutral molecular-dynamics session contract.
//!
//! All quantities cross [`MdSession`] and [`CAbiMdSession`] in rgpot units:
//! positions and plane/wall coordinates in Angstrom, velocities in
//! Angstrom/ps, forces in eV/Angstrom, energies and `kT` in eV, time step in
//! ps, friction in inverse ps, and masses in amu. Flat arrays contain
//! `3 * n_atoms` entries in row-major `(x, y, z)` order.

use std::ffi::{c_char, c_void, CStr};
use std::fmt;
use std::ptr::NonNull;

/// Kilojoules per mole in one electronvolt (exact 2019 SI constants).
pub const KJ_PER_EV: f64 = 1.602_176_634e-19 * 6.022_140_76e23 / 1000.0;

/// Nanometres in one Angstrom.
pub const NM_PER_ANGSTROM: f64 = 0.1;

/// Conversion factor from `eV` to `amu * (Angstrom / ps)^2`.
pub const AMU_A2_PS2_PER_EV: f64 = KJ_PER_EV * 100.0;

/// What an [`MdSessionError`] reports.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MdSessionErrorKind {
    /// The engine or an argument check refused the call.
    Failed,
    /// The engine's operation table has a null required callback, named here.
    MissingOperation(&'static str),
}

/// Error raised by an MD session or its C ABI boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MdSessionError {
    kind: MdSessionErrorKind,
    message: String,
}

impl MdSessionError {
    /// Construct an MD session error with a diagnostic message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            kind: MdSessionErrorKind::Failed,
            message: message.into(),
        }
    }

    /// The error for a null required callback `slot` of [`MdSessionOps`].
    pub fn missing_operation(slot: &'static str) -> Self {
        Self {
            kind: MdSessionErrorKind::MissingOperation(slot),
            message: format!("MD session operation table has a null {slot}"),
        }
    }

    /// What this error reports.
    pub fn kind(&self) -> MdSessionErrorKind {
        self.kind
    }
}

impl fmt::Display for MdSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MdSessionError {}

/// Controls for one projected BAOAB Langevin integration call.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MdLangevinConfig {
    /// Thermal energy `kT` in eV.
    pub k_t: f64,
    /// Langevin friction in inverse picoseconds.
    pub friction: f64,
    /// Integration time step in picoseconds.
    pub time_step: f64,
    /// Name of the random stream.
    ///
    /// A call with the seed of the previous call continues that stream, so a
    /// run split in chunks equals one long run; a new seed starts a new
    /// stream. Setting positions or velocities ends the stream, so the same
    /// seed after a new state replays the same noise.
    pub seed: u64,
    /// How a drift that ends outside a wall is handled by
    /// [`MdSession::run_langevin`].
    pub wall_mode: MdWallMode,
    /// Record the constraint force of every step in
    /// [`MdLangevinOutcome::constraint_force`].
    pub record_constraint_force: bool,
}

impl Default for MdLangevinConfig {
    fn default() -> Self {
        Self {
            k_t: 0.02585,
            friction: 1.0,
            time_step: 0.001,
            seed: 1,
            wall_mode: MdWallMode::Reflect,
            record_constraint_force: false,
        }
    }
}

/// How a drift that ends outside a wall is handled.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MdWallMode {
    /// Mirror the position and reflect the velocity in the mass-weighted
    /// metric.
    #[default]
    Reflect,
    /// Undo the drift and reflect the velocity.
    Reject,
}

impl MdWallMode {
    fn to_c(self) -> i32 {
        match self {
            Self::Reflect => 0,
            Self::Reject => 1,
        }
    }
}

/// What [`MdSession::run_langevin`] reports.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MdLangevinOutcome {
    /// Number of wall reflections.
    pub wall_hits: usize,
    /// One Lagrange multiplier per step when recorded and a plane is given.
    ///
    /// Entry `k` is `lambda = -(n . M^-1 F) / (n . M^-1 n)` in eV/Angstrom for
    /// the unit normal `n` over the `3N` coordinates, the mean of the two force
    /// kicks of step `k`. The constraint force is `lambda n`, and its mean
    /// over the constrained ensemble is `dF/dc` for the plane offset
    /// `c = n . p`.
    pub constraint_force: Vec<f64>,
}

impl MdLangevinConfig {
    /// Set [`Self::k_t`] in eV.
    #[must_use]
    pub fn with_k_t(mut self, k_t: f64) -> Self {
        self.k_t = k_t;
        self
    }

    /// Set [`Self::friction`] in inverse picoseconds.
    #[must_use]
    pub fn with_friction(mut self, friction: f64) -> Self {
        self.friction = friction;
        self
    }

    /// Set [`Self::time_step`] in picoseconds.
    #[must_use]
    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.time_step = time_step;
        self
    }

    /// Set [`Self::seed`].
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set [`Self::wall_mode`].
    #[must_use]
    pub fn with_wall_mode(mut self, wall_mode: MdWallMode) -> Self {
        self.wall_mode = wall_mode;
        self
    }

    /// Set [`Self::record_constraint_force`].
    #[must_use]
    pub fn with_record_constraint_force(mut self, record: bool) -> Self {
        self.record_constraint_force = record;
        self
    }
}

/// SplitMix64 with Box-Muller normals, the stream every engine shares.
///
/// It matches `rgpot::md::SplitMix64` in `ProjectedLangevin.hpp` draw for draw.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
    spare: Option<f64>,
}

impl SplitMix64 {
    /// Start a stream from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in the open interval `(0, 1)`.
    pub fn next_uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// A standard normal draw.
    pub fn next_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

/// Maxwell-Boltzmann velocities in Angstrom/ps for `masses` in amu.
///
/// `k_t` is in eV. With `plane_normal` (`3 * n_atoms` entries) the draw is
/// conditioned on `n . v = 0`. A particle of zero mass is at rest. The draw
/// equals the one `rgpot_md_session_reseed_velocities` makes for the same
/// arguments.
pub fn maxwell_boltzmann_velocities(
    masses: &[f64],
    k_t: f64,
    seed: u64,
    plane_normal: Option<&[f64]>,
) -> Result<Vec<f64>, MdSessionError> {
    if !(k_t.is_finite() && k_t >= 0.0) {
        return Err(MdSessionError::new("kT must be finite and not negative"));
    }
    if let Some(normal) = plane_normal {
        if normal.len() != 3 * masses.len() {
            return Err(MdSessionError::new(
                "plane normal length must be 3 * n_atoms",
            ));
        }
        check_finite_slice(normal, "plane_normal")?;
    }
    let k_t = k_t * AMU_A2_PS2_PER_EV;
    let mut rng = SplitMix64::new(seed);
    let mut velocity = vec![0.0; 3 * masses.len()];
    for (atom, &mass) in masses.iter().enumerate() {
        if mass > 0.0 {
            for axis in 0..3 {
                velocity[3 * atom + axis] = (k_t / mass).sqrt() * rng.next_normal();
            }
        }
    }
    if let Some(normal) = plane_normal {
        let mut along = 0.0;
        let mut weighted = 0.0;
        for (atom, &mass) in masses.iter().enumerate() {
            if mass > 0.0 {
                for axis in 0..3 {
                    let k = 3 * atom + axis;
                    along += normal[k] * velocity[k];
                    weighted += normal[k] * normal[k] / mass;
                }
            }
        }
        if !(weighted > 0.0 && weighted.is_finite()) {
            return Err(MdSessionError::new(
                "the normal must be finite and act on a massive coordinate",
            ));
        }
        let scale = along / weighted;
        for (atom, &mass) in masses.iter().enumerate() {
            if mass > 0.0 {
                for axis in 0..3 {
                    let k = 3 * atom + axis;
                    velocity[k] -= scale * normal[k] / mass;
                }
            }
        }
    }
    Ok(velocity)
}

/// State returned by an rgpot MD session, in rgpot units.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MdSessionState {
    /// Flat xyz positions in Angstrom.
    pub positions: Vec<f64>,
    /// Flat xyz velocities in Angstrom per picosecond.
    pub velocities: Vec<f64>,
    /// Flat xyz forces in eV per Angstrom.
    pub forces: Vec<f64>,
    /// Potential energy in eV.
    pub potential_energy: f64,
    /// Kinetic energy in eV.
    pub kinetic_energy: f64,
}

impl MdSessionState {
    /// Set [`Self::positions`].
    #[must_use]
    pub fn with_positions(mut self, positions: Vec<f64>) -> Self {
        self.positions = positions;
        self
    }

    /// Set [`Self::velocities`].
    #[must_use]
    pub fn with_velocities(mut self, velocities: Vec<f64>) -> Self {
        self.velocities = velocities;
        self
    }

    /// Set [`Self::forces`].
    #[must_use]
    pub fn with_forces(mut self, forces: Vec<f64>) -> Self {
        self.forces = forces;
        self
    }

    /// Set [`Self::potential_energy`].
    #[must_use]
    pub fn with_potential_energy(mut self, potential_energy: f64) -> Self {
        self.potential_energy = potential_energy;
        self
    }

    /// Set [`Self::kinetic_energy`].
    #[must_use]
    pub fn with_kinetic_energy(mut self, kinetic_energy: f64) -> Self {
        self.kinetic_energy = kinetic_energy;
        self
    }
}

/// Neighbour query request ABI shared with `rgpot_nlist_request`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MdNlistRequest {
    pub bytes: u32,
    pub abi: u32,
    pub cutoff: f64,
    pub skin: f64,
    pub shape: i32,
    pub shell: i32,
    pub cut: i32,
    pub periodic: [i32; 3],
    pub want_shifts: i32,
    pub want_vectors: i32,
    pub want_distances: i32,
}

/// `RGPOT_NLIST_ABI` in `rgpot_nlist.h`.
const NLIST_REQUEST_ABI: u32 = 4;

impl MdNlistRequest {
    /// A request the neighbour library accepts.
    ///
    /// `bytes` is the size of this struct and `abi` is `RGPOT_NLIST_ABI`.
    /// [`Default`] leaves both at zero, and the engine refuses that request.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bytes: std::mem::size_of::<Self>() as u32,
            abi: NLIST_REQUEST_ABI,
            ..Self::default()
        }
    }
}

/// Structure-of-arrays pair buffers shared with `rgpot_nlist_soa`.
///
/// This is the raw layout handed to the engine callbacks. Safe callers pass
/// [`MdNlistBuffers`] or [`MdNlistPairs`] to [`MdSession`] instead, which
/// build it from slices of checked length.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MdNlistSoa {
    pub bytes: u32,
    pub first: *mut i32,
    pub second: *mut i32,
    pub shift: *mut i32,
    pub vector: *mut f64,
    pub distance: *mut f64,
    pub mask: *mut u8,
    pub cap: usize,
}

/// Caller-owned columns that [`MdSession::get_neighbors`] fills.
///
/// `first` and `second` hold one entry per pair and their common length is
/// the capacity. `shift` and `vector` hold three entries per pair, `distance`
/// and `mask` one. A column left `None` is not written; a request that asks
/// for it, or a cluster request without `mask`, is refused before the engine
/// is called.
#[derive(Debug)]
pub struct MdNlistBuffers<'a> {
    /// First atom of each pair.
    pub first: &'a mut [i32],
    /// Second atom of each pair.
    pub second: &'a mut [i32],
    /// Integer image shift of each pair, three per pair.
    pub shift: Option<&'a mut [i32]>,
    /// Pair vector in Angstrom, three per pair.
    pub vector: Option<&'a mut [f64]>,
    /// Pair distance in Angstrom.
    pub distance: Option<&'a mut [f64]>,
    /// Cluster mask.
    pub mask: Option<&'a mut [u8]>,
}

impl<'a> MdNlistBuffers<'a> {
    /// Buffers with the atom columns only.
    #[must_use]
    pub fn new(first: &'a mut [i32], second: &'a mut [i32]) -> Self {
        Self {
            first,
            second,
            shift: None,
            vector: None,
            distance: None,
            mask: None,
        }
    }

    /// Set [`Self::shift`].
    #[must_use]
    pub fn with_shift(mut self, shift: &'a mut [i32]) -> Self {
        self.shift = Some(shift);
        self
    }

    /// Set [`Self::vector`].
    #[must_use]
    pub fn with_vector(mut self, vector: &'a mut [f64]) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Set [`Self::distance`].
    #[must_use]
    pub fn with_distance(mut self, distance: &'a mut [f64]) -> Self {
        self.distance = Some(distance);
        self
    }

    /// Set [`Self::mask`].
    #[must_use]
    pub fn with_mask(mut self, mask: &'a mut [u8]) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Number of pairs the buffers hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.first.len()
    }

    fn as_raw(&mut self, request: &MdNlistRequest) -> Result<MdNlistSoa, MdSessionError> {
        let cap = self.capacity();
        check_nlist_columns(
            request,
            cap,
            self.second.len(),
            self.shift.as_deref().map(<[i32]>::len),
            self.vector.as_deref().map(<[f64]>::len),
            self.distance.as_deref().map(<[f64]>::len),
            self.mask.as_deref().map(<[u8]>::len),
        )?;
        Ok(MdNlistSoa {
            bytes: NLIST_SOA_BYTES,
            first: self.first.as_mut_ptr(),
            second: self.second.as_mut_ptr(),
            shift: self
                .shift
                .as_deref_mut()
                .map_or(std::ptr::null_mut(), <[i32]>::as_mut_ptr),
            vector: self
                .vector
                .as_deref_mut()
                .map_or(std::ptr::null_mut(), <[f64]>::as_mut_ptr),
            distance: self
                .distance
                .as_deref_mut()
                .map_or(std::ptr::null_mut(), <[f64]>::as_mut_ptr),
            mask: self
                .mask
                .as_deref_mut()
                .map_or(std::ptr::null_mut(), <[u8]>::as_mut_ptr),
            cap,
        })
    }
}

/// Host-built pairs that [`MdSession::set_neighbors`] hands to the engine.
///
/// The columns follow [`MdNlistBuffers`]; the common length of `first` and
/// `second` is the pair count.
#[derive(Clone, Copy, Debug)]
pub struct MdNlistPairs<'a> {
    /// First atom of each pair.
    pub first: &'a [i32],
    /// Second atom of each pair.
    pub second: &'a [i32],
    /// Integer image shift of each pair, three per pair.
    pub shift: Option<&'a [i32]>,
    /// Pair vector in Angstrom, three per pair.
    pub vector: Option<&'a [f64]>,
    /// Pair distance in Angstrom.
    pub distance: Option<&'a [f64]>,
    /// Cluster mask.
    pub mask: Option<&'a [u8]>,
}

impl<'a> MdNlistPairs<'a> {
    /// Pairs with the atom columns only.
    #[must_use]
    pub fn new(first: &'a [i32], second: &'a [i32]) -> Self {
        Self {
            first,
            second,
            shift: None,
            vector: None,
            distance: None,
            mask: None,
        }
    }

    /// Set [`Self::shift`].
    #[must_use]
    pub fn with_shift(mut self, shift: &'a [i32]) -> Self {
        self.shift = Some(shift);
        self
    }

    /// Set [`Self::vector`].
    #[must_use]
    pub fn with_vector(mut self, vector: &'a [f64]) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Set [`Self::distance`].
    #[must_use]
    pub fn with_distance(mut self, distance: &'a [f64]) -> Self {
        self.distance = Some(distance);
        self
    }

    /// Set [`Self::mask`].
    #[must_use]
    pub fn with_mask(mut self, mask: &'a [u8]) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Number of pairs.
    #[must_use]
    pub fn count(&self) -> usize {
        self.first.len()
    }

    /// The engine reads these columns and does not write them, so the
    /// mutable pointers of the raw layout never carry a write.
    fn as_raw(&self, request: &MdNlistRequest) -> Result<MdNlistSoa, MdSessionError> {
        let count = self.count();
        check_nlist_columns(
            request,
            count,
            self.second.len(),
            self.shift.map(<[i32]>::len),
            self.vector.map(<[f64]>::len),
            self.distance.map(<[f64]>::len),
            self.mask.map(<[u8]>::len),
        )?;
        Ok(MdNlistSoa {
            bytes: NLIST_SOA_BYTES,
            first: self.first.as_ptr().cast_mut(),
            second: self.second.as_ptr().cast_mut(),
            shift: self
                .shift
                .map_or(std::ptr::null_mut(), |v| v.as_ptr().cast_mut()),
            vector: self
                .vector
                .map_or(std::ptr::null_mut(), |v| v.as_ptr().cast_mut()),
            distance: self
                .distance
                .map_or(std::ptr::null_mut(), |v| v.as_ptr().cast_mut()),
            mask: self
                .mask
                .map_or(std::ptr::null_mut(), |v| v.as_ptr().cast_mut()),
            cap: count,
        })
    }
}

const NLIST_SOA_BYTES: u32 = std::mem::size_of::<MdNlistSoa>() as u32;

/// Check the column lengths against `pairs` and the columns `request` asks
/// for.
fn check_nlist_columns(
    request: &MdNlistRequest,
    pairs: usize,
    second: usize,
    shift: Option<usize>,
    vector: Option<usize>,
    distance: Option<usize>,
    mask: Option<usize>,
) -> Result<(), MdSessionError> {
    if second != pairs {
        return Err(MdSessionError::new(format!(
            "neighbour column second holds {second} entries for {pairs} pairs"
        )));
    }
    let triple = pairs
        .checked_mul(3)
        .ok_or_else(|| MdSessionError::new("neighbour pair count overflows"))?;
    let columns = [
        ("shift", shift, triple, request.want_shifts != 0),
        ("vector", vector, triple, request.want_vectors != 0),
        ("distance", distance, pairs, request.want_distances != 0),
        ("mask", mask, pairs, request.shape == 1),
    ];
    for (name, len, expected, wanted) in columns {
        match len {
            Some(len) if len != expected => {
                return Err(MdSessionError::new(format!(
                    "neighbour column {name} holds {len} entries, expected {expected}"
                )));
            }
            None if wanted && pairs > 0 => {
                return Err(MdSessionError::new(format!(
                    "the neighbour request needs the {name} column"
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Return statuses for neighbour exchange callbacks.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MdNlistStatus {
    Ok = 0,
    Error = 1,
    Unsupported = 2,
}

impl MdNlistStatus {
    fn from_raw(value: i32) -> Self {
        match value {
            0 => Self::Ok,
            2 => Self::Unsupported,
            _ => Self::Error,
        }
    }
}

/// Safe Rust contract implemented by C ABI-backed rgpot engines.
pub trait MdSession {
    /// Number of Cartesian coordinates (`3 * n_atoms`).
    fn dof(&self) -> usize;

    /// Particle masses in amu (`n_atoms` entries).
    fn masses(&self) -> Result<Vec<f64>, MdSessionError>;

    /// Replace the current flat xyz positions in Angstrom.
    fn set_positions(&mut self, positions: &[f64]) -> Result<(), MdSessionError>;

    /// Replace the current flat xyz velocities in Angstrom per picosecond.
    fn set_velocities(&mut self, velocities: &[f64]) -> Result<(), MdSessionError>;

    /// Advance projected BAOAB Langevin dynamics and return wall reflection hits.
    fn step_langevin(
        &mut self,
        config: &MdLangevinConfig,
        plane_point: &[f64],
        plane_normal: &[f64],
        wall_points: &[f64],
        wall_normals: &[f64],
        steps: usize,
    ) -> Result<usize, MdSessionError>;

    /// Advance deterministic unconstrained velocity-Verlet NVE dynamics.
    fn step_nve(&mut self, time_step: f64, steps: usize) -> Result<(), MdSessionError>;

    /// Read positions, velocities, forces, and energies.
    fn state(&mut self) -> Result<MdSessionState, MdSessionError>;

    /// Set the periodic box, three lattice vectors as rows, in Angstrom.
    fn set_periodic_box(&mut self, _box9: &[f64; 9]) -> Result<(), MdSessionError> {
        Err(MdSessionError::new(
            "MD engine does not support set_periodic_box",
        ))
    }

    /// Read the periodic box, three lattice vectors as rows, in Angstrom.
    fn periodic_box(&mut self) -> Result<[f64; 9], MdSessionError> {
        Err(MdSessionError::new(
            "MD engine does not support get_periodic_box",
        ))
    }

    /// Evaluate the first forces once the box and positions are set.
    fn initialize(&mut self) -> Result<(), MdSessionError> {
        Ok(())
    }

    /// Advance Langevin dynamics, optionally on a hyperplane and between
    /// walls, and report the wall hits and the constraint force per step.
    ///
    /// `plane` is the point and the normal of the hyperplane, `3 * n_atoms`
    /// entries each, or `None` for unconstrained dynamics.
    fn run_langevin(
        &mut self,
        _config: &MdLangevinConfig,
        _plane: Option<(&[f64], &[f64])>,
        _wall_points: &[f64],
        _wall_normals: &[f64],
        _steps: usize,
    ) -> Result<MdLangevinOutcome, MdSessionError> {
        Err(MdSessionError::new(
            "MD engine does not support run_langevin",
        ))
    }

    /// Draw Maxwell-Boltzmann velocities at `k_t` (eV) and set them.
    ///
    /// With `plane_normal` the draw is conditioned on `n . v = 0`. The draw
    /// is made here from the masses, so it is the same on every engine.
    fn reseed_velocities(
        &mut self,
        k_t: f64,
        seed: u64,
        plane_normal: Option<&[f64]>,
    ) -> Result<(), MdSessionError> {
        let masses = self.masses()?;
        let velocities = maxwell_boltzmann_velocities(&masses, k_t, seed, plane_normal)?;
        self.set_velocities(&velocities)
    }

    /// Query the engine neighbour list for the supplied request.
    ///
    /// Returns the number of pairs written into `out` and the number the
    /// full list holds. With `out` of `None` nothing is written and only the
    /// full number is reported. A buffer shorter than that list is an error.
    /// The message names the pair count the caller must provide. The engine
    /// may already have written the pairs that fit.
    fn get_neighbors(
        &mut self,
        _request: &MdNlistRequest,
        _out: Option<MdNlistBuffers<'_>>,
    ) -> Result<(usize, usize), MdSessionError> {
        Err(MdSessionError::new(
            "MD engine does not expose neighbor queries",
        ))
    }

    /// Provide neighbour pairs built by the host.
    fn set_neighbors(
        &mut self,
        _request: &MdNlistRequest,
        _pairs: &MdNlistPairs<'_>,
    ) -> Result<(), MdSessionError> {
        Err(MdSessionError::new(
            "MD engine does not accept supplied neighbors",
        ))
    }
}

/// C ABI request matching `rgpot_md_langevin_request`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MdLangevinRequest {
    /// `size_of::<MdLangevinRequest>()`.
    pub struct_size: usize,
    /// Thermal energy in eV.
    pub k_t: f64,
    /// Friction in inverse ps.
    pub friction: f64,
    /// Time step in ps.
    pub time_step: f64,
    /// Number of steps.
    pub steps: usize,
    /// Random stream seed.
    pub seed: u64,
    /// `3 * n_atoms`.
    pub n3: usize,
    /// Plane point in Angstrom, or null.
    pub plane_point: *const f64,
    /// Plane normal, or null.
    pub plane_normal: *const f64,
    /// Number of walls.
    pub n_walls: usize,
    /// Wall points in Angstrom, `n_walls * n3`.
    pub wall_points: *const f64,
    /// Wall normals, `n_walls * n3`.
    pub wall_normals: *const f64,
    /// 0 reflects, 1 rejects the drift.
    pub wall_mode: i32,
}

/// C ABI result matching `rgpot_md_langevin_result`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MdLangevinResult {
    /// `size_of::<MdLangevinResult>()`.
    pub struct_size: usize,
    /// Wall reflections.
    pub wall_hits: u64,
    /// Buffer for the constraint force per step, or null.
    pub constraint_force: *mut f64,
    /// Entries in `constraint_force`.
    pub constraint_force_len: usize,
}

/// The `step_langevin` callback of [`MdSessionOps`].
pub type MdStepLangevinFn = unsafe extern "C" fn(
    ctx: *mut c_void,
    k_t: f64,
    friction: f64,
    time_step: f64,
    plane_point: *const f64,
    plane_normal: *const f64,
    wall_points: *const f64,
    wall_normals: *const f64,
    n_walls: usize,
    n3: usize,
    steps: usize,
    seed: u64,
    wall_hits: *mut u64,
    err: *mut c_char,
    err_len: usize,
) -> i32;

/// C ABI vtable matching `rgpot_md_session_ops` in `external_potentials.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MdSessionOps {
    /// Host capacity (`std::mem::size_of::<MdSessionOps>()`) on input to
    /// `rgpot_md_session_engine_create`, and filled byte size on output.
    pub struct_size: usize,
    /// Set flat positions in Angstrom.
    pub set_positions: unsafe extern "C" fn(
        ctx: *mut c_void,
        positions: *const f64,
        n3: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32,
    /// Set flat velocities in Angstrom/ps.
    pub set_velocities: unsafe extern "C" fn(
        ctx: *mut c_void,
        velocities: *const f64,
        n3: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32,
    /// Step projected BAOAB Langevin dynamics.
    pub step_langevin: MdStepLangevinFn,
    /// Step unconstrained NVE dynamics.
    pub step_nve: unsafe extern "C" fn(
        ctx: *mut c_void,
        time_step: f64,
        steps: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32,
    /// Read positions, velocities, forces, and energies.
    pub get_state: unsafe extern "C" fn(
        ctx: *mut c_void,
        positions: *mut f64,
        velocities: *mut f64,
        forces: *mut f64,
        n3: usize,
        potential_energy: *mut f64,
        kinetic_energy: *mut f64,
        err: *mut c_char,
        err_len: usize,
    ) -> i32,
    /// Read particle masses in amu.
    pub get_masses: unsafe extern "C" fn(
        ctx: *mut c_void,
        masses: *mut f64,
        n_atoms: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32,
    /// Destroy the backend context.
    pub destroy: unsafe extern "C" fn(ctx: *mut c_void),
    /// Optional: set the periodic box in Angstrom.
    pub set_periodic_box: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            box9: *const f64,
            err: *mut c_char,
            err_len: usize,
        ) -> i32,
    >,
    /// Optional: read the periodic box in Angstrom.
    pub get_periodic_box: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            box9: *mut f64,
            err: *mut c_char,
            err_len: usize,
        ) -> i32,
    >,
    /// Optional: evaluate the first forces.
    pub initialize:
        Option<unsafe extern "C" fn(ctx: *mut c_void, err: *mut c_char, err_len: usize) -> i32>,
    /// Optional: Langevin dynamics with the constraint force per step.
    pub run_langevin: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            request: *const MdLangevinRequest,
            result: *mut MdLangevinResult,
            err: *mut c_char,
            err_len: usize,
        ) -> i32,
    >,
    /// Optional neighbour query callback.
    pub get_neighbors: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            request: *const MdNlistRequest,
            out: *mut MdNlistSoa,
            count: *mut usize,
            needed: *mut usize,
            err: *mut c_char,
            err_len: usize,
        ) -> i32,
    >,
    /// Optional neighbour setter callback.
    pub set_neighbors: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            request: *const MdNlistRequest,
            pairs: *const MdNlistSoa,
            count: usize,
            err: *mut c_char,
            err_len: usize,
        ) -> i32,
    >,
}

/// The callbacks every [`MdSessionOps`] must hold, by name and offset.
const REQUIRED_OPS: [(&str, usize); 7] = [
    (
        "set_positions",
        std::mem::offset_of!(MdSessionOps, set_positions),
    ),
    (
        "set_velocities",
        std::mem::offset_of!(MdSessionOps, set_velocities),
    ),
    (
        "step_langevin",
        std::mem::offset_of!(MdSessionOps, step_langevin),
    ),
    ("step_nve", std::mem::offset_of!(MdSessionOps, step_nve)),
    ("get_state", std::mem::offset_of!(MdSessionOps, get_state)),
    ("get_masses", std::mem::offset_of!(MdSessionOps, get_masses)),
    ("destroy", std::mem::offset_of!(MdSessionOps, destroy)),
];

/// The smallest `struct_size` that holds every required callback.
const REQUIRED_OPS_SIZE: usize = std::mem::offset_of!(MdSessionOps, destroy)
    + std::mem::size_of::<unsafe extern "C" fn(*mut c_void)>();

#[cfg(feature = "engine")]
struct EngineLibraryHandle {
    library: std::mem::ManuallyDrop<libloading::Library>,
    _pin: Option<std::mem::ManuallyDrop<libloading::Library>>,
    can_unload: bool,
}

#[cfg(feature = "engine")]
unsafe fn is_unload_safe(library: &libloading::Library) -> bool {
    use crate::engine::{
        AbiStampFn, AbiVersionFn, RgpotEngineAbiStamp, UnloadPolicyFn, ENGINE_ABI_VERSION,
        ENGINE_UNLOAD_POLICY_MINOR, ENGINE_UNLOAD_SAFE,
    };

    let version_fn = match unsafe { library.get::<AbiVersionFn>(b"rgpot_engine_abi_version\0") } {
        Ok(f) => f,
        Err(_) => return false,
    };
    let stamp_fn = match unsafe { library.get::<AbiStampFn>(b"rgpot_engine_abi_stamp\0") } {
        Ok(f) => f,
        Err(_) => return false,
    };
    let policy_fn = match unsafe { library.get::<UnloadPolicyFn>(b"rgpot_engine_unload_policy\0") }
    {
        Ok(f) => f,
        Err(_) => return false,
    };
    if unsafe { version_fn() } != ENGINE_ABI_VERSION {
        return false;
    }
    let stamp = unsafe { stamp_fn() };
    if !stamp.serves(RgpotEngineAbiStamp::HOST) || stamp.abi_minor < ENGINE_UNLOAD_POLICY_MINOR {
        return false;
    }
    unsafe { policy_fn() == ENGINE_UNLOAD_SAFE }
}

#[cfg(all(feature = "engine", unix))]
unsafe fn pin_resident_library(
    path: &std::path::Path,
    anchor: *const c_void,
) -> Result<Option<std::mem::ManuallyDrop<libloading::Library>>, libloading::Error> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_LAZY, RTLD_LOCAL};
    use std::os::unix::ffi::OsStrExt;

    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let dladdr_ok = !anchor.is_null() && unsafe { libc::dladdr(anchor, &mut info) } != 0;
    let pin_path = if dladdr_ok && !info.dli_fname.is_null() {
        let bytes = unsafe { CStr::from_ptr(info.dli_fname) }.to_bytes();
        std::path::Path::new(std::ffi::OsStr::from_bytes(bytes)).to_path_buf()
    } else {
        path.to_path_buf()
    };
    let flags = RTLD_LAZY | RTLD_LOCAL | libc::RTLD_NOLOAD | libc::RTLD_NODELETE;
    unsafe { UnixLibrary::open(Some(&pin_path), flags) }
        .map(|pin| Some(std::mem::ManuallyDrop::new(libloading::Library::from(pin))))
}

#[cfg(all(feature = "engine", not(unix)))]
unsafe fn pin_resident_library(
    _path: &std::path::Path,
    _anchor: *const c_void,
) -> Result<Option<std::mem::ManuallyDrop<libloading::Library>>, libloading::Error> {
    Ok(None)
}

/// Rust owner for an engine-agnostic C ABI MD session vtable.
pub struct CAbiMdSession {
    ctx: NonNull<c_void>,
    ops: MdSessionOps,
    n_atoms: usize,
    n3: usize,
    #[cfg(feature = "engine")]
    _library: Option<EngineLibraryHandle>,
}

impl CAbiMdSession {
    /// Wrap an existing C backend context and vtable.
    ///
    /// # Safety
    ///
    /// `ctx` must remain valid until `ops.destroy` is called on drop, and all
    /// function pointers in `ops` must uphold the C ABI contract in
    /// `external_potentials.h`. On error `ops.destroy` is not called and the
    /// caller still owns `ctx`.
    pub unsafe fn from_ops(
        ctx: NonNull<c_void>,
        ops: MdSessionOps,
        n_atoms: usize,
    ) -> Result<Self, MdSessionError> {
        if ops.struct_size < REQUIRED_OPS_SIZE {
            return Err(MdSessionError::new(
                "MD session vtable struct_size is too small",
            ));
        }
        let mut ops = ops;
        // A vtable built against an older header ends before the optional
        // callbacks; whatever the struct holds there is not a callback.
        // Clamp to our own size so an oversized struct_size cannot read past
        // our fields.
        let size = ops.struct_size.min(std::mem::size_of::<MdSessionOps>());
        ops.struct_size = size;
        let holds = |offset: usize, slot: usize| offset + slot <= size;
        if !holds(
            std::mem::offset_of!(MdSessionOps, set_periodic_box),
            std::mem::size_of_val(&ops.set_periodic_box),
        ) {
            ops.set_periodic_box = None;
        }
        if !holds(
            std::mem::offset_of!(MdSessionOps, get_periodic_box),
            std::mem::size_of_val(&ops.get_periodic_box),
        ) {
            ops.get_periodic_box = None;
        }
        if !holds(
            std::mem::offset_of!(MdSessionOps, initialize),
            std::mem::size_of_val(&ops.initialize),
        ) {
            ops.initialize = None;
        }
        if !holds(
            std::mem::offset_of!(MdSessionOps, run_langevin),
            std::mem::size_of_val(&ops.run_langevin),
        ) {
            ops.run_langevin = None;
        }
        if !holds(
            std::mem::offset_of!(MdSessionOps, get_neighbors),
            std::mem::size_of_val(&ops.get_neighbors),
        ) {
            ops.get_neighbors = None;
        }
        if !holds(
            std::mem::offset_of!(MdSessionOps, set_neighbors),
            std::mem::size_of_val(&ops.set_neighbors),
        ) {
            ops.set_neighbors = None;
        }
        let Some(n3) = n_atoms.checked_mul(3) else {
            return Err(MdSessionError::new("MD session atom count overflows"));
        };
        if n_atoms == 0 {
            return Err(MdSessionError::new("MD session requires at least one atom"));
        }
        Ok(Self {
            ctx,
            ops,
            n_atoms,
            n3,
            #[cfg(feature = "engine")]
            _library: None,
        })
    }

    /// Take ownership of the context and the operation table an engine's
    /// create call wrote.
    ///
    /// `ops` is the table the engine filled after it was zeroed, so a slot
    /// it left out reads as null. A null `ctx`, a `struct_size` that does
    /// not reach `destroy`, or a null required callback is an error, the
    /// last of kind [`MdSessionErrorKind::MissingOperation`]. Once `destroy`
    /// is known to be present, every error calls it on `ctx`, so a refused
    /// session is released. A table without `destroy` cannot release its
    /// context, and the engine leaks it.
    ///
    /// # Safety
    ///
    /// Every non-null slot of `ops` inside its `struct_size` must be a
    /// callback that upholds the C ABI contract in `external_potentials.h`
    /// for `ctx`, and `ctx` must stay valid until `destroy` is called.
    pub unsafe fn from_engine_ops(
        ctx: *mut c_void,
        ops: std::mem::MaybeUninit<MdSessionOps>,
        n_atoms: usize,
    ) -> Result<Self, MdSessionError> {
        let Some(ctx) = NonNull::new(ctx) else {
            return Err(MdSessionError::new("MD engine returned null session"));
        };
        let base = ops.as_ptr().cast::<u8>();
        // An Option of a function pointer is defined for every bit pattern,
        // null included, so each slot can be read before the table is.
        let slot = |offset: usize| unsafe {
            base.add(offset)
                .cast::<Option<unsafe extern "C" fn()>>()
                .read()
        };
        let struct_size = unsafe { base.cast::<usize>().read() };
        if struct_size < REQUIRED_OPS_SIZE {
            return Err(MdSessionError::new(
                "MD session vtable struct_size is too small",
            ));
        }
        let Some(destroy) = (unsafe {
            base.add(std::mem::offset_of!(MdSessionOps, destroy))
                .cast::<Option<unsafe extern "C" fn(*mut c_void)>>()
                .read()
        }) else {
            return Err(MdSessionError::missing_operation("destroy"));
        };
        for (name, offset) in REQUIRED_OPS {
            if slot(offset).is_none() {
                unsafe { destroy(ctx.as_ptr()) };
                return Err(MdSessionError::missing_operation(name));
            }
        }
        let ops = unsafe { ops.assume_init() };
        unsafe { Self::from_ops(ctx, ops, n_atoms) }.inspect_err(|_| unsafe {
            destroy(ctx.as_ptr());
        })
    }

    /// Dynamically load an MD engine plugin exporting `rgpot_md_session_engine_create`.
    ///
    /// Zeroes `MdSessionOps` and passes `size_of::<MdSessionOps>()` in
    /// `struct_size` as the byte capacity before calling the engine; the engine
    /// fills at most `min(capacity, sizeof(engine ops))` bytes and sets
    /// `struct_size` to the filled size.
    ///
    /// # Safety
    ///
    /// `engine_path` must point to a valid rgpot MD session plugin implementing
    /// `rgpot_md_session_engine_create_fn` and `rgpot_md_session_ops`.
    #[cfg(feature = "engine")]
    pub unsafe fn open_engine(
        engine_path: &std::path::Path,
        spec: &[u8],
        options: Option<&str>,
    ) -> Result<Self, MdSessionError> {
        type CreateFn = unsafe extern "C" fn(
            spec: *const c_void,
            spec_len: usize,
            options: *const c_char,
            ops: *mut MdSessionOps,
            ctx: *mut *mut c_void,
            n_atoms: *mut usize,
            err: *mut c_char,
            err_len: usize,
        ) -> i32;

        let library = unsafe { libloading::Library::new(engine_path) }
            .map_err(|err| MdSessionError::new(format!("failed to load MD engine: {err}")))?;
        let create = unsafe { library.get::<CreateFn>(b"rgpot_md_session_engine_create\0") }
            .map_err(|err| {
                MdSessionError::new(format!("missing rgpot_md_session_engine_create: {err}"))
            })?;
        let c_options = match options {
            Some(text) => Some(
                std::ffi::CString::new(text)
                    .map_err(|_| MdSessionError::new("options contain interior NUL"))?,
            ),
            None => None,
        };
        let options_ptr = match &c_options {
            Some(value) => value.as_ptr(),
            None => std::ptr::null(),
        };
        let mut ops = std::mem::MaybeUninit::<MdSessionOps>::zeroed();
        // SAFETY: `ops` is allocated on the stack and `struct_size` is an aligned `usize` field.
        unsafe {
            std::ptr::addr_of_mut!((*ops.as_mut_ptr()).struct_size)
                .write(std::mem::size_of::<MdSessionOps>());
        }
        let mut ctx: *mut c_void = std::ptr::null_mut();
        let mut n_atoms: usize = 0;
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            create(
                if spec.is_empty() {
                    std::ptr::null()
                } else {
                    spec.as_ptr().cast()
                },
                spec.len(),
                options_ptr,
                ops.as_mut_ptr(),
                &mut ctx,
                &mut n_atoms,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(MdSessionError::new(error_message(&error)));
        }
        let mut session = unsafe { Self::from_engine_ops(ctx, ops, n_atoms) }?;
        let can_unload = unsafe { is_unload_safe(&library) };
        let pin = if can_unload {
            None
        } else {
            let create_ptr = (*create) as *const () as *const c_void;
            unsafe { pin_resident_library(engine_path, create_ptr) }.map_err(|err| {
                MdSessionError::new(format!("failed to pin resident MD engine: {err}"))
            })?
        };
        session._library = Some(EngineLibraryHandle {
            library: std::mem::ManuallyDrop::new(library),
            _pin: pin,
            can_unload,
        });
        Ok(session)
    }

    #[cfg(feature = "engine")]
    fn release_library(&mut self) {
        if let Some(mut owned) = self._library.take() {
            if owned.can_unload {
                // SAFETY: `self.ops.destroy` has already returned and the
                // engine declared itself `RGPOT_ENGINE_UNLOAD_SAFE` with a
                // compatible ABI stamp at or above `ENGINE_UNLOAD_POLICY_MINOR`.
                unsafe { std::mem::ManuallyDrop::drop(&mut owned.library) };
            }
        }
    }

    #[cfg(not(feature = "engine"))]
    fn release_library(&mut self) {}
}

impl Drop for CAbiMdSession {
    fn drop(&mut self) {
        unsafe { (self.ops.destroy)(self.ctx.as_ptr()) };
        self.release_library();
    }
}

fn error_message(error: &[c_char]) -> String {
    let bytes: Vec<u8> = error.iter().map(|value| *value as u8).collect();
    match CStr::from_bytes_until_nul(&bytes) {
        Ok(value) => {
            let text = value.to_string_lossy().into_owned();
            if text.is_empty() {
                "rgpot MD session failed".to_string()
            } else {
                text
            }
        }
        Err(_) => "rgpot MD session failed".to_string(),
    }
}

fn check_finite_slice(values: &[f64], label: &str) -> Result<(), MdSessionError> {
    if !values.iter().all(|value| value.is_finite()) {
        return Err(MdSessionError::new(format!(
            "{label} contains non-finite values"
        )));
    }
    Ok(())
}

impl MdSession for CAbiMdSession {
    fn dof(&self) -> usize {
        self.n3
    }

    fn masses(&self) -> Result<Vec<f64>, MdSessionError> {
        let mut masses = vec![0.0; self.n_atoms];
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            (self.ops.get_masses)(
                self.ctx.as_ptr(),
                masses.as_mut_ptr(),
                self.n_atoms,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(masses)
        } else {
            Err(MdSessionError::new(error_message(&error)))
        }
    }

    fn set_positions(&mut self, positions: &[f64]) -> Result<(), MdSessionError> {
        if positions.len() != self.n3 {
            return Err(MdSessionError::new(format!(
                "expected {} position coordinates, got {}",
                self.n3,
                positions.len()
            )));
        }
        check_finite_slice(positions, "positions")?;
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            (self.ops.set_positions)(
                self.ctx.as_ptr(),
                positions.as_ptr(),
                positions.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(MdSessionError::new(error_message(&error)))
        }
    }

    fn set_velocities(&mut self, velocities: &[f64]) -> Result<(), MdSessionError> {
        if velocities.len() != self.n3 {
            return Err(MdSessionError::new(format!(
                "expected {} velocity coordinates, got {}",
                self.n3,
                velocities.len()
            )));
        }
        check_finite_slice(velocities, "velocities")?;
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            (self.ops.set_velocities)(
                self.ctx.as_ptr(),
                velocities.as_ptr(),
                velocities.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(MdSessionError::new(error_message(&error)))
        }
    }

    fn step_langevin(
        &mut self,
        config: &MdLangevinConfig,
        plane_point: &[f64],
        plane_normal: &[f64],
        wall_points: &[f64],
        wall_normals: &[f64],
        steps: usize,
    ) -> Result<usize, MdSessionError> {
        if !(config.k_t.is_finite() && config.k_t > 0.0) {
            return Err(MdSessionError::new(
                "Langevin kT must be finite and positive",
            ));
        }
        if !(config.friction.is_finite() && config.friction >= 0.0) {
            return Err(MdSessionError::new(
                "Langevin friction must be finite and non-negative",
            ));
        }
        if !(config.time_step.is_finite() && config.time_step > 0.0) {
            return Err(MdSessionError::new(
                "Langevin time_step must be finite and positive",
            ));
        }
        if plane_point.len() != self.n3 || plane_normal.len() != self.n3 {
            return Err(MdSessionError::new(
                "hyperplane point and normal length must match 3 * n_atoms",
            ));
        }
        if wall_points.len() != wall_normals.len() || !wall_points.len().is_multiple_of(self.n3) {
            return Err(MdSessionError::new(
                "wall points and normals must match and be multiples of 3 * n_atoms",
            ));
        }
        check_finite_slice(plane_point, "plane_point")?;
        check_finite_slice(plane_normal, "plane_normal")?;
        check_finite_slice(wall_points, "wall_points")?;
        check_finite_slice(wall_normals, "wall_normals")?;

        let n_walls = wall_points.len() / self.n3;
        let wp_ptr = if n_walls > 0 {
            wall_points.as_ptr()
        } else {
            std::ptr::null()
        };
        let wn_ptr = if n_walls > 0 {
            wall_normals.as_ptr()
        } else {
            std::ptr::null()
        };

        let mut error = [0 as c_char; 1024];
        let mut hits: u64 = 0;
        let status = unsafe {
            (self.ops.step_langevin)(
                self.ctx.as_ptr(),
                config.k_t,
                config.friction,
                config.time_step,
                plane_point.as_ptr(),
                plane_normal.as_ptr(),
                wp_ptr,
                wn_ptr,
                n_walls,
                self.n3,
                steps,
                config.seed,
                &mut hits,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            usize::try_from(hits).map_err(|_| MdSessionError::new("wall hit count exceeds usize"))
        } else {
            Err(MdSessionError::new(error_message(&error)))
        }
    }

    fn step_nve(&mut self, time_step: f64, steps: usize) -> Result<(), MdSessionError> {
        if !(time_step.is_finite() && time_step > 0.0) {
            return Err(MdSessionError::new(
                "NVE time_step must be finite and positive",
            ));
        }
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            (self.ops.step_nve)(
                self.ctx.as_ptr(),
                time_step,
                steps,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(MdSessionError::new(error_message(&error)))
        }
    }

    fn set_periodic_box(&mut self, box9: &[f64; 9]) -> Result<(), MdSessionError> {
        check_finite_slice(box9, "periodic box")?;
        let Some(set) = self.ops.set_periodic_box else {
            return Err(MdSessionError::new(
                "MD engine does not support set_periodic_box",
            ));
        };
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            set(
                self.ctx.as_ptr(),
                box9.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(MdSessionError::new(error_message(&error)))
        }
    }

    fn periodic_box(&mut self) -> Result<[f64; 9], MdSessionError> {
        let Some(get) = self.ops.get_periodic_box else {
            return Err(MdSessionError::new(
                "MD engine does not support get_periodic_box",
            ));
        };
        let mut box9 = [0.0; 9];
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            get(
                self.ctx.as_ptr(),
                box9.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(box9)
        } else {
            Err(MdSessionError::new(error_message(&error)))
        }
    }

    fn initialize(&mut self) -> Result<(), MdSessionError> {
        let Some(initialize) = self.ops.initialize else {
            return Ok(());
        };
        let mut error = [0 as c_char; 1024];
        let status = unsafe { initialize(self.ctx.as_ptr(), error.as_mut_ptr(), error.len()) };
        if status == 0 {
            Ok(())
        } else {
            Err(MdSessionError::new(error_message(&error)))
        }
    }

    fn run_langevin(
        &mut self,
        config: &MdLangevinConfig,
        plane: Option<(&[f64], &[f64])>,
        wall_points: &[f64],
        wall_normals: &[f64],
        steps: usize,
    ) -> Result<MdLangevinOutcome, MdSessionError> {
        if !(config.k_t.is_finite() && config.k_t > 0.0) {
            return Err(MdSessionError::new(
                "Langevin kT must be finite and positive",
            ));
        }
        if !(config.friction.is_finite() && config.friction >= 0.0) {
            return Err(MdSessionError::new(
                "Langevin friction must be finite and non-negative",
            ));
        }
        if !(config.time_step.is_finite() && config.time_step > 0.0) {
            return Err(MdSessionError::new(
                "Langevin time_step must be finite and positive",
            ));
        }
        if let Some((point, normal)) = plane {
            if point.len() != self.n3 || normal.len() != self.n3 {
                return Err(MdSessionError::new(
                    "hyperplane point and normal length must match 3 * n_atoms",
                ));
            }
            check_finite_slice(point, "plane_point")?;
            check_finite_slice(normal, "plane_normal")?;
        }
        if wall_points.len() != wall_normals.len() || !wall_points.len().is_multiple_of(self.n3) {
            return Err(MdSessionError::new(
                "wall points and normals must match and be multiples of 3 * n_atoms",
            ));
        }
        check_finite_slice(wall_points, "wall_points")?;
        check_finite_slice(wall_normals, "wall_normals")?;
        let Some(run) = self.ops.run_langevin else {
            return Err(MdSessionError::new(
                "MD engine does not support run_langevin",
            ));
        };
        let n_walls = wall_points.len() / self.n3;
        let record = config.record_constraint_force && plane.is_some();
        let mut constraint_force = vec![0.0; if record { steps } else { 0 }];
        let request = MdLangevinRequest {
            struct_size: std::mem::size_of::<MdLangevinRequest>(),
            k_t: config.k_t,
            friction: config.friction,
            time_step: config.time_step,
            steps,
            seed: config.seed,
            n3: self.n3,
            plane_point: plane.map_or(std::ptr::null(), |(point, _)| point.as_ptr()),
            plane_normal: plane.map_or(std::ptr::null(), |(_, normal)| normal.as_ptr()),
            n_walls,
            wall_points: if n_walls > 0 {
                wall_points.as_ptr()
            } else {
                std::ptr::null()
            },
            wall_normals: if n_walls > 0 {
                wall_normals.as_ptr()
            } else {
                std::ptr::null()
            },
            wall_mode: config.wall_mode.to_c(),
        };
        let mut result = MdLangevinResult {
            struct_size: std::mem::size_of::<MdLangevinResult>(),
            wall_hits: 0,
            constraint_force: if record {
                constraint_force.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            constraint_force_len: constraint_force.len(),
        };
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            run(
                self.ctx.as_ptr(),
                &request,
                &mut result,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(MdSessionError::new(error_message(&error)));
        }
        Ok(MdLangevinOutcome {
            wall_hits: usize::try_from(result.wall_hits)
                .map_err(|_| MdSessionError::new("wall hit count exceeds usize"))?,
            constraint_force,
        })
    }

    fn state(&mut self) -> Result<MdSessionState, MdSessionError> {
        let mut state = MdSessionState {
            positions: vec![0.0; self.n3],
            velocities: vec![0.0; self.n3],
            forces: vec![0.0; self.n3],
            ..MdSessionState::default()
        };
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            (self.ops.get_state)(
                self.ctx.as_ptr(),
                state.positions.as_mut_ptr(),
                state.velocities.as_mut_ptr(),
                state.forces.as_mut_ptr(),
                self.n3,
                &mut state.potential_energy,
                &mut state.kinetic_energy,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(state)
        } else {
            Err(MdSessionError::new(error_message(&error)))
        }
    }

    fn get_neighbors(
        &mut self,
        request: &MdNlistRequest,
        out: Option<MdNlistBuffers<'_>>,
    ) -> Result<(usize, usize), MdSessionError> {
        let Some(callback) = self.ops.get_neighbors else {
            return Err(MdSessionError::new(
                "MD engine does not expose neighbor queries",
            ));
        };
        let mut raw = match out {
            Some(mut buffers) => Some(buffers.as_raw(request)?),
            None => None,
        };
        let cap = raw.as_ref().map_or(0, |soa| soa.cap);
        let out_ptr = raw
            .as_mut()
            .map_or(std::ptr::null_mut(), |soa| soa as *mut MdNlistSoa);
        let mut count: usize = 0;
        let mut needed: usize = 0;
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            callback(
                self.ctx.as_ptr(),
                request as *const MdNlistRequest,
                out_ptr,
                &mut count,
                &mut needed,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        match MdNlistStatus::from_raw(status) {
            MdNlistStatus::Ok if count > cap => Err(MdSessionError::new(format!(
                "MD engine reported {count} neighbour pairs for a buffer of {cap}"
            ))),
            MdNlistStatus::Ok => Ok((count, needed)),
            MdNlistStatus::Unsupported => {
                let message = error_message(&error);
                if message == "rgpot MD session failed" {
                    Err(MdSessionError::new(
                        "MD engine does not expose neighbor queries",
                    ))
                } else {
                    Err(MdSessionError::new(message))
                }
            }
            MdNlistStatus::Error => Err(MdSessionError::new(error_message(&error))),
        }
    }

    fn set_neighbors(
        &mut self,
        request: &MdNlistRequest,
        pairs: &MdNlistPairs<'_>,
    ) -> Result<(), MdSessionError> {
        let Some(callback) = self.ops.set_neighbors else {
            return Err(MdSessionError::new(
                "MD engine does not accept supplied neighbors",
            ));
        };
        let raw = pairs.as_raw(request)?;
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            callback(
                self.ctx.as_ptr(),
                request as *const MdNlistRequest,
                &raw,
                raw.cap,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        match MdNlistStatus::from_raw(status) {
            MdNlistStatus::Ok => Ok(()),
            MdNlistStatus::Unsupported => {
                let message = error_message(&error);
                if message == "rgpot MD session failed" {
                    Err(MdSessionError::new(
                        "MD engine does not accept supplied neighbors",
                    ))
                } else {
                    Err(MdSessionError::new(message))
                }
            }
            MdNlistStatus::Error => Err(MdSessionError::new(error_message(&error))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    struct MockCtx {
        positions: Vec<f64>,
        velocities: Vec<f64>,
        masses: Vec<f64>,
        last_seed: u64,
        last_k_t: f64,
        last_time_step: f64,
        last_n_walls: usize,
        last_wall_mode: i32,
        cell: [f64; 9],
        fail_next: bool,
        nlist_calls: usize,
        nlist_pairs: usize,
        received_pairs: Vec<(i32, i32)>,
        destroyed: Arc<AtomicBool>,
    }

    impl MockCtx {
        fn new(destroyed: &Arc<AtomicBool>) -> Self {
            Self {
                positions: vec![0.0; 6],
                velocities: vec![0.0; 6],
                masses: vec![4.0, 39.9],
                last_seed: 0,
                last_k_t: 0.0,
                last_time_step: 0.0,
                last_n_walls: 0,
                last_wall_mode: -1,
                cell: [0.0; 9],
                fail_next: false,
                nlist_calls: 0,
                nlist_pairs: 3,
                received_pairs: Vec::new(),
                destroyed: Arc::clone(destroyed),
            }
        }
    }

    fn write_c_err(err: *mut c_char, err_len: usize, msg: &str) {
        if err.is_null() || err_len == 0 {
            return;
        }
        let bytes = msg.as_bytes();
        let n = bytes.len().min(err_len - 1);
        unsafe {
            for (i, &b) in bytes[..n].iter().enumerate() {
                *err.add(i) = b as c_char;
            }
            *err.add(n) = 0;
        }
    }

    unsafe extern "C" fn mock_set_positions(
        ctx: *mut c_void,
        positions: *const f64,
        n3: usize,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        let state = unsafe { &mut *(ctx as *mut MockCtx) };
        let slice = unsafe { std::slice::from_raw_parts(positions, n3) };
        state.positions.copy_from_slice(slice);
        0
    }

    unsafe extern "C" fn mock_set_velocities(
        ctx: *mut c_void,
        velocities: *const f64,
        n3: usize,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        let state = unsafe { &mut *(ctx as *mut MockCtx) };
        let slice = unsafe { std::slice::from_raw_parts(velocities, n3) };
        state.velocities.copy_from_slice(slice);
        0
    }

    unsafe extern "C" fn mock_step_langevin(
        ctx: *mut c_void,
        k_t: f64,
        _friction: f64,
        time_step: f64,
        _plane_point: *const f64,
        _plane_normal: *const f64,
        _wall_points: *const f64,
        _wall_normals: *const f64,
        n_walls: usize,
        _n3: usize,
        steps: usize,
        seed: u64,
        wall_hits: *mut u64,
        err: *mut c_char,
        err_len: usize,
    ) -> i32 {
        let state = unsafe { &mut *(ctx as *mut MockCtx) };
        if state.fail_next {
            state.fail_next = false;
            write_c_err(err, err_len, "mock langevin error");
            return 1;
        }
        state.last_k_t = k_t;
        state.last_time_step = time_step;
        state.last_seed = seed;
        state.last_n_walls = n_walls;
        for (p, v) in state.positions.iter_mut().zip(&state.velocities) {
            *p += time_step * (steps as f64) * *v;
        }
        if !wall_hits.is_null() {
            unsafe { *wall_hits = (n_walls * steps) as u64 };
        }
        0
    }

    unsafe extern "C" fn mock_step_nve(
        ctx: *mut c_void,
        time_step: f64,
        steps: usize,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        let state = unsafe { &mut *(ctx as *mut MockCtx) };
        state.last_time_step = time_step;
        for (p, v) in state.positions.iter_mut().zip(&state.velocities) {
            *p += time_step * (steps as f64) * *v;
        }
        0
    }

    unsafe extern "C" fn mock_get_state(
        ctx: *mut c_void,
        positions: *mut f64,
        velocities: *mut f64,
        forces: *mut f64,
        n3: usize,
        potential_energy: *mut f64,
        kinetic_energy: *mut f64,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        let state = unsafe { &*(ctx as *mut MockCtx) };
        unsafe {
            std::slice::from_raw_parts_mut(positions, n3).copy_from_slice(&state.positions);
            std::slice::from_raw_parts_mut(velocities, n3).copy_from_slice(&state.velocities);
            std::slice::from_raw_parts_mut(forces, n3).fill(-1.0);
            *potential_energy = 0.5;
            *kinetic_energy = 0.25;
        }
        0
    }

    unsafe extern "C" fn mock_get_masses(
        ctx: *mut c_void,
        masses: *mut f64,
        n_atoms: usize,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        let state = unsafe { &*(ctx as *mut MockCtx) };
        unsafe {
            std::slice::from_raw_parts_mut(masses, n_atoms).copy_from_slice(&state.masses);
        }
        0
    }

    unsafe extern "C" fn mock_destroy(ctx: *mut c_void) {
        let boxed = unsafe { Box::from_raw(ctx as *mut MockCtx) };
        boxed.destroyed.store(true, Ordering::SeqCst);
    }

    /// Reports `nlist_pairs` pairs `(k, k + 1)` at distance `k + 1`.
    /// A buffer with `cap > 0` that cannot hold them is an error and the
    /// message names the required count. `fail_next` still returns success
    /// with a count past `cap`, which the wrapper refuses.
    unsafe extern "C" fn mock_get_neighbors(
        ctx: *mut c_void,
        _request: *const MdNlistRequest,
        out: *mut MdNlistSoa,
        count: *mut usize,
        needed: *mut usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32 {
        let state = unsafe { &mut *(ctx as *mut MockCtx) };
        state.nlist_calls += 1;
        let total = state.nlist_pairs;
        let Some(out) = (unsafe { out.as_ref() }) else {
            unsafe {
                *count = 0;
                *needed = total;
            }
            return MdNlistStatus::Ok as i32;
        };
        if out.cap == 0 {
            unsafe {
                *count = 0;
                *needed = total;
            }
            return MdNlistStatus::Ok as i32;
        }
        let written = total.min(out.cap);
        for k in 0..written {
            unsafe {
                *out.first.add(k) = k as i32;
                *out.second.add(k) = k as i32 + 1;
                if !out.distance.is_null() {
                    *out.distance.add(k) = k as f64 + 1.0;
                }
            }
        }
        if state.fail_next {
            state.fail_next = false;
            unsafe {
                *count = out.cap + 1;
                *needed = total;
            }
            return MdNlistStatus::Ok as i32;
        }
        unsafe {
            *count = written;
            *needed = total;
        }
        if written < total {
            write_c_err(
                err,
                err_len,
                &format!("neighbour buffer holds {written} pairs; {total} are required"),
            );
            return MdNlistStatus::Error as i32;
        }
        MdNlistStatus::Ok as i32
    }

    unsafe extern "C" fn mock_set_neighbors(
        ctx: *mut c_void,
        _request: *const MdNlistRequest,
        pairs: *const MdNlistSoa,
        count: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32 {
        let state = unsafe { &mut *(ctx as *mut MockCtx) };
        let pairs = unsafe { &*pairs };
        if pairs.cap != count {
            write_c_err(err, err_len, "count and cap differ");
            return MdNlistStatus::Error as i32;
        }
        state.received_pairs = (0..count)
            .map(|k| unsafe { (*pairs.first.add(k), *pairs.second.add(k)) })
            .collect();
        MdNlistStatus::Ok as i32
    }

    unsafe extern "C" fn mock_refuse_neighbors(
        _ctx: *mut c_void,
        _request: *const MdNlistRequest,
        _pairs: *const MdNlistSoa,
        _count: usize,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        MdNlistStatus::Unsupported as i32
    }

    fn mock_ops() -> MdSessionOps {
        MdSessionOps {
            struct_size: std::mem::size_of::<MdSessionOps>(),
            set_positions: mock_set_positions,
            set_velocities: mock_set_velocities,
            step_langevin: mock_step_langevin,
            step_nve: mock_step_nve,
            get_state: mock_get_state,
            get_masses: mock_get_masses,
            destroy: mock_destroy,
            set_periodic_box: None,
            get_periodic_box: None,
            initialize: None,
            run_langevin: None,
            get_neighbors: Some(mock_get_neighbors),
            set_neighbors: Some(mock_set_neighbors),
        }
    }

    unsafe extern "C" fn mock_set_periodic_box(
        ctx: *mut c_void,
        box9: *const f64,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        let state = unsafe { &mut *(ctx as *mut MockCtx) };
        state
            .cell
            .copy_from_slice(unsafe { std::slice::from_raw_parts(box9, 9) });
        0
    }

    unsafe extern "C" fn mock_get_periodic_box(
        ctx: *mut c_void,
        box9: *mut f64,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        let state = unsafe { &*(ctx as *mut MockCtx) };
        unsafe { std::slice::from_raw_parts_mut(box9, 9) }.copy_from_slice(&state.cell);
        0
    }

    unsafe extern "C" fn mock_run_langevin(
        ctx: *mut c_void,
        request: *const MdLangevinRequest,
        result: *mut MdLangevinResult,
        err: *mut c_char,
        err_len: usize,
    ) -> i32 {
        let state = unsafe { &mut *(ctx as *mut MockCtx) };
        let request = unsafe { &*request };
        let result = unsafe { &mut *result };
        if request.struct_size < std::mem::size_of::<MdLangevinRequest>() {
            write_c_err(err, err_len, "short request");
            return 1;
        }
        state.last_seed = request.seed;
        state.last_wall_mode = request.wall_mode;
        result.wall_hits = (request.n_walls * request.steps) as u64;
        if !result.constraint_force.is_null() {
            if result.constraint_force_len < request.steps {
                write_c_err(err, err_len, "constraint buffer too short");
                return 1;
            }
            for step in 0..request.steps {
                unsafe { *result.constraint_force.add(step) = 0.5 * (step as f64 + 1.0) };
            }
        }
        0
    }

    #[test]
    fn error_buffers_and_state_builders_are_safe() {
        let state = MdSessionState::default();
        assert!(state.positions.is_empty());
        assert_eq!(
            error_message(&[b'b' as c_char, b'a' as c_char, 0, b'x' as c_char]),
            "ba"
        );
        assert_eq!(error_message(&[0]), "rgpot MD session failed");
    }

    #[test]
    fn cabi_session_forwards_controls_validates_inputs_and_drops() {
        let destroyed = Arc::new(AtomicBool::new(false));
        let ctx = Box::into_raw(Box::new(MockCtx::new(&destroyed)));
        let mut session =
            unsafe { CAbiMdSession::from_ops(NonNull::new_unchecked(ctx.cast()), mock_ops(), 2) }
                .expect("valid vtable");

        assert_eq!(session.dof(), 6);
        assert_eq!(session.masses().unwrap(), vec![4.0, 39.9]);

        // Buffer length and finiteness validation before C boundary.
        assert!(session.set_positions(&[0.0; 3]).is_err());
        assert!(session.set_velocities(&[f64::NAN; 6]).is_err());
        session
            .set_positions(&[1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
            .unwrap();
        session
            .set_velocities(&[0.5, 0.0, 0.0, -0.5, 0.0, 0.0])
            .unwrap();

        let cfg = MdLangevinConfig::default()
            .with_k_t(0.05)
            .with_friction(2.0)
            .with_time_step(0.002)
            .with_seed(99);
        let plane_p = [0.0; 6];
        let plane_n = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let wall_p = [0.0; 12];
        let wall_n = [0.0; 12];
        // Mismatched wall buffers rejected.
        assert!(session
            .step_langevin(&cfg, &plane_p, &plane_n, &wall_p, &[0.0; 6], 1)
            .is_err());
        let hits = session
            .step_langevin(&cfg, &plane_p, &plane_n, &wall_p, &wall_n, 3)
            .unwrap();
        assert_eq!(hits, 6);
        unsafe {
            let inner = &*ctx;
            assert_eq!(inner.last_seed, 99);
            assert_eq!(inner.last_k_t, 0.05);
            assert_eq!(inner.last_time_step, 0.002);
            assert_eq!(inner.last_n_walls, 2);
            (&mut *ctx).fail_next = true;
        }
        let err = session
            .step_langevin(&cfg, &plane_p, &plane_n, &[], &[], 1)
            .unwrap_err();
        assert_eq!(err.to_string(), "mock langevin error");

        session.step_nve(0.004, 2).unwrap();
        let st = session.state().unwrap();
        assert_eq!(st.potential_energy, 0.5);
        assert_eq!(st.kinetic_energy, 0.25);
        assert_eq!(st.forces, vec![-1.0; 6]);
        let expected_x0 = 1.0 + 0.002 * 3.0 * 0.5 + 0.004 * 2.0 * 0.5;
        assert!((st.positions[0] - expected_x0).abs() <= 16.0 * f64::EPSILON);

        drop(session);
        assert!(destroyed.load(Ordering::SeqCst));
    }

    fn mock_session(
        ops: MdSessionOps,
        destroyed: &Arc<AtomicBool>,
    ) -> (CAbiMdSession, *mut MockCtx) {
        let ctx = Box::into_raw(Box::new(MockCtx::new(destroyed)));
        let session =
            unsafe { CAbiMdSession::from_ops(NonNull::new_unchecked(ctx.cast()), ops, 2) }
                .expect("valid vtable");
        (session, ctx)
    }

    #[test]
    fn neighbour_callbacks_forward_and_report_the_typed_status() {
        let destroyed = Arc::new(AtomicBool::new(false));
        let (mut session, ctx) = mock_session(mock_ops(), &destroyed);
        let request = MdNlistRequest {
            want_distances: 1,
            ..MdNlistRequest::new()
        };
        assert_eq!(session.get_neighbors(&request, None).unwrap(), (0, 3));

        let (mut first, mut second, mut distance) = ([-1; 4], [-1; 4], [0.0; 4]);
        let buffers = MdNlistBuffers::new(&mut first, &mut second).with_distance(&mut distance);
        assert_eq!(buffers.capacity(), 4);
        assert_eq!(
            session.get_neighbors(&request, Some(buffers)).unwrap(),
            (3, 3)
        );
        assert_eq!(first, [0, 1, 2, -1]);
        assert_eq!(second, [1, 2, 3, -1]);
        assert_eq!(distance, [1.0, 2.0, 3.0, 0.0]);

        let (mut first, mut second) = ([0; 2], [0; 2]);
        let short = session
            .get_neighbors(
                &MdNlistRequest::new(),
                Some(MdNlistBuffers::new(&mut first, &mut second)),
            )
            .unwrap_err();
        assert!(short.to_string().contains("3 are required"), "{short}");
        assert_eq!(unsafe { (*ctx).nlist_calls }, 3);

        // Columns that disagree with the capacity or the request are refused
        // before the engine is called.
        let (mut first, mut second) = ([0; 2], [0; 3]);
        let mismatch = session
            .get_neighbors(&request, Some(MdNlistBuffers::new(&mut first, &mut second)))
            .unwrap_err();
        assert!(mismatch.to_string().contains("column second"), "{mismatch}");
        let (mut first, mut second) = ([0; 2], [0; 2]);
        let unasked = session
            .get_neighbors(&request, Some(MdNlistBuffers::new(&mut first, &mut second)))
            .unwrap_err();
        assert!(unasked.to_string().contains("distance column"), "{unasked}");
        let (mut first, mut second, mut vector) = ([0; 2], [0; 2], [0.0; 5]);
        let short_vector = session
            .get_neighbors(
                &MdNlistRequest::new(),
                Some(MdNlistBuffers::new(&mut first, &mut second).with_vector(&mut vector)),
            )
            .unwrap_err();
        assert!(
            short_vector.to_string().contains("vector holds 5"),
            "{short_vector}"
        );
        let cluster = MdNlistRequest {
            shape: 1,
            ..MdNlistRequest::new()
        };
        let (mut first, mut second) = ([0; 2], [0; 2]);
        assert!(session
            .get_neighbors(&cluster, Some(MdNlistBuffers::new(&mut first, &mut second)))
            .unwrap_err()
            .to_string()
            .contains("mask column"));
        assert_eq!(unsafe { (*ctx).nlist_calls }, 3);

        // A count beyond the capacity is an engine error, not a length.
        unsafe { (*ctx).fail_next = true };
        let (mut first, mut second) = ([0; 2], [0; 2]);
        let overrun = session
            .get_neighbors(
                &MdNlistRequest::new(),
                Some(MdNlistBuffers::new(&mut first, &mut second)),
            )
            .unwrap_err();
        assert!(overrun
            .to_string()
            .contains("3 neighbour pairs for a buffer of 2"));

        let pairs = MdNlistPairs::new(&[0, 1], &[1, 0]);
        assert_eq!(pairs.count(), 2);
        session
            .set_neighbors(&MdNlistRequest::new(), &pairs)
            .unwrap();
        assert_eq!(unsafe { &(*ctx).received_pairs }, &[(0, 1), (1, 0)]);
        assert!(session
            .set_neighbors(&request, &pairs)
            .unwrap_err()
            .to_string()
            .contains("distance column"));
        session
            .set_neighbors(&request, &pairs.with_distance(&[1.0, 1.0]))
            .unwrap();
        assert!(session
            .set_neighbors(&MdNlistRequest::new(), &MdNlistPairs::new(&[0], &[]))
            .is_err());

        let mut refusing = mock_ops();
        refusing.set_neighbors = Some(mock_refuse_neighbors);
        let (mut refuser, _ctx) = mock_session(refusing, &destroyed);
        let refused = refuser
            .set_neighbors(&MdNlistRequest::new(), &MdNlistPairs::new(&[], &[]))
            .unwrap_err();
        assert!(refused
            .to_string()
            .contains("does not accept supplied neighbors"));

        let mut short = mock_ops();
        short.struct_size = REQUIRED_OPS_SIZE;
        let (mut bare, _ctx) = mock_session(short, &destroyed);
        assert!(bare
            .get_neighbors(&request, None)
            .unwrap_err()
            .to_string()
            .contains("does not expose neighbor queries"));
    }

    #[test]
    fn optional_callbacks_report_unsupported_when_absent() {
        let destroyed = Arc::new(AtomicBool::new(false));
        let (mut session, _ctx) = mock_session(mock_ops(), &destroyed);
        let cell = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        assert!(session
            .set_periodic_box(&cell)
            .unwrap_err()
            .to_string()
            .contains("set_periodic_box"));
        assert!(session.periodic_box().is_err());
        assert!(session.initialize().is_ok());
        let cfg = MdLangevinConfig::default();
        let err = session
            .run_langevin(
                &cfg,
                Some((&[0.0; 6], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
                &[],
                &[],
                1,
            )
            .unwrap_err();
        assert!(err.to_string().contains("run_langevin"));
    }

    #[test]
    fn a_short_vtable_hides_the_optional_callbacks() {
        let destroyed = Arc::new(AtomicBool::new(false));
        let mut ops = mock_ops();
        ops.run_langevin = Some(mock_run_langevin);
        ops.set_periodic_box = Some(mock_set_periodic_box);
        ops.struct_size = REQUIRED_OPS_SIZE;
        let (mut session, _ctx) = mock_session(ops, &destroyed);
        assert!(session.set_periodic_box(&[0.0; 9]).is_err());
        assert!(session
            .run_langevin(&MdLangevinConfig::default(), None, &[], &[], 1)
            .is_err());

        let mut tiny = mock_ops();
        tiny.struct_size = std::mem::offset_of!(MdSessionOps, destroy);
        let ctx = Box::into_raw(Box::new(MockCtx::new(&destroyed)));
        let refused =
            unsafe { CAbiMdSession::from_ops(NonNull::new_unchecked(ctx.cast()), tiny, 2) };
        assert!(refused.is_err());
        drop(unsafe { Box::from_raw(ctx) });
    }

    fn mock_ctx(destroyed: &Arc<AtomicBool>) -> *mut MockCtx {
        Box::into_raw(Box::new(MockCtx::new(destroyed)))
    }

    /// The table an engine writes, with the slot at `offset` left null.
    fn engine_ops_without(offset: usize) -> std::mem::MaybeUninit<MdSessionOps> {
        let mut ops = std::mem::MaybeUninit::new(mock_ops());
        unsafe {
            ops.as_mut_ptr()
                .cast::<u8>()
                .add(offset)
                .write_bytes(0, std::mem::size_of::<Option<unsafe extern "C" fn()>>());
        }
        ops
    }

    #[test]
    fn a_null_required_callback_is_refused_and_the_context_released() {
        for (name, offset) in REQUIRED_OPS {
            if name == "destroy" {
                continue;
            }
            let destroyed = Arc::new(AtomicBool::new(false));
            let ctx = mock_ctx(&destroyed);
            let err = unsafe {
                CAbiMdSession::from_engine_ops(ctx.cast(), engine_ops_without(offset), 2)
            }
            .err()
            .expect("a null callback is refused");
            assert_eq!(err.kind(), MdSessionErrorKind::MissingOperation(name));
            assert!(err.to_string().contains(name));
            assert!(
                destroyed.load(Ordering::SeqCst),
                "{name} leaked the context"
            );
        }
    }

    #[test]
    fn a_null_destroy_is_refused() {
        let destroyed = Arc::new(AtomicBool::new(false));
        let ctx = mock_ctx(&destroyed);
        let ops = engine_ops_without(std::mem::offset_of!(MdSessionOps, destroy));
        let err = unsafe { CAbiMdSession::from_engine_ops(ctx.cast(), ops, 2) }
            .err()
            .expect("a null destroy is refused");
        assert_eq!(err.kind(), MdSessionErrorKind::MissingOperation("destroy"));
        assert!(!destroyed.load(Ordering::SeqCst));
        drop(unsafe { Box::from_raw(ctx) });
    }

    #[test]
    fn a_refused_engine_table_releases_the_context() {
        let destroyed = Arc::new(AtomicBool::new(false));
        let ctx = mock_ctx(&destroyed);
        let ops = std::mem::MaybeUninit::new(mock_ops());
        let err = unsafe { CAbiMdSession::from_engine_ops(ctx.cast(), ops, 0) }
            .err()
            .expect("zero atoms are refused");
        assert_eq!(err.kind(), MdSessionErrorKind::Failed);
        assert!(destroyed.load(Ordering::SeqCst));

        let null = unsafe {
            CAbiMdSession::from_engine_ops(
                std::ptr::null_mut(),
                std::mem::MaybeUninit::new(mock_ops()),
                2,
            )
        };
        assert!(null.is_err());

        let destroyed = Arc::new(AtomicBool::new(false));
        let ctx = mock_ctx(&destroyed);
        let session = unsafe {
            CAbiMdSession::from_engine_ops(ctx.cast(), std::mem::MaybeUninit::new(mock_ops()), 2)
        }
        .expect("a full table is adopted");
        assert_eq!(session.dof(), 6);
        drop(session);
        assert!(destroyed.load(Ordering::SeqCst));
    }

    #[test]
    fn run_langevin_forwards_the_plane_and_returns_the_constraint_force() {
        let destroyed = Arc::new(AtomicBool::new(false));
        let mut ops = mock_ops();
        ops.run_langevin = Some(mock_run_langevin);
        ops.set_periodic_box = Some(mock_set_periodic_box);
        ops.get_periodic_box = Some(mock_get_periodic_box);
        let (mut session, ctx) = mock_session(ops, &destroyed);

        let cell = [10.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 12.0];
        session.set_periodic_box(&cell).unwrap();
        assert_eq!(session.periodic_box().unwrap(), cell);

        let plane = ([0.0; 6], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let cfg = MdLangevinConfig::default()
            .with_seed(5)
            .with_wall_mode(MdWallMode::Reject)
            .with_record_constraint_force(true);
        let out = session
            .run_langevin(&cfg, Some((&plane.0, &plane.1)), &[0.0; 6], &[0.0; 6], 3)
            .unwrap();
        assert_eq!(out.wall_hits, 3);
        assert_eq!(out.constraint_force, vec![0.5, 1.0, 1.5]);
        unsafe {
            assert_eq!((&*ctx).last_seed, 5);
            assert_eq!((&*ctx).last_wall_mode, 1);
        }

        let none = session.run_langevin(&cfg, None, &[], &[], 2).unwrap();
        assert!(none.constraint_force.is_empty());
        assert!(session
            .run_langevin(&cfg, Some((&plane.0[..3], &plane.1)), &[], &[], 1)
            .is_err());
    }

    #[test]
    fn the_random_stream_is_the_cpp_stream() {
        let mut first = SplitMix64::new(1);
        assert_eq!(first.next_u64(), 0x910a_2dec_8902_5cc1);
        assert_eq!(first.next_u64(), 0xbeeb_8da1_658e_ec67);
        let mut normals = SplitMix64::new(7);
        let draws = [
            normals.next_normal(),
            normals.next_normal(),
            normals.next_normal(),
        ];
        let expected = [
            1.364_992_297_457_228_2,
            0.144_521_221_269_415_4,
            -0.396_523_975_253_817_7,
        ];
        for (draw, want) in draws.iter().zip(expected) {
            assert!((draw - want).abs() <= 1.0e-15, "{draw} vs {want}");
        }
    }

    #[test]
    fn maxwell_boltzmann_velocities_match_the_core_draw() {
        let masses = [4.0, 39.9];
        let free = maxwell_boltzmann_velocities(&masses, 0.02585, 7, None).unwrap();
        let free_expected = [
            10.778_569_510_299_807,
            1.141_202_065_438_486,
            -3.131_124_796_627_359,
            -0.569_035_935_022_96,
            0.011_247_207_933_508_375,
            3.148_832_525_364_433,
        ];
        let normal = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let plane = maxwell_boltzmann_velocities(&masses, 0.02585, 7, Some(&normal)).unwrap();
        let plane_expected = [
            1.499_289_563_749_780_3,
            1.141_202_065_438_486,
            -3.131_124_796_627_359,
            -1.499_289_563_749_779_8,
            0.011_247_207_933_508_375,
            3.148_832_525_364_433,
        ];
        for (got, want) in free
            .iter()
            .zip(free_expected)
            .chain(plane.iter().zip(plane_expected))
        {
            assert!(
                (got - want).abs() <= 1.0e-12 * want.abs().max(1.0),
                "{got} vs {want}"
            );
        }
        assert!(plane[0] + plane[3] < 1.0e-12 && plane[0] + plane[3] > -1.0e-12);
        assert!(maxwell_boltzmann_velocities(&masses, -1.0, 7, None).is_err());
        assert!(maxwell_boltzmann_velocities(&masses, 0.02, 7, Some(&normal[..3])).is_err());
    }

    #[test]
    fn reseed_velocities_sets_the_drawn_velocities() {
        let destroyed = Arc::new(AtomicBool::new(false));
        let (mut session, ctx) = mock_session(mock_ops(), &destroyed);
        session.reseed_velocities(0.02585, 7, None).unwrap();
        let expected = maxwell_boltzmann_velocities(&[4.0, 39.9], 0.02585, 7, None).unwrap();
        unsafe {
            assert_eq!((&*ctx).velocities, expected);
        }
    }

    #[test]
    fn a_larger_host_struct_size_is_clamped_and_still_works() {
        let destroyed = Arc::new(AtomicBool::new(false));
        let mut ops = mock_ops();
        ops.set_periodic_box = Some(mock_set_periodic_box);
        ops.get_periodic_box = Some(mock_get_periodic_box);
        ops.run_langevin = Some(mock_run_langevin);
        ops.struct_size = std::mem::size_of::<MdSessionOps>() + 64;

        let ctx = mock_ctx(&destroyed);
        let mut session = unsafe {
            CAbiMdSession::from_engine_ops(ctx.cast(), std::mem::MaybeUninit::new(ops), 2)
        }
        .expect("larger struct_size is accepted and clamped");
        assert_eq!(session.ops.struct_size, std::mem::size_of::<MdSessionOps>());
        let cell = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        session.set_periodic_box(&cell).unwrap();
        assert_eq!(session.periodic_box().unwrap(), cell);
        drop(session);
        assert!(destroyed.load(Ordering::SeqCst));
    }

    #[test]
    fn a_smaller_capacity_copy_keeps_guard_bytes_intact() {
        #[repr(C)]
        struct GuardedBuffer {
            _align: [MdSessionOps; 0],
            bytes: [u8; std::mem::size_of::<MdSessionOps>() + 32],
        }
        let mut buf = GuardedBuffer {
            _align: [],
            bytes: [0xA5; std::mem::size_of::<MdSessionOps>() + 32],
        };
        let cap = REQUIRED_OPS_SIZE;
        buf.bytes[..cap].fill(0);
        let src = mock_ops();
        let filled = cap.min(std::mem::size_of::<MdSessionOps>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&src as *const MdSessionOps).cast::<u8>(),
                buf.bytes.as_mut_ptr(),
                filled,
            );
            buf.bytes.as_mut_ptr().cast::<usize>().write(filled);
        }
        assert!(buf.bytes[cap..].iter().all(|&b| b == 0xA5));

        let mut ops = std::mem::MaybeUninit::<MdSessionOps>::zeroed();
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.bytes.as_ptr(),
                ops.as_mut_ptr().cast::<u8>(),
                filled,
            );
        }
        let destroyed = Arc::new(AtomicBool::new(false));
        let ctx = mock_ctx(&destroyed);
        let mut session = unsafe { CAbiMdSession::from_engine_ops(ctx.cast(), ops, 2) }
            .expect("truncated capacity vtable is adopted");
        assert_eq!(session.ops.struct_size, REQUIRED_OPS_SIZE);
        assert!(session.set_periodic_box(&[0.0; 9]).is_err());
        drop(session);
        assert!(destroyed.load(Ordering::SeqCst));
    }
}
