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

/// Error raised by an MD session or its C ABI boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MdSessionError(String);

impl MdSessionError {
    /// Construct an MD session error with a diagnostic message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for MdSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
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

/// C ABI vtable matching `rgpot_md_session_ops` in `external_potentials.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MdSessionOps {
    /// Byte size of this struct (`std::mem::size_of::<MdSessionOps>()`).
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
    pub step_langevin: unsafe extern "C" fn(
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
    ) -> i32,
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
}

/// Rust owner for an engine-agnostic C ABI MD session vtable.
pub struct CAbiMdSession {
    ctx: NonNull<c_void>,
    ops: MdSessionOps,
    n_atoms: usize,
    n3: usize,
    #[cfg(feature = "profile")]
    _library: Option<libloading::Library>,
}

impl CAbiMdSession {
    /// Wrap an existing C backend context and vtable.
    ///
    /// # Safety
    ///
    /// `ctx` must remain valid until `ops.destroy` is called on drop, and all
    /// function pointers in `ops` must uphold the C ABI contract in
    /// `external_potentials.h`.
    pub unsafe fn from_ops(
        ctx: NonNull<c_void>,
        ops: MdSessionOps,
        n_atoms: usize,
    ) -> Result<Self, MdSessionError> {
        let required = std::mem::offset_of!(MdSessionOps, destroy)
            + std::mem::size_of::<unsafe extern "C" fn(*mut c_void)>();
        if ops.struct_size < required {
            return Err(MdSessionError::new(
                "MD session vtable struct_size is too small",
            ));
        }
        let mut ops = ops;
        // A vtable built against an older header ends before the optional
        // callbacks; whatever the struct holds there is not a callback.
        if ops.struct_size < std::mem::offset_of!(MdSessionOps, set_periodic_box) + 8 {
            ops.set_periodic_box = None;
        }
        if ops.struct_size < std::mem::offset_of!(MdSessionOps, get_periodic_box) + 8 {
            ops.get_periodic_box = None;
        }
        if ops.struct_size < std::mem::offset_of!(MdSessionOps, initialize) + 8 {
            ops.initialize = None;
        }
        if ops.struct_size < std::mem::offset_of!(MdSessionOps, run_langevin) + 8 {
            ops.run_langevin = None;
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
            #[cfg(feature = "profile")]
            _library: None,
        })
    }

    /// Dynamically load an MD engine plugin exporting `rgpot_md_session_engine_create`.
    ///
    /// # Safety
    ///
    /// `engine_path` must point to a valid rgpot MD session plugin implementing
    /// `rgpot_md_session_engine_create_fn` and `rgpot_md_session_ops`.
    #[cfg(feature = "profile")]
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
        let mut ops: MdSessionOps = unsafe { std::mem::zeroed() };
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
                &mut ops,
                &mut ctx,
                &mut n_atoms,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(MdSessionError::new(error_message(&error)));
        }
        let Some(ctx) = NonNull::new(ctx) else {
            return Err(MdSessionError::new("MD engine returned null session"));
        };
        let mut session = unsafe { Self::from_ops(ctx, ops, n_atoms) }?;
        session._library = Some(library);
        Ok(session)
    }
}

impl Drop for CAbiMdSession {
    fn drop(&mut self) {
        unsafe { (self.ops.destroy)(self.ctx.as_ptr()) };
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
        destroyed: Arc<AtomicBool>,
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
        let ctx = Box::into_raw(Box::new(MockCtx {
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
            destroyed: Arc::clone(&destroyed),
        }));
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
        let ctx = Box::into_raw(Box::new(MockCtx {
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
            destroyed: Arc::clone(destroyed),
        }));
        let session =
            unsafe { CAbiMdSession::from_ops(NonNull::new_unchecked(ctx.cast()), ops, 2) }
                .expect("valid vtable");
        (session, ctx)
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
        ops.struct_size = std::mem::offset_of!(MdSessionOps, destroy) + 8;
        let (mut session, _ctx) = mock_session(ops, &destroyed);
        assert!(session.set_periodic_box(&[0.0; 9]).is_err());
        assert!(session
            .run_langevin(&MdLangevinConfig::default(), None, &[], &[], 1)
            .is_err());

        let mut tiny = mock_ops();
        tiny.struct_size = std::mem::offset_of!(MdSessionOps, destroy);
        let ctx = Box::into_raw(Box::new(MockCtx {
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
            destroyed: Arc::clone(&destroyed),
        }));
        let refused =
            unsafe { CAbiMdSession::from_ops(NonNull::new_unchecked(ctx.cast()), tiny, 2) };
        assert!(refused.is_err());
        drop(unsafe { Box::from_raw(ctx) });
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
            0.144_521_221_269_415_41,
            -0.396_523_975_253_817_72,
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
            -0.569_035_935_022_960_02,
            0.011_247_207_933_508_375,
            3.148_832_525_364_432_9,
        ];
        let normal = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let plane = maxwell_boltzmann_velocities(&masses, 0.02585, 7, Some(&normal)).unwrap();
        let plane_expected = [
            1.499_289_563_749_780_3,
            1.141_202_065_438_486,
            -3.131_124_796_627_359,
            -1.499_289_563_749_779_8,
            0.011_247_207_933_508_375,
            3.148_832_525_364_432_9,
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
}
