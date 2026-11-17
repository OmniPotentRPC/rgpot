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
    /// Deterministic 64-bit PRNG seed.
    pub seed: u64,
}

impl Default for MdLangevinConfig {
    fn default() -> Self {
        Self {
            k_t: 0.02585,
            friction: 1.0,
            time_step: 0.001,
            seed: 1,
        }
    }
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

/// Structure-of-arrays pair buffers shared with `rgpot_nlist_soa`.
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

    /// Query the engine neighbour list for the supplied request.
    fn get_neighbors(
        &mut self,
        _request: &MdNlistRequest,
        _out: Option<&mut MdNlistSoa>,
    ) -> Result<(usize, usize), MdSessionError> {
        Err(MdSessionError::new(
            "MD engine does not expose neighbor queries",
        ))
    }

    /// Provide neighbour pairs built by the host.
    fn set_neighbors(
        &mut self,
        _request: &MdNlistRequest,
        _pairs: &MdNlistSoa,
        _count: usize,
    ) -> Result<(), MdSessionError> {
        Err(MdSessionError::new(
            "MD engine does not accept supplied neighbors",
        ))
    }
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
        out: Option<&mut MdNlistSoa>,
    ) -> Result<(usize, usize), MdSessionError> {
        let Some(callback) = self.ops.get_neighbors else {
            return Err(MdSessionError::new(
                "MD engine does not expose neighbor queries",
            ));
        };
        let mut count: usize = 0;
        let mut needed: usize = 0;
        let mut error = [0 as c_char; 1024];
        let out_ptr = match out {
            Some(value) => value as *mut MdNlistSoa,
            None => std::ptr::null_mut(),
        };
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
        pairs: &MdNlistSoa,
        count: usize,
    ) -> Result<(), MdSessionError> {
        let Some(callback) = self.ops.set_neighbors else {
            return Err(MdSessionError::new(
                "MD engine does not accept supplied neighbors",
            ));
        };
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            callback(
                self.ctx.as_ptr(),
                request as *const MdNlistRequest,
                pairs as *const MdNlistSoa,
                count,
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
        fail_next: bool,
        nlist_calls: usize,
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

    unsafe extern "C" fn mock_get_neighbors(
        ctx: *mut c_void,
        _request: *const MdNlistRequest,
        _out: *mut MdNlistSoa,
        count: *mut usize,
        needed: *mut usize,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        let state = unsafe { &mut *(ctx as *mut MockCtx) };
        state.nlist_calls += 1;
        if !count.is_null() {
            unsafe { *count = 4 };
        }
        if !needed.is_null() {
            unsafe { *needed = 4 };
        }
        MdNlistStatus::Ok as i32
    }

    unsafe extern "C" fn mock_set_neighbors(
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
            get_neighbors: Some(mock_get_neighbors),
            set_neighbors: Some(mock_set_neighbors),
            destroy: mock_destroy,
        }
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
            fail_next: false,
            nlist_calls: 0,
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
        let request = MdNlistRequest::default();
        let mut out = MdNlistSoa::default();
        let (count, needed) = session.get_neighbors(&request, Some(&mut out)).unwrap();
        assert_eq!(count, 4);
        assert_eq!(needed, 4);
        let set_error = session
            .set_neighbors(&request, &MdNlistSoa::default(), 0)
            .unwrap_err();
        assert!(set_error
            .to_string()
            .contains("does not accept supplied neighbors"));

        drop(session);
        assert!(destroyed.load(Ordering::SeqCst));
    }
}
