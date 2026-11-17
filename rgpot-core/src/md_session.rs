//! Engine-neutral molecular-dynamics session contract.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr::NonNull;

/// State returned by an rgpot MD session, in the engine's documented units.
#[derive(Clone, Debug, Default)]
pub struct MdSessionState {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub forces: Vec<f64>,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
}

/// Safe Rust contract implemented by C ABI-backed rgpot engines.
pub trait MdSession {
    fn set_positions(&mut self, positions: &[f64]) -> Result<(), String>;
    fn set_velocities(&mut self, velocities: &[f64]) -> Result<(), String>;
    fn step_langevin(
        &mut self,
        k_t: f64,
        friction: f64,
        time_step: f64,
        plane_point: &[f64],
        plane_normal: &[f64],
        wall_points: &[f64],
        wall_normals: &[f64],
        steps: usize,
        seed: u64,
    ) -> Result<usize, String>;
    fn step_nve(&mut self, time_step: f64, steps: usize) -> Result<(), String>;
    fn state(&mut self) -> Result<MdSessionState, String>;
}

#[repr(C)]
struct rgpot_md_session {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn rgpot_md_session_create(
        system_xml: *const c_char,
        platform: *const c_char,
        time_step: f64,
        friction: f64,
        k_t: f64,
        seed: i32,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut rgpot_md_session;
    fn rgpot_md_session_set_positions(
        session: *mut rgpot_md_session,
        values: *const f64,
        n3: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
    fn rgpot_md_session_set_velocities(
        session: *mut rgpot_md_session,
        values: *const f64,
        n3: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
    fn rgpot_md_session_step_langevin(
        session: *mut rgpot_md_session,
        plane_point: *const f64,
        plane_normal: *const f64,
        wall_points: *const f64,
        wall_normals: *const f64,
        n_walls: usize,
        steps: usize,
        wall_hits: *mut u64,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
    fn rgpot_md_session_step_nve(
        session: *mut rgpot_md_session,
        steps: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
    fn rgpot_md_session_get_state(
        session: *mut rgpot_md_session,
        positions: *mut f64,
        velocities: *mut f64,
        forces: *mut f64,
        n3: usize,
        potential_energy: *mut f64,
        kinetic_energy: *mut f64,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
    fn rgpot_md_session_destroy(session: *mut rgpot_md_session);
}

/// Rust owner for the generic rgpot C session handle.
pub struct CAbiMdSession {
    raw: NonNull<rgpot_md_session>,
    n3: usize,
}

impl CAbiMdSession {
    /// Construct a session from serialized OpenMM-compatible system data.
    ///
    /// The ABI is engine-neutral; `platform` selects the backend supplied by
    /// the engine. Coordinates are nanometres, velocities nanometres/ps,
    /// energies kJ/mol, and `k_t` kJ/mol.
    pub fn new(
        system_xml: &str,
        platform: &str,
        time_step: f64,
        friction: f64,
        k_t: f64,
        seed: i32,
        n3: usize,
    ) -> Result<Self, String> {
        let xml = CString::new(system_xml).map_err(|_| "XML contains NUL".to_string())?;
        let platform = CString::new(platform).map_err(|_| "platform contains NUL".to_string())?;
        let mut error = [0_i8; 1024];
        let raw = unsafe {
            rgpot_md_session_create(
                xml.as_ptr(),
                platform.as_ptr(),
                time_step,
                friction,
                k_t,
                seed,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let raw = NonNull::new(raw).ok_or_else(|| error_message(&error))?;
        Ok(Self { raw, n3 })
    }
}

impl Drop for CAbiMdSession {
    fn drop(&mut self) {
        unsafe { rgpot_md_session_destroy(self.raw.as_ptr()) };
    }
}

fn error_message(error: &[i8]) -> String {
    let bytes = error.iter().map(|value| *value as u8).collect::<Vec<_>>();
    match CStr::from_bytes_until_nul(&bytes) {
        Ok(value) => value.to_string_lossy().into_owned(),
        Err(_) => "rgpot MD session failed".to_string(),
    }
}

impl MdSession for CAbiMdSession {
    fn set_positions(&mut self, positions: &[f64]) -> Result<(), String> {
        let mut error = [0_i8; 1024];
        let status = unsafe {
            rgpot_md_session_set_positions(
                self.raw.as_ptr(),
                positions.as_ptr(),
                positions.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        (status == 0).then_some(()).ok_or_else(|| error_message(&error))
    }

    fn set_velocities(&mut self, velocities: &[f64]) -> Result<(), String> {
        let mut error = [0_i8; 1024];
        let status = unsafe {
            rgpot_md_session_set_velocities(
                self.raw.as_ptr(),
                velocities.as_ptr(),
                velocities.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        (status == 0).then_some(()).ok_or_else(|| error_message(&error))
    }

    fn step_langevin(
        &mut self,
        _k_t: f64,
        _friction: f64,
        _time_step: f64,
        plane_point: &[f64],
        plane_normal: &[f64],
        wall_points: &[f64],
        wall_normals: &[f64],
        steps: usize,
        _seed: u64,
    ) -> Result<usize, String> {
        let mut error = [0_i8; 1024];
        let mut hits = 0;
        let status = unsafe {
            rgpot_md_session_step_langevin(
                self.raw.as_ptr(),
                plane_point.as_ptr(),
                plane_normal.as_ptr(),
                wall_points.as_ptr(),
                wall_normals.as_ptr(),
                wall_points.len() / 3,
                steps,
                &mut hits,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        (status == 0)
            .then_some(hits as usize)
            .ok_or_else(|| error_message(&error))
    }

    fn step_nve(&mut self, _time_step: f64, steps: usize) -> Result<(), String> {
        let mut error = [0_i8; 1024];
        let status = unsafe {
            rgpot_md_session_step_nve(self.raw.as_ptr(), steps, error.as_mut_ptr(), error.len())
        };
        (status == 0).then_some(()).ok_or_else(|| error_message(&error))
    }

    fn state(&mut self) -> Result<MdSessionState, String> {
        let mut state = MdSessionState {
            positions: vec![0.0; self.n3],
            velocities: vec![0.0; self.n3],
            forces: vec![0.0; self.n3],
            ..MdSessionState::default()
        };
        let mut error = [0_i8; 1024];
        let status = unsafe {
            rgpot_md_session_get_state(
                self.raw.as_ptr(),
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
        (status == 0).then_some(state).ok_or_else(|| error_message(&error))
    }
}
