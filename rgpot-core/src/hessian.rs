// MIT License
// Copyright 2023--present rgpot developers

//! Dense energy Hessian of a potential handle.
//!
//! The matrix is `d²E/dx_i dx_j`, row-major, with the same `3N` Cartesian
//! layout as the positions. A potential that knows this matrix registers a
//! [`HessianCallback`]. A potential that knows only the product `H v`
//! registers a [`crate::hvp::HessianVectorCallback`]; the dense matrix is
//! then one column per unit direction of that product. Either source is
//! exact curvature. A handle with neither returns [`HessianError::NoHessian`]
//! (`RGPOT_NO_HESSIAN`). This module does not difference forces.
//!
//! The last matrix is kept, keyed on the same geometry bits as the
//! Hessian-vector centre. A later product at that geometry is a matrix-vector
//! multiply and does not call the callback again.
//! `RGPOT_HVP_AUTO` uses a registered product callback first, then this
//! matrix, and only then a forward difference.

use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Largest Cartesian dimension of a dense Hessian this library allocates.
///
/// `8192` components is `8192²` doubles, 512 MiB. A larger shape is a typed
/// invalid-parameter error. The C entry point checks the same limit before
/// it reserves, so an untrusted quadratic size does not abort the process.
pub const RGPOT_HESSIAN_MAX_DIM: usize = 8192;

/// Element count of a dense Hessian, or a typed error when `dim` is above
/// [`RGPOT_HESSIAN_MAX_DIM`] or the byte count overflows.
pub(crate) fn hessian_len(dim: usize) -> Result<usize, HessianError> {
    if dim > RGPOT_HESSIAN_MAX_DIM {
        return Err(HessianError::Invalid(format!(
            "Hessian dimension {dim} exceeds {RGPOT_HESSIAN_MAX_DIM}"
        )));
    }
    let n2 = dim
        .checked_mul(dim)
        .ok_or_else(|| HessianError::Invalid("Hessian dimension overflows".into()))?;
    if n2.checked_mul(std::mem::size_of::<f64>()).is_none() {
        return Err(HessianError::Invalid("Hessian byte count overflows".into()));
    }
    Ok(n2)
}

fn allocate_hessian(dim: usize) -> Result<Vec<f64>, HessianError> {
    let n2 = hessian_len(dim)?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(n2)
        .map_err(|_| HessianError::Invalid(format!("Hessian allocation of {n2} doubles failed")))?;
    matrix.resize(n2, 0.0);
    Ok(matrix)
}

use crate::eindir::rgpot_potential_t;
use crate::fused::{lock, FusedKey};
use crate::hvp::{read_geometry, Geometry, HvpError};
use crate::status::rgpot_status_t;
use crate::types::rgpot_force_input_t;

/// Analytic dense energy Hessian of a potential.
///
/// The callback receives the handle's `user_data`, the geometry, and room
/// for `(3 * n_atoms)²` doubles. It writes the row-major energy Hessian and
/// returns `RGPOT_SUCCESS`, or an error status. It must not call back into
/// this handle's Hessian or Hessian-vector entry points.
pub type HessianCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    hessian_out: *mut f64,
) -> rgpot_status_t;

/// Counters of the dense-Hessian entry points.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct rgpot_hessian_counts_t {
    /// Matrices computed by a callback or by column assembly.
    pub evaluations: u64,
    /// Requests answered from the cached matrix.
    pub cache_hits: u64,
    /// Evaluations assembled from an analytic Hessian-vector callback.
    pub from_hvp: u64,
    /// `RGPOT_HVP_AUTO` products served by multiplying the cached matrix.
    pub products: u64,
}

/// Why a dense-Hessian request failed.
#[derive(Debug, Clone, PartialEq)]
pub enum HessianError {
    /// A malformed argument or callback result. C callers see
    /// `RGPOT_INVALID_PARAMETER`.
    Invalid(String),
    /// The handle has neither a Hessian callback nor an analytic
    /// Hessian-vector callback. C callers see `RGPOT_NO_HESSIAN`.
    NoHessian(String),
    /// A callback returned this status.
    Callback {
        status: rgpot_status_t,
        message: String,
    },
}

impl HessianError {
    /// The status a C caller receives.
    pub fn status(&self) -> rgpot_status_t {
        match self {
            Self::Invalid(_) => rgpot_status_t::RGPOT_INVALID_PARAMETER,
            Self::NoHessian(_) => rgpot_status_t::RGPOT_NO_HESSIAN,
            Self::Callback { status, .. } => *status,
        }
    }
}

impl std::fmt::Display for HessianError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) | Self::NoHessian(why) => f.write_str(why),
            Self::Callback { status, message } => {
                write!(f, "{message} (callback returned {status:?})")
            }
        }
    }
}

impl std::error::Error for HessianError {}

impl From<HvpError> for HessianError {
    fn from(error: HvpError) -> Self {
        match error {
            HvpError::Invalid(why) => Self::Invalid(why),
            HvpError::Callback { status, message } => Self::Callback { status, message },
        }
    }
}

impl From<HessianError> for HvpError {
    fn from(error: HessianError) -> Self {
        match error {
            HessianError::Invalid(why) | HessianError::NoHessian(why) => Self::Invalid(why),
            HessianError::Callback { status, message } => Self::Callback { status, message },
        }
    }
}

struct StoredHessian {
    key: FusedKey,
    matrix: Arc<Vec<f64>>,
}

/// Per-handle dense-Hessian state, kept in the handle's opaque cache.
#[derive(Default)]
pub(crate) struct HessianState {
    callback: Mutex<Option<HessianCallback>>,
    stored: Mutex<Option<StoredHessian>>,
    /// Bumped on callback replacement and cache invalidation. An in-flight
    /// matrix is published only when this is still the value sampled before
    /// the computation.
    generation: AtomicU64,
    evaluations: AtomicU64,
    cache_hits: AtomicU64,
    from_hvp: AtomicU64,
    products: AtomicU64,
}

/// `H v` for a row-major matrix.
pub(crate) fn apply_hessian(matrix: &[f64], direction: &[f64], out: &mut [f64]) {
    let n = direction.len();
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &matrix[i * n..(i + 1) * n];
        let mut acc = 0.0;
        for (entry, component) in row.iter().zip(direction) {
            acc += entry * component;
        }
        *slot = acc;
    }
}

impl rgpot_potential_t {
    /// Register `callback` as the analytic dense Hessian, or clear it.
    pub fn set_hessian_callback(&self, callback: Option<HessianCallback>) {
        *lock(&self.hessian_state().callback) = callback;
        self.invalidate_hessian_cache();
    }

    /// Whether an analytic dense-Hessian callback is registered.
    pub fn has_hessian_callback(&self) -> bool {
        lock(&self.hessian_state().callback).is_some()
    }

    /// Drop the cached matrix. The geometry key already changes when the
    /// fused centre is invalidated; this drops a matrix whose callback changed.
    pub(crate) fn invalidate_hessian_cache(&self) {
        let state = self.hessian_state();
        state.generation.fetch_add(1, Ordering::AcqRel);
        *lock(&state.stored) = None;
    }

    /// The counters of the dense-Hessian entry points.
    pub fn hessian_counts(&self) -> rgpot_hessian_counts_t {
        let state = self.hessian_state();
        rgpot_hessian_counts_t {
            evaluations: state.evaluations.load(Ordering::Relaxed),
            cache_hits: state.cache_hits.load(Ordering::Relaxed),
            from_hvp: state.from_hvp.load(Ordering::Relaxed),
            products: state.products.load(Ordering::Relaxed),
        }
    }

    /// Zero the counters. The cached matrix stays.
    pub fn reset_hessian_counts(&self) {
        let state = self.hessian_state();
        state.evaluations.store(0, Ordering::Relaxed);
        state.cache_hits.store(0, Ordering::Relaxed);
        state.from_hvp.store(0, Ordering::Relaxed);
        state.products.store(0, Ordering::Relaxed);
    }

    /// Record one matrix-vector product served from the cached Hessian.
    pub(crate) fn note_hessian_product(&self) {
        self.hessian_state()
            .products
            .fetch_add(1, Ordering::Relaxed);
    }

    /// The energy Hessian at `input`, from the cache when the geometry matches.
    ///
    /// # Safety
    ///
    /// The tensors of `input` must be valid for the call, and a registered
    /// callback must be safe to call with the handle's `user_data`.
    pub unsafe fn energy_hessian(
        &self,
        input: &rgpot_force_input_t,
    ) -> Result<Arc<Vec<f64>>, HessianError> {
        self.refuse_hessian_peer()?;
        let geometry = unsafe { read_geometry(input) }?;
        unsafe { self.energy_hessian_of(input, &geometry) }
    }

    /// [`Self::energy_hessian`] for a geometry that has already been read.
    ///
    /// # Safety
    ///
    /// As for [`Self::energy_hessian`]. `geometry` must be that of `input`.
    pub(crate) unsafe fn energy_hessian_of(
        &self,
        input: &rgpot_force_input_t,
        geometry: &Geometry,
    ) -> Result<Arc<Vec<f64>>, HessianError> {
        let dim = geometry.positions.len();
        if dim == 0 {
            return Ok(Arc::new(Vec::new()));
        }
        let key = self.fused().key(
            &geometry.positions,
            geometry.atomic_numbers.as_deref(),
            geometry.cell.as_deref(),
        );
        let state = self.hessian_state();
        let generation = state.generation.load(Ordering::Acquire);
        {
            let stored = lock(&state.stored);
            if state.generation.load(Ordering::Acquire) == generation {
                if let Some(hit) = stored.as_ref() {
                    if hit.key == key {
                        state.cache_hits.fetch_add(1, Ordering::Relaxed);
                        return Ok(Arc::clone(&hit.matrix));
                    }
                }
            }
        }
        let matrix = Arc::new(unsafe { self.compute_hessian(input, dim) }?);
        let mut stored = lock(&state.stored);
        if state.generation.load(Ordering::Acquire) != generation {
            return Ok(matrix);
        }
        if let Some(hit) = stored.as_ref() {
            if hit.key == key {
                state.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Arc::clone(&hit.matrix));
            }
        }
        *stored = Some(StoredHessian {
            key,
            matrix: Arc::clone(&matrix),
        });
        Ok(matrix)
    }

    fn refuse_hessian_peer(&self) -> Result<(), HessianError> {
        match self.fused_cache_peer_refusal() {
            Some(why) => Err(HessianError::Invalid(why)),
            None => Ok(()),
        }
    }

    unsafe fn compute_hessian(
        &self,
        input: &rgpot_force_input_t,
        dim: usize,
    ) -> Result<Vec<f64>, HessianError> {
        let state = self.hessian_state();
        hessian_len(dim)?;
        let callback = *lock(&state.callback);
        if let Some(callback) = callback {
            let mut matrix = allocate_hessian(dim)?;
            let status = unsafe { callback(self.pot_user_data, input, matrix.as_mut_ptr()) };
            if status != rgpot_status_t::RGPOT_SUCCESS {
                return Err(callback_error(status, "Hessian callback"));
            }
            finite_matrix(&matrix)?;
            state.evaluations.fetch_add(1, Ordering::Relaxed);
            return Ok(matrix);
        }
        let Some(callback) = self.hessian_vector_callback() else {
            return Err(HessianError::NoHessian(
                "potential has no analytic Hessian".into(),
            ));
        };
        let mut matrix = allocate_hessian(dim)?;
        let mut direction = vec![0.0; dim];
        let mut column = vec![0.0; dim];
        for j in 0..dim {
            direction.fill(0.0);
            direction[j] = 1.0;
            let status = unsafe {
                callback(
                    self.pot_user_data,
                    input,
                    direction.as_ptr(),
                    column.as_mut_ptr(),
                )
            };
            if status != rgpot_status_t::RGPOT_SUCCESS {
                return Err(callback_error(status, "Hessian-vector callback"));
            }
            finite_matrix(&column)?;
            for i in 0..dim {
                matrix[i * dim + j] = column[i];
            }
        }
        state.evaluations.fetch_add(1, Ordering::Relaxed);
        state.from_hvp.fetch_add(1, Ordering::Relaxed);
        Ok(matrix)
    }
}

fn finite_matrix(matrix: &[f64]) -> Result<(), HessianError> {
    if let Some(i) = matrix.iter().position(|v| !v.is_finite()) {
        return Err(HessianError::Invalid(format!(
            "Hessian callback: non-finite value at element {i}"
        )));
    }
    Ok(())
}

fn callback_error(status: rgpot_status_t, what: &str) -> HessianError {
    let message = crate::status::last_error_message();
    HessianError::Callback {
        status,
        message: if message.is_empty() {
            format!("{what} returned {status:?}")
        } else {
            message
        },
    }
}
