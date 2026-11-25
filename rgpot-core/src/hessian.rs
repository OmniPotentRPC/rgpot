// MIT License
// Copyright 2023--present rgpot developers

//! Dense Hessians of a potential handle, and what every derivative entry
//! point of a handle can do and has done.
//!
//! The Hessian is the `3 * n_atoms` square matrix of second derivatives of
//! the energy with respect to the positions, row-major, with coordinate
//! `3 i + k` for axis `k` of atom `i`. A request names its scheme, and
//! there is no automatic choice:
//!
//! | Scheme | Force calls | Error |
//! |--------|-------------|-------|
//! | `RGPOT_HESSIAN_ANALYTIC` | 0 | that of the callback |
//! | `RGPOT_HESSIAN_AD` | 0 | that of the callback |
//! | `RGPOT_HESSIAN_EXACT` | 0 | that of the callback it takes |
//! | `RGPOT_HESSIAN_CENTRAL` | `6 * n_atoms` | `O(h^2)` |
//!
//! `RGPOT_HESSIAN_EXACT` takes the analytic callback, else the dual-number
//! one, and is refused when the handle has neither. The central scheme
//! differences the forces at `x +- h e_c` for every coordinate `c` and
//! writes the symmetric part of the result, so it is symmetric to the last
//! bit; its calls bypass the cached centre and are counted in
//! `rgpot_derivative_counts_t::hessian_force_calls`, not in
//! `rgpot_hvp_counts_t`. [`rgpot_potential_t::hessian`] returns the scheme
//! that ran.
//!
//! [`rgpot_derivative_caps_t`] reports which schemes a handle has for the
//! product and for the Hessian, and [`rgpot_derivative_counts_t`] counts
//! the requests each scheme answered.

use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::eindir::rgpot_potential_t;
use crate::fused::lock;
use crate::hvp::{checked_step, invalid, read_geometry, run_callback, HvpError};
use crate::status::rgpot_status_t;
use crate::tensor::{checked_f64_data, checked_i32_data};
use crate::types::rgpot_force_input_t;

/// Dense Hessian of a potential.
///
/// The callback receives the `user_data` of the potential handle, the
/// geometry and room for `(3 * n_atoms)^2` doubles. It writes the Hessian,
/// row-major, to `hessian_out` and returns `RGPOT_SUCCESS`, or an error
/// status code. It must not call back into this handle's Hessian entry
/// points, and must not replace or clear the callback: the call holds the
/// registration, and replacement waits for it to finish.
pub type HessianCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    hessian_out: *mut f64,
) -> rgpot_status_t;

/// How a dense Hessian is computed. The values share the numbering of
/// `rgpot_hvp_scheme_t`; 0 and 1 have no dense meaning and are refused.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum rgpot_hessian_scheme_t {
    /// Central differences of the forces, `6 * n_atoms` force calls.
    RGPOT_HESSIAN_CENTRAL = 2,
    /// The registered Hessian callback. Refused when none is registered.
    RGPOT_HESSIAN_ANALYTIC = 3,
    /// The registered dual-number Hessian callback. Refused when none is
    /// registered.
    RGPOT_HESSIAN_AD = 4,
    /// The analytic callback, else the dual-number one. Refused when
    /// neither is registered; never a finite difference.
    RGPOT_HESSIAN_EXACT = 5,
}

impl rgpot_hessian_scheme_t {
    /// The scheme with this value, or `None` for a value the enum lacks.
    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            2 => Some(Self::RGPOT_HESSIAN_CENTRAL),
            3 => Some(Self::RGPOT_HESSIAN_ANALYTIC),
            4 => Some(Self::RGPOT_HESSIAN_AD),
            5 => Some(Self::RGPOT_HESSIAN_EXACT),
            _ => None,
        }
    }
}

/// Which schemes a handle has for Hessian-vector products and dense
/// Hessians, and what the differenced ones cost. Analytic and dual-number
/// schemes make no force call.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct rgpot_derivative_caps_t {
    /// An analytic Hessian-vector callback is registered.
    pub hvp_analytic: bool,
    /// A dual-number Hessian-vector callback is registered.
    pub hvp_ad: bool,
    /// Forward and central products are available.
    pub hvp_fd: bool,
    /// An analytic Hessian callback is registered.
    pub hessian_analytic: bool,
    /// A dual-number Hessian callback is registered.
    pub hessian_ad: bool,
    /// The central Hessian is available.
    pub hessian_fd: bool,
    /// Force calls of a forward product at the cached centre; a new centre
    /// adds one.
    pub hvp_forward_force_calls: u32,
    /// Force calls of a central product.
    pub hvp_central_force_calls: u32,
    /// Force calls of a central Hessian per atom, two per coordinate.
    pub hessian_central_force_calls_per_atom: u32,
}

/// Requests answered by each derivative scheme of one handle. A product of
/// a zero direction counts under the scheme its request resolved to.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct rgpot_derivative_counts_t {
    /// Products by forward differences.
    pub hvp_forward: u64,
    /// Products by central differences.
    pub hvp_central: u64,
    /// Products by the analytic Hessian-vector callback.
    pub hvp_analytic: u64,
    /// Products by the dual-number Hessian-vector callback.
    pub hvp_ad: u64,
    /// Hessians by central differences.
    pub hessian_central: u64,
    /// Hessians by the analytic Hessian callback.
    pub hessian_analytic: u64,
    /// Hessians by the dual-number Hessian callback.
    pub hessian_ad: u64,
    /// Force calls made by central Hessians.
    pub hessian_force_calls: u64,
}

/// Element count of a dense Hessian.
///
/// The structural limit is a byte count that fits in `usize`. `budget` is the
/// caller's optional ceiling in bytes. `None` installs no ceiling: there is
/// no built-in component count or MiB default. A `Some` budget, including
/// zero, refuses a larger matrix before any callback or allocation.
pub(crate) fn hessian_len(dim: usize, budget: Option<u64>) -> Result<usize, HvpError> {
    let n2 = dim
        .checked_mul(dim)
        .ok_or_else(|| invalid("Hessian dimension overflows"))?;
    let bytes = n2
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| invalid("Hessian byte count overflows"))?;
    if let Some(limit) = budget {
        let bytes = u64::try_from(bytes)
            .map_err(|_| invalid("Hessian byte count does not fit in 64 bits"))?;
        if bytes > limit {
            return Err(invalid(format!(
                "Hessian of {bytes} bytes exceeds the caller budget of {limit} bytes"
            )));
        }
    }
    Ok(n2)
}

/// Per-handle dense-Hessian state, kept in the handle's opaque cache.
#[derive(Default)]
pub(crate) struct HessianState {
    /// An evaluation holds a read guard for the whole callback call.
    /// Replacing or clearing the callback takes the write guard, so it
    /// returns only after every in-flight call of the old callback has
    /// finished, and no call of it can start afterwards.
    analytic: RwLock<Option<HessianCallback>>,
    ad: RwLock<Option<HessianCallback>>,
    /// `None` means no caller byte budget. The only structural limit is a
    /// representable byte count.
    byte_budget: Mutex<Option<u64>>,
    by_central: AtomicU64,
    by_analytic: AtomicU64,
    by_ad: AtomicU64,
    force_calls: AtomicU64,
}

fn read_callback(
    lock: &RwLock<Option<HessianCallback>>,
) -> RwLockReadGuard<'_, Option<HessianCallback>> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_callback(
    lock: &RwLock<Option<HessianCallback>>,
) -> RwLockWriteGuard<'_, Option<HessianCallback>> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The scheme a request runs: `EXACT` resolved against the registered
/// callbacks, and the callback schemes refused without theirs.
fn resolve(
    scheme: rgpot_hessian_scheme_t,
    analytic: bool,
    ad: bool,
) -> Result<rgpot_hessian_scheme_t, HvpError> {
    use rgpot_hessian_scheme_t::*;
    match scheme {
        RGPOT_HESSIAN_ANALYTIC if !analytic => Err(invalid(
            "RGPOT_HESSIAN_ANALYTIC: no Hessian callback is registered",
        )),
        RGPOT_HESSIAN_AD if !ad => Err(invalid(
            "RGPOT_HESSIAN_AD: no dual-number Hessian callback is registered",
        )),
        RGPOT_HESSIAN_EXACT if analytic => Ok(RGPOT_HESSIAN_ANALYTIC),
        RGPOT_HESSIAN_EXACT if ad => Ok(RGPOT_HESSIAN_AD),
        RGPOT_HESSIAN_EXACT => Err(invalid(
            "RGPOT_HESSIAN_EXACT: neither an analytic nor a dual-number Hessian \
             callback is registered",
        )),
        other => Ok(other),
    }
}

impl rgpot_potential_t {
    /// Register `callback` as the analytic Hessian of this handle, or clear
    /// it with `None`. It is called with the handle's `user_data`.
    ///
    /// Returns only after every call of the previous callback that was
    /// already running has returned; no call of it starts after this
    /// returns. The caller may therefore release the old callback and its
    /// `user_data` as soon as this returns.
    ///
    /// This waits on in-flight evaluations, so it must not be called from a
    /// Hessian callback, or from any thread that is inside one of this
    /// handle's Hessian entry points. Doing so deadlocks.
    pub fn set_hessian_callback(&self, callback: Option<HessianCallback>) {
        *write_callback(&self.hessian_state().analytic) = callback;
    }

    /// Register `callback` as the dual-number Hessian of this handle, or
    /// clear it with `None`.
    ///
    /// Waits for in-flight calls of the previous dual-number callback to
    /// finish, as [`Self::set_hessian_callback`] does.
    pub fn set_ad_hessian_callback(&self, callback: Option<HessianCallback>) {
        *write_callback(&self.hessian_state().ad) = callback;
    }

    /// Install or clear the dense-Hessian byte budget.
    ///
    /// `None` clears it. There is no default size. `Some(0)` refuses every
    /// positive matrix.
    pub fn set_hessian_byte_budget(&self, max_bytes: Option<u64>) {
        *lock(&self.hessian_state().byte_budget) = max_bytes;
    }

    /// The installed dense-Hessian byte budget, if any.
    pub fn hessian_byte_budget(&self) -> Option<u64> {
        *lock(&self.hessian_state().byte_budget)
    }

    /// Which derivative schemes this handle has, from the callbacks
    /// registered now.
    pub fn derivative_caps(&self) -> rgpot_derivative_caps_t {
        let hvp = self.hvp_state();
        let hessian = self.hessian_state();
        rgpot_derivative_caps_t {
            hvp_analytic: hvp.has_analytic(),
            hvp_ad: hvp.has_ad(),
            hvp_fd: true,
            hessian_analytic: read_callback(&hessian.analytic).is_some(),
            hessian_ad: read_callback(&hessian.ad).is_some(),
            hessian_fd: true,
            hvp_forward_force_calls: 1,
            hvp_central_force_calls: 2,
            hessian_central_force_calls_per_atom: 6,
        }
    }

    /// Requests answered by each derivative scheme so far.
    /// [`Self::reset_hvp_counts`] zeros them.
    pub fn derivative_counts(&self) -> rgpot_derivative_counts_t {
        let hvp = self.hvp_state();
        let hessian = self.hessian_state();
        rgpot_derivative_counts_t {
            hvp_forward: hvp.by_forward.load(Ordering::Relaxed),
            hvp_central: hvp.by_central.load(Ordering::Relaxed),
            hvp_analytic: hvp.by_analytic.load(Ordering::Relaxed),
            hvp_ad: hvp.by_ad.load(Ordering::Relaxed),
            hessian_central: hessian.by_central.load(Ordering::Relaxed),
            hessian_analytic: hessian.by_analytic.load(Ordering::Relaxed),
            hessian_ad: hessian.by_ad.load(Ordering::Relaxed),
            hessian_force_calls: hessian.force_calls.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn reset_hessian_counts(&self) {
        let state = self.hessian_state();
        for counter in [
            &state.by_central,
            &state.by_analytic,
            &state.by_ad,
            &state.force_calls,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    /// Write the dense Hessian at the geometry of `input` to `hessian_out`
    /// and return the scheme that computed it: `EXACT` resolves to
    /// `ANALYTIC` or `AD`, and any other scheme to itself.
    ///
    /// `hessian_out` holds `(3 * n_atoms)^2` doubles. `step` is the central
    /// displacement, and 0 selects
    /// [`RGPOT_HVP_DEFAULT_STEP`](crate::hvp::RGPOT_HVP_DEFAULT_STEP). A
    /// refused scheme, or a `hessian_out` that overlaps a tensor of
    /// `input`, makes no call. A non-finite entry fails every scheme. On a
    /// failure the content of `hessian_out` is unspecified.
    ///
    /// # Safety
    ///
    /// The tensors of `input` must be valid for the call, and the handle's
    /// callbacks safe to call with its `user_data`.
    pub unsafe fn hessian(
        &self,
        input: &rgpot_force_input_t,
        step: f64,
        scheme: rgpot_hessian_scheme_t,
        hessian_out: &mut [f64],
    ) -> Result<rgpot_hessian_scheme_t, HvpError> {
        self.refuse_if_peer_refused()?;
        let n_atoms =
            unsafe { input.n_atoms() }.ok_or_else(|| invalid("positions: tensor is NULL"))?;
        let entries = n_atoms
            .checked_mul(3)
            .and_then(|dim| dim.checked_mul(dim))
            .ok_or_else(|| invalid("(3 * n_atoms)^2 overflows"))?;
        if hessian_out.len() != entries {
            return Err(invalid(format!(
                "hessian_out: expected {entries} elements, got {}",
                hessian_out.len()
            )));
        }
        unsafe { refuse_overlap(input, hessian_out.as_ptr(), hessian_out.len()) }?;
        let dim = n_atoms * 3;
        hessian_len(dim, self.hessian_byte_budget())?;
        let geometry = unsafe { read_geometry(input) }?;
        let h = checked_step(step)?;
        let state = self.hessian_state();
        let analytic_guard = read_callback(&state.analytic);
        let ad_guard = read_callback(&state.ad);
        let scheme = resolve(scheme, analytic_guard.is_some(), ad_guard.is_some())?;
        let call = |callback: HessianCallback, out: *mut f64| unsafe {
            callback(self.pot_user_data, input, out)
        };
        match scheme {
            rgpot_hessian_scheme_t::RGPOT_HESSIAN_ANALYTIC => {
                drop(ad_guard);
                let callback = analytic_guard.expect("resolved against it");
                run_callback("Hessian callback", "Hessian callback", hessian_out, |out| {
                    call(callback, out)
                })?;
                drop(analytic_guard);
                state.by_analytic.fetch_add(1, Ordering::Relaxed);
            }
            rgpot_hessian_scheme_t::RGPOT_HESSIAN_AD => {
                drop(analytic_guard);
                let callback = ad_guard.expect("resolved against it");
                run_callback(
                    "dual-number Hessian callback",
                    "dual-number Hessian callback",
                    hessian_out,
                    |out| call(callback, out),
                )?;
                drop(ad_guard);
                state.by_ad.fetch_add(1, Ordering::Relaxed);
            }
            rgpot_hessian_scheme_t::RGPOT_HESSIAN_CENTRAL => {
                drop(analytic_guard);
                drop(ad_guard);
                let scale = 1.0 / (2.0 * h);
                let mut x = geometry.positions.clone();
                for c in 0..dim {
                    x[c] = geometry.positions[c] + h;
                    let plus = unsafe {
                        self.forces_at(input, x.clone(), geometry.n_atoms, &state.force_calls)
                    }?;
                    x[c] = geometry.positions[c] - h;
                    let minus = unsafe {
                        self.forces_at(input, x.clone(), geometry.n_atoms, &state.force_calls)
                    }?;
                    x[c] = geometry.positions[c];
                    for r in 0..dim {
                        hessian_out[r * dim + c] = (minus.forces[r] - plus.forces[r]) * scale;
                    }
                }
                for r in 0..dim {
                    for c in (r + 1)..dim {
                        let mean = 0.5 * (hessian_out[r * dim + c] + hessian_out[c * dim + r]);
                        hessian_out[r * dim + c] = mean;
                        hessian_out[c * dim + r] = mean;
                    }
                }
                if let Some(i) = hessian_out.iter().position(|v| !v.is_finite()) {
                    return Err(invalid(format!(
                        "central Hessian: non-finite value at element {i}"
                    )));
                }
                state.by_central.fetch_add(1, Ordering::Relaxed);
            }
            rgpot_hessian_scheme_t::RGPOT_HESSIAN_EXACT => unreachable!("resolved above"),
        }
        Ok(scheme)
    }
}

/// Refuse an output of `entries` doubles at `out` that shares memory with a
/// tensor of `input`, which the callbacks read while the Hessian is written.
/// Only addresses are compared, so the check can run before a slice over
/// the output exists; reading tensor data that such a slice overlaps would
/// break its exclusivity.
///
/// # Safety
///
/// The tensors of `input` must be NULL or valid.
pub(crate) unsafe fn refuse_overlap(
    input: &rgpot_force_input_t,
    out: *const f64,
    entries: usize,
) -> Result<(), HvpError> {
    if entries == 0 {
        return Ok(());
    }
    let n_atoms = unsafe { input.n_atoms() }.ok_or_else(|| invalid("positions: tensor is NULL"))?;
    let dim = n_atoms
        .checked_mul(3)
        .ok_or_else(|| invalid("n_atoms * 3 overflows"))?;
    let out = out as usize;
    let f64_bytes = std::mem::size_of::<f64>();
    let out_bytes = entries.saturating_mul(f64_bytes);
    let mut tensors = vec![(
        "positions",
        checked_f64_data(input.positions, "positions", dim).map_err(invalid)? as usize,
        dim.saturating_mul(f64_bytes),
    )];
    if !input.atomic_numbers.is_null() {
        let data =
            checked_i32_data(input.atomic_numbers, "atomic_numbers", n_atoms).map_err(invalid)?;
        tensors.push((
            "atomic_numbers",
            data as usize,
            n_atoms.saturating_mul(std::mem::size_of::<i32>()),
        ));
    }
    if !input.box_matrix.is_null() {
        let data = checked_f64_data(input.box_matrix, "box_matrix", 9).map_err(invalid)?;
        tensors.push(("box_matrix", data as usize, 9 * f64_bytes));
    }
    for (label, start, bytes) in tensors {
        if bytes > 0 && start < out.saturating_add(out_bytes) && out < start.saturating_add(bytes) {
            return Err(invalid(format!("hessian_out overlaps the {label} tensor")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};

    #[test]
    fn an_empty_tensor_inside_the_output_does_not_overlap_it() {
        let mut out = [0.0; 9];
        let mut z = [78];
        let mut cell = [0.0; 9];
        let inside = unsafe { out.as_mut_ptr().add(1) };
        let mut input = |n_atoms| unsafe {
            rgpot_force_input_create(n_atoms, inside, z.as_mut_ptr(), cell.as_mut_ptr())
        };
        let mut empty = input(0);
        assert!(unsafe { refuse_overlap(&empty, out.as_ptr(), out.len()) }.is_ok());

        let mut one = input(1);
        let err = unsafe { refuse_overlap(&one, out.as_ptr(), out.len()) }.unwrap_err();
        assert_eq!(err.to_string(), "hessian_out overlaps the positions tensor");
        unsafe {
            rgpot_force_input_free(&mut empty);
            rgpot_force_input_free(&mut one);
        }
    }

    #[test]
    fn a_byte_count_that_does_not_fit_in_usize_is_refused() {
        let dim = 1usize << (usize::BITS / 2);
        let err = hessian_len(dim, None).expect_err("overflow");
        let text = err.to_string();
        assert!(text.contains("overflows"), "{text}");
    }

    #[test]
    fn a_caller_budget_is_the_only_practical_ceiling() {
        assert_eq!(hessian_len(3, None).expect("no budget"), 9);
        let err = hessian_len(3, Some(8)).expect_err("over budget");
        assert!(err.to_string().contains("caller budget"));
        assert_eq!(hessian_len(1, Some(8)).expect("one double fits"), 1);
        assert!(hessian_len(1, Some(0)).is_err());
    }
}
