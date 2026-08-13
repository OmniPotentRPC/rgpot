// MIT License
// Copyright 2023--present rgpot developers

//! rgpot as a strict superset of eindir: first-member struct embedding.
//!
//! `rgpot_potential_t` embeds `eindir_objective_t` as its FIRST member, so a
//! `rgpot_potential_t*` IS-A `eindir_objective_t*` at the C ABI with zero cost:
//!
//! ```c
//! rgpot_potential_t *pot = rgpot_potential_new_eindir(...);
//! eindir_objective_t *obj = (eindir_objective_t*)pot;  // zero-cost cast
//! double val;
//! eindir_objective_eval(obj, x, &val);                 // works
//! ```
//!
//! The eval_fn / grad_fn stored in the embedded base call through to the
//! rgpot callback, so evaluation logic lives once -- there is no duplicate
//! eval path and no conversion method.

use std::os::raw::c_void;
use std::sync::Mutex;

use dlpk::sys::DLManagedTensorVersioned;

use crate::potential::PotentialCallback;
use crate::status::rgpot_status_t;
use crate::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3,
    rgpot_tensor_cpu_i32_1d, rgpot_tensor_free,
};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

// ---------------------------------------------------------------------------
// eindir C ABI types -- the real ones, from the shared eindir-core crate
//
// rgpot does NOT mirror eindir's #[repr(C)] types. It re-uses eindir-core's own
// definitions, so `rgpot_potential_t` embeds the exact same `eindir_objective_t`
// and the eval/grad entry points resolve through one compilation of eindir-core.
// That is what lets a downstream Rust consumer (e.g. anneal-core) link both
// crates without the duplicate-symbol / two-Rust-runtime conflict that a
// separate prebuilt `libeindir_core.a` would cause.
// ---------------------------------------------------------------------------

pub use eindir_core::ffi::eindir_status_t;
pub use eindir_core::ffi::{
    eindir_objective_eval, eindir_objective_grad, eindir_objective_has_grad,
    eindir_objective_t, EindirEvalFn, EindirFreeFn, EindirGradFn,
};

// ---------------------------------------------------------------------------
// rgpot_potential_t: rgpot_potential_t* IS-A eindir_objective_t*
// ---------------------------------------------------------------------------

/// Superset potential: embeds `eindir_objective_t` as its first member.
///
/// Because `base` is first and both structs are `#[repr(C)]`, casting a
/// `*mut rgpot_potential_t` to `*mut eindir_objective_t` is defined behaviour
/// at the C ABI.  No conversion method is needed.
#[repr(C)]
pub struct rgpot_potential_t {
    /// Embedded eindir base: the IS-A relationship lives here.
    /// MUST remain the first field.
    pub base: eindir_objective_t,
    /// The underlying rgpot force/energy callback.
    pub callback: PotentialCallback,
    /// Opaque context for `callback` (distinct from `base.user_data`).
    pub pot_user_data: *mut c_void,
    /// Optional destructor for `pot_user_data`.
    pub pot_free_fn: Option<unsafe extern "C" fn(*mut c_void)>,
    /// Atom count; set by `rgpot_potential_new_eindir`.
    pub n_atoms: usize,
    /// Atomic numbers, heap-allocated (len = n_atoms), owned by this struct.
    pub atomic_numbers: *mut i32,
    /// Box matrix (column-major, 3x3), copied at construction.
    pub box_matrix: [f64; 9],
    /// Energy/gradient result shared by a matching eval-then-grad request.
    fused_cache: Mutex<Option<FusedEvaluation>>,
}

struct FusedEvaluation {
    positions: Vec<f64>,
    gradient: Vec<f64>,
}

// Safety: user pointers are opaque; caller guarantees thread safety.
unsafe impl Send for rgpot_potential_t {}

// ---------------------------------------------------------------------------
// Eval / grad callbacks wired into the embedded base
//
// base.user_data = ptr to the rgpot_potential_t itself (set in the constructor
// after Box::into_raw so the address is stable).  The callbacks cast it back
// to access molecular data and the rgpot callback.
// ---------------------------------------------------------------------------

unsafe extern "C" fn rgpot_eval_cb(
    user_data: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> eindir_status_t {
    let pot = unsafe { &*(user_data as *const rgpot_potential_t) };
    let xt = unsafe { &(*x).dl_tensor };
    let n = pot.n_atoms * 3;
    let x_data = unsafe { std::slice::from_raw_parts(xt.data as *const f64, n) };
    let mut pos = x_data.to_vec();
    let mut atmnrs =
        unsafe { std::slice::from_raw_parts(pot.atomic_numbers, pot.n_atoms) }.to_vec();
    let mut box_ = pot.box_matrix;
    let input = rgpot_force_input_t {
        positions: unsafe {
            rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), pot.n_atoms as i64, 3)
        },
        atomic_numbers: unsafe {
            rgpot_tensor_cpu_i32_1d(atmnrs.as_mut_ptr(), pot.n_atoms as i64)
        },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(box_.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    let status =
        unsafe { (pot.callback)(pot.pot_user_data, &input, &mut output) };
    let fused = if status == rgpot_status_t::RGPOT_SUCCESS && !output.forces.is_null() {
        let ft = unsafe { &(*output.forces).dl_tensor };
        let forces = unsafe { std::slice::from_raw_parts(ft.data as *const f64, n) };
        Some(FusedEvaluation {
            positions: x_data.to_vec(),
            gradient: forces.iter().map(|force| -*force).collect(),
        })
    } else {
        None
    };
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    if status != rgpot_status_t::RGPOT_SUCCESS {
        *pot.fused_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        return eindir_status_t::EINDIR_INTERNAL_ERROR;
    }
    *pot.fused_cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = fused;
    unsafe { *value_out = output.energy };
    eindir_status_t::EINDIR_SUCCESS
}

unsafe extern "C" fn rgpot_grad_cb(
    user_data: *mut c_void,
    x: *const DLManagedTensorVersioned,
    grad_out: *mut DLManagedTensorVersioned,
) -> eindir_status_t {
    let pot = unsafe { &*(user_data as *const rgpot_potential_t) };
    let xt = unsafe { &(*x).dl_tensor };
    let n = pot.n_atoms * 3;
    let x_data = unsafe { std::slice::from_raw_parts(xt.data as *const f64, n) };
    let cached = pot
        .fused_cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    if let Some(cached) = cached {
        if cached.positions == x_data {
            let gt = unsafe { &(*grad_out).dl_tensor };
            let dst = unsafe { std::slice::from_raw_parts_mut(gt.data as *mut f64, n) };
            dst.copy_from_slice(&cached.gradient);
            return eindir_status_t::EINDIR_SUCCESS;
        }
    }
    let mut pos = x_data.to_vec();
    let mut atmnrs =
        unsafe { std::slice::from_raw_parts(pot.atomic_numbers, pot.n_atoms) }.to_vec();
    let mut box_ = pot.box_matrix;
    let input = rgpot_force_input_t {
        positions: unsafe {
            rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), pot.n_atoms as i64, 3)
        },
        atomic_numbers: unsafe {
            rgpot_tensor_cpu_i32_1d(atmnrs.as_mut_ptr(), pot.n_atoms as i64)
        },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(box_.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    let status =
        unsafe { (pot.callback)(pot.pot_user_data, &input, &mut output) };
    if status == rgpot_status_t::RGPOT_SUCCESS && !output.forces.is_null() {
        let ft = unsafe { &(*output.forces).dl_tensor };
        let src = unsafe { std::slice::from_raw_parts(ft.data as *const f64, n) };
        let gt = unsafe { &(*grad_out).dl_tensor };
        let dst = unsafe { std::slice::from_raw_parts_mut(gt.data as *mut f64, n) };
        for i in 0..n {
            dst[i] = -src[i]; // gradient = -force
        }
    }
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    if status != rgpot_status_t::RGPOT_SUCCESS {
        return eindir_status_t::EINDIR_INTERNAL_ERROR;
    }
    eindir_status_t::EINDIR_SUCCESS
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

/// Create a potential that is ALSO a valid eindir objective.
///
/// The returned pointer is simultaneously a `rgpot_potential_t*` and (via
/// zero-cost cast to the first member) an `eindir_objective_t*`.
///
/// - `callback`: the rgpot force/energy callback.
/// - `user_data` / `free_fn`: opaque context for the rgpot callback.
/// - `n_atoms`, `atomic_numbers`, `box_matrix`: molecular context.
/// - `bounds_low`, `bounds_high`: f64 arrays of length `n_atoms * 3`
///   (may be NULL; defaults to [-50, 50] per dimension).
///
/// The caller must eventually pass the returned pointer to
/// [`rgpot_potential_free`].
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_new_eindir(
    callback: PotentialCallback,
    user_data: *mut c_void,
    free_fn: Option<unsafe extern "C" fn(*mut c_void)>,
    n_atoms: usize,
    atomic_numbers: *const i32,
    box_matrix: *const f64,
    bounds_low: *const f64,
    bounds_high: *const f64,
) -> *mut rgpot_potential_t {
    let dim = n_atoms * 3;

    let alloc_f64 = |src: *const f64, default: f64| -> *mut f64 {
        let mut v = vec![default; dim];
        if !src.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(src, v.as_mut_ptr(), dim) };
        }
        let ptr = v.as_mut_ptr();
        std::mem::forget(v);
        ptr
    };

    let low = alloc_f64(bounds_low, -50.0);
    let high = alloc_f64(bounds_high, 50.0);

    let atmnrs = {
        let mut v = vec![0i32; n_atoms];
        if !atomic_numbers.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(atomic_numbers, v.as_mut_ptr(), n_atoms) };
        }
        let ptr = v.as_mut_ptr();
        std::mem::forget(v);
        ptr
    };

    let mut box_arr = [0.0f64; 9];
    if !box_matrix.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(box_matrix, box_arr.as_mut_ptr(), 9) };
    }

    let pot = Box::new(rgpot_potential_t {
        base: eindir_objective_t {
            dim,
            low,
            high,
            eval_fn: rgpot_eval_cb,
            grad_fn: Some(rgpot_grad_cb),
            // user_data points to self; set below after Box::into_raw.
            user_data: std::ptr::null_mut(),
            // No free_fn in the embedded base: rgpot_potential_free handles all cleanup.
            free_fn: None,
        },
        callback,
        pot_user_data: user_data,
        pot_free_fn: free_fn,
        n_atoms,
        atomic_numbers: atmnrs,
        box_matrix: box_arr,
        fused_cache: Mutex::new(None),
    });
    let ptr = Box::into_raw(pot);
    // Self-referential: the eindir base's user_data points to the owning struct
    // so the callbacks can access molecular data via a back-cast.
    unsafe { (*ptr).base.user_data = ptr as *mut c_void };
    ptr
}

/// Free a potential created by [`rgpot_potential_new_eindir`].
///
/// Calls `pot_free_fn(pot_user_data)` if provided, frees all owned arrays,
/// then frees the struct.  Do NOT call `eindir_objective_free` on the embedded
/// base; call this function.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_free_eindir(pot: *mut rgpot_potential_t) {
    if pot.is_null() {
        return;
    }
    let p = unsafe { &*pot };
    if let Some(ff) = p.pot_free_fn {
        if !p.pot_user_data.is_null() {
            unsafe { ff(p.pot_user_data) };
        }
    }
    // Free embedded base's bounds arrays.
    let dim = p.base.dim;
    if !p.base.low.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.base.low, dim, dim)) };
    }
    if !p.base.high.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.base.high, dim, dim)) };
    }
    // Free molecular arrays.
    if !p.atomic_numbers.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.atomic_numbers, p.n_atoms, p.n_atoms)) };
    }
    unsafe { drop(Box::from_raw(pot)) };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::create_owned_f64_tensor;
    use eindir_core::ffi::EindirObjectiveWrapper;
    use eindir_core::gradient::DifferentiableObjective;
    use ndarray::Array1;

    struct Ctx1d {
        shape: [i64; 1],
        strides: [i64; 1],
    }

    unsafe extern "C" fn del1d(ptr: *mut DLManagedTensorVersioned) {
        if ptr.is_null() {
            return;
        }
        let ctx = unsafe { (*ptr).manager_ctx.cast::<Ctx1d>() };
        if !ctx.is_null() {
            drop(unsafe { Box::from_raw(ctx) });
        }
        drop(unsafe { Box::from_raw(ptr) });
    }

    fn make_1d(data: *mut f64, n: usize) -> *mut DLManagedTensorVersioned {
        use dlpk::sys::{
            DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLPackVersion, DLTensor,
        };
        let mut ctx = Box::new(Ctx1d {
            shape: [n as i64],
            strides: [1],
        });
        let t = DLTensor {
            data: data.cast(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: DLDataType {
                code: DLDataTypeCode::kDLFloat,
                bits: 64,
                lanes: 1,
            },
            shape: ctx.shape.as_mut_ptr(),
            strides: ctx.strides.as_mut_ptr(),
            byte_offset: 0,
        };
        let managed = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(del1d),
            flags: 0,
            dl_tensor: t,
        });
        Box::into_raw(managed)
    }

    // Mock rgpot callback: energy = sum(positions), forces = -1 everywhere.
    unsafe extern "C" fn mock_energy_callback(
        _user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };
        let n = unsafe { inp.n_atoms() }.unwrap_or(0);
        let pt = unsafe { &(*inp.positions).dl_tensor };
        let pos = unsafe { std::slice::from_raw_parts(pt.data as *const f64, n * 3) };
        out.energy = pos.iter().sum();
        out.variance = 0.0;
        out.forces = create_owned_f64_tensor(vec![-1.0f64; n * 3], vec![n as i64, 3]);
        rgpot_status_t::RGPOT_SUCCESS
    }

    unsafe extern "C" fn counting_energy_callback(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let calls = unsafe { &mut *(user_data as *mut usize) };
        *calls += 1;
        unsafe { mock_energy_callback(std::ptr::null_mut(), input, output) }
    }

    #[test]
    fn fused_value_and_gradient_executes_one_force_request() {
        let n_atoms = 2usize;
        let atmnrs = [1i32, 1];
        let box_ = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0];
        let mut calls = 0usize;
        let pot = unsafe {
            rgpot_potential_new_eindir(
                counting_energy_callback,
                (&mut calls as *mut usize).cast(),
                None,
                n_atoms,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert!(!pot.is_null());
        let objective = unsafe { EindirObjectiveWrapper::new(&(*pot).base) };
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let (energy, gradient) = objective.value_and_gradient(x.view());

        assert_eq!(energy, 21.0);
        assert!(gradient.iter().all(|&value| value == 1.0));
        assert_eq!(calls, 1, "a fused request must execute the engine once");
        unsafe { rgpot_potential_free_eindir(pot) };
    }

    /// The IS-A cast: a rgpot_potential_t* is passed to eindir functions by
    /// casting to eindir_objective_t* -- no conversion method, no duplicate
    /// eval path.  Energy = sum(positions) = 1+2+3+4+5+6 = 21.
    #[test]
    fn rgpot_is_eindir_objective() {
        let n_atoms = 2usize;
        let dim = n_atoms * 3;
        let atmnrs = [1i32, 1];
        let box_ = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let low_data = vec![-50.0f64; dim];
        let high_data = vec![50.0f64; dim];

        let pot = unsafe {
            rgpot_potential_new_eindir(
                mock_energy_callback,
                std::ptr::null_mut(),
                None,
                n_atoms,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                low_data.as_ptr(),
                high_data.as_ptr(),
            )
        };
        assert!(!pot.is_null());

        // THE IS-A CAST: zero cost, no conversion method.
        let obj = pot as *mut eindir_objective_t;

        let mut x_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_t = make_1d(x_data.as_mut_ptr(), dim);

        let mut value = 0.0f64;
        let s = unsafe { eindir_objective_eval(obj, x_t, &mut value) };
        assert_eq!(s, eindir_status_t::EINDIR_SUCCESS);
        assert_eq!(value, 21.0); // 1+2+3+4+5+6

        assert_eq!(unsafe { eindir_objective_has_grad(obj) }, 1);

        // Gradient = -forces = 1.0 everywhere.
        let mut g_data = vec![0.0f64; dim];
        let g_t = make_1d(g_data.as_mut_ptr(), dim);
        let s = unsafe { eindir_objective_grad(obj, x_t, g_t) };
        assert_eq!(s, eindir_status_t::EINDIR_SUCCESS);
        assert!(g_data.iter().all(|&v| v == 1.0));

        unsafe {
            del1d(x_t);
            del1d(g_t);
            rgpot_potential_free_eindir(pot);
        }
    }

    /// Verify struct field values survive a round-trip through the IS-A cast
    /// without calling into the shared library.
    #[test]
    fn base_fields_via_cast() {
        let n_atoms = 2usize;
        let atmnrs = [1i32, 1];
        let box_ = [10.0f64; 9];
        let pot = unsafe {
            rgpot_potential_new_eindir(
                mock_energy_callback,
                std::ptr::null_mut(),
                None,
                n_atoms,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert!(!pot.is_null());
        let obj = pot as *const eindir_objective_t;
        let dim = unsafe { (*obj).dim };
        let user_data = unsafe { (*obj).user_data };
        assert_eq!(dim, 6); // n_atoms * 3
        assert_eq!(user_data, pot as *mut c_void);
        unsafe { rgpot_potential_free_eindir(pot) };
    }

    /// Verify that the embedded base is the first member at offset 0.
    #[test]
    fn base_is_first_member() {
        use std::mem::offset_of;
        assert_eq!(offset_of!(rgpot_potential_t, base), 0);
    }

    /// Existing rgpot calculations still work through the direct callback path.
    #[test]
    fn direct_callback_path_still_works() {
        let n_atoms = 3usize;
        let dim = n_atoms * 3;
        let atmnrs = [1i32; 3];
        let box_ = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

        let pot = unsafe {
            rgpot_potential_new_eindir(
                mock_energy_callback,
                std::ptr::null_mut(),
                None,
                n_atoms,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };

        // Evaluate through the eindir path to check correctness.
        let obj = pot as *mut eindir_objective_t;
        let mut positions = vec![1.0f64; dim];
        let x_t = make_1d(positions.as_mut_ptr(), dim);
        let mut value = 0.0f64;
        let s = unsafe { eindir_objective_eval(obj, x_t, &mut value) };
        assert_eq!(s, eindir_status_t::EINDIR_SUCCESS);
        assert_eq!(value, dim as f64); // sum of 9 ones

        unsafe {
            del1d(x_t);
            rgpot_potential_free_eindir(pot);
        }
    }

    /// The gradient is the negative of the forces (sign convention).
    #[test]
    fn bridge_grad_returns_neg_forces() {
        let n_atoms = 2usize;
        let dim = n_atoms * 3;
        let atmnrs = [1i32; 2];
        let box_ = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

        let pot = unsafe {
            rgpot_potential_new_eindir(
                mock_energy_callback,
                std::ptr::null_mut(),
                None,
                n_atoms,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };

        let obj = pot as *mut eindir_objective_t;
        let mut x_data = vec![0.5f64; dim];
        let x_t = make_1d(x_data.as_mut_ptr(), dim);
        let mut g_data = vec![0.0f64; dim];
        let g_t = make_1d(g_data.as_mut_ptr(), dim);

        let s = unsafe { eindir_objective_grad(obj, x_t, g_t) };
        assert_eq!(s, eindir_status_t::EINDIR_SUCCESS);
        // mock_energy_callback sets forces = -1.0; gradient = -force = 1.0.
        assert!(g_data.iter().all(|&v| v == 1.0), "grad = -(-1) = 1");

        unsafe {
            del1d(x_t);
            del1d(g_t);
            rgpot_potential_free_eindir(pot);
        }
    }
}
