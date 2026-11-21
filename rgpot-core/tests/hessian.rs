// MIT License
// Copyright 2023--present rgpot developers

//! A quadratic potential whose Hessian is the constant matrix `A`.
//! The dense entry point must use a registered analytic callback or the
//! analytic product, and must refuse rather than difference forces.

use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use rgpot_core::c_api::hessian::{
    rgpot_potential_hessian, rgpot_potential_hessian_counts, rgpot_potential_reset_hessian_counts,
    rgpot_potential_set_hessian_callback,
};
use rgpot_core::c_api::hvp::{
    rgpot_potential_hvp, rgpot_potential_hvp_counts, rgpot_potential_hvp_invalidate,
    rgpot_potential_set_hvp_callback,
};
use rgpot_core::c_api::potential::{
    rgpot_potential_calculate, rgpot_potential_free, rgpot_potential_new,
};
use rgpot_core::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
use rgpot_core::hessian::rgpot_hessian_counts_t;
use rgpot_core::hvp::{rgpot_hvp_counts_t, rgpot_hvp_scheme_t};
use rgpot_core::potential::{rgpot_potential_t, PotentialCallback};
use rgpot_core::status::{rgpot_last_error, rgpot_status_t};
use rgpot_core::tensor::{rgpot_tensor_free, rgpot_tensor_owned_cpu_f64_2d};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

const A: [f64; 9] = [2.0, 0.5, 0.0, 0.5, 3.0, -0.25, 0.0, -0.25, 4.0];
const AUTO: i32 = rgpot_hvp_scheme_t::RGPOT_HVP_AUTO as i32;
const FORWARD: i32 = rgpot_hvp_scheme_t::RGPOT_HVP_FORWARD as i32;
const ANALYTIC: i32 = rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC as i32;

struct Quad {
    force_calls: AtomicUsize,
    hessian_calls: AtomicUsize,
    hvp_calls: AtomicUsize,
}

impl Quad {
    fn new() -> Self {
        Self {
            force_calls: AtomicUsize::new(0),
            hessian_calls: AtomicUsize::new(0),
            hvp_calls: AtomicUsize::new(0),
        }
    }
}

fn matvec(matrix: &[f64], vector: &[f64]) -> Vec<f64> {
    (0..vector.len())
        .map(|row| {
            (0..vector.len())
                .map(|column| matrix[row * vector.len() + column] * vector[column])
                .sum()
        })
        .collect()
}

fn read_positions(input: &rgpot_force_input_t) -> Vec<f64> {
    let n = unsafe { input.n_atoms() }.unwrap();
    unsafe { std::slice::from_raw_parts((*input.positions).dl_tensor.data as *const f64, 3 * n) }
        .to_vec()
}

unsafe extern "C" fn quad_force(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let quad = unsafe { &*(user_data as *const Quad) };
    let x = read_positions(unsafe { &*input });
    quad.force_calls.fetch_add(1, Ordering::Relaxed);
    let gradient = matvec(&A, &x);
    let energy = 0.5 * gradient.iter().zip(&x).map(|(g, x)| g * x).sum::<f64>();
    let forces: Vec<f64> = gradient.iter().map(|g| -g).collect();
    let out = unsafe { &mut *output };
    out.energy = energy;
    out.variance = 0.0;
    out.forces = unsafe { rgpot_tensor_owned_cpu_f64_2d(forces.as_ptr(), 1, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn quad_hessian(
    user_data: *mut c_void,
    _input: *const rgpot_force_input_t,
    hessian_out: *mut f64,
) -> rgpot_status_t {
    let quad = unsafe { &*(user_data as *const Quad) };
    quad.hessian_calls.fetch_add(1, Ordering::Relaxed);
    unsafe { std::ptr::copy_nonoverlapping(A.as_ptr(), hessian_out, A.len()) };
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn quad_hvp(
    user_data: *mut c_void,
    _input: *const rgpot_force_input_t,
    direction: *const f64,
    hv_out: *mut f64,
) -> rgpot_status_t {
    let quad = unsafe { &*(user_data as *const Quad) };
    quad.hvp_calls.fetch_add(1, Ordering::Relaxed);
    let direction = unsafe { std::slice::from_raw_parts(direction, 3) };
    let product = matvec(&A, direction);
    unsafe { std::ptr::copy_nonoverlapping(product.as_ptr(), hv_out, product.len()) };
    rgpot_status_t::RGPOT_SUCCESS
}

struct Handle {
    pot: *mut rgpot_potential_t,
    quad: Box<Quad>,
}

impl Handle {
    fn bare() -> Self {
        let quad = Box::new(Quad::new());
        let pot = unsafe {
            rgpot_potential_new(
                quad_force as PotentialCallback,
                &*quad as *const Quad as *mut c_void,
                None,
            )
        };
        assert!(!pot.is_null());
        Self { pot, quad }
    }
}

struct Geom {
    x: [f64; 3],
    z: [i32; 1],
    cell: [f64; 9],
}

impl Geom {
    fn new(x: [f64; 3]) -> Self {
        let mut cell = [0.0; 9];
        cell[0] = 10.0;
        cell[4] = 10.0;
        cell[8] = 10.0;
        Self { x, z: [1], cell }
    }

    fn input(&mut self) -> rgpot_force_input_t {
        unsafe {
            rgpot_force_input_create(
                1,
                self.x.as_mut_ptr(),
                self.z.as_mut_ptr(),
                self.cell.as_mut_ptr(),
            )
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { rgpot_potential_free(self.pot) };
    }
}

fn last_error() -> String {
    let ptr = unsafe { rgpot_last_error() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn a_hessian_callback_serves_the_matrix_and_auto_products_without_forces() {
    let handle = Handle::bare();
    let status = unsafe { rgpot_potential_set_hessian_callback(handle.pot, Some(quad_hessian)) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let mut geom = Geom::new([0.3, -0.2, 0.7]);
    let mut input = geom.input();
    let mut matrix = [0.0; 9];
    let status = unsafe { rgpot_potential_hessian(handle.pot, &input, matrix.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(matrix, A);
    let status = unsafe { rgpot_potential_hessian(handle.pot, &input, matrix.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(handle.quad.hessian_calls.load(Ordering::Relaxed), 1);

    let direction = [1.0, -2.0, 0.5];
    let mut product = [0.0; 3];
    let status = unsafe {
        rgpot_potential_hvp(
            handle.pot,
            &input,
            direction.as_ptr(),
            0.0,
            AUTO,
            product.as_mut_ptr(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let expected = matvec(&A, &direction);
    for (got, want) in product.iter().zip(&expected) {
        assert!((got - want).abs() <= 1e-12, "{got} vs {want}");
    }
    assert_eq!(handle.quad.force_calls.load(Ordering::Relaxed), 0);

    let mut counted = rgpot_hessian_counts_t::default();
    unsafe { rgpot_potential_hessian_counts(handle.pot, &mut counted) };
    assert_eq!(counted.evaluations, 1);
    assert!(counted.cache_hits >= 1);
    assert_eq!(counted.products, 1);
    assert_eq!(counted.from_hvp, 0);

    let status = unsafe {
        rgpot_potential_hvp(
            handle.pot,
            &input,
            direction.as_ptr(),
            1.0e-4,
            FORWARD,
            product.as_mut_ptr(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert!(handle.quad.force_calls.load(Ordering::Relaxed) >= 2);
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn an_hvp_callback_assembles_the_hessian_and_a_second_call_is_cached() {
    let handle = Handle::bare();
    let status = unsafe { rgpot_potential_set_hvp_callback(handle.pot, Some(quad_hvp)) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let mut geom = Geom::new([0.3, -0.2, 0.7]);
    let mut input = geom.input();
    let mut matrix = [0.0; 9];
    let status = unsafe { rgpot_potential_hessian(handle.pot, &input, matrix.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
    assert_eq!(matrix, A);
    assert_eq!(handle.quad.hvp_calls.load(Ordering::Relaxed), 3);
    assert_eq!(handle.quad.force_calls.load(Ordering::Relaxed), 0);
    let status = unsafe { rgpot_potential_hessian(handle.pot, &input, matrix.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(handle.quad.hvp_calls.load(Ordering::Relaxed), 3);
    unsafe { rgpot_potential_hvp_invalidate(handle.pot) };
    let status = unsafe { rgpot_potential_hessian(handle.pot, &input, matrix.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(handle.quad.hvp_calls.load(Ordering::Relaxed), 6);
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn neither_callback_is_a_typed_refusal_and_makes_no_force_call() {
    let handle = Handle::bare();
    let mut geom = Geom::new([0.3, -0.2, 0.7]);
    let mut input = geom.input();
    let mut matrix = [0.0; 9];
    let status = unsafe { rgpot_potential_hessian(handle.pot, &input, matrix.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_NO_HESSIAN);
    assert!(
        last_error().contains("no analytic Hessian"),
        "{}",
        last_error()
    );
    assert_eq!(handle.quad.force_calls.load(Ordering::Relaxed), 0);
    assert_eq!(matrix, [0.0; 9]);
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn auto_prefers_the_product_callback_when_both_are_registered() {
    let handle = Handle::bare();
    unsafe {
        assert_eq!(
            rgpot_potential_set_hessian_callback(handle.pot, Some(quad_hessian)),
            rgpot_status_t::RGPOT_SUCCESS
        );
        assert_eq!(
            rgpot_potential_set_hvp_callback(handle.pot, Some(quad_hvp)),
            rgpot_status_t::RGPOT_SUCCESS
        );
    }
    let mut geom = Geom::new([0.1, 0.2, -0.3]);
    let mut input = geom.input();
    let direction = [0.4, -0.1, 0.2];
    let mut product = [0.0; 3];
    let status = unsafe {
        rgpot_potential_hvp(
            handle.pot,
            &input,
            direction.as_ptr(),
            0.0,
            AUTO,
            product.as_mut_ptr(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(handle.quad.hvp_calls.load(Ordering::Relaxed), 1);
    assert_eq!(handle.quad.hessian_calls.load(Ordering::Relaxed), 0);
    let mut hvp_counts = rgpot_hvp_counts_t::default();
    unsafe { rgpot_potential_hvp_counts(handle.pot, &mut hvp_counts) };
    assert_eq!(hvp_counts.analytic, 1);
    assert_eq!(hvp_counts.force_calls, 0);

    let analytic = unsafe {
        rgpot_potential_hvp(
            handle.pot,
            &input,
            direction.as_ptr(),
            0.0,
            ANALYTIC,
            product.as_mut_ptr(),
        )
    };
    assert_eq!(analytic, rgpot_status_t::RGPOT_SUCCESS);

    unsafe { rgpot_potential_reset_hessian_counts(handle.pot) };
    let mut matrix = [0.0; 9];
    let status = unsafe { rgpot_potential_hessian(handle.pot, &input, matrix.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(matrix, A);
    assert_eq!(handle.quad.hessian_calls.load(Ordering::Relaxed), 1);
    let mut counted = rgpot_hessian_counts_t::default();
    unsafe { rgpot_potential_hessian_counts(handle.pot, &mut counted) };
    assert_eq!(counted.evaluations, 1);
    assert_eq!(counted.from_hvp, 0);
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn analytic_scheme_uses_a_dense_hessian_when_that_is_the_exact_source() {
    let handle = Handle::bare();
    unsafe { rgpot_potential_set_hessian_callback(handle.pot, Some(quad_hessian)) };
    let mut geom = Geom::new([0.1, 0.2, -0.3]);
    let mut input = geom.input();
    let direction = [1.0, 0.0, 0.0];
    let mut product = [0.0; 3];
    let status = unsafe {
        rgpot_potential_hvp(
            handle.pot,
            &input,
            direction.as_ptr(),
            0.0,
            ANALYTIC,
            product.as_mut_ptr(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(product, [2.0, 0.5, 0.0]);
    assert_eq!(handle.quad.force_calls.load(Ordering::Relaxed), 0);
    assert_eq!(handle.quad.hessian_calls.load(Ordering::Relaxed), 1);
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn a_caller_byte_budget_refuses_before_the_callback() {
    use rgpot_core::c_api::hessian::rgpot_potential_set_hessian_byte_budget;
    let handle = Handle::bare();
    unsafe { rgpot_potential_set_hessian_callback(handle.pot, Some(quad_hessian)) };
    let budget = 8u64;
    let status = unsafe { rgpot_potential_set_hessian_byte_budget(handle.pot, &budget) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let mut geom = Geom::new([0.1, 0.2, -0.3]);
    let mut input = geom.input();
    let mut scratch = [0.0; 9];
    let status = unsafe { rgpot_potential_hessian(handle.pot, &input, scratch.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert!(last_error().contains("caller budget"));
    assert_eq!(handle.quad.hessian_calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        unsafe { rgpot_potential_set_hessian_byte_budget(handle.pot, std::ptr::null()) },
        rgpot_status_t::RGPOT_SUCCESS
    );
    let status = unsafe { rgpot_potential_hessian(handle.pot, &input, scratch.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(handle.quad.hessian_calls.load(Ordering::Relaxed), 1);
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn a_zero_atom_hessian_without_a_provider_is_refused() {
    let handle = Handle::bare();
    let mut positions = [];
    let mut species = [];
    let mut cell = [0.0; 9];
    let mut input = unsafe {
        rgpot_force_input_create(
            0,
            positions.as_mut_ptr(),
            species.as_mut_ptr(),
            cell.as_mut_ptr(),
        )
    };
    let mut scratch = [0.0];
    let status = unsafe { rgpot_potential_hessian(handle.pot, &input, scratch.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_NO_HESSIAN);
    assert!(last_error().contains("no analytic Hessian"));
    unsafe { rgpot_force_input_free(&mut input) };
}

struct Gate {
    entered: AtomicBool,
    release: AtomicBool,
    finished: AtomicBool,
}

unsafe extern "C" fn idle_force(
    _user_data: *mut c_void,
    _input: *const rgpot_force_input_t,
    _out: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn gated_hessian(
    user_data: *mut c_void,
    _input: *const rgpot_force_input_t,
    hessian_out: *mut f64,
) -> rgpot_status_t {
    let gate = unsafe { &*(user_data as *const Gate) };
    gate.entered.store(true, Ordering::Release);
    while !gate.release.load(Ordering::Acquire) {
        std::thread::yield_now();
    }
    for value in unsafe { std::slice::from_raw_parts_mut(hessian_out, 9) } {
        *value = 1.0;
    }
    gate.finished.store(true, Ordering::Release);
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn second_hessian(
    _user_data: *mut c_void,
    _input: *const rgpot_force_input_t,
    hessian_out: *mut f64,
) -> rgpot_status_t {
    for value in unsafe { std::slice::from_raw_parts_mut(hessian_out, 9) } {
        *value = 2.0;
    }
    rgpot_status_t::RGPOT_SUCCESS
}

type DenseCallback =
    unsafe extern "C" fn(*mut c_void, *const rgpot_force_input_t, *mut f64) -> rgpot_status_t;

/// Start a dense evaluation that blocks inside the callback, ask a second
/// thread to retire that callback, and check that the retirement waits for
/// the in-flight call. Returns the handle, the matrix the blocked call
/// produced, and whether the old call had finished when retirement returned.
fn retire_while_in_flight(
    replacement: Option<DenseCallback>,
) -> (*mut rgpot_potential_t, [f64; 9], bool) {
    let gate = Arc::new(Gate {
        entered: AtomicBool::new(false),
        release: AtomicBool::new(false),
        finished: AtomicBool::new(false),
    });
    let pot = unsafe {
        rgpot_potential_new(
            idle_force as PotentialCallback,
            Arc::as_ptr(&gate) as *mut c_void,
            None,
        )
    };
    assert!(!pot.is_null());
    unsafe { rgpot_potential_set_hessian_callback(pot, Some(gated_hessian)) };
    let pot_addr = pot as usize;
    let worker = std::thread::spawn(move || {
        let mut geom = Geom::new([0.1, 0.2, -0.3]);
        let mut input = geom.input();
        let mut matrix = [0.0; 9];
        let status = unsafe {
            rgpot_potential_hessian(
                pot_addr as *mut rgpot_potential_t,
                &input,
                matrix.as_mut_ptr(),
            )
        };
        unsafe { rgpot_force_input_free(&mut input) };
        (status, matrix)
    });
    while !gate.entered.load(Ordering::Acquire) {
        std::thread::yield_now();
    }

    let retiring = Arc::new(AtomicBool::new(false));
    let returned = Arc::new(AtomicBool::new(false));
    let retirer = {
        let gate = Arc::clone(&gate);
        let retiring = Arc::clone(&retiring);
        let returned = Arc::clone(&returned);
        std::thread::spawn(move || {
            retiring.store(true, Ordering::Release);
            let status = unsafe {
                rgpot_potential_set_hessian_callback(
                    pot_addr as *mut rgpot_potential_t,
                    replacement,
                )
            };
            assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
            let old_call_finished = gate.finished.load(Ordering::Acquire);
            returned.store(true, Ordering::Release);
            old_call_finished
        })
    };
    while !retiring.load(Ordering::Acquire) {
        std::thread::yield_now();
    }
    std::thread::sleep(std::time::Duration::from_millis(150));
    assert!(
        !returned.load(Ordering::Acquire),
        "retiring the callback returned while the old callback was still running"
    );

    gate.release.store(true, Ordering::Release);
    let (status, first_matrix) = worker.join().expect("hessian thread");
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let old_call_finished = retirer.join().expect("retiring thread");
    (pot, first_matrix, old_call_finished)
}

#[test]
fn replacing_the_callback_waits_for_the_call_in_flight() {
    let (pot, first_matrix, old_call_finished) = retire_while_in_flight(Some(second_hessian));
    assert!(
        old_call_finished,
        "replacement returned before the old call ended"
    );
    assert_eq!(first_matrix, [1.0; 9]);
    let mut geom = Geom::new([0.1, 0.2, -0.3]);
    let mut input = geom.input();
    let mut matrix = [0.0; 9];
    let status = unsafe { rgpot_potential_hessian(pot, &input, matrix.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(matrix, [2.0; 9], "the old matrix must not be published");
    unsafe {
        rgpot_force_input_free(&mut input);
        rgpot_potential_free(pot);
    }
}

#[test]
fn clearing_the_callback_waits_for_the_call_in_flight() {
    let (pot, first_matrix, old_call_finished) = retire_while_in_flight(None);
    assert!(
        old_call_finished,
        "clearing returned before the old call ended"
    );
    assert_eq!(first_matrix, [1.0; 9]);
    let mut geom = Geom::new([0.1, 0.2, -0.3]);
    let mut input = geom.input();
    let mut matrix = [0.0; 9];
    let status = unsafe { rgpot_potential_hessian(pot, &input, matrix.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_NO_HESSIAN);
    unsafe {
        rgpot_force_input_free(&mut input);
        rgpot_potential_free(pot);
    }
}

const COUPLED_DIM: usize = 6;

/// Symmetric, non-diagonal, with terms between the two atoms.
const COUPLED_K: [f64; 36] = [
    2.0, 0.5, -0.2, 0.3, 0.0, 0.1, //
    0.5, 3.0, 0.4, 0.0, -0.6, 0.2, //
    -0.2, 0.4, 2.5, 0.2, 0.1, -0.3, //
    0.3, 0.0, 0.2, 1.8, 0.7, 0.0, //
    0.0, -0.6, 0.1, 0.7, 2.2, -0.4, //
    0.1, 0.2, -0.3, 0.0, -0.4, 3.1,
];

/// `E = x.K.x / 2 + 0.3 (x0 x3)^2 + 0.2 x1 x4 x5`, a coupled potential whose
/// Hessian depends on the geometry.
fn coupled_gradient(x: &[f64]) -> Vec<f64> {
    let mut gradient = matvec(&COUPLED_K, x);
    gradient[0] += 0.6 * x[0] * x[3] * x[3];
    gradient[3] += 0.6 * x[0] * x[0] * x[3];
    gradient[1] += 0.2 * x[4] * x[5];
    gradient[4] += 0.2 * x[1] * x[5];
    gradient[5] += 0.2 * x[1] * x[4];
    gradient
}

fn coupled_hessian(x: &[f64]) -> Vec<f64> {
    let mut h = COUPLED_K.to_vec();
    let mut add = |i: usize, j: usize, value: f64| {
        h[i * COUPLED_DIM + j] += value;
        if i != j {
            h[j * COUPLED_DIM + i] += value;
        }
    };
    add(0, 0, 0.6 * x[3] * x[3]);
    add(0, 3, 1.2 * x[0] * x[3]);
    add(3, 3, 0.6 * x[0] * x[0]);
    add(1, 4, 0.2 * x[5]);
    add(1, 5, 0.2 * x[4]);
    add(4, 5, 0.2 * x[1]);
    h
}

unsafe extern "C" fn coupled_force(
    _user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let x = read_positions(unsafe { &*input });
    let gradient = coupled_gradient(&x);
    let forces: Vec<f64> = gradient.iter().map(|g| -g).collect();
    let mut energy = 0.5
        * matvec(&COUPLED_K, &x)
            .iter()
            .zip(&x)
            .map(|(a, b)| a * b)
            .sum::<f64>();
    energy += 0.3 * (x[0] * x[3]).powi(2) + 0.2 * x[1] * x[4] * x[5];
    let out = unsafe { &mut *output };
    out.energy = energy;
    out.variance = 0.0;
    out.forces = unsafe { rgpot_tensor_owned_cpu_f64_2d(forces.as_ptr(), 2, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn coupled_hessian_callback(
    _user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    hessian_out: *mut f64,
) -> rgpot_status_t {
    let x = read_positions(unsafe { &*input });
    let h = coupled_hessian(&x);
    unsafe { std::ptr::copy_nonoverlapping(h.as_ptr(), hessian_out, h.len()) };
    rgpot_status_t::RGPOT_SUCCESS
}

fn coupled_forces(pot: *const rgpot_potential_t, x: &[f64; COUPLED_DIM]) -> Vec<f64> {
    let mut x = *x;
    let mut z = [1, 1];
    let mut cell = [0.0; 9];
    cell[0] = 10.0;
    cell[4] = 10.0;
    cell[8] = 10.0;
    let mut input =
        unsafe { rgpot_force_input_create(2, x.as_mut_ptr(), z.as_mut_ptr(), cell.as_mut_ptr()) };
    let mut out = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    let status = unsafe { rgpot_potential_calculate(pot, &input, &mut out) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let forces = unsafe {
        std::slice::from_raw_parts((*out.forces).dl_tensor.data as *const f64, COUPLED_DIM)
    }
    .to_vec();
    unsafe {
        rgpot_tensor_free(out.forces);
        rgpot_force_input_free(&mut input);
    }
    forces
}

#[test]
fn the_dense_hessian_matches_a_finite_difference_of_forces_in_every_direction() {
    let pot = unsafe {
        rgpot_potential_new(
            coupled_force as PotentialCallback,
            std::ptr::null_mut(),
            None,
        )
    };
    assert!(!pot.is_null());
    let status =
        unsafe { rgpot_potential_set_hessian_callback(pot, Some(coupled_hessian_callback)) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);

    let centre = [0.4, -0.7, 0.9, 1.1, 0.6, -0.5];
    let mut positions = centre;
    let mut z = [1, 1];
    let mut cell = [0.0; 9];
    cell[0] = 10.0;
    cell[4] = 10.0;
    cell[8] = 10.0;
    let mut input = unsafe {
        rgpot_force_input_create(2, positions.as_mut_ptr(), z.as_mut_ptr(), cell.as_mut_ptr())
    };
    let mut dense = [0.0; COUPLED_DIM * COUPLED_DIM];
    let status = unsafe { rgpot_potential_hessian(pot, &input, dense.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
    unsafe { rgpot_force_input_free(&mut input) };

    let step = 1.0e-5;
    let mut largest = 0.0f64;
    let mut off_diagonal = 0.0f64;
    for j in 0..COUPLED_DIM {
        let mut plus = centre;
        let mut minus = centre;
        plus[j] += step;
        minus[j] -= step;
        let f_plus = coupled_forces(pot, &plus);
        let f_minus = coupled_forces(pot, &minus);
        for i in 0..COUPLED_DIM {
            let numeric = -(f_plus[i] - f_minus[i]) / (2.0 * step);
            let analytic = dense[i * COUPLED_DIM + j];
            largest = largest.max((numeric - analytic).abs());
            if i != j {
                off_diagonal = off_diagonal.max(analytic.abs());
            }
        }
    }
    assert!(off_diagonal > 0.5, "the test potential must be coupled");
    assert!(largest < 1.0e-7, "max |H - FD(-F)| = {largest:e}");
    unsafe { rgpot_potential_free(pot) };
}
