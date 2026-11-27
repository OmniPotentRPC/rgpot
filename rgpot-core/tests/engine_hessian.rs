// MIT License
// Copyright 2023--present rgpot developers

//! How `EnginePotential::hessian`, and the handle of
//! `EnginePotential::into_potential`, pick the source of a dense Hessian,
//! against fake engines made of Rust functions with
//! `EnginePotential::from_symbols`.
//!
//! The fake surface is `E = sum_k Q x_k^4 / 4 + C sum_k x_k x_{k+1}^2` over
//! the flattened coordinates. Its dense Hessian and its Hessian-vector
//! product are separate code, and each hook counts its calls on the calling
//! thread. The create message is two bytes: whether `rgpot_engine_caps`
//! advertises `analytic_hessian`, and whether it advertises `analytic_hvp`.

use std::cell::Cell;
use std::ffi::{c_char, c_int, c_long, c_uint, c_void};

use rgpot_core::engine::{
    CoordTransformFn, EngineError, EnginePotential, EngineRequest, EngineSymbols,
    RgpotEngineAbiStamp, RgpotEngineCaps, RgpotEnginePot, ENGINE_ABI_VERSION,
};
use rgpot_core::hessian::rgpot_hessian_scheme_t;
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_free,
};
use rgpot_core::types::rgpot_force_input_t;

const Q: f64 = 0.7;
const C: f64 = 0.3;
const CELL: [f64; 9] = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
const POSITIONS: [f64; 6] = [0.4, -0.7, 1.1, -0.3, 0.9, 0.25];
const NUMBERS: [i32; 2] = [1, 8];
const DIM: usize = POSITIONS.len();

struct FakePot {
    dense_cap: bool,
    hvp_cap: bool,
}

thread_local! {
    static DENSE_CALLS: Cell<usize> = const { Cell::new(0) };
    static HVP_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Calls of the dense hook and of the product since the last take.
fn take_calls() -> (usize, usize) {
    (DENSE_CALLS.replace(0), HVP_CALLS.replace(0))
}

fn gradient(x: &[f64]) -> Vec<f64> {
    let mut g: Vec<f64> = x.iter().map(|v| Q * v * v * v).collect();
    for k in 0..x.len() - 1 {
        g[k] += C * x[k + 1] * x[k + 1];
        g[k + 1] += 2.0 * C * x[k] * x[k + 1];
    }
    g
}

fn dense(x: &[f64]) -> Vec<f64> {
    let d = x.len();
    let mut h = vec![0.0; d * d];
    for k in 0..d {
        h[k * d + k] = 3.0 * Q * x[k] * x[k];
    }
    for k in 0..d - 1 {
        h[(k + 1) * d + k + 1] += 2.0 * C * x[k];
        h[k * d + k + 1] = 2.0 * C * x[k + 1];
        h[(k + 1) * d + k] = 2.0 * C * x[k + 1];
    }
    h
}

fn product(x: &[f64], v: &[f64]) -> Vec<f64> {
    let d = x.len();
    (0..d)
        .map(|k| {
            let mut hv = 3.0 * Q * x[k] * x[k] * v[k];
            if k > 0 {
                hv += 2.0 * C * (x[k - 1] * v[k] + x[k] * v[k - 1]);
            }
            if k + 1 < d {
                hv += 2.0 * C * x[k + 1] * v[k + 1];
            }
            hv
        })
        .collect()
}

unsafe fn coordinates<'a>(n: c_long, positions: *const f64) -> &'a [f64] {
    unsafe { std::slice::from_raw_parts(positions, 3 * n as usize) }
}

unsafe extern "C" fn fake_abi_version() -> c_int {
    ENGINE_ABI_VERSION
}

unsafe extern "C" fn fake_abi_stamp() -> RgpotEngineAbiStamp {
    RgpotEngineAbiStamp::HOST
}

unsafe extern "C" fn fake_create(
    config: *const c_void,
    config_len: usize,
    _errbuf: *mut c_char,
    _errlen: usize,
) -> *mut RgpotEnginePot {
    if config.is_null() || config_len != 2 {
        return std::ptr::null_mut();
    }
    let flags = unsafe { std::slice::from_raw_parts(config.cast::<u8>(), 2) };
    Box::into_raw(Box::new(FakePot {
        dense_cap: flags[0] != 0,
        hvp_cap: flags[1] != 0,
    }))
    .cast()
}

unsafe extern "C" fn fake_destroy(pot: *mut RgpotEnginePot) {
    if !pot.is_null() {
        drop(unsafe { Box::from_raw(pot.cast::<FakePot>()) });
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn fake_force(
    _pot: *mut RgpotEnginePot,
    n: c_long,
    positions: *const f64,
    _atomic_numbers: *const c_int,
    forces: *mut f64,
    energy: *mut f64,
    _variance: *mut f64,
    _box: *const f64,
    _transform: Option<CoordTransformFn>,
    _user: *mut c_void,
) -> c_int {
    let x = unsafe { coordinates(n, positions) };
    let quartic: f64 = x.iter().map(|v| 0.25 * Q * v.powi(4)).sum();
    let chain: f64 = x.windows(2).map(|w| C * w[0] * w[1] * w[1]).sum();
    let out = unsafe { std::slice::from_raw_parts_mut(forces, x.len()) };
    for (f, g) in out.iter_mut().zip(gradient(x)) {
        *f = -g;
    }
    unsafe { *energy = quartic + chain };
    0
}

unsafe extern "C" fn fake_hvp(
    _pot: *mut RgpotEnginePot,
    n: c_long,
    positions: *const f64,
    _atomic_numbers: *const c_int,
    direction: *const f64,
    hv_out: *mut f64,
    _box: *const f64,
) -> c_int {
    HVP_CALLS.set(HVP_CALLS.get() + 1);
    let x = unsafe { coordinates(n, positions) };
    let v = unsafe { std::slice::from_raw_parts(direction, x.len()) };
    let out = unsafe { std::slice::from_raw_parts_mut(hv_out, x.len()) };
    out.copy_from_slice(&product(x, v));
    0
}

unsafe extern "C" fn fake_hessian(
    _pot: *mut RgpotEnginePot,
    n: c_long,
    positions: *const f64,
    _atomic_numbers: *const c_int,
    hessian_out: *mut f64,
    _box: *const f64,
) -> c_int {
    DENSE_CALLS.set(DENSE_CALLS.get() + 1);
    let x = unsafe { coordinates(n, positions) };
    let out = unsafe { std::slice::from_raw_parts_mut(hessian_out, x.len() * x.len()) };
    out.copy_from_slice(&dense(x));
    0
}

unsafe extern "C" fn fake_caps(pot: *const RgpotEnginePot, out: *mut RgpotEngineCaps) -> c_int {
    let pot = unsafe { &*pot.cast::<FakePot>() };
    unsafe {
        *out = RgpotEngineCaps {
            size: std::mem::size_of::<RgpotEngineCaps>() as c_uint,
            periodic: 1,
            analytic_hvp: c_int::from(pot.hvp_cap),
            analytic_hessian: c_int::from(pot.dense_cap),
            ..Default::default()
        };
    }
    0
}

fn base_symbols() -> EngineSymbols {
    EngineSymbols::new(fake_abi_version, fake_create, fake_destroy, fake_force)
        .with_abi_stamp(fake_abi_stamp)
}

/// The fake engine exporting the dense hook and the product as asked, with
/// `rgpot_engine_caps`.
fn symbols(dense_hook: bool, hvp_hook: bool) -> EngineSymbols {
    let mut symbols = base_symbols().with_caps(fake_caps);
    if dense_hook {
        symbols = symbols.with_hessian(fake_hessian);
    }
    if hvp_hook {
        symbols = symbols.with_hvp(fake_hvp);
    }
    symbols
}

fn engine(symbols: EngineSymbols, dense_cap: bool, hvp_cap: bool) -> EnginePotential {
    unsafe { EnginePotential::from_symbols(symbols, &[u8::from(dense_cap), u8::from(hvp_cap)]) }
        .expect("the fake engine loads")
}

fn request() -> EngineRequest<'static> {
    EngineRequest {
        positions: &POSITIONS,
        atomic_numbers: &NUMBERS,
        box_matrix: &CELL,
    }
}

fn scale(matrix: &[f64]) -> f64 {
    matrix.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Each entry within `ulps` machine epsilons of the largest entry of `want`.
fn assert_close(got: &[f64], want: &[f64], ulps: f64) {
    assert_eq!(got.len(), want.len());
    let tolerance = ulps * f64::EPSILON * scale(want);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance,
            "entry {i}: {g} against {w}, tolerance {tolerance}"
        );
    }
}

fn assert_symmetric_and_finite(matrix: &[f64]) {
    assert_eq!(matrix.len(), DIM * DIM);
    if let Some(i) = matrix.iter().position(|v| !v.is_finite()) {
        panic!("entry {i} is not finite");
    }
    let tolerance = 4.0 * f64::EPSILON * scale(matrix);
    for r in 0..DIM {
        for c in (r + 1)..DIM {
            let (upper, lower) = (matrix[r * DIM + c], matrix[c * DIM + r]);
            assert!(
                (upper - lower).abs() <= tolerance,
                "entries ({r}, {c}) and ({c}, {r}) differ: {upper} against {lower}"
            );
        }
    }
}

/// The Hessian by central differences of the engine's own forces, which
/// checks the fake's closed forms independently of either hook.
fn central_differences(engine: &mut EnginePotential) -> Vec<f64> {
    let h = 1e-4;
    let mut matrix = vec![0.0; DIM * DIM];
    for c in 0..DIM {
        let mut forces_at = |shift: f64| {
            let mut x = POSITIONS;
            x[c] += shift;
            engine
                .force(&EngineRequest {
                    positions: &x,
                    atomic_numbers: &NUMBERS,
                    box_matrix: &CELL,
                })
                .expect("force")
                .forces
        };
        let (plus, minus) = (forces_at(h), forces_at(-h));
        for r in 0..DIM {
            matrix[r * DIM + c] = (minus[r] - plus[r]) / (2.0 * h);
        }
    }
    matrix
}

fn assert_matches_central_differences(matrix: &[f64], engine: &mut EnginePotential) {
    let reference = central_differences(engine);
    for (i, (got, want)) in matrix.iter().zip(&reference).enumerate() {
        assert!(
            (got - want).abs() <= 1e-6,
            "entry {i}: {got} against the central difference {want}"
        );
    }
}

#[test]
fn the_dense_hook_serves_the_hessian_when_it_is_advertised() {
    take_calls();
    let mut engine = engine(symbols(true, false), true, false);
    let matrix = engine.hessian(&request()).expect("dense Hessian");
    assert_eq!(take_calls(), (1, 0));
    assert_close(&matrix, &dense(&POSITIONS), 0.0);
    assert_symmetric_and_finite(&matrix);
    assert_matches_central_differences(&matrix, &mut engine);
}

#[test]
fn the_columns_come_from_the_product_when_only_analytic_hvp_is_advertised() {
    take_calls();
    let mut hvp_only = engine(symbols(false, true), false, true);
    let matrix = hvp_only.hessian(&request()).expect("Hessian from columns");
    assert_eq!(take_calls(), (0, DIM));
    assert_close(&matrix, &dense(&POSITIONS), 4.0);
    assert_symmetric_and_finite(&matrix);
    assert_matches_central_differences(&matrix, &mut hvp_only);

    // A dense hook the caps do not advertise is not called, as in the C table.
    let mut unadvertised = engine(symbols(true, true), false, true);
    let columns = unadvertised
        .hessian(&request())
        .expect("Hessian from columns");
    assert_eq!(take_calls(), (0, DIM));
    assert_eq!(columns, matrix);
}

#[test]
fn the_assembled_hessian_matches_the_dense_one_on_an_engine_with_both() {
    take_calls();
    let both = symbols(true, true);
    let mut dense_route = engine(both, true, true);
    assert!(dense_route.caps().expect("caps").analytic_hessian);
    assert!(dense_route.caps().expect("caps").analytic_hvp);
    let from_dense = dense_route.hessian(&request()).expect("dense Hessian");
    assert_eq!(take_calls(), (1, 0));

    let mut column_route = engine(both, false, true);
    let from_columns = column_route
        .hessian(&request())
        .expect("Hessian from columns");
    assert_eq!(take_calls(), (0, DIM));

    assert_symmetric_and_finite(&from_dense);
    assert_symmetric_and_finite(&from_columns);
    assert_close(&from_columns, &from_dense, 4.0);
    assert_matches_central_differences(&from_columns, &mut column_route);
}

#[test]
fn an_engine_with_neither_source_is_unsupported_without_a_call() {
    take_calls();
    let cases = [
        (
            symbols(false, false),
            true,
            true,
            "no dense Hessian (rgpot_engine_hessian is not exported) and no \
             Hessian-vector product (rgpot_engine_hvp is not exported)",
        ),
        (
            symbols(true, true),
            false,
            false,
            "no dense Hessian (analytic_hessian is not advertised) and no \
             Hessian-vector product (analytic_hvp is not advertised)",
        ),
        (
            symbols(true, false),
            false,
            true,
            "no dense Hessian (analytic_hessian is not advertised) and no \
             Hessian-vector product (rgpot_engine_hvp is not exported)",
        ),
        (
            base_symbols().with_hessian(fake_hessian).with_hvp(fake_hvp),
            true,
            true,
            "no dense Hessian (rgpot_engine_caps is not exported) and no \
             Hessian-vector product (rgpot_engine_caps is not exported)",
        ),
    ];
    for (symbols, dense_cap, hvp_cap, message) in cases {
        let mut engine = engine(symbols, dense_cap, hvp_cap);
        let err = engine.hessian(&request()).unwrap_err();
        assert_eq!(
            err,
            EngineError::Unsupported {
                operation: "rgpot_engine_hessian",
                message: message.into(),
            }
        );
        assert_eq!(err.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
        assert_eq!(take_calls(), (0, 0));
        let potential = engine.into_potential();
        assert!(!potential.as_handle().derivative_caps().hessian_analytic);
    }
}

#[test]
fn the_potential_handle_takes_the_same_route() {
    take_calls();
    let routes = [
        (symbols(true, false), true, false, (1, 0)),
        (symbols(false, true), false, true, (0, DIM)),
        (symbols(true, true), true, true, (1, 0)),
        (symbols(true, true), false, true, (0, DIM)),
        (symbols(true, true), false, false, (0, 0)),
    ];
    let mut positions = POSITIONS;
    let mut numbers = NUMBERS;
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 2, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 2) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    for (symbols, dense_cap, hvp_cap, calls) in routes {
        let potential = engine(symbols, dense_cap, hvp_cap).into_potential();
        let handle = potential.as_handle();
        assert_eq!(handle.has_hessian_vector_callback(), hvp_cap);
        let mut matrix = vec![0.0; DIM * DIM];
        let answer = unsafe {
            handle.hessian(
                &input,
                0.0,
                rgpot_hessian_scheme_t::RGPOT_HESSIAN_EXACT,
                &mut matrix,
            )
        };
        if calls == (0, 0) {
            assert!(!handle.derivative_caps().hessian_analytic);
            assert!(answer.is_err());
        } else {
            assert!(handle.derivative_caps().hessian_analytic);
            assert_eq!(
                answer.expect("analytic Hessian through the handle"),
                rgpot_hessian_scheme_t::RGPOT_HESSIAN_ANALYTIC
            );
            assert_close(&matrix, &dense(&POSITIONS), 4.0);
            assert_eq!(handle.derivative_counts().hessian_analytic, 1);
            assert_eq!(handle.derivative_counts().hessian_force_calls, 0);
        }
        assert_eq!(take_calls(), calls);
    }
    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
}
