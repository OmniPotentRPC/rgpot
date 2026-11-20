// MIT License
// Copyright 2023--present rgpot developers

//! Ownership and failure-class tests for [`EngineSession`] and its callbacks.
//!
//! The engine is a set of in-process Rust functions standing in for a shared
//! library, so nothing here needs `dlopen` and the whole file runs under Miri.

use super::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type OracleFn = unsafe extern "C" fn(
    *mut c_void,
    c_long,
    *const f64,
    *const c_int,
    *mut f64,
    *mut f64,
    *const f64,
) -> c_int;

thread_local! {
    static CTX: Cell<*mut c_void> = const { Cell::new(std::ptr::null_mut()) };
    static ORACLE: Cell<Option<OracleFn>> = const { Cell::new(None) };
    static FORCE_CODE: Cell<Option<c_int>> = const { Cell::new(None) };
    static DESTROY_RUNS_ORACLE: Cell<bool> = const { Cell::new(false) };
}

const POSITIONS: [f64; 3] = [0.0, 0.0, 0.0];
const NUMBERS: [c_int; 1] = [1];
const BOX: [f64; 9] = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

unsafe fn run_stored_oracle() -> Option<c_int> {
    let oracle = ORACLE.with(Cell::get)?;
    let mut forces = [0.0; 3];
    let mut energy = 0.0;
    Some(unsafe {
        oracle(
            CTX.with(Cell::get),
            1,
            POSITIONS.as_ptr(),
            NUMBERS.as_ptr(),
            forces.as_mut_ptr(),
            &mut energy,
            BOX.as_ptr(),
        )
    })
}

unsafe extern "C" fn fake_abi_version() -> c_int {
    ENGINE_ABI_VERSION
}

unsafe extern "C" fn fake_abi_stamp() -> RgpotEngineAbiStamp {
    RgpotEngineAbiStamp::HOST
}

unsafe extern "C" fn fake_create(
    _config: *const c_void,
    _len: usize,
    _error: *mut c_char,
    _capacity: usize,
) -> *mut RgpotEnginePot {
    NonNull::<RgpotEnginePot>::dangling().as_ptr()
}

unsafe extern "C" fn fake_destroy(_pot: *mut RgpotEnginePot) {
    if DESTROY_RUNS_ORACLE.with(Cell::get) {
        unsafe { run_stored_oracle() };
    }
}

unsafe extern "C" fn fake_force(
    _pot: *mut RgpotEnginePot,
    _n_atoms: c_long,
    _positions: *const f64,
    _atomic_numbers: *const c_int,
    _forces: *mut f64,
    _energy: *mut f64,
    _variance: *mut f64,
    _box_matrix: *const f64,
    _transform: Option<CoordTransformFn>,
    _user: *mut c_void,
) -> c_int {
    let oracle_code = unsafe { run_stored_oracle() }.unwrap_or(0);
    FORCE_CODE.with(Cell::get).unwrap_or(oracle_code)
}

unsafe extern "C" fn fake_set_callbacks(
    _pot: *mut RgpotEnginePot,
    table: *const RgpotEngineCallbacks,
) -> c_int {
    match unsafe { table.as_ref() } {
        Some(table) => {
            CTX.with(|ctx| ctx.set(table.ctx));
            ORACLE.with(|oracle| oracle.set(table.oracle));
        }
        None => {
            CTX.with(|ctx| ctx.set(std::ptr::null_mut()));
            ORACLE.with(|oracle| oracle.set(None));
        }
    }
    0
}

fn symbols() -> EngineSymbols {
    EngineSymbols {
        abi_version: fake_abi_version,
        abi_stamp: Some(fake_abi_stamp),
        create: fake_create,
        destroy: fake_destroy,
        force: fake_force,
        force_batch: None,
        hvp: None,
        caps: None,
        last_error: None,
        model_state: None,
        set_callbacks: Some(fake_set_callbacks),
        hint: None,
    }
}

fn engine() -> EnginePotential {
    CTX.with(|ctx| ctx.set(std::ptr::null_mut()));
    ORACLE.with(|oracle| oracle.set(None));
    FORCE_CODE.with(|code| code.set(None));
    DESTROY_RUNS_ORACLE.with(|flag| flag.set(false));
    unsafe { EnginePotential::from_symbols(symbols(), &[]) }.expect("the fake engine loads")
}

fn session() -> EngineSession {
    EngineSession {
        engine: engine(),
        callbacks: None,
    }
}

fn request() -> EngineRequest<'static> {
    EngineRequest {
        positions: &POSITIONS,
        atomic_numbers: &NUMBERS,
        box_matrix: &BOX,
    }
}

fn recoverable(code: c_int, message: &str) -> EngineFailure {
    EngineFailure {
        status: EngineStatus::recoverable(code).expect("a positive code"),
        message: message.into(),
    }
}

fn fatal(code: c_int, message: &str) -> EngineFailure {
    EngineFailure {
        status: EngineStatus::fatal(code).expect("a negative code"),
        message: message.into(),
    }
}

/// Host callbacks whose oracle answers from a script and counts its calls.
struct Scripted {
    calls: Arc<AtomicUsize>,
    answers: Vec<Result<EngineEvaluation, EngineFailure>>,
}

impl Scripted {
    fn new(answers: Vec<Result<EngineEvaluation, EngineFailure>>) -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                calls: Arc::clone(&calls),
                answers,
            },
            calls,
        )
    }
}

impl EngineCallbacks for Scripted {
    fn oracle(&mut self, _request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let last = self.answers.len() - 1;
        match &self.answers[call.min(last)] {
            Ok(answer) => Ok(answer.clone()),
            Err(failure) => Err(failure.clone()),
        }
    }

    fn on_acquire(&mut self, _reason: AcquireReason) {
        self.calls.fetch_add(1, Ordering::SeqCst);
    }

    fn on_model_update(&mut self, _model_state: u64) {
        self.calls.fetch_add(1, Ordering::SeqCst);
    }
}

/// Panics on its first call and would succeed on any later one.
struct PanicsOnce {
    calls: Arc<AtomicUsize>,
}

impl EngineCallbacks for PanicsOnce {
    fn oracle(&mut self, _request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            panic!("the first oracle call panics");
        }
        Ok(EngineEvaluation {
            energy: 1.0,
            forces: vec![0.0; 3],
        })
    }

    fn on_acquire(&mut self, _reason: AcquireReason) {
        self.calls.fetch_add(1, Ordering::SeqCst);
    }

    fn on_model_update(&mut self, _model_state: u64) {
        self.calls.fetch_add(1, Ordering::SeqCst);
    }
}

fn evaluation(energy: f64) -> EngineEvaluation {
    EngineEvaluation {
        energy,
        forces: vec![0.5; 3],
    }
}

fn state_of(callbacks: impl EngineCallbacks + 'static) -> CallbackState {
    CallbackState {
        callbacks: Mutex::new(Box::new(callbacks)),
        failure: Mutex::new(None),
    }
}

unsafe fn call_oracle(state: &CallbackState) -> c_int {
    let mut forces = [0.0; 3];
    let mut energy = 0.0;
    unsafe {
        oracle_trampoline(
            (state as *const CallbackState).cast_mut().cast(),
            1,
            POSITIONS.as_ptr(),
            NUMBERS.as_ptr(),
            forces.as_mut_ptr(),
            &mut energy,
            BOX.as_ptr(),
        )
    }
}

#[test]
fn registered_context_stays_valid_after_the_state_moves_into_the_session() {
    let mut session = session();
    let (first, first_calls) = Scripted::new(vec![Ok(evaluation(2.0))]);
    session
        .set_callbacks(Some(Box::new(first)))
        .expect("register");
    // The native side now holds the context pointer; reading it must be sound
    // after `set_callbacks` has moved the owning state into the session.
    assert_eq!(unsafe { run_stored_oracle() }, Some(0));
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);

    let (second, second_calls) = Scripted::new(vec![Ok(evaluation(3.0))]);
    session
        .set_callbacks(Some(Box::new(second)))
        .expect("replace");
    assert_eq!(unsafe { run_stored_oracle() }, Some(0));
    assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);

    session.set_callbacks(None).expect("clear");
    assert_eq!(unsafe { run_stored_oracle() }, None);
}

#[test]
fn engine_destroy_runs_before_the_callback_state_is_freed() {
    let mut session = session();
    let (callbacks, calls) = Scripted::new(vec![Ok(evaluation(2.0))]);
    session
        .set_callbacks(Some(Box::new(callbacks)))
        .expect("register");
    DESTROY_RUNS_ORACLE.with(|flag| flag.set(true));
    drop(session);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn callback_ownership_is_send() {
    fn is_send<T: Send>() {}
    is_send::<EngineSession>();
}

#[test]
fn a_fatal_callback_failure_is_never_replaced() {
    let (callbacks, _) = Scripted::new(vec![Ok(evaluation(0.0))]);
    let state = state_of(callbacks);
    state.record(fatal(-9, "first fatal"));
    state.record(recoverable(5, "later recoverable"));
    state.record(fatal(-3, "later fatal"));
    let held = state.take_failure().expect("a failure is held");
    assert_eq!(held.status.code(), -9);
    assert_eq!(held.message, "first fatal");
    assert!(state.take_failure().is_none());
}

#[test]
fn a_recoverable_callback_failure_gives_way_to_a_later_one() {
    let (callbacks, _) = Scripted::new(vec![Ok(evaluation(0.0))]);
    let state = state_of(callbacks);
    state.record(recoverable(5, "first"));
    state.record(recoverable(6, "second"));
    assert_eq!(state.take_failure().expect("held").status.code(), 6);
    state.record(recoverable(5, "recoverable"));
    state.record(fatal(-9, "fatal"));
    assert_eq!(state.take_failure().expect("held").status.code(), -9);
}

#[test]
fn a_poisoned_callback_lock_is_fatal_and_runs_nothing() {
    let calls = Arc::new(AtomicUsize::new(0));
    let state = state_of(PanicsOnce {
        calls: Arc::clone(&calls),
    });
    assert_eq!(unsafe { call_oracle(&state) }, -1, "the panic is fatal");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let first = state.take_failure().expect("the panic is recorded");
    assert_eq!(first.status.code(), -1);

    assert!(
        unsafe { call_oracle(&state) } < 0,
        "a poisoned oracle must not be reported as recovered"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1, "the oracle was re-entered");
    let second = state.take_failure().expect("the poisoning is recorded");
    assert_eq!(second.status.code(), -1);

    let ctx = (&state as *const CallbackState).cast_mut().cast();
    unsafe {
        acquire_trampoline(ctx, 1);
        model_update_trampoline(ctx, 7);
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a hook ran on poisoned state"
    );
    assert_eq!(state.take_failure().expect("recorded").status.code(), -1);
}

#[test]
fn a_negative_engine_code_beats_a_recoverable_callback_failure_in_the_session() {
    let mut session = session();
    let (callbacks, _) = Scripted::new(vec![Err(recoverable(5, "bad input"))]);
    session
        .set_callbacks(Some(Box::new(callbacks)))
        .expect("register");
    FORCE_CODE.with(|code| code.set(Some(-3)));
    let error = session.force(&request(), None).expect_err("the call fails");
    assert!(
        matches!(error, EngineError::Fatal { .. }),
        "an engine code below zero is fatal, got {error:?}"
    );
    assert!(matches!(
        session.force(&request(), None),
        Err(EngineError::Fatal { .. })
    ));
}

#[test]
fn a_negative_engine_code_beats_a_recoverable_failure_in_answer_with_failure() {
    let engine = engine();
    let error = engine
        .answer_with_failure(-4, Some(recoverable(5, "bad input")))
        .expect_err("the call fails");
    assert!(
        matches!(error, EngineError::Fatal { code: -4, .. }),
        "an engine code below zero is fatal, got {error:?}"
    );
    assert!(matches!(
        engine.check_alive(),
        Err(EngineError::Fatal { code: -4, .. })
    ));
}

#[test]
fn a_negative_engine_code_beats_a_success_status_failure() {
    let engine = engine();
    let success = EngineFailure {
        status: EngineStatus::Success,
        message: "success".into(),
    };
    let error = engine
        .answer_with_failure(-4, Some(success))
        .expect_err("the call fails");
    assert!(matches!(error, EngineError::Fatal { code: -4, .. }));
}

#[test]
fn a_second_fatal_failure_reports_the_stored_first_one() {
    let engine = engine();
    let first = engine
        .fail(EngineStatus::from_code(-3), "first".into())
        .expect_err("fatal");
    assert_eq!(
        first,
        EngineError::Fatal {
            code: -3,
            message: "first".into()
        }
    );
    let second = engine
        .fail(EngineStatus::from_code(-9), "second".into())
        .expect_err("fatal");
    assert_eq!(second, first, "the stored failure is what is reported");
    let third = engine.answer(-7).expect_err("fatal");
    assert!(matches!(third, EngineError::Fatal { code: -3, .. }));
}
