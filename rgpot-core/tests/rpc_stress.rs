// MIT License
// Copyright 2023--present rgpot developers

//! Stress over the RPC pair, end to end on a loopback socket.
//!
//! A force-only call never asks the server for stress and never runs the
//! stress callback. A stress call reaches the stress callback and returns
//! its block. A server whose potential has no stress callback refuses the
//! stress call with a message and leaves the block at zero.
//!
//! A peer that speaks protocol minor 1 predates `ForceInput.computeStress`.
//! It still serves force-only calls, where the stress is not requested, and
//! a stress call to it is refused naming the protocol minor.

use std::ffi::{c_void, CStr, CString};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rgpot_core::c_api::potential::{
    rgpot_potential_new, rgpot_potential_set_stress_callback, rgpot_stress_out_create,
};
use rgpot_core::c_api::rpc::{
    rgpot_rpc_calculate, rgpot_rpc_calculate_stress, rgpot_rpc_client_free, rgpot_rpc_client_new,
};
use rgpot_core::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
use rgpot_core::eindir::rgpot_potential_t;
use rgpot_core::rpc::server::rgpot_rpc_server_start;
use rgpot_core::status::{rgpot_last_error, rgpot_status_t};
use rgpot_core::tensor::{rgpot_tensor_data, rgpot_tensor_free, rgpot_tensor_owned_cpu_f64_2d};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t, rgpot_stress_out_t};

struct Counts {
    force: AtomicUsize,
    stress: AtomicUsize,
}

fn counts(user_data: *mut c_void) -> &'static Counts {
    unsafe { &*(user_data as *const Counts) }
}

const STRESS: [f64; 9] = [0.5, 0.25, 0.125, 0.25, 1.5, 0.0625, 0.125, 0.0625, 2.5];

fn forces_of(input: *const rgpot_force_input_t, output: &mut rgpot_force_out_t) {
    let n = unsafe { (*input).n_atoms() }.expect("positions");
    output.energy = -3.5;
    let values = vec![0.25; 3 * n];
    output.forces = unsafe { rgpot_tensor_owned_cpu_f64_2d(values.as_ptr(), n as i64, 3) };
}

unsafe extern "C" fn force_only(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    counts(user_data).force.fetch_add(1, Ordering::SeqCst);
    forces_of(input, unsafe { &mut *output });
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn with_stress(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
    stress: *mut rgpot_stress_out_t,
) -> rgpot_status_t {
    counts(user_data).stress.fetch_add(1, Ordering::SeqCst);
    forces_of(input, unsafe { &mut *output });
    let block = unsafe { &mut *stress };
    block.stress = STRESS;
    block.has_stress = 1;
    rgpot_status_t::RGPOT_SUCCESS
}

struct SendPtr(*const rgpot_potential_t);
unsafe impl Send for SendPtr {}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("addr")
        .port()
}

/// Start a server on its own thread and wait until it accepts connections.
/// The thread blocks for the life of the process, as the C function does.
fn serve(stress: bool) -> (u16, &'static Counts) {
    let port = free_port();
    let seen: &'static Counts = Box::leak(Box::new(Counts {
        force: AtomicUsize::new(0),
        stress: AtomicUsize::new(0),
    }));
    let pot =
        unsafe { rgpot_potential_new(force_only, seen as *const Counts as *mut c_void, None) };
    assert!(!pot.is_null());
    if stress {
        let status = unsafe { rgpot_potential_set_stress_callback(pot, Some(with_stress)) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    }
    let handle = SendPtr(pot);
    std::thread::spawn(move || {
        let handle = handle;
        let host = CString::new("127.0.0.1").unwrap();
        unsafe { rgpot_rpc_server_start(handle.0, host.as_ptr(), port) };
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        std::thread::sleep(Duration::from_millis(20));
    }
    (port, seen)
}

struct Case {
    _positions: Vec<f64>,
    _numbers: Vec<i32>,
    _cell: Box<[f64; 9]>,
    input: rgpot_force_input_t,
}

impl Case {
    fn new() -> Self {
        let mut positions = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.5];
        let mut numbers = vec![29, 29, 29];
        let mut cell = Box::new([9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0]);
        let input = unsafe {
            rgpot_force_input_create(
                numbers.len(),
                positions.as_mut_ptr(),
                numbers.as_mut_ptr(),
                cell.as_mut_ptr(),
            )
        };
        Self {
            _positions: positions,
            _numbers: numbers,
            _cell: cell,
            input,
        }
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        unsafe { rgpot_force_input_free(&mut self.input) };
    }
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(rgpot_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn forces(output: &rgpot_force_out_t) -> Vec<f64> {
    let data = unsafe { rgpot_tensor_data(output.forces) } as *const f64;
    unsafe { std::slice::from_raw_parts(data, 9) }.to_vec()
}

#[test]
fn force_only_call_never_runs_the_stress_callback_and_a_stress_call_does() {
    let (port, seen) = serve(true);
    let host = CString::new("127.0.0.1").unwrap();
    let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), port) };
    assert!(!client.is_null());
    let case = Case::new();

    let forces_before = seen.force.load(Ordering::SeqCst);
    let stress_before = seen.stress.load(Ordering::SeqCst);
    let mut output = rgpot_force_out_t::default();
    let status = unsafe { rgpot_rpc_calculate(client, &case.input, &mut output) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
    assert_eq!(output.energy, -3.5);
    assert_eq!(forces(&output), vec![0.25; 9]);
    assert_eq!(seen.force.load(Ordering::SeqCst), forces_before + 1);
    assert_eq!(seen.stress.load(Ordering::SeqCst), stress_before);
    unsafe { rgpot_tensor_free(output.forces) };

    let mut output = rgpot_force_out_t::default();
    let mut stress = rgpot_stress_out_create();
    let status =
        unsafe { rgpot_rpc_calculate_stress(client, &case.input, &mut output, &mut stress) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
    assert_eq!(output.energy, -3.5);
    assert_eq!(stress.has_stress, 1);
    assert_eq!(stress.stress, STRESS);
    assert_eq!(seen.force.load(Ordering::SeqCst), forces_before + 1);
    assert_eq!(seen.stress.load(Ordering::SeqCst), stress_before + 1);
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_rpc_client_free(client);
    }
}

#[test]
fn a_server_without_stress_refuses_the_stress_call_and_still_answers_force_only() {
    let (port, _) = serve(false);
    let host = CString::new("127.0.0.1").unwrap();
    let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), port) };
    assert!(!client.is_null());
    let case = Case::new();

    let mut output = rgpot_force_out_t::default();
    let mut stress = rgpot_stress_out_create();
    let status =
        unsafe { rgpot_rpc_calculate_stress(client, &case.input, &mut output, &mut stress) };
    assert_eq!(status, rgpot_status_t::RGPOT_RPC_ERROR);
    assert!(
        last_error().to_lowercase().contains("stress"),
        "message must name the missing capability: {}",
        last_error()
    );
    assert_eq!(stress.has_stress, 0);
    assert_eq!(stress.stress, [0.0; 9]);

    let mut output = rgpot_force_out_t::default();
    let status = unsafe { rgpot_rpc_calculate(client, &case.input, &mut output) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
    assert_eq!(forces(&output), vec![0.25; 9]);
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_rpc_client_free(client);
    }
}

#[test]
fn an_undersized_stress_block_is_refused_before_any_connection() {
    let host = CString::new("127.0.0.1").unwrap();
    let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), free_port()) };
    assert!(!client.is_null());
    let case = Case::new();
    let mut output = rgpot_force_out_t::default();
    let mut stress = rgpot_stress_out_create();
    stress.struct_size = 8;
    let status =
        unsafe { rgpot_rpc_calculate_stress(client, &case.input, &mut output, &mut stress) };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert!(last_error().contains("struct_size"), "{}", last_error());
    unsafe { rgpot_rpc_client_free(client) };
}

/// A server frozen at protocol minor 1. It advertises the stress operation
/// and answers every call with a stress, because it cannot tell a request
/// for one from its absence.
mod minor_one {
    use super::*;
    use capnp::capability::Promise;
    use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
    use futures::AsyncReadExt;
    use rgpot_core::compat::{describe_server_with, PROTOCOL_MINOR};
    use rgpot_core::Potentials_capnp::potential;

    pub static CALCULATES: AtomicUsize = AtomicUsize::new(0);
    pub static ASKED_FOR_STRESS: AtomicUsize = AtomicUsize::new(0);

    struct Frozen;

    impl potential::Server for Frozen {
        fn get_capabilities(
            &mut self,
            _params: potential::GetCapabilitiesParams,
            mut results: potential::GetCapabilitiesResults,
        ) -> Promise<(), capnp::Error> {
            let mut caps = results.get().init_capabilities();
            describe_server_with(caps.reborrow(), false, true);
            caps.set_protocol_minor(PROTOCOL_MINOR - 1);
            Promise::ok(())
        }

        fn calculate(
            &mut self,
            params: potential::CalculateParams,
            mut results: potential::CalculateResults,
        ) -> Promise<(), capnp::Error> {
            CALCULATES.fetch_add(1, Ordering::SeqCst);
            let fip = params.get().and_then(|p| p.get_fip());
            let fip = match fip {
                Ok(fip) => fip,
                Err(e) => return Promise::err(e),
            };
            if fip.get_compute_stress() {
                ASKED_FOR_STRESS.fetch_add(1, Ordering::SeqCst);
            }
            let n = fip.get_atmnrs().map(|a| a.len()).unwrap_or(0);
            let mut result = results.get().init_result();
            result.set_energy(-3.5);
            let mut forces = result.reborrow().init_forces(3 * n);
            for i in 0..3 * n {
                forces.set(i, 0.25);
            }
            let mut stress = result.init_stress(9);
            for i in 0..9 {
                stress.set(i, STRESS[i as usize]);
            }
            Promise::ok(())
        }
    }

    pub fn serve() -> u16 {
        let port = free_port();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let local = tokio::task::LocalSet::new();
            local.block_on(&runtime, async move {
                let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
                    .await
                    .unwrap();
                loop {
                    let (stream, _) = listener.accept().await.unwrap();
                    let _ = stream.set_nodelay(true);
                    let (reader, writer) =
                        tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
                    let network = twoparty::VatNetwork::new(
                        futures::io::BufReader::new(reader),
                        futures::io::BufWriter::new(writer),
                        rpc_twoparty_capnp::Side::Server,
                        Default::default(),
                    );
                    let client = capnp_rpc::new_client::<potential::Client, _>(Frozen);
                    let system = RpcSystem::new(Box::new(network), Some(client.client));
                    tokio::task::spawn_local(system);
                }
            });
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
            assert!(Instant::now() < deadline, "server did not start");
            std::thread::sleep(Duration::from_millis(20));
        }
        port
    }
}

#[test]
fn a_minor_1_server_serves_force_only_and_refuses_a_stress_call_by_minor() {
    let port = minor_one::serve();
    let host = CString::new("127.0.0.1").unwrap();
    let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), port) };
    assert!(!client.is_null());
    let case = Case::new();

    let mut output = rgpot_force_out_t::default();
    let status = unsafe { rgpot_rpc_calculate(client, &case.input, &mut output) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
    assert_eq!(forces(&output), vec![0.25; 9]);
    assert_eq!(minor_one::CALCULATES.load(Ordering::SeqCst), 1);
    assert_eq!(
        minor_one::ASKED_FOR_STRESS.load(Ordering::SeqCst),
        0,
        "a force-only call must not request stress from a minor-1 peer"
    );
    unsafe { rgpot_tensor_free(output.forces) };

    let mut output = rgpot_force_out_t::default();
    let mut stress = rgpot_stress_out_create();
    let status =
        unsafe { rgpot_rpc_calculate_stress(client, &case.input, &mut output, &mut stress) };
    assert_eq!(status, rgpot_status_t::RGPOT_RPC_ERROR);
    assert!(
        last_error().contains("protocol minor 1 is below the required 2"),
        "{}",
        last_error()
    );
    assert_eq!(stress.has_stress, 0);
    assert_eq!(stress.stress, [0.0; 9]);
    assert_eq!(
        minor_one::CALCULATES.load(Ordering::SeqCst),
        1,
        "the refusal comes before any calculate"
    );
    unsafe { rgpot_rpc_client_free(client) };
}

#[test]
fn a_minor_2_server_honours_the_stress_request() {
    let (port, seen) = serve(true);
    let host = CString::new("127.0.0.1").unwrap();
    let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), port) };
    let case = Case::new();
    let mut output = rgpot_force_out_t::default();
    let mut stress = rgpot_stress_out_create();
    let status =
        unsafe { rgpot_rpc_calculate_stress(client, &case.input, &mut output, &mut stress) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
    assert_eq!(stress.has_stress, 1);
    assert_eq!(stress.stress, STRESS);
    assert_eq!(seen.stress.load(Ordering::SeqCst), 1);
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_rpc_client_free(client);
    }
}
