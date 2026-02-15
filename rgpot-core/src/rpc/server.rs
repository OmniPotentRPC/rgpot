// MIT License
// Copyright 2023--present rgpot developers

//! Cap'n Proto RPC server that dispatches incoming `calculate` calls
//! to a `rgpot_potential_t` callback.
//!
//! ## DLPack Integration
//!
//! Incoming data is deserialized from Cap'n Proto into owned `Vec`s, then
//! wrapped in non-owning DLPack tensors for the callback.  The callback's
//! output forces (an owning DLPack tensor) are read back and serialized
//! into the Cap'n Proto response.

use capnp::Error as CapnpError;
use capnp_rpc::{pry, rpc_twoparty_capnp, twoparty, RpcSystem};
use futures::AsyncReadExt;
use std::os::raw::c_void;
use tokio::runtime::Runtime;

use crate::potential::{PotentialCallback, rgpot_potential_t};
use crate::rpc::schema::potential;
use crate::status::rgpot_status_t;
use crate::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_free,
};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// RPC server state wrapping a potential callback.
struct PotentialServer {
    callback: PotentialCallback,
    user_data: *mut c_void,
}

// Safety: the user_data pointer is only accessed through the callback
// which is expected to be thread-safe by the caller.
unsafe impl Send for PotentialServer {}
unsafe impl Sync for PotentialServer {}

impl potential::Server for PotentialServer {
    fn calculate(
        &mut self,
        params: potential::CalculateParams,
        mut results: potential::CalculateResults,
    ) -> capnp::capability::Promise<(), CapnpError> {
        let fip = pry!(pry!(params.get()).get_fip());

        let positions = pry!(fip.get_pos());
        let atmnrs = pry!(fip.get_atmnrs());
        let box_data = pry!(fip.get_box());

        let n_atoms = atmnrs.len() as usize;

        // Copy capnp data into owned buffers
        let mut pos_vec: Vec<f64> = (0..positions.len()).map(|i| positions.get(i)).collect();
        let mut atm_vec: Vec<i32> = (0..atmnrs.len()).map(|i| atmnrs.get(i)).collect();
        let mut box_vec: Vec<f64> = (0..box_data.len()).map(|i| box_data.get(i)).collect();

        // Create non-owning DLPack tensors wrapping the owned buffers
        let pos_tensor =
            unsafe { rgpot_tensor_cpu_f64_2d(pos_vec.as_mut_ptr(), n_atoms as i64, 3) };
        let atm_tensor =
            unsafe { rgpot_tensor_cpu_i32_1d(atm_vec.as_mut_ptr(), n_atoms as i64) };
        let box_tensor = unsafe { rgpot_tensor_cpu_f64_matrix3(box_vec.as_mut_ptr()) };

        let input = rgpot_force_input_t {
            positions: pos_tensor,
            atomic_numbers: atm_tensor,
            box_matrix: box_tensor,
        };
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };

        let status = unsafe { (self.callback)(self.user_data, &input, &mut output) };

        // Read forces from the output DLPack tensor before freeing
        let forces_data = if !output.forces.is_null() {
            let ft = unsafe { &(*output.forces).dl_tensor };
            let len = n_atoms * 3;
            unsafe { std::slice::from_raw_parts(ft.data as *const f64, len) }
        } else {
            &[] as &[f64]
        };

        // Free all tensors
        let forces_copy: Vec<f64> = forces_data.to_vec();
        unsafe {
            rgpot_tensor_free(output.forces);
            rgpot_tensor_free(input.positions);
            rgpot_tensor_free(input.atomic_numbers);
            rgpot_tensor_free(input.box_matrix);
        }

        if status != rgpot_status_t::RGPOT_SUCCESS {
            return capnp::capability::Promise::err(CapnpError::failed(
                "potential callback returned an error".to_string(),
            ));
        }

        let mut result_builder = results.get().init_result();
        result_builder.set_energy(output.energy);

        let mut forces_builder = result_builder.init_forces(forces_copy.len() as u32);
        for (i, &f) in forces_copy.iter().enumerate() {
            forces_builder.set(i as u32, f);
        }

        capnp::capability::Promise::ok(())
    }
}

/// Start an RPC server listening on `host:port`, dispatching to `pot`.
///
/// This function blocks the current thread. It creates its own tokio runtime.
///
/// # Safety
/// `pot` must be a valid pointer obtained from `rgpot_potential_new`.
/// The potential and its user_data must remain valid for the lifetime
/// of the server.
#[no_mangle]
pub unsafe extern "C" fn rgpot_rpc_server_start(
    pot: *const rgpot_potential_t,
    host: *const std::os::raw::c_char,
    port: u16,
) -> rgpot_status_t {
    use crate::status::{catch_unwind, set_last_error};

    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() {
            set_last_error("rgpot_rpc_server_start: pot is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        if host.is_null() {
            set_last_error("rgpot_rpc_server_start: host is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }

        let pot_ref = unsafe { &*pot };
        let host_str = match unsafe { std::ffi::CStr::from_ptr(host) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("invalid host string: {e}"));
                return rgpot_status_t::RGPOT_INVALID_PARAMETER;
            }
        };

        let addr = format!("{host_str}:{port}");

        let runtime = match Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                set_last_error(&format!("failed to create runtime: {e}"));
                return rgpot_status_t::RGPOT_INTERNAL_ERROR;
            }
        };

        let local = tokio::task::LocalSet::new();
        local.block_on(&runtime, async move {
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    set_last_error(&format!("failed to bind {addr}: {e}"));
                    return rgpot_status_t::RGPOT_RPC_ERROR;
                }
            };

            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(e) => {
                        set_last_error(&format!("accept failed: {e}"));
                        return rgpot_status_t::RGPOT_RPC_ERROR;
                    }
                };

                let _ = stream.set_nodelay(true);

                let server_impl = PotentialServer {
                    callback: pot_ref.callback,
                    user_data: pot_ref.user_data,
                };

                let potential_client =
                    capnp_rpc::new_client::<potential::Client, _>(server_impl);

                let (reader, writer) =
                    tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();

                let network = twoparty::VatNetwork::new(
                    futures::io::BufReader::new(reader),
                    futures::io::BufWriter::new(writer),
                    rpc_twoparty_capnp::Side::Server,
                    Default::default(),
                );

                let rpc_system =
                    RpcSystem::new(Box::new(network), Some(potential_client.client));

                tokio::task::spawn_local(rpc_system);
            }
        })
    }))
}
