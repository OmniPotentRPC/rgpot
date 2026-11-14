// MIT License
// Copyright 2023--present rgpot developers

//! Cap'n Proto RPC server that dispatches incoming `calculate` calls
//! to a `rgpot_potential_t` callback.
//!
//! ## DLPack Integration
//!
//! Incoming data is deserialized from Cap'n Proto into owned `Vec` values, then
//! wrapped in non-owning DLPack tensors for the callback.  The callback's
//! output forces (an owning DLPack tensor) are read back and serialized
//! into the Cap'n Proto response.

use capnp::Error as CapnpError;
use capnp_rpc::{pry, rpc_twoparty_capnp, twoparty, RpcSystem};
use futures::AsyncReadExt;
use std::os::raw::c_void;
use tokio::runtime::Runtime;

use crate::potential::{rgpot_potential_t, PotentialCallback};
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
    fn get_capabilities(
        &mut self,
        _params: potential::GetCapabilitiesParams,
        mut results: potential::GetCapabilitiesResults,
    ) -> capnp::capability::Promise<(), CapnpError> {
        crate::compat::describe_server(results.get().init_capabilities());
        capnp::capability::Promise::ok(())
    }

    fn calculate(
        &mut self,
        params: potential::CalculateParams,
        mut results: potential::CalculateResults,
    ) -> capnp::capability::Promise<(), CapnpError> {
        let fip = pry!(pry!(params.get()).get_fip());
        pry!(self.write_result(fip, results.get().init_result()));
        capnp::capability::Promise::ok(())
    }

    fn calculate_batch(
        &mut self,
        params: potential::CalculateBatchParams,
        mut results: potential::CalculateBatchResults,
    ) -> capnp::capability::Promise<(), CapnpError> {
        let items = pry!(pry!(params.get()).get_items());
        let mut out = results.get().init_results(items.len());
        for i in 0..items.len() {
            let fip = items.get(i);
            pry!(self.write_result(fip, out.reborrow().get(i)));
        }
        capnp::capability::Promise::ok(())
    }
}

impl PotentialServer {
    fn write_result(
        &self,
        fip: crate::Potentials_capnp::force_input::Reader<'_>,
        mut result_builder: crate::Potentials_capnp::potential_result::Builder<'_>,
    ) -> Result<(), CapnpError> {
        let positions = fip.get_pos()?;
        let atmnrs = fip.get_atmnrs()?;
        let box_data = fip.get_box()?;
        let n_atoms = atmnrs.len() as usize;

        let mut pos_vec: Vec<f64> = (0..positions.len()).map(|i| positions.get(i)).collect();
        let mut atm_vec: Vec<i32> = (0..atmnrs.len()).map(|i| atmnrs.get(i)).collect();
        let mut box_vec: Vec<f64> = (0..box_data.len()).map(|i| box_data.get(i)).collect();

        let pos_tensor =
            unsafe { rgpot_tensor_cpu_f64_2d(pos_vec.as_mut_ptr(), n_atoms as i64, 3) };
        let atm_tensor = unsafe { rgpot_tensor_cpu_i32_1d(atm_vec.as_mut_ptr(), n_atoms as i64) };
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

        let forces_data = if !output.forces.is_null() {
            let ft = unsafe { &(*output.forces).dl_tensor };
            let len = n_atoms * 3;
            unsafe { std::slice::from_raw_parts(ft.data as *const f64, len) }
        } else {
            &[] as &[f64]
        };
        let forces_copy: Vec<f64> = forces_data.to_vec();
        unsafe {
            rgpot_tensor_free(output.forces);
            rgpot_tensor_free(input.positions);
            rgpot_tensor_free(input.atomic_numbers);
            rgpot_tensor_free(input.box_matrix);
        }

        if status != rgpot_status_t::RGPOT_SUCCESS {
            return Err(CapnpError::failed(
                "potential callback returned an error".to_string(),
            ));
        }

        result_builder.set_energy(output.energy);
        let mut forces_builder = result_builder.init_forces(forces_copy.len() as u32);
        for (i, &f) in forces_copy.iter().enumerate() {
            forces_builder.set(i as u32, f);
        }
        Ok(())
    }
}

/// Remove `path` only when `lstat` says it is a socket. A regular file,
/// a directory, or a symlink is left untouched and reported.
#[cfg(unix)]
fn claim_socket_path(path: &str) -> Result<(), String> {
    use std::os::unix::fs::FileTypeExt;
    if path.is_empty() {
        return Err("unix socket path is empty".to_string());
    }
    match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            if meta.file_type().is_socket() {
                std::fs::remove_file(path).map_err(|e| format!("cannot remove socket {path}: {e}"))
            } else {
                Err(format!("{path} exists and is not a socket"))
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!("cannot stat {path}: {err}")),
    }
}

fn accept_client<S>(stream: S, pot: *const rgpot_potential_t)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + 'static,
{
    let pot_ref = unsafe { &*pot };
    let server_impl = PotentialServer {
        callback: pot_ref.callback,
        user_data: pot_ref.pot_user_data,
    };
    let potential_client = capnp_rpc::new_client::<potential::Client, _>(server_impl);
    let (reader, writer) = tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
    let network = twoparty::VatNetwork::new(
        futures::io::BufReader::new(reader),
        futures::io::BufWriter::new(writer),
        rpc_twoparty_capnp::Side::Server,
        Default::default(),
    );
    let rpc_system = RpcSystem::new(Box::new(network), Some(potential_client.client));
    tokio::task::spawn_local(rpc_system);
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

        let host_str = match unsafe { std::ffi::CStr::from_ptr(host) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("invalid host string: {e}"));
                return rgpot_status_t::RGPOT_INVALID_PARAMETER;
            }
        };

        // `unix:/path` listens on a domain socket. Anything else is TCP
        // `host:port`. TCP_NODELAY is set on each accepted TCP stream.
        // Windows has no UnixListener in this build, so that address fails
        // before a socket is opened.
        #[cfg(not(unix))]
        if host_str.starts_with("unix:") {
            set_last_error("unix sockets are not supported on this platform");
            return rgpot_status_t::RGPOT_RPC_ERROR;
        }
        #[cfg(unix)]
        let unix_path = host_str.strip_prefix("unix:").map(str::to_owned);
        let addr = if host_str.starts_with("unix:") {
            host_str.to_owned()
        } else {
            format!("{host_str}:{port}")
        };

        let runtime = match Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                set_last_error(&format!("failed to create runtime: {e}"));
                return rgpot_status_t::RGPOT_INTERNAL_ERROR;
            }
        };

        let local = tokio::task::LocalSet::new();
        local.block_on(&runtime, async move {
            #[cfg(unix)]
            if let Some(path) = unix_path.as_deref() {
                if let Err(message) = claim_socket_path(path) {
                    set_last_error(&message);
                    return rgpot_status_t::RGPOT_RPC_ERROR;
                }
                let listener = match tokio::net::UnixListener::bind(path) {
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
                    accept_client(stream, pot);
                }
            }

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
                accept_client(stream, pot);
            }
        })
    }))
}

#[cfg(all(test, unix))]
mod tests {
    use super::claim_socket_path;

    #[test]
    fn a_regular_file_is_not_removed() {
        let path = std::env::temp_dir().join(format!("rgpot-unix-file-{}", std::process::id()));
        std::fs::write(&path, b"keep").unwrap();
        let err = claim_socket_path(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("not a socket"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), b"keep");
        std::fs::remove_file(&path).unwrap();
    }
}
