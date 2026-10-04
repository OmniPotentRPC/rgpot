// MIT License
// Copyright 2023--present rgpot developers

//! Async RPC client wrapping Cap'n Proto `Potential.calculate()`.
//!
//! The client owns a tokio runtime so that the C API can call it
//! synchronously.
//!
//! ## DLPack Integration
//!
//! Input tensors are read from `DLManagedTensorVersioned` pointers (CPU only
//! for RPC — GPU tensors would need a device-to-host copy first).  The
//! response forces are wrapped in an **owning** DLPack tensor so the caller
//! can free them via `rgpot_tensor_free`.

use capnp::Error as CapnpError;
use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
use dlpk::sys::DLDeviceType;
use futures::AsyncReadExt;
use tokio::runtime::Runtime;

use crate::rpc::schema::potential;
use crate::tensor::create_owned_f64_tensor;
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// RPC client that connects to a remote rgpot server.
pub struct RpcClient {
    runtime: Runtime,
    addr: String,
}

impl RpcClient {
    /// Create a new RPC client targeting `host:port`.
    ///
    /// The connection is established lazily on the first `calculate` call.
    pub fn new(host: &str, port: u16) -> Result<Self, String> {
        let runtime = Runtime::new().map_err(|e| format!("failed to create tokio runtime: {e}"))?;
        Ok(Self {
            runtime,
            addr: format!("{host}:{port}"),
        })
    }

    /// Perform a synchronous RPC calculation.
    ///
    /// Internally this blocks on the tokio runtime with a `LocalSet`
    /// (required because `capnp_rpc::RpcSystem` is `!Send`).
    pub fn calculate(
        &mut self,
        input: &rgpot_force_input_t,
        output: &mut rgpot_force_out_t,
    ) -> Result<(), String> {
        let local = tokio::task::LocalSet::new();
        local.block_on(&self.runtime, self.calculate_async(input, output))
    }

    async fn calculate_async(
        &self,
        input: &rgpot_force_input_t,
        output: &mut rgpot_force_out_t,
    ) -> Result<(), String> {
        // --- Extract data from DLPack tensors (CPU only) ---
        let n = unsafe { input.n_atoms() }
            .ok_or_else(|| "cannot determine n_atoms from input tensors".to_string())?;

        let (positions, atmnrs, box_data) = unsafe { extract_cpu_input(input, n)? };

        // --- Connect ---
        let stream = tokio::net::TcpStream::connect(&self.addr)
            .await
            .map_err(|e| format!("connection failed: {e}"))?;
        stream
            .set_nodelay(true)
            .map_err(|e| format!("set_nodelay failed: {e}"))?;

        let (reader, writer) = tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();

        let network = twoparty::VatNetwork::new(
            futures::io::BufReader::new(reader),
            futures::io::BufWriter::new(writer),
            rpc_twoparty_capnp::Side::Client,
            Default::default(),
        );

        let mut rpc_system = RpcSystem::new(Box::new(network), None);
        let potential_client: potential::Client =
            rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);

        tokio::task::spawn_local(rpc_system);

        let caps_reply = potential_client
            .get_capabilities_request()
            .send()
            .promise
            .await
            .map_err(|e| format!("capabilities: required a Capabilities message, received {e}"))?;
        let caps = caps_reply
            .get()
            .map_err(|e| format!("capabilities: required a Capabilities message, received {e}"))?
            .get_capabilities()
            .map_err(|e| format!("capabilities: required a Capabilities message, received {e}"))?;
        crate::compat::check_capabilities(caps, &crate::compat::Expectation::default())?;

        // --- Build capnp request ---
        let mut request = potential_client.calculate_request();
        {
            let mut fip = request.get().init_fip();

            let mut pos_builder = fip.reborrow().init_pos(positions.len() as u32);
            for (i, &val) in positions.iter().enumerate() {
                pos_builder.set(i as u32, val);
            }

            let mut atm_builder = fip.reborrow().init_atmnrs(atmnrs.len() as u32);
            for (i, &val) in atmnrs.iter().enumerate() {
                atm_builder.set(i as u32, val);
            }

            let mut box_builder = fip.init_box(9);
            for (i, &val) in box_data.iter().enumerate() {
                box_builder.set(i as u32, val);
            }
        }

        // --- Send and receive ---
        let response = request
            .send()
            .promise
            .await
            .map_err(|e: CapnpError| format!("RPC call failed: {e}"))?;

        let result = response
            .get()
            .map_err(|e| format!("failed to read response: {e}"))?
            .get_result()
            .map_err(|e| format!("failed to get result: {e}"))?;

        output.energy = result.get_energy();

        let forces = result
            .get_forces()
            .map_err(|e| format!("failed to read forces: {e}"))?;

        if forces.len() as usize != n * 3 {
            return Err(format!(
                "force array size mismatch: expected {}, got {}",
                n * 3,
                forces.len()
            ));
        }

        // Create an owning DLPack tensor for the forces
        let forces_vec: Vec<f64> = (0..forces.len()).map(|i| forces.get(i)).collect();
        output.forces = create_owned_f64_tensor(forces_vec, vec![n as i64, 3]);

        Ok(())
    }
}

/// Extract CPU data slices from DLPack input tensors.
///
/// # Safety
/// All tensor pointers in `input` must be valid DLPack tensors on kDLCPU.
unsafe fn extract_cpu_input(
    input: &rgpot_force_input_t,
    n: usize,
) -> Result<(&[f64], &[i32], &[f64]), String> {
    // Positions
    if input.positions.is_null() {
        return Err("positions tensor is NULL".into());
    }
    let pos_t = unsafe { &(*input.positions).dl_tensor };
    if pos_t.device.device_type != DLDeviceType::kDLCPU {
        return Err("RPC requires CPU tensors; positions is not on CPU".into());
    }
    let positions = unsafe { std::slice::from_raw_parts(pos_t.data as *const f64, n * 3) };

    // Atomic numbers
    if input.atomic_numbers.is_null() {
        return Err("atomic_numbers tensor is NULL".into());
    }
    let atm_t = unsafe { &(*input.atomic_numbers).dl_tensor };
    if atm_t.device.device_type != DLDeviceType::kDLCPU {
        return Err("RPC requires CPU tensors; atomic_numbers is not on CPU".into());
    }
    let atmnrs = unsafe { std::slice::from_raw_parts(atm_t.data as *const i32, n) };

    // Box matrix
    if input.box_matrix.is_null() {
        return Err("box_matrix tensor is NULL".into());
    }
    let box_t = unsafe { &(*input.box_matrix).dl_tensor };
    if box_t.device.device_type != DLDeviceType::kDLCPU {
        return Err("RPC requires CPU tensors; box_matrix is not on CPU".into());
    }
    let box_data = unsafe { std::slice::from_raw_parts(box_t.data as *const f64, 9) };

    Ok((positions, atmnrs, box_data))
}

#[cfg(test)]
mod tests {
    use super::RpcClient;
    use crate::rpc::schema::potential;
    use crate::tensor::{create_owned_f64_tensor, create_owned_i32_tensor, rgpot_tensor_free};
    use crate::types::{rgpot_force_input_t, rgpot_force_out_t};
    use crate::Potentials_capnp::capabilities::Operation;
    use capnp::Error as CapnpError;
    use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio::runtime::Runtime;
    use tokio_util::compat::TokioAsyncReadCompatExt;

    struct Peer {
        major: u16,
        called: Arc<AtomicBool>,
    }

    impl potential::Server for Peer {
        fn get_capabilities(
            &mut self,
            _: potential::GetCapabilitiesParams,
            mut results: potential::GetCapabilitiesResults,
        ) -> capnp::capability::Promise<(), CapnpError> {
            let mut caps = capnp_rpc::pry!(results.get()).init_capabilities();
            crate::compat::fill_compatibility(caps.reborrow());
            caps.set_protocol_major(self.major);
            let mut ops = caps.reborrow().init_operations(2);
            ops.set(0, Operation::Energy);
            ops.set(1, Operation::Forces);
            capnp::capability::Promise::ok(())
        }

        fn calculate(
            &mut self,
            _: potential::CalculateParams,
            mut results: potential::CalculateResults,
        ) -> capnp::capability::Promise<(), CapnpError> {
            self.called.store(true, Ordering::SeqCst);
            let mut result = capnp_rpc::pry!(results.get()).init_result();
            result.set_energy(1.25);
            let mut forces = result.init_forces(3);
            for i in 0..3 {
                forces.set(i, 0.0);
            }
            capnp::capability::Promise::ok(())
        }
    }

    fn serve(peer: Peer) -> u16 {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = Runtime::new().expect("runtime");
            let local = tokio::task::LocalSet::new();
            local.block_on(&runtime, async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                tx.send(listener.local_addr().expect("addr").port())
                    .expect("port");
                let (stream, _) = listener.accept().await.expect("accept");
                let _ = stream.set_nodelay(true);
                let client = capnp_rpc::new_client::<potential::Client, _>(peer);
                let (reader, writer) = TokioAsyncReadCompatExt::compat(stream).split();
                let network = twoparty::VatNetwork::new(
                    futures::io::BufReader::new(reader),
                    futures::io::BufWriter::new(writer),
                    rpc_twoparty_capnp::Side::Server,
                    Default::default(),
                );
                let rpc = RpcSystem::new(Box::new(network), Some(client.client));
                let _ = rpc.await;
            });
        });
        rx.recv_timeout(Duration::from_secs(5))
            .expect("server port")
    }

    fn one_atom() -> rgpot_force_input_t {
        rgpot_force_input_t {
            positions: create_owned_f64_tensor(vec![0.1, 0.2, 0.3], vec![1, 3]),
            atomic_numbers: create_owned_i32_tensor(vec![8], vec![1]),
            box_matrix: create_owned_f64_tensor(
                vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
                vec![3, 3],
            ),
        }
    }

    fn free_input(input: &rgpot_force_input_t) {
        unsafe {
            rgpot_tensor_free(input.positions);
            rgpot_tensor_free(input.atomic_numbers);
            rgpot_tensor_free(input.box_matrix);
        }
    }

    #[test]
    fn rpc_client_refuses_an_incompatible_peer_before_calculate() {
        let called = Arc::new(AtomicBool::new(false));
        let port = serve(Peer {
            major: 2,
            called: Arc::clone(&called),
        });
        let mut client = RpcClient::new("127.0.0.1", port).unwrap();
        let input = one_atom();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let err = client.calculate(&input, &mut output).unwrap_err();
        assert_eq!(err, "protocolMajor: required 1, received 2");
        assert!(!called.load(Ordering::SeqCst));
        free_input(&input);
    }

    #[test]
    fn rpc_client_dispatches_a_compatible_peer() {
        let called = Arc::new(AtomicBool::new(false));
        let port = serve(Peer {
            major: 1,
            called: Arc::clone(&called),
        });
        let mut client = RpcClient::new("127.0.0.1", port).unwrap();
        let input = one_atom();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        client.calculate(&input, &mut output).unwrap();
        assert!(called.load(Ordering::SeqCst));
        assert_eq!(output.energy, 1.25);
        unsafe { rgpot_tensor_free(output.forces) };
        free_input(&input);
    }
}
