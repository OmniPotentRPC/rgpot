// MIT License
// Copyright 2023--present rgpot developers

//! Async RPC client wrapping Cap'n Proto `Potential.calculate()`.
//!
//! The client owns a tokio runtime so that the C API can call it
//! synchronously.

use capnp::Error as CapnpError;
use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
use tokio::runtime::Runtime;

use crate::rpc::schema::potential;
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
    /// Internally this blocks on the tokio runtime.
    pub fn calculate(
        &mut self,
        input: &rgpot_force_input_t,
        output: &mut rgpot_force_out_t,
    ) -> Result<(), String> {
        self.runtime.block_on(self.calculate_async(input, output))
    }

    async fn calculate_async(
        &self,
        input: &rgpot_force_input_t,
        output: &mut rgpot_force_out_t,
    ) -> Result<(), String> {
        let stream = tokio::net::TcpStream::connect(&self.addr)
            .await
            .map_err(|e| format!("connection failed: {e}"))?;
        stream
            .set_nodelay(true)
            .map_err(|e| format!("set_nodelay failed: {e}"))?;

        let (reader, writer) =
            tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();

        let network = twoparty::VatNetwork::new(
            reader,
            writer,
            rpc_twoparty_capnp::Side::Client,
            Default::default(),
        );

        let mut rpc_system = RpcSystem::new(Box::new(network), None);
        let potential_client: potential::Client =
            rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);

        tokio::task::spawn_local(rpc_system);

        let n = input.n_atoms;

        let mut request = potential_client.calculate_request();
        {
            let mut fip = request.get().init_fip();

            // Positions
            let positions = unsafe { std::slice::from_raw_parts(input.pos, n * 3) };
            let mut pos_builder = fip.reborrow().init_pos(positions.len() as u32);
            for (i, &val) in positions.iter().enumerate() {
                pos_builder.set(i as u32, val);
            }

            // Atomic numbers
            let atmnrs = unsafe { std::slice::from_raw_parts(input.atmnrs, n) };
            let mut atm_builder = fip.reborrow().init_atmnrs(atmnrs.len() as u32);
            for (i, &val) in atmnrs.iter().enumerate() {
                atm_builder.set(i as u32, val);
            }

            // Box
            let box_data = unsafe { std::slice::from_raw_parts(input.box_, 9) };
            let mut box_builder = fip.init_box(9);
            for (i, &val) in box_data.iter().enumerate() {
                box_builder.set(i as u32, val);
            }
        }

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

        let out_forces = unsafe { std::slice::from_raw_parts_mut(output.forces, n * 3) };
        for i in 0..forces.len() {
            out_forces[i as usize] = forces.get(i);
        }

        Ok(())
    }
}
