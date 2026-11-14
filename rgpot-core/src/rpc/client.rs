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
        // `unix:/path` is the same address KJ accepts. A hostname still
        // joins the port, which is the TCP default.
        let addr = if host.starts_with("unix:") {
            host.to_owned()
        } else {
            format!("{host}:{port}")
        };
        Ok(Self { runtime, addr })
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
        // TCP_NODELAY is set on TCP. A unix socket has no such option.
        // The two streams have different types, so each side calls exchange.
        if let Some(path) = self.addr.strip_prefix("unix:") {
            #[cfg(unix)]
            {
                let stream = tokio::net::UnixStream::connect(path)
                    .await
                    .map_err(|e| format!("connection failed: {e}"))?;
                let (reader, writer) =
                    tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
                return exchange(reader, writer, n, positions, atmnrs, box_data, output).await;
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                return Err("unix sockets are not supported on this platform".to_string());
            }
        }
        let stream = tokio::net::TcpStream::connect(&self.addr)
            .await
            .map_err(|e| format!("connection failed: {e}"))?;
        stream
            .set_nodelay(true)
            .map_err(|e| format!("set_nodelay failed: {e}"))?;
        let (reader, writer) = tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
        exchange(reader, writer, n, positions, atmnrs, box_data, output).await
    }

    /// Several geometries in one round trip. Image `i` of the result is the
    /// same energy and forces `calculate` returns for that geometry.
    pub fn calculate_batch(&mut self, images: &[ImageIn<'_>]) -> Result<Vec<ImageOut>, String> {
        let local = tokio::task::LocalSet::new();
        local.block_on(&self.runtime, self.calculate_batch_async(images))
    }

    async fn calculate_batch_async(&self, images: &[ImageIn<'_>]) -> Result<Vec<ImageOut>, String> {
        if let Some(path) = self.addr.strip_prefix("unix:") {
            #[cfg(unix)]
            {
                let stream = tokio::net::UnixStream::connect(path)
                    .await
                    .map_err(|e| format!("connection failed: {e}"))?;
                let (reader, writer) =
                    tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
                return exchange_batch(reader, writer, images).await;
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                return Err("unix sockets are not supported on this platform".to_string());
            }
        }
        let stream = tokio::net::TcpStream::connect(&self.addr)
            .await
            .map_err(|e| format!("connection failed: {e}"))?;
        stream
            .set_nodelay(true)
            .map_err(|e| format!("set_nodelay failed: {e}"))?;
        let (reader, writer) = tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
        exchange_batch(reader, writer, images).await
    }
}

/// One geometry passed to [`RpcClient::calculate_batch`].
pub struct ImageIn<'a> {
    pub positions: &'a [f64],
    pub atomic_numbers: &'a [i32],
    pub cell: &'a [f64],
}

/// Energy and forces for one image, in the request order.
pub struct ImageOut {
    pub energy: f64,
    pub forces: Vec<f64>,
}

async fn exchange<R, W>(
    reader: R,
    writer: W,
    n: usize,
    positions: &[f64],
    atmnrs: &[i32],
    box_data: &[f64],
    output: &mut rgpot_force_out_t,
) -> Result<(), String>
where
    R: futures::io::AsyncRead + Unpin + 'static,
    W: futures::io::AsyncWrite + Unpin + 'static,
{
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

    // --- Refuse an incompatible or outdated server before any work ---
    crate::compat::check_server(&potential_client, &crate::compat::Expectation::default()).await?;

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

async fn exchange_batch<R, W>(
    reader: R,
    writer: W,
    images: &[ImageIn<'_>],
) -> Result<Vec<ImageOut>, String>
where
    R: futures::io::AsyncRead + Unpin + 'static,
    W: futures::io::AsyncWrite + Unpin + 'static,
{
    for image in images {
        if image.positions.len() != image.atomic_numbers.len() * 3 {
            return Err("positions length is not 3 * atomic_numbers".to_string());
        }
        if image.cell.len() != 9 {
            return Err("cell must hold 9 values".to_string());
        }
    }

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
    crate::compat::check_server(&potential_client, &crate::compat::Expectation::default()).await?;

    let mut request = potential_client.calculate_batch_request();
    {
        let mut items = request.get().init_items(images.len() as u32);
        for (k, image) in images.iter().enumerate() {
            let mut fip = items.reborrow().get(k as u32);
            let mut pos_builder = fip.reborrow().init_pos(image.positions.len() as u32);
            for (i, &val) in image.positions.iter().enumerate() {
                pos_builder.set(i as u32, val);
            }
            let mut atm_builder = fip
                .reborrow()
                .init_atmnrs(image.atomic_numbers.len() as u32);
            for (i, &val) in image.atomic_numbers.iter().enumerate() {
                atm_builder.set(i as u32, val);
            }
            let mut box_builder = fip.init_box(9);
            for (i, &val) in image.cell.iter().enumerate() {
                box_builder.set(i as u32, val);
            }
        }
    }

    let response = request
        .send()
        .promise
        .await
        .map_err(|e: CapnpError| format!("RPC call failed: {e}"))?;
    let results = response
        .get()
        .map_err(|e| format!("failed to read response: {e}"))?
        .get_results()
        .map_err(|e| format!("failed to get results: {e}"))?;
    if results.len() as usize != images.len() {
        return Err(format!(
            "batch result count mismatch: expected {}, got {}",
            images.len(),
            results.len()
        ));
    }

    let mut out = Vec::with_capacity(images.len());
    for (k, image) in images.iter().enumerate() {
        let result = results.get(k as u32);
        let forces = result
            .get_forces()
            .map_err(|e| format!("failed to read forces: {e}"))?;
        let n3 = image.atomic_numbers.len() * 3;
        if forces.len() as usize != n3 {
            return Err(format!(
                "force array size mismatch: expected {n3}, got {}",
                forces.len()
            ));
        }
        out.push(ImageOut {
            energy: result.get_energy(),
            forces: (0..forces.len()).map(|i| forces.get(i)).collect(),
        });
    }
    Ok(out)
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
    use super::{ImageIn, RpcClient};
    use crate::potential::PotentialCallback;
    use crate::status::rgpot_status_t;
    use crate::tensor::{create_owned_f64_tensor, rgpot_tensor_free};
    use crate::types::{rgpot_force_input_t, rgpot_force_out_t};
    use std::ffi::CString;
    use std::os::raw::c_void;

    unsafe extern "C" fn echo_positions(
        _user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let input = unsafe { &*input };
        let n = unsafe { input.n_atoms() }.unwrap_or(0);
        let pos = unsafe {
            let t = &(*input.positions).dl_tensor;
            std::slice::from_raw_parts(t.data as *const f64, n * 3)
        };
        let energy = pos.iter().sum();
        unsafe {
            (*output).energy = energy;
            (*output).forces = create_owned_f64_tensor(pos.to_vec(), vec![n as i64, 3]);
        }
        rgpot_status_t::RGPOT_SUCCESS
    }

    fn bits(x: f64) -> u64 {
        u64::from_ne_bytes(x.to_ne_bytes())
    }

    struct SendPot(*mut crate::potential::rgpot_potential_t);
    unsafe impl Send for SendPot {}

    fn serve(pot: SendPot, host: &std::ffi::CStr, port: u16) {
        unsafe { crate::rpc::server::rgpot_rpc_server_start(pot.0, host.as_ptr(), port) };
    }

    #[test]
    fn batch_matches_calculate() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let pot = unsafe {
            crate::c_api::potential::rgpot_potential_new(
                echo_positions as PotentialCallback,
                std::ptr::null_mut(),
                None,
            )
        };
        assert!(!pot.is_null());
        let pot = SendPot(pot);
        let host = CString::new("127.0.0.1").unwrap();
        std::thread::spawn(move || serve(pot, &host, port));

        let mut client = None;
        for _ in 0..50 {
            if let Ok(c) = RpcClient::new("127.0.0.1", port) {
                client = Some(c);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let mut client = client.expect("client");
        // The constructor does not connect. Probe until the server accepts.
        let probe_pos = vec![0.0_f64, 0.0, 0.0];
        let probe_z = vec![1_i32];
        let probe_cell = vec![10.0_f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let mut ready = false;
        for _ in 0..50 {
            let images = [ImageIn {
                positions: &probe_pos,
                atomic_numbers: &probe_z,
                cell: &probe_cell,
            }];
            if client.calculate_batch(&images).is_ok() {
                ready = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(ready, "server did not accept a batch");

        let images = [
            (
                vec![0.2_f64, -0.4, 0.6, 1.5, 0.0, -0.25],
                vec![13_i32, 13],
                vec![8.0, 0.1, 0.0, 0.0, 8.0, 0.0, 0.0, 0.2, 8.0],
            ),
            (
                vec![1.0_f64, 2.0, 3.0],
                vec![18_i32],
                vec![12.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 12.0],
            ),
        ];
        let batch = client
            .calculate_batch(&[
                ImageIn {
                    positions: &images[0].0,
                    atomic_numbers: &images[0].1,
                    cell: &images[0].2,
                },
                ImageIn {
                    positions: &images[1].0,
                    atomic_numbers: &images[1].1,
                    cell: &images[1].2,
                },
            ])
            .unwrap();

        for (k, (pos, z, cell)) in images.iter().enumerate() {
            let mut pos_buf = pos.clone();
            let mut z_buf = z.clone();
            let mut cell_buf = cell.clone();
            let input = rgpot_force_input_t {
                positions: unsafe {
                    crate::tensor::rgpot_tensor_cpu_f64_2d(
                        pos_buf.as_mut_ptr(),
                        (pos_buf.len() / 3) as i64,
                        3,
                    )
                },
                atomic_numbers: unsafe {
                    crate::tensor::rgpot_tensor_cpu_i32_1d(z_buf.as_mut_ptr(), z_buf.len() as i64)
                },
                box_matrix: unsafe {
                    crate::tensor::rgpot_tensor_cpu_f64_matrix3(cell_buf.as_mut_ptr())
                },
            };
            let mut output = rgpot_force_out_t {
                forces: std::ptr::null_mut(),
                energy: 0.0,
                variance: 0.0,
            };
            client.calculate(&input, &mut output).unwrap();
            assert_eq!(bits(output.energy), bits(batch[k].energy));
            let got = unsafe {
                let t = &(*output.forces).dl_tensor;
                std::slice::from_raw_parts(t.data as *const f64, pos.len())
            };
            assert_eq!(got.len(), batch[k].forces.len());
            for (a, b) in got.iter().zip(batch[k].forces.iter()) {
                assert_eq!(bits(*a), bits(*b));
            }
            unsafe {
                rgpot_tensor_free(output.forces);
                rgpot_tensor_free(input.positions);
                rgpot_tensor_free(input.atomic_numbers);
                rgpot_tensor_free(input.box_matrix);
            }
        }
    }
}
