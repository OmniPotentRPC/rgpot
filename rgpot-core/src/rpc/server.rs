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
use std::os::raw::{c_char, c_void};
use tokio::runtime::Runtime;

use crate::hvp::rgpot_hvp_scheme_t;
use crate::potential::{rgpot_potential_t, PotentialCallback};
use crate::rpc::schema::potential;
use crate::status::rgpot_status_t;
use crate::tensor::{
    copy_f64_tensor, rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3,
    rgpot_tensor_cpu_i32_1d, rgpot_tensor_free,
};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

const DEFAULT_STREAM_LIMIT: usize = 4096;

#[repr(C)]
struct NlistRequest {
    bytes: u32,
    abi: u32,
    cutoff: f64,
    skin: f64,
    shape: i32,
    shell: i32,
    cut: i32,
    periodic: [i32; 3],
    want_shifts: i32,
    want_vectors: i32,
    want_distances: i32,
}

#[repr(C)]
struct NlistSoa {
    bytes: u32,
    first: *mut i32,
    second: *mut i32,
    shift: *mut i32,
    vector: *mut f64,
    distance: *mut f64,
    mask: *mut u8,
    cap: usize,
}

extern "C" {
    fn rgpot_nlist_request_init(request: *mut NlistRequest);
    fn rgpot_nlist_build(
        positions: *const f64,
        n: usize,
        box9: *const f64,
        request: *const NlistRequest,
        out: *mut NlistSoa,
        count: *mut usize,
        needed: *mut usize,
    ) -> i32;
    fn rgpot_nlist_build_systems(
        positions: *const f64,
        counts: *const i32,
        nsystems: usize,
        box9: *const f64,
        request: *const NlistRequest,
        out: *mut NlistSoa,
        system: *mut i32,
        count: *mut usize,
        needed: *mut usize,
    ) -> i32;
    fn rgpot_nlist_strerror(code: i32) -> *const c_char;
}

fn nlist_error_text(code: i32) -> String {
    let ptr = unsafe { rgpot_nlist_strerror(code) };
    if ptr.is_null() {
        return "unknown neighbour error".to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// RPC server state wrapping a potential callback.
struct PotentialServer {
    callback: PotentialCallback,
    user_data: *mut c_void,
    potential: *const rgpot_potential_t,
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
        let analytic_hvp =
            !self.potential.is_null() && unsafe { &*self.potential }.has_hessian_vector_callback();
        crate::compat::describe_server(results.get().init_capabilities(), analytic_hvp);
        capnp::capability::Promise::ok(())
    }

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
        if positions.len() as usize != 3 * n_atoms {
            return capnp::capability::Promise::err(CapnpError::failed(format!(
                "positions: expected {} values for {n_atoms} atoms, got {}",
                3 * n_atoms,
                positions.len()
            )));
        }
        if box_data.len() != 9 {
            return capnp::capability::Promise::err(CapnpError::failed(format!(
                "box: expected 9 values, got {}",
                box_data.len()
            )));
        }

        // Copy capnp data into owned buffers
        let mut pos_vec: Vec<f64> = (0..positions.len()).map(|i| positions.get(i)).collect();
        let mut atm_vec: Vec<i32> = (0..atmnrs.len()).map(|i| atmnrs.get(i)).collect();
        let mut box_vec: Vec<f64> = (0..box_data.len()).map(|i| box_data.get(i)).collect();

        // Create non-owning DLPack tensors wrapping the owned buffers
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

        let forces_copy = if status == rgpot_status_t::RGPOT_SUCCESS {
            copy_f64_tensor(output.forces, "callback forces", n_atoms * 3, false)
        } else {
            Ok(Vec::new())
        };

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
        let forces_copy = match forces_copy {
            Ok(f) => f,
            Err(why) => return capnp::capability::Promise::err(CapnpError::failed(why)),
        };

        let mut result_builder = results.get().init_result();
        result_builder.set_energy(output.energy);

        let mut forces_builder = result_builder
            .reborrow()
            .init_forces(forces_copy.len() as u32);
        for (i, &f) in forces_copy.iter().enumerate() {
            forces_builder.set(i as u32, f);
        }

        if fip.has_neighbors() {
            let mut list_out = result_builder.reborrow().init_neighbor_list();
            let neighbor_request = match fip.get_neighbors() {
                Ok(req) => req,
                Err(error) => {
                    list_out.set_status(5);
                    list_out.set_message(format!("invalid neighbors field: {error}"));
                    list_out.set_more(false);
                    list_out.set_next(0);
                    return capnp::capability::Promise::ok(());
                }
            };
            if let Err((code, msg)) = answer_neighbours(fip, neighbor_request, list_out.reborrow())
            {
                list_out.set_status(code);
                list_out.set_message(&msg);
                list_out.set_more(false);
                list_out.set_next(0);
            }
        }

        capnp::capability::Promise::ok(())
    }

    fn calculate_hvp(
        &mut self,
        params: potential::CalculateHvpParams,
        mut results: potential::CalculateHvpResults,
    ) -> capnp::capability::Promise<(), CapnpError> {
        if self.potential.is_null() {
            return capnp::capability::Promise::err(CapnpError::failed(
                "server has no potential handle".to_string(),
            ));
        }
        let params = pry!(params.get());
        let fip = pry!(params.get_fip());
        let positions = pry!(fip.get_pos());
        let atmnrs = pry!(fip.get_atmnrs());
        let box_data = pry!(fip.get_box());
        let direction = pry!(params.get_direction());
        let n_atoms = atmnrs.len() as usize;
        let dimension = 3 * n_atoms;
        if positions.len() as usize != dimension || direction.len() as usize != dimension {
            return capnp::capability::Promise::err(CapnpError::failed(
                "HVP position or direction size mismatch".to_string(),
            ));
        }
        if box_data.len() != 9 {
            return capnp::capability::Promise::err(CapnpError::failed(
                "HVP box size mismatch".to_string(),
            ));
        }
        let mut pos_vec: Vec<f64> = (0..positions.len()).map(|i| positions.get(i)).collect();
        let mut atom_vec: Vec<i32> = (0..atmnrs.len()).map(|i| atmnrs.get(i)).collect();
        let mut box_vec: Vec<f64> = (0..box_data.len()).map(|i| box_data.get(i)).collect();
        let direction_vec: Vec<f64> = (0..direction.len()).map(|i| direction.get(i)).collect();
        let pos_tensor =
            unsafe { rgpot_tensor_cpu_f64_2d(pos_vec.as_mut_ptr(), n_atoms as i64, 3) };
        let atom_tensor = unsafe { rgpot_tensor_cpu_i32_1d(atom_vec.as_mut_ptr(), n_atoms as i64) };
        let box_tensor = unsafe { rgpot_tensor_cpu_f64_matrix3(box_vec.as_mut_ptr()) };
        let input = rgpot_force_input_t {
            positions: pos_tensor,
            atomic_numbers: atom_tensor,
            box_matrix: box_tensor,
        };
        let mut product = vec![0.0; dimension];
        let answer = unsafe {
            (&*self.potential).hessian_vector(
                &input,
                &direction_vec,
                0.0,
                rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
                &mut product,
            )
        };
        unsafe {
            rgpot_tensor_free(input.positions);
            rgpot_tensor_free(input.atomic_numbers);
            rgpot_tensor_free(input.box_matrix);
        }
        if let Err(error) = answer {
            return capnp::capability::Promise::err(CapnpError::failed(error.to_string()));
        }
        let mut builder = results.get().init_product(dimension as u32);
        for (i, value) in product.into_iter().enumerate() {
            builder.set(i as u32, value);
        }
        capnp::capability::Promise::ok(())
    }

    fn neighbours(
        &mut self,
        params: potential::NeighboursParams,
        mut results: potential::NeighboursResults,
    ) -> capnp::capability::Promise<(), CapnpError> {
        let params = pry!(params.get());
        let fip = pry!(params.get_fip());
        let request = pry!(params.get_request());
        let mut out = results.get().init_neighbors();
        if let Err((code, msg)) = answer_neighbours(fip, request, out.reborrow()) {
            out.set_status(code);
            out.set_message(&msg);
            out.set_more(false);
            out.set_next(0);
        }
        capnp::capability::Promise::ok(())
    }
}

fn answer_neighbours(
    fip: crate::rpc::schema::force_input::Reader<'_>,
    request: crate::rpc::schema::neighbor_request::Reader<'_>,
    mut out: crate::rpc::schema::neighbor_list::Builder<'_>,
) -> Result<(), (i32, String)> {
    let mode = request.get_mode();
    if mode != 1 && mode != 2 {
        return Err((5, "mode must be 1 (stored) or 2 (stream)".to_string()));
    }
    let format = request.get_format();
    if format > 2 {
        return Err((
            5,
            "format must be 0 (half pairs), 1 (full pairs) or 2 (cluster of four)".to_string(),
        ));
    }
    let cut = request.get_cut();
    if cut > 1 {
        return Err((5, "cut must be 0 (strict) or 1 (loose)".to_string()));
    }
    let periodic_reader = request
        .get_periodic()
        .map_err(|e| (5, format!("invalid periodic list: {e}")))?;
    if !periodic_reader.is_empty() && periodic_reader.len() != 3 {
        return Err((5, "periodic must be empty or hold 3 flags".to_string()));
    }
    let pos_reader = fip
        .get_pos()
        .map_err(|e| (2, format!("invalid positions: {e}")))?;
    if pos_reader.len() % 3 != 0 {
        return Err((5, "positions length must be a multiple of 3".to_string()));
    }
    let n_atoms = (pos_reader.len() / 3) as usize;
    let atm_reader = fip
        .get_atmnrs()
        .map_err(|e| (2, format!("invalid atomic numbers: {e}")))?;
    if !atm_reader.is_empty() && atm_reader.len() as usize != n_atoms {
        return Err((5, "atomic numbers size mismatch".to_string()));
    }
    let positions: Vec<f64> = (0..pos_reader.len()).map(|i| pos_reader.get(i)).collect();

    let mut c_req = unsafe { std::mem::zeroed::<NlistRequest>() };
    unsafe { rgpot_nlist_request_init(&mut c_req) };
    c_req.cutoff = request.get_cutoff();
    c_req.skin = request.get_skin();
    if format == 2 {
        c_req.shape = 1;
        c_req.shell = 1;
    } else {
        c_req.shape = 0;
        c_req.shell = i32::from(format == 1);
    }
    c_req.cut = i32::from(cut);
    if periodic_reader.len() == 3 {
        for (k, periodic) in c_req.periodic.iter_mut().enumerate() {
            *periodic = i32::from(periodic_reader.get(k as u32));
        }
    }
    c_req.want_shifts = i32::from(request.get_shifts());
    c_req.want_vectors = i32::from(request.get_vectors());
    c_req.want_distances = i32::from(request.get_distances());

    let systems_reader = request
        .get_systems()
        .map_err(|e| (5, format!("invalid systems list: {e}")))?;
    let mut first = Vec::new();
    let mut second = Vec::new();
    let mut shift = Vec::new();
    let mut vector = Vec::new();
    let mut distance = Vec::new();
    let mut mask = Vec::new();
    let mut system_col = Vec::new();

    if !systems_reader.is_empty() {
        let mut counts = Vec::with_capacity(systems_reader.len() as usize);
        let mut boxes = Vec::with_capacity(systems_reader.len() as usize * 9);
        let mut total_atoms = 0usize;
        for s in 0..systems_reader.len() {
            let entry = systems_reader.get(s);
            let cnt = entry.get_count() as usize;
            counts.push(cnt as i32);
            total_atoms += cnt;
            let b = entry
                .get_box()
                .map_err(|e| (4, format!("invalid system box: {e}")))?;
            if b.len() != 9 {
                return Err((4, "each system box must have 9 elements".to_string()));
            }
            for k in 0..9 {
                boxes.push(b.get(k));
            }
        }
        if total_atoms != n_atoms {
            return Err((
                5,
                "systems counts do not sum to the number of positions".to_string(),
            ));
        }
        let mut count = 0usize;
        let mut needed = 0usize;
        let rc = unsafe {
            rgpot_nlist_build_systems(
                positions.as_ptr(),
                counts.as_ptr(),
                counts.len(),
                boxes.as_ptr(),
                &c_req,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut count,
                &mut needed,
            )
        };
        if rc != 0 {
            return Err((rc, nlist_error_text(rc)));
        }
        first.resize(needed, 0);
        second.resize(needed, 0);
        mask.resize(needed, 0);
        system_col.resize(needed, 0);
        if c_req.want_shifts != 0 {
            shift.resize(needed * 3, 0);
        }
        if c_req.want_vectors != 0 {
            vector.resize(needed * 3, 0.0);
        }
        if c_req.want_distances != 0 {
            distance.resize(needed, 0.0);
        }
        let mut soa = NlistSoa {
            bytes: std::mem::size_of::<NlistSoa>() as u32,
            first: first.as_mut_ptr(),
            second: second.as_mut_ptr(),
            shift: if c_req.want_shifts != 0 {
                shift.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            vector: if c_req.want_vectors != 0 {
                vector.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            distance: if c_req.want_distances != 0 {
                distance.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            mask: mask.as_mut_ptr(),
            cap: needed,
        };
        let rc = unsafe {
            rgpot_nlist_build_systems(
                positions.as_ptr(),
                counts.as_ptr(),
                counts.len(),
                boxes.as_ptr(),
                &c_req,
                &mut soa,
                system_col.as_mut_ptr(),
                &mut count,
                &mut needed,
            )
        };
        if rc != 0 {
            return Err((rc, nlist_error_text(rc)));
        }
    } else {
        let box_reader = fip
            .get_box()
            .map_err(|e| (4, format!("invalid box: {e}")))?;
        if box_reader.len() != 9 {
            return Err((4, "box must have 9 elements".to_string()));
        }
        let mut box9 = [0.0; 9];
        for (k, slot) in box9.iter_mut().enumerate() {
            *slot = box_reader.get(k as u32);
        }
        let mut count = 0usize;
        let mut needed = 0usize;
        let rc = unsafe {
            rgpot_nlist_build(
                positions.as_ptr(),
                n_atoms,
                box9.as_ptr(),
                &c_req,
                std::ptr::null_mut(),
                &mut count,
                &mut needed,
            )
        };
        if rc != 0 {
            return Err((rc, nlist_error_text(rc)));
        }
        first.resize(needed, 0);
        second.resize(needed, 0);
        mask.resize(needed, 0);
        if c_req.want_shifts != 0 {
            shift.resize(needed * 3, 0);
        }
        if c_req.want_vectors != 0 {
            vector.resize(needed * 3, 0.0);
        }
        if c_req.want_distances != 0 {
            distance.resize(needed, 0.0);
        }
        let mut soa = NlistSoa {
            bytes: std::mem::size_of::<NlistSoa>() as u32,
            first: first.as_mut_ptr(),
            second: second.as_mut_ptr(),
            shift: if c_req.want_shifts != 0 {
                shift.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            vector: if c_req.want_vectors != 0 {
                vector.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            distance: if c_req.want_distances != 0 {
                distance.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            mask: mask.as_mut_ptr(),
            cap: needed,
        };
        let rc = unsafe {
            rgpot_nlist_build(
                positions.as_ptr(),
                n_atoms,
                box9.as_ptr(),
                &c_req,
                &mut soa,
                &mut count,
                &mut needed,
            )
        };
        if rc != 0 {
            return Err((rc, nlist_error_text(rc)));
        }
    }

    let total = first.len();
    let (start, count, more, next) = if mode == 2 {
        let cursor = request.get_cursor() as usize;
        if cursor > total {
            return Err((
                6,
                "cursor is past the end of the neighbour list".to_string(),
            ));
        }
        let raw_limit = request.get_limit() as usize;
        let limit = if raw_limit == 0 {
            DEFAULT_STREAM_LIMIT
        } else {
            raw_limit
        };
        let count = limit.min(total - cursor);
        let next = cursor + count;
        (cursor, count, next < total, next as u64)
    } else {
        (0, total, false, total as u64)
    };

    let mut first_out = out.reborrow().init_first(count as u32);
    for i in 0..count {
        first_out.set(i as u32, first[start + i]);
    }
    let mut second_out = out.reborrow().init_second(count as u32);
    for i in 0..count {
        second_out.set(i as u32, second[start + i]);
    }
    let mut mask_out = out.reborrow().init_mask(count as u32);
    for i in 0..count {
        mask_out.set(i as u32, mask[start + i]);
    }
    if c_req.want_shifts != 0 {
        let mut shift_out = out.reborrow().init_shift((count * 3) as u32);
        for i in 0..count {
            let src = start + i;
            shift_out.set((3 * i) as u32, shift[3 * src]);
            shift_out.set((3 * i + 1) as u32, shift[3 * src + 1]);
            shift_out.set((3 * i + 2) as u32, shift[3 * src + 2]);
        }
    }
    if c_req.want_vectors != 0 {
        let mut vec_out = out.reborrow().init_vector((count * 3) as u32);
        for i in 0..count {
            let src = start + i;
            vec_out.set((3 * i) as u32, vector[3 * src]);
            vec_out.set((3 * i + 1) as u32, vector[3 * src + 1]);
            vec_out.set((3 * i + 2) as u32, vector[3 * src + 2]);
        }
    }
    if c_req.want_distances != 0 {
        let mut dist_out = out.reborrow().init_distance(count as u32);
        for i in 0..count {
            dist_out.set(i as u32, distance[start + i]);
        }
    }
    if !system_col.is_empty() {
        let mut sys_out = out.reborrow().init_system(count as u32);
        for i in 0..count {
            sys_out.set(i as u32, system_col[start + i]);
        }
    }
    out.set_more(more);
    out.set_next(next);
    out.set_status(0);
    out.set_message("ok");
    Ok(())
}

/// Start an RPC server listening on `host:port`, dispatching to `pot`.
///
/// This function blocks the current thread. It creates its own tokio runtime.
///
/// A request whose `pos` does not hold three values per entry of `atmnrs`,
/// or whose `box` does not hold 9 values, fails without calling the
/// callback.  A successful callback whose forces tensor is not 3 doubles
/// per atom on the CPU also fails the request.
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
                    user_data: pot_ref.pot_user_data,
                    potential: pot,
                };

                let potential_client = capnp_rpc::new_client::<potential::Client, _>(server_impl);

                let (reader, writer) =
                    tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();

                let network = twoparty::VatNetwork::new(
                    futures::io::BufReader::new(reader),
                    futures::io::BufWriter::new(writer),
                    rpc_twoparty_capnp::Side::Server,
                    Default::default(),
                );

                let rpc_system = RpcSystem::new(Box::new(network), Some(potential_client.client));

                tokio::task::spawn_local(rpc_system);
            }
        })
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static mut SHORT_FORCES: [f64; 6] = [0.0; 6];

    struct Probe {
        calls: AtomicUsize,
        short_forces: bool,
    }

    unsafe extern "C" fn probe_callback(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let probe = unsafe { &*(user_data as *const Probe) };
        probe.calls.fetch_add(1, Ordering::SeqCst);
        let n = unsafe { (*input).n_atoms() }.unwrap_or(0);
        let out = unsafe { &mut *output };
        out.energy = 1.0;
        out.forces = if probe.short_forces {
            // Shape [n - 1, 3] over a buffer large enough for [n, 3].
            unsafe {
                rgpot_tensor_cpu_f64_2d(
                    std::ptr::addr_of_mut!(SHORT_FORCES).cast(),
                    n as i64 - 1,
                    3,
                )
            }
        } else {
            crate::tensor::create_owned_f64_tensor(vec![0.5; 3 * n], vec![n as i64, 3])
        };
        rgpot_status_t::RGPOT_SUCCESS
    }

    unsafe extern "C" fn probe_hvp(
        _user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        direction: *const f64,
        product: *mut f64,
    ) -> rgpot_status_t {
        let dimension = unsafe { (*input).n_atoms() }
            .and_then(|n| n.checked_mul(3))
            .expect("test input");
        for i in 0..dimension {
            unsafe { *product.add(i) = 2.0 * *direction.add(i) };
        }
        rgpot_status_t::RGPOT_SUCCESS
    }

    fn call(probe: &Probe, pos: &[f64], atm: &[i32], cell: &[f64]) -> Result<Vec<f64>, String> {
        let client = capnp_rpc::new_client::<potential::Client, _>(PotentialServer {
            callback: probe_callback,
            user_data: probe as *const Probe as *mut c_void,
            potential: std::ptr::null(),
        });
        let mut request = client.calculate_request();
        {
            let mut fip = request.get().init_fip();
            let mut b = fip.reborrow().init_pos(pos.len() as u32);
            for (i, &v) in pos.iter().enumerate() {
                b.set(i as u32, v);
            }
            let mut b = fip.reborrow().init_atmnrs(atm.len() as u32);
            for (i, &v) in atm.iter().enumerate() {
                b.set(i as u32, v);
            }
            let mut b = fip.init_box(cell.len() as u32);
            for (i, &v) in cell.iter().enumerate() {
                b.set(i as u32, v);
            }
        }
        futures::executor::block_on(async {
            let response = request.send().promise.await.map_err(|e| e.to_string())?;
            let forces = response
                .get()
                .and_then(|r| r.get_result())
                .and_then(|r| r.get_forces())
                .map_err(|e| e.to_string())?;
            Ok((0..forces.len()).map(|i| forces.get(i)).collect())
        })
    }

    fn probe(short_forces: bool) -> Probe {
        Probe {
            calls: AtomicUsize::new(0),
            short_forces,
        }
    }

    const CELL: [f64; 9] = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

    #[test]
    fn analytic_hvp_is_served_through_the_rpc_method() {
        let probe = probe(false);
        let pot = unsafe {
            crate::c_api::potential::rgpot_potential_new(
                probe_callback,
                &probe as *const Probe as *mut c_void,
                None,
            )
        };
        assert!(!pot.is_null());
        unsafe { &*pot }.set_hessian_vector_callback(Some(probe_hvp));
        let client = capnp_rpc::new_client::<potential::Client, _>(PotentialServer {
            callback: probe_callback,
            user_data: &probe as *const Probe as *mut c_void,
            potential: pot,
        });
        let direction = [0.2, -0.3, 0.4];
        let mut request = client.calculate_hvp_request();
        {
            let mut root = request.get();
            let mut fip = root.reborrow().init_fip();
            let mut positions = fip.reborrow().init_pos(3);
            for (i, value) in [0.1, 0.2, 0.3].into_iter().enumerate() {
                positions.set(i as u32, value);
            }
            fip.reborrow().init_atmnrs(1).set(0, 1);
            let mut cell = fip.init_box(9);
            for (i, value) in CELL.into_iter().enumerate() {
                cell.set(i as u32, value);
            }
            let mut vector = root.init_direction(3);
            for (i, value) in direction.into_iter().enumerate() {
                vector.set(i as u32, value);
            }
        }
        let response = futures::executor::block_on(request.send().promise).expect("HVP response");
        let product = response
            .get()
            .and_then(|value| value.get_product())
            .expect("HVP product");
        for (i, value) in direction.into_iter().enumerate() {
            assert!((product.get(i as u32) - 2.0 * value).abs() <= f64::EPSILON);
        }
        unsafe { crate::c_api::potential::rgpot_potential_free(pot) };
    }

    #[test]
    fn well_formed_request_succeeds() {
        let p = probe(false);
        let forces = call(&p, &[0.0; 6], &[1, 1], &CELL).unwrap();
        assert_eq!(forces, vec![0.5; 6]);
        assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn short_positions_are_rejected_before_the_callback() {
        let p = probe(false);
        let err = call(&p, &[0.0; 3], &[1, 1], &CELL).unwrap_err();
        assert!(err.contains("positions"), "{err}");
        assert_eq!(p.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn short_box_is_rejected_before_the_callback() {
        let p = probe(false);
        let err = call(&p, &[0.0; 6], &[1, 1], &CELL[..6]).unwrap_err();
        assert!(err.contains("box"), "{err}");
        assert_eq!(p.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn mismatched_forces_shape_is_rejected() {
        let p = probe(true);
        let err = call(&p, &[0.0; 6], &[1, 1], &CELL).unwrap_err();
        assert!(err.contains("forces"), "{err}");
        assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn neighbours_are_served_through_the_rpc_method() {
        let p = probe(false);
        let client = capnp_rpc::new_client::<potential::Client, _>(PotentialServer {
            callback: probe_callback,
            user_data: &p as *const Probe as *mut c_void,
            potential: std::ptr::null(),
        });
        let pos = [0.2, 0.3, 0.4, 3.6, 0.5, 0.2, 1.0, 3.8, 0.7, 2.2, 2.1, 3.9];
        let box9 = [4.0, 0.2, 0.1, 0.3, 4.5, -0.2, 0.05, 0.15, 4.2];
        for full in [false, true] {
            let mut request = client.neighbours_request();
            {
                let mut root = request.get();
                let mut fip = root.reborrow().init_fip();
                let mut b_pos = fip.reborrow().init_pos(pos.len() as u32);
                for (i, &v) in pos.iter().enumerate() {
                    b_pos.set(i as u32, v);
                }
                let mut b_atm = fip.reborrow().init_atmnrs(4);
                for i in 0..4 {
                    b_atm.set(i, 1);
                }
                let mut b_box = fip.init_box(9);
                for (i, &v) in box9.iter().enumerate() {
                    b_box.set(i as u32, v);
                }
                let mut nreq = root.init_request();
                nreq.set_cutoff(3.2);
                nreq.set_skin(0.2);
                nreq.set_format(u8::from(full));
                nreq.set_cut(1);
                let mut pbc = nreq.reborrow().init_periodic(3);
                pbc.set(0, true);
                pbc.set(1, true);
                pbc.set(2, false);
                nreq.set_shifts(true);
                nreq.set_vectors(true);
                nreq.set_distances(true);
                nreq.set_mode(1);
            }
            let response =
                futures::executor::block_on(request.send().promise).expect("neighbours response");
            let list = response
                .get()
                .and_then(|r| r.get_neighbors())
                .expect("neighbours list");
            assert_eq!(list.get_status(), 0);
            let first = list.get_first().expect("first");
            let second = list.get_second().expect("second");
            let shift = list.get_shift().expect("shift");
            let vector = list.get_vector().expect("vector");
            let distance = list.get_distance().expect("distance");
            assert!(!first.is_empty());
            assert_eq!(first.len(), second.len());
            assert_eq!(shift.len(), first.len() * 3);
            assert_eq!(vector.len(), first.len() * 3);
            assert_eq!(distance.len(), first.len());
        }
    }

    #[test]
    fn calculate_includes_neighbours_when_force_input_requests_them() {
        let p = probe(false);
        let client = capnp_rpc::new_client::<potential::Client, _>(PotentialServer {
            callback: probe_callback,
            user_data: &p as *const Probe as *mut c_void,
            potential: std::ptr::null(),
        });
        let mut request = client.calculate_request();
        {
            let root = request.get();
            let mut fip = root.init_fip();
            let pos = [0.2, 0.3, 0.4, 3.6, 0.5, 0.2, 1.0, 3.8, 0.7, 2.2, 2.1, 3.9];
            let box9 = [4.0, 0.2, 0.1, 0.3, 4.5, -0.2, 0.05, 0.15, 4.2];
            let mut b_pos = fip.reborrow().init_pos(pos.len() as u32);
            for (i, &value) in pos.iter().enumerate() {
                b_pos.set(i as u32, value);
            }
            let mut b_atm = fip.reborrow().init_atmnrs(4);
            for i in 0..4 {
                b_atm.set(i, 1);
            }
            let mut b_box = fip.reborrow().init_box(9);
            for (i, &value) in box9.iter().enumerate() {
                b_box.set(i as u32, value);
            }
            let mut nreq = fip.init_neighbors();
            nreq.set_cutoff(3.2);
            nreq.set_skin(0.2);
            nreq.set_format(1);
            nreq.set_cut(1);
            let mut pbc = nreq.reborrow().init_periodic(3);
            pbc.set(0, true);
            pbc.set(1, true);
            pbc.set(2, false);
            nreq.set_shifts(true);
            nreq.set_vectors(true);
            nreq.set_distances(true);
            nreq.set_mode(1);
        }
        let response =
            futures::executor::block_on(request.send().promise).expect("calculate response");
        let result = response
            .get()
            .and_then(|r| r.get_result())
            .expect("calculate result");
        let list = result.get_neighbor_list().expect("neighbor list");
        assert_eq!(list.get_status(), 0);
        assert!(!list.get_first().expect("first").is_empty());
    }
}
