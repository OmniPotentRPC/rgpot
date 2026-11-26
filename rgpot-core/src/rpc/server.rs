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

use crate::hvp::rgpot_hvp_scheme_t;
use crate::potential::{rgpot_potential_t, PotentialCallback, PotentialStressCallback};
use crate::rpc::schema::potential;
use crate::status::rgpot_status_t;
use crate::tensor::{
    copy_f64_tensor, rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3,
    rgpot_tensor_cpu_i32_1d, rgpot_tensor_free,
};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t, rgpot_stress_out_t};

/// RPC server state wrapping a potential callback.
struct PotentialServer {
    callback: PotentialCallback,
    /// Set when the potential computes stress. A request for stress without
    /// it is refused instead of answered with zeros.
    stress_callback: Option<PotentialStressCallback>,
    user_data: *mut c_void,
    potential: *const rgpot_potential_t,
}

// Safety: the user_data pointer is only accessed through the callback
// which is expected to be thread-safe by the caller.
unsafe impl Send for PotentialServer {}
unsafe impl Sync for PotentialServer {}

/// What one evaluation produced, in the units of the callback.
struct Answer {
    energy: f64,
    forces: Vec<f64>,
    stress: Option<[f64; 9]>,
}

impl Answer {
    fn write(self, mut result: crate::rpc::schema::potential_result::Builder<'_>) {
        result.set_energy(self.energy);
        let mut forces_builder = result.reborrow().init_forces(self.forces.len() as u32);
        for (i, &f) in self.forces.iter().enumerate() {
            forces_builder.set(i as u32, f);
        }
        if let Some(stress) = self.stress {
            let mut stress_builder = result.init_stress(9);
            for (i, &value) in stress.iter().enumerate() {
                stress_builder.set(i as u32, value);
            }
        }
    }
}

impl PotentialServer {
    /// Run the callback on one request.
    ///
    /// A request for stress is served only by a potential that computes it,
    /// and fails when the callback leaves `has_stress` clear: a stress
    /// request never succeeds without a stress.
    fn evaluate(
        &self,
        fip: crate::rpc::schema::force_input::Reader<'_>,
        require_stress: bool,
    ) -> Result<Answer, CapnpError> {
        let want_stress = require_stress || fip.get_want_stress();
        if want_stress && self.stress_callback.is_none() {
            return Err(CapnpError::failed(
                "the selected potential does not provide stress".to_string(),
            ));
        }

        let positions = fip.get_pos()?;
        let atmnrs = fip.get_atmnrs()?;
        let box_data = fip.get_box()?;

        let n_atoms = atmnrs.len() as usize;
        if positions.len() as usize != 3 * n_atoms {
            return Err(CapnpError::failed(format!(
                "positions: expected {} values for {n_atoms} atoms, got {}",
                3 * n_atoms,
                positions.len()
            )));
        }
        if box_data.len() != 9 {
            return Err(CapnpError::failed(format!(
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
        let mut output = rgpot_force_out_t::default();
        let mut stress = rgpot_stress_out_t::default();

        let status = match self.stress_callback.filter(|_| want_stress) {
            Some(callback) => unsafe { callback(self.user_data, &input, &mut output, &mut stress) },
            None => unsafe { (self.callback)(self.user_data, &input, &mut output) },
        };

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
            return Err(CapnpError::failed(
                "potential callback returned an error".to_string(),
            ));
        }
        let forces = forces_copy.map_err(CapnpError::failed)?;
        if want_stress && stress.has_stress == 0 {
            return Err(CapnpError::failed(
                "the selected potential returned no stress".to_string(),
            ));
        }

        Ok(Answer {
            energy: output.energy,
            forces,
            stress: (want_stress && stress.has_stress != 0).then_some(stress.stress),
        })
    }
}

impl potential::Server for PotentialServer {
    fn get_capabilities(
        &mut self,
        _params: potential::GetCapabilitiesParams,
        mut results: potential::GetCapabilitiesResults,
    ) -> capnp::capability::Promise<(), CapnpError> {
        let analytic_hvp =
            !self.potential.is_null() && unsafe { &*self.potential }.has_hessian_vector_callback();
        crate::compat::describe_server_with(
            results.get().init_capabilities(),
            analytic_hvp,
            self.stress_callback.is_some(),
        );
        capnp::capability::Promise::ok(())
    }

    fn calculate(
        &mut self,
        params: potential::CalculateParams,
        mut results: potential::CalculateResults,
    ) -> capnp::capability::Promise<(), CapnpError> {
        let fip = pry!(pry!(params.get()).get_fip());
        let answer = pry!(self.evaluate(fip, false));
        answer.write(results.get().init_result());
        capnp::capability::Promise::ok(())
    }

    fn calculate_stress(
        &mut self,
        params: potential::CalculateStressParams,
        mut results: potential::CalculateStressResults,
    ) -> capnp::capability::Promise<(), CapnpError> {
        let fip = pry!(pry!(params.get()).get_fip());
        let answer = pry!(self.evaluate(fip, true));
        answer.write(results.get().init_result());
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
                    stress_callback: pot_ref.stress_callback(),
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
        stress_calls: AtomicUsize,
        short_forces: bool,
        /// The stress callback runs but leaves `has_stress` clear, as a
        /// kernel does for a cell it cannot measure.
        withhold_stress: bool,
    }

    fn probe_forces(probe: &Probe, input: *const rgpot_force_input_t, out: &mut rgpot_force_out_t) {
        let n = unsafe { (*input).n_atoms() }.unwrap_or(0);
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
    }

    unsafe extern "C" fn probe_callback(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let probe = unsafe { &*(user_data as *const Probe) };
        probe.calls.fetch_add(1, Ordering::SeqCst);
        probe_forces(probe, input, unsafe { &mut *output });
        rgpot_status_t::RGPOT_SUCCESS
    }

    unsafe extern "C" fn probe_stress_callback(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
        stress: *mut rgpot_stress_out_t,
    ) -> rgpot_status_t {
        let probe = unsafe { &*(user_data as *const Probe) };
        probe.stress_calls.fetch_add(1, Ordering::SeqCst);
        probe_forces(probe, input, unsafe { &mut *output });
        let block = unsafe { &mut *stress };
        if !probe.withhold_stress {
            block.stress = [0.125; 9];
            block.has_stress = 1;
        }
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

    fn call(
        probe: &Probe,
        pos: &[f64],
        atm: &[i32],
        cell: &[f64],
        compute_stress: bool,
        serves_stress: bool,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let client = capnp_rpc::new_client::<potential::Client, _>(PotentialServer {
            callback: probe_callback,
            stress_callback: serves_stress.then_some(probe_stress_callback as _),
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
            fip.set_want_stress(compute_stress);
            let mut b = fip.init_box(cell.len() as u32);
            for (i, &v) in cell.iter().enumerate() {
                b.set(i as u32, v);
            }
        }
        futures::executor::block_on(async {
            let response = request.send().promise.await.map_err(|e| e.to_string())?;
            let result = response
                .get()
                .and_then(|r| r.get_result())
                .map_err(|e| e.to_string())?;
            let forces = result.get_forces().map_err(|e| e.to_string())?;
            let stress = result.get_stress().map_err(|e| e.to_string())?;
            Ok((
                (0..forces.len()).map(|i| forces.get(i)).collect(),
                (0..stress.len()).map(|i| stress.get(i)).collect(),
            ))
        })
    }

    fn probe(short_forces: bool) -> Probe {
        Probe {
            calls: AtomicUsize::new(0),
            stress_calls: AtomicUsize::new(0),
            short_forces,
            withhold_stress: false,
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
            stress_callback: None,
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
        let (forces, stress) = call(&p, &[0.0; 6], &[1, 1], &CELL, false, true).unwrap();
        assert_eq!(forces, vec![0.5; 6]);
        assert!(stress.is_empty(), "a force-only request carries no stress");
        assert_eq!(p.calls.load(Ordering::SeqCst), 1);
        assert_eq!(p.stress_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn stress_request_and_result_round_trip() {
        let p = probe(false);
        let (forces, stress) = call(&p, &[0.0; 6], &[1, 1], &CELL, true, true).unwrap();
        assert_eq!(forces, vec![0.5; 6]);
        assert_eq!(stress, vec![0.125; 9]);
        assert_eq!(p.calls.load(Ordering::SeqCst), 0);
        assert_eq!(p.stress_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stress_request_to_a_potential_without_stress_is_refused() {
        let p = probe(false);
        let err = call(&p, &[0.0; 6], &[1, 1], &CELL, true, false).unwrap_err();
        assert!(err.contains("does not provide stress"), "{err}");
        assert_eq!(p.calls.load(Ordering::SeqCst), 0);
        assert_eq!(p.stress_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn stress_request_the_callback_cannot_answer_fails_like_the_cpp_server() {
        let p = Probe {
            withhold_stress: true,
            ..probe(false)
        };
        let err = call(&p, &[0.0; 6], &[1, 1], &CELL, true, true).unwrap_err();
        assert!(err.contains("returned no stress"), "{err}");
        assert_eq!(p.stress_calls.load(Ordering::SeqCst), 1);
        let (forces, stress) = call(&p, &[0.0; 6], &[1, 1], &CELL, false, true).unwrap();
        assert_eq!(forces.len(), 6);
        assert!(stress.is_empty(), "a force-only request carries no stress");
    }

    #[test]
    fn short_positions_are_rejected_before_the_callback() {
        let p = probe(false);
        let err = call(&p, &[0.0; 3], &[1, 1], &CELL, false, true).unwrap_err();
        assert!(err.contains("positions"), "{err}");
        assert_eq!(p.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn short_box_is_rejected_before_the_callback() {
        let p = probe(false);
        let err = call(&p, &[0.0; 6], &[1, 1], &CELL[..6], false, true).unwrap_err();
        assert!(err.contains("box"), "{err}");
        assert_eq!(p.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn mismatched_forces_shape_is_rejected() {
        let p = probe(true);
        let err = call(&p, &[0.0; 6], &[1, 1], &CELL, false, true).unwrap_err();
        assert!(err.contains("forces"), "{err}");
        assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    }
}
