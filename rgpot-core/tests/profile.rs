use capnp::message::Builder;
use capnp::serialize;
use rgpot_core::profile::{
    decode_potential_result, encode_force_input, library_candidates, ProfileRequest,
};
use rgpot_core::Potentials_capnp::potential_result;

#[test]
fn candidates_are_prefix_parameterized() {
    let nwchem = library_candidates("nwchemc", Some("/opt/lib/libnwchemc.so"));
    let cpmd = library_candidates("cpmdc", None);

    assert_eq!(nwchem, ["/opt/lib/libnwchemc.so"]);
    assert!(cpmd.iter().any(|path| path == "libcpmdc.so"));
    assert!(!cpmd.iter().any(|path| path.contains("nwchem")));
}

#[test]
fn force_input_codec_preserves_geometry_and_units() {
    let positions = [0.0, 0.1, 0.2, 0.9, 1.0, 1.1];
    let atomic_numbers = [8, 1];
    let box_matrix = [12.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 14.0];
    let encoded = encode_force_input(&ProfileRequest {
        positions: &positions,
        atomic_numbers: &atomic_numbers,
        box_matrix: Some(&box_matrix),
        length_unit: "angstrom",
        energy_unit: "eV",
    })
    .expect("encode ForceInput");

    let mut bytes = encoded.as_slice();
    let message =
        serialize::read_message_from_flat_slice(&mut bytes, capnp::message::ReaderOptions::new())
            .expect("read ForceInput");
    let input = message
        .get_root::<rgpot_core::Potentials_capnp::force_input::Reader>()
        .expect("ForceInput root");

    let got_positions = input.get_pos().expect("positions");
    let got_atomic_numbers = input.get_atmnrs().expect("atomic numbers");
    let got_box = input.get_box().expect("box");
    assert_eq!(
        (0..got_positions.len())
            .map(|i| got_positions.get(i))
            .collect::<Vec<_>>(),
        positions
    );
    assert_eq!(
        (0..got_atomic_numbers.len())
            .map(|i| got_atomic_numbers.get(i))
            .collect::<Vec<_>>(),
        atomic_numbers
    );
    assert_eq!(
        (0..got_box.len())
            .map(|i| got_box.get(i))
            .collect::<Vec<_>>(),
        box_matrix
    );
    assert_eq!(
        input.get_length_unit().unwrap().to_str().unwrap(),
        "angstrom"
    );
    assert_eq!(input.get_energy_unit().unwrap().to_str().unwrap(), "eV");
}

#[test]
fn force_input_codec_omits_cell_for_molecular_geometry() {
    let positions = [0.0, 0.0, 0.0, 0.9, 0.0, 0.0];
    let atomic_numbers = [8, 1];
    let encoded = encode_force_input(&ProfileRequest {
        positions: &positions,
        atomic_numbers: &atomic_numbers,
        box_matrix: None,
        length_unit: "angstrom",
        energy_unit: "eV",
    })
    .expect("encode molecular ForceInput");

    let mut bytes = encoded.as_slice();
    let message =
        serialize::read_message_from_flat_slice(&mut bytes, capnp::message::ReaderOptions::new())
            .expect("read ForceInput");
    let input = message
        .get_root::<rgpot_core::Potentials_capnp::force_input::Reader>()
        .expect("ForceInput root");

    assert_eq!(input.get_box().expect("box list").len(), 0);
}

#[test]
fn potential_result_codec_preserves_one_fused_evaluation() {
    let mut message = Builder::new_default();
    {
        let mut result = message.init_root::<potential_result::Builder>();
        result.set_energy(-12.75);
        let mut forces = result.reborrow().init_forces(6);
        for (i, value) in [1.0, 2.0, 3.0, -1.0, -2.0, -3.0].into_iter().enumerate() {
            forces.set(i as u32, value);
        }
    }
    let encoded = serialize::write_message_to_words(&message);
    let evaluation = decode_potential_result(&encoded, 6).expect("decode PotentialResult");

    assert_eq!(evaluation.energy, -12.75);
    assert_eq!(evaluation.forces, [1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
}

#[cfg(unix)]
mod md_session_capacity_tests {
    use std::ffi::{c_char, c_void, CString};
    use std::path::{Path, PathBuf};

    use rgpot_core::md_session::{CAbiMdSession, MdSession, MdSessionOps};

    fn build_capacity_engine() -> PathBuf {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/md_session_capacity_engine.c");
        let suffix = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let out = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("libmd_session_capacity_engine.{suffix}"));
        let target = env!("RGPOT_BUILD_TARGET");
        let compiler = cc::Build::new()
            .target(target)
            .host(target)
            .opt_level(0)
            .try_get_compiler()
            .expect("C compiler for capacity fixture");
        let mut command = compiler.to_command();
        if cfg!(target_os = "macos") {
            command.arg("-dynamiclib");
        } else {
            command.args(["-shared", "-fPIC"]);
        }
        command.arg("-o").arg(&out).arg(&source);
        let status = command.status().expect("compile capacity fixture");
        assert!(
            status.success(),
            "capacity fixture compile failed: {status}"
        );
        out
    }

    #[test]
    fn open_engine_passes_capacity_and_handles_older_newer_and_oversized_engines() {
        let lib_path = build_capacity_engine();
        let host_size = std::mem::size_of::<MdSessionOps>();
        let required_size = std::mem::offset_of!(MdSessionOps, destroy)
            + std::mem::size_of::<unsafe extern "C" fn(*mut c_void)>();

        // Current engine: sees host capacity and fills all callbacks.
        let mut current = unsafe { CAbiMdSession::open_engine(&lib_path, &[], Some("current")) }
            .expect("open current engine");
        let masses = current.masses().unwrap();
        assert_eq!(masses[0], host_size as f64);
        assert_eq!(masses[1], host_size as f64);
        let cell = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        current.set_periodic_box(&cell).unwrap();
        assert_eq!(current.periodic_box().unwrap(), cell);

        // Older engine: fills only up to destroy, optional ops report unsupported.
        let mut older = unsafe { CAbiMdSession::open_engine(&lib_path, &[], Some("older")) }
            .expect("open older engine");
        let masses = older.masses().unwrap();
        assert_eq!(masses[0], host_size as f64);
        assert_eq!(masses[1], required_size as f64);
        assert!(older.set_periodic_box(&cell).is_err());

        // Newer engine: larger vtable writes only min(capacity, newer_size) = host_size.
        let mut newer = unsafe { CAbiMdSession::open_engine(&lib_path, &[], Some("newer")) }
            .expect("open newer engine");
        let masses = newer.masses().unwrap();
        assert_eq!(masses[0], host_size as f64);
        assert_eq!(masses[1], host_size as f64);
        newer.set_periodic_box(&cell).unwrap();

        // Oversized reported struct_size is clamped by from_ops without failing.
        let mut oversized =
            unsafe { CAbiMdSession::open_engine(&lib_path, &[], Some("oversized")) }
                .expect("open oversized engine");
        oversized.set_periodic_box(&cell).unwrap();
    }

    #[test]
    fn direct_engine_calls_respect_smaller_and_larger_host_capacities() {
        type CreateFn = unsafe extern "C" fn(
            spec: *const c_void,
            spec_len: usize,
            options: *const c_char,
            ops: *mut MdSessionOps,
            ctx: *mut *mut c_void,
            n_atoms: *mut usize,
            err: *mut c_char,
            err_len: usize,
        ) -> i32;

        let lib_path = build_capacity_engine();
        let library = unsafe { libloading::Library::new(&lib_path) }.expect("load fixture");
        let create = unsafe { library.get::<CreateFn>(b"rgpot_md_session_engine_create\0") }
            .expect("resolve create");

        let required_size = std::mem::offset_of!(MdSessionOps, destroy)
            + std::mem::size_of::<unsafe extern "C" fn(*mut c_void)>();
        let mid_size = std::mem::offset_of!(MdSessionOps, set_periodic_box)
            + std::mem::size_of::<
                Option<unsafe extern "C" fn(*mut c_void, *const f64, *mut c_char, usize) -> i32>,
            >()
            + 1;

        let newer_opt = CString::new("newer").unwrap();
        let current_opt = CString::new("current").unwrap();

        #[repr(C)]
        struct AlignedOpsBuffer {
            _align: [MdSessionOps; 0],
            bytes: [u8; std::mem::size_of::<MdSessionOps>() + 64],
        }

        // Capacity smaller than destroy (0 and required_size - 1) is refused cleanly.
        for tiny_cap in [0usize, required_size - 1] {
            let mut buf = AlignedOpsBuffer {
                _align: [],
                bytes: [0xA5u8; std::mem::size_of::<MdSessionOps>() + 64],
            };
            unsafe {
                buf.bytes.as_mut_ptr().cast::<usize>().write(tiny_cap);
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            let mut n_atoms: usize = 0;
            let mut err = [0 as c_char; 512];
            let status = unsafe {
                create(
                    std::ptr::null(),
                    0,
                    newer_opt.as_ptr(),
                    buf.bytes.as_mut_ptr().cast::<MdSessionOps>(),
                    &mut ctx,
                    &mut n_atoms,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            assert_ne!(status, 0);
            assert!(ctx.is_null());
            assert!(buf.bytes[std::mem::size_of::<usize>()..]
                .iter()
                .all(|&b| b == 0xA5));
        }

        // Smaller host capacity against "newer" engine: guard bytes stay intact.
        for cap in [required_size, mid_size] {
            let mut buf = AlignedOpsBuffer {
                _align: [],
                bytes: [0xA5u8; std::mem::size_of::<MdSessionOps>() + 64],
            };
            buf.bytes[..cap].fill(0);
            unsafe {
                buf.bytes.as_mut_ptr().cast::<usize>().write(cap);
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            let mut n_atoms: usize = 0;
            let mut err = [0 as c_char; 512];
            let status = unsafe {
                create(
                    std::ptr::null(),
                    0,
                    newer_opt.as_ptr(),
                    buf.bytes.as_mut_ptr().cast::<MdSessionOps>(),
                    &mut ctx,
                    &mut n_atoms,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            assert_eq!(status, 0);
            let reported = unsafe { buf.bytes.as_ptr().cast::<usize>().read() };
            assert_eq!(reported, cap);
            assert!(buf.bytes[cap..].iter().all(|&b| b == 0xA5));

            let mut ops = std::mem::MaybeUninit::<MdSessionOps>::zeroed();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf.bytes.as_ptr(),
                    ops.as_mut_ptr().cast::<u8>(),
                    cap,
                );
            }
            let session = unsafe { CAbiMdSession::from_engine_ops(ctx, ops, n_atoms) }
                .expect("adopt truncated ops");
            let masses = session.masses().unwrap();
            assert_eq!(masses[0], cap as f64);
            assert_eq!(masses[1], cap as f64);
        }

        // Larger host capacity against "current" and "newer" engines.
        #[repr(C)]
        struct LargerHostOps {
            base: MdSessionOps,
            future_cb: Option<unsafe extern "C" fn(*mut c_void)>,
            future_tag: u64,
            guard: [u8; 32],
        }

        {
            let mut host = std::mem::MaybeUninit::<LargerHostOps>::uninit();
            let base_len = std::mem::size_of::<MdSessionOps>();
            unsafe {
                host.as_mut_ptr().write_bytes(0xA5, 1);
                host.as_mut_ptr().cast::<u8>().write_bytes(0, base_len);
                std::ptr::addr_of_mut!((*host.as_mut_ptr()).base.struct_size)
                    .write(std::mem::size_of::<LargerHostOps>());
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            let mut n_atoms: usize = 0;
            let mut err = [0 as c_char; 512];
            let status = unsafe {
                create(
                    std::ptr::null(),
                    0,
                    current_opt.as_ptr(),
                    std::ptr::addr_of_mut!((*host.as_mut_ptr()).base),
                    &mut ctx,
                    &mut n_atoms,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            assert_eq!(status, 0);
            let tail = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<u8>().add(base_len),
                    std::mem::size_of::<LargerHostOps>() - base_len,
                )
            };
            assert!(tail.iter().all(|&b| b == 0xA5));
            let base = unsafe { std::ptr::addr_of!((*host.as_ptr()).base).read() };
            assert_eq!(base.struct_size, base_len);
            let mut session = unsafe {
                CAbiMdSession::from_engine_ops(ctx, std::mem::MaybeUninit::new(base), n_atoms)
            }
            .expect("larger host works with current engine");
            let cell = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
            session.set_periodic_box(&cell).unwrap();
        }

        {
            let mut host = std::mem::MaybeUninit::<LargerHostOps>::uninit();
            let newer_end =
                std::mem::offset_of!(LargerHostOps, future_tag) + std::mem::size_of::<u64>();
            unsafe {
                host.as_mut_ptr().write_bytes(0xA5, 1);
                host.as_mut_ptr().cast::<u8>().write_bytes(0, newer_end);
                std::ptr::addr_of_mut!((*host.as_mut_ptr()).base.struct_size)
                    .write(std::mem::size_of::<LargerHostOps>());
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            let mut n_atoms: usize = 0;
            let mut err = [0 as c_char; 512];
            let status = unsafe {
                create(
                    std::ptr::null(),
                    0,
                    newer_opt.as_ptr(),
                    std::ptr::addr_of_mut!((*host.as_mut_ptr()).base),
                    &mut ctx,
                    &mut n_atoms,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            assert_eq!(status, 0);
            let host = unsafe { host.assume_init() };
            assert_eq!(host.base.struct_size, newer_end);
            assert!(host.future_cb.is_some());
            assert_eq!(host.future_tag, 0x1234_5678_9abc_def0);
            assert!(host.guard.iter().all(|&b| b == 0xA5));
            let session = unsafe {
                CAbiMdSession::from_engine_ops(ctx, std::mem::MaybeUninit::new(host.base), n_atoms)
            }
            .expect("larger host works with newer engine");
            assert_eq!(session.dof(), 6);
        }
    }
}
