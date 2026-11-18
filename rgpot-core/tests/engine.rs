#![cfg(unix)]

use std::ffi::c_int;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rgpot_core::engine::{
    CoupledWellConfig, EngineError, EnginePotential, EngineRequest, PotlibcConfig,
    RgpotEngineAbiStamp, COUPLED_WELL_CONFIG_MIN_SIZE, ENGINE_ABI_LAYOUT_REVISION,
    ENGINE_ABI_MINOR, ENGINE_ABI_VERSION, ENGINE_ERR_UNSUPPORTED, POTLIBC_CONFIG_MAGIC,
    POTLIBC_CONFIG_MIN_SIZE, POTLIBC_CONFIG_VERSION, POTLIBC_SURFACE_ID_CAPACITY,
};
use rgpot_core::hvp::rgpot_hvp_scheme_t;
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_data, rgpot_tensor_free,
};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

const SPRING: f64 = 2.5;
const CELL: [f64; 9] = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

fn build(name: &str, defines: &[&str]) -> PathBuf {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/harmonic_engine.c");
    let include = Path::new(env!("CARGO_MANIFEST_DIR")).join("../CppCore");
    let suffix = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("lib{name}.{suffix}"));
    let target = env!("RGPOT_BUILD_TARGET");
    let compiler = match cc::Build::new()
        .target(target)
        .host(target)
        .opt_level(0)
        .try_get_compiler()
    {
        Ok(compiler) => compiler,
        Err(why) => panic!(
            "cannot compile C fixture '{name}': no C compiler available for target {target} ({why})"
        ),
    };
    let mut command = compiler.to_command();
    if cfg!(target_os = "macos") {
        command.arg("-dynamiclib");
    } else {
        command.args(["-shared", "-fPIC"]);
    }
    command
        .arg("-I")
        .arg(include)
        .arg("-o")
        .arg(&out)
        .arg(&source);
    for define in defines {
        command.arg(format!("-D{define}"));
    }
    let status = match command.status() {
        Ok(status) => status,
        Err(why) if why.kind() == std::io::ErrorKind::NotFound => panic!(
            "cannot compile C fixture '{name}': compiler {:?} not found ({why})",
            compiler.path()
        ),
        Err(why) => panic!("spawn C compiler {:?}: {why}", compiler.path()),
    };
    assert!(status.success(), "the fixture did not compile: {status}");
    out
}

/// The engine is C++ and uses the C++ runtime, so it is built with the C++ driver.
/// Built once: every test below loads the same file.
fn build_coupled_well() -> &'static Path {
    static LIBRARY: OnceLock<PathBuf> = OnceLock::new();
    LIBRARY.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let source = root.join("engines/coupled_well_engine.cpp");
        let suffix = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let out =
            Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("libcoupled_well.{suffix}"));
        let target = env!("RGPOT_BUILD_TARGET");
        let compiler = match cc::Build::new()
            .cpp(true)
            .target(target)
            .host(target)
            .opt_level(0)
            .std("c++17")
            .try_get_compiler()
        {
            Ok(compiler) => compiler,
            Err(why) => panic!(
                "cannot compile the coupled-well engine: no C++ compiler available for target {target} ({why})"
            ),
        };
        let mut command = compiler.to_command();
        if cfg!(target_os = "macos") {
            command.arg("-dynamiclib");
        } else {
            command.args(["-shared", "-fPIC"]);
        }
        let status = command
            .arg("-I")
            .arg(root.join("CppCore"))
            .arg("-I")
            .arg(root.join("include"))
            .arg("-o")
            .arg(&out)
            .arg(&source)
            .status()
            .unwrap_or_else(|why| panic!("run the C++ compiler {:?}: {why}", compiler.path()));
        assert!(status.success(), "the coupled-well engine did not compile: {status}");
        out
    })
}

/// Rounding bar of a sum of `terms` products: each costs one rounding of its own size.
fn within_roundoff(got: f64, want: f64, terms: usize) -> bool {
    (got - want).abs() <= terms as f64 * f64::EPSILON * got.abs().max(want.abs())
}

fn all_within_roundoff(got: &[f64], want: &[f64]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| within_roundoff(*g, *w, 1))
}

fn config() -> [u8; 8] {
    SPRING.to_ne_bytes()
}

fn load(path: &Path) -> EnginePotential {
    unsafe { EnginePotential::load(path, &config()) }.expect("load the harmonic engine")
}

fn created(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_create_count\0") }.expect("create count symbol");
    unsafe { count() }
}

fn destroyed(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_destroy_count\0") }.expect("count symbol");
    unsafe { count() }
}

fn unsupported_code(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let value: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_unsupported_code\0") }.expect("code symbol");
    unsafe { value() }
}

#[test]
fn force_is_the_harmonic_energy_and_minus_the_gradient() {
    let path = build("harmonic", &[]);
    let mut engine = load(&path);
    let positions = [0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let answer = engine
        .force(&EngineRequest {
            positions: &positions,
            atomic_numbers: &[1, 8],
            box_matrix: &CELL,
        })
        .expect("force");
    let sum: f64 = positions.iter().map(|x| x * x).sum();
    assert!(within_roundoff(
        answer.energy,
        0.5 * SPRING * sum,
        3 * positions.len()
    ));
    let expected: Vec<f64> = positions.iter().map(|x| -SPRING * x).collect();
    assert!(all_within_roundoff(&answer.forces, &expected));
}

#[test]
fn batch_is_the_same_answer_as_one_call_per_system() {
    let path = build("harmonic_batch", &[]);
    let mut engine = load(&path);
    let numbers = [1, 8];
    let first = [0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let second = [1.5, 1.0, -2.0, 0.5, -3.0, 0.75];
    let mut positions = first.to_vec();
    positions.extend_from_slice(&second);
    let mut boxes = CELL.to_vec();
    boxes.extend_from_slice(&CELL);
    let batch = engine
        .force_batch(&numbers, &positions, &boxes)
        .expect("batch");
    assert_eq!(batch.len(), 2);
    for (system, one) in [first, second].iter().enumerate() {
        let single = engine
            .force(&EngineRequest {
                positions: one,
                atomic_numbers: &numbers,
                box_matrix: &CELL,
            })
            .expect("single");
        assert!(within_roundoff(
            batch[system].energy,
            single.energy,
            3 * one.len()
        ));
        assert!(all_within_roundoff(&batch[system].forces, &single.forces));
    }
}

#[test]
fn batch_without_the_symbol_is_unsupported() {
    let path = build("harmonic_nobatch", &["NO_BATCH"]);
    let mut engine = load(&path);
    let error = engine
        .force_batch(&[1], &[0.0, 0.0, 0.0], &CELL)
        .unwrap_err();
    assert!(matches!(
        error,
        EngineError::Unsupported {
            operation: "rgpot_engine_force_batch",
            ..
        }
    ));
}

#[test]
fn create_failure_carries_the_engine_message() {
    let path = build("harmonic_create", &[]);
    let error = unsafe { EnginePotential::load(&path, &[1, 2, 3]) }
        .err()
        .expect("create must fail");
    match error {
        EngineError::Create(message) => assert!(message.contains("config is 3 bytes"), "{message}"),
        other => panic!("expected a create error, got {other}"),
    }
}

#[test]
fn positive_code_is_recoverable_and_negative_is_fatal() {
    let path = build("harmonic_codes", &[]);
    let mut engine = load(&path);
    let positions = [0.0; 6];
    let recoverable = engine
        .force(&EngineRequest {
            positions: &positions,
            atomic_numbers: &[1, 0],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert_eq!(
        recoverable,
        EngineError::Recoverable {
            code: 1,
            message: "atom 1 has atomic number 0".into()
        }
    );
    let fatal = engine
        .force(&EngineRequest {
            positions: &positions,
            atomic_numbers: &[-1, 1],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert_eq!(
        fatal,
        EngineError::Fatal {
            code: -1,
            message: "atom 0 has a negative atomic number".into()
        }
    );
    let repeated = engine
        .force(&EngineRequest {
            positions: &positions,
            atomic_numbers: &[1, 1],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert_eq!(repeated, fatal);
}

#[test]
fn a_failure_without_last_error_names_the_code() {
    let path = build("harmonic_nolast", &["NO_LAST_ERROR"]);
    let mut engine = load(&path);
    let error = engine
        .force(&EngineRequest {
            positions: &[0.0; 3],
            atomic_numbers: &[0],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert_eq!(
        error,
        EngineError::Recoverable {
            code: 1,
            message: "no rgpot_engine_last_error; code 1".into()
        }
    );
}

#[test]
fn an_abi_major_other_than_the_loader_is_refused() {
    let path = build("harmonic_abi2", &["FIXTURE_ABI=2"]);
    let error = unsafe { EnginePotential::load(&path, &config()) }
        .err()
        .expect("ABI must be refused");
    assert_eq!(
        error,
        EngineError::Abi {
            expected: ENGINE_ABI_VERSION,
            found: 2
        }
    );
}

#[test]
fn an_incompatible_abi_stamp_is_refused() {
    let path = build("harmonic_layout2", &["FIXTURE_LAYOUT=2"]);
    let error = unsafe { EnginePotential::load(&path, &config()) }
        .err()
        .expect("layout revision must be refused");
    assert!(matches!(error, EngineError::Incompatible { .. }), "{error}");
}

#[test]
fn abi_minor_rule_accepts_unstamped_and_minors_0_2_3_and_rejects_minor_4() {
    let request = EngineRequest {
        positions: &[1.0, 0.0, 0.0],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };

    // Unstamped engine explicitly means minor 0, layout 1: accepted under 0 <= 3.
    let unstamped = build(
        "harmonic_minor0_unstamped",
        &["NO_STAMP", "NO_CAPS", "NO_HVP"],
    );
    let mut engine = load(&unstamped);
    let eval = engine.force(&request).expect("unstamped minor 0 force");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING, 3));
    assert!(matches!(
        engine.hvp(&request, &[1.0, 0.0, 0.0]),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        })
    ));

    // Unstamped engine exporting HVP: minor 0 is accepted and symbol presence discovers HVP.
    let unstamped_hvp = build("harmonic_minor0_unstamped_hvp", &["NO_STAMP", "NO_CAPS"]);
    let mut engine = load(&unstamped_hvp);
    let product = engine
        .hvp(&request, &[1.0, -2.0, 0.5])
        .expect("symbol-presence HVP on unstamped engine");
    assert!(all_within_roundoff(
        &product,
        &[SPRING, -2.0 * SPRING, 0.5 * SPRING]
    ));

    // Stamped minor 0: accepted under 0 <= 3.
    let minor0 = build(
        "harmonic_minor0_stamped",
        &["FIXTURE_MINOR=0", "NO_CAPS", "NO_HVP"],
    );
    let mut engine = load(&minor0);
    let eval = engine.force(&request).expect("stamped minor 0 force");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING, 3));

    // Stamped minor 2 without minor-3 HVP symbol: accepted under 2 <= 3; HVP is Unsupported.
    let minor2 = build(
        "harmonic_minor2_nohvp",
        &["FIXTURE_MINOR=2", "CAPS_SIZE_BEFORE_HVP", "NO_HVP"],
    );
    let mut engine = load(&minor2);
    let eval = engine.force(&request).expect("stamped minor 2 force");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING, 3));
    assert!(!engine.caps().expect("minor 2 caps").analytic_hvp);
    assert!(matches!(
        engine.hvp(&request, &[1.0, 0.0, 0.0]),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        })
    ));
    let handle = load(&minor2)
        .into_potential()
        .expect("minor 2 potential handle");
    assert!(!handle.as_handle().has_hessian_vector_callback());

    // Stamped minor 2 exporting HVP: accepted under 2 <= 3; symbol presence discovers HVP.
    let minor2_hvp = build(
        "harmonic_minor2_hvp",
        &["FIXTURE_MINOR=2", "CAPS_SIZE_BEFORE_HVP"],
    );
    let mut engine = load(&minor2_hvp);
    let product = engine
        .hvp(&request, &[1.0, 2.0, -0.5])
        .expect("minor 2 with HVP symbol");
    assert!(all_within_roundoff(
        &product,
        &[SPRING, 2.0 * SPRING, -0.5 * SPRING]
    ));

    // Stamped minor 3: accepted under 3 <= 3.
    let minor3 = build("harmonic_minor3", &["FIXTURE_MINOR=3"]);
    let mut engine = load(&minor3);
    let eval = engine.force(&request).expect("stamped minor 3 force");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING, 3));
    let product = engine
        .hvp(&request, &[2.0, 0.0, -1.0])
        .expect("stamped minor 3 hvp");
    assert!(all_within_roundoff(&product, &[2.0 * SPRING, 0.0, -SPRING]));

    // Stamped minor 4: rejected because 4 > ENGINE_ABI_MINOR (3).
    let minor4 = build("harmonic_minor4", &["FIXTURE_MINOR=4"]);
    let before = created(&minor4);
    let error = unsafe { EnginePotential::load(&minor4, &config()) }
        .err()
        .expect("minor 4 must be rejected by minor 3 loader");
    assert_eq!(
        error,
        EngineError::Incompatible {
            host: RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: ENGINE_ABI_MINOR,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            },
            engine: Some(RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: 4,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            }),
        }
    );
    assert_eq!(created(&minor4), before);
}

#[test]
fn a_missing_required_symbol_is_named() {
    let path = build("harmonic_noforce", &["NO_FORCE", "NO_BATCH"]);
    let error = unsafe { EnginePotential::load(&path, &config()) }
        .err()
        .expect("force is required");
    assert_eq!(error, EngineError::Missing("rgpot_engine_force".into()));

    let no_available = build("harmonic_noavailable", &["NO_AVAILABLE"]);
    let error = unsafe { EnginePotential::load(&no_available, &config()) }
        .err()
        .expect("available is required");
    assert_eq!(error, EngineError::Missing("rgpot_engine_available".into()));
}

#[test]
fn an_unavailable_engine_is_refused_before_create() {
    let path = build("harmonic_unavailable", &["FIXTURE_AVAILABLE=0"]);
    let before = created(&path);
    let error = unsafe { EnginePotential::load(&path, &config()) }
        .err()
        .expect("unavailable engine must be refused");
    assert_eq!(error, EngineError::Unavailable);
    assert_eq!(created(&path), before);
}

#[test]
fn a_request_that_does_not_fit_is_refused_before_the_engine() {
    let path = build("harmonic_shape", &[]);
    let mut engine = load(&path);
    let error = engine
        .force(&EngineRequest {
            positions: &[0.0; 4],
            atomic_numbers: &[1, 1],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert!(matches!(error, EngineError::Invalid(_)), "{error}");
    let empty = engine
        .force(&EngineRequest {
            positions: &[],
            atomic_numbers: &[],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert!(matches!(empty, EngineError::Invalid(_)), "{empty}");
}

#[test]
fn dropping_the_instance_destroys_the_engine() {
    let path = build("harmonic_drop", &[]);
    let before = destroyed(&path);
    drop(load(&path));
    assert_eq!(destroyed(&path), before + 1);
}

#[test]
fn the_potential_handle_evaluates_and_destroys_the_engine() {
    let path = build("harmonic_handle", &[]);
    let before = destroyed(&path);
    let potential = load(&path).into_potential().expect("potential handle");
    assert!(potential.as_handle().has_hessian_vector_callback());
    let mut positions = vec![0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let mut numbers = vec![1, 8];
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 2, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 2) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    let status = unsafe { potential.calculate(&input, &mut output) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let sum: f64 = positions.iter().map(|x| x * x).sum();
    assert!(within_roundoff(
        output.energy,
        0.5 * SPRING * sum,
        3 * positions.len()
    ));
    let forces =
        unsafe { std::slice::from_raw_parts(rgpot_tensor_data(output.forces).cast::<f64>(), 6) };
    let expected: Vec<f64> = positions.iter().map(|x| -SPRING * x).collect();
    assert!(all_within_roundoff(forces, &expected));
    let direction = [1.0, -0.5, 0.25, -2.0, 0.0, 4.0];
    let mut product = [0.0; 6];
    unsafe {
        potential
            .as_handle()
            .hessian_vector(
                &input,
                &direction,
                0.0,
                rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
                &mut product,
            )
            .expect("analytic HVP through the potential handle");
    }
    let expected_product: Vec<f64> = direction.iter().map(|value| SPRING * value).collect();
    assert!(all_within_roundoff(&product, &expected_product));
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    drop(potential);
    assert_eq!(destroyed(&path), before + 1);
}

#[test]
fn the_potential_handle_reports_an_engine_failure_as_a_status() {
    let path = build("harmonic_status", &[]);
    let potential = load(&path).into_potential().expect("potential handle");
    let mut positions = vec![0.0; 3];
    let mut numbers = vec![0];
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 1, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 1) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    let status = unsafe { potential.calculate(&input, &mut output) };
    assert_eq!(status, rgpot_status_t::RGPOT_INTERNAL_ERROR);
    assert!(output.forces.is_null());
    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
}

#[test]
fn unaligned_config_slice_is_word_aligned_for_create() {
    let path = build("harmonic_align", &[]);
    #[repr(align(8))]
    struct Padded([u8; 16]);
    let mut buf = Padded([0; 16]);
    buf.0[1..9].copy_from_slice(&config());
    let unaligned = &buf.0[1..9];
    assert_ne!(unaligned.as_ptr() as usize % 8, 0);
    let mut engine = unsafe { EnginePotential::load(&path, unaligned) }.expect("aligned copy");
    let answer = engine
        .force(&EngineRequest {
            positions: &[1.0, 0.0, 0.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        })
        .expect("force");
    assert!(within_roundoff(answer.energy, 0.5 * SPRING, 3));
}

#[test]
fn hvp_returns_the_harmonic_hessian_product() {
    let path = build("harmonic_hvp", &[]);
    let mut engine = load(&path);
    assert!(engine.caps().expect("caps").analytic_hvp);
    let positions = [0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let direction = [1.0, -0.5, 0.25, -2.0, 0.0, 4.0];
    let product = engine
        .hvp(
            &EngineRequest {
                positions: &positions,
                atomic_numbers: &[1, 8],
                box_matrix: &CELL,
            },
            &direction,
        )
        .expect("hvp");
    let expected: Vec<f64> = direction.iter().map(|value| SPRING * value).collect();
    assert!(all_within_roundoff(&product, &expected));
}

#[test]
fn hvp_is_discovered_by_symbol_presence_without_caps() {
    let path = build("harmonic_nocaps_hvp", &["NO_CAPS"]);
    let mut engine = load(&path);
    assert!(matches!(
        engine.caps(),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_caps",
            ..
        })
    ));
    let positions = [0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let direction = [1.0, -0.5, 0.25, -2.0, 0.0, 4.0];
    let product = engine
        .hvp(
            &EngineRequest {
                positions: &positions,
                atomic_numbers: &[1, 8],
                box_matrix: &CELL,
            },
            &direction,
        )
        .expect("hvp without rgpot_engine_caps");
    let expected: Vec<f64> = direction.iter().map(|value| SPRING * value).collect();
    assert!(all_within_roundoff(&product, &expected));

    let potential = load(&path)
        .into_potential()
        .expect("potential handle without rgpot_engine_caps");
    assert!(potential.as_handle().has_hessian_vector_callback());
    let mut pos_vec = positions.to_vec();
    let mut numbers = vec![1, 8];
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(pos_vec.as_mut_ptr(), 2, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 2) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let mut handle_product = [0.0; 6];
    unsafe {
        potential
            .as_handle()
            .hessian_vector(
                &input,
                &direction,
                0.0,
                rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
                &mut handle_product,
            )
            .expect("analytic HVP through handle without rgpot_engine_caps");
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    assert!(all_within_roundoff(&handle_product, &expected));
}

#[test]
fn hvp_without_the_symbol_or_narrowed_by_caps_is_unsupported() {
    let no_symbol = build("harmonic_nohvp", &["NO_HVP"]);
    let mut engine = load(&no_symbol);
    assert!(!engine.caps().expect("caps").analytic_hvp);
    let error = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 3],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        )
        .unwrap_err();
    assert!(matches!(
        error,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        }
    ));
    let potential = load(&no_symbol)
        .into_potential()
        .expect("handle without HVP symbol");
    assert!(!potential.as_handle().has_hessian_vector_callback());

    let bogus_caps = build("harmonic_nohvp_forcecap", &["NO_HVP", "FORCE_HVP_CAP"]);
    let mut engine = load(&bogus_caps);
    assert!(!engine.caps().expect("caps").analytic_hvp);
    assert!(matches!(
        engine.hvp(
            &EngineRequest {
                positions: &[0.0; 3],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        ),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        })
    ));
    let potential = load(&bogus_caps)
        .into_potential()
        .expect("caps cannot enable a missing HVP symbol");
    assert!(!potential.as_handle().has_hessian_vector_callback());

    let no_capability = build("harmonic_nohvpcap", &["NO_HVP_CAP"]);
    let mut engine = load(&no_capability);
    assert!(!engine.caps().expect("caps").analytic_hvp);
    let error = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 3],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        )
        .unwrap_err();
    assert!(matches!(
        error,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        }
    ));
    let narrowed_handle = load(&no_capability)
        .into_potential()
        .expect("handle with HVP narrowed off by caps");
    assert!(!narrowed_handle.as_handle().has_hessian_vector_callback());

    let short_caps = build("harmonic_shortcaps", &["CAPS_SIZE_BEFORE_HVP"]);
    let mut engine = load(&short_caps);
    assert!(engine.caps().expect("short caps").analytic_hvp);
    let short_product = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 3],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[2.0, -1.0, 0.5],
        )
        .expect("short caps preserves symbol-presence HVP");
    assert!(all_within_roundoff(
        &short_product,
        &[2.0 * SPRING, -SPRING, 0.5 * SPRING]
    ));
}

#[test]
fn caps_failure_is_not_silently_downgraded() {
    let path = build("harmonic_capserror", &["CAPS_ERROR"]);
    let error = load(&path)
        .into_potential()
        .err()
        .expect("capability error must be returned");
    assert!(matches!(error, EngineError::Recoverable { code: 2, .. }));
}

#[test]
fn unsupported_code_keeps_the_engine_message_and_matches_the_header() {
    let path = build("harmonic_hvpunsupported", &["HVP_UNSUPPORTED"]);
    assert_eq!(unsupported_code(&path), ENGINE_ERR_UNSUPPORTED);
    let mut engine = load(&path);
    let error = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 3],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        )
        .unwrap_err();
    assert_eq!(
        error,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            message: "harmonic engine built without analytic HVP".into(),
        }
    );
}

#[test]
fn force_uses_the_header_unsupported_code() {
    let path = build("harmonic_forceunsupported", &["FORCE_UNSUPPORTED"]);
    let mut engine = load(&path);
    let error = engine
        .force(&EngineRequest {
            positions: &[0.0; 3],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert_eq!(
        error,
        EngineError::Unsupported {
            operation: "rgpot_engine_force",
            message: "force operation unavailable".into(),
        }
    );
}

#[test]
fn hvp_maps_recoverable_and_fatal_codes() {
    let path = build("harmonic_hvpcodes", &[]);
    let mut recoverable_engine = load(&path);
    let positions = [0.0; 3];
    assert_eq!(
        recoverable_engine
            .hvp(
                &EngineRequest {
                    positions: &positions,
                    atomic_numbers: &[0],
                    box_matrix: &CELL,
                },
                &[1.0, 0.0, 0.0],
            )
            .unwrap_err(),
        EngineError::Recoverable {
            code: 1,
            message: "atom 0 has atomic number 0".into(),
        }
    );
    let mut fatal_engine = load(&path);
    let error = fatal_engine
        .hvp(
            &EngineRequest {
                positions: &positions,
                atomic_numbers: &[-1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        )
        .unwrap_err();
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "atom 0 has a negative atomic number".into(),
        }
    );
    assert_eq!(
        fatal_engine
            .hvp(
                &EngineRequest {
                    positions: &positions,
                    atomic_numbers: &[1],
                    box_matrix: &CELL,
                },
                &[1.0, 0.0, 0.0],
            )
            .unwrap_err(),
        error
    );
}

#[test]
fn hvp_rejects_mismatched_direction_length() {
    let path = build("harmonic_hvplen", &[]);
    let mut engine = load(&path);
    let error = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 6],
                atomic_numbers: &[1, 8],
                box_matrix: &CELL,
            },
            &[1.0, 2.0, 3.0],
        )
        .unwrap_err();
    assert!(matches!(error, EngineError::Invalid(_)), "{error}");
}

fn build_with_abs_zero(name: &str, defines: &[&str], zero_symbols: &[&str]) -> PathBuf {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/harmonic_engine.c");
    let include = Path::new(env!("CARGO_MANIFEST_DIR")).join("../CppCore");
    let suffix = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let tmpdir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let asm = tmpdir.join(format!("{name}_abs_zero.s"));
    let prefix = if cfg!(target_os = "macos") { "_" } else { "" };
    let mut asm_src = String::new();
    for sym in zero_symbols {
        asm_src.push_str(&format!(".globl {prefix}{sym}\n.set {prefix}{sym}, 0\n"));
    }
    if !cfg!(target_os = "macos") {
        asm_src.push_str(".section .note.GNU-stack,\"\",%progbits\n");
    }
    std::fs::write(&asm, asm_src).expect("write ABS-zero assembly fixture");

    let out = tmpdir.join(format!("lib{name}.{suffix}"));
    let target = env!("RGPOT_BUILD_TARGET");
    let compiler = match cc::Build::new()
        .target(target)
        .host(target)
        .opt_level(0)
        .try_get_compiler()
    {
        Ok(compiler) => compiler,
        Err(why) => panic!(
            "cannot compile C fixture '{name}': no C compiler available for target {target} ({why})"
        ),
    };
    let mut command = compiler.to_command();
    if cfg!(target_os = "macos") {
        command.arg("-dynamiclib");
    } else {
        command.args(["-shared", "-fPIC"]);
    }
    command
        .arg("-I")
        .arg(include)
        .arg("-o")
        .arg(&out)
        .arg(&source)
        .arg(&asm);
    for define in defines {
        command.arg(format!("-D{define}"));
    }
    let status = match command.status() {
        Ok(status) => status,
        Err(why) => panic!("spawn C compiler {:?}: {why}", compiler.path()),
    };
    assert!(
        status.success(),
        "the ABS-zero fixture did not compile: {status}"
    );
    out
}

#[test]
fn abs_zero_symbols_are_none_when_optional_and_missing_when_required() {
    let null_hvp_with_caps = build_with_abs_zero(
        "harmonic_abs0_hvp_caps",
        &["NO_HVP", "FORCE_HVP_CAP"],
        &["rgpot_engine_hvp"],
    );
    let mut engine = load(&null_hvp_with_caps);
    assert!(!engine.caps().expect("caps").analytic_hvp);
    assert_eq!(
        engine.hvp(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        ),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            message: "symbol is not exported".into(),
        })
    );
    let handle = load(&null_hvp_with_caps)
        .into_potential()
        .expect("potential handle with ABS-zero hvp");
    assert!(!handle.as_handle().has_hessian_vector_callback());

    let null_all_optional = build_with_abs_zero(
        "harmonic_abs0_all_opt",
        &["NO_STAMP", "NO_BATCH", "NO_CAPS", "NO_LAST_ERROR", "NO_HVP"],
        &[
            "rgpot_engine_abi_stamp",
            "rgpot_engine_abi_compatible",
            "rgpot_engine_force_batch",
            "rgpot_engine_hvp",
            "rgpot_engine_caps",
            "rgpot_engine_last_error",
        ],
    );
    let mut engine = load(&null_all_optional);
    let eval = engine
        .force(&EngineRequest {
            positions: &[1.0, 0.0, 0.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        })
        .expect("force with ABS-zero optional symbols");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING, 3));
    let handle = load(&null_all_optional)
        .into_potential()
        .expect("handle with ABS-zero optional symbols");
    assert!(!handle.as_handle().has_hessian_vector_callback());

    let null_force = build_with_abs_zero(
        "harmonic_abs0_force",
        &["NO_FORCE", "NO_BATCH"],
        &["rgpot_engine_force"],
    );
    let before = created(&null_force);
    let err = unsafe { EnginePotential::load(&null_force, &config()) }
        .err()
        .expect("ABS-zero required rgpot_engine_force must fail load");
    assert_eq!(err, EngineError::Missing("rgpot_engine_force".into()));
    assert_eq!(created(&null_force), before);

    // Build a standalone shared object where all required symbols are exported,
    // with one required symbol set to ABS 0 at a time, so every required lookup
    // is verified without invoking a null function pointer.
    let required_names = [
        "rgpot_engine_abi_version",
        "rgpot_engine_available",
        "rgpot_engine_create",
        "rgpot_engine_destroy",
        "rgpot_engine_force",
    ];
    let suffix = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let prefix = if cfg!(target_os = "macos") { "_" } else { "" };
    let target = env!("RGPOT_BUILD_TARGET");
    let compiler = cc::Build::new()
        .target(target)
        .host(target)
        .opt_level(0)
        .try_get_compiler()
        .expect("C compiler");
    for (idx, missing_sym) in required_names.iter().enumerate() {
        let asm = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("req_abs0_{idx}.s"));
        let out =
            Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("libreq_abs0_{idx}.{suffix}"));
        let mut text = String::from(".text\n");
        for sym in required_names {
            if sym == *missing_sym {
                text.push_str(&format!(".globl {prefix}{sym}\n.set {prefix}{sym}, 0\n"));
            } else {
                text.push_str(&format!(
                    ".globl {prefix}{sym}\n{prefix}{sym}:\n  .byte 0\n"
                ));
            }
        }
        if !cfg!(target_os = "macos") {
            text.push_str(".section .note.GNU-stack,\"\",%progbits\n");
        }
        std::fs::write(&asm, text).expect("write required ABS-zero asm");
        let mut cmd = compiler.to_command();
        if cfg!(target_os = "macos") {
            cmd.arg("-dynamiclib");
        } else {
            cmd.args(["-shared", "-fPIC"]);
        }
        let status = cmd
            .arg("-o")
            .arg(&out)
            .arg(&asm)
            .status()
            .expect("link asm");
        assert!(status.success(), "link {missing_sym} fixture: {status}");
        let err = unsafe { EnginePotential::load(&out, &config()) }
            .err()
            .expect("ABS-zero required symbol must be rejected during resolve");
        assert_eq!(err, EngineError::Missing((*missing_sym).into()));
    }
}

#[test]
fn lock_error_code_matches_header_maps_on_rust_side_and_marks_handle_unusable() {
    use rgpot_core::engine::ENGINE_ERR_LOCK;

    let force_lock = build("harmonic_forcelock", &["FORCE_LOCK_ERROR"]);
    let library = unsafe { libloading::Library::new(&force_lock) }.expect("reopen fixture");
    let header_code: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_lock_error_code\0") }.expect("lock error code symbol");
    assert_eq!(unsafe { header_code() }, ENGINE_ERR_LOCK);

    let mut engine = load(&force_lock);
    let request = EngineRequest {
        positions: &[1.0, 0.0, 0.0],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };
    let expected = EngineError::Lock {
        operation: "rgpot_engine_force",
        message: "fixture handle lock failed".into(),
    };
    assert_eq!(engine.force(&request).unwrap_err(), expected);
    assert_eq!(engine.force(&request).unwrap_err(), expected);
    assert_eq!(expected.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);

    let hvp_lock = build("harmonic_hvplock", &["HVP_LOCK_ERROR"]);
    let mut hvp_engine = load(&hvp_lock);
    let expected_hvp = EngineError::Lock {
        operation: "rgpot_engine_hvp",
        message: "fixture hvp lock failed".into(),
    };
    assert_eq!(
        hvp_engine.hvp(&request, &[1.0, 0.0, 0.0]).unwrap_err(),
        expected_hvp
    );
    assert_eq!(hvp_engine.force(&request).unwrap_err(), expected_hvp);

    let caps_lock = build("harmonic_capslock", &["CAPS_LOCK_ERROR"]);
    let caps_err = load(&caps_lock)
        .into_potential()
        .err()
        .expect("caps lock failure must fail into_potential");
    assert_eq!(
        caps_err,
        EngineError::Lock {
            operation: "rgpot_engine_caps",
            message: "fixture caps lock failed".into(),
        }
    );
}

fn fixture_max_concurrent(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen fixture");
    let f: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_max_concurrent\0") }.expect("max concurrent symbol");
    unsafe { f() }
}

fn fixture_max_handle_concurrent(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen fixture");
    let f: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_max_handle_concurrent\0") }
            .expect("max handle concurrent symbol");
    unsafe { f() }
}

fn fixture_reset_concurrent(path: &Path) {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen fixture");
    let f: libloading::Symbol<unsafe extern "C" fn()> =
        unsafe { library.get(b"fixture_reset_concurrent\0") }.expect("reset concurrent symbol");
    unsafe { f() };
}

#[test]
fn concurrent_handles_serialize_each_handle_and_parallelize_distinct_handles() {
    use rgpot_core::c_api::hvp::rgpot_potential_hvp;
    use rgpot_core::c_api::potential::rgpot_potential_calculate;
    use std::sync::{Arc, Barrier};

    let path = build(
        "harmonic_concurrent_handles",
        &["FIXTURE_REENTRANCY=1", "CONCURRENCY_PROBE"],
    );
    let mut probe = load(&path);
    assert_eq!(probe.caps().expect("caps").reentrancy, 1);
    drop(probe);

    // 1. Overlapping force and analytic HVP calls on a single PotentialImpl handle
    //    must serialize through the handle mutex.
    fixture_reset_concurrent(&path);
    let shared_potential = load(&path).into_potential().expect("shared handle");
    let raw_addr = shared_potential.as_handle() as *const _ as usize;
    let start = Arc::new(Barrier::new(4));
    std::thread::scope(|scope| {
        for worker in 0..4usize {
            let start = Arc::clone(&start);
            scope.spawn(move || {
                let pot = raw_addr as *const rgpot_core::potential::rgpot_potential_t;
                let scale = (worker + 1) as f64;
                let mut positions = [0.5 * scale, -scale, 2.0 * scale];
                let mut numbers = [1_i32];
                let mut cell = CELL;
                let input = rgpot_force_input_t {
                    positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 1, 3) },
                    atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 1) },
                    box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
                };
                start.wait();
                for step_idx in 0..4usize {
                    if step_idx % 2 == 0 {
                        let mut output = rgpot_force_out_t {
                            forces: std::ptr::null_mut(),
                            energy: 0.0,
                            variance: 0.0,
                        };
                        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
                        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
                        let sum: f64 = positions.iter().map(|v| v * v).sum();
                        assert!(within_roundoff(output.energy, 0.5 * SPRING * sum, 9));
                        unsafe { rgpot_tensor_free(output.forces) };
                    } else {
                        let direction = [scale, -0.5 * scale, 0.25 * scale];
                        let mut hv = [0.0; 3];
                        let status = unsafe {
                            rgpot_potential_hvp(
                                pot,
                                &input,
                                direction.as_ptr(),
                                0.0,
                                rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC as i32,
                                hv.as_mut_ptr(),
                            )
                        };
                        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
                        let want: Vec<f64> = direction.iter().map(|d| SPRING * d).collect();
                        assert!(all_within_roundoff(&hv, &want));
                    }
                }
                unsafe {
                    rgpot_tensor_free(input.positions);
                    rgpot_tensor_free(input.atomic_numbers);
                    rgpot_tensor_free(input.box_matrix);
                }
            });
        }
    });
    assert_eq!(fixture_max_concurrent(&path), 1);
    assert_eq!(fixture_max_handle_concurrent(&path), 1);
    drop(shared_potential);

    // 2. Distinct PotentialImpl handles of a reentrancy==1 engine evaluate
    //    concurrently on separate worker threads without blocking each other.
    fixture_reset_concurrent(&path);
    let before_destroy = destroyed(&path);
    let handles: Vec<rgpot_core::potential::PotentialImpl> = (0..4)
        .map(|_| load(&path).into_potential().expect("worker handle"))
        .collect();
    let gate = Arc::new(Barrier::new(handles.len()));
    std::thread::scope(|scope| {
        for (idx, handle) in handles.into_iter().enumerate() {
            let gate = Arc::clone(&gate);
            scope.spawn(move || {
                let factor = (idx + 1) as f64;
                let mut positions = [factor, -2.0 * factor, 0.5 * factor];
                let mut numbers = [8_i32];
                let mut cell = CELL;
                let input = rgpot_force_input_t {
                    positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 1, 3) },
                    atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 1) },
                    box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
                };
                gate.wait();
                for _ in 0..4 {
                    let mut output = rgpot_force_out_t {
                        forces: std::ptr::null_mut(),
                        energy: 0.0,
                        variance: 0.0,
                    };
                    let status = unsafe { handle.calculate(&input, &mut output) };
                    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
                    let sum: f64 = positions.iter().map(|v| v * v).sum();
                    assert!(within_roundoff(output.energy, 0.5 * SPRING * sum, 9));
                    unsafe { rgpot_tensor_free(output.forces) };
                }
                unsafe {
                    rgpot_tensor_free(input.positions);
                    rgpot_tensor_free(input.atomic_numbers);
                    rgpot_tensor_free(input.box_matrix);
                }
            });
        }
    });
    assert_eq!(fixture_max_handle_concurrent(&path), 1);
    let cpus = match std::thread::available_parallelism() {
        Ok(count) => count.get(),
        Err(_) => 1,
    };
    if cpus >= 2 {
        assert!(
            fixture_max_concurrent(&path) >= 2,
            "expected parallel execution across distinct reentrancy==1 handles, got max_concurrent={}",
            fixture_max_concurrent(&path)
        );
    }
    assert_eq!(destroyed(&path), before_destroy + 4);
}

#[test]
fn reentrancy_two_engines_serialize_across_handles() {
    use std::sync::{Arc, Barrier};

    let path = build(
        "harmonic_reentrancy2",
        &["FIXTURE_REENTRANCY=2", "CONCURRENCY_PROBE"],
    );
    let mut probe = load(&path);
    let caps = probe.caps().expect("reentrancy==2 caps");
    assert_eq!(caps.reentrancy, 2);
    assert!(caps.analytic_hvp);
    drop(probe);

    // Direct EnginePotential instances on separate threads must serialize across handles.
    fixture_reset_concurrent(&path);
    let direct_gate = Arc::new(Barrier::new(3));
    std::thread::scope(|scope| {
        for idx in 0..3usize {
            let path_ref = &path;
            let direct_gate = Arc::clone(&direct_gate);
            scope.spawn(move || {
                let mut engine = load(path_ref);
                let factor = (idx + 1) as f64;
                let positions = [factor, -factor, 0.5 * factor];
                let request = EngineRequest {
                    positions: &positions,
                    atomic_numbers: &[1],
                    box_matrix: &CELL,
                };
                direct_gate.wait();
                for step_idx in 0..4usize {
                    if step_idx % 2 == 0 {
                        let eval = engine.force(&request).expect("reentrancy==2 direct force");
                        let sum: f64 = positions.iter().map(|v| v * v).sum();
                        assert!(within_roundoff(eval.energy, 0.5 * SPRING * sum, 9));
                    } else {
                        let dir = [1.0, -2.0, 0.5];
                        let hv = engine
                            .hvp(&request, &dir)
                            .expect("reentrancy==2 direct hvp");
                        assert!(all_within_roundoff(
                            &hv,
                            &[SPRING, -2.0 * SPRING, 0.5 * SPRING]
                        ));
                    }
                }
            });
        }
    });
    assert_eq!(
        fixture_max_concurrent(&path),
        1,
        "reentrancy==2 EnginePotential instances overlapped"
    );

    // PotentialImpl handles created via into_potential() must also serialize across handles.
    fixture_reset_concurrent(&path);
    let before_destroy = destroyed(&path);
    let handles: Vec<rgpot_core::potential::PotentialImpl> = (0..4)
        .map(|_| load(&path).into_potential().expect("reentrancy==2 handle"))
        .collect();
    let handle_gate = Arc::new(Barrier::new(handles.len()));
    std::thread::scope(|scope| {
        for (idx, handle) in handles.into_iter().enumerate() {
            let handle_gate = Arc::clone(&handle_gate);
            scope.spawn(move || {
                let factor = (idx + 1) as f64;
                let mut positions = [0.5 * factor, 1.5 * factor, -factor];
                let mut numbers = [1_i32];
                let mut cell = CELL;
                let input = rgpot_force_input_t {
                    positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 1, 3) },
                    atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 1) },
                    box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
                };
                handle_gate.wait();
                for step_idx in 0..4usize {
                    if step_idx % 2 == 0 {
                        let mut output = rgpot_force_out_t {
                            forces: std::ptr::null_mut(),
                            energy: 0.0,
                            variance: 0.0,
                        };
                        let status = unsafe { handle.calculate(&input, &mut output) };
                        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
                        let sum: f64 = positions.iter().map(|v| v * v).sum();
                        assert!(within_roundoff(output.energy, 0.5 * SPRING * sum, 9));
                        unsafe { rgpot_tensor_free(output.forces) };
                    } else {
                        let direction = [factor, -0.5 * factor, 2.0 * factor];
                        let mut hv = [0.0; 3];
                        unsafe {
                            handle
                                .as_handle()
                                .hessian_vector(
                                    &input,
                                    &direction,
                                    0.0,
                                    rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
                                    &mut hv,
                                )
                                .expect("reentrancy==2 handle HVP");
                        }
                        let want: Vec<f64> = direction.iter().map(|d| SPRING * d).collect();
                        assert!(all_within_roundoff(&hv, &want));
                    }
                }
                unsafe {
                    rgpot_tensor_free(input.positions);
                    rgpot_tensor_free(input.atomic_numbers);
                    rgpot_tensor_free(input.box_matrix);
                }
            });
        }
    });
    assert_eq!(
        fixture_max_concurrent(&path),
        1,
        "reentrancy==2 PotentialImpl handles overlapped"
    );
    assert_eq!(destroyed(&path), before_destroy + 4);
}

#[test]
fn an_engine_without_stamp_symbols_uses_the_major_contract() {
    let path = build("harmonic_nostamp", &["NO_STAMP"]);
    let mut engine = load(&path);
    let eval = engine
        .force(&EngineRequest {
            positions: &[1.0, -2.0, 0.5],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        })
        .expect("force on unstamped engine");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING * 5.25, 9));
    assert!(
        engine
            .caps()
            .expect("caps on unstamped engine")
            .analytic_hvp
    );
}

#[test]
fn unstamped_compatible_and_older_stamped_minors_load_while_future_minors_fail() {
    let request = EngineRequest {
        positions: &[1.0, 0.0, 0.0],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };

    // Unstamped engine that still exports rgpot_engine_abi_compatible at minor 0:
    // loader must query baseline minor 0 rather than host minor 3.
    let unstamped_compat = build(
        "harmonic_nostamp_compat_minor0",
        &["NO_STAMP_ONLY", "FIXTURE_MINOR=0"],
    );
    let mut engine = load(&unstamped_compat);
    let eval = engine
        .force(&request)
        .expect("unstamped engine with minor-0 abi_compatible");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING, 3));

    // Stamped minor 1 is accepted by a minor 3 loader (1 <= 3).
    let minor1 = build("harmonic_stamped_minor1", &["FIXTURE_MINOR=1"]);
    let mut engine = load(&minor1);
    let eval = engine.force(&request).expect("stamped minor 1 force");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING, 3));

    // Stamped minor above ENGINE_ABI_MINOR is rejected before create.
    let minor5 = build("harmonic_stamped_minor5", &["FIXTURE_MINOR=5"]);
    let before = created(&minor5);
    let err = unsafe { EnginePotential::load(&minor5, &config()) }
        .err()
        .expect("minor 5 must be rejected");
    assert_eq!(
        err,
        EngineError::Incompatible {
            host: RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: ENGINE_ABI_MINOR,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            },
            engine: Some(RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: 5,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            }),
        }
    );
    assert_eq!(created(&minor5), before);
}

/// Parameters of the coupled-well tests. Every value is a dyadic rational, so the
/// closed forms below are exact in binary floating point.
const WELL_NUMBER: i32 = 1;
const WELL_BARRIER: f64 = 3.0;
const WELL_POSITION: f64 = 2.0;
const WELL_SPRING: f64 = 4.0;
const WELL_COUPLING: f64 = 0.25;
/// Terms in the longest expression the oracle and the engine each evaluate.
const WELL_TERMS: usize = 16;

fn well_config() -> CoupledWellConfig {
    CoupledWellConfig::default()
        .with_atomic_number(WELL_NUMBER)
        .with_barrier_energy(WELL_BARRIER)
        .with_well_position(WELL_POSITION)
        .with_transverse_spring(WELL_SPRING)
        .with_coupling(WELL_COUPLING)
}

fn load_well() -> EnginePotential {
    unsafe { EnginePotential::load(build_coupled_well(), well_config().as_bytes()) }
        .expect("coupled-well engine")
}

fn well_request(positions: &[f64; 3]) -> EngineRequest<'_> {
    EngineRequest {
        positions,
        atomic_numbers: &[WELL_NUMBER],
        box_matrix: &CELL,
    }
}

/// Energy, gradient and Hessian of the model from the expanded polynomial
///   E = V0 (x^4/x0^4 - 2 x^2/x0^2 + 1) + k/2 (y^2 - 2 a x^2 y + a^2 x^4),
/// differentiated by hand, not from the engine's factored form.
struct Expansion {
    energy: f64,
    gradient: [f64; 2],
    hessian: [f64; 3],
}

fn expansion(x: f64, y: f64) -> Expansion {
    let (v0, x0, k, a) = (WELL_BARRIER, WELL_POSITION, WELL_SPRING, WELL_COUPLING);
    let (x2, x02) = (x * x, x0 * x0);
    let (x4, x04) = (x2 * x2, x02 * x02);
    Expansion {
        energy: v0 * (x4 / x04 - 2.0 * x2 / x02 + 1.0)
            + 0.5 * k * (y * y - 2.0 * a * x2 * y + a * a * x4),
        gradient: [
            v0 * (4.0 * x * x2 / x04 - 4.0 * x / x02)
                + k * (2.0 * a * a * x * x2 - 2.0 * a * x * y),
            k * (y - a * x2),
        ],
        hessian: [
            v0 * (12.0 * x2 / x04 - 4.0 / x02) + k * (6.0 * a * a * x2 - 2.0 * a * y),
            -2.0 * k * a * x,
            k,
        ],
    }
}

const WELL_XS: [f64; 7] = [-3.0, -2.0, -0.5, 0.0, 1.0, 2.0, 2.5];
const WELL_YS: [f64; 5] = [-1.5, 0.0, 0.5, 1.0, 3.0];

#[test]
fn coupled_well_force_and_hvp_are_the_analytic_derivatives_on_a_grid() {
    let mut engine = load_well();
    let directions = [[1.0, 0.0, 9.0], [0.0, 1.0, 9.0], [0.5, -1.5, 4.0]];
    for x in WELL_XS {
        for y in WELL_YS {
            let positions = [x, y, 7.0];
            let want = expansion(x, y);
            let answer = engine.force(&well_request(&positions)).expect("force");
            assert!(
                within_roundoff(answer.energy, want.energy, WELL_TERMS),
                "energy at ({x}, {y}): {} vs {}",
                answer.energy,
                want.energy
            );
            for (component, gradient) in want.gradient.iter().enumerate() {
                assert!(
                    within_roundoff(answer.forces[component], -gradient, WELL_TERMS),
                    "force {component} at ({x}, {y}): {} vs {}",
                    answer.forces[component],
                    -gradient
                );
            }
            assert_eq!(answer.forces[2], 0.0, "the energy does not depend on z");
            for direction in directions {
                let product = engine
                    .hvp(&well_request(&positions), &direction)
                    .expect("hvp");
                let [hxx, hxy, hyy] = want.hessian;
                let expected = [
                    hxx * direction[0] + hxy * direction[1],
                    hxy * direction[0] + hyy * direction[1],
                    0.0,
                ];
                for component in 0..3 {
                    assert!(
                        within_roundoff(product[component], expected[component], WELL_TERMS),
                        "hvp {component} at ({x}, {y}) along {direction:?}: {} vs {}",
                        product[component],
                        expected[component]
                    );
                }
            }
        }
    }
}

#[test]
fn coupled_well_stationary_points_have_the_hand_computed_curvature() {
    let mut engine = load_well();
    let unit_x = [1.0, 0.0, 0.0];
    let unit_y = [0.0, 1.0, 0.0];

    // x = x0, y = alpha x0^2: both terms vanish. With u = 1 and a zero
    // displacement, H = [[8 V0 / x0^2 + 4 k a^2 x0^2, -2 k a x0], [., k]].
    let minimum = [2.0, 1.0, 0.0];
    let answer = engine.force(&well_request(&minimum)).expect("minimum");
    assert_eq!(answer.energy, 0.0);
    assert_eq!(answer.forces, [0.0, 0.0, 0.0]);
    let column_x = engine.hvp(&well_request(&minimum), &unit_x).expect("Hx");
    let column_y = engine.hvp(&well_request(&minimum), &unit_y).expect("Hy");
    assert_eq!(column_x, [10.0, -4.0, 0.0]);
    assert_eq!(column_y, [-4.0, 4.0, 0.0]);

    // The barrier top x = 0, y = 0 has energy V0 and H = diag(-4 V0 / x0^2, k).
    let top = [0.0, 0.0, 0.0];
    let answer = engine.force(&well_request(&top)).expect("barrier top");
    assert_eq!(answer.energy, WELL_BARRIER);
    assert_eq!(answer.forces, [0.0, 0.0, 0.0]);
    assert_eq!(
        engine.hvp(&well_request(&top), &unit_x).expect("Hx"),
        [-3.0, 0.0, 0.0]
    );
    assert_eq!(
        engine.hvp(&well_request(&top), &unit_y).expect("Hy"),
        [0.0, 4.0, 0.0]
    );

    // (1, 1/2): E = 29/16, F = (11/4, -1), H = [[-1/4, -2], [-2, 4]].
    let generic = [1.0, 0.5, 0.0];
    let answer = engine
        .force(&well_request(&generic))
        .expect("generic point");
    assert_eq!(answer.energy, 1.8125);
    assert_eq!(answer.forces, [2.75, -1.0, 0.0]);
    let product = engine
        .hvp(&well_request(&generic), &[1.0, 2.0, 5.0])
        .expect("Hd");
    assert_eq!(product, [-4.25, 6.0, 0.0]);
}

#[test]
fn coupled_well_extreme_finite_inputs_and_hvp_edge_cases_stay_servable() {
    // Regression: finite answers at this point used to overflow intermediates.
    let tiny = CoupledWellConfig::default()
        .with_atomic_number(WELL_NUMBER)
        .with_barrier_energy(1.0e-320)
        .with_well_position(1.0e-155)
        .with_transverse_spring(1.0)
        .with_coupling(0.0);
    let mut tiny_engine =
        unsafe { EnginePotential::load(build_coupled_well(), tiny.as_bytes()) }.expect("tiny cfg");
    let tiny_eval = tiny_engine
        .force(&well_request(&[1.0, 0.0, 0.0]))
        .expect("finite tiny-scale answer");
    assert!(tiny_eval.energy.is_finite());
    assert!(tiny_eval.forces.iter().all(|v| v.is_finite()));
    let scaled_min = [1.0e-155, 0.0, 0.0];
    let min_eval = tiny_engine
        .force(&well_request(&scaled_min))
        .expect("scaled minimum answer");
    assert_eq!(min_eval.energy, 0.0);
    assert_eq!(min_eval.forces, vec![0.0, 0.0, 0.0]);
    let min_hv = tiny_engine
        .hvp(&well_request(&scaled_min), &[1.0, 0.0, 0.0])
        .expect("scaled minimum HVP");
    assert!(min_hv[0].is_finite());
    assert_eq!(&min_hv[1..], &[0.0, 0.0]);

    // Regression: force used to require a finite Hessian, and HVP used inf*0.
    let huge_alpha = CoupledWellConfig::default()
        .with_atomic_number(WELL_NUMBER)
        .with_barrier_energy(1.0)
        .with_well_position(1.0)
        .with_transverse_spring(1.0)
        .with_coupling(1.0e200);
    let mut edge_engine =
        unsafe { EnginePotential::load(build_coupled_well(), huge_alpha.as_bytes()) }
            .expect("huge-alpha cfg");
    let edge_point = [1.0, 1.0e200, 0.0];
    let edge_eval = edge_engine
        .force(&well_request(&edge_point))
        .expect("force must not require finite Hessian");
    assert_eq!(edge_eval.energy, 0.0);
    assert_eq!(edge_eval.forces, vec![0.0, 0.0, 0.0]);
    let hv = edge_engine
        .hvp(&well_request(&edge_point), &[0.0, 1.0, 0.0])
        .expect("direct HVP must avoid inf*0");
    assert_eq!(hv, vec![-2.0e200, 1.0, 0.0]);
}

#[test]
fn coupled_well_batch_is_the_analytic_answer_for_each_system() {
    let mut engine = load_well();
    let systems = [[2.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.5, -1.0]];
    let positions: Vec<f64> = systems.iter().flatten().copied().collect();
    let boxes: Vec<f64> = CELL.repeat(systems.len());
    let batch = engine
        .force_batch(&[WELL_NUMBER], &positions, &boxes)
        .expect("batch");
    assert_eq!(batch.len(), systems.len());
    for (system, answer) in systems.iter().zip(&batch) {
        let want = expansion(system[0], system[1]);
        assert!(within_roundoff(answer.energy, want.energy, WELL_TERMS));
        assert!(within_roundoff(
            answer.forces[0],
            -want.gradient[0],
            WELL_TERMS
        ));
        assert!(within_roundoff(
            answer.forces[1],
            -want.gradient[1],
            WELL_TERMS
        ));
        assert_eq!(answer.forces[2], 0.0);
        let single = engine.force(&well_request(system)).expect("single");
        assert_eq!(*answer, single);
    }
    assert_eq!(batch[0].energy, 0.0);
    assert_eq!(batch[1].energy, WELL_BARRIER);
}

#[test]
fn coupled_well_batch_refuses_a_ragged_or_mismatched_request() {
    let mut engine = load_well();
    let positions = [1.0, 0.5, 0.0, 2.0, 1.0, 0.0];
    let ragged = engine
        .force_batch(&[WELL_NUMBER], &positions[..5], &CELL)
        .unwrap_err();
    assert!(matches!(ragged, EngineError::Invalid(_)), "{ragged}");
    let wrong_boxes = engine
        .force_batch(&[WELL_NUMBER], &positions, &CELL)
        .unwrap_err();
    assert!(
        matches!(wrong_boxes, EngineError::Invalid(_)),
        "{wrong_boxes}"
    );
    let empty = engine.force_batch(&[WELL_NUMBER], &[], &[]).unwrap_err();
    assert!(matches!(empty, EngineError::Invalid(_)), "{empty}");
    let two_atoms = engine
        .force_batch(&[WELL_NUMBER, WELL_NUMBER], &positions, &CELL)
        .unwrap_err();
    assert!(
        matches!(two_atoms, EngineError::Recoverable { code: 1, .. }),
        "{two_atoms}"
    );
    let species = engine
        .force_batch(&[WELL_NUMBER + 1], &positions, &CELL.repeat(2))
        .unwrap_err();
    assert!(
        matches!(&species, EngineError::Recoverable { code: 1, message } if message.contains("atomic number")),
        "{species}"
    );
}

#[test]
fn coupled_well_refuses_requests_it_cannot_serve_and_keeps_serving() {
    let mut engine = load_well();
    let two_atoms = engine
        .force(&EngineRequest {
            positions: &[0.0; 6],
            atomic_numbers: &[WELL_NUMBER, WELL_NUMBER],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert!(
        matches!(&two_atoms, EngineError::Recoverable { code: 1, message } if message.contains("one atom")),
        "{two_atoms}"
    );
    let species = engine
        .force(&EngineRequest {
            positions: &[0.0; 3],
            atomic_numbers: &[WELL_NUMBER + 1],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert!(
        matches!(&species, EngineError::Recoverable { code: 1, message } if message.contains("atomic number")),
        "{species}"
    );
    let not_finite = engine
        .force(&well_request(&[f64::NAN, 0.0, 0.0]))
        .unwrap_err();
    assert!(
        matches!(&not_finite, EngineError::Recoverable { code: 1, message } if message.contains("finite")),
        "{not_finite}"
    );
    let overflow = engine
        .force(&well_request(&[1.0e200, 0.0, 0.0]))
        .unwrap_err();
    assert!(
        matches!(overflow, EngineError::Recoverable { code: 2, .. }),
        "{overflow}"
    );
    let direction = engine
        .hvp(&well_request(&[0.0; 3]), &[f64::INFINITY, 0.0, 0.0])
        .unwrap_err();
    assert!(matches!(direction, EngineError::Invalid(_)), "{direction}");
    // None of these is fatal: the instance still answers.
    let answer = engine
        .force(&well_request(&[0.0; 3]))
        .expect("still usable");
    assert_eq!(answer.energy, WELL_BARRIER);
}

#[test]
fn coupled_well_parameters_are_validated_in_rust_and_by_the_engine_alike() {
    let valid = [
        well_config(),
        CoupledWellConfig::default(),
        well_config().with_coupling(-WELL_COUPLING),
        well_config().with_coupling(0.0),
        CoupledWellConfig::new(8, 0.5, 1.5, 2.0, 1.0),
    ];
    for config in valid {
        config.validate().expect("valid in Rust");
        unsafe { EnginePotential::load(build_coupled_well(), config.as_bytes()) }
            .unwrap_or_else(|why| panic!("{config:?} refused by the engine: {why}"));
    }

    let mut wrong_magic = well_config();
    wrong_magic.magic ^= 1;
    let mut wrong_version = well_config();
    wrong_version.version += 1;
    let mut reserved0 = well_config();
    reserved0.reserved0 = 1;
    let mut reserved1 = well_config();
    reserved1.reserved1 = 1;
    let mut too_small = well_config();
    too_small.size = COUPLED_WELL_CONFIG_MIN_SIZE - 8;
    let invalid = [
        ("magic", wrong_magic),
        ("version", wrong_version),
        ("reserved0", reserved0),
        ("reserved1", reserved1),
        ("too-small size", too_small),
        ("zero atomic number", well_config().with_atomic_number(0)),
        (
            "negative atomic number",
            well_config().with_atomic_number(-1),
        ),
        ("zero barrier", well_config().with_barrier_energy(0.0)),
        ("negative barrier", well_config().with_barrier_energy(-1.0)),
        ("NaN barrier", well_config().with_barrier_energy(f64::NAN)),
        ("zero well position", well_config().with_well_position(0.0)),
        (
            "infinite well position",
            well_config().with_well_position(f64::INFINITY),
        ),
        (
            "negative spring",
            well_config().with_transverse_spring(-1.0),
        ),
        ("NaN spring", well_config().with_transverse_spring(f64::NAN)),
        ("NaN coupling", well_config().with_coupling(f64::NAN)),
        (
            "infinite coupling",
            well_config().with_coupling(f64::NEG_INFINITY),
        ),
    ];
    for (name, config) in invalid {
        let rust = config.validate();
        assert!(
            matches!(rust, Err(EngineError::Invalid(_))),
            "{name}: Rust accepted it"
        );
        let engine = unsafe { EnginePotential::load(build_coupled_well(), config.as_bytes()) };
        assert!(
            matches!(engine.err(), Some(EngineError::Create(why)) if !why.is_empty()),
            "{name}: the engine accepted it or gave no reason"
        );
    }

    let truncated = unsafe {
        EnginePotential::load(
            build_coupled_well(),
            &well_config().as_bytes()[..std::mem::size_of::<CoupledWellConfig>() - 8],
        )
    };
    assert!(matches!(
        truncated.err(),
        Some(EngineError::Create(why))
            if why.contains("CoupledWellConfigV1")
                || why.contains("shorter than config.size")
    ));

    let mut extended = vec![0_u8; std::mem::size_of::<CoupledWellConfig>() + 16];
    extended[..std::mem::size_of::<CoupledWellConfig>()].copy_from_slice(well_config().as_bytes());
    let new_size = extended.len() as u32;
    extended[4..8].copy_from_slice(&new_size.to_ne_bytes());
    unsafe { EnginePotential::load(build_coupled_well(), &extended) }
        .expect("future coupled-well config with zero trailing bytes");
    extended[std::mem::size_of::<CoupledWellConfig>()] = 1;
    assert!(matches!(
        unsafe { EnginePotential::load(build_coupled_well(), &extended) }.err(),
        Some(EngineError::Create(why)) if why.contains("trailing bytes")
    ));
}

#[test]
fn coupled_well_config_bytes_follow_the_header_layout() {
    let config = well_config();
    let bytes = config.as_bytes();
    assert_eq!(bytes.len(), COUPLED_WELL_CONFIG_MIN_SIZE as usize);
    let word = |offset: usize| u32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let real = |offset: usize| f64::from_ne_bytes(bytes[offset..offset + 8].try_into().unwrap());
    assert_eq!(word(0), 0x4357_5054, "magic");
    assert_eq!(word(4), COUPLED_WELL_CONFIG_MIN_SIZE, "size");
    assert_eq!(word(8), 1, "version");
    assert_eq!(word(12), 0, "reserved0");
    assert_eq!(word(16), WELL_NUMBER as u32, "atomic_number");
    assert_eq!(word(20), 0, "reserved1");
    assert_eq!(real(24), WELL_BARRIER);
    assert_eq!(real(32), WELL_POSITION);
    assert_eq!(real(40), WELL_SPRING);
    assert_eq!(real(48), WELL_COUPLING);
    assert_eq!(
        CoupledWellConfig::new(WELL_NUMBER, 3.0, 2.0, 4.0, 0.25),
        config
    );
    assert_eq!(CoupledWellConfig::default().validate().ok(), Some(()));
}

#[test]
fn potlibc_config_bytes_follow_the_header_layout() {
    let config = PotlibcConfig::new("vbo-beta1", 4).expect("config");
    let bytes = config.as_bytes();
    assert_eq!(bytes.len(), POTLIBC_CONFIG_MIN_SIZE as usize);
    assert_eq!(
        u32::from_ne_bytes(bytes[0..4].try_into().unwrap()),
        POTLIBC_CONFIG_MAGIC
    );
    assert_eq!(
        u32::from_ne_bytes(bytes[4..8].try_into().unwrap()),
        POTLIBC_CONFIG_MIN_SIZE
    );
    assert_eq!(
        u32::from_ne_bytes(bytes[8..12].try_into().unwrap()),
        POTLIBC_CONFIG_VERSION
    );
    assert_eq!(u32::from_ne_bytes(bytes[12..16].try_into().unwrap()), 0);
    assert_eq!(u64::from_ne_bytes(bytes[16..24].try_into().unwrap()), 4);
    assert_eq!(&bytes[24..33], b"vbo-beta1");
    assert!(bytes[33..].iter().all(|byte| *byte == 0));
    config.validate().expect("valid");

    let fixed = PotlibcConfig::default()
        .with_surface_id("brh2leps")
        .expect("id");
    assert_eq!(fixed, PotlibcConfig::new("brh2leps", 0).expect("config"));
    let resized = fixed.with_atom_count(7);
    assert_eq!(
        u64::from_ne_bytes(resized.as_bytes()[16..24].try_into().unwrap()),
        7
    );
    let replaced = resized.with_surface_id("fh25sec").expect("id");
    assert_eq!(&replaced.as_bytes()[24..], &{
        let mut expected = [0_u8; POTLIBC_SURFACE_ID_CAPACITY];
        expected[..7].copy_from_slice(b"fh25sec");
        expected
    });
}

#[test]
fn potlibc_config_refuses_an_unusable_surface_id() {
    let longest = "a".repeat(POTLIBC_SURFACE_ID_CAPACITY - 1);
    assert!(PotlibcConfig::new(&longest, 0).is_ok());
    for (name, id) in [
        ("empty", String::new()),
        (
            "no room for the terminator",
            "a".repeat(POTLIBC_SURFACE_ID_CAPACITY),
        ),
        ("embedded NUL", "br\0h2".to_owned()),
    ] {
        assert!(
            matches!(PotlibcConfig::new(&id, 0), Err(EngineError::Invalid(_))),
            "{name}"
        );
    }
    assert!(matches!(
        PotlibcConfig::default().validate(),
        Err(EngineError::Invalid(_))
    ));
    let mut unterminated = PotlibcConfig::new("brh2leps", 0).expect("config");
    unterminated.surface_id = [b'a'; POTLIBC_SURFACE_ID_CAPACITY];
    assert!(matches!(
        unterminated.validate(),
        Err(EngineError::Invalid(_))
    ));
    let mut wrong_magic = PotlibcConfig::new("brh2leps", 0).expect("config");
    wrong_magic.magic = 0;
    assert!(matches!(
        wrong_magic.validate(),
        Err(EngineError::Invalid(_))
    ));
    let mut too_small = PotlibcConfig::new("brh2leps", 0).expect("config");
    too_small.size = POTLIBC_CONFIG_MIN_SIZE - 8;
    assert!(matches!(too_small.validate(), Err(EngineError::Invalid(_))));
}

#[test]
fn coupled_well_honours_the_engine_identity_contract() {
    let library = unsafe { libloading::Library::new(build_coupled_well()) }.expect("open engine");
    let available: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"rgpot_engine_available\0") }.expect("available symbol");
    let stamp: libloading::Symbol<unsafe extern "C" fn() -> RgpotEngineAbiStamp> =
        unsafe { library.get(b"rgpot_engine_abi_stamp\0") }.expect("stamp symbol");
    let compatible: libloading::Symbol<unsafe extern "C" fn(*const RgpotEngineAbiStamp) -> c_int> =
        unsafe { library.get(b"rgpot_engine_abi_compatible\0") }.expect("compatible symbol");
    assert_eq!(unsafe { available() }, 1);
    let engine = unsafe { stamp() };
    let host = RgpotEngineAbiStamp {
        abi_major: ENGINE_ABI_VERSION as u16,
        abi_minor: ENGINE_ABI_MINOR,
        layout_revision: ENGINE_ABI_LAYOUT_REVISION,
    };
    assert_eq!(engine, host, "the engine and the loader share one header");

    // The documented rule: equal major and layout, engine minor at least the host's.
    let accepts = |host: RgpotEngineAbiStamp| unsafe { compatible(&host) } != 0;
    assert!(accepts(host));
    assert!(accepts(RgpotEngineAbiStamp {
        abi_minor: 0,
        ..host
    }));
    assert!(!accepts(RgpotEngineAbiStamp {
        abi_minor: host.abi_minor + 1,
        ..host
    }));
    assert!(!accepts(RgpotEngineAbiStamp {
        abi_major: host.abi_major + 1,
        ..host
    }));
    assert!(!accepts(RgpotEngineAbiStamp {
        layout_revision: host.layout_revision + 1,
        ..host
    }));
    assert_eq!(unsafe { compatible(std::ptr::null()) }, 0);
}
