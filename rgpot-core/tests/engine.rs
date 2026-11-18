#![cfg(unix)]

use std::ffi::c_int;
use std::path::{Path, PathBuf};

use rgpot_core::engine::{
    EngineError, EnginePotential, EngineRequest, RgpotEngineAbiStamp, ENGINE_ABI_LAYOUT_REVISION,
    ENGINE_ABI_MINOR, ENGINE_ABI_VERSION, ENGINE_ERR_UNSUPPORTED,
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
fn abi_minor_rule_accepts_unstamped_and_minors_0_2_3_4_and_rejects_minor_5() {
    let request = EngineRequest {
        positions: &[1.0, 0.0, 0.0],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };

    // Unstamped engine explicitly means minor 0, layout 1: accepted under 0 <= 4.
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

    // Stamped minor 0: accepted under 0 <= 4.
    let minor0 = build(
        "harmonic_minor0_stamped",
        &["FIXTURE_MINOR=0", "NO_CAPS", "NO_HVP"],
    );
    let mut engine = load(&minor0);
    let eval = engine.force(&request).expect("stamped minor 0 force");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING, 3));

    // Stamped minor 2 without minor-4 HVP symbol: accepted under 2 <= 4; HVP is Unsupported.
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

    // Stamped minor 2 exporting HVP: accepted under 2 <= 4; symbol presence discovers HVP.
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

    // Stamped minor 3: accepted under 3 <= 4.
    let minor3 = build(
        "harmonic_minor3",
        &["FIXTURE_MINOR=3", "CAPS_SIZE_BEFORE_HVP"],
    );
    let mut engine = load(&minor3);
    let eval = engine.force(&request).expect("stamped minor 3 force");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING, 3));
    let product = engine
        .hvp(&request, &[2.0, 0.0, -1.0])
        .expect("stamped minor 3 hvp");
    assert!(all_within_roundoff(&product, &[2.0 * SPRING, 0.0, -SPRING]));

    // Stamped minor 4: accepted under 4 <= 4.
    let minor4 = build("harmonic_minor4", &["FIXTURE_MINOR=4"]);
    let mut engine = load(&minor4);
    let eval = engine.force(&request).expect("stamped minor 4 force");
    assert!(within_roundoff(eval.energy, 0.5 * SPRING, 3));
    let product = engine
        .hvp(&request, &[2.0, 0.0, -1.0])
        .expect("stamped minor 4 hvp");
    assert!(all_within_roundoff(&product, &[2.0 * SPRING, 0.0, -SPRING]));

    // Stamped minor 5: rejected because 5 > ENGINE_ABI_MINOR (4).
    let minor5 = build("harmonic_minor5", &["FIXTURE_MINOR=5"]);
    let before = created(&minor5);
    let error = unsafe { EnginePotential::load(&minor5, &config()) }
        .err()
        .expect("minor 5 must be rejected by minor 4 loader");
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
                abi_minor: 5,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            }),
        }
    );
    assert_eq!(created(&minor5), before);
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
