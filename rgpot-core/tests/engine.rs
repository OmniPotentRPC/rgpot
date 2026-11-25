#![cfg(unix)]

use std::ffi::{c_int, c_long};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[cfg(feature = "schema")]
use rgpot_core::engine::builtin_params;
use rgpot_core::engine::{
    AcquireReason, CoordTransformFn, CoordinateTransform, CoupledWellConfig, DerivativeReport,
    EngineCallbacks, EngineCaps, EngineError, EngineEvaluation, EngineFailure, EnginePotential,
    EngineRequest, EngineSession, EngineStatus, EngineSymbols, GradientJet, JetBatchRequest,
    PotlibcConfig, RgpotEngineAbiStamp, RgpotEngineCaps, RgpotEngineComponent, RgpotEnginePot,
    COUPLED_WELL_CONFIG_MAGIC, COUPLED_WELL_CONFIG_MIN_SIZE, COUPLED_WELL_CONFIG_V1_SIZE,
    COUPLED_WELL_CONFIG_VERSION, COUPLED_WELL_CONFIG_VERSION_V1, ENGINE_ABI_LAYOUT_REVISION,
    ENGINE_ABI_MINOR, ENGINE_ABI_VERSION, ENGINE_ERR_INVALID, ENGINE_ERR_UNSUPPORTED,
    ENGINE_UNLOAD_POLICY_MINOR, POTLIBC_CONFIG_MAGIC, POTLIBC_CONFIG_MIN_SIZE,
    POTLIBC_CONFIG_V1_SIZE, POTLIBC_CONFIG_VERSION, POTLIBC_CONFIG_VERSION_V1,
    POTLIBC_SURFACE_ID_CAPACITY,
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

#[cfg(feature = "cpmd-fixture")]
#[derive(Debug)]
enum CpmdFixtureError {
    MissingEnvironment(&'static str),
    Read(std::io::Error),
    Engine(EngineError),
}

#[cfg(feature = "cpmd-fixture")]
impl std::fmt::Display for CpmdFixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingEnvironment(name) => {
                write!(f, "required CPMD fixture variable {name} is not set")
            }
            Self::Read(error) => write!(f, "read CPMD fixture parameters: {error}"),
            Self::Engine(error) => error.fmt(f),
        }
    }
}

#[cfg(feature = "cpmd-fixture")]
impl std::error::Error for CpmdFixtureError {}

#[cfg(feature = "cpmd-fixture")]
impl From<std::io::Error> for CpmdFixtureError {
    fn from(error: std::io::Error) -> Self {
        Self::Read(error)
    }
}

#[cfg(feature = "cpmd-fixture")]
impl From<EngineError> for CpmdFixtureError {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

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
        // nextest runs each test in its own process, so a shared path would
        // let one process load the library while another rewrites it.
        let out = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("libcoupled_well.{}.{suffix}", std::process::id()));
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

/// Compile `engines/potlibc_engine.c` against a minimal C `potlibc` stub so
/// the Rust test suite exercises the real potlibc engine plugin directly.
fn build_potlibc() -> &'static Path {
    static LIBRARY: OnceLock<PathBuf> = OnceLock::new();
    LIBRARY.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let source = root.join("engines/potlibc_engine.c");
        let tmpdir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("potlibc_stub.{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).expect("create potlibc stub dir");
        let header = tmpdir.join("potlibc.h");
        let stub = tmpdir.join("potlibc_stub.c");
        std::fs::write(
            &header,
            r#"#ifndef POTLIBC_H
#define POTLIBC_H
#include <stddef.h>
#include <stdint.h>
typedef struct potlibc_handle potlibc_handle;
typedef enum potlibc_status {
  POTLIBC_SUCCESS = 0,
  POTLIBC_ERROR_INVALID_SURFACE_ID = 2,
  POTLIBC_ERROR_ATOM_COUNT_REQUIRED = 11,
  POTLIBC_ERROR_ATOM_COUNT_UNEXPECTED = 12
} potlibc_status;
typedef enum potlibc_capability {
  POTLIBC_CAP_ENERGY = 1u << 0,
  POTLIBC_CAP_FORCES = 1u << 1,
  POTLIBC_CAP_HVP = 1u << 2
} potlibc_capability;
typedef struct potlibc_metadata {
  const char *surface_id;
  size_t atom_count;
  const int *atomic_numbers;
  const double *reference_geometry;
  const char *atom_ordering;
  const char *length_unit;
  const char *energy_unit;
  const char *force_unit;
  uint32_t capabilities;
} potlibc_metadata;
potlibc_status potlibc_create(const char *surface_id, potlibc_handle **handle);
potlibc_status potlibc_create_sized(const char *surface_id, size_t atom_count,
                                    potlibc_handle **handle);
void potlibc_destroy(potlibc_handle *handle);
potlibc_status potlibc_get_metadata(const potlibc_handle *handle,
                                    potlibc_metadata *metadata);
potlibc_status potlibc_evaluate(potlibc_handle *handle, size_t count,
                                const double *x, double *energy, double *force);
potlibc_status potlibc_hessian_vector_product(potlibc_handle *handle,
                                              size_t count, const double *x,
                                              const double *direction,
                                              double *product);
const char *potlibc_status_string(potlibc_status status);
#endif
"#,
        )
        .expect("write potlibc.h stub");
        std::fs::write(
            &stub,
            r#"#include "potlibc.h"
#include <stdlib.h>
#include <string.h>
struct potlibc_handle {
  potlibc_metadata meta;
  int species[8];
};
static const int BRH2_SPECIES[3] = {35, 1, 1};
potlibc_status potlibc_create(const char *surface_id, potlibc_handle **out) {
  potlibc_handle *h;
  if (strcmp(surface_id, "vbo-beta1") == 0) {
    return POTLIBC_ERROR_ATOM_COUNT_REQUIRED;
  }
  if (strcmp(surface_id, "brh2leps") != 0) {
    return POTLIBC_ERROR_INVALID_SURFACE_ID;
  }
  h = (potlibc_handle *)calloc(1, sizeof *h);
  h->meta.surface_id = "brh2leps";
  h->meta.atom_count = 3;
  h->meta.atomic_numbers = BRH2_SPECIES;
  h->meta.length_unit = "bohr";
  h->meta.energy_unit = "hartree";
  h->meta.force_unit = "hartree/bohr";
  h->meta.capabilities =
      POTLIBC_CAP_ENERGY | POTLIBC_CAP_FORCES | POTLIBC_CAP_HVP;
  *out = h;
  return POTLIBC_SUCCESS;
}
potlibc_status potlibc_create_sized(const char *surface_id, size_t atom_count,
                                    potlibc_handle **out) {
  potlibc_handle *h;
  size_t i;
  if (strcmp(surface_id, "brh2leps") == 0) {
    return POTLIBC_ERROR_ATOM_COUNT_UNEXPECTED;
  }
  if (strcmp(surface_id, "vbo-beta1") != 0 || atom_count < 2 || atom_count > 8) {
    return POTLIBC_ERROR_INVALID_SURFACE_ID;
  }
  h = (potlibc_handle *)calloc(1, sizeof *h);
  for (i = 0; i < atom_count; ++i) {
    h->species[i] = 13;
  }
  h->meta.surface_id = "vbo-beta1";
  h->meta.atom_count = atom_count;
  h->meta.atomic_numbers = h->species;
  h->meta.length_unit = "bohr";
  h->meta.energy_unit = "hartree";
  h->meta.force_unit = "hartree/bohr";
  h->meta.capabilities =
      POTLIBC_CAP_ENERGY | POTLIBC_CAP_FORCES | POTLIBC_CAP_HVP;
  *out = h;
  return POTLIBC_SUCCESS;
}
void potlibc_destroy(potlibc_handle *handle) { free(handle); }
potlibc_status potlibc_get_metadata(const potlibc_handle *handle,
                                    potlibc_metadata *metadata) {
  *metadata = handle->meta;
  return POTLIBC_SUCCESS;
}
potlibc_status potlibc_evaluate(potlibc_handle *handle, size_t count,
                                const double *x, double *energy,
                                double *force) {
  const size_t dim = 3 * handle->meta.atom_count;
  size_t s, i;
  for (s = 0; s < count; ++s) {
    double e = 0.0;
    for (i = 0; i < dim; ++i) {
      const double xi = x[s * dim + i];
      e += 0.5 * xi * xi;
      force[s * dim + i] = -xi;
    }
    energy[s] = e;
  }
  return POTLIBC_SUCCESS;
}
potlibc_status potlibc_hessian_vector_product(potlibc_handle *handle,
                                              size_t count, const double *x,
                                              const double *direction,
                                              double *product) {
  const size_t total = count * 3 * handle->meta.atom_count;
  size_t i;
  (void)x;
  for (i = 0; i < total; ++i) {
    product[i] = direction[i];
  }
  return POTLIBC_SUCCESS;
}
const char *potlibc_status_string(potlibc_status status) {
  switch (status) {
  case POTLIBC_SUCCESS:
    return "success";
  case POTLIBC_ERROR_ATOM_COUNT_REQUIRED:
    return "atom count required";
  case POTLIBC_ERROR_ATOM_COUNT_UNEXPECTED:
    return "atom count unexpected";
  default:
    return "invalid surface id";
  }
}
"#,
        )
        .expect("write potlibc_stub.c");
        let suffix = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let out = tmpdir.join(format!("libpotlibc.{suffix}"));
        let target = env!("RGPOT_BUILD_TARGET");
        let compiler = match cc::Build::new()
            .target(target)
            .host(target)
            .opt_level(0)
            .try_get_compiler()
        {
            Ok(compiler) => compiler,
            Err(why) => panic!(
                "cannot compile the potlibc engine: no C compiler available for target {target} ({why})"
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
            .arg(&tmpdir)
            .arg("-I")
            .arg(root.join("CppCore"))
            .arg("-I")
            .arg(root.join("include"))
            .arg("-o")
            .arg(&out)
            .arg(&source)
            .arg(&stub)
            .arg("-lpthread")
            .status()
            .unwrap_or_else(|why| panic!("run the C compiler {:?}: {why}", compiler.path()));
        assert!(status.success(), "the potlibc engine did not compile: {status}");
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

/// Opens the engine library named by `RGPOT_TEST_ENGINE` with an empty
/// configuration and checks that the host accepts its ABI stamp.
///
/// Whether `rgpot_engine_create` then accepts the empty message is the
/// engine's own choice: the Lennard-Jones plugin builds its documented default
/// potential, while a backend that needs a params message refuses with
/// `EngineError::Create`. Any other error means the loader rejected the library
/// before create, which is the failure this test exists to catch.
///
/// The meson test `engine_abi_stamp_rust_loader` runs it with
/// `--ignored` against the Lennard-Jones plugin built in the same tree.
#[test]
#[ignore = "needs RGPOT_TEST_ENGINE to name a generic engine library"]
fn configured_external_engine_passes_the_abi_check() {
    let path = std::env::var_os("RGPOT_TEST_ENGINE")
        .expect("RGPOT_TEST_ENGINE names the engine library to open");
    let path = Path::new(&path);
    assert_eq!(
        RgpotEngineAbiStamp::HOST,
        RgpotEngineAbiStamp {
            abi_major: 1,
            abi_minor: 11,
            layout_revision: 1,
        }
    );
    match unsafe { EnginePotential::load(path, &[]) } {
        Ok(engine) => assert_eq!(
            engine.abi_stamp(),
            RgpotEngineAbiStamp::HOST,
            "{} reports another ABI stamp",
            path.display()
        ),
        Err(EngineError::Create(_)) => {}
        Err(error) => panic!(
            "rgpot host rejected {} before create: {error}",
            path.display()
        ),
    }
}

const POTLIBC_ENGINE_ENV: &str = "RGPOT_TEST_POTLIBC_ENGINE";
const POTLIBC_FAIL_LOCK_ENV: &str = "RGPOT_TEST_POTLIBC_ENGINE_FAIL_LOCK";
const BOHR_TO_ANGSTROM: f64 = 0.529177210903;

/// `RgpotPotlibcConfigV1` of `rgpot/potlibc_engine.h`.
#[repr(C)]
struct PotlibcConfigV1 {
    magic: u32,
    version: u32,
    atom_count: u64,
    surface_id: [u8; 128],
}

fn potlibc_config(surface_id: &str) -> Vec<u8> {
    let mut message = PotlibcConfigV1 {
        magic: 0x504F_544C,
        version: 1,
        atom_count: 0,
        surface_id: [0; 128],
    };
    message.surface_id[..surface_id.len()].copy_from_slice(surface_id.as_bytes());
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (&message as *const PotlibcConfigV1).cast::<u8>(),
            std::mem::size_of::<PotlibcConfigV1>(),
        )
    };
    bytes.to_vec()
}

fn potlibc_library(variable: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(variable)
            .unwrap_or_else(|| panic!("{variable} names a built libpotlibc_engine")),
    )
}

/// The BrH2 LEPS surface in Angstrom, the first atom at the origin.
fn brh2_positions() -> [f64; 9] {
    let bohr = [0.0, 0.0, 0.0, 3.0802, 0.0, 0.0, 4.8202, 0.0, 0.0];
    bohr.map(|value| value * BOHR_TO_ANGSTROM)
}

/// Loads the potlibc engine through the generic loader and evaluates the
/// `brh2leps` surface.
///
/// The energy is the reference value of the plugin's own regression test, and
/// the forces are checked against a central difference of the energy. The
/// meson test `potlibc_engine_rust_loader` runs it with `--ignored` against
/// the plugin built in the same tree.
#[test]
#[ignore = "needs RGPOT_TEST_POTLIBC_ENGINE to name libpotlibc_engine"]
fn potlibc_engine_evaluates_through_the_rust_loader() {
    assert_eq!(std::mem::size_of::<PotlibcConfigV1>(), 144);
    let path = potlibc_library(POTLIBC_ENGINE_ENV);
    let mut engine = unsafe { EnginePotential::load(&path, &potlibc_config("brh2leps")) }
        .expect("the host loads the potlibc engine");
    assert_eq!(engine.abi_stamp(), RgpotEngineAbiStamp::HOST);
    assert!(engine.symbols().available.is_some());
    assert_eq!(engine.caps().expect("caps").reentrancy, 1);

    let positions = brh2_positions();
    let numbers = [35, 1, 1];
    let cell = [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0];
    let evaluation = engine
        .force(&EngineRequest {
            positions: &positions,
            atomic_numbers: &numbers,
            box_matrix: &cell,
        })
        .expect("potlibc evaluates the brh2leps surface");
    assert_eq!(evaluation.forces.len(), 9);
    assert!(
        (evaluation.energy - f64::from_bits(0x3fdf_2d7a_1092_72ec)).abs() < 1e-9,
        "energy {} differs from the reference",
        evaluation.energy
    );

    let step = 1e-5;
    for component in [0_usize, 3, 6] {
        let mut energy_at = |offset: f64| {
            let mut shifted = positions;
            shifted[component] += offset;
            engine
                .force(&EngineRequest {
                    positions: &shifted,
                    atomic_numbers: &numbers,
                    box_matrix: &cell,
                })
                .expect("displaced evaluation")
                .energy
        };
        let gradient = (energy_at(step) - energy_at(-step)) / (2.0 * step);
        assert!(
            (evaluation.forces[component] + gradient).abs() < 1e-6,
            "force component {component} is {} but -dE/dx is {}",
            evaluation.forces[component],
            -gradient
        );
    }

    let wrong_count = engine
        .force(&EngineRequest {
            positions: &positions[..6],
            atomic_numbers: &numbers[..2],
            box_matrix: &cell,
        })
        .unwrap_err();
    assert!(
        matches!(wrong_count, EngineError::Recoverable { code: 1, .. }),
        "{wrong_count}"
    );
    engine
        .force(&EngineRequest {
            positions: &positions,
            atomic_numbers: &numbers,
            box_matrix: &cell,
        })
        .expect("a recoverable refusal leaves the handle usable");
}

/// Drives the lock-failure build of the potlibc engine, which fails every
/// handle lock acquisition while `rgpot_potlibc_test_set_lock_fail(1)` is set.
/// The Rust loader maps the code to `EngineError::Lock`, latches it, keeps the
/// handle unusable after locking works again, and the plugin still releases its
/// call mutex when the handle is dropped while locking fails.
#[test]
#[ignore = "needs RGPOT_TEST_POTLIBC_ENGINE_FAIL_LOCK to name the lock-failure build"]
fn potlibc_lock_failure_is_a_latched_lock_error_through_the_rust_loader() {
    use rgpot_core::engine::ENGINE_ERR_LOCK;

    let path = potlibc_library(POTLIBC_FAIL_LOCK_ENV);
    let mut engine = unsafe { EnginePotential::load(&path, &potlibc_config("brh2leps")) }
        .expect("the host loads the lock-failure build");
    let observer = unsafe { libloading::Library::new(&path) }.expect("reopen the plugin");
    let set_fail: libloading::Symbol<unsafe extern "C" fn(c_int)> =
        unsafe { observer.get(b"rgpot_potlibc_test_set_lock_fail\0") }
            .expect("lock failure switch");
    let live: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { observer.get(b"rgpot_potlibc_test_live_call_locks\0") }.expect("mutex counter");
    assert_eq!(unsafe { live() }, 1);

    let positions = brh2_positions();
    let request = EngineRequest {
        positions: &positions,
        atomic_numbers: &[35, 1, 1],
        box_matrix: &[20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0],
    };
    engine
        .force(&request)
        .expect("an evaluation before the failure");

    unsafe { set_fail(1) };
    let first = engine.force(&request).unwrap_err();
    assert!(
        matches!(
            &first,
            EngineError::Lock { operation: "rgpot_engine_force", message }
                if message == "potlibc engine handle lock failed"
        ),
        "{first}"
    );
    assert_eq!(first.engine_status(), EngineStatus::fatal(ENGINE_ERR_LOCK));

    unsafe { set_fail(0) };
    assert_eq!(engine.force(&request).unwrap_err(), first);
    assert_eq!(engine.caps().unwrap_err(), first);

    unsafe { set_fail(1) };
    drop(engine);
    unsafe { set_fail(0) };
    assert_eq!(
        unsafe { live() },
        0,
        "the call mutex was not released by a destroy whose lock failed"
    );
}

fn load_session(path: &Path) -> EngineSession {
    // SAFETY: `path` names the fixture compiled from the ABI header above.
    unsafe { EngineSession::load(path, &config()) }.expect("load the harmonic engine session")
}

struct FixtureObserver(libloading::Library);

impl FixtureObserver {
    fn open(path: &Path) -> Self {
        Self(unsafe { libloading::Library::new(path) }.expect("open the fixture observer"))
    }

    fn created(&self) -> c_int {
        let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
            unsafe { self.0.get(b"fixture_create_count\0") }.expect("create count symbol");
        unsafe { count() }
    }

    fn destroyed(&self) -> c_int {
        let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
            unsafe { self.0.get(b"fixture_destroy_count\0") }.expect("count symbol");
        unsafe { count() }
    }

    fn native_calls(&self) -> c_int {
        let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
            unsafe { self.0.get(b"fixture_native_call_count\0") }
                .expect("native call count symbol");
        unsafe { count() }
    }
}

#[cfg(target_os = "linux")]
fn library_is_mapped(path: &Path) -> bool {
    let canonical = path.canonicalize().expect("canonical fixture path");
    let maps = std::fs::read_to_string("/proc/self/maps").expect("read /proc/self/maps");
    maps.lines()
        .filter_map(|line| line.split_whitespace().last())
        .any(|mapped| Path::new(mapped) == canonical)
}

fn native_calls(path: &Path) -> c_int {
    FixtureObserver::open(path).native_calls()
}

fn batch_calls(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_batch_call_count\0") }.expect("batch call count symbol");
    unsafe { count() }
}

fn null_callback_clears(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_null_callback_clear_count\0") }
            .expect("null callback clear count symbol");
    unsafe { count() }
}

fn last_oracle_rc(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let value: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_last_oracle_rc\0") }.expect("oracle code symbol");
    unsafe { value() }
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

struct Shift;

impl CoordinateTransform for Shift {
    fn transform(
        &mut self,
        positions: &mut [f64],
        _box_matrix: &mut [f64; 9],
    ) -> Result<(), EngineFailure> {
        for value in positions {
            *value += 1.0;
        }
        Ok(())
    }
}

#[test]
fn session_transform_changes_only_engine_scratch() {
    let path = build("harmonic_session_transform", &[]);
    let mut session = load_session(&path);
    let positions = [1.0, 2.0, 3.0];
    let mut shift = Shift;
    let answer = session
        .force(
            &EngineRequest {
                positions: &positions,
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            Some(&mut shift),
        )
        .expect("transformed force");
    assert_eq!(positions, [1.0, 2.0, 3.0]);
    assert_eq!(answer.energy.to_bits(), 36.25f64.to_bits());
    assert_eq!(answer.forces, [-5.0, -7.5, -10.0]);
}

#[derive(Default)]
struct CallbackRecord {
    acquisitions: Vec<AcquireReason>,
    states: Vec<u64>,
}

struct Oracle {
    record: Arc<Mutex<CallbackRecord>>,
}

impl EngineCallbacks for Oracle {
    fn oracle(&mut self, request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        Ok(EngineEvaluation {
            energy: request.positions.iter().sum(),
            forces: request.positions.iter().map(|value| -*value).collect(),
        })
    }

    fn on_acquire(&mut self, reason: AcquireReason) {
        self.record
            .lock()
            .expect("callback record")
            .acquisitions
            .push(reason);
    }

    fn on_model_update(&mut self, model_state: u64) {
        self.record
            .lock()
            .expect("callback record")
            .states
            .push(model_state);
    }
}

#[test]
fn session_owns_callbacks_and_reports_model_state() {
    let path = build("harmonic_session_callbacks", &[]);
    let record = Arc::new(Mutex::new(CallbackRecord::default()));
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(Oracle {
            record: Arc::clone(&record),
        })))
        .expect("set callbacks");
    let answer = session
        .force(
            &EngineRequest {
                positions: &[1.0, 2.0, 3.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect("oracle force");
    assert_eq!(answer.energy, 6.0);
    assert_eq!(answer.forces, [-1.0, -2.0, -3.0]);
    assert_eq!(session.model_state().expect("model state"), 1);
    let record = record.lock().expect("callback record");
    assert_eq!(record.acquisitions, [AcquireReason::Variance]);
    assert_eq!(record.states, [1]);
}

#[test]
fn failed_callback_replacement_keeps_old_registration_and_none_passes_null() {
    let path = build("harmonic_callback_transaction", &[]);
    let old = Arc::new(Mutex::new(CallbackRecord::default()));
    let replacement = Arc::new(Mutex::new(CallbackRecord::default()));
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(Oracle {
            record: Arc::clone(&old),
        })))
        .expect("set initial callbacks");

    let error = session
        .set_callbacks(Some(Box::new(Oracle {
            record: Arc::clone(&replacement),
        })))
        .expect_err("fixture refuses callback replacement");
    assert_eq!(
        error,
        EngineError::Recoverable {
            code: 2,
            message: "callback replacement refused".into(),
        }
    );
    session
        .force(
            &EngineRequest {
                positions: &[1.0, 2.0, 3.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect("old oracle remains registered");
    assert_eq!(
        old.lock().expect("old callback record").acquisitions,
        [AcquireReason::Variance]
    );
    assert!(replacement
        .lock()
        .expect("replacement callback record")
        .acquisitions
        .is_empty());

    let before = null_callback_clears(&path);
    session.set_callbacks(None).expect("clear callbacks");
    assert_eq!(null_callback_clears(&path), before + 1);
    let answer = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect("harmonic force after clearing callbacks");
    assert_eq!(answer.energy.to_bits(), 1.25f64.to_bits());
}

/// An oracle that reports when it is dropped, and can panic when it is.
struct Tracked {
    dropped: Arc<AtomicBool>,
    oracle_calls: Arc<AtomicUsize>,
    panic_on_drop: bool,
}

impl Tracked {
    fn new(panic_on_drop: bool) -> (Self, Arc<AtomicBool>, Arc<AtomicUsize>) {
        let dropped = Arc::new(AtomicBool::new(false));
        let oracle_calls = Arc::new(AtomicUsize::new(0));
        let tracked = Self {
            dropped: Arc::clone(&dropped),
            oracle_calls: Arc::clone(&oracle_calls),
            panic_on_drop,
        };
        (tracked, dropped, oracle_calls)
    }
}

impl EngineCallbacks for Tracked {
    fn oracle(&mut self, request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        self.oracle_calls.fetch_add(1, Ordering::SeqCst);
        Ok(EngineEvaluation {
            energy: request.positions.iter().sum(),
            forces: request.positions.iter().map(|value| -*value).collect(),
        })
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
        if self.panic_on_drop && !std::thread::panicking() {
            panic!("callback destructor panic for fixture");
        }
    }
}

fn force_one(session: &mut EngineSession) -> Result<EngineEvaluation, EngineError> {
    session.force(
        &EngineRequest {
            positions: &[1.0, 2.0, 3.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        },
        None,
    )
}

#[test]
fn a_panicking_destructor_of_replaced_callbacks_does_not_unwind_out_of_the_swap() {
    let path = build(
        "harmonic_callback_replacement_destructor",
        &["ACCEPT_CALLBACK_REPLACEMENT"],
    );
    let (old, old_dropped, _) = Tracked::new(true);
    let (new, new_dropped, new_calls) = Tracked::new(false);
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(old)))
        .expect("set initial callbacks");

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        session.set_callbacks(Some(Box::new(new)))
    }));
    let error = outcome
        .expect("the destructor panic must not unwind out of set_callbacks")
        .expect_err("a panicking destructor is reported");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "callback destructor panicked".into(),
        }
    );
    assert!(old_dropped.load(Ordering::SeqCst));
    assert!(
        !new_dropped.load(Ordering::SeqCst),
        "the registered replacement must stay owned by the session"
    );
    assert_eq!(force_one(&mut session), Err(error));
    assert_eq!(new_calls.load(Ordering::SeqCst), 0);

    let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(session)));
    assert!(dropped.is_ok(), "dropping the session must not panic");
    assert!(new_dropped.load(Ordering::SeqCst));
}

#[test]
fn clearing_callbacks_whose_destructor_panics_is_contained_and_reported() {
    let path = build("harmonic_callback_clear_destructor", &[]);
    let (old, old_dropped, _) = Tracked::new(true);
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(old)))
        .expect("set initial callbacks");

    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.set_callbacks(None)));
    let error = outcome
        .expect("the destructor panic must not unwind out of set_callbacks")
        .expect_err("a panicking destructor is reported");
    assert!(matches!(error, EngineError::Fatal { code: -1, .. }));
    assert!(old_dropped.load(Ordering::SeqCst));
    assert_eq!(null_callback_clears(&path), 1);
}

#[test]
fn a_refused_callback_context_stays_alive_until_the_engine_is_destroyed() {
    let path = build(
        "harmonic_callback_refused_retained",
        &["RETAIN_REFUSED_CALLBACKS"],
    );
    let observer = FixtureObserver::open(&path);
    let (old, old_dropped, old_calls) = Tracked::new(false);
    let (refused, refused_dropped, refused_calls) = Tracked::new(true);
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(old)))
        .expect("set initial callbacks");
    session
        .set_callbacks(Some(Box::new(refused)))
        .expect_err("the fixture refuses the replacement");

    assert!(
        !refused_dropped.load(Ordering::SeqCst),
        "an engine that kept the refused context must not see it freed"
    );
    force_one(&mut session).expect("the engine evaluates through the context it kept");
    assert_eq!(old_calls.load(Ordering::SeqCst), 0);
    assert_eq!(refused_calls.load(Ordering::SeqCst), 1);
    assert!(!old_dropped.load(Ordering::SeqCst));

    let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(session)));
    assert!(
        dropped.is_ok(),
        "a panicking destructor must not escape the drop"
    );
    assert_eq!(observer.destroyed(), 1);
    assert!(refused_dropped.load(Ordering::SeqCst));
    assert!(old_dropped.load(Ordering::SeqCst));
}

struct PanicOracle;

impl EngineCallbacks for PanicOracle {
    fn oracle(&mut self, _request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        panic!("oracle panic for fixture")
    }
}

struct PanicModelUpdate;

impl EngineCallbacks for PanicModelUpdate {
    fn oracle(&mut self, request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        Ok(EngineEvaluation {
            energy: 0.0,
            forces: vec![0.0; request.positions.len()],
        })
    }

    fn on_model_update(&mut self, _model_state: u64) {
        panic!("model update panic for fixture")
    }
}

struct PanicTransform;

impl CoordinateTransform for PanicTransform {
    fn transform(
        &mut self,
        _positions: &mut [f64],
        _box_matrix: &mut [f64; 9],
    ) -> Result<(), EngineFailure> {
        panic!("transform panic for fixture")
    }
}

/// Every later call on a poisoned session returns `expected`, code and
/// message, without reaching the engine.
fn assert_poisoned_without_an_extra_force(
    session: &mut EngineSession,
    path: &Path,
    expected: &EngineError,
) {
    assert!(
        matches!(expected, EngineError::Fatal { .. }),
        "{expected:?}"
    );
    let observer = FixtureObserver::open(path);
    let before = observer.native_calls();
    for _ in 0..2 {
        let error = session
            .force(
                &EngineRequest {
                    positions: &[1.0, 0.0, 0.0],
                    atomic_numbers: &[1],
                    box_matrix: &CELL,
                },
                None,
            )
            .expect_err("fatal session remains poisoned");
        assert_eq!(&error, expected);
    }
    assert_eq!(session.model_state().as_ref(), Err(expected));
    assert_eq!(session.hint(&[1]).as_ref(), Err(expected));
    assert_eq!(observer.native_calls(), before);
}

#[test]
fn oracle_panic_is_fatal_and_poisons_the_session() {
    let path = build("harmonic_oracle_panic", &[]);
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(PanicOracle)))
        .expect("set panic callback");
    let error = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect_err("oracle panic is fatal");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "oracle callback panicked".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

#[test]
fn transform_panic_is_fatal_and_poisons_the_session() {
    let path = build("harmonic_transform_panic", &[]);
    let mut session = load_session(&path);
    let error = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            Some(&mut PanicTransform),
        )
        .expect_err("transform panic is fatal");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "coordinate transform panicked".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

#[test]
fn hint_reports_status_updates_state_and_surfaces_callback_panics() {
    let path = build("harmonic_hint", &[]);
    let mut session = load_session(&path);
    session.hint(&[1]).expect("valid hint");
    assert_eq!(session.model_state().expect("model state"), 1);
    let error = session.hint(&[0xff]).expect_err("fixture refuses hint");
    assert_eq!(
        error,
        EngineError::Recoverable {
            code: 2,
            message: "hint refused".into(),
        }
    );
    assert_eq!(session.model_state().expect("model state"), 1);

    let panic_path = build("harmonic_hint_callback_panic", &[]);
    let mut panic_session = load_session(&panic_path);
    panic_session
        .set_callbacks(Some(Box::new(PanicModelUpdate)))
        .expect("set model-update callback");
    let error = panic_session
        .hint(&[1])
        .expect_err("model-update callback panic is fatal");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "model-update callback panicked".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut panic_session, &panic_path, &error);
}

#[test]
fn a_panic_in_a_retained_refused_callback_is_fatal() {
    let path = build(
        "harmonic_refused_model_update_panic",
        &["RETAIN_REFUSED_CALLBACKS"],
    );
    let (old, _, _) = Tracked::new(false);
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(old)))
        .expect("set initial callbacks");
    session
        .set_callbacks(Some(Box::new(PanicModelUpdate)))
        .expect_err("the fixture refuses the replacement");

    let error = session
        .hint(&[1])
        .expect_err("the retained refused callback panics in the model update");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "model-update callback panicked".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

#[test]
fn a_panic_in_a_retained_refused_oracle_is_fatal() {
    let path = build(
        "harmonic_refused_oracle_panic",
        &["RETAIN_REFUSED_CALLBACKS"],
    );
    let (old, _, _) = Tracked::new(false);
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(old)))
        .expect("set initial callbacks");
    session
        .set_callbacks(Some(Box::new(PanicOracle)))
        .expect_err("the fixture refuses the replacement");

    let error = force_one(&mut session).expect_err("the retained refused oracle panics");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "oracle callback panicked".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

#[test]
fn native_fatal_status_poisons_an_engine_session() {
    let path = build("harmonic_session_fatal_once", &["FATAL_ONCE"]);
    let observer = FixtureObserver::open(&path);
    let mut session = load_session(&path);
    let error = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect_err("fixture fires one fatal status");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -7,
            message: "fatal once".into(),
        }
    );
    let before = observer.native_calls();
    let later = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect_err("session remains poisoned");
    assert_eq!(later, error);
    assert_eq!(observer.native_calls(), before);
}

#[test]
fn typed_status_does_not_retry_from_error_text() {
    assert_eq!(EngineStatus::from_code(0), EngineStatus::Success);
    assert!(matches!(
        EngineStatus::from_code(2),
        EngineStatus::RecoverableInputFailure(magnitude) if magnitude.get() == 2
    ));
    assert!(matches!(
        EngineStatus::from_code(-2),
        EngineStatus::FatalInstanceFailure(magnitude) if magnitude.get() == 2
    ));

    let path = build("harmonic_session_no_retry", &[]);
    let observer = FixtureObserver::open(&path);
    let mut session = load_session(&path);
    let before = observer.native_calls();
    let error = session
        .force(
            &EngineRequest {
                positions: &[0.0, 0.0, 0.0],
                atomic_numbers: &[0],
                box_matrix: &CELL,
            },
            None,
        )
        .expect_err("recoverable input");
    assert_eq!(error.engine_status(), Some(EngineStatus::from_code(1)));
    assert_eq!(observer.native_calls() - before, 2);
}

#[test]
fn status_codes_round_trip_in_range_and_saturate_beyond_it() {
    for code in [0, 1, -1, 2, -2, c_int::MAX, c_int::MIN] {
        assert_eq!(
            EngineStatus::from_code(EngineStatus::from_code(code).code()),
            EngineStatus::from_code(code)
        );
        assert_eq!(EngineStatus::from_code(code).code(), code);
    }
    let huge = NonZeroU32::new(u32::MAX).expect("nonzero");
    assert_eq!(
        EngineStatus::RecoverableInputFailure(huge).code(),
        c_int::MAX
    );
    assert_eq!(EngineStatus::FatalInstanceFailure(huge).code(), c_int::MIN);
}

unsafe extern "C" fn stub_abi_version() -> c_int {
    ENGINE_ABI_VERSION
}

unsafe extern "C" fn stub_available() -> c_int {
    1
}

unsafe extern "C" fn stub_create(
    _config: *const std::ffi::c_void,
    _config_len: usize,
    _error: *mut std::ffi::c_char,
    _error_len: usize,
) -> *mut RgpotEnginePot {
    std::ptr::null_mut()
}

unsafe extern "C" fn stub_destroy(_pot: *mut RgpotEnginePot) {}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn stub_force(
    _pot: *mut RgpotEnginePot,
    _n_atoms: c_long,
    _positions: *const f64,
    _atomic_numbers: *const c_int,
    _forces: *mut f64,
    _energy: *mut f64,
    _variance: *mut f64,
    _box: *const f64,
    _transform: Option<CoordTransformFn>,
    _user: *mut std::ffi::c_void,
) -> c_int {
    0
}

unsafe extern "C" fn stub_model_state(_pot: *const RgpotEnginePot) -> u64 {
    0
}

#[test]
fn symbols_are_built_with_a_constructor_and_optional_setters() {
    let required = EngineSymbols::new(stub_abi_version, stub_create, stub_destroy, stub_force);
    assert!(required.available.is_none());
    assert!(required.abi_stamp.is_none());
    assert!(required.force_batch.is_none());
    assert!(required.hvp.is_none());
    assert!(required.caps.is_none());
    assert!(required.last_error.is_none());
    assert!(required.last_components.is_none());
    assert!(required.component_name.is_none());
    assert!(required.model_state.is_none());
    assert!(required.set_callbacks.is_none());
    assert!(required.hint.is_none());

    let with_available = required.with_available(stub_available);
    assert!(with_available.available.is_some());
    assert!(with_available.model_state.is_none());

    let with_state = required.with_model_state(stub_model_state);
    assert!(with_state.model_state.is_some());
    assert!(with_state.set_callbacks.is_none());
    assert!(with_state.hint.is_none());
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
    let before = batch_calls(&path);
    let batch = engine
        .force_batch(&numbers, &positions, &boxes)
        .expect("batch");
    assert_eq!(batch_calls(&path), before + 1);
    assert_eq!(batch.len(), 2);
    for (system, one) in [first, second].iter().enumerate() {
        let single = engine
            .force(&EngineRequest {
                positions: one,
                atomic_numbers: &numbers,
                box_matrix: &CELL,
            })
            .expect("single");
        assert_eq!(batch[system].energy.to_bits(), single.energy.to_bits());
        assert_eq!(
            batch[system]
                .forces
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            single
                .forces
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[cfg(feature = "cpmd-fixture")]
#[test]
#[ignore = "needs RGPOT_ENGINE_CPMD_FIXTURE and RGPOT_ENGINE_CPMD_PARAMS; run with --ignored"]
fn cpmd_fixture_batch_preserves_single_call_bits_and_order() -> Result<(), CpmdFixtureError> {
    let engine_path = std::env::var_os("RGPOT_ENGINE_CPMD_FIXTURE").ok_or(
        CpmdFixtureError::MissingEnvironment("RGPOT_ENGINE_CPMD_FIXTURE"),
    )?;
    let params_path = std::env::var_os("RGPOT_ENGINE_CPMD_PARAMS").ok_or(
        CpmdFixtureError::MissingEnvironment("RGPOT_ENGINE_CPMD_PARAMS"),
    )?;
    let config = std::fs::read(params_path)?;
    let positions = [0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 1.5, -0.2, 0.3, 0.1, 0.4, -0.7];
    let numbers = [7, 1];
    let mut boxes = CELL.to_vec();
    boxes.extend_from_slice(&CELL);

    // SAFETY: this feature deliberately trusts the externally supplied CPMD
    // fixture to implement the versioned engine ABI.
    let mut singles = unsafe { EngineSession::load(Path::new(&engine_path), &config) }?;
    let first = singles.force(
        &EngineRequest {
            positions: &positions[..6],
            atomic_numbers: &numbers,
            box_matrix: &CELL,
        },
        None,
    )?;
    let second = singles.force(
        &EngineRequest {
            positions: &positions[6..],
            atomic_numbers: &numbers,
            box_matrix: &CELL,
        },
        None,
    )?;

    // SAFETY: the same trusted CPMD fixture is loaded into a fresh instance.
    let mut batch = unsafe { EngineSession::load(Path::new(&engine_path), &config) }?;
    let answers = batch.force_batch(&numbers, &positions, &boxes, None)?;
    assert_eq!(answers[0].energy.to_bits(), first.energy.to_bits());
    assert_eq!(answers[1].energy.to_bits(), second.energy.to_bits());
    assert_eq!(answers[0].forces, first.forces);
    assert_eq!(answers[1].forces, second.forces);
    Ok(())
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
}

#[test]
fn fatal_status_poisons_the_handle_and_makes_zero_later_native_calls() {
    let path = build("harmonic_fatal_once", &["FATAL_ONCE"]);
    let observer = FixtureObserver::open(&path);
    let before_destroy = observer.destroyed();
    let mut engine = load(&path);
    let expected = EngineError::Fatal {
        code: -7,
        message: "fatal once".into(),
    };
    let valid = EngineRequest {
        positions: &[1.0, 0.0, 0.0],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };

    assert_eq!(engine.force(&valid).unwrap_err(), expected);
    let after_fatal = observer.native_calls();
    assert!(after_fatal > 0);

    // The fixture returns -7 only once and 0 thereafter, so an unpoisoned
    // handle would both increment native_calls and return Ok here.
    assert_eq!(engine.force(&valid).unwrap_err(), expected);
    assert_eq!(
        engine
            .force(&EngineRequest {
                positions: &[0.0; 4],
                atomic_numbers: &[1, 1],
                box_matrix: &CELL,
            })
            .unwrap_err(),
        expected
    );
    assert_eq!(
        engine
            .force_batch(&[1], &[1.0, 0.0, 0.0], &CELL)
            .unwrap_err(),
        expected
    );
    assert_eq!(engine.hvp(&valid, &[1.0, 0.0, 0.0]).unwrap_err(), expected);
    assert_eq!(engine.caps().unwrap_err(), expected);
    assert_eq!(observer.native_calls(), after_fatal);

    let potential = engine.into_potential();
    assert!(!potential.as_handle().has_hessian_vector_callback());
    assert_eq!(observer.native_calls(), after_fatal);
    let mut positions = vec![1.0, 0.0, 0.0];
    let mut numbers = vec![1];
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
    assert_eq!(
        unsafe { potential.calculate(&input, &mut output) },
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    );
    assert_eq!(observer.native_calls(), after_fatal);
    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    drop(potential);
    assert_eq!(observer.destroyed(), before_destroy + 1);
}

#[test]
fn fatal_hvp_poisons_a_live_potential_handle_without_reentering_native_code() {
    let path = build("harmonic_fatal_hvp_handle", &["FATAL_ONCE"]);
    let observer = FixtureObserver::open(&path);
    let before_destroy = observer.destroyed();
    let potential = load(&path).into_potential();
    assert!(potential.as_handle().has_hessian_vector_callback());

    let mut positions = vec![0.5, -1.0, 2.0];
    let mut numbers = vec![1];
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 1, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 1) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let mut product = [0.0; 3];
    let err = unsafe {
        potential.as_handle().hessian_vector(
            &input,
            &[1.0, 0.0, 0.0],
            0.0,
            rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
            &mut product,
        )
    }
    .unwrap_err();
    assert_eq!(err.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
    let after_fatal = observer.native_calls();

    let err_again = unsafe {
        potential.as_handle().hessian_vector(
            &input,
            &[1.0, 0.0, 0.0],
            0.0,
            rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
            &mut product,
        )
    }
    .unwrap_err();
    assert_eq!(err_again.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    assert_eq!(
        unsafe { potential.calculate(&input, &mut output) },
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    );
    assert_eq!(observer.native_calls(), after_fatal);

    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    drop(potential);
    assert_eq!(observer.destroyed(), before_destroy + 1);
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
fn abi_stamp_serves_equal_and_older_plugin_minors() {
    let host = RgpotEngineAbiStamp::HOST;
    assert!(host.serves(host));

    let older = RgpotEngineAbiStamp {
        abi_minor: host.abi_minor - 1,
        ..host
    };
    assert!(older.serves(host));

    let newer = RgpotEngineAbiStamp {
        abi_minor: host.abi_minor + 1,
        ..host
    };
    assert!(!newer.serves(host));
}

#[test]
fn an_incompatible_abi_stamp_is_refused_and_names_both_stamps() {
    let newer_minor = format!("FIXTURE_ABI_MINOR={}", ENGINE_ABI_MINOR + 1);
    for (name, define, found) in [
        (
            "harmonic_layout2",
            "FIXTURE_ABI_LAYOUT=2",
            RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: ENGINE_ABI_MINOR,
                layout_revision: 2,
            },
        ),
        (
            "harmonic_layout0",
            "FIXTURE_ABI_LAYOUT=0",
            RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: ENGINE_ABI_MINOR,
                layout_revision: 0,
            },
        ),
        (
            "harmonic_stamp_newer",
            newer_minor.as_str(),
            RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: ENGINE_ABI_MINOR + 1,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            },
        ),
        (
            "harmonic_stamp_major2",
            "FIXTURE_STAMP_MAJOR=2",
            RgpotEngineAbiStamp {
                abi_major: 2,
                abi_minor: ENGINE_ABI_MINOR,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            },
        ),
    ] {
        let path = build(name, &[define]);
        let error = unsafe { EnginePotential::load(&path, &config()) }
            .err()
            .expect("incompatible ABI stamp must be refused");
        assert_eq!(
            error,
            EngineError::AbiStamp {
                expected: RgpotEngineAbiStamp::HOST,
                found,
            }
        );
        let text = error.to_string();
        assert!(
            text.contains(&RgpotEngineAbiStamp::HOST.to_string()),
            "{text}"
        );
        assert!(text.contains(&found.to_string()), "{text}");
    }
}

#[test]
fn a_missing_abi_stamp_symbol_is_served_as_minor_zero() {
    let path = build("harmonic_nostamp", &["NO_STAMP"]);
    let mut engine = load(&path);
    assert_eq!(
        engine.abi_stamp(),
        RgpotEngineAbiStamp {
            abi_major: ENGINE_ABI_VERSION as u16,
            abi_minor: 0,
            layout_revision: 1,
        }
    );
    let answer = engine
        .force(&EngineRequest {
            positions: &[1.0, 0.0, 0.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        })
        .expect("force on minor-0 engine");
    assert!(within_roundoff(answer.energy, 0.5 * SPRING, 3));
}

#[test]
fn minor_4_lacks_components_minor_5_lacks_the_jet_and_a_newer_minor_is_rejected() {
    assert_eq!(ENGINE_ABI_MINOR, 11);

    // A minor-4 plugin omits rgpot_engine_last_components and
    // rgpot_engine_component_name; the host serves it, resolves those symbols
    // as absent, and evaluates force and HVP normally.
    let minor4_path = build("harmonic_stamp_minor4", &["FIXTURE_ABI_MINOR=4"]);
    let mut minor4 = load(&minor4_path);
    assert_eq!(
        minor4.abi_stamp(),
        RgpotEngineAbiStamp {
            abi_major: ENGINE_ABI_VERSION as u16,
            abi_minor: 4,
            layout_revision: ENGINE_ABI_LAYOUT_REVISION,
        }
    );
    assert!(minor4.symbols().last_components.is_none());
    assert!(minor4.symbols().component_name.is_none());
    assert_eq!(
        minor4.last_components().unwrap_err(),
        EngineError::Unsupported {
            operation: "rgpot_engine_last_components",
            message: "symbol is not exported".into(),
        }
    );
    assert_eq!(
        minor4.component_name(0).unwrap_err(),
        EngineError::Unsupported {
            operation: "rgpot_engine_component_name",
            message: "symbol is not exported".into(),
        }
    );
    let req = EngineRequest {
        positions: &[1.0, -2.0, 0.5],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };
    let answer4 = minor4.force(&req).expect("force on minor-4 plugin");
    let sum: f64 = req.positions.iter().map(|x| x * x).sum();
    assert!(within_roundoff(answer4.energy, 0.5 * SPRING * sum, 9));
    let hv4 = minor4
        .hvp(&req, &[1.0, 0.0, -1.0])
        .expect("hvp on minor-4 plugin");
    assert!(all_within_roundoff(&hv4, &[SPRING, 0.0, -SPRING]));

    // A minor-5 plugin exports both component symbols and the host resolves
    // and calls them.
    let minor5_path = build("harmonic_stamp_minor5", &["FIXTURE_ABI_MINOR=5"]);
    let mut minor5 = load(&minor5_path);
    assert_eq!(
        minor5.abi_stamp(),
        RgpotEngineAbiStamp {
            abi_minor: 5,
            ..RgpotEngineAbiStamp::HOST
        }
    );
    assert!(minor5.symbols().taylor_jet.is_none());
    assert_eq!(minor5.caps().expect("caps").jet_order, 0);
    assert!(minor5.symbols().last_components.is_some());
    assert!(minor5.symbols().component_name.is_some());
    assert_eq!(minor5.component_name(0).expect("component 0"), "harmonic");
    assert_eq!(
        minor5.component_name(1).expect("out-of-range component"),
        ""
    );
    assert_eq!(
        minor5.last_components().expect("components before force"),
        vec![RgpotEngineComponent {
            weight: 1.0,
            energy: 0.0,
            stress: [0.0; 9],
            enabled: 1,
            evaluated: 0,
            has_stress: 0,
        }]
    );
    let answer5 = minor5.force(&req).expect("force on minor-5 plugin");
    assert_eq!(
        minor5.last_components().expect("components after force"),
        vec![RgpotEngineComponent {
            weight: 1.0,
            energy: answer5.energy,
            stress: [0.0; 9],
            enabled: 1,
            evaluated: 1,
            has_stress: 0,
        }]
    );

    // A minor-11 plugin is the current minor and also exports the jet.
    let minor11_path = build("harmonic_stamp_minor11", &["FIXTURE_ABI_MINOR=11"]);
    let minor11 = load(&minor11_path);
    assert_eq!(minor11.abi_stamp(), RgpotEngineAbiStamp::HOST);
    assert!(minor11.symbols().taylor_jet.is_some());
    assert_eq!(minor11.caps().expect("caps").jet_order, 3);

    // A plugin whose minor is newer than the host's is rejected.
    let newer_define = format!("FIXTURE_ABI_MINOR={}", ENGINE_ABI_MINOR + 1);
    let newer_path = build("harmonic_stamp_minor_too_new", &[&newer_define]);
    let error = unsafe { EnginePotential::load(&newer_path, &config()) }
        .err()
        .expect("plugin newer than the host minor must be refused");
    assert_eq!(
        error,
        EngineError::AbiStamp {
            expected: RgpotEngineAbiStamp::HOST,
            found: RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: ENGINE_ABI_MINOR + 1,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            },
        }
    );
}

#[test]
fn a_missing_required_symbol_is_named() {
    let path = build("harmonic_noforce", &["NO_FORCE", "NO_BATCH"]);
    let error = unsafe { EnginePotential::load(&path, &config()) }
        .err()
        .expect("force is required");
    assert_eq!(error, EngineError::Missing("rgpot_engine_force".into()));
}

#[test]
fn a_plugin_without_available_is_treated_as_available() {
    let path = build("harmonic_noavailable", &["NO_AVAILABLE"]);
    let mut engine = load(&path);
    assert!(engine.symbols().available.is_none());
    engine
        .force(&one_atom_request())
        .expect("a plugin that does not export rgpot_engine_available evaluates");

    let null_available = build_with_abs_zero(
        "harmonic_abs0_available",
        &["NO_AVAILABLE"],
        &["rgpot_engine_available"],
    );
    let engine = load(&null_available);
    assert!(
        engine.symbols().available.is_none(),
        "a null rgpot_engine_available address is treated as absent"
    );
}

#[test]
fn an_unavailable_engine_is_refused_before_create() {
    let path = build("harmonic_unavailable", &["FIXTURE_AVAILABLE=0"]);
    let observer = FixtureObserver::open(&path);
    let before = observer.created();
    let error = unsafe { EnginePotential::load(&path, &config()) }
        .err()
        .expect("unavailable engine must be refused");
    assert_eq!(error, EngineError::Unavailable);
    assert_eq!(observer.created(), before);
}

/// Every refusal of a load leaves no engine instance behind: an instance the
/// failed load created is destroyed once, and a refusal before creation never
/// creates one. The library handle of each refused first load is closed, which
/// `failed_first_loads_are_closed_and_can_be_retried` checks by mapping.
#[test]
fn every_rejected_load_balances_created_and_destroyed_instances() {
    let too_new = format!("FIXTURE_ABI_MINOR={}", ENGINE_ABI_MINOR + 1);
    let cases: [(&str, Vec<&str>, c_int); 6] = [
        (
            "harmonic_balance_unavailable",
            vec!["FIXTURE_AVAILABLE=0"],
            0,
        ),
        ("harmonic_balance_abi", vec!["FIXTURE_ABI=2"], 0),
        ("harmonic_balance_too_new", vec![too_new.as_str()], 0),
        ("harmonic_balance_layout", vec!["FIXTURE_ABI_LAYOUT=2"], 0),
        ("harmonic_balance_caps_lock", vec!["CAPS_LOCK_ERROR"], 1),
        ("harmonic_balance_caps_fatal", vec!["CAPS_ERROR_CODE=-1"], 1),
    ];
    for (name, defines, creates) in cases {
        let path = build(name, &defines);
        let observer = FixtureObserver::open(&path);
        let created = observer.created();
        let destroyed = observer.destroyed();
        for attempt in 0..3 {
            unsafe { EnginePotential::load(&path, &config()) }
                .err()
                .unwrap_or_else(|| panic!("{name} must be refused"));
            assert_eq!(
                observer.created() - created,
                creates * (attempt + 1),
                "{name}: instances created by attempt {attempt}"
            );
            assert_eq!(
                observer.destroyed() - destroyed,
                creates * (attempt + 1),
                "{name}: instances destroyed by attempt {attempt}"
            );
        }
    }
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
    let observer = FixtureObserver::open(&path);
    let before = observer.destroyed();
    drop(load(&path));
    assert_eq!(observer.destroyed(), before + 1);
}

/// `dl_iterate_phdr` entries whose object file name is `name`.
#[cfg(target_os = "linux")]
fn link_maps_named(name: &str) -> usize {
    use std::ffi::{c_char, c_void, CStr};

    #[repr(C)]
    struct DlPhdrInfo {
        _addr: usize,
        name: *const c_char,
    }

    struct Hits<'a> {
        name: &'a str,
        count: usize,
    }

    unsafe extern "C" fn visit(info: *mut DlPhdrInfo, _size: usize, data: *mut c_void) -> c_int {
        let hits = unsafe { &mut *data.cast::<Hits<'_>>() };
        let name = unsafe { (*info).name };
        if !name.is_null() {
            let object = unsafe { CStr::from_ptr(name) }.to_string_lossy();
            if Path::new(object.as_ref()).file_name() == Some(hits.name.as_ref()) {
                hits.count += 1;
            }
        }
        0
    }

    extern "C" {
        fn dl_iterate_phdr(
            callback: unsafe extern "C" fn(*mut DlPhdrInfo, usize, *mut c_void) -> c_int,
            data: *mut c_void,
        ) -> c_int;
    }

    let mut hits = Hits { name, count: 0 };
    unsafe { dl_iterate_phdr(visit, (&mut hits as *mut Hits<'_>).cast()) };
    hits.count
}

fn file_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .expect("fixture file name")
}

#[cfg(target_os = "linux")]
#[test]
fn dropping_the_instance_keeps_the_library_resident() {
    let path = build("harmonic_resident", &[]);
    assert!(!library_is_mapped(&path));
    let request = EngineRequest {
        positions: &[1.0, 0.0, 0.0],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };

    let first = load(&path);
    let create = first.symbols().create as usize;
    drop(first);
    for iteration in 0..200 {
        let mut engine = load(&path);
        assert_eq!(
            engine.symbols().create as usize,
            create,
            "iteration {iteration} resolved another copy of the plugin"
        );
        engine.force(&request).expect("force on a reused plugin");
        drop(engine);
        drop(load(&path).into_potential());
        assert!(
            library_is_mapped(&path),
            "fixture was unmapped after iteration {iteration}"
        );
    }
    assert_eq!(link_maps_named(file_name(&path)), 1);
}

#[cfg(target_os = "linux")]
#[test]
fn every_path_to_one_file_shares_one_registry_entry() {
    let path = build("harmonic_aliases", &[]);
    let dir = path.parent().expect("fixture directory");
    let symlink = dir.join("libharmonic_aliases_symlink.so");
    let hardlink = dir.join("libharmonic_aliases_hardlink.so");
    for alias in [&symlink, &hardlink] {
        let _ = std::fs::remove_file(alias);
    }
    std::os::unix::fs::symlink(&path, &symlink).expect("symlink the fixture");
    std::fs::hard_link(&path, &hardlink).expect("hard-link the fixture");
    let dotted = dir
        .join("..")
        .join(dir.file_name().expect("fixture directory name"))
        .join(file_name(&path));

    let engines: Vec<EnginePotential> = [&path, &symlink, &hardlink, &dotted]
        .into_iter()
        .map(|alias| load(alias))
        .collect();
    let create = engines[0].symbols().create as usize;
    for engine in &engines {
        assert_eq!(engine.symbols().create as usize, create);
    }
    assert_eq!(link_maps_named(file_name(&path)), 1);
    assert_eq!(link_maps_named(file_name(&hardlink)), 0);
    assert_eq!(link_maps_named(file_name(&symlink)), 0);
}

#[cfg(target_os = "linux")]
#[test]
fn a_file_replaced_at_a_loaded_path_is_refused() {
    let path = build("harmonic_replaced", &[]);
    let mut engine = load(&path);

    let next = build("harmonic_replaced_next", &[]);
    let staging = path.with_extension("staging");
    std::fs::copy(&next, &staging).expect("stage the replacement");
    std::fs::rename(&staging, &path).expect("replace the fixture");

    let error = unsafe { EnginePotential::load(&path, &config()) }
        .err()
        .expect("a replaced file must be refused");
    match &error {
        EngineError::Load(text) => assert!(text.contains("was replaced"), "{text}"),
        other => panic!("expected EngineError::Load, got {other:?}"),
    }

    let request = EngineRequest {
        positions: &[1.0, 0.0, 0.0],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };
    engine
        .force(&request)
        .expect("the resident plugin still evaluates");
    load(&next)
        .force(&request)
        .expect("the new file loads from its own path");
}

#[test]
fn a_bare_library_name_is_refused() {
    let path = build("harmonic_bare", &[]);
    let error = unsafe { EnginePotential::load(Path::new(file_name(&path)), &config()) }
        .err()
        .expect("a bare library name must be refused");
    match &error {
        EngineError::Load(text) => assert!(text.contains("bare library name"), "{text}"),
        other => panic!("expected EngineError::Load, got {other:?}"),
    }
    #[cfg(target_os = "linux")]
    assert!(!library_is_mapped(&path));
}

static INITIALIZER_REENTRY_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

extern "C" fn initializer_reentry_callback() -> c_int {
    std::panic::catch_unwind(|| {
        let path = INITIALIZER_REENTRY_PATH
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .expect("initializer reentry path");
        unsafe { EnginePotential::load(&path, &config()) }.map(drop)
    })
    .map_or(1, |result| if result.is_ok() { 0 } else { 1 })
}

/// Re-runs the calling test in a child process and reports whether this
/// process is that child. The parent returns `true` once the child succeeds
/// and panics if it fails or does not finish in time, which is how a deadlock
/// shows. The child returns `false` and runs the body of the test.
fn run_in_child(test_name: &str, marker: &str, deadlock: &str) -> bool {
    if std::env::var_os(marker).is_some() {
        return false;
    }
    let mut child = std::process::Command::new(
        std::env::current_exe().expect("path to the engine test executable"),
    )
    .args(["--exact", test_name, "--nocapture"])
    .env(marker, "1")
    .spawn()
    .expect("spawn the child test process");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().expect("wait for the child test process") {
            assert!(status.success(), "{test_name} child failed: {status}");
            return true;
        }
        if Instant::now() >= deadline {
            child.kill().expect("kill the deadlocked child");
            child.wait().expect("reap the deadlocked child");
            panic!("{deadlock}");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn an_initializer_can_load_another_engine_without_holding_the_registry_lock() {
    const TEST_NAME: &str =
        "an_initializer_can_load_another_engine_without_holding_the_registry_lock";
    if run_in_child(
        TEST_NAME,
        "RGPOT_INITIALIZER_REENTRY_CHILD",
        "plugin initializer deadlocked while loading another engine",
    ) {
        return;
    }

    let outer = build("harmonic_initializer_reentry", &["REENTER_IN_INITIALIZER"]);
    let inner = build("harmonic_initializer_inner", &[]);
    *INITIALIZER_REENTRY_PATH
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(inner);
    std::env::set_var(
        "RGPOT_FIXTURE_INIT_CALLBACK",
        format!("{:x}", initializer_reentry_callback as *const () as usize),
    );
    drop(load(&outer));
    std::env::remove_var("RGPOT_FIXTURE_INIT_CALLBACK");
    *INITIALIZER_REENTRY_PATH
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

#[cfg(target_os = "linux")]
#[test]
fn failed_first_loads_are_closed_and_can_be_retried() {
    let failures = [
        (
            build("harmonic_failed_missing", &["NO_FORCE", "NO_BATCH"]),
            config().to_vec(),
        ),
        (
            build("harmonic_failed_abi", &["FIXTURE_ABI=2"]),
            config().to_vec(),
        ),
        (
            build("harmonic_failed_stamp", &["FIXTURE_ABI_LAYOUT=2"]),
            config().to_vec(),
        ),
        (build("harmonic_failed_create", &[]), vec![1, 2, 3]),
        (
            build("harmonic_failed_unavailable", &["FIXTURE_AVAILABLE=0"]),
            config().to_vec(),
        ),
        (
            build(
                "harmonic_failed_minor_too_new",
                &[&format!("FIXTURE_ABI_MINOR={}", ENGINE_ABI_MINOR + 1)],
            ),
            config().to_vec(),
        ),
        (
            build("harmonic_failed_caps_lock", &["CAPS_LOCK_ERROR"]),
            config().to_vec(),
        ),
        (
            build("harmonic_failed_caps_fatal", &["CAPS_ERROR_CODE=-1"]),
            config().to_vec(),
        ),
    ];

    for (path, config) in failures {
        assert!(!library_is_mapped(&path));
        let first = unsafe { EnginePotential::load(&path, &config) }
            .err()
            .expect("fixture must fail to load");
        assert!(
            !library_is_mapped(&path),
            "{} stayed mapped after load failed",
            path.display()
        );
        let again = unsafe { EnginePotential::load(&path, &config) }
            .err()
            .expect("fixture must fail to load again");
        assert_eq!(again, first);
        assert!(!library_is_mapped(&path));
        assert_eq!(link_maps_named(file_name(&path)), 0);
    }
}

const UNLOAD_SAFE: &str = "FIXTURE_UNLOAD_POLICY=RGPOT_ENGINE_UNLOAD_SAFE";

/// A plugin whose file is replaced while its first engine is created is
/// refused, and the handle opened for that attempt is closed rather than kept.
#[cfg(target_os = "linux")]
#[test]
fn a_file_replaced_during_load_is_refused_and_its_handle_is_closed() {
    for (name, defines) in [
        (
            "harmonic_swap_in_create_safe",
            vec!["REPLACE_SELF_IN_CREATE", UNLOAD_SAFE],
        ),
        (
            "harmonic_swap_in_create_resident",
            vec!["REPLACE_SELF_IN_CREATE"],
        ),
    ] {
        let path = build(name, &defines);
        let replacement = build(&format!("{name}_next"), &[]);
        let staged = PathBuf::from(format!("{}.next", path.display()));
        std::fs::copy(&replacement, &staged).expect("stage the replacement");

        let error = unsafe { EnginePotential::load(&path, &config()) }
            .err()
            .expect("a file replaced during load must be refused");
        match &error {
            EngineError::Load(text) => assert!(text.contains("was replaced"), "{name}: {text}"),
            other => panic!("{name}: expected EngineError::Load, got {other:?}"),
        }
        assert_eq!(
            link_maps_named(file_name(&path)),
            0,
            "{name}: the refused attempt kept its handle open"
        );

        let not_loaded = unsafe { EnginePotential::unload(&path) }.unwrap_err();
        assert!(
            matches!(&not_loaded, EngineError::Invalid(text) if text.contains("was not loaded")),
            "{name}: {not_loaded:?}"
        );
        let mut engine = load(&path);
        engine
            .force(&EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            })
            .expect("the replacement file loads and evaluates");
    }
}

static FINALIZER_REENTRY_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

extern "C" fn finalizer_reentry_callback() -> c_int {
    std::panic::catch_unwind(|| {
        let path = FINALIZER_REENTRY_PATH
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .expect("finalizer reentry path");
        unsafe { EnginePotential::load(&path, &config()) }.map(drop)
    })
    .map_or(1, |result| if result.is_ok() { 0 } else { 1 })
}

/// The destructor of an unloaded plugin may load another engine, which needs
/// the registry lock, so `unload` must not hold it across `dlclose`.
#[cfg(target_os = "linux")]
#[test]
fn unload_closes_the_library_without_holding_the_registry_lock() {
    const TEST_NAME: &str = "unload_closes_the_library_without_holding_the_registry_lock";
    if run_in_child(
        TEST_NAME,
        "RGPOT_FINALIZER_REENTRY_CHILD",
        "unload deadlocked: the plugin destructor could not load another engine",
    ) {
        return;
    }

    let outer = build(
        "harmonic_finalizer_reentry",
        &["REENTER_IN_FINALIZER", UNLOAD_SAFE],
    );
    let inner = build("harmonic_finalizer_inner", &[]);
    *FINALIZER_REENTRY_PATH
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(inner.clone());
    std::env::set_var(
        "RGPOT_FIXTURE_FINI_CALLBACK",
        format!("{:x}", finalizer_reentry_callback as *const () as usize),
    );
    drop(load(&outer));
    unsafe { EnginePotential::unload(&outer) }.expect("unload the plugin");
    std::env::remove_var("RGPOT_FIXTURE_FINI_CALLBACK");
    assert!(!library_is_mapped(&outer));
    assert!(
        library_is_mapped(&inner),
        "the destructor did not load the other engine"
    );
}

fn unload_refusal(path: &Path) -> String {
    match unsafe { EnginePotential::unload(path) } {
        Err(EngineError::Unsupported { operation, message }) => {
            assert_eq!(operation, "rgpot_engine_unload_policy");
            message
        }
        other => panic!("expected the unload to be refused, got {other:?}"),
    }
}

#[test]
fn unload_is_refused_without_the_policy_symbol() {
    let path = build("harmonic_unload_no_policy", &[]);
    let create = load(&path).symbols().create as usize;
    let message = unload_refusal(&path);
    assert!(message.contains("is not exported"), "{message}");
    #[cfg(target_os = "linux")]
    assert!(library_is_mapped(&path));
    assert_eq!(load(&path).symbols().create as usize, create);
}

#[test]
fn unload_is_refused_for_a_resident_policy_or_a_stamp_before_the_policy_minor() {
    let older_minor = format!("FIXTURE_ABI_MINOR={}", ENGINE_UNLOAD_POLICY_MINOR - 1);
    for (name, defines, reason) in [
        (
            "harmonic_unload_resident_policy",
            vec!["FIXTURE_UNLOAD_POLICY=RGPOT_ENGINE_UNLOAD_RESIDENT"],
            "returned 0",
        ),
        (
            "harmonic_unload_older_minor",
            vec![UNLOAD_SAFE, older_minor.as_str()],
            "is ignored",
        ),
        (
            "harmonic_unload_no_stamp",
            vec![UNLOAD_SAFE, "NO_STAMP"],
            "is ignored",
        ),
    ] {
        let path = build(name, &defines);
        drop(load(&path));
        let message = unload_refusal(&path);
        assert!(message.contains(reason), "{name}: {message}");
        #[cfg(target_os = "linux")]
        assert!(library_is_mapped(&path), "{name} was unmapped");
    }
}

#[test]
fn an_unload_safe_plugin_is_unloaded_only_without_live_instances() {
    let path = build("harmonic_unload_safe", &[UNLOAD_SAFE]);
    let not_loaded = unsafe { EnginePotential::unload(&path) }.unwrap_err();
    assert!(
        matches!(&not_loaded, EngineError::Invalid(text) if text.contains("was not loaded")),
        "{not_loaded:?}"
    );

    let engine = load(&path);
    let potential = load(&path).into_potential();
    let busy = unsafe { EnginePotential::unload(&path) }.unwrap_err();
    assert!(
        matches!(&busy, EngineError::Invalid(text) if text.contains("2 engine instance(s)")),
        "{busy:?}"
    );
    drop(engine);
    let busy = unsafe { EnginePotential::unload(&path) }.unwrap_err();
    assert!(
        matches!(&busy, EngineError::Invalid(text) if text.contains("1 engine instance(s)")),
        "{busy:?}"
    );
    drop(potential);

    unsafe { EnginePotential::unload(&path) }.expect("unload the unload-safe plugin");
    #[cfg(target_os = "linux")]
    assert!(!library_is_mapped(&path));
    let mut again = load(&path);
    let answer = again
        .force(&EngineRequest {
            positions: &[1.0, 0.0, 0.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        })
        .expect("force after a reload");
    assert!(within_roundoff(answer.energy, 0.5 * SPRING, 3));
}

/// The load and unload loop the sanitizer run exercises.
#[test]
fn an_unload_safe_plugin_survives_repeated_load_and_unload() {
    let path = build("harmonic_unload_loop", &[UNLOAD_SAFE]);
    let request = EngineRequest {
        positions: &[0.5, -1.0, 2.0],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };
    for iteration in 0..200 {
        let mut engine = load(&path);
        engine.force(&request).expect("force");
        drop(engine);
        drop(load(&path).into_potential());
        unsafe { EnginePotential::unload(&path) }
            .unwrap_or_else(|why| panic!("unload {iteration}: {why}"));
        #[cfg(target_os = "linux")]
        assert!(
            !library_is_mapped(&path),
            "fixture stayed mapped after unload {iteration}"
        );
    }
}

/// Median nanoseconds per load, create, destroy cycle for a resident plugin
/// and for an unload-safe plugin that is unloaded every cycle. Run with
/// `cargo test --release --features engine --test engine -- --ignored
/// load_cycle_cost --nocapture`.
#[test]
#[ignore = "timing measurement; prints medians instead of asserting"]
fn load_cycle_cost() {
    const CYCLES: u32 = 2000;
    const RUNS: usize = 11;
    let resident = build("harmonic_cost_resident", &[]);
    let unloadable = build("harmonic_cost_unload", &[UNLOAD_SAFE]);
    let median = |mut samples: Vec<f64>| {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    };
    let time = |cycle: &dyn Fn()| {
        cycle();
        let mut runs = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = std::time::Instant::now();
            for _ in 0..CYCLES {
                cycle();
            }
            runs.push(start.elapsed().as_nanos() as f64 / f64::from(CYCLES));
        }
        runs
    };
    let resident_runs = time(&|| drop(load(&resident)));
    let unload_runs = time(&|| {
        drop(load(&unloadable));
        unsafe { EnginePotential::unload(&unloadable) }.expect("unload");
    });
    println!(
        "resident: median {:.0} ns/cycle over {RUNS} runs of {CYCLES} cycles {:?}",
        median(resident_runs.clone()),
        resident_runs
            .iter()
            .map(|ns| ns.round() as u64)
            .collect::<Vec<_>>()
    );
    println!(
        "unload:   median {:.0} ns/cycle over {RUNS} runs of {CYCLES} cycles {:?}",
        median(unload_runs.clone()),
        unload_runs
            .iter()
            .map(|ns| ns.round() as u64)
            .collect::<Vec<_>>()
    );
}

#[test]
fn the_potential_handle_evaluates_and_destroys_the_engine() {
    let path = build("harmonic_handle", &[]);
    let observer = FixtureObserver::open(&path);
    let before = observer.destroyed();
    let potential = load(&path).into_potential();
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
    let counts = potential.as_handle().hvp_counts();
    assert_eq!(counts.force_calls, 0);
    assert_eq!(counts.analytic, 1);
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    drop(potential);
    assert_eq!(observer.destroyed(), before + 1);
}

#[test]
fn the_potential_handle_reports_an_engine_failure_as_a_status() {
    let path = build("harmonic_status", &[]);
    let potential = load(&path).into_potential();
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
    let caps = engine.caps().expect("caps");
    assert!(caps.analytic_hvp);
    let positions = [0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let direction = [1.0, -0.5, 0.25, -2.0, 0.0, 4.0];
    let hv = engine
        .hvp(
            &EngineRequest {
                positions: &positions,
                atomic_numbers: &[1, 8],
                box_matrix: &CELL,
            },
            &direction,
        )
        .expect("hvp");
    let expected: Vec<f64> = direction.iter().map(|d| SPRING * d).collect();
    assert!(all_within_roundoff(&hv, &expected));
}

#[test]
fn hvp_without_the_symbol_or_capability_is_unsupported() {
    let no_sym = build("harmonic_nohvp", &["NO_HVP"]);
    let mut engine = load(&no_sym);
    assert!(!engine.caps().expect("caps").analytic_hvp);
    let err = engine
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
        err,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        }
    ));
    assert_eq!(err.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
    let potential = load(&no_sym).into_potential();
    assert!(!potential.as_handle().has_hessian_vector_callback());

    let no_cap = build("harmonic_nohvpcap", &["NO_HVP_CAP"]);
    let mut engine = load(&no_cap);
    assert!(!engine.caps().expect("caps").analytic_hvp);
    let err = engine
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
        err,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        }
    ));
    let potential = load(&no_cap).into_potential();
    assert!(!potential.as_handle().has_hessian_vector_callback());

    let no_caps = build("harmonic_nocaps", &["NO_CAPS"]);
    let mut engine = load(&no_caps);
    assert_eq!(
        engine.caps().unwrap_err(),
        EngineError::Unsupported {
            operation: "rgpot_engine_caps",
            message: "symbol is not exported".into(),
        }
    );
    let err = engine
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
        err,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        }
    ));
    let potential = load(&no_caps).into_potential();
    assert!(!potential.as_handle().has_hessian_vector_callback());
}

#[test]
fn unsupported_code_keeps_the_engine_message_and_matches_the_header() {
    let path = build("harmonic_hvpunsupported", &["HVP_UNSUPPORTED"]);
    assert_eq!(unsupported_code(&path), ENGINE_ERR_UNSUPPORTED);
    let mut engine = load(&path);
    let err = engine
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
        err,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            message: "harmonic engine built without analytic HVP".into(),
        }
    );
}

#[test]
fn hvp_maps_recoverable_and_fatal_codes() {
    let path = build("harmonic_hvpcodes", &[]);
    let mut engine = load(&path);
    let positions = [0.0; 3];
    let recoverable = EngineRequest {
        positions: &positions,
        atomic_numbers: &[0],
        box_matrix: &CELL,
    };
    assert_eq!(
        engine.hvp(&recoverable, &[1.0, 0.0, 0.0]).unwrap_err(),
        EngineError::Recoverable {
            code: 1,
            message: "atom 0 has atomic number 0".into(),
        }
    );
    let fatal = EngineRequest {
        positions: &positions,
        atomic_numbers: &[-1],
        box_matrix: &CELL,
    };
    assert_eq!(
        engine.hvp(&fatal, &[1.0, 0.0, 0.0]).unwrap_err(),
        EngineError::Fatal {
            code: -1,
            message: "atom 0 has a negative atomic number".into(),
        }
    );
}

#[test]
fn hvp_rejects_a_non_finite_product() {
    let path = build("harmonic_hvpnan", &["HVP_NONFINITE"]);
    let mut engine = load(&path);
    let err = engine
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
        err,
        EngineError::Invalid("the engine Hessian-vector product at element 1 is not finite".into())
    );
}

#[test]
fn hvp_rejects_mismatched_direction_length() {
    let path = build("harmonic_hvplen", &[]);
    let mut engine = load(&path);
    let err = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 6],
                atomic_numbers: &[1, 8],
                box_matrix: &CELL,
            },
            &[1.0, 2.0, 3.0],
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Invalid(_)), "{err}");
    assert_eq!(err.status(), rgpot_status_t::RGPOT_INVALID_PARAMETER);
}

#[test]
fn hvp_c_api_rejects_overlapping_direction_and_output() {
    let path = build("harmonic_hvpalias", &[]);
    let potential = load(&path).into_potential();
    let mut positions = vec![0.5, -1.0, 2.0];
    let mut numbers = vec![1];
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 1, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 1) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let initial = [1.0, -0.5, 0.25];
    let mut aliased = initial;
    let status = unsafe {
        rgpot_core::c_api::hvp::rgpot_potential_hvp(
            potential.as_handle() as *const _,
            &input,
            aliased.as_ptr(),
            0.0,
            rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC as c_int,
            aliased.as_mut_ptr(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert_eq!(aliased, initial);
    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
}

/// `RgpotEngineCaps` holds the fields of `engine_c_abi.h` in the same order,
/// each four bytes wide, so every offset an engine writes is the one the
/// loader reads.
#[test]
fn engine_caps_layout_is_the_c_header_layout() {
    use std::mem::{offset_of, size_of};

    let header = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../CppCore/rgpot/engine_c_abi.h"),
    )
    .expect("read engine_c_abi.h");
    let end = header
        .find("} RgpotEngineCaps;")
        .expect("RgpotEngineCaps in engine_c_abi.h");
    let open = "typedef struct {";
    let start = header[..end].rfind(open).expect("start of RgpotEngineCaps") + open.len();
    let declared: Vec<&str> = header[start..end]
        .lines()
        .filter_map(|line| {
            let mut words = line.split_once(';')?.0.split_whitespace();
            let (kind, name) = (words.next()?, words.next()?);
            assert!(matches!(kind, "int" | "unsigned"), "{name} is a {kind}");
            Some(name)
        })
        .collect();

    let loader = [
        ("size", offset_of!(RgpotEngineCaps, size)),
        ("reentrancy", offset_of!(RgpotEngineCaps, reentrancy)),
        (
            "per_image_instances",
            offset_of!(RgpotEngineCaps, per_image_instances),
        ),
        ("batched", offset_of!(RgpotEngineCaps, batched)),
        ("periodic", offset_of!(RgpotEngineCaps, periodic)),
        ("stress", offset_of!(RgpotEngineCaps, stress)),
        (
            "group_collective",
            offset_of!(RgpotEngineCaps, group_collective),
        ),
        (
            "orbital_selection",
            offset_of!(RgpotEngineCaps, orbital_selection),
        ),
        ("orbital_copy", offset_of!(RgpotEngineCaps, orbital_copy)),
        ("analytic_hvp", offset_of!(RgpotEngineCaps, analytic_hvp)),
        ("jet_order", offset_of!(RgpotEngineCaps, jet_order)),
    ];
    let names: Vec<&str> = loader.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, declared, "field order differs from engine_c_abi.h");
    for (index, (name, offset)) in loader.iter().enumerate() {
        assert_eq!(*offset, 4 * index, "offset of {name}");
    }
    assert_eq!(size_of::<RgpotEngineCaps>(), 4 * declared.len());
}

#[test]
fn coincident_pair_reports_recoverable_eval_error_and_internal_status() {
    let path = build("harmonic_coincident", &["REJECT_COINCIDENT"]);
    let mut engine = load(&path);
    let coincident = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    let numbers = [1, 1];
    let direction = [0.5, -0.25, 0.75, -0.5, 0.25, -0.75];
    let expected = EngineError::Recoverable {
        code: 2,
        message: "coincident atoms".into(),
    };

    assert_eq!(
        engine
            .force(&EngineRequest {
                positions: &coincident,
                atomic_numbers: &numbers,
                box_matrix: &CELL,
            })
            .unwrap_err(),
        expected
    );
    assert_eq!(
        engine
            .force_batch(&numbers, &coincident, &CELL)
            .unwrap_err(),
        expected
    );
    assert_eq!(
        engine
            .hvp(
                &EngineRequest {
                    positions: &coincident,
                    atomic_numbers: &numbers,
                    box_matrix: &CELL,
                },
                &direction,
            )
            .unwrap_err(),
        expected
    );

    let potential = engine.into_potential();
    let mut positions = coincident.to_vec();
    let mut atomic_numbers = numbers.to_vec();
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 2, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(atomic_numbers.as_mut_ptr(), 2) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    assert_eq!(
        unsafe { potential.calculate(&input, &mut output) },
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    );
    assert!(output.forces.is_null());

    assert_eq!(
        unsafe {
            rgpot_core::c_api::hvp::rgpot_potential_hvp_centre(
                potential.as_handle() as *const _,
                &input,
                &mut output,
            )
        },
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    );
    let mut hv_out = [0.0; 6];
    for scheme in [
        rgpot_hvp_scheme_t::RGPOT_HVP_AUTO,
        rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
        rgpot_hvp_scheme_t::RGPOT_HVP_FORWARD,
    ] {
        assert_eq!(
            unsafe {
                rgpot_core::c_api::hvp::rgpot_potential_hvp(
                    potential.as_handle() as *const _,
                    &input,
                    direction.as_ptr(),
                    0.0,
                    scheme as c_int,
                    hv_out.as_mut_ptr(),
                )
            },
            rgpot_status_t::RGPOT_INTERNAL_ERROR
        );
    }

    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }

    let mut separated = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
    let sep_input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(separated.as_mut_ptr(), 2, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(atomic_numbers.as_mut_ptr(), 2) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    assert_eq!(
        unsafe { potential.calculate(&sep_input, &mut output) },
        rgpot_status_t::RGPOT_SUCCESS
    );
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_tensor_free(sep_input.positions);
        rgpot_tensor_free(sep_input.atomic_numbers);
        rgpot_tensor_free(sep_input.box_matrix);
    }
}

struct SuccessStatusFailure;

impl CoordinateTransform for SuccessStatusFailure {
    fn transform(
        &mut self,
        _positions: &mut [f64],
        _box_matrix: &mut [f64; 9],
    ) -> Result<(), EngineFailure> {
        Err(EngineFailure {
            status: EngineStatus::Success,
            message: "failure with success status".into(),
        })
    }
}

#[test]
fn transform_failure_with_success_status_is_an_error() {
    let path = build("harmonic_transform_success_status", &[]);
    let mut session = load_session(&path);
    let answer = session.force(
        &EngineRequest {
            positions: &[1.0, 2.0, 3.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        },
        Some(&mut SuccessStatusFailure),
    );
    assert!(matches!(answer, Err(EngineError::Invalid(_))), "{answer:?}");
}

#[test]
fn status_encoding_has_one_sign_per_class_and_round_trips() {
    let samples = [
        c_int::MIN,
        c_int::MIN + 1,
        -1000,
        -7,
        -2,
        -1,
        0,
        1,
        2,
        3,
        7,
        1000,
        c_int::MAX - 1,
        c_int::MAX,
    ];
    for code in samples {
        let status = EngineStatus::from_code(code);
        assert_eq!(status.code(), code, "code {code}");
        assert_eq!(EngineStatus::from_code(status.code()), status);
        match status {
            EngineStatus::Success => assert_eq!(code, 0),
            EngineStatus::RecoverableInputFailure(magnitude) => {
                assert!(code > 0, "code {code}");
                assert_eq!(magnitude.get(), code.unsigned_abs());
            }
            EngineStatus::FatalInstanceFailure(magnitude) => {
                assert!(code < 0, "code {code}");
                assert_eq!(magnitude.get(), code.unsigned_abs());
            }
        }
    }

    assert_eq!(
        EngineStatus::recoverable(2),
        Some(EngineStatus::from_code(2))
    );
    assert_eq!(EngineStatus::recoverable(0), None);
    assert_eq!(EngineStatus::recoverable(-2), None);
    assert_eq!(EngineStatus::fatal(-2), Some(EngineStatus::from_code(-2)));
    assert_eq!(EngineStatus::fatal(0), None);
    assert_eq!(EngineStatus::fatal(2), None);
    assert_eq!(
        EngineStatus::Success.or_invalid().code(),
        ENGINE_ERR_INVALID
    );
    assert_eq!(EngineStatus::from_code(-4).or_invalid().code(), -4);
}

#[test]
fn engine_errors_report_a_status_of_their_own_class() {
    let recoverable = EngineError::Recoverable {
        code: 5,
        message: String::new(),
    };
    let fatal = EngineError::Fatal {
        code: -5,
        message: String::new(),
    };
    assert_eq!(
        recoverable.engine_status(),
        Some(EngineStatus::from_code(5))
    );
    assert_eq!(fatal.engine_status(), Some(EngineStatus::from_code(-5)));
    let contradictory = EngineError::Recoverable {
        code: -5,
        message: String::new(),
    };
    assert_eq!(contradictory.engine_status(), None);
}

struct ScriptedOracle {
    answer: Result<EngineEvaluation, EngineFailure>,
}

impl EngineCallbacks for ScriptedOracle {
    fn oracle(&mut self, _request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        self.answer.clone()
    }
}

fn oracle_force(
    name: &str,
    answer: Result<EngineEvaluation, EngineFailure>,
) -> (
    PathBuf,
    EngineSession,
    Result<EngineEvaluation, EngineError>,
) {
    oracle_force_with(name, &[], answer)
}

fn oracle_force_with(
    name: &str,
    defines: &[&str],
    answer: Result<EngineEvaluation, EngineFailure>,
) -> (
    PathBuf,
    EngineSession,
    Result<EngineEvaluation, EngineError>,
) {
    let path = build(name, defines);
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(ScriptedOracle { answer })))
        .expect("set callbacks");
    let answer = session.force(
        &EngineRequest {
            positions: &[1.0, 2.0, 3.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        },
        None,
    );
    (path, session, answer)
}

#[test]
fn oracle_failure_with_success_status_reaches_native_code_as_an_error() {
    let (path, session, answer) = oracle_force(
        "harmonic_oracle_success_status",
        Err(EngineFailure {
            status: EngineStatus::Success,
            message: "failure with success status".into(),
        }),
    );
    assert_eq!(last_oracle_rc(&path), ENGINE_ERR_INVALID);
    assert_eq!(
        answer,
        Err(EngineError::Recoverable {
            code: ENGINE_ERR_INVALID,
            message: "failure with success status".into(),
        })
    );
    assert_eq!(session.model_state().expect("model state"), 0);
}

#[test]
fn every_oracle_failure_status_reaches_native_code_as_its_own_nonzero_code() {
    let (path, session, answer) = oracle_force(
        "harmonic_oracle_recoverable",
        Err(EngineFailure {
            status: EngineStatus::from_code(5),
            message: "oracle refused this geometry".into(),
        }),
    );
    assert_eq!(last_oracle_rc(&path), 5);
    assert_eq!(
        answer
            .expect_err("recoverable oracle failure")
            .engine_status(),
        Some(EngineStatus::from_code(5))
    );
    assert_eq!(session.model_state().expect("model state"), 0);

    let (path, mut session, answer) = oracle_force(
        "harmonic_oracle_fatal",
        Err(EngineFailure {
            status: EngineStatus::from_code(-9),
            message: "oracle lost its backend".into(),
        }),
    );
    assert_eq!(last_oracle_rc(&path), -9);
    let error = answer.expect_err("fatal oracle failure");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -9,
            message: "oracle lost its backend".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);

    let (path, _session, answer) = oracle_force(
        "harmonic_oracle_wrong_length",
        Ok(EngineEvaluation {
            energy: 1.0,
            forces: vec![0.0],
        }),
    );
    assert_eq!(last_oracle_rc(&path), ENGINE_ERR_INVALID);
    assert!(
        matches!(answer, Err(EngineError::Recoverable { code, .. }) if code == ENGINE_ERR_INVALID),
        "{answer:?}"
    );
}

#[test]
fn oracle_output_that_is_not_finite_is_an_error_not_a_success() {
    let cases = [
        ("energy_nan", f64::NAN, vec![0.0, 0.0, 0.0], "energy"),
        ("energy_inf", f64::INFINITY, vec![0.0, 0.0, 0.0], "energy"),
        (
            "force_nan",
            1.0,
            vec![0.0, f64::NAN, 0.0],
            "force at element 1",
        ),
        (
            "force_inf",
            1.0,
            vec![0.0, 0.0, f64::NEG_INFINITY],
            "force at element 2",
        ),
    ];
    for (name, energy, forces, what) in cases {
        let (path, session, answer) = oracle_force(
            &format!("harmonic_oracle_{name}"),
            Ok(EngineEvaluation { energy, forces }),
        );
        assert_eq!(last_oracle_rc(&path), ENGINE_ERR_INVALID, "{name}");
        match answer {
            Err(EngineError::Recoverable { code, message }) => {
                assert_eq!(code, ENGINE_ERR_INVALID, "{name}");
                assert!(message.contains(what), "{name}: {message}");
                assert!(message.contains("not finite"), "{name}: {message}");
            }
            other => panic!("{name}: expected a recoverable error, got {other:?}"),
        }
        assert_eq!(session.model_state().expect("model state"), 0, "{name}");
    }
}

#[test]
fn a_finite_oracle_answer_still_succeeds() {
    let (path, session, answer) = oracle_force(
        "harmonic_oracle_finite",
        Ok(EngineEvaluation {
            energy: -2.5,
            forces: vec![0.5, -0.5, 0.0],
        }),
    );
    assert_eq!(last_oracle_rc(&path), 0);
    let answer = answer.expect("finite oracle answer");
    assert_eq!(answer.energy, -2.5);
    assert_eq!(answer.forces, [0.5, -0.5, 0.0]);
    assert_eq!(session.model_state().expect("model state"), 1);
}

#[test]
fn a_fatal_oracle_failure_is_latched_over_the_engines_own_fatal_code() {
    let (path, mut session, answer) = oracle_force_with(
        "harmonic_oracle_fatal_then_engine_fatal",
        &["FATAL_AFTER_HOST_FAILURE"],
        Err(EngineFailure {
            status: EngineStatus::from_code(-9),
            message: "oracle lost its backend".into(),
        }),
    );
    assert_eq!(last_oracle_rc(&path), -9);
    let error = answer.expect_err("fatal oracle failure");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -9,
            message: "oracle lost its backend".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

#[test]
fn a_fatal_hint_with_an_empty_error_buffer_latches_a_message_naming_the_code() {
    let path = build("harmonic_hint_fatal_without_message", &[]);
    let mut session = load_session(&path);
    let error = session.hint(&[0xfe]).expect_err("fixture fails the hint");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -5,
            message: "rgpot_engine_hint failed with code -5 and wrote no message".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

struct AcquirePanicThenRefuse;

impl EngineCallbacks for AcquirePanicThenRefuse {
    fn oracle(&mut self, _request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        Err(EngineFailure {
            status: EngineStatus::from_code(5),
            message: "oracle refused this geometry".into(),
        })
    }

    fn on_acquire(&mut self, _reason: AcquireReason) {
        panic!("acquisition panic for fixture")
    }
}

#[test]
fn a_recoverable_oracle_failure_does_not_replace_an_earlier_acquisition_panic() {
    let path = build("harmonic_acquire_panic_then_refuse", &[]);
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(AcquirePanicThenRefuse)))
        .expect("set callbacks");
    let error = session
        .force(
            &EngineRequest {
                positions: &[1.0, 2.0, 3.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect_err("acquisition panic is fatal");
    assert_eq!(last_oracle_rc(&path), 5);
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "acquisition callback panicked".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

/// Fails each call with the next status of `script`.
struct ScriptedTransform {
    script: Vec<EngineFailure>,
}

impl CoordinateTransform for ScriptedTransform {
    fn transform(
        &mut self,
        _positions: &mut [f64],
        _box_matrix: &mut [f64; 9],
    ) -> Result<(), EngineFailure> {
        Err(self.script.remove(0))
    }
}

#[test]
fn a_later_fatal_transform_failure_wins_over_an_earlier_recoverable_one() {
    let path = build(
        "harmonic_batch_transform_recoverable_then_fatal",
        &["BATCH_ALL_SYSTEMS"],
    );
    let mut session = load_session(&path);
    let mut transform = ScriptedTransform {
        script: vec![
            EngineFailure {
                status: EngineStatus::from_code(4),
                message: "first system refused".into(),
            },
            EngineFailure {
                status: EngineStatus::from_code(-6),
                message: "transform lost its frame".into(),
            },
        ],
    };
    let boxes: Vec<f64> = CELL.iter().chain(CELL.iter()).copied().collect();
    let error = session
        .force_batch(
            &[1],
            &[1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            &boxes,
            Some(&mut transform),
        )
        .expect_err("fatal transform failure");
    assert!(transform.script.is_empty());
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -6,
            message: "transform lost its frame".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

#[test]
fn a_fatal_engine_code_wins_over_a_recoverable_transform_failure() {
    let path = build(
        "harmonic_transform_recoverable_engine_fatal",
        &["FATAL_AFTER_HOST_FAILURE"],
    );
    let mut session = load_session(&path);
    let mut transform = ScriptedTransform {
        script: vec![EngineFailure {
            status: EngineStatus::from_code(4),
            message: "transform refused".into(),
        }],
    };
    let error = session
        .force(
            &EngineRequest {
                positions: &[1.0, 2.0, 3.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            Some(&mut transform),
        )
        .expect_err("fatal engine code");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -8,
            message: "host failure left the engine unusable".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

#[test]
fn a_fatal_engine_code_wins_over_a_recoverable_oracle_failure() {
    let (path, mut session, answer) = oracle_force_with(
        "harmonic_oracle_recoverable_engine_fatal",
        &["FATAL_AFTER_HOST_FAILURE"],
        Err(EngineFailure {
            status: EngineStatus::from_code(5),
            message: "oracle refused this geometry".into(),
        }),
    );
    assert_eq!(last_oracle_rc(&path), 5);
    let error = answer.expect_err("fatal engine code");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -8,
            message: "host failure left the engine unusable".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

#[test]
fn result_cache_queries_preserve_native_status_and_counter_width() {
    use rgpot_core::engine::EngineCacheCounts;

    let mut engine = load(&build("harmonic_cache", &["FIXTURE_CACHE"]));
    assert_eq!(
        engine.cache_counts().unwrap_err(),
        EngineError::Unsupported {
            operation: "rgpot_engine_cache_counts",
            message: "cache detached".into()
        }
    );
    engine
        .set_cache(Some(Path::new("cache/geometry results")))
        .unwrap();
    assert_eq!(
        engine.cache_counts().unwrap(),
        EngineCacheCounts {
            computed: 1_u64 << 32,
            served: 7,
        }
    );
    engine.set_cache(None).unwrap();
    assert!(matches!(
        engine.cache_counts(),
        Err(EngineError::Unsupported { .. })
    ));

    let mut older = load(&build("harmonic_no_cache", &["FIXTURE_ABI_MINOR=5"]));
    assert!(matches!(
        older.set_cache(None),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_set_cache",
            ..
        })
    ));
    assert!(matches!(
        older.cache_counts(),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_cache_counts",
            ..
        })
    ));
}

const CACHE_DIRECTORY: &str = "cache/geometry results";

fn one_atom_request() -> EngineRequest<'static> {
    EngineRequest {
        positions: &[1.0, 2.0, 3.0],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    }
}

fn jet_call(
    engine: &mut EnginePotential,
    positions: &[f64],
    numbers: &[i32],
    directions: &[f64],
    order: usize,
) -> Result<GradientJet, EngineError> {
    assert!(!numbers.is_empty() && positions.len().is_multiple_of(3 * numbers.len()));
    let systems = positions.len() / (3 * numbers.len());
    let boxes: Vec<f64> = CELL.iter().copied().cycle().take(9 * systems).collect();
    engine.taylor_jet(
        &JetBatchRequest {
            atomic_numbers: numbers,
            positions,
            directions,
            boxes: &boxes,
        },
        order,
    )
}

fn hvp_request<'a>(positions: &'a [f64], numbers: &'a [i32]) -> EngineRequest<'a> {
    EngineRequest {
        positions,
        atomic_numbers: numbers,
        box_matrix: &CELL,
    }
}

#[test]
fn a_session_attaches_queries_and_clears_a_result_cache() {
    use rgpot_core::engine::EngineCacheCounts;

    let mut session = load_session(&build("harmonic_session_cache", &["FIXTURE_CACHE"]));
    assert_eq!(
        session.cache_counts().unwrap_err(),
        EngineError::Unsupported {
            operation: "rgpot_engine_cache_counts",
            message: "cache detached".into()
        }
    );

    session.attach_cache(Path::new(CACHE_DIRECTORY)).unwrap();
    assert_eq!(
        session.cache_counts().unwrap(),
        EngineCacheCounts {
            computed: 1_u64 << 32,
            served: 7,
        }
    );
    session
        .force(&one_atom_request(), None)
        .expect("an evaluation with a cache attached");

    session.clear_cache().unwrap();
    assert!(matches!(
        session.cache_counts(),
        Err(EngineError::Unsupported { .. })
    ));
    session
        .force(&one_atom_request(), None)
        .expect("an evaluation after the cache is cleared");
}

#[test]
fn a_refused_cache_attachment_is_recoverable_and_keeps_the_session_usable() {
    let mut session = load_session(&build("harmonic_session_refused", &["FIXTURE_CACHE"]));
    let refused = session
        .attach_cache(Path::new("another/directory"))
        .expect_err("the fixture refuses an unexpected directory");
    assert!(
        matches!(&refused, EngineError::Recoverable { code, .. } if *code == ENGINE_ERR_INVALID),
        "{refused:?}"
    );
    assert!(session.cache_counts().is_err());
    session
        .force(&one_atom_request(), None)
        .expect("a refused attachment does not poison the session");
    session.attach_cache(Path::new(CACHE_DIRECTORY)).unwrap();
    assert_eq!(session.cache_counts().unwrap().served, 7);
}

#[test]
fn a_cache_path_with_a_null_byte_never_reaches_the_engine() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let mut session = load_session(&build("harmonic_session_null_path", &["FIXTURE_CACHE"]));
    let path = Path::new(OsStr::from_bytes(b"cache/geometry\0results"));
    assert_eq!(
        session.attach_cache(path).unwrap_err(),
        EngineError::Invalid("cache path contains a null byte".into())
    );
    session
        .force(&one_atom_request(), None)
        .expect("the session stays usable");
}

#[test]
fn a_session_on_an_engine_without_cache_exports_reports_unsupported() {
    let mut session = load_session(&build("harmonic_session_no_cache", &[]));
    assert!(matches!(
        session.attach_cache(Path::new(CACHE_DIRECTORY)),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_set_cache",
            ..
        })
    ));
    assert!(matches!(
        session.clear_cache(),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_set_cache",
            ..
        })
    ));
    assert!(matches!(
        session.cache_counts(),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_cache_counts",
            ..
        })
    ));
    session
        .force(&one_atom_request(), None)
        .expect("an engine without a cache still evaluates");
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
    let handle = load(&null_hvp_with_caps).into_potential();
    assert!(!handle.as_handle().has_hessian_vector_callback());

    let null_all_optional = build_with_abs_zero(
        "harmonic_abs0_all_opt",
        &["NO_BATCH", "NO_CAPS", "NO_LAST_ERROR", "NO_HVP"],
        &[
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
    let handle = load(&null_all_optional).into_potential();
    assert!(!handle.as_handle().has_hessian_vector_callback());

    // An ABS-zero stamp symbol is absent, and a loader that finds no stamp
    // treats the engine as minor 0, layout 1, which this loader still serves.
    let null_stamp = build_with_abs_zero(
        "harmonic_abs0_stamp",
        &["NO_STAMP"],
        &["rgpot_engine_abi_stamp", "rgpot_engine_abi_compatible"],
    );
    let unstamped = unsafe { EnginePotential::load(&null_stamp, &config()) }
        .expect("an ABS-zero stamp symbol loads as an unstamped engine");
    assert_eq!(unstamped.abi_stamp().abi_minor, 0);
    assert_eq!(unstamped.abi_stamp().layout_revision, 1);
    drop(unstamped);

    let null_force = build_with_abs_zero(
        "harmonic_abs0_force",
        &["NO_FORCE", "NO_BATCH"],
        &["rgpot_engine_force"],
    );
    let observer = FixtureObserver::open(&null_force);
    let before = observer.created();
    let err = unsafe { EnginePotential::load(&null_force, &config()) }
        .err()
        .expect("ABS-zero required rgpot_engine_force must fail load");
    assert_eq!(err, EngineError::Missing("rgpot_engine_force".into()));
    assert_eq!(observer.created(), before);

    // Build a standalone shared object where all required symbols are exported,
    // with one required symbol set to ABS 0 at a time, so every required lookup
    // is verified without invoking a null function pointer.
    let required_names = [
        "rgpot_engine_abi_version",
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
    let rejected = unsafe { EnginePotential::load(&caps_lock, &config()) }
        .err()
        .expect("a caps lock failure at load rejects the engine");
    assert_eq!(
        rejected,
        EngineError::Lock {
            operation: "rgpot_engine_caps",
            message: "fixture caps lock failed".into(),
        }
    );
}

#[test]
fn a_negative_caps_code_at_load_rejects_the_engine_and_frees_the_handle() {
    for (name, define, expected) in [
        (
            "harmonic_capsfatal",
            "CAPS_ERROR_CODE=-1",
            EngineError::Fatal {
                code: -1,
                message: "fixture caps failed".into(),
            },
        ),
        (
            "harmonic_capsfatal_other",
            "CAPS_ERROR_CODE=-7",
            EngineError::Fatal {
                code: -7,
                message: "fixture caps failed".into(),
            },
        ),
    ] {
        let path = build(name, &[define]);
        let observer = FixtureObserver::open(&path);
        let before_created = observer.created();
        let before_destroyed = observer.destroyed();
        let rejected = unsafe { EnginePotential::load(&path, &config()) }
            .err()
            .expect("a negative caps code at load rejects the engine");
        assert_eq!(rejected, expected);
        assert_eq!(rejected.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
        assert_eq!(observer.created(), before_created + 1);
        assert_eq!(
            observer.destroyed(),
            before_destroyed + 1,
            "the rejected handle must be destroyed exactly once"
        );
    }
}

#[test]
fn a_positive_caps_code_at_load_keeps_process_wide_serialization() {
    use std::sync::{Arc, Barrier};

    let path = build(
        "harmonic_capspositive",
        &[
            "CAPS_ERROR_CODE=1",
            "FIXTURE_REENTRANCY=1",
            "CONCURRENCY_PROBE",
        ],
    );
    let _observer = FixtureObserver::open(&path);
    fixture_reset_concurrent(&path);
    let gate = Arc::new(Barrier::new(3));
    std::thread::scope(|scope| {
        for idx in 0..3usize {
            let path_ref = &path;
            let gate = Arc::clone(&gate);
            scope.spawn(move || {
                let mut engine = load(path_ref);
                let factor = (idx + 1) as f64;
                let positions = [factor, -factor, 0.5 * factor];
                let request = EngineRequest {
                    positions: &positions,
                    atomic_numbers: &[1],
                    box_matrix: &CELL,
                };
                gate.wait();
                for _ in 0..4 {
                    engine
                        .force(&request)
                        .expect("force after a failed caps probe");
                }
            });
        }
    });
    assert_eq!(
        fixture_max_concurrent(&path),
        1,
        "a failed caps probe must not skip the process-wide lock"
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
    let observer = FixtureObserver::open(&path);
    let probe = load(&path);
    assert_eq!(probe.caps().expect("caps").reentrancy, 1);
    drop(probe);

    // 1. Overlapping force and analytic HVP calls on a single PotentialImpl handle
    //    must serialize through the handle mutex.
    fixture_reset_concurrent(&path);
    let shared_potential = load(&path).into_potential();
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
    let before_destroy = observer.destroyed();
    let handles: Vec<rgpot_core::potential::PotentialImpl> =
        (0..4).map(|_| load(&path).into_potential()).collect();
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
    assert_eq!(observer.destroyed(), before_destroy + 4);
}

#[test]
fn reentrancy_two_engines_serialize_across_handles() {
    use std::sync::{Arc, Barrier};

    let path = build(
        "harmonic_reentrancy2",
        &["FIXTURE_REENTRANCY=2", "CONCURRENCY_PROBE"],
    );
    let observer = FixtureObserver::open(&path);
    let probe = load(&path);
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
    let before_destroy = observer.destroyed();
    let handles: Vec<rgpot_core::potential::PotentialImpl> =
        (0..4).map(|_| load(&path).into_potential()).collect();
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
    assert_eq!(observer.destroyed(), before_destroy + 4);
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
    assert_eq!(word(8), COUPLED_WELL_CONFIG_VERSION, "version");
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

/// A version-1 client writes 48 bytes: magic, version 1,
/// atomic number, a zero reserved word, then the four parameters.
#[test]
fn coupled_well_accepts_a_version_one_message_with_the_same_answer() {
    assert_eq!(COUPLED_WELL_CONFIG_V1_SIZE, 48);
    let mut legacy = [0_u8; 48];
    legacy[0..4].copy_from_slice(&COUPLED_WELL_CONFIG_MAGIC.to_ne_bytes());
    legacy[4..8].copy_from_slice(&COUPLED_WELL_CONFIG_VERSION_V1.to_ne_bytes());
    legacy[8..12].copy_from_slice(&WELL_NUMBER.to_ne_bytes());
    legacy[16..24].copy_from_slice(&WELL_BARRIER.to_ne_bytes());
    legacy[24..32].copy_from_slice(&WELL_POSITION.to_ne_bytes());
    legacy[32..40].copy_from_slice(&WELL_SPRING.to_ne_bytes());
    legacy[40..48].copy_from_slice(&WELL_COUPLING.to_ne_bytes());

    let mut from_v1 = unsafe { EnginePotential::load(build_coupled_well(), &legacy) }
        .expect("v1 coupled-well message");
    let mut from_v2 = load_well();
    let direction = [0.5, -1.5, 4.0];
    for x in WELL_XS {
        for y in WELL_YS {
            let positions = [x, y, 7.0];
            let request = well_request(&positions);
            assert_eq!(
                from_v1.force(&request).expect("v1 force"),
                from_v2.force(&request).expect("v2 force")
            );
            assert_eq!(
                from_v1.hvp(&request, &direction).expect("v1 hvp"),
                from_v2.hvp(&request, &direction).expect("v2 hvp")
            );
        }
    }

    for len in [
        COUPLED_WELL_CONFIG_V1_SIZE as usize - 1,
        COUPLED_WELL_CONFIG_V1_SIZE as usize + 1,
        COUPLED_WELL_CONFIG_MIN_SIZE as usize,
    ] {
        let mut sized = vec![0_u8; len];
        let copy_len = len.min(legacy.len());
        sized[..copy_len].copy_from_slice(&legacy[..copy_len]);
        let wrong_len = unsafe { EnginePotential::load(build_coupled_well(), &sized) }
            .err()
            .unwrap_or_else(|| panic!("{len}-byte version-1 coupled-well message must be refused"));
        assert!(
            matches!(wrong_len, EngineError::Create(ref why) if why.contains("RgpotCoupledWellConfigV1")),
            "{len}-byte v1: {wrong_len}"
        );
    }

    for atomic_number in [WELL_NUMBER, 2_i32] {
        legacy[8..12].copy_from_slice(&atomic_number.to_ne_bytes());
        for bad_version in [
            COUPLED_WELL_CONFIG_MIN_SIZE - 1,
            COUPLED_WELL_CONFIG_VERSION,
        ] {
            legacy[4..8].copy_from_slice(&bad_version.to_ne_bytes());
            let unknown_v1 = unsafe { EnginePotential::load(build_coupled_well(), &legacy) }
                .err()
                .expect("a version-1 buffer with a non-V1 version must be refused");
            assert_eq!(
                unknown_v1,
                EngineError::Create("coupled-well config version is unknown".into()),
                "atomic_number={atomic_number}, version={bad_version}"
            );
        }
        // 99 is at least the version-2 size. atomic_number 2 makes the version
        // word 2, so the 48-byte buffer is shorter than config.size. Any other
        // atomic number leaves a version other than 2.
        legacy[4..8].copy_from_slice(&99_u32.to_ne_bytes());
        let sized = unsafe { EnginePotential::load(build_coupled_well(), &legacy) }
            .err()
            .expect("a 48-byte buffer with size 99 must be refused");
        let expected = if atomic_number == 2 {
            "coupled-well config buffer is shorter than config.size"
        } else {
            "coupled-well config version is unknown"
        };
        assert_eq!(
            sized,
            EngineError::Create(expected.into()),
            "atomic_number={atomic_number}, size=99"
        );
    }
    for bad_version in [COUPLED_WELL_CONFIG_VERSION_V1, 3_u32] {
        let mut unknown_v2 = well_config();
        unknown_v2.version = bad_version;
        let unknown_v2 =
            unsafe { EnginePotential::load(build_coupled_well(), unknown_v2.as_bytes()) }
                .err()
                .expect("a version-2 buffer with a non-V2 version must be refused");
        assert_eq!(
            unknown_v2,
            EngineError::Create("coupled-well config version is unknown".into()),
            "v2 version={bad_version}"
        );
    }
}

#[test]
fn potlibc_accepts_a_version_one_message_and_pins_its_exact_length() {
    assert_eq!(POTLIBC_CONFIG_V1_SIZE, 144);
    let mut legacy = [0_u8; 144];
    legacy[0..4].copy_from_slice(&POTLIBC_CONFIG_MAGIC.to_ne_bytes());
    legacy[4..8].copy_from_slice(&POTLIBC_CONFIG_VERSION_V1.to_ne_bytes());
    legacy[8..16].copy_from_slice(&0_u64.to_ne_bytes());
    legacy[16..24].copy_from_slice(b"brh2leps");

    let mut from_v1 = unsafe { EnginePotential::load(build_potlibc(), &legacy) }
        .expect("144-byte version-1 potlibc message");
    let v2 = PotlibcConfig::new("brh2leps", 0).expect("v2 config");
    let mut from_v2 =
        unsafe { EnginePotential::load(build_potlibc(), v2.as_bytes()) }.expect("v2 potlibc");
    let positions = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 2.5, 0.0, 0.0];
    let species = [35, 1, 1];
    let direction = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9];
    let request = EngineRequest {
        positions: &positions,
        atomic_numbers: &species,
        box_matrix: &CELL,
    };
    assert_eq!(
        from_v1.force(&request).expect("v1 force"),
        from_v2.force(&request).expect("v2 force")
    );
    assert_eq!(
        from_v1.hvp(&request, &direction).expect("v1 hvp"),
        from_v2.hvp(&request, &direction).expect("v2 hvp")
    );

    for len in [
        POTLIBC_CONFIG_V1_SIZE as usize - 1,
        POTLIBC_CONFIG_V1_SIZE as usize + 1,
        POTLIBC_CONFIG_MIN_SIZE as usize,
    ] {
        let mut sized = vec![0_u8; len];
        let copy_len = len.min(legacy.len());
        sized[..copy_len].copy_from_slice(&legacy[..copy_len]);
        let wrong_len = unsafe { EnginePotential::load(build_potlibc(), &sized) }
            .err()
            .unwrap_or_else(|| panic!("{len}-byte version-1 potlibc message must be refused"));
        assert!(
            matches!(wrong_len, EngineError::Create(ref why) if why.contains("RgpotPotlibcConfigV1")),
            "{len}-byte v1: {wrong_len}"
        );
    }

    for atom_count in [0_u64, 2_u64] {
        legacy[8..16].copy_from_slice(&atom_count.to_ne_bytes());
        for bad_version in [99_u32, POTLIBC_CONFIG_VERSION] {
            legacy[4..8].copy_from_slice(&bad_version.to_ne_bytes());
            let unknown_v1 = unsafe { EnginePotential::load(build_potlibc(), &legacy) }
                .err()
                .expect("a version-1 potlibc buffer with a non-V1 version must be refused");
            assert_eq!(
                unknown_v1,
                EngineError::Create("potlibc config version is unknown".into()),
                "atom_count={atom_count}, version={bad_version}"
            );
        }
        // 200 is a plausible version-2 size. atom_count 0 leaves version 0 at
        // offset 8, so the version is unknown. atom_count 2 leaves version 2
        // there, so the 144-byte buffer is a truncated version-2 message.
        legacy[4..8].copy_from_slice(&200_u32.to_ne_bytes());
        let sized = unsafe { EnginePotential::load(build_potlibc(), &legacy) }
            .err()
            .expect("a 144-byte buffer with size 200 must be refused");
        let expected = if atom_count == 2 {
            "potlibc config buffer is shorter than config.size"
        } else {
            "potlibc config version is unknown"
        };
        assert_eq!(
            sized,
            EngineError::Create(expected.into()),
            "atom_count={atom_count}, size=200"
        );
    }
    for bad_version in [POTLIBC_CONFIG_VERSION_V1, 3_u32] {
        let mut unknown_v2 = PotlibcConfig::new("brh2leps", 0).expect("v2 config");
        unknown_v2.version = bad_version;
        let unknown_v2 = unsafe { EnginePotential::load(build_potlibc(), unknown_v2.as_bytes()) }
            .err()
            .expect("a version-2 potlibc buffer with a non-V2 version must be refused");
        assert_eq!(
            unknown_v2,
            EngineError::Create("potlibc config version is unknown".into()),
            "v2 version={bad_version}"
        );
    }
}

/// A version-2 message cut to the version-1 length, with a size word of 56,
/// 152 or 200, is shorter than `config.size`. A genuine version-1 message
/// still decodes.
#[test]
fn a_v2_message_cut_to_the_v1_length_is_shorter_than_its_size() {
    let mut legacy = [0_u8; 48];
    legacy[0..4].copy_from_slice(&COUPLED_WELL_CONFIG_MAGIC.to_ne_bytes());
    legacy[4..8].copy_from_slice(&COUPLED_WELL_CONFIG_VERSION_V1.to_ne_bytes());
    legacy[8..12].copy_from_slice(&WELL_NUMBER.to_ne_bytes());
    legacy[16..24].copy_from_slice(&WELL_BARRIER.to_ne_bytes());
    legacy[24..32].copy_from_slice(&WELL_POSITION.to_ne_bytes());
    legacy[32..40].copy_from_slice(&WELL_SPRING.to_ne_bytes());
    legacy[40..48].copy_from_slice(&WELL_COUPLING.to_ne_bytes());
    unsafe { EnginePotential::load(build_coupled_well(), &legacy) }
        .expect("genuine coupled-well version-1 message");

    let well = well_config().as_bytes().to_vec();
    for size in [COUPLED_WELL_CONFIG_MIN_SIZE, 152_u32, 200_u32] {
        let mut truncated = well[..COUPLED_WELL_CONFIG_V1_SIZE as usize].to_vec();
        truncated[4..8].copy_from_slice(&size.to_ne_bytes());
        let refused = unsafe { EnginePotential::load(build_coupled_well(), &truncated) }
            .err()
            .expect("truncated coupled-well version-2 message");
        assert_eq!(
            refused,
            EngineError::Create("coupled-well config buffer is shorter than config.size".into()),
            "coupled-well size={size}"
        );
    }

    let mut legacy = [0_u8; 144];
    legacy[0..4].copy_from_slice(&POTLIBC_CONFIG_MAGIC.to_ne_bytes());
    legacy[4..8].copy_from_slice(&POTLIBC_CONFIG_VERSION_V1.to_ne_bytes());
    legacy[8..16].copy_from_slice(&0_u64.to_ne_bytes());
    legacy[16..24].copy_from_slice(b"brh2leps");
    unsafe { EnginePotential::load(build_potlibc(), &legacy) }
        .expect("genuine potlibc version-1 message");

    let pot = PotlibcConfig::new("brh2leps", 0)
        .expect("v2 config")
        .as_bytes()
        .to_vec();
    for size in [POTLIBC_CONFIG_MIN_SIZE, 200_u32] {
        let mut truncated = pot[..POTLIBC_CONFIG_V1_SIZE as usize].to_vec();
        truncated[4..8].copy_from_slice(&size.to_ne_bytes());
        let refused = unsafe { EnginePotential::load(build_potlibc(), &truncated) }
            .err()
            .expect("truncated potlibc version-2 message");
        assert_eq!(
            refused,
            EngineError::Create("potlibc config buffer is shorter than config.size".into()),
            "potlibc size={size}"
        );
    }
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

    // The documented rule: equal major and layout, engine minor at most the host's.
    let accepts = |host: RgpotEngineAbiStamp| unsafe { compatible(&host) } != 0;
    assert!(accepts(host));
    assert!(accepts(RgpotEngineAbiStamp {
        abi_minor: host.abi_minor + 1,
        ..host
    }));
    assert!(!accepts(RgpotEngineAbiStamp {
        abi_minor: host.abi_minor - 1,
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

#[test]
fn taylor_jet_returns_the_harmonic_derivatives_and_the_report() {
    let path = build("harmonic_jet", &[]);
    let mut engine = load(&path);
    assert_eq!(engine.caps().expect("caps").jet_order, 3);
    let positions = [0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let direction = [1.0, -0.5, 0.25, -2.0, 0.0, 4.0];
    let jet = jet_call(&mut engine, &positions, &[1, 8], &direction, 3).expect("jet");
    assert_eq!(jet.order(), 3);
    assert_eq!(jet.n_systems(), 1);
    let expected: Vec<f64> = direction.iter().map(|d| SPRING * d).collect();
    assert!(all_within_roundoff(
        jet.derivative(1).expect("order 1"),
        &expected
    ));
    assert_eq!(jet.derivative(2), Some(&[0.0; 6][..]));
    assert_eq!(jet.derivative(3), Some(&[0.0; 6][..]));
    assert_eq!(jet.derivative(0), None);
    assert_eq!(jet.derivative(4), None);
    assert_eq!(
        jet.report(),
        Some(DerivativeReport {
            operations: 6,
            scale: SPRING * 4.0,
        })
    );
    let one = jet_call(&mut engine, &positions, &[1, 8], &direction, 1).expect("order one");
    assert_eq!(one.order(), 1);
    assert_eq!(one.derivative(1), jet.derivative(1));
    assert_eq!(one.derivative(2), None);
    let hvp = engine
        .hvp(&hvp_request(&positions, &[1, 8]), &direction)
        .expect("hvp");
    assert_eq!(Some(hvp.as_slice()), one.derivative(1));
}

#[test]
fn a_jet_batch_is_one_engine_call_with_summed_operations_and_largest_scale() {
    let path = build("harmonic_jet_batch", &[]);
    let mut engine = load(&path);
    let _ = engine.caps().expect("caps");
    let positions = [
        0.5, -1.0, 2.0, 0.0, 3.0, -0.25, 1.0, 1.0, 1.0, -2.0, 0.5, 0.0,
    ];
    let directions = [
        1.0, -0.5, 0.25, -2.0, 0.0, 4.0, 0.5, 0.5, 0.5, 8.0, 0.0, -1.0,
    ];
    let before = native_calls(&path);
    let jet = jet_call(&mut engine, &positions, &[1, 8], &directions, 2).expect("batch");
    assert_eq!(native_calls(&path) - before, 1, "one engine call");
    assert_eq!(jet.n_systems(), 2);
    let first: Vec<f64> = directions[..6].iter().map(|d| SPRING * d).collect();
    let second: Vec<f64> = directions[6..].iter().map(|d| SPRING * d).collect();
    assert!(all_within_roundoff(
        jet.system_derivative(1, 0).expect("system 0"),
        &first
    ));
    assert!(all_within_roundoff(
        jet.system_derivative(1, 1).expect("system 1"),
        &second
    ));
    assert_eq!(jet.system_derivative(1, 2), None);
    assert_eq!(jet.derivative(2).map(<[f64]>::len), Some(12));
    assert_eq!(
        jet.report(),
        Some(DerivativeReport {
            operations: 12,
            scale: SPRING * 8.0,
        })
    );
    let boxes = [CELL[0]; 9 * 3];
    let err = engine
        .taylor_jet(
            &JetBatchRequest {
                atomic_numbers: &[1, 8],
                positions: &positions,
                directions: &directions,
                boxes: &boxes,
            },
            1,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
}

#[test]
fn taylor_jet_without_a_report_is_reported_as_untracked() {
    let path = build("harmonic_jet_untracked", &["JET_UNTRACKED"]);
    let mut engine = load(&path);
    let jet = jet_call(&mut engine, &[0.5; 3], &[1], &[1.0, 0.0, 0.0], 2).expect("jet");
    assert_eq!(jet.report(), None);
}

#[test]
fn taylor_jet_absence_is_a_typed_status_and_never_a_fallback() {
    let direction = [1.0, 0.0, 0.0];
    let expect_unsupported = |engine: &mut EnginePotential, order: usize| {
        let err = jet_call(engine, &[0.0; 3], &[1], &direction, order).unwrap_err();
        assert!(
            matches!(
                err,
                EngineError::Unsupported {
                    operation: "rgpot_engine_taylor_jet",
                    ..
                }
            ),
            "{err:?}"
        );
        assert_eq!(err.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
    };

    let no_symbol = build("harmonic_nojet", &["NO_JET"]);
    let mut engine = load(&no_symbol);
    assert_eq!(engine.caps().expect("caps").jet_order, 0);
    expect_unsupported(&mut engine, 1);

    let no_capability = build("harmonic_nojetcap", &["NO_JET_CAP"]);
    let mut engine = load(&no_capability);
    assert_eq!(engine.caps().expect("caps").jet_order, 0);
    expect_unsupported(&mut engine, 1);

    let order_two = build("harmonic_jetorder2", &["FIXTURE_JET_ORDER=2"]);
    let mut engine = load(&order_two);
    assert_eq!(engine.caps().expect("caps").jet_order, 2);
    let before = native_calls(&order_two);
    expect_unsupported(&mut engine, 3);
    assert_eq!(
        native_calls(&order_two) - before,
        0,
        "the capabilities are queried once per instance, so a refused order never reaches the engine"
    );
    assert!(jet_call(&mut engine, &[0.0; 3], &[1], &direction, 2).is_ok());

    let refused = build("harmonic_jetrefused", &["JET_UNSUPPORTED"]);
    let mut engine = load(&refused);
    let err = jet_call(&mut engine, &[0.0; 3], &[1], &direction, 1).unwrap_err();
    assert_eq!(
        err,
        EngineError::Unsupported {
            operation: "rgpot_engine_taylor_jet",
            message: "harmonic engine built without a jet".into(),
        }
    );

    let no_caps = build("harmonic_jetnocaps", &["NO_CAPS"]);
    let mut engine = load(&no_caps);
    assert_eq!(
        jet_call(&mut engine, &[0.0; 3], &[1], &direction, 1).unwrap_err(),
        EngineError::Unsupported {
            operation: "rgpot_engine_taylor_jet",
            message: "rgpot_engine_caps is not exported".into(),
        }
    );
}

#[test]
fn a_minor_four_engine_without_the_jet_loads_and_refuses_only_the_jet() {
    let path = build(
        "harmonic_minor4",
        &["FIXTURE_ABI_MINOR=4", "NO_JET", "FIXTURE_CAPS_MINOR4"],
    );
    let mut engine = load(&path);
    assert_eq!(
        engine.abi_stamp(),
        RgpotEngineAbiStamp {
            abi_major: ENGINE_ABI_VERSION as u16,
            abi_minor: 4,
            layout_revision: ENGINE_ABI_LAYOUT_REVISION,
        }
    );
    assert!(engine.abi_stamp().serves(RgpotEngineAbiStamp::HOST));
    let caps = engine.caps().expect("caps of a minor-4 engine");
    assert!(caps.analytic_hvp);
    assert_eq!(
        caps.jet_order, 0,
        "a caps struct without the field serves no jet"
    );
    let answer = engine
        .force(&hvp_request(&[1.0, 0.0, 0.0], &[1]))
        .expect("force on a minor-4 engine");
    assert!(within_roundoff(answer.energy, 0.5 * SPRING, 3));
    let hvp = engine
        .hvp(&hvp_request(&[1.0, 0.0, 0.0], &[1]), &[1.0, 0.0, 0.0])
        .expect("hvp on a minor-4 engine");
    assert!(within_roundoff(hvp[0], SPRING, 3));
    let err = jet_call(&mut engine, &[1.0, 0.0, 0.0], &[1], &[1.0, 0.0, 0.0], 1).unwrap_err();
    assert!(
        matches!(
            err,
            EngineError::Unsupported {
                operation: "rgpot_engine_taylor_jet",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn taylor_jet_refuses_malformed_requests_before_the_engine() {
    let path = build("harmonic_jetinvalid", &[]);
    let mut engine = load(&path);
    let before = native_calls(&path);
    for (direction, order) in [
        (&[1.0, 0.0][..], 1),
        (&[1.0, f64::NAN, 0.0][..], 1),
        (&[1.0, 0.0, 0.0][..], 0),
    ] {
        let err = jet_call(&mut engine, &[0.0; 3], &[1], direction, order).unwrap_err();
        assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
    }
    let err = jet_call(&mut engine, &[f64::INFINITY, 0.0, 0.0], &[1], &[1.0; 3], 1).unwrap_err();
    assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
    assert_eq!(
        native_calls(&path) - before,
        1,
        "one capability query, cached for the rest"
    );
}

#[test]
fn taylor_jet_maps_recoverable_codes() {
    let path = build("harmonic_jetrecoverable", &[]);
    let mut engine = load(&path);
    let err = jet_call(&mut engine, &[0.0; 3], &[0], &[1.0, 0.0, 0.0], 1).unwrap_err();
    assert!(
        matches!(&err, EngineError::Recoverable { code: 1, .. }),
        "{err:?}"
    );
}

#[test]
fn a_jet_the_engine_cannot_vouch_for_fails_closed() {
    let direction = [1.0, 0.0, 0.0];
    for (name, define, what) in [
        (
            "harmonic_jet_nan_output",
            "JET_NONFINITE_OUTPUT=NAN",
            "the engine jet at element 2 is not finite",
        ),
        (
            "harmonic_jet_inf_output",
            "JET_NONFINITE_OUTPUT=INFINITY",
            "the engine jet at element 2 is not finite",
        ),
        (
            "harmonic_jet_nan_scale",
            "JET_NONFINITE_SCALE",
            "the engine jet scale is not finite",
        ),
        (
            "harmonic_jet_negative_operations",
            "JET_NEGATIVE_OPERATIONS",
            "the engine reports a negative operation count -1",
        ),
        (
            "harmonic_jet_negative_scale",
            "JET_NEGATIVE_SCALE",
            "the engine jet scale is negative",
        ),
    ] {
        let path = build(name, &[define, "FIXTURE_JET_ORDER=1"]);
        let mut engine = load(&path);
        let error = jet_call(&mut engine, &[0.0; 3], &[1], &direction, 1)
            .expect_err("a jet that is not finite or not countable is refused");
        assert_eq!(error, EngineError::Invalid(what.into()), "{name}");
        assert_eq!(error.status(), rgpot_status_t::RGPOT_INVALID_PARAMETER);
        let mut session = load_session(&path);
        assert_eq!(
            session_jet(&mut session, 1).expect_err("session refuses it too"),
            error,
            "{name}"
        );
    }

    let negative = build("harmonic_jet_negative_cap", &["JET_NEGATIVE_CAP"]);
    let mut engine = load(&negative);
    let error = engine.caps().expect_err("a negative jet order is refused");
    assert_eq!(
        error,
        EngineError::Invalid("the engine reports a negative jet order -1".into())
    );
    assert_eq!(
        jet_call(&mut engine, &[0.0; 3], &[1], &direction, 1),
        Err(error)
    );
}

#[test]
fn a_non_finite_box_is_refused_before_the_engine() {
    let path = build("harmonic_jet_nonfinite_box", &[]);
    let mut engine = load(&path);
    let _ = engine.caps().expect("caps");
    let before = native_calls(&path);
    let positions = [0.0; 3];
    let directions = [1.0, 0.0, 0.0];
    for (index, value, what) in [
        (
            0_usize,
            f64::NAN,
            "the engine box at element 0 is not finite",
        ),
        (
            8_usize,
            f64::INFINITY,
            "the engine box at element 8 is not finite",
        ),
        (
            4_usize,
            f64::NEG_INFINITY,
            "the engine box at element 4 is not finite",
        ),
    ] {
        let mut boxes = CELL;
        boxes[index] = value;
        assert_eq!(
            engine.taylor_jet(
                &JetBatchRequest {
                    atomic_numbers: &[1],
                    positions: &positions,
                    directions: &directions,
                    boxes: &boxes,
                },
                1,
            ),
            Err(EngineError::Invalid(what.into())),
            "box element {index}"
        );
    }
    assert_eq!(native_calls(&path), before);
    let mut session = load_session(&path);
    let mut boxes = CELL;
    boxes[2] = f64::NAN;
    assert_eq!(
        session.taylor_jet(
            &JetBatchRequest {
                atomic_numbers: &[1],
                positions: &positions,
                directions: &directions,
                boxes: &boxes,
            },
            1,
        ),
        Err(EngineError::Invalid(
            "the engine box at element 2 is not finite".into()
        ))
    );
}

#[test]
fn a_non_finite_jet_input_is_refused_before_the_engine() {
    let path = build("harmonic_jet_nonfinite_input", &[]);
    let mut engine = load(&path);
    let positions = [0.0; 3];
    let _ = engine.caps().expect("caps");
    let before = native_calls(&path);
    for (positions, directions, what) in [
        (
            [f64::NAN, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            "the engine position at element 0 is not finite",
        ),
        (
            positions,
            [0.0, f64::NAN, 0.0],
            "the engine direction at element 1 is not finite",
        ),
        (
            positions,
            [0.0, 0.0, f64::INFINITY],
            "the engine direction at element 2 is not finite",
        ),
    ] {
        assert_eq!(
            jet_call(&mut engine, &positions, &[1], &directions, 1),
            Err(EngineError::Invalid(what.into()))
        );
    }
    assert_eq!(native_calls(&path), before);
}

#[test]
fn caps_carry_every_field_and_ignore_bytes_past_the_reported_size() {
    let path = build("harmonic_caps_fields", &[]);
    assert_eq!(
        load(&path).caps().expect("caps"),
        EngineCaps {
            reentrancy: 0,
            per_image_instances: false,
            batched: false,
            periodic: true,
            stress: false,
            group_collective: false,
            analytic_hvp: true,
            jet_order: 3,
        }
    );

    let dirty = build(
        "harmonic_caps_dirty_tail",
        &[
            "FIXTURE_ABI_MINOR=4",
            "FIXTURE_CAPS_MINOR4",
            "FIXTURE_CAPS_DIRTY_TAIL",
        ],
    );
    let mut engine = load(&dirty);
    assert_eq!(engine.caps().expect("caps").jet_order, 0);
    assert!(matches!(
        jet_call(&mut engine, &[0.0; 3], &[1], &[1.0, 0.0, 0.0], 1),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_taylor_jet",
            ..
        })
    ));
}

#[test]
fn a_jet_request_that_is_not_whole_systems_is_refused_before_the_engine() {
    let path = build("harmonic_jet_shapes", &[]);
    let mut engine = load(&path);
    let _ = engine.caps().expect("caps");
    let before = native_calls(&path);
    let zeros = [0.0; 9];
    for (numbers, coordinates) in [(&[][..], 3), (&[1][..], 0), (&[1][..], 4), (&[1, 8][..], 7)] {
        let error = engine
            .taylor_jet(
                &JetBatchRequest {
                    atomic_numbers: numbers,
                    positions: &zeros[..coordinates],
                    directions: &zeros[..coordinates],
                    boxes: &CELL,
                },
                1,
            )
            .expect_err("not a whole number of systems");
        assert!(
            matches!(&error, EngineError::Invalid(text) if text.contains("are not a whole number")),
            "{numbers:?} with {coordinates} coordinates: {error:?}"
        );
    }
    assert_eq!(native_calls(&path), before);
}

fn session_jet(session: &mut EngineSession, order: usize) -> Result<GradientJet, EngineError> {
    session_jet_of(session, &[1.0, -0.5, 0.25], order)
}

fn session_jet_of(
    session: &mut EngineSession,
    directions: &[f64],
    order: usize,
) -> Result<GradientJet, EngineError> {
    session.taylor_jet(
        &JetBatchRequest {
            atomic_numbers: &[1],
            positions: &[0.5, -1.0, 2.0],
            directions,
            boxes: &CELL,
        },
        order,
    )
}

fn call_oracle_outside_a_call(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let call: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_call_oracle_outside_a_call\0") }.expect("oracle hook");
    unsafe { call() }
}

fn session_with_oracle(
    name: &str,
    defines: &[&str],
    answer: Result<EngineEvaluation, EngineFailure>,
) -> (PathBuf, EngineSession) {
    let path = build(name, defines);
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(ScriptedOracle { answer })))
        .expect("set callbacks");
    (path, session)
}

#[test]
fn a_session_serves_the_jet_the_engine_serves() {
    let path = build("harmonic_session_jet", &[]);
    let mut session = load_session(&path);
    let jet = session_jet(&mut session, 3).expect("session jet");
    assert_eq!(jet.order(), 3);
    assert_eq!(jet.n_systems(), 1);
    assert!(all_within_roundoff(
        jet.derivative(1).expect("order 1"),
        &[SPRING, -0.5 * SPRING, 0.25 * SPRING]
    ));
    assert_eq!(jet.derivative(2), Some(&[0.0; 3][..]));
    assert_eq!(jet.derivative(3), Some(&[0.0; 3][..]));
    assert_eq!(
        jet.report(),
        Some(DerivativeReport {
            operations: 3,
            scale: SPRING * 2.0,
        })
    );

    let before = native_calls(&path);
    assert!(matches!(
        session_jet(&mut session, 4),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_taylor_jet",
            ..
        })
    ));
    assert!(matches!(
        session_jet_of(&mut session, &[1.0, 0.0], 1),
        Err(EngineError::Invalid(_))
    ));
    assert_eq!(
        native_calls(&path) - before,
        0,
        "the capabilities are cached, so a refused request never reaches the engine"
    );

    let no_jet = build("harmonic_session_nojet", &["NO_JET"]);
    let mut session = load_session(&no_jet);
    assert!(matches!(
        session_jet(&mut session, 1),
        Err(EngineError::Unsupported {
            operation: "rgpot_engine_taylor_jet",
            ..
        })
    ));
}

#[test]
fn a_session_jet_reaches_the_registered_oracle_and_succeeds_when_it_answers() {
    let (path, mut session) = session_with_oracle(
        "harmonic_session_jet_oracle",
        &[],
        Ok(EngineEvaluation {
            energy: 0.0,
            forces: vec![0.0; 3],
        }),
    );
    let jet = session_jet(&mut session, 1).expect("jet through a registered oracle");
    assert_eq!(last_oracle_rc(&path), 0);
    assert_eq!(jet.order(), 1);
}

#[test]
fn a_fatal_oracle_failure_during_a_jet_is_latched_over_the_engines_own_fatal_code() {
    let (path, mut session) = session_with_oracle(
        "harmonic_session_jet_fatal",
        &["FATAL_AFTER_HOST_FAILURE"],
        Err(EngineFailure {
            status: EngineStatus::from_code(-9),
            message: "oracle lost its backend".into(),
        }),
    );
    let error = session_jet(&mut session, 1).expect_err("fatal oracle failure");
    assert_eq!(last_oracle_rc(&path), -9);
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -9,
            message: "oracle lost its backend".into(),
        }
    );
    let before = native_calls(&path);
    for _ in 0..2 {
        assert_eq!(session_jet(&mut session, 1), Err(error.clone()));
    }
    assert_eq!(native_calls(&path), before);
    assert_poisoned_without_an_extra_force(&mut session, &path, &error);
}

#[test]
fn a_recoverable_oracle_failure_during_a_jet_is_not_an_unsupported_jet() {
    let (path, mut session) = session_with_oracle(
        "harmonic_session_jet_recoverable",
        &[],
        Err(EngineFailure {
            status: EngineStatus::from_code(ENGINE_ERR_UNSUPPORTED),
            message: "oracle declined this geometry".into(),
        }),
    );
    let error = session_jet(&mut session, 1).expect_err("recoverable oracle failure");
    assert_eq!(last_oracle_rc(&path), ENGINE_ERR_UNSUPPORTED);
    assert_eq!(
        error,
        EngineError::Recoverable {
            code: ENGINE_ERR_UNSUPPORTED,
            message: "oracle declined this geometry".into(),
        }
    );
    session.set_callbacks(None).expect("clear callbacks");
    assert!(
        session_jet(&mut session, 1).is_ok(),
        "the session is usable"
    );
}

#[test]
fn a_fatal_callback_failure_no_call_has_settled_wins_over_a_refused_jet() {
    let (path, mut session) = session_with_oracle(
        "harmonic_session_jet_unsettled_fatal",
        &[],
        Err(EngineFailure {
            status: EngineStatus::from_code(-9),
            message: "oracle lost its backend".into(),
        }),
    );
    assert_eq!(call_oracle_outside_a_call(&path), -9);
    let before = native_calls(&path);
    let error = session_jet(&mut session, 0).expect_err("refused jet");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -9,
            message: "oracle lost its backend".into(),
        }
    );
    assert_eq!(native_calls(&path), before);
    assert_eq!(session_jet(&mut session, 1), Err(error));
}

#[cfg(feature = "schema")]
#[test]
fn builtin_params_is_a_flat_capnp_message_naming_the_backend() {
    use rgpot_core::Potentials_capnp::builtin_params;

    let bytes = builtin_params("cbe");
    assert_eq!(bytes.len() % 8, 0);
    let mut words = vec![0_u64; bytes.len() / 8];
    let (chunks, tail) = bytes.as_chunks::<8>();
    assert!(tail.is_empty());
    for (word, chunk) in words.iter_mut().zip(chunks) {
        *word = u64::from_ne_bytes(*chunk);
    }
    let slice = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), bytes.len()) };
    let reader = capnp::serialize::read_message_from_flat_slice(
        &mut &slice[..],
        capnp::message::ReaderOptions::new(),
    )
    .expect("flat message");
    let params: builtin_params::Reader<'_> = reader.get_root().expect("BuiltinParams");
    assert_eq!(
        params.get_name().expect("name").to_str().expect("utf8"),
        "cbe"
    );
}

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
