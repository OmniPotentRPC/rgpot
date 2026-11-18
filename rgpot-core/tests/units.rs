// MIT License
// Copyright 2023--present rgpot developers

#[allow(dead_code)]
#[path = "../cxx_host.rs"]
mod cxx_host;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use rgpot_core::units;

static NEXT_SCRATCH_ID: AtomicU64 = AtomicU64::new(0);

struct ScratchDir {
    dir: PathBuf,
    files: Vec<PathBuf>,
}

impl ScratchDir {
    fn new(tag: &str) -> Self {
        let id = NEXT_SCRATCH_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rgpot-units-test-{}-{}-{}",
            tag,
            std::process::id(),
            id
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self {
            dir,
            files: Vec::new(),
        }
    }

    fn track(&mut self, path: PathBuf) -> PathBuf {
        self.files.push(path.clone());
        path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        for file in self.files.iter().rev() {
            let _ = std::fs::remove_file(file);
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn cpp_core_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("CppCore")
}

fn run_cpp_probe(tag: &str, cpp_source: &str, extra_sources: &[&Path], count: usize) -> Vec<f64> {
    let cpp_core = cpp_core_dir();
    let mut scratch = ScratchDir::new(tag);
    let src = scratch.track(scratch.dir.join("probe.cc"));
    let bin = scratch.track(cxx_host::executable_path(&scratch.dir, "probe"));
    let dump = scratch.track(scratch.dir.join("values.bin"));

    std::fs::write(&src, cpp_source).expect("write probe.cc");
    let mut sources: Vec<&Path> = vec![&src];
    sources.extend_from_slice(extra_sources);
    cxx_host::compile_cpp_binary(&cpp_core, &sources, &bin);

    let mut run = Command::new(&bin);
    cxx_host::configure_run_env(&mut run);
    run.arg(&dump);
    let out = run.output().expect("run probe");
    assert!(
        out.status.success(),
        "C++ probe failed ({}): {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&dump).expect("read values.bin");
    assert_eq!(bytes.len(), count * std::mem::size_of::<f64>());
    bytes
        .chunks_exact(std::mem::size_of::<f64>())
        .map(|chunk| {
            let mut arr = [0_u8; 8];
            arr.copy_from_slice(chunk);
            f64::from_ne_bytes(arr)
        })
        .collect()
}

#[test]
fn every_exported_constant_has_the_bits_of_units_hpp() {
    let probe = r#"
#include "rgpot/units.hpp"
#include <cstdio>

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const double values[13] = {
      rgpot::units::BOLTZMANN_CONSTANT_J_PER_K,
      rgpot::units::ELEMENTARY_CHARGE_C,
      rgpot::units::HARTREE_ENERGY_J,
      rgpot::units::REDUCED_PLANCK_CONSTANT_J_S,
      rgpot::units::BOHR_TO_ANGSTROM,
      rgpot::units::ANGSTROM_TO_BOHR,
      rgpot::units::HARTREE_TO_EV,
      rgpot::units::EV_TO_HARTREE,
      rgpot::units::KB_HARTREE,
      rgpot::units::KB_EV,
      rgpot::units::ELECTRON_MASS_IN_DALTON,
      rgpot::units::DALTON_IN_ELECTRON_MASSES,
      rgpot::units::ATOMIC_UNIT_OF_TIME_S,
  };
  std::FILE *fp = std::fopen(argv[1], "wb");
  if (!fp) return 3;
  const std::size_t wrote = std::fwrite(values, sizeof(double), 13, fp);
  std::fclose(fp);
  return wrote == 13 ? 0 : 4;
}
"#;
    let cpp = run_cpp_probe("header-bits", probe, &[], 13);
    let rust = [
        (
            "BOLTZMANN_CONSTANT_J_PER_K",
            units::BOLTZMANN_CONSTANT_J_PER_K,
        ),
        ("ELEMENTARY_CHARGE_C", units::ELEMENTARY_CHARGE_C),
        ("HARTREE_ENERGY_J", units::HARTREE_ENERGY_J),
        (
            "REDUCED_PLANCK_CONSTANT_J_S",
            units::REDUCED_PLANCK_CONSTANT_J_S,
        ),
        ("BOHR_TO_ANGSTROM", units::BOHR_TO_ANGSTROM),
        ("ANGSTROM_TO_BOHR", units::ANGSTROM_TO_BOHR),
        ("HARTREE_TO_EV", units::HARTREE_TO_EV),
        ("EV_TO_HARTREE", units::EV_TO_HARTREE),
        ("KB_HARTREE", units::KB_HARTREE),
        ("KB_EV", units::KB_EV),
        ("ELECTRON_MASS_IN_DALTON", units::ELECTRON_MASS_IN_DALTON),
        (
            "DALTON_IN_ELECTRON_MASSES",
            units::DALTON_IN_ELECTRON_MASSES,
        ),
        ("ATOMIC_UNIT_OF_TIME_S", units::ATOMIC_UNIT_OF_TIME_S),
    ];

    let header_text = std::fs::read_to_string(cpp_core_dir().join("rgpot").join("units.hpp"))
        .expect("read units.hpp");
    for ((name, got), want) in rust.into_iter().zip(cpp) {
        assert!(
            header_text.contains(&format!("inline constexpr double {name}")),
            "units.hpp lacks constexpr declaration for {name}"
        );
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{name} bits {:#018x} ({got:?}) != C++ header bits {:#018x} ({want:?})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

#[test]
fn exported_constants_agree_with_cpp_unit_parser() {
    let probe = r#"
#include "rgpot/units.hpp"
#include <cstdio>

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  using rgpot::units::unit_conversion_factor;
  const double values[6] = {
      unit_conversion_factor("bohr", "angstrom"),
      unit_conversion_factor("angstrom", "bohr"),
      unit_conversion_factor("hartree", "eV"),
      unit_conversion_factor("eV", "hartree"),
      unit_conversion_factor("dalton", "electron_mass"),
      unit_conversion_factor("hbar", "hartree*s"),
  };
  std::FILE *fp = std::fopen(argv[1], "wb");
  if (!fp) return 3;
  const std::size_t wrote = std::fwrite(values, sizeof(double), 6, fp);
  std::fclose(fp);
  return wrote == 6 ? 0 : 4;
}
"#;
    let parser = cpp_core_dir().join("rgpot").join("units.cc");
    let parsed = run_cpp_probe("parser-factors", probe, &[&parser], 6);
    let pairs = [
        ("BOHR_TO_ANGSTROM", units::BOHR_TO_ANGSTROM, parsed[0]),
        ("ANGSTROM_TO_BOHR", units::ANGSTROM_TO_BOHR, parsed[1]),
        ("HARTREE_TO_EV", units::HARTREE_TO_EV, parsed[2]),
        ("EV_TO_HARTREE", units::EV_TO_HARTREE, parsed[3]),
        (
            "DALTON_IN_ELECTRON_MASSES",
            units::DALTON_IN_ELECTRON_MASSES,
            parsed[4],
        ),
        (
            "ATOMIC_UNIT_OF_TIME_S",
            units::ATOMIC_UNIT_OF_TIME_S,
            parsed[5],
        ),
    ];
    for (name, got, from_parser) in pairs {
        let rel = ((got - from_parser) / got).abs();
        assert!(
            rel < 1e-11,
            "{name} ({got:?}) differs from unit_conversion_factor ({from_parser:?}): rel={rel:e}"
        );
    }
}

#[test]
fn codata_2018_values_and_identities_hold() {
    assert_eq!(
        units::BOHR_TO_ANGSTROM.to_bits(),
        0.529_177_210_903_f64.to_bits()
    );
    assert_eq!(
        units::ANGSTROM_TO_BOHR.to_bits(),
        (1.0_f64 / units::BOHR_TO_ANGSTROM).to_bits()
    );
    assert_eq!(
        units::HARTREE_TO_EV.to_bits(),
        27.211_386_245_988_f64.to_bits()
    );
    assert_eq!(
        units::EV_TO_HARTREE.to_bits(),
        (1.0_f64 / units::HARTREE_TO_EV).to_bits()
    );
    assert_eq!(
        units::KB_EV.to_bits(),
        (units::BOLTZMANN_CONSTANT_J_PER_K / units::ELEMENTARY_CHARGE_C).to_bits()
    );
    assert_eq!(
        units::KB_HARTREE.to_bits(),
        3.166_811_563_455_6e-6_f64.to_bits()
    );
    assert_eq!(
        units::DALTON_IN_ELECTRON_MASSES.to_bits(),
        (1.0_f64 / units::ELECTRON_MASS_IN_DALTON).to_bits()
    );
    assert_eq!(
        units::ATOMIC_UNIT_OF_TIME_S.to_bits(),
        (units::REDUCED_PLANCK_CONSTANT_J_S / units::HARTREE_ENERGY_J).to_bits()
    );

    // Exact 2019 SI k_B and elementary charge e against CODATA 2018 Hartree.
    let kb_si = units::BOLTZMANN_CONSTANT_J_PER_K;
    let elem_charge = units::ELEMENTARY_CHARGE_C;
    let planck_h = 6.626_070_15e-34_f64;
    let kb_hartree_from_si = (kb_si / elem_charge) / units::HARTREE_TO_EV;
    assert!(((units::KB_HARTREE - kb_hartree_from_si) / units::KB_HARTREE).abs() < 1e-14);

    let kb_hartree_from_kb_ev = units::KB_EV / units::HARTREE_TO_EV;
    assert!(((units::KB_HARTREE - kb_hartree_from_kb_ev) / units::KB_HARTREE).abs() < 2e-11);

    let hbar_si = planck_h / std::f64::consts::TAU;
    let au_time_from_si = hbar_si / (units::HARTREE_TO_EV * elem_charge);
    assert!(
        ((units::ATOMIC_UNIT_OF_TIME_S - au_time_from_si) / units::ATOMIC_UNIT_OF_TIME_S).abs()
            < 1e-13
    );
}

#[test]
fn linking_rgpot_core_c_abi_with_cpp_units_has_no_duplicate_symbols() {
    let cpp_core = cpp_core_dir();
    let parser = cpp_core.join("rgpot").join("units.cc");

    // `cargo test` does not necessarily emit the crate's staticlib. Build it
    // through Cargo in an isolated target directory so this check cannot pass
    // vacuously in a clean checkout or under nextest.
    let mut scratch = ScratchDir::new("no-dup-link");
    let cargo_target = scratch.dir.join("cargo-target");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let built = Command::new(cargo)
        .args(["build", "--lib", "--no-default-features", "--manifest-path"])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&cargo_target)
        .env_remove("CARGO_BUILD_TARGET")
        .output()
        .expect("invoke cargo to build librgpot_core static archive");
    assert!(
        built.status.success(),
        "could not build librgpot_core static archive required by the duplicate-symbol test:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&built.stdout),
        String::from_utf8_lossy(&built.stderr)
    );

    let profile_dir = cargo_target.join("debug");
    let staticlib = if cfg!(windows) {
        profile_dir.join("rgpot_core.lib")
    } else {
        profile_dir.join("librgpot_core.a")
    };
    assert!(
        staticlib.is_file(),
        "cargo succeeded but did not produce the required static archive at {}",
        staticlib.display()
    );
    let archive = std::fs::read(&staticlib).expect("read librgpot_core static archive");
    for forbidden in [
        b"unit_conversion_factor".as_slice(),
        b"validate_unit".as_slice(),
    ] {
        assert!(
            !archive
                .windows(forbidden.len())
                .any(|window| window == forbidden),
            "{} unexpectedly embeds C++ units symbol {:?}",
            staticlib.display(),
            std::str::from_utf8(forbidden)
        );
    }

    let src = scratch.track(scratch.dir.join("link_check.cc"));
    let bin = scratch.track(cxx_host::executable_path(&scratch.dir, "link_check"));
    std::fs::write(
        &src,
        r#"
#include "rgpot/units.hpp"
#include <cstring>

extern "C" const char *rgpot_version(void);

int main() {
  const char *ver = rgpot_version();
  double factor = rgpot::units::unit_conversion_factor("bohr", "angstrom");
  return (ver != nullptr && std::strlen(ver) > 0 && factor > 0.0) ? 0 : 1;
}
"#,
    )
    .expect("write link_check.cc");

    let (program, extra_args) = cxx_host::resolve_cxx();
    let mut cmd = Command::new(&program);
    cxx_host::configure_compile_env(&mut cmd);
    cmd.args(&extra_args)
        .args(["-std=c++17", "-O2", "-I"])
        .arg(&cpp_core)
        .arg(&src)
        .arg(&parser);
    if cfg!(target_os = "macos") {
        cmd.arg("-Wl,-force_load").arg(&staticlib);
    } else if cfg!(target_os = "linux") {
        cmd.arg("-Wl,--whole-archive")
            .arg(&staticlib)
            .arg("-Wl,--no-whole-archive");
    } else {
        cmd.arg(&staticlib);
    }
    if !cfg!(windows) {
        cmd.args(["-lpthread", "-ldl", "-lm"]);
    }
    cmd.arg("-o").arg(&bin);
    let compiled = cmd
        .output()
        .expect("link C++ units.cc with librgpot_core.a");
    assert!(
        compiled.status.success(),
        "linking units.cc together with librgpot_core.a failed:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );

    let mut run = Command::new(&bin);
    cxx_host::configure_run_env(&mut run);
    let out = run.output().expect("run link_check");
    assert!(out.status.success());
}
