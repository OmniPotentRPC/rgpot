// MIT License
// Copyright 2023--present rgpot developers

// Host C++17 compiler invocation shared by `build.rs` (`units` feature) and
// the `units` integration test. Kept separate from Cargo link directives so no
// C++ objects or runtime libraries are linked into `librgpot_core`.

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const ENV_KEYS: [&str; 5] = [
    "HOST_CXX",
    "CXX",
    "SDKROOT",
    "CONDA_PREFIX",
    "CONDA_BUILD_SYSROOT",
];

fn non_empty_env(key: &str) -> Option<String> {
    match env::var(key) {
        Ok(val) => {
            let trimmed = val.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_owned())
            }
        }
        Err(_) => None,
    }
}

fn conda_darwin_sysroot_missing() -> bool {
    match non_empty_env("CONDA_BUILD_SYSROOT") {
        Some(path) => !Path::new(&path).is_dir(),
        None => true,
    }
}

fn split_compiler_spec(spec: &str) -> Option<(String, Vec<String>)> {
    if Path::new(spec).is_file() {
        return Some((spec.to_owned(), Vec::new()));
    }
    let mut parts = spec.split_whitespace();
    let program = parts.next()?.to_owned();
    let args: Vec<String> = parts.map(str::to_owned).collect();
    Some((program, args))
}

fn probe_compiler(candidate: &str) -> bool {
    match Command::new(candidate).arg("--version").output() {
        Ok(out) => out.status.success(),
        Err(_) => false,
    }
}

pub fn resolve_cxx() -> (String, Vec<String>) {
    let configured = match non_empty_env("HOST_CXX") {
        Some(val) => Some(val),
        None => non_empty_env("CXX"),
    };
    if let Some(spec) = configured {
        if cfg!(target_os = "macos")
            && spec.contains("-apple-darwin")
            && conda_darwin_sysroot_missing()
            && Path::new("/usr/bin/clang++").is_file()
        {
            return ("/usr/bin/clang++".to_owned(), Vec::new());
        }
        if let Some(parsed) = split_compiler_spec(&spec) {
            return parsed;
        }
    }

    let candidates: [&str; 3] = if cfg!(target_os = "macos") {
        ["clang++", "c++", "g++"]
    } else {
        ["c++", "g++", "clang++"]
    };
    for candidate in candidates {
        if probe_compiler(candidate) {
            return (candidate.to_owned(), Vec::new());
        }
    }
    (candidates[0].to_owned(), Vec::new())
}

fn macos_sdkroot() -> Option<String> {
    if let Some(existing) = non_empty_env("SDKROOT") {
        if Path::new(&existing).is_dir() {
            return Some(existing);
        }
    }
    let out = Command::new("xcrun").arg("--show-sdk-path").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let sdk = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    if sdk.is_empty() || !Path::new(&sdk).is_dir() {
        None
    } else {
        Some(sdk)
    }
}

pub fn configure_compile_env(cmd: &mut Command) {
    if cfg!(target_os = "macos") {
        if conda_darwin_sysroot_missing() {
            cmd.env_remove("CONDA_BUILD_SYSROOT");
        }
        if let Some(sdk) = macos_sdkroot() {
            cmd.env("SDKROOT", sdk);
        }
    }
}

fn prepend_library_dir(var: &str, dir: &Path) -> Option<OsString> {
    let mut paths = vec![dir.to_path_buf()];
    if let Some(existing) = env::var_os(var) {
        for entry in env::split_paths(&existing) {
            if !entry.as_os_str().is_empty() {
                paths.push(entry);
            }
        }
    }
    env::join_paths(paths).ok()
}

pub fn configure_run_env(cmd: &mut Command) {
    if let Some(prefix) = non_empty_env("CONDA_PREFIX") {
        let lib_dir = Path::new(&prefix).join("lib");
        if lib_dir.is_dir() {
            let key = if cfg!(target_os = "macos") {
                "DYLD_FALLBACK_LIBRARY_PATH"
            } else {
                "LD_LIBRARY_PATH"
            };
            if let Some(joined) = prepend_library_dir(key, &lib_dir) {
                cmd.env(key, joined);
            }
        }
    }
}

pub fn executable_path(dir: &Path, stem: &str) -> PathBuf {
    let mut path = dir.join(stem);
    if !env::consts::EXE_EXTENSION.is_empty() {
        path.set_extension(env::consts::EXE_EXTENSION);
    }
    path
}

pub fn compile_cpp_binary(include_dir: &Path, sources: &[&Path], output_bin: &Path) {
    for src in sources {
        if !src.is_file() {
            panic!(
                "missing C++ source required for units export: {}",
                src.display()
            );
        }
    }
    let (program, extra_args) = resolve_cxx();
    let mut cmd = Command::new(&program);
    configure_compile_env(&mut cmd);
    cmd.args(&extra_args)
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-I"])
        .arg(include_dir);
    for src in sources {
        cmd.arg(src);
    }
    cmd.arg("-o").arg(output_bin);

    let out = match cmd.output() {
        Ok(out) => out,
        Err(err) => panic!("failed to invoke C++ compiler `{program}`: {err}"),
    };
    if !out.status.success() {
        panic!(
            "C++ units compilation with `{program}` failed ({}):\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
