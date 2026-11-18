// MIT License
// Copyright 2023--present rgpot developers

// Host C++17 compiler invocation for the `units` integration test. Kept
// separate from Cargo link directives so no C++ objects or runtime libraries
// are linked into `librgpot_core`.

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

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

static NEXT_PROBE_ID: AtomicU64 = AtomicU64::new(0);

fn probe_compiler(program: &str, extra_args: &[String]) -> bool {
    let id = NEXT_PROBE_ID.fetch_add(1, Ordering::Relaxed);
    let stem = format!("rgpot-cxx-probe-{}-{id}", std::process::id());
    let source = env::temp_dir().join(format!("{stem}.cc"));
    let object = env::temp_dir().join(format!("{stem}.o"));
    if std::fs::write(
        &source,
        "#include <string>\nint main() { std::string value; return value.size(); }\n",
    )
    .is_err()
    {
        return false;
    }

    let mut cmd = Command::new(program);
    configure_compile_env(&mut cmd);
    let status = cmd
        .args(extra_args)
        .args(["-std=c++17", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status();
    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&object);
    matches!(status, Ok(status) if status.success())
}

fn select_fallback(candidates: &[&str]) -> Option<(String, Vec<String>)> {
    candidates.iter().find_map(|candidate| {
        let parsed = split_compiler_spec(candidate)?;
        if probe_compiler(&parsed.0, &parsed.1) {
            Some(parsed)
        } else {
            None
        }
    })
}

pub fn resolve_cxx() -> (String, Vec<String>) {
    let configured = match non_empty_env("HOST_CXX") {
        Some(val) => Some(val),
        None => non_empty_env("CXX"),
    };
    if let Some(spec) = configured {
        if let Some(parsed) = split_compiler_spec(&spec) {
            return parsed;
        }
    }

    let candidates: [&str; 3] = if cfg!(target_os = "macos") {
        ["clang++", "c++", "g++"]
    } else {
        ["c++", "g++", "clang++"]
    };
    if let Some(selected) = select_fallback(&candidates) {
        return selected;
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

#[cfg(all(test, unix))]
mod tests {
    use super::{probe_compiler, select_fallback};
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn write_executable(path: &Path, body: &str) {
        std::fs::write(path, body).expect("write fake compiler");
        let mut permissions = std::fs::metadata(path)
            .expect("stat fake compiler")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(path, permissions).expect("chmod fake compiler");
    }

    /// Another test thread may fork while the script is still open for
    /// writing, which makes the first `exec` fail with `ETXTBSY`. The window
    /// closes once that child has executed, so the probe waits it out and
    /// leaves the script runnable for the code under test.
    fn run_version_probe(path: &Path) -> std::process::ExitStatus {
        const ATTEMPTS: u32 = 200;
        const PAUSE: std::time::Duration = std::time::Duration::from_millis(10);
        for _ in 1..ATTEMPTS {
            match std::process::Command::new(path).arg("--version").status() {
                Err(error) if error.kind() == std::io::ErrorKind::ExecutableFileBusy => {
                    std::thread::sleep(PAUSE);
                }
                outcome => return outcome.expect("run fake compiler version probe"),
            }
        }
        std::process::Command::new(path)
            .arg("--version")
            .status()
            .expect("run fake compiler version probe")
    }

    #[test]
    fn fallback_rejects_compiler_without_standard_library_headers() {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("rgpot-cxx-host-test-{}-{id}", std::process::id()));
        std::fs::create_dir(&dir).expect("create compiler test directory");
        let bad = dir.join("c++");
        let good = dir.join("g++");
        write_executable(
            &bad,
            "#!/bin/sh\nif test \"${1-}\" = --version; then exit 0; fi\nexit 1\n",
        );
        write_executable(&good, "#!/bin/sh\nexec g++ \"$@\"\n");

        let bad_text = bad.to_string_lossy().into_owned();
        let good_text = good.to_string_lossy().into_owned();
        assert!(run_version_probe(&bad).success());
        assert!(run_version_probe(&good).success());
        assert!(!probe_compiler(&bad_text, &[]));
        assert_eq!(
            select_fallback(&[&bad_text, &good_text]),
            Some((good_text.clone(), Vec::new()))
        );

        std::fs::remove_file(bad).expect("remove fake c++");
        std::fs::remove_file(good).expect("remove fake g++");
        std::fs::remove_dir(dir).expect("remove compiler test directory");
    }
}
