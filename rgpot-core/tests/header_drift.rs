// MIT License
// Copyright 2023--present rgpot developers

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../version_macros.rs"]
mod version_macros;

fn format_c_header(crate_dir: &Path, header: &Path) -> bool {
    let repo_root = crate_dir.join("..");
    let style_file = repo_root.join("CppCore").join(".clang-format");
    let style_arg = if style_file.is_file() {
        format!("--style=file:{}", style_file.display())
    } else {
        "--style=LLVM".to_owned()
    };

    let pixi_cf = repo_root
        .join(".pixi")
        .join("envs")
        .join("devbld")
        .join("bin")
        .join("clang-format");
    let mut candidates: Vec<PathBuf> = vec![PathBuf::from("clang-format")];
    if pixi_cf.is_file() {
        candidates.push(pixi_cf);
    }
    candidates.push(PathBuf::from("clang-format-19"));
    candidates.push(PathBuf::from("clang-format-16"));

    for bin in candidates {
        match Command::new(&bin)
            .args([&style_arg, "-i"])
            .arg(header)
            .status()
        {
            Ok(status) if status.success() => return true,
            Ok(status) => panic!(
                "{} failed while formatting {} with status {status}",
                bin.display(),
                header.display()
            ),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => panic!(
                "failed to invoke {} on {}: {err}",
                bin.display(),
                header.display()
            ),
        }
    }
    false
}

fn regenerate_raw_header(crate_dir: &Path) -> String {
    let mut config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("load rgpot-core/cbindgen.toml");

    let version = env!("CARGO_PKG_VERSION");
    let cargo_core = [
        env!("CARGO_PKG_VERSION_MAJOR"),
        env!("CARGO_PKG_VERSION_MINOR"),
        env!("CARGO_PKG_VERSION_PATCH"),
    ]
    .map(str::to_owned);
    let parsed = version_macros::numeric_core(version)
        .map(|(major, minor, patch)| [major, minor, patch].map(|n| n.to_string()));
    assert_eq!(
        parsed,
        Some(cargo_core),
        "version_macros.rs parses {version:?} differently from cargo"
    );

    let version_block = format!(
        "/* [[[cog import rgpot_version as v; cog.outl(v.header_macros())]]] */\n\
         {macros}\n\
         /* [[[end]]] */",
        macros = version_macros::header_macros(version),
    );
    config.after_includes = Some(match config.after_includes {
        Some(existing) => format!("{existing}\n\n{version_block}"),
        None => version_block,
    });

    let bindings = cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("regenerate C bindings with cbindgen");
    let mut buf = Vec::new();
    bindings.write(&mut buf);
    let raw = String::from_utf8(buf).expect("cbindgen output must be UTF-8");
    format!("{}\n", raw.trim_end())
}

#[test]
fn regenerated_rgpot_h_matches_committed_header() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let committed_path = crate_dir.join("include/rgpot.h");
    let committed = std::fs::read_to_string(&committed_path)
        .expect("read committed rgpot-core/include/rgpot.h");
    let raw = regenerate_raw_header(crate_dir);

    let generated_path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("rgpot.generated.h");
    std::fs::write(&generated_path, &raw).expect("write regenerated rgpot.h");
    assert!(
        format_c_header(crate_dir, &generated_path),
        "clang-format is required to verify rgpot.h formatting"
    );
    let formatted =
        std::fs::read_to_string(&generated_path).expect("read formatted rgpot.generated.h");
    let generated = format!("{}\n", formatted.trim_end());
    std::fs::write(&generated_path, &generated).expect("rewrite normalized rgpot.generated.h");

    if committed != generated {
        let diff_text = match Command::new("diff")
            .arg("-u")
            .arg(&committed_path)
            .arg(&generated_path)
            .output()
        {
            Ok(out) => String::from_utf8_lossy(&out.stdout).into_owned(),
            Err(_) => String::from("(diff tool unavailable)"),
        };
        panic!(
            "rgpot-core/include/rgpot.h drifted from freshly generated cbindgen output:\n{diff_text}"
        );
    }
}

/// `ENGINE_ERR_LOCK` mirrors the `RGPOT_ENGINE_ERR_LOCK` macro of
/// `engine_c_abi.h` and carries `cbindgen:ignore`, so the generated header
/// must not define it.
#[test]
fn generated_rgpot_h_leaves_out_the_ignored_engine_lock_constant() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let generated = regenerate_raw_header(crate_dir);
    assert!(
        !generated.contains("ENGINE_ERR_LOCK"),
        "regenerated rgpot.h leaked the ignored constant ENGINE_ERR_LOCK"
    );
}
