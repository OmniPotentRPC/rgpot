// MIT License
// Copyright 2023--present rgpot developers

use std::path::Path;
use std::process::Command;

#[path = "../version_macros.rs"]
mod version_macros;

fn regenerate_header() -> String {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
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
    let generated = regenerate_header();

    let generated_path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("rgpot.generated.h");
    std::fs::write(&generated_path, &generated).expect("write regenerated rgpot.h");

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

#[test]
fn generated_rgpot_h_excludes_rust_only_engine_constants() {
    let generated = regenerate_header();
    for constant in [
        "ENGINE_ABI_VERSION",
        "ENGINE_ABI_MINOR",
        "ENGINE_ABI_LAYOUT_REVISION",
        "ENGINE_ERR_UNSUPPORTED",
        "ENGINE_ERR_LOCK",
    ] {
        assert!(
            !generated.contains(constant),
            "regenerated rgpot.h leaked Rust-only constant {constant}"
        );
    }
}
