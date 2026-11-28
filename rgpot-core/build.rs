use std::env;
use std::path::PathBuf;

#[cfg(feature = "gen-header")]
#[path = "version_macros.rs"]
mod version_macros;

/// Generate C header via cbindgen (only when `gen-header` feature is active).
/// Run `cargo build --features gen-header` or `pixi r gen-header` to regenerate.
#[cfg(feature = "gen-header")]
fn generate_c_header(crate_dir: &str) {
    // Once any rerun-if line is printed, cargo stops rescanning the
    // package, so name every input of the header. Without these a
    // changed doc comment would not regenerate it.
    for input in ["src", "cbindgen.toml", "version_macros.rs", "Cargo.toml"] {
        println!("cargo::rerun-if-changed={input}");
    }

    let output_dir = PathBuf::from(crate_dir).join("include");
    std::fs::create_dir_all(&output_dir).unwrap();

    let mut config =
        cbindgen::Config::from_file("cbindgen.toml").expect("Unable to find cbindgen.toml");

    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let cargo_core = ["MAJOR", "MINOR", "PATCH"]
        .map(|part| env::var(format!("CARGO_PKG_VERSION_{part}")).unwrap());
    let ours = version_macros::numeric_core(&version)
        .map(|(major, minor, patch)| [major, minor, patch].map(|n| n.to_string()));
    assert_eq!(
        ours,
        Some(cargo_core),
        "version_macros.rs parses {version:?} differently from cargo"
    );
    // The cog markers let the cogapp check see the same block
    // that ci/version/rgpot_version.py writes, so neither copy can drift.
    let version_block = format!(
        "/* [[[cog import rgpot_version as v; cog.outl(v.header_macros())]]] */\n\
         {macros}\n\
         /* [[[end]]] */",
        macros = version_macros::header_macros(&version),
    );
    config.after_includes = Some(match config.after_includes {
        Some(existing) => format!("{existing}\n\n{version_block}"),
        None => version_block,
    });

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(output_dir.join("rgpot.h"));

    let path = output_dir.join("rgpot.h");
    format_c_header(crate_dir, &path);

    // Ensure the file ends with exactly one newline (keeps end-of-file-fixer happy).
    let content = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, format!("{}\n", content.trim_end())).unwrap();
}

#[cfg(feature = "gen-header")]
fn format_c_header(crate_dir: &str, header: &std::path::Path) {
    let repo_root = PathBuf::from(crate_dir).join("..");
    let style_file = repo_root.join("CppCore").join(".clang-format");
    let style_arg = if style_file.is_file() {
        println!("cargo::rerun-if-changed={}", style_file.display());
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
        match std::process::Command::new(&bin)
            .args([&style_arg, "-i"])
            .arg(header)
            .status()
        {
            Ok(status) if status.success() => return,
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
    println!(
        "cargo::warning=clang-format not found; {} left unformatted",
        header.display()
    );
}

/// Source revision of the build: `RGPOT_SOURCE_REVISION`, else the git
/// short hash of the checkout, else empty.
fn source_revision(crate_dir: &str) -> String {
    if let Ok(rev) = env::var("RGPOT_SOURCE_REVISION") {
        return rev.trim().to_owned();
    }
    std::process::Command::new("git")
        .args(["-C", crate_dir, "rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn cppcore_root(crate_dir: &str) -> PathBuf {
    if let Ok(path) = env::var("RGPOT_CPPCORE_ROOT") {
        return PathBuf::from(path);
    }
    let monorepo = PathBuf::from(crate_dir).join("../CppCore");
    if monorepo.join("rgpot").is_dir() {
        return monorepo;
    }
    PathBuf::from(crate_dir).join("cppcore-staging")
}

fn main() {
    #[allow(unused_variables)]
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo::rerun-if-env-changed=RGPOT_SOURCE_REVISION");
    println!("cargo::rerun-if-changed=src/version.rs");
    println!("cargo::rerun-if-changed=src/engine.rs");
    println!("cargo::rerun-if-changed=cbindgen.toml");
    println!("cargo::rerun-if-changed=../CppCore/rgpot/engine_c_abi.h");
    println!("cargo::rerun-if-changed=cpp/eam_table_bridge.cpp");
    println!("cargo::rerun-if-env-changed=RGPOT_CPPCORE_ROOT");
    println!(
        "cargo::rustc-env=RGPOT_SOURCE_REVISION={}",
        source_revision(&crate_dir)
    );
    if let Ok(target) = env::var("TARGET") {
        println!("cargo::rustc-env=RGPOT_BUILD_TARGET={target}");
    }

    let cppcore = cppcore_root(&crate_dir);
    println!("cargo::rerun-if-changed={}", cppcore.display());
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        // The bridge archive links statically, so RGPOT_API must not
        // expand to dllimport on Windows.
        .define("RGPOT_STATIC", None)
        // Isolate the packaged CppCore symbols from a CppCore library linked
        // by an embedding application, and avoid the rgpot_ C ABI prefix so
        // symbol-export baselines only count public C ABI entry points.
        .define("rgpot", "eam_bridge_cpp")
        .define("rgpot_nlist_abi_version", "eam_bridge_nlist_abi_version")
        .define("rgpot_nlist_strerror", "eam_bridge_nlist_strerror")
        .define("rgpot_nlist_request_init", "eam_bridge_nlist_request_init")
        .define("rgpot_nlist_build", "eam_bridge_nlist_build")
        .define("rgpot_nlist_walk_open", "eam_bridge_nlist_walk_open")
        .define("rgpot_nlist_walk_pull", "eam_bridge_nlist_walk_pull")
        .define("rgpot_nlist_walk_close", "eam_bridge_nlist_walk_close")
        .define(
            "rgpot_nlist_build_systems",
            "eam_bridge_nlist_build_systems",
        )
        .define("rgpot_nlist_builder_open", "eam_bridge_nlist_builder_open")
        .define(
            "rgpot_nlist_builder_update",
            "eam_bridge_nlist_builder_update",
        )
        .define(
            "rgpot_nlist_builder_close",
            "eam_bridge_nlist_builder_close",
        )
        .define("rgpot_nlist_fortran_open", "eam_bridge_nlist_fortran_open")
        .define(
            "rgpot_nlist_fortran_close",
            "eam_bridge_nlist_fortran_close",
        )
        .define(
            "rgpot_nlist_fortran_build",
            "eam_bridge_nlist_fortran_build",
        )
        .flag_if_supported("-fexceptions")
        .flag_if_supported("-ffp-contract=off")
        .flag_if_supported("/fp:precise")
        .include(&cppcore)
        .include(cppcore.parent().unwrap().join("include"))
        .file("cpp/eam_table_bridge.cpp")
        .file(cppcore.join("rgpot").join("PotHelpers.cc"))
        .file(cppcore.join("rgpot").join("simd").join("exact_fma_hw.cc"))
        .file(cppcore.join("rgpot").join("simd").join("exp.cc"))
        .file(cppcore.join("rgpot").join("nlist").join("MinimumImage.cpp"))
        .file(cppcore.join("rgpot").join("Slices.cc"))
        .file(cppcore.join("rgpot").join("nlist").join("HalfShell.cc"))
        .file(cppcore.join("rgpot").join("nlist").join("HalfShellWalk.cc"))
        .file(cppcore.join("rgpot").join("nlist").join("NeighbourList.cc"))
        .file(cppcore.join("rgpot").join("nlist").join("Neighbours.cc"))
        .file(
            cppcore
                .join("rgpot")
                .join("EAMTable")
                .join("EAMTablePot.cc"),
        )
        .file(
            cppcore
                .join("rgpot")
                .join("EAMTable")
                .join("EAMTableImages.cc"),
        )
        .compile("rgpot_eam_table_bridge");

    // eindir-core is a normal Cargo dependency now; its capi #[no_mangle] symbols
    // resolve through the shared crate, so no prebuilt static lib link is needed.

    #[cfg(feature = "gen-header")]
    generate_c_header(&crate_dir);

    #[cfg(feature = "schema")]
    compile_capnp_schema(&crate_dir);
}

#[cfg(feature = "schema")]
fn compile_capnp_schema(crate_dir: &str) {
    // Prefer the bundled schema (works on crates.io).
    // Fall back to the monorepo location for local development.
    let bundled = PathBuf::from(crate_dir)
        .join("schema")
        .join("Potentials.capnp");
    let monorepo = PathBuf::from(crate_dir)
        .join("..")
        .join("CppCore")
        .join("rgpot")
        .join("rpc")
        .join("Potentials.capnp");

    let (capnp_dir, capnp_schema) = if bundled.exists() {
        (PathBuf::from(crate_dir).join("schema"), bundled)
    } else if monorepo.exists() {
        (
            PathBuf::from(crate_dir)
                .join("..")
                .join("CppCore")
                .join("rgpot")
                .join("rpc"),
            monorepo,
        )
    } else {
        // Neither found — generate stub so rpc/mod.rs include! doesn't fail.
        let out_dir = env::var("OUT_DIR").unwrap();
        for name in ["Potentials", "common"] {
            std::fs::write(
                PathBuf::from(&out_dir).join(format!("{name}_capnp.rs")),
                format!("// Auto-generated stub: {name}.capnp not found at build time.\n"),
            )
            .expect("Failed to write Cap'n Proto stub");
        }
        return;
    };

    // The build-identity fields exist only in schema revisions that carry
    // them; the producer code is compiled in when they are present.
    println!("cargo::rustc-check-cfg=cfg(rgpot_schema_build_identity)");
    if std::fs::read_to_string(&capnp_schema)
        .map(|text| text.contains("buildVersion"))
        .unwrap_or(false)
    {
        println!("cargo::rustc-cfg=rgpot_schema_build_identity");
    }
    println!("cargo::rerun-if-changed={}", capnp_schema.display());

    // The vendored copy is rgconf's potential.capnp as it is; it imports
    // common.capnp from the same directory, so both are compiled and the
    // crate carries a common_capnp module next to Potentials_capnp.
    let common = capnp_dir.join("common.capnp");
    println!("cargo::rerun-if-changed={}", common.display());

    capnpc::CompilerCommand::new()
        .src_prefix(&capnp_dir)
        .file(&capnp_schema)
        .file(&common)
        .run()
        .expect("Failed to compile Cap'n Proto schema");
}
