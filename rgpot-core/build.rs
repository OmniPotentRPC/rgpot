use std::env;
use std::path::PathBuf;

/// Generate C header via cbindgen (only when `gen-header` feature is active).
/// Run `cargo build --features gen-header` or `pixi r gen-header` to regenerate.
#[cfg(feature = "gen-header")]
fn generate_c_header(crate_dir: &str) {
    let output_dir = PathBuf::from(crate_dir).join("include");
    std::fs::create_dir_all(&output_dir).unwrap();

    let config = cbindgen::Config::from_file("cbindgen.toml")
        .expect("Unable to find cbindgen.toml");

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(output_dir.join("rgpot.h"));
}

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    #[cfg(feature = "gen-header")]
    generate_c_header(&crate_dir);

    // Compile Cap'n Proto schema if the rpc feature is enabled.
    // Build scripts don't get #[cfg(feature = ...)] — check the env var instead.
    if env::var("CARGO_FEATURE_RPC").is_ok() {
        // Prefer the bundled schema (works on crates.io).
        // Fall back to the monorepo location for local development.
        let bundled = PathBuf::from(&crate_dir)
            .join("schema")
            .join("Potentials.capnp");
        let monorepo = PathBuf::from(&crate_dir)
            .join("..")
            .join("CppCore")
            .join("rgpot")
            .join("rpc")
            .join("Potentials.capnp");

        let (capnp_dir, capnp_schema) = if bundled.exists() {
            (PathBuf::from(&crate_dir).join("schema"), bundled)
        } else if monorepo.exists() {
            (
                PathBuf::from(&crate_dir)
                    .join("..")
                    .join("CppCore")
                    .join("rgpot")
                    .join("rpc"),
                monorepo,
            )
        } else {
            // Neither found — generate stub so rpc/mod.rs include! doesn't fail.
            let out_dir = env::var("OUT_DIR").unwrap();
            let stub = PathBuf::from(&out_dir).join("Potentials_capnp.rs");
            std::fs::write(
                &stub,
                "// Auto-generated stub: Potentials.capnp not found at build time.\n",
            )
            .expect("Failed to write Cap'n Proto stub");
            return;
        };

        capnpc::CompilerCommand::new()
            .src_prefix(&capnp_dir)
            .file(&capnp_schema)
            .run()
            .expect("Failed to compile Cap'n Proto schema");
    }
}
