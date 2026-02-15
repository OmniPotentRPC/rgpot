use std::env;
use std::path::PathBuf;

fn main() {
    // Generate C header via cbindgen
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = PathBuf::from(&crate_dir).join("include");
    std::fs::create_dir_all(&output_dir).unwrap();

    let config = cbindgen::Config::from_file("cbindgen.toml")
        .expect("Unable to find cbindgen.toml");

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(output_dir.join("rgpot.h"));

    // Compile Cap'n Proto schema if the rpc feature is enabled.
    // Build scripts don't get #[cfg(feature = ...)] — check the env var instead.
    if env::var("CARGO_FEATURE_RPC").is_ok() {
        let capnp_dir = PathBuf::from(&crate_dir)
            .join("..")
            .join("CppCore")
            .join("rgpot")
            .join("rpc");
        let capnp_schema = capnp_dir.join("Potentials.capnp");

        if capnp_schema.exists() {
            capnpc::CompilerCommand::new()
                .src_prefix(&capnp_dir)
                .file(&capnp_schema)
                .run()
                .expect("Failed to compile Cap'n Proto schema");
        } else {
            // Generate an empty stub so the include! in rpc/mod.rs doesn't fail.
            let out_dir = env::var("OUT_DIR").unwrap();
            let stub = PathBuf::from(&out_dir).join("Potentials_capnp.rs");
            std::fs::write(
                &stub,
                "// Auto-generated stub: Potentials.capnp not found at build time.\n",
            )
            .expect("Failed to write Cap'n Proto stub");
        }
    }
}
