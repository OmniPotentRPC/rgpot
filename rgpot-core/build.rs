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

    // Compile Cap'n Proto schema if the rpc feature is enabled
    #[cfg(feature = "rpc")]
    {
        let capnp_schema = PathBuf::from(&crate_dir)
            .join("..")
            .join("CppCore")
            .join("rgpot")
            .join("rpc")
            .join("Potentials.capnp");

        if capnp_schema.exists() {
            capnpc::CompilerCommand::new()
                .file(&capnp_schema)
                .run()
                .expect("Failed to compile Cap'n Proto schema");
        }
    }
}
