fn main() {
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel = manifest.join("../SuttonChenKernel.cc");
    let wrapper = manifest.join("bench_wrap.cc");
    let include = manifest.join("../../..");
    cc::Build::new()
        .cpp(true)
        .compiler("g++")
        .std("c++20")
        .file(kernel.clone())
        .file(wrapper.clone())
        .include(include)
        .flag_if_supported("-ffp-contract=off")
        .flag_if_supported("-fno-math-errno")
        .flag_if_supported("-O3")
        .warnings(false)
        .compile("sutton_chen_kernel");
    println!("cargo:rerun-if-changed={}", kernel.display());
    println!("cargo:rerun-if-changed={}", wrapper.display());
}
