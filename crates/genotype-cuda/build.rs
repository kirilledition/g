use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=native/packed8_deflate_ffi.cc");
    println!("cargo:rerun-if-changed=native/nvcomp_abi.h");
    println!("cargo:rerun-if-changed=../../native/cuda-driver/cuda_driver.h");
    println!("cargo:rerun-if-changed=native/packed8_kernel.cu");
    println!("cargo:rerun-if-changed=native/packed8_kernel.compute_70.ptx");
    println!("cargo:rerun-if-changed=../../vendor/openxla/xla/ffi/api/api.h");
    println!("cargo:rerun-if-changed=../../vendor/openxla/xla/ffi/api/c_api.h");
    println!("cargo:rerun-if-changed=../../vendor/openxla/xla/ffi/api/ffi.h");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    let output_directory = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    write_embedded_ptx(&output_directory);

    let mut native_build = cc::Build::new();
    native_build
        .cpp(true)
        .std("c++20")
        .define("NDEBUG", None)
        .include("native")
        .include("../../native/cuda-driver")
        .include(&output_directory)
        .include("../../vendor/openxla")
        .file("native/packed8_deflate_ffi.cc")
        .flag_if_supported("-O3")
        .flag_if_supported("-fPIC")
        .flag_if_supported("-isystem../../vendor/openxla")
        .warnings(true)
        .extra_warnings(true)
        .compile("g_genotype_cuda_native");

    println!("cargo:rustc-link-lib=dl");
}

fn write_embedded_ptx(output_directory: &Path) {
    const RAW_STRING_DELIMITER: &str = "g_cuda_ptx";

    let ptx = fs::read_to_string("native/packed8_kernel.compute_70.ptx")
        .expect("the checked-in compute_70 PTX artifact must be readable");
    let closing_delimiter = format!("){RAW_STRING_DELIMITER}\"");
    assert!(!ptx.contains(&closing_delimiter), "PTX contains the reserved C++ raw-string delimiter");

    let include = format!("R\"{RAW_STRING_DELIMITER}(\n{ptx}\n){RAW_STRING_DELIMITER}\"\n");
    fs::write(output_directory.join("packed8_kernel_ptx.inc"), include)
        .expect("the generated PTX include must be writable");
}
