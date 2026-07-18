use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const KERNEL_SOURCE_SHA256: &str = "4a823918e8b198ef8079cf54e159467c0942ee3d59c99924558d413f7c43585c";
const KERNEL_PTX_SHA256: &str = "a22c9866447f21c7f7cd484ec1e12c3c249a5a84acf3850cb3eb3a56697c736f";

fn main() {
    println!("cargo:rerun-if-changed=native/firth_components_ffi.cc");
    println!("cargo:rerun-if-changed=native/cuda_driver_abi.h");
    println!("cargo:rerun-if-changed=native/firth_components_kernel.cu");
    println!("cargo:rerun-if-changed=native/firth_components_kernel.compute_70.ptx");
    println!("cargo:rerun-if-changed=../../vendor/openxla/xla/ffi/api/api.h");
    println!("cargo:rerun-if-changed=../../vendor/openxla/xla/ffi/api/c_api.h");
    println!("cargo:rerun-if-changed=../../vendor/openxla/xla/ffi/api/ffi.h");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    verify_kernel_artifacts();
    let output_directory = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    write_embedded_ptx(&output_directory);

    let mut native_build = cc::Build::new();
    native_build
        .cpp(true)
        .std("c++20")
        .define("NDEBUG", None)
        .include("native")
        .include(&output_directory)
        .include("../../vendor/openxla")
        .file("native/firth_components_ffi.cc")
        .flag_if_supported("-O3")
        .flag_if_supported("-fPIC")
        .flag_if_supported("-isystem../../vendor/openxla")
        .warnings(true)
        .extra_warnings(true)
        .compile("g_compute_cuda_native");

    println!("cargo:rustc-link-lib=dl");
}

fn verify_kernel_artifacts() {
    verify_sha256("native/firth_components_kernel.cu", KERNEL_SOURCE_SHA256);
    verify_sha256("native/firth_components_kernel.compute_70.ptx", KERNEL_PTX_SHA256);
}

fn verify_sha256(path: &str, expected_sha256: &str) {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read maintained CUDA artifact {path}: {error}"));
    let mut observed_sha256 = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(observed_sha256, "{byte:02x}").expect("writing to a String cannot fail");
    }
    assert_eq!(
        observed_sha256, expected_sha256,
        "maintained CUDA artifact {path} changed without a regenerated and reviewed provenance hash"
    );
}

fn write_embedded_ptx(output_directory: &Path) {
    const RAW_STRING_DELIMITER: &str = "g_cuda_firth";

    let ptx = fs::read_to_string("native/firth_components_kernel.compute_70.ptx")
        .expect("the checked-in compute_70 Firth PTX artifact must be readable");
    let closing_delimiter = format!("){RAW_STRING_DELIMITER}\"");
    assert!(!ptx.contains(&closing_delimiter), "PTX contains the reserved C++ raw-string delimiter");
    let include = format!("R\"{RAW_STRING_DELIMITER}(\n{ptx}\n){RAW_STRING_DELIMITER}\"\n");
    fs::write(output_directory.join("firth_components_kernel_ptx.inc"), include)
        .expect("the generated Firth PTX include must be writable");
}
