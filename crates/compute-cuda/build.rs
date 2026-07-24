use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[path = "../../native/cuda-build/artifact_verification.rs"]
mod artifact_verification;

const FIRTH_COMPONENTS_FFI_API_VERSION: u32 = 1;
const MINIMUM_CUDA_DRIVER_VERSION: i32 = 12_020;
const MINIMUM_CUDA_DRIVER_PTX_ISA: &str = "8.2";
const KERNEL_SOURCE_SHA256: &str = "1d15fd1aad609023c849942478764c8d2c67a74ff5acd0909652f2dfa180fce0";
const KERNEL_PTX_SHA256: &str = "a22c9866447f21c7f7cd484ec1e12c3c249a5a84acf3850cb3eb3a56697c736f";
const KERNEL_PTX_PATH: &str = "native/firth_components_kernel.compute_70.ptx";

fn main() {
    println!("cargo:rerun-if-changed=../../native/cuda-build/artifact_verification.rs");
    println!("cargo:rerun-if-changed=native/firth_components_ffi.cc");
    println!("cargo:rerun-if-changed=../../native/cuda-driver/cuda_driver.h");
    println!("cargo:rerun-if-changed=native/firth_components_kernel.cu");
    println!("cargo:rerun-if-changed=native/firth_components_kernel.compute_70.ptx");
    println!("cargo:rerun-if-changed=../../vendor/openxla/xla/ffi/api/api.h");
    println!("cargo:rerun-if-changed=../../vendor/openxla/xla/ffi/api/c_api.h");
    println!("cargo:rerun-if-changed=../../vendor/openxla/xla/ffi/api/ffi.h");

    verify_build_artifacts();
    let output_directory = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    let ptx =
        fs::read_to_string(KERNEL_PTX_PATH).expect("the checked-in compute_70 Firth PTX artifact must be readable");
    let ptx_identity = artifact_verification::parse_ptx_identity(&ptx, "checked-in Firth PTX artifact")
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        ptx_identity.isa, MINIMUM_CUDA_DRIVER_PTX_ISA,
        "the reviewed CUDA driver floor must be updated together with the embedded PTX ISA"
    );
    let handler_sha256 = artifact_verification::framed_file_set_sha256(
        "g-compute-cuda-firth-handler-v0",
        &[
            ("ffi", Path::new("native/firth_components_ffi.cc")),
            ("ptx", Path::new(KERNEL_PTX_PATH)),
            ("cuda-driver", Path::new("../../native/cuda-driver/cuda_driver.h")),
            ("openxla-api", Path::new("../../vendor/openxla/xla/ffi/api/api.h")),
            ("openxla-c-api", Path::new("../../vendor/openxla/xla/ffi/api/c_api.h")),
            ("openxla-ffi", Path::new("../../vendor/openxla/xla/ffi/api/ffi.h")),
        ],
        "raw-CUDA Firth handler identity input",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    write_artifact_identity(&output_directory, &ptx_identity, &handler_sha256);
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    write_embedded_ptx(&output_directory, &ptx);

    let mut native_build = cc::Build::new();
    native_build
        .cpp(true)
        .std("c++20")
        .define("NDEBUG", None)
        .include("native")
        .include("../../native/cuda-driver")
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

fn verify_build_artifacts() {
    artifact_verification::verify_sha256(
        Path::new("native/firth_components_kernel.cu"),
        KERNEL_SOURCE_SHA256,
        "maintained CUDA source",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    artifact_verification::verify_sha256(Path::new(KERNEL_PTX_PATH), KERNEL_PTX_SHA256, "checked-in CUDA PTX")
        .unwrap_or_else(|error| panic!("{error}"));
    artifact_verification::verify_openxla_headers(Path::new("../../vendor/openxla"))
        .unwrap_or_else(|error| panic!("{error}"));
}

fn write_artifact_identity(
    output_directory: &Path,
    ptx_identity: &artifact_verification::PtxIdentity<'_>,
    handler_sha256: &str,
) {
    let rust_identity = format!(
        "/// JAX typed-FFI registration API used by the Firth handler.\n\
         pub const FIRTH_COMPONENTS_FFI_API_VERSION: u32 = {FIRTH_COMPONENTS_FFI_API_VERSION};\n\
         /// SHA-256 of the framed native handler, PTX, driver support, and XLA ABI inputs.\n\
         pub const FIRTH_COMPONENTS_HANDLER_SHA256: &str = \"{handler_sha256}\";\n\
         /// Minimum CUDA driver API version reviewed for the embedded PTX ISA.\n\
         pub const FIRTH_COMPONENTS_MINIMUM_CUDA_DRIVER_VERSION: i32 = {MINIMUM_CUDA_DRIVER_VERSION};\n\
         /// Minimum compute-capability major version declared by the embedded PTX target.\n\
         pub const FIRTH_COMPONENTS_MINIMUM_COMPUTE_CAPABILITY_MAJOR: i32 = {};\n\
         /// Minimum compute-capability minor version declared by the embedded PTX target.\n\
         pub const FIRTH_COMPONENTS_MINIMUM_COMPUTE_CAPABILITY_MINOR: i32 = {};\n\
         /// Canonical SHA-256 of the verified embedded Firth PTX artifact.\n\
         pub const FIRTH_COMPONENTS_PTX_SHA256: &str = \"{KERNEL_PTX_SHA256}\";\n\
         /// PTX ISA declared by the verified embedded Firth artifact.\n\
         pub const FIRTH_COMPONENTS_PTX_ISA: &str = \"{}\";\n\
         /// Compilation target declared by the verified embedded Firth artifact.\n\
         pub const FIRTH_COMPONENTS_PTX_TARGET: &str = \"{}\";\n",
        ptx_identity.minimum_compute_capability_major,
        ptx_identity.minimum_compute_capability_minor,
        ptx_identity.isa,
        ptx_identity.target
    );
    fs::write(output_directory.join("firth_components_artifact_identity.rs"), rust_identity)
        .expect("the generated Rust Firth artifact identity must be writable");

    let native_identity = format!(
        "constexpr std::int32_t kMinimumCudaDriverVersion = {MINIMUM_CUDA_DRIVER_VERSION};\n\
         constexpr std::int32_t kMinimumComputeCapabilityMajor = {};\n\
         constexpr std::int32_t kMinimumComputeCapabilityMinor = {};\n\
         constexpr char kFirthComponentsPtxIsa[] = \"{}\";\n\
         constexpr char kFirthComponentsPtxTarget[] = \"{}\";\n",
        ptx_identity.minimum_compute_capability_major,
        ptx_identity.minimum_compute_capability_minor,
        ptx_identity.isa,
        ptx_identity.target
    );
    fs::write(output_directory.join("firth_components_artifact_identity.inc"), native_identity)
        .expect("the generated native Firth artifact identity must be writable");
}

fn write_embedded_ptx(output_directory: &Path, ptx: &str) {
    const RAW_STRING_DELIMITER: &str = "g_cuda_firth";

    let closing_delimiter = format!("){RAW_STRING_DELIMITER}\"");
    assert!(!ptx.contains(&closing_delimiter), "PTX contains the reserved C++ raw-string delimiter");
    let include = format!("R\"{RAW_STRING_DELIMITER}({ptx}){RAW_STRING_DELIMITER}\"\n");
    fs::write(output_directory.join("firth_components_kernel_ptx.inc"), include)
        .expect("the generated Firth PTX include must be writable");
}
