use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

const OPENXLA_API_HEADER_SHA256: &str = "7f76572a80ed2097e5924e6d02d84891c725300172280bd009c0a7c9ac7961eb";
const OPENXLA_C_API_HEADER_SHA256: &str = "85fc385c2d3a6b539a05b9cf4c3535aa24b4b41040f9e111c1f2c11b0e2fa539";
const OPENXLA_FFI_HEADER_SHA256: &str = "4e4a1d8f9825e88e15a2bcbb7c08eb6233f020b952cab5bbbb8510e3017515c5";

pub(crate) fn verify_openxla_headers(openxla_directory: &Path) -> Result<(), String> {
    verify_sha256(
        &openxla_directory.join("xla/ffi/api/api.h"),
        OPENXLA_API_HEADER_SHA256,
        "vendored OpenXLA FFI header",
    )?;
    verify_sha256(
        &openxla_directory.join("xla/ffi/api/c_api.h"),
        OPENXLA_C_API_HEADER_SHA256,
        "vendored OpenXLA FFI header",
    )?;
    verify_sha256(
        &openxla_directory.join("xla/ffi/api/ffi.h"),
        OPENXLA_FFI_HEADER_SHA256,
        "vendored OpenXLA FFI header",
    )
}

pub(crate) fn verify_sha256(path: &Path, expected_sha256: &str, artifact_description: &str) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|error| format!("read {artifact_description} {}: {error}", path.display()))?;
    let mut observed_sha256 = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(observed_sha256, "{byte:02x}").expect("writing to a String cannot fail");
    }
    if observed_sha256 == expected_sha256 {
        return Ok(());
    }

    Err(format!(
        "{artifact_description} {} changed without a regenerated and reviewed provenance hash: expected sha256:{expected_sha256}, observed sha256:{observed_sha256}",
        path.display()
    ))
}
