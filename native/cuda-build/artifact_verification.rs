use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

const OPENXLA_API_HEADER_SHA256: &str = "7f76572a80ed2097e5924e6d02d84891c725300172280bd009c0a7c9ac7961eb";
const OPENXLA_C_API_HEADER_SHA256: &str = "85fc385c2d3a6b539a05b9cf4c3535aa24b4b41040f9e111c1f2c11b0e2fa539";
const OPENXLA_FFI_HEADER_SHA256: &str = "4e4a1d8f9825e88e15a2bcbb7c08eb6233f020b952cab5bbbb8510e3017515c5";

pub(crate) struct PtxIdentity<'ptx> {
    pub(crate) isa: &'ptx str,
    pub(crate) target: &'ptx str,
    pub(crate) minimum_compute_capability_major: i32,
    pub(crate) minimum_compute_capability_minor: i32,
}

pub(crate) fn parse_ptx_identity<'ptx>(
    ptx: &'ptx str,
    artifact_description: &str,
) -> Result<PtxIdentity<'ptx>, String> {
    let isa = unique_ptx_directive_value(ptx, ".version")
        .ok_or_else(|| format!("{artifact_description} must declare exactly one single-value PTX ISA version"))?;
    if !isa.bytes().all(|byte| byte.is_ascii_digit() || byte == b'.') {
        return Err(format!("{artifact_description} PTX ISA must contain only ASCII digits and periods"));
    }

    let target = unique_ptx_directive_value(ptx, ".target")
        .ok_or_else(|| format!("{artifact_description} must declare exactly one single-value PTX target"))?;
    let compute_capability = target
        .strip_prefix("sm_")
        .ok_or_else(|| format!("{artifact_description} PTX target must use the sm_<major><minor> form"))?;
    if compute_capability.len() < 2 || !compute_capability.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("{artifact_description} PTX target must use the sm_<major><minor> form"));
    }
    let split_index = compute_capability.len() - 1;
    let minimum_compute_capability_major = compute_capability[..split_index]
        .parse::<i32>()
        .map_err(|error| format!("{artifact_description} PTX target has an invalid major version: {error}"))?;
    let minimum_compute_capability_minor = compute_capability[split_index..]
        .parse::<i32>()
        .map_err(|error| format!("{artifact_description} PTX target has an invalid minor version: {error}"))?;
    if minimum_compute_capability_major <= 0 {
        return Err(format!("{artifact_description} PTX target must encode a positive major version"));
    }

    Ok(PtxIdentity { isa, target, minimum_compute_capability_major, minimum_compute_capability_minor })
}

fn unique_ptx_directive_value<'ptx>(ptx: &'ptx str, directive: &str) -> Option<&'ptx str> {
    let mut observed_value = None;
    for line in ptx.lines() {
        let mut fields = line.split_whitespace();
        if fields.next() != Some(directive) {
            continue;
        }
        let value = fields.next()?;
        if fields.next().is_some() || observed_value.replace(value).is_some() {
            return None;
        }
    }
    observed_value
}

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
    let observed_sha256 = lowercase_sha256(Sha256::digest(bytes));
    if observed_sha256 == expected_sha256 {
        return Ok(());
    }

    Err(format!(
        "{artifact_description} {} changed without a regenerated and reviewed provenance hash: expected sha256:{expected_sha256}, observed sha256:{observed_sha256}",
        path.display()
    ))
}

pub(crate) fn framed_file_set_sha256(
    domain: &str,
    files: &[(&str, &Path)],
    artifact_description: &str,
) -> Result<String, String> {
    let mut digest = Sha256::new();
    update_framed_digest(&mut digest, domain.as_bytes())?;
    for (role, path) in files {
        let bytes =
            fs::read(path).map_err(|error| format!("read {artifact_description} {}: {error}", path.display()))?;
        update_framed_digest(&mut digest, role.as_bytes())?;
        update_framed_digest(&mut digest, &bytes)?;
    }
    Ok(lowercase_sha256(digest.finalize()))
}

fn update_framed_digest(digest: &mut Sha256, bytes: &[u8]) -> Result<(), String> {
    let byte_count =
        u64::try_from(bytes.len()).map_err(|_| "CUDA artifact identity input exceeds uint64 length".to_string())?;
    digest.update(byte_count.to_le_bytes());
    digest.update(bytes);
    Ok(())
}

fn lowercase_sha256(digest: impl AsRef<[u8]>) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest.as_ref() {
        write!(encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TemporaryDirectory {
        path: std::path::PathBuf,
    }

    impl TemporaryDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "g-cuda-artifact-verification-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            fs::create_dir(&path).expect("the CUDA verifier test directory should be created");
            Self { path }
        }
    }

    impl Drop for TemporaryDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.path).expect("the CUDA verifier test directory should be removed");
        }
    }

    #[test]
    fn ptx_identity_requires_one_canonical_version_and_target() {
        let identity =
            parse_ptx_identity(".version 8.2\n.target sm_70\n.address_size 64\n", "test PTX").expect("valid PTX");

        assert_eq!(identity.isa, "8.2");
        assert_eq!(identity.target, "sm_70");
        assert_eq!(identity.minimum_compute_capability_major, 7);
        assert_eq!(identity.minimum_compute_capability_minor, 0);
        assert!(parse_ptx_identity(".version 8.2\n.version 8.3\n.target sm_70\n", "test PTX").is_err());
        assert!(parse_ptx_identity(".version 8.2\n.target compute_70\n", "test PTX").is_err());
    }

    #[test]
    fn framed_file_set_digest_binds_domain_roles_order_and_bytes() {
        let temporary_directory = TemporaryDirectory::new();
        let first_path = temporary_directory.path.join("first");
        let second_path = temporary_directory.path.join("second");
        fs::write(&first_path, b"alpha").expect("the first verifier fixture should be written");
        fs::write(&second_path, b"beta").expect("the second verifier fixture should be written");
        let files = [("first", first_path.as_path()), ("second", second_path.as_path())];

        let digest = framed_file_set_sha256("g-test-handler-v0", &files, "test handler")
            .expect("the framed test handler should hash");
        assert_eq!(digest, "4f74988afe378886c9f71c6395cae1f316f181235603bd1971e809050d2dbf42");
        assert_ne!(
            framed_file_set_sha256("g-other-handler-v0", &files, "test handler")
                .expect("the alternate domain should hash"),
            digest
        );
        assert_ne!(
            framed_file_set_sha256(
                "g-test-handler-v0",
                &[("second", second_path.as_path()), ("first", first_path.as_path())],
                "test handler",
            )
            .expect("the reverse order should hash"),
            digest
        );
        assert_ne!(
            framed_file_set_sha256(
                "g-test-handler-v0",
                &[("renamed", first_path.as_path()), ("second", second_path.as_path())],
                "test handler",
            )
            .expect("the renamed role should hash"),
            digest
        );

        fs::write(&second_path, b"changed").expect("the second verifier fixture should be changed");
        assert_ne!(
            framed_file_set_sha256("g-test-handler-v0", &files, "test handler")
                .expect("the changed content should hash"),
            digest
        );
    }
}
