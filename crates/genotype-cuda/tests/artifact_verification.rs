use std::path::{Path, PathBuf};

#[path = "../../../native/cuda-build/artifact_verification.rs"]
mod artifact_verification;

const INCORRECT_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const TRACKED_KERNEL_ARTIFACTS: [(&str, &str); 4] = [
    (
        "crates/genotype-cuda/native/packed8_kernel.cu",
        "5d08dac719a190b201f9c11cdf4a031fb69e4f02b4a20adef1e4837b6842b180",
    ),
    (
        "crates/genotype-cuda/native/packed8_kernel.compute_70.ptx",
        "3e89b8d8277de8b10113c4d0d27979a1449e34593c4141c9dc1a230137fdba32",
    ),
    (
        "crates/compute-cuda/native/firth_components_kernel.cu",
        "1d15fd1aad609023c849942478764c8d2c67a74ff5acd0909652f2dfa180fce0",
    ),
    (
        "crates/compute-cuda/native/firth_components_kernel.compute_70.ptx",
        "a22c9866447f21c7f7cd484ec1e12c3c249a5a84acf3850cb3eb3a56697c736f",
    ),
];

fn repository_path(relative_path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(relative_path)
}

#[test]
fn all_tracked_artifacts_match_the_frozen_manifests() {
    artifact_verification::verify_openxla_headers(&repository_path("vendor/openxla"))
        .expect("the vendored OpenXLA headers should match their reviewed hashes");
    for (relative_path, expected_sha256) in TRACKED_KERNEL_ARTIFACTS {
        artifact_verification::verify_sha256(&repository_path(relative_path), expected_sha256, "tracked CUDA artifact")
            .expect("the tracked CUDA artifact should match its reviewed hash");
    }
}

#[test]
fn changed_artifact_is_rejected_with_both_hashes() {
    let artifact_path = repository_path("crates/genotype-cuda/native/packed8_kernel.cu");
    let error = artifact_verification::verify_sha256(&artifact_path, INCORRECT_SHA256, "test artifact")
        .expect_err("a changed artifact should be rejected");

    assert!(error.contains(&artifact_path.display().to_string()));
    assert!(error.contains(&format!("expected sha256:{INCORRECT_SHA256}")));
    assert!(error.contains("observed sha256:"));
    assert!(!error.contains(&format!("observed sha256:{INCORRECT_SHA256}")));
}

#[test]
fn missing_artifact_is_rejected_with_its_path() {
    let artifact_path = repository_path("vendor/openxla/xla/ffi/api/not-vendored.h");
    let error = artifact_verification::verify_sha256(&artifact_path, INCORRECT_SHA256, "test artifact")
        .expect_err("a missing artifact should be rejected");

    assert!(error.contains("read test artifact"));
    assert!(error.contains(&artifact_path.display().to_string()));
}
