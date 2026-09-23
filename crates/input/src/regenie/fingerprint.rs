use std::fs::Metadata;
use std::path::{Path, PathBuf};

/// Exact file digest captured during LOCO indexing and checked against its source.
#[derive(Debug)]
pub struct IndexedPredictionFileFingerprint {
    pub(super) path: PathBuf,
    pub(super) metadata: Metadata,
    pub(super) content_sha256: [u8; 32],
}

impl IndexedPredictionFileFingerprint {
    /// Return the configured prediction path verified against the indexed source.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return source metadata captured from the same open file as the digest.
    #[must_use]
    pub const fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Return the SHA-256 digest of all original file bytes.
    #[must_use]
    pub const fn content_sha256(&self) -> &[u8; 32] {
        &self.content_sha256
    }
}
