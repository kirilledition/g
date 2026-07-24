//! Best-effort persistent cache for packed8 BGEN compatibility scans.

use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use g_genotype_contracts::BgenContentEvidence;

const CACHE_APPLICATION_DIRECTORY: &str = "g";
const CACHE_DIRECTORY_NAME: &str = "packed8_validation";
const CACHE_SCHEMA_VERSION: i64 = 0;
const COMPATIBLE_MARKER: &[u8] = b"compatible\n";
const REQUIRES_DOSAGE_MARKER: &[u8] = b"requires_dosage\n";
static TEMPORARY_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) struct ValidationCacheEntry {
    cache_path: PathBuf,
    fingerprint: String,
}

impl ValidationCacheEntry {
    pub(super) fn build(
        content_evidence: &BgenContentEvidence,
        sample_count: usize,
        variant_count: usize,
    ) -> std::io::Result<Option<Self>> {
        let Some(cache_directory) = default_cache_directory() else {
            return Ok(None);
        };
        Self::build_in_directory(content_evidence, sample_count, variant_count, &cache_directory)
    }

    fn build_in_directory(
        content_evidence: &BgenContentEvidence,
        sample_count: usize,
        variant_count: usize,
        cache_directory: &Path,
    ) -> std::io::Result<Option<Self>> {
        let BgenContentEvidence::OwnedSnapshot(content_fingerprint) = content_evidence else {
            return Ok(None);
        };
        let sample_count = u64::try_from(sample_count).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "BGEN sample count exceeds the cache range.")
        })?;
        let variant_count = u64::try_from(variant_count).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "BGEN variant count exceeds the cache range.")
        })?;
        let mut fingerprint_hash = Sha256::new();
        fingerprint_hash.update(CACHE_SCHEMA_VERSION.to_le_bytes());
        fingerprint_hash.update(content_fingerprint.content_sha256.as_bytes());
        fingerprint_hash.update(content_fingerprint.byte_count.to_le_bytes());
        fingerprint_hash.update(sample_count.to_le_bytes());
        fingerprint_hash.update(variant_count.to_le_bytes());
        let fingerprint = hex::encode(fingerprint_hash.finalize());
        Ok(Some(Self { cache_path: cache_directory.join(format!("{fingerprint}.marker")), fingerprint }))
    }

    pub(super) fn read(&self) -> std::io::Result<Option<crate::common::Packed8Compatibility>> {
        let marker = match fs::read(&self.cache_path) {
            Ok(marker) => marker,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        match marker.as_slice() {
            COMPATIBLE_MARKER => Ok(Some(crate::common::Packed8Compatibility::Compatible)),
            REQUIRES_DOSAGE_MARKER => Ok(Some(crate::common::Packed8Compatibility::RequiresDosage)),
            _ => Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "Packed8 validation cache marker has an unknown value.",
            )),
        }
    }

    pub(super) fn write(self, compatibility: crate::common::Packed8Compatibility) -> std::io::Result<()> {
        if let Some(parent_path) = self.cache_path.parent() {
            fs::create_dir_all(parent_path)?;
        }
        let marker = match compatibility {
            crate::common::Packed8Compatibility::Compatible => COMPATIBLE_MARKER,
            crate::common::Packed8Compatibility::RequiresDosage => REQUIRES_DOSAGE_MARKER,
        };
        let temporary_file_counter = TEMPORARY_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary_cache_path = self.cache_path.with_file_name(format!(
            ".{}.{}.{}.tmp",
            self.fingerprint,
            std::process::id(),
            temporary_file_counter,
        ));
        let write_result = OpenOptions::new().write(true).create_new(true).open(&temporary_cache_path).and_then(
            |mut temporary_cache_file| {
                temporary_cache_file.write_all(marker)?;
                temporary_cache_file.sync_all()
            },
        );
        if let Err(error) = write_result {
            let _ = fs::remove_file(&temporary_cache_path);
            return Err(error);
        }
        if let Err(error) = fs::rename(&temporary_cache_path, &self.cache_path) {
            let _ = fs::remove_file(&temporary_cache_path);
            if !self.cache_path.is_file() {
                return Err(error);
            }
        }
        Ok(())
    }
}

fn default_cache_directory() -> Option<PathBuf> {
    non_empty_environment_path("XDG_CACHE_HOME")
        .map(|cache_root| cache_root.join(CACHE_APPLICATION_DIRECTORY).join(CACHE_DIRECTORY_NAME))
        .or_else(|| {
            non_empty_environment_path("HOME").map(|home_directory| {
                home_directory.join(".cache").join(CACHE_APPLICATION_DIRECTORY).join(CACHE_DIRECTORY_NAME)
            })
        })
}

fn non_empty_environment_path(variable_name: &str) -> Option<PathBuf> {
    std::env::var_os(variable_name).filter(|value| !value.is_empty()).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use g_genotype_contracts::{BgenContentEvidence, BgenContentFingerprint, BgenContentSha256, BgenSourceIdentity};

    use super::*;

    fn owned_evidence(digest_byte: u8, byte_count: u64) -> BgenContentEvidence {
        BgenContentEvidence::OwnedSnapshot(BgenContentFingerprint {
            content_sha256: BgenContentSha256::from_bytes([digest_byte; 32]),
            byte_count,
        })
    }

    fn positioned_evidence(path: &str) -> BgenContentEvidence {
        BgenContentEvidence::PositionedUnattested(BgenSourceIdentity {
            configured_path: PathBuf::from(path),
            canonical_path: Some(PathBuf::from(format!("/canonical/{path}"))),
            device_identifier: 2,
            inode_identifier: 3,
            change_time_nanoseconds: 5,
            modification_time_nanoseconds: 7,
            file_size: 11,
        })
    }

    fn cache_entry(
        content_evidence: &BgenContentEvidence,
        sample_count: usize,
        variant_count: usize,
    ) -> ValidationCacheEntry {
        ValidationCacheEntry::build_in_directory(content_evidence, sample_count, variant_count, Path::new("/cache"))
            .expect("cache fingerprint should build")
            .expect("owned content should produce a cache entry")
    }

    #[test]
    fn packed8_cache_fingerprint_binds_content_length_revision_and_geometry() {
        let baseline = cache_entry(&owned_evidence(13, 101), 103, 107);
        let same_content = cache_entry(&owned_evidence(13, 101), 103, 107);
        let different_digest = cache_entry(&owned_evidence(17, 101), 103, 107);
        let different_length = cache_entry(&owned_evidence(13, 109), 103, 107);
        let different_sample_count = cache_entry(&owned_evidence(13, 101), 127, 107);
        let different_variant_count = cache_entry(&owned_evidence(13, 101), 103, 131);

        assert_eq!(CACHE_SCHEMA_VERSION, 0);
        assert_eq!(baseline.fingerprint, same_content.fingerprint);
        assert_eq!(baseline.cache_path, same_content.cache_path);
        assert_ne!(baseline.fingerprint, different_digest.fingerprint);
        assert_ne!(baseline.fingerprint, different_length.fingerprint);
        assert_ne!(baseline.fingerprint, different_sample_count.fingerprint);
        assert_ne!(baseline.fingerprint, different_variant_count.fingerprint);
    }

    #[test]
    fn packed8_cache_requires_authoritative_content_and_has_no_path_component() {
        let first_path = positioned_evidence("first/input.bgen");
        let second_path = positioned_evidence("second/input.bgen");

        assert!(
            ValidationCacheEntry::build_in_directory(&first_path, 3, 5, Path::new("/cache"))
                .expect("unattested cache decision should succeed")
                .is_none(),
        );
        assert!(
            ValidationCacheEntry::build_in_directory(&second_path, 3, 5, Path::new("/cache"))
                .expect("unattested cache decision should succeed")
                .is_none(),
        );

        let first_content = cache_entry(&owned_evidence(19, 23), 3, 5);
        let same_content_for_another_locator = cache_entry(&owned_evidence(19, 23), 3, 5);
        assert_eq!(first_content.fingerprint, same_content_for_another_locator.fingerprint);
    }
}
