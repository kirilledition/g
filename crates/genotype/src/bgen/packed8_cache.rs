//! Best-effort persistent cache for packed8 BGEN compatibility scans.

use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use g_genotype_contracts::BgenSourceIdentity;

const CACHE_APPLICATION_DIRECTORY: &str = "g";
const CACHE_DIRECTORY_NAME: &str = "packed8_validation";
const CACHE_SCHEMA_VERSION: i64 = 5;
const COMPATIBLE_MARKER: &[u8] = b"compatible\n";
const REQUIRES_DOSAGE_MARKER: &[u8] = b"requires_dosage\n";
static TEMPORARY_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) struct ValidationCacheEntry {
    cache_path: PathBuf,
    fingerprint: String,
}

impl ValidationCacheEntry {
    pub(super) fn build(
        source_identity: &BgenSourceIdentity,
        sample_count: usize,
        variant_count: usize,
    ) -> std::io::Result<Option<Self>> {
        let Some(cache_directory) = default_cache_directory() else {
            return Ok(None);
        };
        let Some(canonical_bgen_path) = source_identity.canonical_path.as_ref() else {
            return Ok(None);
        };
        let sample_count = i64::try_from(sample_count).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "BGEN sample count exceeds the cache range.")
        })?;
        let variant_count = i64::try_from(variant_count).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "BGEN variant count exceeds the cache range.")
        })?;
        let canonical_path_bytes = canonical_bgen_path.as_os_str().as_bytes();
        let canonical_path_byte_count = u64::try_from(canonical_path_bytes.len()).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "BGEN canonical path exceeds the cache range.")
        })?;
        let mut fingerprint_hash = Sha256::new();
        fingerprint_hash.update(CACHE_SCHEMA_VERSION.to_le_bytes());
        fingerprint_hash.update(canonical_path_byte_count.to_le_bytes());
        fingerprint_hash.update(canonical_path_bytes);
        fingerprint_hash.update(source_identity.change_time_nanoseconds.to_le_bytes());
        fingerprint_hash.update(source_identity.device_identifier.to_le_bytes());
        fingerprint_hash.update(source_identity.inode_identifier.to_le_bytes());
        fingerprint_hash.update(source_identity.modification_time_nanoseconds.to_le_bytes());
        fingerprint_hash.update(sample_count.to_le_bytes());
        fingerprint_hash.update(source_identity.file_size.to_le_bytes());
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
