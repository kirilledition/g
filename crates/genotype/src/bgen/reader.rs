use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;

use g_genotype_contracts::{VariantMetadataColumns, VariantMetadataStore};

use crate::common::{ChunkSpec, Packed8BufferPool, Packed8Compatibility};
use crate::error::GenotypeResult;

use super::decode::{ThreadScratch, VariantDecodeFailure, read_exact_bytes, read_u32_at, u32_to_usize};
use super::error::{BgenError, contextualize_variant_metadata_invariant};
use super::format::CompressionType;
use super::metadata::VariantRecord;
use super::sample_selection::{SampleSelection, build_sample_selection};
use super::{index, index_cache, packed8};

mod variant_major;

#[derive(Debug)]
pub struct BgenReaderCore {
    pub(super) mmap: Mmap,
    source: super::packed8_cache::ValidationCacheSource,
    sample_count: usize,
    variant_count: usize,
    pub(super) compression_type: CompressionType,
    packed8_validation_complete: AtomicBool,
    pub(super) variant_records: Arc<[VariantRecord]>,
    variant_metadata: Arc<VariantMetadataStore>,
    chromosome_boundary_indices: Arc<[usize]>,
}

/// Immutable per-delivery BGEN decoding context.
#[derive(Debug)]
pub struct BgenReadSession<'reader> {
    pub(super) reader: &'reader BgenReaderCore,
    pub(super) sample_selection: SampleSelection,
    pub(super) packed8_buffer_pool: Arc<Packed8BufferPool>,
    pub(super) compressed_packed8_state: OnceLock<super::raw_deflate::CompressedPacked8SessionState>,
}

impl BgenReaderCore {
    /// Open and index a Layout 2 BGEN source.
    ///
    /// Repeated opens can share the last verified index within this process.
    /// The cache limits estimated index allocation to 256 `MiB` and one descriptor,
    /// without retaining a mapping, until eviction or process exit. Each open
    /// still maps its own source and checks source identity before publishing
    /// any reused metadata.
    /// Admission requires observing the same opened-file identity for two
    /// monotonic seconds, then parsing it afresh. Initial opens therefore
    /// still parse even old files; recent index snapshots are never reused
    /// across the ambiguity window of second-resolution filesystem timestamps.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened or mapped, its header or
    /// variant index is invalid, its layout is unsupported, or it changes while
    /// being indexed.
    pub fn open(bgen_path: &Path) -> Result<Self, BgenError> {
        Self::open_with_index_cache(bgen_path, index_cache::process_index_cache())
    }

    fn open_with_index_cache(
        bgen_path: &Path,
        index_cache: &Mutex<index_cache::IndexCache>,
    ) -> Result<Self, BgenError> {
        let source = super::packed8_cache::ValidationCacheSource::open(bgen_path)?;
        let mmap = unsafe { MmapOptions::new().map(&source.file)? };
        if !source.is_unchanged()? {
            return Err(BgenError::InvalidFormat("BGEN source changed while it was being opened.".to_string()));
        }
        // Acquire admission before reading any parse input, including the
        // header and embedded samples. A long open must not promote header
        // fields captured inside the coarse-timestamp probation window.
        let cached_variant_index =
            index_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner).lookup(&source);

        let first_variant_offset = 4 + u32_to_usize(read_u32_at(&mmap, 0)?)?;
        let header_block_length = u32_to_usize(read_u32_at(&mmap, 4)?)?;
        if header_block_length < 20 {
            return Err(BgenError::InvalidFormat(format!(
                "BGEN header block length must be at least 20 bytes. Observed {header_block_length}.",
            )));
        }
        let variant_count = u32_to_usize(read_u32_at(&mmap, 8)?)?;
        let sample_count = u32_to_usize(read_u32_at(&mmap, 12)?)?;

        let magic_offset = 16;
        let magic_number = read_exact_bytes(&mmap, magic_offset, 4)?;
        if magic_number != b"bgen" && magic_number != [0_u8, 0, 0, 0] {
            return Err(BgenError::InvalidFormat(
                "BGEN header magic number must be `bgen` or four zero bytes.".to_string(),
            ));
        }

        let header_flags_offset = 4 + header_block_length - 4;
        let header_flags = read_u32_at(&mmap, header_flags_offset)?;
        let compression_type = CompressionType::try_from(header_flags & 0b11)?;
        let layout_identifier = (header_flags >> 2) & 0b1111;
        if layout_identifier != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Only BGEN Layout 2 is supported by the native Rust reader. Observed layout {layout_identifier}.",
            )));
        }
        let contains_embedded_samples = ((header_flags >> 31) & 1) == 1;

        let sample_block_offset = 4 + header_block_length;
        if contains_embedded_samples {
            index::validate_sample_identifier_block(&mmap, sample_block_offset, first_variant_offset, sample_count)?;
        }

        let parsed_variant_index = match cached_variant_index.index {
            Some(cached_index) => cached_index,
            None => {
                index::parse_variant_index(&mmap, first_variant_offset, variant_count, sample_count, compression_type)?
            }
        };
        if !source.is_unchanged()? {
            return Err(BgenError::InvalidFormat(
                "BGEN source changed while its header and variant index were being read.".to_string(),
            ));
        }
        if let Some(admission_observed_at) = cached_variant_index.admission_observed_at {
            index_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(
                &source,
                &parsed_variant_index,
                admission_observed_at,
            );
        }

        Ok(Self {
            mmap,
            source,
            sample_count,
            variant_count,
            compression_type,
            packed8_validation_complete: AtomicBool::new(false),
            variant_records: parsed_variant_index.variant_records,
            variant_metadata: parsed_variant_index.variant_metadata,
            chromosome_boundary_indices: parsed_variant_index.chromosome_boundary_indices,
        })
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn variant_count(&self) -> usize {
        self.variant_count
    }

    /// Return the identity captured from the exact BGEN file opened by this reader.
    pub fn source_identity(&self) -> &g_genotype_contracts::BgenSourceIdentity {
        &self.source.identity
    }

    /// Plan uncommitted chunks without crossing chromosome boundaries.
    ///
    /// # Errors
    ///
    /// Returns an error when the chunk size is zero or chromosome boundaries
    /// cannot be normalized for this reader.
    pub fn plan_chromosome_homogeneous_chunks(
        &self,
        chunk_size: usize,
        committed_chunk_identifiers: &BTreeSet<usize>,
    ) -> GenotypeResult<Vec<ChunkSpec>> {
        crate::planner::plan_chromosome_homogeneous_chunks(
            self.variant_count,
            chunk_size,
            &self.chromosome_boundary_indices,
            committed_chunk_identifiers,
        )
    }

    /// Build an immutable decoding session for one aligned sample selection.
    ///
    /// # Errors
    ///
    /// Returns an error when the sample selection is invalid.
    pub fn read_session(&self, sample_indices: &[usize]) -> Result<BgenReadSession<'_>, BgenError> {
        self.ensure_source_unchanged("BGEN source changed before genotype delivery began.")?;
        let sample_selection = build_sample_selection(self.sample_count, sample_indices)?;
        Ok(BgenReadSession {
            reader: self,
            sample_selection,
            packed8_buffer_pool: Arc::new(Packed8BufferPool::default()),
            compressed_packed8_state: OnceLock::new(),
        })
    }

    fn scan_packed8_compatibility(&self) -> Result<Packed8Compatibility, BgenError> {
        if self.packed8_validation_complete.load(Ordering::Acquire) {
            return Ok(Packed8Compatibility::Compatible);
        }
        self.variant_records
            .par_iter()
            .enumerate()
            .map_init(ThreadScratch::default, |thread_scratch, (variant_index, variant_record)| {
                packed8::validate_variant_compatible_with_packed8(
                    &self.mmap,
                    self.compression_type,
                    variant_record,
                    self.sample_count,
                    thread_scratch,
                )
                .map_err(|error| self.contextualize_variant_error(variant_index, error))
            })
            .try_reduce(
                || Packed8Compatibility::Compatible,
                |left, right| {
                    Ok(
                        if left == Packed8Compatibility::RequiresDosage || right == Packed8Compatibility::RequiresDosage
                        {
                            Packed8Compatibility::RequiresDosage
                        } else {
                            Packed8Compatibility::Compatible
                        },
                    )
                },
            )
    }

    /// Resolve packed8 compatibility, reusing a matching persistent scan when available.
    ///
    /// Cache lookup and write failures are deliberately non-fatal: the reader
    /// performs the compatibility scan and preserves packed8 execution for the
    /// current process. BGEN parsing and globally unsupported formats remain errors;
    /// valid inputs that need dosage delivery return a typed outcome.
    ///
    /// # Errors
    ///
    /// Returns an error when the BGEN stream is corrupt or unsupported by both
    /// packed8 and dosage delivery.
    pub fn packed8_compatibility_with_cache(&self) -> Result<Packed8Compatibility, BgenError> {
        if self.packed8_validation_complete.load(Ordering::Acquire) {
            return Ok(Packed8Compatibility::Compatible);
        }
        self.ensure_source_unchanged("BGEN source changed before packed8 compatibility validation began.")?;
        let cache_entry = super::packed8_cache::ValidationCacheEntry::build(self, &self.source).ok().flatten();
        if let Some(cached_compatibility) = cache_entry.as_ref().and_then(|entry| entry.read().ok().flatten()) {
            if cached_compatibility == Packed8Compatibility::Compatible {
                self.packed8_validation_complete.store(true, Ordering::Release);
            }
            return Ok(cached_compatibility);
        }
        let compatibility = self.scan_packed8_compatibility()?;
        self.ensure_source_unchanged("BGEN source changed while packed8 compatibility was being validated.")?;
        if compatibility == Packed8Compatibility::Compatible {
            self.packed8_validation_complete.store(true, Ordering::Release);
        }
        if let Some(cache_entry) = cache_entry {
            let _ = cache_entry.write(compatibility);
        }
        Ok(compatibility)
    }

    /// Return shared metadata for a validated half-open variant range.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested range is reversed or exceeds the
    /// indexed variant count.
    pub fn variant_metadata_slice(
        &self,
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<VariantMetadataColumns, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;

        VariantMetadataColumns::new(Arc::clone(&self.variant_metadata), variant_start..variant_stop).map_err(|error| {
            contextualize_variant_metadata_invariant("Indexed BGEN variant metadata violates its invariants", error)
        })
    }

    pub(super) fn validate_packed8_probability_pair_preconditions(&self) -> Result<(), BgenError> {
        if self.packed8_validation_complete.load(Ordering::Acquire) {
            return Ok(());
        }
        Err(BgenError::UnsupportedFormat(
            "Packed8 BGEN probability-pair delivery requires packed8 compatibility validation.".to_string(),
        ))
    }

    pub(super) fn contextualize_variant_error(&self, variant_index: usize, error: BgenError) -> BgenError {
        let variant_identifier = self.variant_metadata.variant_identifier(variant_index);
        match error {
            BgenError::InvalidFormat(message) => {
                BgenError::InvalidFormat(format!("Variant '{variant_identifier}': {message}"))
            }
            BgenError::UnsupportedFormat(message) => {
                BgenError::UnsupportedFormat(format!("Variant '{variant_identifier}': {message}"))
            }
            BgenError::Range(message) => BgenError::Range(format!("Variant '{variant_identifier}': {message}")),
            BgenError::Io(source) => BgenError::Io(source),
        }
    }

    fn contextualize_variant_decode_failure(&self, variant_start: usize, failure: VariantDecodeFailure) -> BgenError {
        let Some(relative_variant_index) = failure.relative_variant_index else {
            return failure.source;
        };
        self.contextualize_variant_error(variant_start + relative_variant_index, failure.source)
    }

    fn ensure_source_unchanged(&self, message: &'static str) -> Result<(), BgenError> {
        if self.source.is_unchanged()? {
            return Ok(());
        }
        Err(BgenError::InvalidFormat(message.to_string()))
    }
}

impl BgenReadSession<'_> {
    /// Close a delivery session after verifying that its mapped source stayed stable.
    ///
    /// # Errors
    ///
    /// Returns an error when the opened BGEN changed during genotype delivery.
    pub fn finish(self) -> Result<(), BgenError> {
        self.reader.ensure_source_unchanged("BGEN source changed while genotype delivery was in progress.")
    }
}

pub(super) fn validate_variant_bounds(
    variant_start: usize,
    variant_stop: usize,
    variant_count: usize,
) -> Result<(), BgenError> {
    if variant_start > variant_stop || variant_stop > variant_count {
        return Err(BgenError::Range(format!(
            "Variant bounds must satisfy 0 <= start <= stop <= {variant_count}. Received start={variant_start}, stop={variant_stop}.",
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Seek, SeekFrom, Write};
    use std::os::unix::fs::symlink;
    use std::path::{Path, PathBuf};
    use std::sync::Barrier;
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::common::{ChunkStatisticsPolicy, GenotypeBatchPayload, OwnedGenotypeBuffer};

    const COMPLETE_STATISTICS_POLICY: ChunkStatisticsPolicy =
        ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: true, collect_sparse_candidate_mask: true };
    const TEST_INDEX_STORAGE_BYTES: usize = 1024 * 1024;

    fn temporary_bgen_path(label: &str) -> PathBuf {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after unix epoch").as_nanos();
        std::env::temp_dir().join(format!("g-reader-{label}-{}-{timestamp}.bgen", std::process::id()))
    }

    fn minimal_bgen_header_bytes(variant_count: u32, sample_count: u32, flags: u32) -> Vec<u8> {
        let mut bytes = vec![0_u8; 24];
        bytes[0..4].copy_from_slice(&20_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&20_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&variant_count.to_le_bytes());
        bytes[12..16].copy_from_slice(&sample_count.to_le_bytes());
        bytes[16..20].copy_from_slice(b"bgen");
        bytes[20..24].copy_from_slice(&flags.to_le_bytes());
        bytes
    }

    fn append_bgen_string(bytes: &mut Vec<u8>, value: &str) {
        let value_length = u16::try_from(value.len()).expect("BGEN string length should fit u16");
        bytes.extend_from_slice(&value_length.to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn trusted_probability_block(probability_bytes: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.push(2);
        bytes.push(2);
        bytes.extend_from_slice(&[2, 2, 2]);
        bytes.push(0);
        bytes.push(8);
        bytes.extend_from_slice(probability_bytes);
        bytes
    }

    fn variant_payload(probability_block: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_bgen_string(&mut bytes, "var");
        append_bgen_string(&mut bytes, "rs");
        append_bgen_string(&mut bytes, "22");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(b"A");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(b"G");
        let block_length = u32::try_from(probability_block.len()).expect("probability block should fit u32");
        bytes.extend_from_slice(&block_length.to_le_bytes());
        bytes.extend_from_slice(probability_block);
        bytes
    }

    fn write_single_variant_bgen(path: &Path) {
        let probability_block = trusted_probability_block(&[0, 0, 255, 0, 0, 255]);
        let payload = variant_payload(&probability_block);
        let mut bytes = minimal_bgen_header_bytes(1, 3, 2 << 2);
        bytes.extend_from_slice(&payload);
        fs::write(path, bytes).expect("BGEN test fixture should be written");
    }

    fn fixture_identifier_offset(bytes: &[u8]) -> usize {
        bytes.windows(4).position(|window| window == b"\x02\x00rs").expect("fixture should contain its identifier") + 2
    }

    fn replace_fixture_identifier(path: &Path, identifier: [u8; 2]) {
        let bytes = fs::read(path).expect("fixture bytes should be readable");
        let identifier_offset =
            u64::try_from(fixture_identifier_offset(&bytes)).expect("fixture offset should fit u64");
        let mut file = fs::OpenOptions::new().write(true).open(path).expect("fixture should open for editing");
        file.seek(SeekFrom::Start(identifier_offset)).expect("fixture identifier should be seekable");
        // Keep the mapped file length stable; truncation would invalidate readers' mappings.
        file.write_all(&identifier).expect("fixture identifier should be replaced");
    }

    fn restore_modification_time(path: &Path, modification_time: SystemTime) {
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("fixture should open for timestamp restoration")
            .set_times(fs::FileTimes::new().set_modified(modification_time))
            .expect("fixture modification time should be restored");
    }

    fn assert_reader_decodes_fixture(reader: &BgenReaderCore) {
        let session = reader.read_session(&[0, 1, 2]).expect("fixture session should build");
        let batch = session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect("fixture should decode after index reuse");
        let GenotypeBatchPayload::Decoded { genotypes, .. } = batch.payload else {
            panic!("dosage decode should return a decoded payload");
        };
        let OwnedGenotypeBuffer::Dosage(values) = genotypes else {
            panic!("dosage decode should return f32 values");
        };
        assert_eq!(values, vec![2.0, 0.0, 1.0]);
        session.finish().expect("fixture source should remain stable during decoding");
    }

    fn open_with_mature_index(path: &Path, cache: &Mutex<index_cache::IndexCache>) -> BgenReaderCore {
        drop(BgenReaderCore::open_with_index_cache(path, cache).expect("fixture should establish an observation"));
        cache.lock().expect("local cache should not be poisoned").age_observation_for_test(Duration::from_secs(2));
        BgenReaderCore::open_with_index_cache(path, cache)
            .expect("stable fixture should promote a freshly parsed index")
    }

    #[test]
    fn index_cache_probation_discards_initial_indexes_and_promotes_a_fresh_parse() {
        let path = temporary_bgen_path("index-cache-probation");
        write_single_variant_bgen(&path);
        let cache = Mutex::new(index_cache::IndexCache::new(TEST_INDEX_STORAGE_BYTES));
        let initial = BgenReaderCore::open_with_index_cache(&path, &cache).expect("initial fixture should open");
        let initial_records = Arc::downgrade(&initial.variant_records);
        let initial_metadata = Arc::downgrade(&initial.variant_metadata);
        drop(initial);
        assert!(initial_records.upgrade().is_none());
        assert!(initial_metadata.upgrade().is_none());

        let pending = BgenReaderCore::open_with_index_cache(&path, &cache).expect("pending fixture should reopen");
        let pending_again =
            BgenReaderCore::open_with_index_cache(&path, &cache).expect("pending fixture should reparse");
        assert!(!Arc::ptr_eq(&pending.variant_records, &pending_again.variant_records));
        assert!(!Arc::ptr_eq(&pending.variant_metadata, &pending_again.variant_metadata));
        cache.lock().expect("local cache should not be poisoned").age_observation_for_test(Duration::from_secs(2));
        let promoted = BgenReaderCore::open_with_index_cache(&path, &cache).expect("stable fixture should be promoted");
        assert!(!Arc::ptr_eq(&pending_again.variant_records, &promoted.variant_records));
        assert!(!Arc::ptr_eq(&pending_again.variant_metadata, &promoted.variant_metadata));
        let reused = BgenReaderCore::open_with_index_cache(&path, &cache).expect("promoted fixture should be reused");
        assert!(Arc::ptr_eq(&promoted.variant_records, &reused.variant_records));
        assert!(Arc::ptr_eq(&promoted.variant_metadata, &reused.variant_metadata));
        assert_reader_decodes_fixture(&reused);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn index_cache_source_replacement_restarts_probation() {
        let path = temporary_bgen_path("index-cache-probation-source");
        let replacement_path = temporary_bgen_path("index-cache-probation-replacement");
        write_single_variant_bgen(&path);
        let cache = Mutex::new(index_cache::IndexCache::new(TEST_INDEX_STORAGE_BYTES));
        let original = BgenReaderCore::open_with_index_cache(&path, &cache).expect("original fixture should open");
        cache.lock().expect("local cache should not be poisoned").age_observation_for_test(Duration::from_secs(2));
        write_single_variant_bgen(&replacement_path);
        replace_fixture_identifier(&replacement_path, *b"xy");
        fs::rename(&replacement_path, &path).expect("replacement should replace the observed source");

        let replacement = BgenReaderCore::open_with_index_cache(&path, &cache).expect("replacement should open");
        let pending = BgenReaderCore::open_with_index_cache(&path, &cache).expect("replacement should remain pending");
        assert_ne!(original.source_identity().inode_identifier, replacement.source_identity().inode_identifier);
        assert_eq!(pending.variant_metadata.variant_identifier(0), "xy");
        assert!(!Arc::ptr_eq(&replacement.variant_records, &pending.variant_records));
        assert!(!Arc::ptr_eq(&replacement.variant_metadata, &pending.variant_metadata));
        cache.lock().expect("local cache should not be poisoned").age_observation_for_test(Duration::from_secs(2));
        let promoted = BgenReaderCore::open_with_index_cache(&path, &cache).expect("stable replacement should promote");
        assert!(!Arc::ptr_eq(&pending.variant_records, &promoted.variant_records));
        assert!(!Arc::ptr_eq(&pending.variant_metadata, &promoted.variant_metadata));
        let reused = BgenReaderCore::open_with_index_cache(&path, &cache).expect("promoted replacement should reuse");
        assert!(Arc::ptr_eq(&promoted.variant_records, &reused.variant_records));
        assert!(Arc::ptr_eq(&promoted.variant_metadata, &reused.variant_metadata));
        assert_reader_decodes_fixture(&reused);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn index_cache_reuses_index_after_previous_reader_drops() {
        let path = temporary_bgen_path("index-cache-reopen");
        write_single_variant_bgen(&path);
        let cache = Mutex::new(index_cache::IndexCache::new(TEST_INDEX_STORAGE_BYTES));
        let reader = open_with_mature_index(&path, &cache);
        let variant_records = Arc::downgrade(&reader.variant_records);
        let variant_metadata = Arc::downgrade(&reader.variant_metadata);
        let chromosome_boundaries = Arc::downgrade(&reader.chromosome_boundary_indices);
        drop(reader);

        let reopened = BgenReaderCore::open_with_index_cache(&path, &cache).expect("unchanged fixture should reopen");
        assert!(Arc::ptr_eq(
            &variant_records.upgrade().expect("cache should retain indexed records"),
            &reopened.variant_records,
        ));
        assert!(Arc::ptr_eq(
            &variant_metadata.upgrade().expect("cache should retain metadata"),
            &reopened.variant_metadata,
        ));
        assert!(Arc::ptr_eq(
            &chromosome_boundaries.upgrade().expect("cache should retain chromosome boundaries"),
            &reopened.chromosome_boundary_indices,
        ));
        assert_reader_decodes_fixture(&reopened);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn index_cache_reuses_symlink_alias_with_fresh_configured_path() {
        let path = temporary_bgen_path("index-cache-source");
        let alias_path = temporary_bgen_path("index-cache-alias");
        write_single_variant_bgen(&path);
        symlink(&path, &alias_path).expect("fixture alias should be created");
        let cache = Mutex::new(index_cache::IndexCache::new(TEST_INDEX_STORAGE_BYTES));
        let reader = open_with_mature_index(&path, &cache);
        let alias_reader = BgenReaderCore::open_with_index_cache(&alias_path, &cache).expect("alias should open");

        assert!(Arc::ptr_eq(&reader.variant_records, &alias_reader.variant_records));
        assert!(Arc::ptr_eq(&reader.variant_metadata, &alias_reader.variant_metadata));
        assert_eq!(reader.source_identity().configured_path, path);
        assert_eq!(alias_reader.source_identity().configured_path, alias_path);
        assert_eq!(reader.source_identity().canonical_path, alias_reader.source_identity().canonical_path);
        assert_reader_decodes_fixture(&alias_reader);
        let _ = fs::remove_file(alias_path);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn index_cache_invalidates_same_length_edit_with_restored_modification_time() {
        let path = temporary_bgen_path("index-cache-edit");
        write_single_variant_bgen(&path);
        let modification_time = fs::metadata(&path)
            .expect("fixture metadata should exist")
            .modified()
            .expect("fixture modification time should exist");
        let cache = Mutex::new(index_cache::IndexCache::new(TEST_INDEX_STORAGE_BYTES));
        let original = BgenReaderCore::open_with_index_cache(&path, &cache).expect("fixture should open");
        replace_fixture_identifier(&path, *b"xy");
        restore_modification_time(&path, modification_time);
        let edited = BgenReaderCore::open_with_index_cache(&path, &cache).expect("edited fixture should open");

        assert_eq!(original.source_identity().file_size, edited.source_identity().file_size);
        assert_eq!(original.source_identity().inode_identifier, edited.source_identity().inode_identifier);
        assert_eq!(
            original.source_identity().modification_time_nanoseconds,
            edited.source_identity().modification_time_nanoseconds,
        );
        assert!(!Arc::ptr_eq(&original.variant_records, &edited.variant_records));
        assert!(!Arc::ptr_eq(&original.variant_metadata, &edited.variant_metadata));
        assert_eq!(original.variant_metadata.variant_identifier(0), "rs");
        assert_eq!(edited.variant_metadata.variant_identifier(0), "xy");
        if original.source_identity().change_time_nanoseconds != edited.source_identity().change_time_nanoseconds {
            assert!(matches!(original.read_session(&[0, 1, 2]), Err(BgenError::InvalidFormat(_))));
        }
        assert_reader_decodes_fixture(&edited);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn index_cache_invalidates_atomic_replacement_with_same_size_and_modification_time() {
        let path = temporary_bgen_path("index-cache-replace");
        let replacement_path = temporary_bgen_path("index-cache-replacement");
        write_single_variant_bgen(&path);
        let modification_time = fs::metadata(&path)
            .expect("fixture metadata should exist")
            .modified()
            .expect("fixture modification time should exist");
        let cache = Mutex::new(index_cache::IndexCache::new(TEST_INDEX_STORAGE_BYTES));
        let original = open_with_mature_index(&path, &cache);
        write_single_variant_bgen(&replacement_path);
        replace_fixture_identifier(&replacement_path, *b"xy");
        restore_modification_time(&replacement_path, modification_time);
        fs::rename(&replacement_path, &path).expect("replacement should atomically replace the configured source");
        let replaced = BgenReaderCore::open_with_index_cache(&path, &cache).expect("replacement should open");

        assert_eq!(original.source_identity().file_size, replaced.source_identity().file_size);
        assert_eq!(
            original.source_identity().modification_time_nanoseconds,
            replaced.source_identity().modification_time_nanoseconds,
        );
        assert_ne!(original.source_identity().inode_identifier, replaced.source_identity().inode_identifier);
        assert!(!Arc::ptr_eq(&original.variant_records, &replaced.variant_records));
        assert!(!Arc::ptr_eq(&original.variant_metadata, &replaced.variant_metadata));
        assert_eq!(replaced.variant_metadata.variant_identifier(0), "xy");
        assert!(matches!(original.read_session(&[0, 1, 2]), Err(BgenError::InvalidFormat(_))));
        assert_reader_decodes_fixture(&replaced);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn index_cache_invalidates_retargeted_symlink() {
        let first_path = temporary_bgen_path("index-cache-first-target");
        let second_path = temporary_bgen_path("index-cache-second-target");
        let alias_path = temporary_bgen_path("index-cache-retarget");
        write_single_variant_bgen(&first_path);
        write_single_variant_bgen(&second_path);
        replace_fixture_identifier(&second_path, *b"xy");
        symlink(&first_path, &alias_path).expect("first alias target should be created");
        let cache = Mutex::new(index_cache::IndexCache::new(TEST_INDEX_STORAGE_BYTES));
        let first = open_with_mature_index(&alias_path, &cache);
        fs::remove_file(&alias_path).expect("first alias should be removed");
        symlink(&second_path, &alias_path).expect("second alias target should be created");
        let second = BgenReaderCore::open_with_index_cache(&alias_path, &cache).expect("second target should open");

        assert!(!Arc::ptr_eq(&first.variant_records, &second.variant_records));
        assert!(!Arc::ptr_eq(&first.variant_metadata, &second.variant_metadata));
        assert_eq!(first.source_identity().configured_path, second.source_identity().configured_path);
        assert_ne!(first.source_identity().canonical_path, second.source_identity().canonical_path);
        assert_eq!(first.variant_metadata.variant_identifier(0), "rs");
        assert_eq!(second.variant_metadata.variant_identifier(0), "xy");
        assert!(matches!(first.read_session(&[0, 1, 2]), Err(BgenError::InvalidFormat(_))));
        assert_reader_decodes_fixture(&second);
        let _ = fs::remove_file(alias_path);
        let _ = fs::remove_file(first_path);
        let _ = fs::remove_file(second_path);
    }

    #[test]
    fn index_cache_rejects_corrupt_replacement_instead_of_reusing_metadata() {
        let path = temporary_bgen_path("index-cache-corrupt-source");
        let replacement_path = temporary_bgen_path("index-cache-corrupt-replacement");
        write_single_variant_bgen(&path);
        let modification_time = fs::metadata(&path)
            .expect("fixture metadata should exist")
            .modified()
            .expect("fixture modification time should exist");
        let cache = Mutex::new(index_cache::IndexCache::new(TEST_INDEX_STORAGE_BYTES));
        let original = open_with_mature_index(&path, &cache);
        let mut bytes = fs::read(&path).expect("fixture bytes should be readable");
        let identifier_offset = fixture_identifier_offset(&bytes);
        // Keep a valid header and unchanged file length, but break the variant index.
        bytes[identifier_offset - 2..identifier_offset].copy_from_slice(&u16::MAX.to_le_bytes());
        fs::write(&replacement_path, bytes).expect("corrupt replacement should be written");
        restore_modification_time(&replacement_path, modification_time);
        fs::rename(&replacement_path, &path).expect("corrupt replacement should replace the source");

        assert_eq!(
            fs::metadata(&path).expect("replacement metadata should exist").len(),
            original.source_identity().file_size
        );
        assert!(BgenReaderCore::open_with_index_cache(&path, &cache).is_err());
        assert!(BgenReaderCore::open_with_index_cache(&path, &cache).is_err());
        assert_eq!(original.variant_metadata.variant_identifier(0), "rs");
        assert!(matches!(original.read_session(&[0, 1, 2]), Err(BgenError::InvalidFormat(_))));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn index_cache_bypasses_indexes_exceeding_storage_budget() {
        let path = temporary_bgen_path("index-cache-zero-budget");
        write_single_variant_bgen(&path);
        let cache = Mutex::new(index_cache::IndexCache::new(0));
        let first = BgenReaderCore::open_with_index_cache(&path, &cache).expect("uncached fixture should open");
        cache.lock().expect("local cache should not be poisoned").age_observation_for_test(Duration::from_secs(2));
        let second = BgenReaderCore::open_with_index_cache(&path, &cache).expect("uncached fixture should reopen");
        let third =
            BgenReaderCore::open_with_index_cache(&path, &cache).expect("oversized index should remain uncached");

        assert!(!Arc::ptr_eq(&first.variant_records, &second.variant_records));
        assert!(!Arc::ptr_eq(&first.variant_metadata, &second.variant_metadata));
        assert!(!Arc::ptr_eq(&second.variant_records, &third.variant_records));
        assert!(!Arc::ptr_eq(&second.variant_metadata, &third.variant_metadata));
        assert_eq!(first.source_identity(), second.source_identity());
        assert_reader_decodes_fixture(&first);
        assert_reader_decodes_fixture(&second);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn index_cache_eviction_preserves_existing_reader() {
        let first_path = temporary_bgen_path("index-cache-eviction-first");
        let second_path = temporary_bgen_path("index-cache-eviction-second");
        write_single_variant_bgen(&first_path);
        write_single_variant_bgen(&second_path);
        replace_fixture_identifier(&second_path, *b"xy");
        let cache = Mutex::new(index_cache::IndexCache::new(TEST_INDEX_STORAGE_BYTES));
        let first = open_with_mature_index(&first_path, &cache);
        let second = open_with_mature_index(&second_path, &cache);

        assert_reader_decodes_fixture(&first);
        assert_reader_decodes_fixture(&second);
        assert_eq!(first.variant_metadata.variant_identifier(0), "rs");
        assert_eq!(second.variant_metadata.variant_identifier(0), "xy");
        let reopened =
            BgenReaderCore::open_with_index_cache(&first_path, &cache).expect("evicted fixture should reopen");
        assert!(!Arc::ptr_eq(&first.variant_records, &reopened.variant_records));
        assert!(!Arc::ptr_eq(&first.variant_metadata, &reopened.variant_metadata));
        assert_reader_decodes_fixture(&reopened);
        let _ = fs::remove_file(first_path);
        let _ = fs::remove_file(second_path);
    }

    #[test]
    fn index_cache_concurrent_warmed_opens_share_metadata_and_decode() {
        let path = temporary_bgen_path("index-cache-concurrent");
        write_single_variant_bgen(&path);
        let cache = Mutex::new(index_cache::IndexCache::new(TEST_INDEX_STORAGE_BYTES));
        let warmed = open_with_mature_index(&path, &cache);
        let barrier = Barrier::new(4);
        let readers = thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        let reader = BgenReaderCore::open_with_index_cache(&path, &cache)
                            .expect("concurrent warmed fixture should open");
                        assert_reader_decodes_fixture(&reader);
                        reader
                    })
                })
                .collect();
            workers.into_iter().map(|worker| worker.join().expect("reader thread should complete")).collect::<Vec<_>>()
        });

        for reader in readers {
            assert!(Arc::ptr_eq(&warmed.variant_records, &reader.variant_records));
            assert!(Arc::ptr_eq(&warmed.variant_metadata, &reader.variant_metadata));
            assert!(Arc::ptr_eq(&warmed.chromosome_boundary_indices, &reader.chromosome_boundary_indices));
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn reader_decodes_variant_major_batches_for_empty_and_selected_samples() {
        let path = temporary_bgen_path("optional-stats");
        write_single_variant_bgen(&path);
        let reader = BgenReaderCore::open(&path).expect("BGEN reader should open");

        let empty_session = reader.read_session(&[]).expect("empty selection session should build");
        let empty_batch = empty_session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect("empty selected samples should decode");
        let GenotypeBatchPayload::Decoded { genotypes: empty_genotypes, statistics: empty_statistics } =
            empty_batch.payload
        else {
            panic!("dosage decode should return a decoded payload");
        };
        let OwnedGenotypeBuffer::Dosage(empty_values) = empty_genotypes else {
            panic!("dosage decode should return f32 values");
        };
        assert!(empty_values.is_empty());
        assert_eq!(empty_statistics.output.observation_count, vec![0]);
        assert_eq!(empty_statistics.compute.genotype_mean, vec![0.0]);

        let selected_session = reader.read_session(&[0, 2]).expect("non-contiguous selection session should build");
        let selected_batch = selected_session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect("selected samples should decode");
        let GenotypeBatchPayload::Decoded { genotypes, statistics } = selected_batch.payload else {
            panic!("dosage decode should return a decoded payload");
        };
        let OwnedGenotypeBuffer::Dosage(output_values) = genotypes else {
            panic!("dosage decode should return f32 values");
        };
        assert_eq!(output_values, vec![2.0, 1.0]);
        assert_eq!(statistics.output.observation_count, vec![2]);
        assert_eq!(statistics.output.allele_one_frequency, vec![0.75]);
        assert_eq!(statistics.compute.genotype_mean, vec![1.5]);
        assert_eq!(statistics.compute.imputed_dosage_square_sum, Some(vec![5.0]));
        assert_eq!(statistics.compute.sparse_candidate_mask, Some(vec![true]));

        empty_session.finish().expect("empty delivery session source should remain stable");
        selected_session.finish().expect("selected delivery session source should remain stable");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reader_reports_invalid_metadata_ranges_as_range_errors() {
        let path = temporary_bgen_path("metadata-range");
        write_single_variant_bgen(&path);
        let reader = BgenReaderCore::open(&path).expect("BGEN reader should open");

        assert_metadata_range_error(
            &reader,
            1,
            0,
            "Variant bounds must satisfy 0 <= start <= stop <= 1. Received start=1, stop=0.",
        );
        assert_metadata_range_error(
            &reader,
            0,
            2,
            "Variant bounds must satisfy 0 <= start <= stop <= 1. Received start=0, stop=2.",
        );

        let _ = fs::remove_file(path);
    }

    fn assert_metadata_range_error(
        reader: &BgenReaderCore,
        variant_start: usize,
        variant_stop: usize,
        expected_message: &str,
    ) {
        match reader.variant_metadata_slice(variant_start, variant_stop) {
            Err(BgenError::Range(message)) => assert_eq!(message, expected_message),
            Err(other) => panic!("expected a range error, observed {other:?}"),
            Ok(_) => panic!("expected invalid metadata bounds to fail"),
        }
    }
}
