use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use rayon::prelude::*;

use g_genotype_contracts::{VariantMetadataColumns, VariantMetadataStore};

use crate::common::{ChunkSpec, Packed8BufferPool, Packed8Compatibility, SessionBufferPool};
use crate::error::GenotypeResult;

use super::decode::{VariantDecodeFailure, u32_to_usize, with_worker_thread_scratch};
use super::error::{BgenError, contextualize_variant_metadata_invariant};
use super::format::CompressionType;
use super::metadata::VariantRecord;
use super::request::BgenOpenRequest;
use super::sample_selection::{SampleSelection, build_sample_selection};
use super::source::{BgenSource, coalesced_variant_window_stop};
use super::{index, packed8};

mod variant_major;

#[derive(Debug)]
pub struct BgenReaderCore {
    pub(super) source: BgenSource,
    positioned_index: Option<PositionedBgenIndex>,
    packed8_validation_complete: AtomicBool,
}

#[derive(Debug)]
struct PositionedBgenIndex {
    sample_count: usize,
    compression_type: CompressionType,
    variant_records: Vec<VariantRecord>,
    variant_metadata: Arc<VariantMetadataStore>,
    chromosome_boundary_indices: Vec<usize>,
}

/// Immutable per-delivery BGEN decoding context.
#[derive(Debug)]
pub struct BgenReadSession<'reader> {
    pub(super) reader: &'reader BgenReaderCore,
    pub(super) sample_selection: SampleSelection,
    pub(super) packed8_buffer_pool: Arc<Packed8BufferPool>,
    pub(super) positioned_source_window_pool: Arc<SessionBufferPool<Vec<u8>>>,
    pub(super) compressed_packed8_state: OnceLock<super::raw_deflate::CompressedPacked8SessionState>,
}

#[cfg(test)]
std::thread_local! {
    static TEST_DELIVERY_SOURCE_VALIDATION_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
#[must_use]
struct TestDeliverySourceValidationHookGuard;

#[cfg(test)]
impl Drop for TestDeliverySourceValidationHookGuard {
    fn drop(&mut self) {
        TEST_DELIVERY_SOURCE_VALIDATION_HOOK.with(|hook_slot| {
            drop(hook_slot.borrow_mut().take());
        });
    }
}

#[cfg(test)]
fn install_test_delivery_source_validation_hook<ValidationHook>(
    validation_hook: ValidationHook,
) -> TestDeliverySourceValidationHookGuard
where
    ValidationHook: FnOnce() + 'static,
{
    TEST_DELIVERY_SOURCE_VALIDATION_HOOK.with(|hook_slot| {
        assert!(
            hook_slot.replace(Some(Box::new(validation_hook))).is_none(),
            "a delivery source validation hook is already installed on this test thread",
        );
    });
    TestDeliverySourceValidationHookGuard
}

#[cfg(test)]
fn run_test_delivery_source_validation_hook() {
    let validation_hook = TEST_DELIVERY_SOURCE_VALIDATION_HOOK.with(|hook_slot| hook_slot.borrow_mut().take());
    if let Some(validation_hook) = validation_hook {
        validation_hook();
    }
}

impl BgenReaderCore {
    /// Open and index a Layout 2 BGEN source.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened or read, its header or
    /// variant index is invalid, its layout is unsupported, or it changes while
    /// being indexed.
    pub fn open(bgen_path: &Path) -> Result<Self, BgenError> {
        Self::open_request(BgenOpenRequest { locator: bgen_path, content_selector: None })
    }

    /// Resolve and index one typed BGEN open request.
    ///
    /// A content-selected process-cache hit resolves entirely from authenticated
    /// immutable bytes and does not access the request locator. Unselected opens
    /// always acquire their locator independently and never consult or publish
    /// the process snapshot cache.
    ///
    /// # Errors
    ///
    /// Returns an error when acquisition fails, selected content does not match,
    /// a selected input cannot use an owned snapshot, or BGEN parsing fails.
    pub fn open_request(request: BgenOpenRequest<'_>) -> Result<Self, BgenError> {
        Self::open_source(BgenSource::open_request(request)?)
    }

    /// Open with positioned I/O for same-input benchmark comparisons.
    ///
    /// This does not change the production [`Self::open`] policy.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::open`].
    #[cfg(feature = "benchmark-positioned-source")]
    #[doc(hidden)]
    pub fn open_positioned_for_benchmark(bgen_path: &Path) -> Result<Self, BgenError> {
        Self::open_source(BgenSource::open_with_snapshot_limit(bgen_path, 0)?)
    }

    fn open_source(mut source: BgenSource) -> Result<Self, BgenError> {
        if source.snapshot_payload().is_some() {
            return Ok(Self { source, positioned_index: None, packed8_validation_complete: AtomicBool::new(false) });
        }
        if source.snapshot_bytes().is_none() {
            ensure_source_unchanged(&source, "BGEN source changed before its header was read.")?;
        }
        let header = parse_bgen_header(&source)?;
        if header.contains_embedded_samples {
            index::validate_sample_identifier_block(
                &source,
                header.sample_block_offset,
                header.first_variant_offset,
                header.sample_count,
            )?;
        }

        let parsed_variant_index = index::parse_variant_index(
            &source,
            header.first_variant_offset,
            header.variant_count,
            header.sample_count,
            header.compression_type,
        )?;
        ensure_source_unchanged(&source, "BGEN source changed while its header and variant index were being read.")?;

        let positioned_index = if source.snapshot_bytes().is_some() {
            source.publish_snapshot_payload(
                header.sample_count,
                header.compression_type,
                parsed_variant_index.variant_records,
                parsed_variant_index.variant_metadata,
                parsed_variant_index.chromosome_boundary_indices,
            )?;
            None
        } else {
            Some(PositionedBgenIndex {
                sample_count: header.sample_count,
                compression_type: header.compression_type,
                variant_records: parsed_variant_index.variant_records,
                variant_metadata: parsed_variant_index.variant_metadata,
                chromosome_boundary_indices: parsed_variant_index.chromosome_boundary_indices,
            })
        };
        Ok(Self { source, positioned_index, packed8_validation_complete: AtomicBool::new(false) })
    }

    pub fn sample_count(&self) -> usize {
        self.source
            .snapshot_payload()
            .map_or_else(|| self.positioned_index().sample_count, |payload| payload.sample_count)
    }

    pub fn variant_count(&self) -> usize {
        self.variant_records().len()
    }

    pub(super) fn compression_type(&self) -> CompressionType {
        self.source
            .snapshot_payload()
            .map_or_else(|| self.positioned_index().compression_type, |payload| payload.compression_type)
    }

    pub(super) fn variant_records(&self) -> &[VariantRecord] {
        self.source.snapshot_payload().map_or_else(
            || self.positioned_index().variant_records.as_slice(),
            |payload| payload.variant_records.as_slice(),
        )
    }

    fn variant_metadata(&self) -> &Arc<VariantMetadataStore> {
        self.source
            .snapshot_payload()
            .map_or_else(|| &self.positioned_index().variant_metadata, |payload| &payload.variant_metadata)
    }

    fn chromosome_boundary_indices(&self) -> &[usize] {
        self.source.snapshot_payload().map_or_else(
            || self.positioned_index().chromosome_boundary_indices.as_slice(),
            |payload| payload.chromosome_boundary_indices.as_slice(),
        )
    }

    fn positioned_index(&self) -> &PositionedBgenIndex {
        self.positioned_index.as_ref().expect("a BGEN reader without a parsed snapshot must own a positioned index")
    }

    /// Return descriptor metadata from the source acquisition.
    ///
    /// On a process snapshot-cache hit this is the original capture identity,
    /// not evidence that the current request locator was opened. This accessor
    /// is provenance-only; use [`Self::content_evidence`] for content authority.
    pub fn source_identity(&self) -> &g_genotype_contracts::BgenSourceIdentity {
        self.source.identity()
    }

    /// Return authoritative snapshot evidence or positioned unattested evidence.
    pub fn content_evidence(&self) -> &g_genotype_contracts::BgenContentEvidence {
        self.source.content_evidence()
    }

    /// Return acquisition provenance for this reader request.
    pub fn source_provenance(&self) -> &g_genotype_contracts::BgenSourceProvenance {
        self.source.provenance()
    }

    /// Report whether this reader opened from the canonical process snapshot.
    #[cfg(feature = "benchmark-internals")]
    #[doc(hidden)]
    #[must_use]
    pub fn opened_from_process_snapshot_cache(&self) -> bool {
        self.source.snapshot_cache_hit()
    }

    /// Validate an external sample count against the BGEN header.
    ///
    /// # Errors
    ///
    /// Returns an error when the expected sample count differs from the count
    /// stored in the BGEN header.
    pub fn validate_expected_sample_count(&self, expected_sample_count: usize) -> Result<(), BgenError> {
        if expected_sample_count == self.sample_count() {
            return Ok(());
        }
        Err(BgenError::InvalidFormat(format!(
            "BGEN header reports {} samples, but aligned sample metadata reports {expected_sample_count}.",
            self.sample_count(),
        )))
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
            self.variant_count(),
            chunk_size,
            self.chromosome_boundary_indices(),
            committed_chunk_identifiers,
        )
    }

    /// Build an immutable decoding session for one aligned sample selection.
    ///
    /// # Errors
    ///
    /// Returns an error when the sample selection is invalid or a positioned
    /// source changed before delivery began.
    pub fn read_session(&self, sample_indices: &[usize]) -> Result<BgenReadSession<'_>, BgenError> {
        self.ensure_delivery_source_unchanged("BGEN source changed before genotype delivery began.")?;
        let sample_selection = build_sample_selection(self.sample_count(), sample_indices)?;
        Ok(BgenReadSession {
            reader: self,
            sample_selection,
            packed8_buffer_pool: Arc::new(Packed8BufferPool::default()),
            positioned_source_window_pool: Arc::new(SessionBufferPool::default()),
            compressed_packed8_state: OnceLock::new(),
        })
    }

    fn scan_packed8_compatibility(&self) -> Result<Packed8Compatibility, BgenError> {
        if self.packed8_validation_complete.load(Ordering::Acquire) {
            return Ok(Packed8Compatibility::Compatible);
        }
        if let Some(source_window) = self.source.full_snapshot_window() {
            return self.scan_packed8_compatibility_window(0, self.variant_records(), source_window);
        }

        let mut compatibility = Packed8Compatibility::Compatible;
        let mut source_window_buffer = Vec::new();
        let mut window_variant_start = 0_usize;
        while window_variant_start < self.variant_records().len() {
            let window_variant_stop = coalesced_variant_window_stop(self.variant_records(), window_variant_start)
                .map_err(|error| self.contextualize_variant_error(window_variant_start, error))?;
            let window_variant_records = &self.variant_records()[window_variant_start..window_variant_stop];
            let source_window = self
                .source
                .read_variant_window(window_variant_records, &mut source_window_buffer)
                .map_err(|error| self.contextualize_variant_error(window_variant_start, error))?;
            let window_compatibility =
                self.scan_packed8_compatibility_window(window_variant_start, window_variant_records, source_window)?;
            if window_compatibility == Packed8Compatibility::RequiresDosage {
                compatibility = Packed8Compatibility::RequiresDosage;
            }
            window_variant_start = window_variant_stop;
        }
        Ok(compatibility)
    }

    fn scan_packed8_compatibility_window(
        &self,
        window_variant_start: usize,
        variant_records: &[VariantRecord],
        source_window: super::source::BgenByteWindow<'_>,
    ) -> Result<Packed8Compatibility, BgenError> {
        variant_records
            .par_iter()
            .enumerate()
            .map(|(relative_variant_index, variant_record)| {
                with_worker_thread_scratch(|thread_scratch| {
                    packed8::validate_variant_compatible_with_packed8(
                        source_window,
                        self.compression_type(),
                        variant_record,
                        self.sample_count(),
                        thread_scratch,
                    )
                    .map_err(|error| {
                        self.contextualize_variant_error(window_variant_start + relative_variant_index, error)
                    })
                })
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
        self.ensure_delivery_source_unchanged("BGEN source changed before packed8 compatibility validation began.")?;
        if self.packed8_validation_complete.load(Ordering::Acquire) {
            return Ok(Packed8Compatibility::Compatible);
        }
        let cache_entry = super::packed8_cache::ValidationCacheEntry::build(
            self.source.content_evidence(),
            self.sample_count(),
            self.variant_count(),
        )
        .ok()
        .flatten();
        if let Some(cached_compatibility) = cache_entry.as_ref().and_then(|entry| entry.read().ok().flatten()) {
            self.ensure_delivery_source_unchanged(
                "BGEN source changed while cached packed8 compatibility was being read.",
            )?;
            if cached_compatibility == Packed8Compatibility::Compatible {
                self.packed8_validation_complete.store(true, Ordering::Release);
            }
            return Ok(cached_compatibility);
        }
        let compatibility = self.scan_packed8_compatibility()?;
        self.ensure_delivery_source_unchanged("BGEN source changed while packed8 compatibility was being validated.")?;
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
        validate_variant_bounds(variant_start, variant_stop, self.variant_count())?;

        VariantMetadataColumns::new(Arc::clone(self.variant_metadata()), variant_start..variant_stop).map_err(|error| {
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
        let variant_identifier = self.variant_metadata().variant_identifier(variant_index);
        let context = format!("Variant index {variant_index} ('{variant_identifier}')");
        match error {
            BgenError::InvalidFormat(message) => BgenError::InvalidFormat(format!("{context}: {message}")),
            BgenError::UnsupportedFormat(message) => BgenError::UnsupportedFormat(format!("{context}: {message}")),
            BgenError::Range(message) => BgenError::Range(format!("{context}: {message}")),
            BgenError::Io(source) => BgenError::Io(source),
            error @ (BgenError::ContentSha256Mismatch { .. }
            | BgenError::ContentByteCountMismatch { .. }
            | BgenError::ContentSelectionRequiresOwnedSnapshot { .. }) => error,
        }
    }

    fn contextualize_variant_decode_failure(&self, variant_start: usize, failure: VariantDecodeFailure) -> BgenError {
        let Some(relative_variant_index) = failure.relative_variant_index else {
            return failure.source;
        };
        self.contextualize_variant_error(variant_start + relative_variant_index, failure.source)
    }

    pub(super) fn ensure_source_unchanged(&self, message: &'static str) -> Result<(), BgenError> {
        ensure_source_unchanged(&self.source, message)
    }

    pub(super) fn ensure_delivery_source_unchanged(&self, message: &'static str) -> Result<(), BgenError> {
        #[cfg(test)]
        run_test_delivery_source_validation_hook();
        if self.source.snapshot_bytes().is_some() {
            return Ok(());
        }
        self.ensure_source_unchanged(message)
    }
}

struct ParsedBgenHeader {
    first_variant_offset: u64,
    sample_block_offset: u64,
    variant_count: usize,
    sample_count: usize,
    compression_type: CompressionType,
    contains_embedded_samples: bool,
}

fn parse_bgen_header(source: &BgenSource) -> Result<ParsedBgenHeader, BgenError> {
    const SUPPORTED_HEADER_FLAG_MASK: u32 = 0x8000_003F;

    let relative_first_variant_offset = u64::from(source.read_u32_at(0)?);
    let first_variant_offset = 4_u64
        .checked_add(relative_first_variant_offset)
        .ok_or_else(|| BgenError::Range("BGEN first-variant offset overflowed uint64.".to_string()))?;
    let header_block_length = u64::from(source.read_u32_at(4)?);
    if header_block_length < 20 {
        return Err(BgenError::InvalidFormat(format!(
            "BGEN header block length must be at least 20 bytes. Observed {header_block_length}.",
        )));
    }
    if header_block_length > relative_first_variant_offset {
        return Err(BgenError::InvalidFormat(format!(
            "BGEN header block length {header_block_length} exceeds the first-variant offset {relative_first_variant_offset}.",
        )));
    }
    if first_variant_offset > source.length() {
        return Err(BgenError::InvalidFormat(format!(
            "BGEN first-variant offset {first_variant_offset} exceeds the source length {}.",
            source.length(),
        )));
    }

    let variant_count = u32_to_usize(source.read_u32_at(8)?)?;
    let sample_count = u32_to_usize(source.read_u32_at(12)?)?;
    let mut magic_number = [0_u8; 4];
    source.read_exact_at(16, &mut magic_number)?;
    if magic_number != *b"bgen" && magic_number != [0_u8; 4] {
        return Err(BgenError::InvalidFormat(
            "BGEN header magic number must be `bgen` or four zero bytes.".to_string(),
        ));
    }

    let header_flags = source.read_u32_at(header_block_length)?;
    let reserved_header_flags = header_flags & !SUPPORTED_HEADER_FLAG_MASK;
    if reserved_header_flags != 0 {
        return Err(BgenError::InvalidFormat(format!(
            "BGEN header sets reserved flag bits 0x{reserved_header_flags:08x}.",
        )));
    }
    let compression_type = CompressionType::try_from(header_flags & 0b11)?;
    let layout_identifier = (header_flags >> 2) & 0b1111;
    if layout_identifier != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Only BGEN Layout 2 is supported by the native Rust reader. Observed layout {layout_identifier}.",
        )));
    }
    let contains_embedded_samples = ((header_flags >> 31) & 1) == 1;
    let sample_block_offset = 4_u64
        .checked_add(header_block_length)
        .ok_or_else(|| BgenError::Range("BGEN sample-block offset overflowed uint64.".to_string()))?;
    if sample_block_offset > first_variant_offset {
        return Err(BgenError::InvalidFormat(
            "BGEN first-variant offset precedes the end of the header block.".to_string(),
        ));
    }
    if variant_count == 0 && first_variant_offset != source.length() {
        return Err(BgenError::InvalidFormat(format!(
            "A BGEN source with zero variants must point its first-variant offset to the end of the file at byte {}. Observed {first_variant_offset}.",
            source.length(),
        )));
    }

    Ok(ParsedBgenHeader {
        first_variant_offset,
        sample_block_offset,
        variant_count,
        sample_count,
        compression_type,
        contains_embedded_samples,
    })
}

fn ensure_source_unchanged(source: &BgenSource, message: &'static str) -> Result<(), BgenError> {
    if source.is_unchanged()? {
        return Ok(());
    }
    Err(BgenError::InvalidFormat(message.to_string()))
}

impl BgenReadSession<'_> {
    /// Close a delivery session.
    ///
    /// Immutable snapshot sessions need no terminal source check. Positioned
    /// sessions verify that the opened descriptor and configured path stayed
    /// stable.
    ///
    /// # Errors
    ///
    /// Returns an error when a positioned BGEN changed during genotype
    /// delivery.
    pub fn finish(self) -> Result<(), BgenError> {
        self.reader.ensure_delivery_source_unchanged("BGEN source changed while genotype delivery was in progress.")
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
    use std::fs::{self, File, FileTimes, OpenOptions};
    use std::io::{Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use g_genotype_contracts::{BgenContentEvidence, BgenContentSha256, BgenSnapshotResolution};
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::bgen::BgenContentSelector;
    use crate::common::{ChunkStatisticsPolicy, GenotypeBatchPayload, OwnedGenotypeBuffer};

    const COMPLETE_STATISTICS_POLICY: ChunkStatisticsPolicy =
        ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: true, collect_sparse_candidate_mask: true };
    const TEST_COORDINATION_TIMEOUT: Duration = Duration::from_secs(10);

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

    fn append_bgen_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
        let value_length = u16::try_from(value.len()).expect("BGEN byte-string length should fit u16");
        bytes.extend_from_slice(&value_length.to_le_bytes());
        bytes.extend_from_slice(value);
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
        variant_payload_with_metadata(b"var", b"rs", b"22", b"A", b"G", probability_block)
    }

    fn variant_payload_with_metadata(
        variant_identifier: &[u8],
        rsid: &[u8],
        chromosome: &[u8],
        reference_allele: &[u8],
        counted_allele: &[u8],
        probability_block: &[u8],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_bgen_bytes(&mut bytes, variant_identifier);
        append_bgen_bytes(&mut bytes, rsid);
        append_bgen_bytes(&mut bytes, chromosome);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        let reference_allele_length =
            u32::try_from(reference_allele.len()).expect("reference allele length should fit u32");
        bytes.extend_from_slice(&reference_allele_length.to_le_bytes());
        bytes.extend_from_slice(reference_allele);
        let counted_allele_length = u32::try_from(counted_allele.len()).expect("counted allele length should fit u32");
        bytes.extend_from_slice(&counted_allele_length.to_le_bytes());
        bytes.extend_from_slice(counted_allele);
        let block_length = u32::try_from(probability_block.len()).expect("probability block should fit u32");
        bytes.extend_from_slice(&block_length.to_le_bytes());
        bytes.extend_from_slice(probability_block);
        bytes
    }

    fn compressed_variant_payload_with_declared_length(declared_uncompressed_block_length: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_bgen_bytes(&mut bytes, b"var");
        append_bgen_bytes(&mut bytes, b"rs");
        append_bgen_bytes(&mut bytes, b"22");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(b"A");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(b"G");
        bytes.extend_from_slice(&5_u32.to_le_bytes());
        bytes.extend_from_slice(&declared_uncompressed_block_length.to_le_bytes());
        bytes.push(0);
        bytes
    }

    fn write_single_variant_bgen(path: &Path) {
        write_single_variant_bgen_with_probabilities(path, &[0, 0, 255, 0, 0, 255]);
    }

    fn write_single_variant_bgen_with_probabilities(path: &Path, probability_bytes: &[u8]) {
        let probability_block = trusted_probability_block(probability_bytes);
        let payload = variant_payload(&probability_block);
        write_variant_payloads_bgen(path, &[payload]);
    }

    fn write_variant_payloads_bgen(path: &Path, payloads: &[Vec<u8>]) {
        let variant_count = u32::try_from(payloads.len()).expect("BGEN test variant count should fit u32");
        let mut bytes = minimal_bgen_header_bytes(variant_count, 3, 2 << 2);
        for payload in payloads {
            bytes.extend_from_slice(payload);
        }
        fs::write(path, bytes).expect("BGEN test fixture should be written");
    }

    fn content_selector(path: &Path) -> BgenContentSelector {
        let bytes = fs::read(path).expect("selected BGEN fixture should be readable");
        BgenContentSelector {
            content_sha256: BgenContentSha256::from_bytes(Sha256::digest(&bytes).into()),
            expected_byte_count: Some(u64::try_from(bytes.len()).expect("BGEN fixture length should fit uint64")),
        }
    }

    fn open_selected_reader(
        locator: &Path,
        content_selector: BgenContentSelector,
        snapshot_cache: &'static super::super::source::BgenSnapshotCache,
    ) -> BgenReaderCore {
        let source = BgenSource::open_request_with_test_snapshot_cache(
            BgenOpenRequest { locator, content_selector: Some(content_selector) },
            snapshot_cache,
        )
        .expect("selected BGEN source should resolve");
        BgenReaderCore::open_source(source).expect("selected BGEN source should parse")
    }

    fn snapshot_cache_is_empty(snapshot_cache: &super::super::source::BgenSnapshotCache) -> bool {
        snapshot_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_none()
    }

    fn decoded_dosages(reader: &BgenReaderCore) -> Vec<f32> {
        let session = reader.read_session(&[0, 1, 2]).expect("snapshot read session should build");
        let batch = session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect("snapshot batch should decode");
        session.finish().expect("owned snapshot should finish without locator access");
        let GenotypeBatchPayload::Decoded { genotypes, .. } = batch.payload else {
            panic!("snapshot decode should return decoded values");
        };
        let OwnedGenotypeBuffer::Dosage(values) = genotypes else {
            panic!("snapshot decode should return dosage values");
        };
        values
    }

    fn write_positioned_single_variant_bgen(path: &Path) {
        write_positioned_variant_count_bgen(path, 1);
    }

    fn write_positioned_single_variant_bgen_with_probabilities(path: &Path, probability_bytes: &[u8]) {
        let probability_block = trusted_probability_block(probability_bytes);
        let payload = variant_payload(&probability_block);
        write_positioned_variant_payloads_bgen(path, &[payload]);
    }

    fn write_positioned_variant_count_bgen(path: &Path, variant_count: u32) {
        let probability_block = trusted_probability_block(&[0, 0, 255, 0, 0, 255]);
        let payload = variant_payload(&probability_block);
        let payloads = vec![payload; usize::try_from(variant_count).expect("test variant count should fit usize")];
        write_positioned_variant_payloads_bgen(path, &payloads);
    }

    fn write_positioned_variant_payloads_bgen(path: &Path, payloads: &[Vec<u8>]) {
        let variant_count = u32::try_from(payloads.len()).expect("BGEN test variant count should fit u32");
        let first_variant_offset = crate::bgen::source::MAXIMUM_OWNED_SNAPSHOT_BYTE_COUNT + 1;
        let relative_first_variant_offset =
            u32::try_from(first_variant_offset - 4).expect("positioned test offset should fit u32");
        let mut header = minimal_bgen_header_bytes(variant_count, 3, 2 << 2);
        header[0..4].copy_from_slice(&relative_first_variant_offset.to_le_bytes());
        let mut file = File::create(path).expect("positioned BGEN test fixture should be created");
        file.write_all(&header).expect("positioned BGEN header should be written");
        file.seek(SeekFrom::Start(first_variant_offset)).expect("positioned BGEN payload offset should be seekable");
        for payload in payloads {
            file.write_all(payload).expect("positioned BGEN payload should be written");
        }
    }

    fn large_metadata_variant_payload(probability_bytes: &[u8]) -> Vec<u8> {
        let reference_allele = vec![b'A'; crate::bgen::source::MAXIMUM_SOURCE_WINDOW_BYTE_COUNT];
        variant_payload_with_metadata(
            b"large",
            b"rs-large",
            b"22",
            &reference_allele,
            b"G",
            &trusted_probability_block(probability_bytes),
        )
    }

    fn set_first_variant_offset(bytes: &mut [u8], first_variant_offset: usize) {
        let relative_first_variant_offset =
            u32::try_from(first_variant_offset.checked_sub(4).expect("BGEN offset should include prefix"))
                .expect("test BGEN offset should fit u32");
        bytes[0..4].copy_from_slice(&relative_first_variant_offset.to_le_bytes());
    }

    fn assert_invalid_bgen(bytes: &[u8], label: &str, expected_message: &str) {
        let path = temporary_bgen_path(label);
        fs::write(&path, bytes).expect("invalid BGEN fixture should be written");
        let error = BgenReaderCore::open(&path).expect_err("invalid BGEN fixture should fail");
        assert!(
            error.to_string().contains(expected_message),
            "expected `{expected_message}` in BGEN error, observed `{error}`"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn reader_decodes_variant_major_batches_for_empty_and_selected_samples() {
        let path = temporary_bgen_path("optional-stats");
        write_single_variant_bgen(&path);
        let reader = BgenReaderCore::open(&path).expect("BGEN reader should open");
        reader.validate_expected_sample_count(3).expect("matching external sample count should pass");
        assert!(
            reader
                .validate_expected_sample_count(2)
                .expect_err("mismatched external sample count should fail")
                .to_string()
                .contains("aligned sample metadata reports 2")
        );
        assert!(reader.source.snapshot_bytes().is_some(), "small BGEN inputs should use an owned snapshot");

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
    fn selected_snapshot_hit_uses_only_content_and_unselected_open_stays_isolated() {
        let original_path = temporary_bgen_path("content-cache-original");
        let alternate_path = temporary_bgen_path("content-cache-alternate");
        let missing_path = temporary_bgen_path("content-cache-missing");
        write_single_variant_bgen(&original_path);
        write_single_variant_bgen_with_probabilities(&alternate_path, &[255, 0, 255, 0, 255, 0]);
        let original_selector = content_selector(&original_path);
        let snapshot_cache = super::super::source::new_test_snapshot_cache();

        let original_reader = open_selected_reader(&original_path, original_selector, snapshot_cache);
        let original_payload =
            original_reader.source.snapshot_payload_arc().expect("selected reader should own a parsed snapshot");
        let BgenContentEvidence::OwnedSnapshot(original_fingerprint) = original_reader.content_evidence() else {
            panic!("selected reader should expose owned-snapshot evidence");
        };
        assert_eq!(original_fingerprint.content_sha256, original_selector.content_sha256);
        assert_eq!(original_reader.source_provenance().requested_path, original_path);
        assert_eq!(original_reader.source_provenance().resolution, BgenSnapshotResolution::CapturedFromLocator,);

        let alternate_hit_reader = {
            let locator_hook_guard = super::super::source::install_test_locator_acquisition_hook(|| {
                panic!("selected cache hit must not access its request locator");
            });
            let reader = open_selected_reader(&alternate_path, original_selector, snapshot_cache);
            drop(locator_hook_guard);
            reader
        };
        let alternate_hit_payload =
            alternate_hit_reader.source.snapshot_payload_arc().expect("cache-hit reader should own a parsed snapshot");
        assert!(Arc::ptr_eq(&original_payload, &alternate_hit_payload));
        assert_eq!(decoded_dosages(&alternate_hit_reader), vec![2.0, 0.0, 1.0]);
        assert_eq!(alternate_hit_reader.source_provenance().requested_path, alternate_path);
        assert_eq!(
            alternate_hit_reader.source_provenance().captured_source_identity,
            original_reader.source_provenance().captured_source_identity,
        );
        assert_eq!(alternate_hit_reader.source_provenance().resolution, BgenSnapshotResolution::ProcessSnapshotCache,);

        let missing_hit_reader = {
            let locator_hook_guard = super::super::source::install_test_locator_acquisition_hook(|| {
                panic!("selected cache hit must not inspect a missing locator");
            });
            let reader = open_selected_reader(&missing_path, original_selector, snapshot_cache);
            drop(locator_hook_guard);
            reader
        };
        assert_eq!(decoded_dosages(&missing_hit_reader), vec![2.0, 0.0, 1.0]);
        assert_eq!(missing_hit_reader.source_provenance().requested_path, missing_path);
        assert_eq!(missing_hit_reader.source_provenance().resolution, BgenSnapshotResolution::ProcessSnapshotCache,);

        let digest_only_selector = BgenContentSelector { expected_byte_count: None, ..original_selector };
        let digest_only_hit_reader = {
            let locator_hook_guard = super::super::source::install_test_locator_acquisition_hook(|| {
                panic!("digest-only selected cache hit must not inspect its request locator");
            });
            let reader = open_selected_reader(&missing_path, digest_only_selector, snapshot_cache);
            drop(locator_hook_guard);
            reader
        };
        let BgenContentEvidence::OwnedSnapshot(digest_only_fingerprint) = digest_only_hit_reader.content_evidence()
        else {
            panic!("digest-only selected cache hit should inherit the retained fingerprint");
        };
        assert_eq!(*digest_only_fingerprint, *original_fingerprint);
        assert_eq!(digest_only_hit_reader.source_provenance().resolution, BgenSnapshotResolution::ProcessSnapshotCache,);

        let unselected_source =
            BgenSource::open_unselected_for_test(&alternate_path).expect("unselected alternate source should open");
        assert!(!unselected_source.snapshot_cache_hit());
        let unselected_reader =
            BgenReaderCore::open_source(unselected_source).expect("unselected alternate source should parse");
        let unselected_payload =
            unselected_reader.source.snapshot_payload_arc().expect("unselected reader should own a parsed snapshot");
        assert!(!Arc::ptr_eq(&original_payload, &unselected_payload));
        assert_eq!(decoded_dosages(&unselected_reader), vec![0.0, 0.0, 0.0]);
        assert_eq!(unselected_reader.source_provenance().resolution, BgenSnapshotResolution::CapturedFromLocator,);

        let selected_after_unselected = open_selected_reader(&missing_path, original_selector, snapshot_cache);
        let selected_after_unselected_payload =
            selected_after_unselected.source.snapshot_payload_arc().expect("retained selected snapshot should resolve");
        assert!(Arc::ptr_eq(&original_payload, &selected_after_unselected_payload));

        let mismatched_count_selector =
            BgenContentSelector { expected_byte_count: Some(original_fingerprint.byte_count + 1), ..original_selector };
        let locator_hook_guard = super::super::source::install_test_locator_acquisition_hook(|| {
            panic!("cached byte-count mismatch must not inspect the locator");
        });
        let count_error = BgenSource::open_request_with_test_snapshot_cache(
            BgenOpenRequest { locator: &missing_path, content_selector: Some(mismatched_count_selector) },
            snapshot_cache,
        )
        .expect_err("cached byte-count mismatch should fail");
        drop(locator_hook_guard);
        assert!(matches!(
            count_error,
            BgenError::ContentByteCountMismatch {
                expected_byte_count,
                observed_byte_count,
            } if expected_byte_count == original_fingerprint.byte_count + 1
                && observed_byte_count == original_fingerprint.byte_count
        ));

        let _ = fs::remove_file(original_path);
        let _ = fs::remove_file(alternate_path);
    }

    #[test]
    fn unselected_open_never_reuses_identical_or_rewritten_locator_content() {
        let path = temporary_bgen_path("unselected-equal-metadata");
        write_single_variant_bgen(&path);
        let original_metadata = fs::metadata(&path).expect("original metadata should resolve");
        let original_modified = original_metadata.modified().expect("original modification time should resolve");
        let snapshot_cache = super::super::source::new_test_snapshot_cache();

        let first_source = BgenSource::open_unselected_for_test(&path).expect("first unselected source should open");
        let first_reader = BgenReaderCore::open_source(first_source).expect("first unselected source should parse");
        let first_payload = first_reader.source.snapshot_payload_arc().expect("first source should own its payload");
        let BgenContentEvidence::OwnedSnapshot(first_fingerprint) = first_reader.content_evidence() else {
            panic!("first unselected source should have owned evidence");
        };
        let first_fingerprint = *first_fingerprint;

        let identical_source =
            BgenSource::open_unselected_for_test(&path).expect("identical unselected source should open");
        let identical_reader =
            BgenReaderCore::open_source(identical_source).expect("identical unselected source should parse");
        let identical_payload =
            identical_reader.source.snapshot_payload_arc().expect("identical source should own its payload");
        assert!(!Arc::ptr_eq(&first_payload, &identical_payload));
        assert_eq!(first_reader.content_evidence(), identical_reader.content_evidence());

        write_single_variant_bgen_with_probabilities(&path, &[255, 0, 255, 0, 255, 0]);
        File::options()
            .write(true)
            .open(&path)
            .expect("rewritten fixture should reopen")
            .set_times(FileTimes::new().set_modified(original_modified))
            .expect("rewritten fixture modification time should be restored");
        let rewritten_metadata = fs::metadata(&path).expect("rewritten metadata should resolve");
        assert_eq!(rewritten_metadata.len(), original_metadata.len());

        let second_source = BgenSource::open_unselected_for_test(&path).expect("second unselected source should open");
        let second_reader = BgenReaderCore::open_source(second_source).expect("second unselected source should parse");
        let second_payload = second_reader.source.snapshot_payload_arc().expect("second source should own its payload");
        let BgenContentEvidence::OwnedSnapshot(second_fingerprint) = second_reader.content_evidence() else {
            panic!("second unselected source should have owned evidence");
        };
        let second_fingerprint = *second_fingerprint;

        assert!(!Arc::ptr_eq(&first_payload, &second_payload));
        assert_ne!(first_fingerprint, second_fingerprint);
        assert_eq!(decoded_dosages(&first_reader), vec![2.0, 0.0, 1.0]);
        assert_eq!(decoded_dosages(&identical_reader), vec![2.0, 0.0, 1.0]);
        assert_eq!(decoded_dosages(&second_reader), vec![0.0, 0.0, 0.0]);
        assert!(snapshot_cache_is_empty(snapshot_cache), "unselected opens must never publish");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn public_unselected_open_never_participates_in_the_snapshot_cache() {
        let selected_path = temporary_bgen_path("public-unselected-cache-selected");
        let unselected_path = temporary_bgen_path("public-unselected-cache-independent");
        let missing_path = temporary_bgen_path("public-unselected-cache-missing");
        write_single_variant_bgen(&selected_path);
        write_single_variant_bgen_with_probabilities(&unselected_path, &[255, 0, 255, 0, 255, 0]);
        let selector = content_selector(&selected_path);

        let selected_reader =
            BgenReaderCore::open_request(BgenOpenRequest { locator: &selected_path, content_selector: Some(selector) })
                .expect("selected public open should seed the process cache");
        let selected_payload =
            selected_reader.source.snapshot_payload_arc().expect("selected public open should publish its payload");

        let cache_observation = super::super::source::observe_test_snapshot_cache_operations();
        let unselected_reader =
            BgenReaderCore::open(&unselected_path).expect("unselected public open should parse independently");
        assert_eq!(decoded_dosages(&unselected_reader), vec![0.0, 0.0, 0.0]);
        assert_eq!(
            cache_observation.operation_counts().observed_counts(),
            super::super::source::TestSnapshotCacheOperationCountSnapshot::default(),
            "unselected public open must not resolve, initialize, consult, lock, publish, or evict the snapshot cache",
        );
        drop(cache_observation);

        let selected_reopen =
            BgenReaderCore::open_request(BgenOpenRequest { locator: &missing_path, content_selector: Some(selector) })
                .expect("selected payload should remain cached after an unselected public open");
        let selected_reopen_payload = selected_reopen
            .source
            .snapshot_payload_arc()
            .expect("selected cache hit should retain the canonical payload");
        assert!(Arc::ptr_eq(&selected_payload, &selected_reopen_payload));

        let _ = fs::remove_file(selected_path);
        let _ = fs::remove_file(unselected_path);
    }

    #[test]
    fn selected_failures_preserve_the_retained_valid_content() {
        let original_path = temporary_bgen_path("selected-failure-original");
        let alternate_path = temporary_bgen_path("selected-failure-alternate");
        let malformed_path = temporary_bgen_path("selected-failure-malformed");
        let missing_path = temporary_bgen_path("selected-failure-missing");
        write_single_variant_bgen(&original_path);
        write_single_variant_bgen_with_probabilities(&alternate_path, &[255, 0, 255, 0, 255, 0]);
        fs::write(&malformed_path, minimal_bgen_header_bytes(0, 0, (2 << 2) | (1 << 6)))
            .expect("malformed BGEN fixture should be written");
        let original_selector = content_selector(&original_path);
        let alternate_selector = content_selector(&alternate_path);
        let wrong_digest_selector = BgenContentSelector {
            content_sha256: BgenContentSha256::from_bytes([29; 32]),
            expected_byte_count: alternate_selector.expected_byte_count,
        };
        let malformed_selector = content_selector(&malformed_path);
        let snapshot_cache = super::super::source::new_test_snapshot_cache();

        let retained_reader = open_selected_reader(&original_path, original_selector, snapshot_cache);
        let retained_payload =
            retained_reader.source.snapshot_payload_arc().expect("retained reader should own a parsed payload");

        let wrong_digest_error = BgenSource::open_request_with_test_snapshot_cache(
            BgenOpenRequest { locator: &alternate_path, content_selector: Some(wrong_digest_selector) },
            snapshot_cache,
        )
        .expect_err("wrong selected digest should fail");
        assert!(matches!(
            wrong_digest_error,
            BgenError::ContentSha256Mismatch { expected, observed }
                if expected == wrong_digest_selector.content_sha256 && observed == alternate_selector.content_sha256
        ));

        let malformed_source = BgenSource::open_request_with_test_snapshot_cache(
            BgenOpenRequest { locator: &malformed_path, content_selector: Some(malformed_selector) },
            snapshot_cache,
        )
        .expect("digest-matching malformed bytes should be captured");
        let malformed_error =
            BgenReaderCore::open_source(malformed_source).expect_err("malformed selected BGEN must not publish");
        assert!(malformed_error.to_string().contains("reserved flag bits 0x00000040"));

        let reopened_reader = open_selected_reader(&missing_path, original_selector, snapshot_cache);
        let reopened_payload =
            reopened_reader.source.snapshot_payload_arc().expect("retained selected content should reopen");
        assert!(Arc::ptr_eq(&retained_payload, &reopened_payload));

        let _ = fs::remove_file(original_path);
        let _ = fs::remove_file(alternate_path);
        let _ = fs::remove_file(malformed_path);
    }

    #[test]
    fn selected_replacement_preserves_live_readers_and_releases_after_drop() {
        let original_path = temporary_bgen_path("selected-replacement-original");
        let replacement_path = temporary_bgen_path("selected-replacement-new");
        let missing_path = temporary_bgen_path("selected-replacement-missing");
        write_single_variant_bgen(&original_path);
        write_single_variant_bgen_with_probabilities(&replacement_path, &[255, 0, 255, 0, 255, 0]);
        let original_selector = content_selector(&original_path);
        let replacement_selector = content_selector(&replacement_path);
        let snapshot_cache = super::super::source::new_test_snapshot_cache();

        let original_reader = open_selected_reader(&original_path, original_selector, snapshot_cache);
        let original_payload =
            original_reader.source.snapshot_payload_arc().expect("original reader should own its payload");
        let original_payload_weak = Arc::downgrade(&original_payload);
        let replacement_reader = open_selected_reader(&replacement_path, replacement_selector, snapshot_cache);
        let replacement_payload =
            replacement_reader.source.snapshot_payload_arc().expect("replacement reader should own its payload");

        assert!(!Arc::ptr_eq(&original_payload, &replacement_payload));
        assert_eq!(decoded_dosages(&original_reader), vec![2.0, 0.0, 1.0]);
        assert_eq!(decoded_dosages(&replacement_reader), vec![0.0, 0.0, 0.0]);
        let reopened_replacement = open_selected_reader(&missing_path, replacement_selector, snapshot_cache);
        let reopened_replacement_payload =
            reopened_replacement.source.snapshot_payload_arc().expect("replacement cache hit should own its payload");
        assert!(Arc::ptr_eq(&replacement_payload, &reopened_replacement_payload));

        drop(original_payload);
        drop(original_reader);
        assert!(
            original_payload_weak.upgrade().is_none(),
            "replaced content should release after its final live reader drops",
        );

        let _ = fs::remove_file(original_path);
        let _ = fs::remove_file(replacement_path);
    }

    #[test]
    fn concurrent_selected_misses_canonicalize_after_independent_parsing() {
        let path = temporary_bgen_path("selected-concurrent-canonicalization");
        write_single_variant_bgen(&path);
        let selector = content_selector(&path);
        let snapshot_cache = super::super::source::new_test_snapshot_cache();
        let capture_barrier = Arc::new(std::sync::Barrier::new(2));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let thread_path = path.clone();
            let thread_barrier = Arc::clone(&capture_barrier);
            handles.push(thread::spawn(move || {
                let capture_hook_guard = super::super::source::install_test_snapshot_capture_hook(move || {
                    thread_barrier.wait();
                });
                let reader = open_selected_reader(&thread_path, selector, snapshot_cache);
                drop(capture_hook_guard);
                reader
            }));
        }
        let first_reader = handles.remove(0).join().expect("first selected open thread should finish");
        let second_reader = handles.remove(0).join().expect("second selected open thread should finish");
        let first_payload = first_reader.source.snapshot_payload_arc().expect("first reader should own its payload");
        let second_payload = second_reader.source.snapshot_payload_arc().expect("second reader should own its payload");

        assert!(Arc::ptr_eq(&first_payload, &second_payload));
        assert_eq!(first_reader.source_provenance().resolution, BgenSnapshotResolution::CapturedFromLocator,);
        assert_eq!(second_reader.source_provenance().resolution, BgenSnapshotResolution::CapturedFromLocator,);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn selected_positioned_source_is_rejected_without_cache_publication() {
        let path = temporary_bgen_path("selected-positioned-rejection");
        write_positioned_single_variant_bgen(&path);
        let snapshot_cache = super::super::source::new_test_snapshot_cache();
        let selector =
            BgenContentSelector { content_sha256: BgenContentSha256::from_bytes([23; 32]), expected_byte_count: None };

        let error = BgenSource::open_request_with_test_snapshot_cache(
            BgenOpenRequest { locator: &path, content_selector: Some(selector) },
            snapshot_cache,
        )
        .expect_err("selected positioned source should be rejected");
        assert!(matches!(
            error,
            BgenError::ContentSelectionRequiresOwnedSnapshot {
                source_byte_count,
                maximum_snapshot_byte_count,
            } if source_byte_count > maximum_snapshot_byte_count
                && maximum_snapshot_byte_count == crate::bgen::source::MAXIMUM_OWNED_SNAPSHOT_BYTE_COUNT
        ));
        assert!(snapshot_cache_is_empty(snapshot_cache));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn selected_capture_uses_the_open_descriptor_after_locator_deletion_or_retargeting() {
        let deleted_path = temporary_bgen_path("selected-capture-deleted-locator");
        write_single_variant_bgen(&deleted_path);
        let deleted_selector = content_selector(&deleted_path);
        let deleted_cache = super::super::source::new_test_snapshot_cache();
        let deleted_hook_path = deleted_path.clone();
        let deletion_hook_guard = super::super::source::install_test_snapshot_capture_hook(move || {
            fs::remove_file(deleted_hook_path).expect("capture hook should delete the configured locator");
        });
        let deleted_reader = open_selected_reader(&deleted_path, deleted_selector, deleted_cache);
        drop(deletion_hook_guard);
        assert_eq!(decoded_dosages(&deleted_reader), vec![2.0, 0.0, 1.0]);
        assert_eq!(deleted_reader.source_provenance().requested_path, deleted_path);

        let retargeted_path = temporary_bgen_path("selected-capture-retargeted-locator");
        let replacement_path = temporary_bgen_path("selected-capture-retargeted-replacement");
        write_single_variant_bgen(&retargeted_path);
        write_single_variant_bgen_with_probabilities(&replacement_path, &[255, 0, 255, 0, 255, 0]);
        let retargeted_selector = content_selector(&retargeted_path);
        let retargeted_cache = super::super::source::new_test_snapshot_cache();
        let retargeted_hook_path = retargeted_path.clone();
        let replacement_hook_path = replacement_path.clone();
        let retarget_hook_guard = super::super::source::install_test_snapshot_capture_hook(move || {
            fs::rename(replacement_hook_path, retargeted_hook_path)
                .expect("capture hook should retarget the configured locator");
        });
        let retargeted_reader = open_selected_reader(&retargeted_path, retargeted_selector, retargeted_cache);
        drop(retarget_hook_guard);
        assert_eq!(decoded_dosages(&retargeted_reader), vec![2.0, 0.0, 1.0]);
        let current_reader = BgenReaderCore::open(&retargeted_path).expect("retargeted locator should remain readable");
        assert_eq!(decoded_dosages(&current_reader), vec![0.0, 0.0, 0.0]);

        let _ = fs::remove_file(retargeted_path);
    }

    #[test]
    fn reader_rejects_malformed_header_offsets_flags_and_variant_capacity() {
        let mut header_overlap = minimal_bgen_header_bytes(0, 0, 2 << 2);
        header_overlap[0..4].copy_from_slice(&19_u32.to_le_bytes());
        assert_invalid_bgen(&header_overlap, "header-overlap", "exceeds the first-variant offset");

        let mut offset_beyond_source = minimal_bgen_header_bytes(0, 0, 2 << 2);
        offset_beyond_source[0..4].copy_from_slice(&100_u32.to_le_bytes());
        assert_invalid_bgen(&offset_beyond_source, "offset-beyond-source", "exceeds the source length");

        let reserved_flags = minimal_bgen_header_bytes(0, 0, (2 << 2) | (1 << 6));
        assert_invalid_bgen(&reserved_flags, "reserved-flags", "reserved flag bits 0x00000040");

        let mut zero_variant_trailing_bytes = minimal_bgen_header_bytes(0, 0, 2 << 2);
        zero_variant_trailing_bytes.push(0);
        assert_invalid_bgen(
            &zero_variant_trailing_bytes,
            "zero-variant-offset",
            "zero variants must point its first-variant offset to the end",
        );

        let impossible_variant_count = minimal_bgen_header_bytes(u32::MAX, 0, 2 << 2);
        assert_invalid_bgen(
            &impossible_variant_count,
            "variant-capacity",
            "but only 0 bytes remain after the first-variant offset",
        );

        let oversized_uncompressed_length = u32::try_from(crate::bgen::source::MAXIMUM_SOURCE_WINDOW_BYTE_COUNT + 1)
            .expect("oversized test block length should fit u32");
        let mut oversized_uncompressed_block = minimal_bgen_header_bytes(1, 3, 1 | (2 << 2));
        oversized_uncompressed_block
            .extend_from_slice(&compressed_variant_payload_with_declared_length(oversized_uncompressed_length));
        assert_invalid_bgen(
            &oversized_uncompressed_block,
            "decompression-capacity",
            "uncompressed probability block contains 8388609 bytes",
        );
    }

    #[test]
    fn reader_rejects_malformed_embedded_sample_blocks() {
        let embedded_sample_flags = (2 << 2) | (1 << 31);

        let mut short_block = minimal_bgen_header_bytes(0, 0, embedded_sample_flags);
        short_block.extend_from_slice(&4_u32.to_le_bytes());
        let short_block_length = short_block.len();
        set_first_variant_offset(&mut short_block, short_block_length);
        assert_invalid_bgen(&short_block, "sample-short", "must be at least 8 bytes");

        let mut overlapping_block = minimal_bgen_header_bytes(0, 0, embedded_sample_flags);
        overlapping_block.extend_from_slice(&100_u32.to_le_bytes());
        overlapping_block.extend_from_slice(&0_u32.to_le_bytes());
        let overlapping_block_length = overlapping_block.len();
        set_first_variant_offset(&mut overlapping_block, overlapping_block_length);
        assert_invalid_bgen(&overlapping_block, "sample-overlap", "overlaps the first variant block");

        let mut mismatched_count = minimal_bgen_header_bytes(0, 1, embedded_sample_flags);
        mismatched_count.extend_from_slice(&8_u32.to_le_bytes());
        mismatched_count.extend_from_slice(&0_u32.to_le_bytes());
        let mismatched_count_length = mismatched_count.len();
        set_first_variant_offset(&mut mismatched_count, mismatched_count_length);
        assert_invalid_bgen(&mismatched_count, "sample-count", "reports 0 samples");

        let mut truncated_identifier = minimal_bgen_header_bytes(0, 1, embedded_sample_flags);
        truncated_identifier.extend_from_slice(&11_u32.to_le_bytes());
        truncated_identifier.extend_from_slice(&1_u32.to_le_bytes());
        truncated_identifier.extend_from_slice(&2_u16.to_le_bytes());
        truncated_identifier.push(b'a');
        let truncated_identifier_length = truncated_identifier.len();
        set_first_variant_offset(&mut truncated_identifier, truncated_identifier_length);
        assert_invalid_bgen(
            &truncated_identifier,
            "sample-identifier",
            "identifier 0 extends beyond its declared block length",
        );

        let mut trailing_block_byte = minimal_bgen_header_bytes(0, 0, embedded_sample_flags);
        trailing_block_byte.extend_from_slice(&9_u32.to_le_bytes());
        trailing_block_byte.extend_from_slice(&0_u32.to_le_bytes());
        trailing_block_byte.push(0);
        let trailing_block_length = trailing_block_byte.len();
        set_first_variant_offset(&mut trailing_block_byte, trailing_block_length);
        assert_invalid_bgen(&trailing_block_byte, "sample-trailing", "does not match the encoded sample identifiers");
    }

    #[test]
    fn reader_rejects_invalid_utf8_in_every_variant_metadata_field() {
        let probability_block = trusted_probability_block(&[0, 0, 255, 0, 0, 255]);
        let invalid_metadata_cases: [(&str, [&[u8]; 5], &str); 5] = [
            ("variant-identifier", [b"\xff", b"rs", b"22", b"A", b"G"], "variant identifier"),
            ("rsid", [b"var", b"\xff", b"22", b"A", b"G"], "rsid"),
            ("chromosome", [b"var", b"rs", b"\xff", b"A", b"G"], "chromosome"),
            ("reference-allele", [b"var", b"rs", b"22", b"\xff", b"G"], "reference allele"),
            ("counted-allele", [b"var", b"rs", b"22", b"A", b"\xff"], "counted allele"),
        ];
        for (label, fields, expected_field) in invalid_metadata_cases {
            let [variant_identifier, rsid, chromosome, reference_allele, counted_allele] = fields;
            let payload = variant_payload_with_metadata(
                variant_identifier,
                rsid,
                chromosome,
                reference_allele,
                counted_allele,
                &probability_block,
            );
            let mut bytes = minimal_bgen_header_bytes(1, 3, 2 << 2);
            bytes.extend_from_slice(&payload);
            assert_invalid_bgen(&bytes, label, &format!("{expected_field} contains invalid UTF-8"));
        }
    }

    #[test]
    fn owned_snapshot_remains_isolated_after_source_truncation() {
        let path = temporary_bgen_path("snapshot-truncation");
        write_single_variant_bgen(&path);
        let reader = BgenReaderCore::open(&path).expect("snapshot BGEN reader should open");
        let session = reader.read_session(&[0, 1, 2]).expect("snapshot read session should build");
        OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("snapshot test source should reopen")
            .set_len(0)
            .expect("snapshot test source should truncate");

        let batch = session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect("in-place mutation cannot change captured snapshot bytes");
        let GenotypeBatchPayload::Decoded { genotypes, .. } = batch.payload else {
            panic!("snapshot dosage decode should return decoded values");
        };
        let OwnedGenotypeBuffer::Dosage(values) = genotypes else {
            panic!("snapshot dosage decode should return f32 values");
        };
        assert_eq!(values, vec![2.0, 0.0, 1.0]);

        session.finish().expect("immutable snapshot sessions need no terminal source restat");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn owned_snapshot_remains_isolated_after_path_replacement() {
        let path = temporary_bgen_path("snapshot-path-replacement");
        let replacement_path = temporary_bgen_path("snapshot-path-replacement-new");
        write_single_variant_bgen(&path);
        let reader = BgenReaderCore::open(&path).expect("snapshot BGEN reader should open");
        let session = reader.read_session(&[0, 1, 2]).expect("snapshot read session should build");

        write_single_variant_bgen_with_probabilities(&replacement_path, &[255, 0, 255, 0, 255, 0]);
        fs::rename(&replacement_path, &path).expect("replacement BGEN should atomically replace the configured path");

        let batch = session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect("path replacement cannot change bytes in an owned snapshot");
        let GenotypeBatchPayload::Decoded { genotypes, .. } = batch.payload else {
            panic!("snapshot dosage decode should return decoded values");
        };
        let OwnedGenotypeBuffer::Dosage(values) = genotypes else {
            panic!("snapshot dosage decode should return f32 values");
        };
        assert_eq!(values, vec![2.0, 0.0, 1.0]);

        session.finish().expect("path replacement cannot affect an immutable snapshot session");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn positioned_source_decodes_and_reports_typed_eof_after_truncation() {
        let path = temporary_bgen_path("positioned-source");
        write_positioned_single_variant_bgen(&path);
        let reader = BgenReaderCore::open(&path).expect("positioned BGEN reader should open");
        assert!(reader.source.snapshot_bytes().is_none(), "large BGEN inputs should use positioned reads");
        let session = reader.read_session(&[0, 2]).expect("positioned read session should build");
        let batch = session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect("positioned BGEN batch should decode");
        let GenotypeBatchPayload::Decoded { genotypes, .. } = batch.payload else {
            panic!("positioned dosage decode should return decoded values");
        };
        let OwnedGenotypeBuffer::Dosage(values) = genotypes else {
            panic!("positioned dosage decode should return f32 values");
        };
        assert_eq!(values, vec![2.0, 1.0]);

        OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("positioned test source should reopen")
            .set_len(0)
            .expect("positioned test source should truncate");
        let error = session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect_err("truncated positioned source should fail");
        match error {
            BgenError::Io(source) => {
                assert_eq!(source.kind(), std::io::ErrorKind::UnexpectedEof);
                let message = source.to_string();
                assert!(message.starts_with("Unexpected end of file while reading BGEN bytes:"));
                assert!(message.contains("positioned read at offset"));
                assert!(message.contains("requested"));
                assert!(message.ends_with("observed 0."));
            }
            other => panic!("expected a positioned-read EOF, observed {other:?}"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn positioned_batch_rejects_same_length_configured_path_replacement_after_read() {
        let path = temporary_bgen_path("positioned-post-read-replacement");
        let replacement_path = temporary_bgen_path("positioned-post-read-replacement-new");
        write_positioned_single_variant_bgen(&path);
        write_positioned_single_variant_bgen_with_probabilities(&replacement_path, &[255, 0, 255, 0, 255, 0]);
        assert_eq!(
            fs::metadata(&path).expect("original positioned metadata should resolve").len(),
            fs::metadata(&replacement_path).expect("replacement positioned metadata should resolve").len(),
        );
        let reader = BgenReaderCore::open(&path).expect("positioned BGEN reader should open");
        let session = reader.read_session(&[0, 1, 2]).expect("positioned read session should build");
        fs::rename(&replacement_path, &path)
            .expect("same-length positioned replacement should atomically replace the configured path");

        let error = session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect_err("post-read identity validation must reject configured-path replacement");
        match error {
            BgenError::InvalidFormat(message) => {
                assert_eq!(message, "BGEN source changed while a genotype batch was being read.");
            }
            other => panic!("expected a post-read source-identity error, observed {other:?}"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn positioned_batch_rejects_concurrent_same_length_mutation_after_read() {
        let path = temporary_bgen_path("positioned-post-read-in-place-mutation");
        write_positioned_single_variant_bgen(&path);
        let source_length = fs::metadata(&path).expect("positioned source metadata should resolve").len();
        let reader = BgenReaderCore::open(&path).expect("positioned BGEN reader should open");
        assert!(reader.source.snapshot_bytes().is_none(), "positioned mutation test must bypass the snapshot cache");
        let session = reader.read_session(&[0, 1, 2]).expect("positioned read session should build");
        let (post_read_sender, post_read_receiver) = mpsc::sync_channel(1);
        let (mutation_complete_sender, mutation_complete_receiver) = mpsc::sync_channel(1);
        let mutation_path = path.clone();
        let mutation_handle = thread::spawn(move || {
            post_read_receiver
                .recv_timeout(TEST_COORDINATION_TIMEOUT)
                .expect("positioned decode should reach post-read validation");
            let mut source_file =
                OpenOptions::new().write(true).open(&mutation_path).expect("positioned source should reopen");
            source_file.seek(SeekFrom::End(-1)).expect("final probability byte should be seekable");
            source_file.write_all(&[0_u8]).expect("final probability byte should be overwritten");
            source_file
                .set_times(FileTimes::new().set_modified(UNIX_EPOCH))
                .expect("in-place mutation should force a distinct modification timestamp");
            source_file.sync_all().expect("in-place positioned mutation should reach the filesystem");
            let mutated_source_length =
                fs::metadata(&mutation_path).expect("mutated positioned metadata should resolve").len();
            mutation_complete_sender.send(()).expect("positioned mutation completion should be observed");
            mutated_source_length
        });
        let validation_hook_guard = install_test_delivery_source_validation_hook(move || {
            post_read_sender.send(()).expect("post-read hook should release the mutation thread");
            mutation_complete_receiver
                .recv_timeout(TEST_COORDINATION_TIMEOUT)
                .expect("positioned mutation should complete before identity validation");
        });

        let decode_result = session.decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY);
        drop(validation_hook_guard);
        let mutated_source_length = mutation_handle.join().expect("positioned mutation thread should finish");
        let error = decode_result.expect_err("post-read identity validation must reject opened-descriptor mutation");
        assert_eq!(mutated_source_length, source_length);
        assert!(
            !reader.source.is_open_file_unchanged().expect("opened positioned descriptor should remain statable"),
            "same-length in-place mutation must change opened-descriptor identity metadata",
        );
        match error {
            BgenError::InvalidFormat(message) => {
                assert_eq!(message, "BGEN source changed while a genotype batch was being read.");
            }
            other => panic!("expected a post-read opened-descriptor identity error, observed {other:?}"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn positioned_session_finish_rejects_concurrent_replacement_after_final_batch() {
        let path = temporary_bgen_path("positioned-finish-replacement");
        let replacement_path = temporary_bgen_path("positioned-finish-replacement-new");
        write_positioned_single_variant_bgen(&path);
        write_positioned_single_variant_bgen_with_probabilities(&replacement_path, &[255, 0, 255, 0, 255, 0]);
        assert_eq!(
            fs::metadata(&path).expect("original positioned metadata should resolve").len(),
            fs::metadata(&replacement_path).expect("replacement positioned metadata should resolve").len(),
        );
        let reader = BgenReaderCore::open(&path).expect("positioned BGEN reader should open");
        assert!(reader.source.snapshot_bytes().is_none(), "positioned finish test must bypass the snapshot cache");
        let session = reader.read_session(&[0, 1, 2]).expect("positioned read session should build");
        session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect("final positioned batch should succeed before replacement");
        let (finish_sender, finish_receiver) = mpsc::sync_channel(1);
        let (replacement_complete_sender, replacement_complete_receiver) = mpsc::sync_channel(1);
        let replaced_path = path.clone();
        let replacement_source_path = replacement_path.clone();
        let replacement_handle = thread::spawn(move || {
            finish_receiver
                .recv_timeout(TEST_COORDINATION_TIMEOUT)
                .expect("positioned session should reach finish validation");
            fs::rename(&replacement_source_path, &replaced_path)
                .expect("same-length positioned replacement should atomically replace the configured path");
            replacement_complete_sender.send(()).expect("positioned replacement completion should be observed");
        });
        let validation_hook_guard = install_test_delivery_source_validation_hook(move || {
            finish_sender.send(()).expect("finish hook should release the replacement thread");
            replacement_complete_receiver
                .recv_timeout(TEST_COORDINATION_TIMEOUT)
                .expect("positioned replacement should complete before finish validation");
        });

        let finish_result = session.finish();
        drop(validation_hook_guard);
        replacement_handle.join().expect("positioned replacement thread should finish");
        let error = finish_result.expect_err("session finish must reject configured-path replacement");
        match error {
            BgenError::InvalidFormat(message) => {
                assert_eq!(message, "BGEN source changed while genotype delivery was in progress.");
            }
            other => panic!("expected a session-finish source-identity error, observed {other:?}"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn positioned_source_coalesces_io_across_parallel_decode_tiles() {
        let path = temporary_bgen_path("positioned-coarse-window");
        write_positioned_variant_count_bgen(&path, 65);
        let reader = BgenReaderCore::open(&path).expect("positioned multi-tile BGEN reader should open");
        assert!(reader.source.snapshot_bytes().is_none(), "large BGEN inputs should use positioned reads");
        assert_eq!(
            reader.packed8_compatibility_with_cache().expect("positioned compatibility scan should succeed"),
            crate::common::Packed8Compatibility::Compatible,
        );
        let session = reader.read_session(&[0, 2]).expect("positioned multi-tile session should build");
        let batch = session
            .decode_variant_major_batch(0, 65, 65, true, COMPLETE_STATISTICS_POLICY)
            .expect("one positioned source window should feed multiple parallel decode tiles");
        let GenotypeBatchPayload::Decoded { genotypes, statistics } = batch.payload else {
            panic!("positioned packed8 decode should return decoded values");
        };
        let OwnedGenotypeBuffer::Packed8(values) = genotypes else {
            panic!("positioned packed8 decode should return probability pairs");
        };
        assert_eq!(values.len(), 65 * 2 * 2);
        assert!(values.chunks_exact(4).all(|row| row == [0, 0, 0, 255]));
        assert_eq!(statistics.output.observation_count, vec![2; 65]);
        drop(values);
        let retained_source_window = session
            .positioned_source_window_pool
            .take_matching(|buffer| !buffer.is_empty())
            .expect("positioned decode should retain its initialized source window");
        let retained_source_window_pointer = retained_source_window.as_ptr();
        session.positioned_source_window_pool.release(retained_source_window);

        let second_batch = session
            .decode_variant_major_batch(0, 65, 65, true, COMPLETE_STATISTICS_POLICY)
            .expect("a later batch should reuse the session source window");
        drop(second_batch);
        let reused_source_window = session
            .positioned_source_window_pool
            .take_matching(|buffer| !buffer.is_empty())
            .expect("later positioned decode should return its source window");
        assert_eq!(reused_source_window.as_ptr(), retained_source_window_pointer);
        session.positioned_source_window_pool.release(reused_source_window);
        session.finish().expect("positioned multi-tile source should remain stable");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn positioned_later_window_errors_report_absolute_file_variant_index() {
        let path = temporary_bgen_path("positioned-later-window-error");
        let valid_payload = variant_payload(&trusted_probability_block(&[0, 0, 255, 0, 0, 255]));
        let corrupt_later_payload = large_metadata_variant_payload(&[255, 1, 255, 0, 0, 255]);
        write_positioned_variant_payloads_bgen(&path, &[valid_payload.clone(), valid_payload, corrupt_later_payload]);
        let reader = BgenReaderCore::open(&path).expect("positioned multi-window BGEN reader should open");
        let session = reader.read_session(&[0, 2]).expect("positioned multi-window session should build");

        let error = session
            .decode_variant_major_batch(1, 3, 2, false, COMPLETE_STATISTICS_POLICY)
            .expect_err("a corrupt probability pair in the later source window should fail");
        assert!(
            error.to_string().contains("Variant index 2 ('rs-large')"),
            "later-window error should report absolute file variant index 2, observed {error}",
        );
        assert!(error.to_string().contains("sum above 255"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn positioned_sparse_output_and_statistics_match_snapshot_across_window_boundary() {
        let snapshot_path = temporary_bgen_path("snapshot-window-reference");
        let positioned_path = temporary_bgen_path("positioned-window-candidate");
        let payloads = vec![
            variant_payload(&trusted_probability_block(&[0, 0, 255, 0, 0, 255])),
            large_metadata_variant_payload(&[255, 0, 0, 255, 255, 0]),
            variant_payload(&trusted_probability_block(&[0, 255, 0, 0, 255, 0])),
        ];
        write_variant_payloads_bgen(&snapshot_path, &payloads);
        write_positioned_variant_payloads_bgen(&positioned_path, &payloads);
        let snapshot_reader = BgenReaderCore::open(&snapshot_path).expect("snapshot reference BGEN should open");
        let positioned_reader = BgenReaderCore::open(&positioned_path).expect("positioned candidate BGEN should open");
        assert!(snapshot_reader.source.snapshot_bytes().is_some());
        assert!(positioned_reader.source.snapshot_bytes().is_none());
        let snapshot_session =
            snapshot_reader.read_session(&[0, 2]).expect("snapshot sparse-selection session should build");
        let positioned_session =
            positioned_reader.read_session(&[0, 2]).expect("positioned sparse-selection session should build");

        let snapshot_batch = snapshot_session
            .decode_variant_major_batch(0, 3, 3, false, COMPLETE_STATISTICS_POLICY)
            .expect("snapshot reference batch should decode");
        let positioned_batch = positioned_session
            .decode_variant_major_batch(0, 3, 3, false, COMPLETE_STATISTICS_POLICY)
            .expect("positioned batch should decode across source windows");
        match (snapshot_batch.payload, positioned_batch.payload) {
            (
                GenotypeBatchPayload::Decoded { genotypes: snapshot_genotypes, statistics: snapshot_statistics },
                GenotypeBatchPayload::Decoded { genotypes: positioned_genotypes, statistics: positioned_statistics },
            ) => {
                assert_eq!(positioned_genotypes, snapshot_genotypes);
                assert_eq!(positioned_statistics, snapshot_statistics);
                assert_eq!(positioned_statistics.compute.sparse_candidate_mask, Some(vec![true, true, true]));
            }
            payloads => panic!("both window-boundary batches should return decoded payloads, observed {payloads:?}"),
        }
        snapshot_session.finish().expect("snapshot source should remain stable");
        positioned_session.finish().expect("positioned source should remain stable");

        let _ = fs::remove_file(snapshot_path);
        let _ = fs::remove_file(positioned_path);
    }

    #[test]
    fn oversized_positioned_payload_returns_bounded_error_without_publishing_batch() {
        let path = temporary_bgen_path("positioned-oversized-payload");
        write_positioned_single_variant_bgen(&path);
        let mut reader = BgenReaderCore::open(&path).expect("positioned BGEN reader should open");
        reader
            .positioned_index
            .as_mut()
            .expect("an oversized positioned source should own its index")
            .variant_records[0]
            .probability_payload_length = u32::try_from(crate::bgen::source::MAXIMUM_SOURCE_WINDOW_BYTE_COUNT + 1)
            .expect("oversized payload length should fit u32");
        let session = reader.read_session(&[0, 1, 2]).expect("positioned read session should build");

        let error = session
            .decode_variant_major_batch(0, 1, 1, false, COMPLETE_STATISTICS_POLICY)
            .expect_err("an oversized positioned payload must not publish a partial genotype batch");
        match error {
            BgenError::Range(message) => {
                assert!(message.contains("Variant index 0 ('rs')"));
                assert!(message.contains("BGEN source windows cannot exceed 8388608 bytes"));
                assert!(message.contains("Requested 8388609 bytes"));
            }
            other => panic!("expected a typed bounded-window range error, observed {other:?}"),
        }

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
