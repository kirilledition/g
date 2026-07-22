use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

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
use super::{index, packed8};

mod variant_major;

#[derive(Debug)]
pub struct BgenReaderCore {
    pub(super) mmap: Mmap,
    source: super::packed8_cache::ValidationCacheSource,
    sample_count: usize,
    variant_count: usize,
    pub(super) compression_type: CompressionType,
    packed8_validation_complete: AtomicBool,
    pub(super) variant_records: Vec<VariantRecord>,
    variant_metadata: Arc<VariantMetadataStore>,
    chromosome_boundary_indices: Vec<usize>,
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
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened or mapped, its header or
    /// variant index is invalid, its layout is unsupported, or it changes while
    /// being indexed.
    pub fn open(bgen_path: &Path) -> Result<Self, BgenError> {
        let source = super::packed8_cache::ValidationCacheSource::open(bgen_path)?;
        let mmap = unsafe { MmapOptions::new().map(&source.file)? };

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

        let parsed_variant_index =
            index::parse_variant_index(&mmap, first_variant_offset, variant_count, sample_count, compression_type)?;
        if !source.is_unchanged()? {
            return Err(BgenError::InvalidFormat(
                "BGEN source changed while its header and variant index were being read.".to_string(),
            ));
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
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::common::{ChunkStatisticsPolicy, GenotypeBatchPayload, OwnedGenotypeBuffer};

    const COMPLETE_STATISTICS_POLICY: ChunkStatisticsPolicy =
        ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: true, collect_sparse_candidate_mask: true };

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
