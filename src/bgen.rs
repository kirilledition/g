use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use flate2::{Decompress, FlushDecompress, Status};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use thiserror::Error;

const MISSING_SAMPLE_FLAG_MASK: u8 = 0x80;
const PLOIDY_MASK: u8 = 0x3F;
const VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES: usize = 2;
const ALLELE_LENGTH_SIZE_IN_BYTES: usize = 4;
const DEFAULT_DECODE_TILE_VARIANT_COUNT: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompressionType {
    None,
    Zlib,
}

impl TryFrom<u32> for CompressionType {
    type Error = BgenError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Zlib),
            unsupported_value => Err(BgenError::UnsupportedFormat(format!(
                "Unsupported BGEN compression flag {unsupported_value}. Only uncompressed and zlib-compressed blocks are supported.",
            ))),
        }
    }
}

#[derive(Debug)]
pub struct BgenReaderCore {
    bgen_path: PathBuf,
    mmap: Mmap,
    sample_count: usize,
    variant_count: usize,
    contains_embedded_samples: bool,
    sample_identifiers: Vec<String>,
    compression_type: CompressionType,
    trusted_no_missing_diploid: bool,
    variant_records: Vec<VariantRecord>,
    chromosome_boundary_indices: Vec<usize>,
    prepared_sample_selection: Mutex<Option<Arc<SampleSelection>>>,
    profiling: ReaderProfiling,
}

#[derive(Debug)]
struct VariantRecord {
    probability_payload_offset: usize,
    probability_payload_length: usize,
    declared_uncompressed_block_length: usize,
    chromosome: String,
    resolved_variant_identifier: String,
    position: i64,
    counted_allele: String,
    reference_allele: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReaderProfileSnapshot {
    pub sample_selection_prepare_ns: u64,
    pub sample_selection_prepare_count: u64,
    pub compressed_block_fetch_ns: u64,
    pub compressed_block_fetch_count: u64,
    pub compressed_byte_count: u64,
    pub decompression_ns: u64,
    pub decompression_count: u64,
    pub uncompressed_byte_count: u64,
    pub zlib_stream_count: u64,
    pub probability_decode_ns: u64,
    pub probability_decode_count: u64,
    pub variant_decode_count: u64,
    pub output_write_ns: u64,
    pub output_write_count: u64,
    pub output_byte_count: u64,
    pub decode_tile_count: u64,
    pub selected_sample_count: u64,
    pub metadata_slice_ns: u64,
    pub metadata_slice_count: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ThreadLocalProfileSnapshot {
    compressed_block_fetch_ns: u64,
    compressed_block_fetch_count: u64,
    compressed_byte_count: u64,
    decompression_ns: u64,
    decompression_count: u64,
    uncompressed_byte_count: u64,
    zlib_stream_count: u64,
    probability_decode_ns: u64,
    probability_decode_count: u64,
    variant_decode_count: u64,
    output_write_ns: u64,
    output_write_count: u64,
    output_byte_count: u64,
    decode_tile_count: u64,
    selected_sample_count: u64,
}

#[derive(Debug, Default)]
struct ReaderProfiling {
    enabled: AtomicBool,
    sample_selection_prepare_ns: AtomicU64,
    sample_selection_prepare_count: AtomicU64,
    compressed_block_fetch_ns: AtomicU64,
    compressed_block_fetch_count: AtomicU64,
    compressed_byte_count: AtomicU64,
    decompression_ns: AtomicU64,
    decompression_count: AtomicU64,
    uncompressed_byte_count: AtomicU64,
    zlib_stream_count: AtomicU64,
    probability_decode_ns: AtomicU64,
    probability_decode_count: AtomicU64,
    variant_decode_count: AtomicU64,
    output_write_ns: AtomicU64,
    output_write_count: AtomicU64,
    output_byte_count: AtomicU64,
    decode_tile_count: AtomicU64,
    selected_sample_count: AtomicU64,
    metadata_slice_ns: AtomicU64,
    metadata_slice_count: AtomicU64,
}

#[derive(Error, Debug)]
pub enum BgenError {
    #[error("{0}")]
    InvalidFormat(String),
    #[error("{0}")]
    UnsupportedFormat(String),
    #[error("{0}")]
    Range(String),
    #[error("I/O error while reading BGEN file: {0}")]
    Io(#[from] std::io::Error),
}

pub type VariantMetadataLists = (Vec<String>, Vec<String>, Vec<i64>, Vec<String>, Vec<String>);

impl ReaderProfiling {
    fn reset(&self) {
        self.enabled.store(true, Ordering::Relaxed);
        self.sample_selection_prepare_ns.store(0, Ordering::Relaxed);
        self.sample_selection_prepare_count.store(0, Ordering::Relaxed);
        self.compressed_block_fetch_ns.store(0, Ordering::Relaxed);
        self.compressed_block_fetch_count.store(0, Ordering::Relaxed);
        self.compressed_byte_count.store(0, Ordering::Relaxed);
        self.decompression_ns.store(0, Ordering::Relaxed);
        self.decompression_count.store(0, Ordering::Relaxed);
        self.uncompressed_byte_count.store(0, Ordering::Relaxed);
        self.zlib_stream_count.store(0, Ordering::Relaxed);
        self.probability_decode_ns.store(0, Ordering::Relaxed);
        self.probability_decode_count.store(0, Ordering::Relaxed);
        self.variant_decode_count.store(0, Ordering::Relaxed);
        self.output_write_ns.store(0, Ordering::Relaxed);
        self.output_write_count.store(0, Ordering::Relaxed);
        self.output_byte_count.store(0, Ordering::Relaxed);
        self.decode_tile_count.store(0, Ordering::Relaxed);
        self.selected_sample_count.store(0, Ordering::Relaxed);
        self.metadata_slice_ns.store(0, Ordering::Relaxed);
        self.metadata_slice_count.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> ReaderProfileSnapshot {
        ReaderProfileSnapshot {
            sample_selection_prepare_ns: self.sample_selection_prepare_ns.load(Ordering::Relaxed),
            sample_selection_prepare_count: self.sample_selection_prepare_count.load(Ordering::Relaxed),
            compressed_block_fetch_ns: self.compressed_block_fetch_ns.load(Ordering::Relaxed),
            compressed_block_fetch_count: self.compressed_block_fetch_count.load(Ordering::Relaxed),
            compressed_byte_count: self.compressed_byte_count.load(Ordering::Relaxed),
            decompression_ns: self.decompression_ns.load(Ordering::Relaxed),
            decompression_count: self.decompression_count.load(Ordering::Relaxed),
            uncompressed_byte_count: self.uncompressed_byte_count.load(Ordering::Relaxed),
            zlib_stream_count: self.zlib_stream_count.load(Ordering::Relaxed),
            probability_decode_ns: self.probability_decode_ns.load(Ordering::Relaxed),
            probability_decode_count: self.probability_decode_count.load(Ordering::Relaxed),
            variant_decode_count: self.variant_decode_count.load(Ordering::Relaxed),
            output_write_ns: self.output_write_ns.load(Ordering::Relaxed),
            output_write_count: self.output_write_count.load(Ordering::Relaxed),
            output_byte_count: self.output_byte_count.load(Ordering::Relaxed),
            decode_tile_count: self.decode_tile_count.load(Ordering::Relaxed),
            selected_sample_count: self.selected_sample_count.load(Ordering::Relaxed),
            metadata_slice_ns: self.metadata_slice_ns.load(Ordering::Relaxed),
            metadata_slice_count: self.metadata_slice_count.load(Ordering::Relaxed),
        }
    }

    fn merge_thread_local_snapshot(&self, thread_local_snapshot: &ThreadLocalProfileSnapshot) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.compressed_block_fetch_ns.fetch_add(thread_local_snapshot.compressed_block_fetch_ns, Ordering::Relaxed);
        self.compressed_block_fetch_count
            .fetch_add(thread_local_snapshot.compressed_block_fetch_count, Ordering::Relaxed);
        self.compressed_byte_count.fetch_add(thread_local_snapshot.compressed_byte_count, Ordering::Relaxed);
        self.decompression_ns.fetch_add(thread_local_snapshot.decompression_ns, Ordering::Relaxed);
        self.decompression_count.fetch_add(thread_local_snapshot.decompression_count, Ordering::Relaxed);
        self.uncompressed_byte_count.fetch_add(thread_local_snapshot.uncompressed_byte_count, Ordering::Relaxed);
        self.zlib_stream_count.fetch_add(thread_local_snapshot.zlib_stream_count, Ordering::Relaxed);
        self.probability_decode_ns.fetch_add(thread_local_snapshot.probability_decode_ns, Ordering::Relaxed);
        self.probability_decode_count.fetch_add(thread_local_snapshot.probability_decode_count, Ordering::Relaxed);
        self.variant_decode_count.fetch_add(thread_local_snapshot.variant_decode_count, Ordering::Relaxed);
        self.output_write_ns.fetch_add(thread_local_snapshot.output_write_ns, Ordering::Relaxed);
        self.output_write_count.fetch_add(thread_local_snapshot.output_write_count, Ordering::Relaxed);
        self.output_byte_count.fetch_add(thread_local_snapshot.output_byte_count, Ordering::Relaxed);
        self.decode_tile_count.fetch_add(thread_local_snapshot.decode_tile_count, Ordering::Relaxed);
        self.selected_sample_count.fetch_add(thread_local_snapshot.selected_sample_count, Ordering::Relaxed);
    }

    fn record_sample_selection_prepare(&self, duration_nanoseconds: u64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.sample_selection_prepare_ns.fetch_add(duration_nanoseconds, Ordering::Relaxed);
        self.sample_selection_prepare_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_metadata_slice(&self, duration_nanoseconds: u64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.metadata_slice_ns.fetch_add(duration_nanoseconds, Ordering::Relaxed);
        self.metadata_slice_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_selected_sample_count(&self, selected_sample_count: usize) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.selected_sample_count
            .fetch_add(u64::try_from(selected_sample_count).unwrap_or(u64::MAX), Ordering::Relaxed);
    }
}

fn elapsed_nanoseconds(start_time: Instant) -> u64 {
    u64::try_from(start_time.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn decode_tile_variant_count() -> usize {
    static DECODE_TILE_VARIANT_COUNT: OnceLock<usize> = OnceLock::new();
    *DECODE_TILE_VARIANT_COUNT.get_or_init(|| {
        std::env::var("G_BGEN_DECODE_TILE_VARIANT_COUNT")
            .ok()
            .and_then(|raw_value| raw_value.parse::<usize>().ok())
            .filter(|tile_variant_count| *tile_variant_count > 0)
            .unwrap_or(DEFAULT_DECODE_TILE_VARIANT_COUNT)
    })
}

fn unphased_eight_bit_dosage_lookup() -> &'static [f32] {
    static UNPHASED_EIGHT_BIT_DOSAGE_LOOKUP: OnceLock<Vec<f32>> = OnceLock::new();
    UNPHASED_EIGHT_BIT_DOSAGE_LOOKUP.get_or_init(|| {
        let reciprocal_scale = 1.0_f32 / 255.0_f32;
        let mut dosage_lookup = Vec::with_capacity(usize::from(u16::MAX) + 1);
        for packed_probability_index in 0..=u16::MAX {
            let homozygous_reference_probability =
                f32::from((packed_probability_index & 0x00FF) as u8) * reciprocal_scale;
            let heterozygous_probability =
                f32::from(((packed_probability_index & 0xFF00) >> 8) as u8) * reciprocal_scale;
            dosage_lookup.push(2.0_f32 - ((2.0_f32 * homozygous_reference_probability) + heterozygous_probability));
        }
        dosage_lookup
    })
}

fn all_samples_present_diploid(sample_ploidy_and_missingness: &[u8]) -> bool {
    const PRESENT_DIPLOID_BYTE_GROUP: [u8; 16] = [2_u8; 16];
    let mut ploidy_chunks = sample_ploidy_and_missingness.chunks_exact(PRESENT_DIPLOID_BYTE_GROUP.len());
    for ploidy_chunk in &mut ploidy_chunks {
        if ploidy_chunk != PRESENT_DIPLOID_BYTE_GROUP {
            return false;
        }
    }
    ploidy_chunks.remainder().iter().all(|ploidy_byte| *ploidy_byte == 2)
}

fn read_packed_probability_index_eight_bit(packed_probability_bytes: &[u8], probability_offset: usize) -> usize {
    usize::from(packed_probability_bytes[probability_offset])
        | (usize::from(packed_probability_bytes[probability_offset + 1]) << 8)
}

#[allow(clippy::missing_errors_doc)]
impl BgenReaderCore {
    pub fn open(bgen_path: &Path, trusted_no_missing_diploid: bool) -> Result<Self, BgenError> {
        let file = File::open(bgen_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

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
        let sample_identifiers = if contains_embedded_samples {
            parse_sample_identifier_block(&mmap, sample_block_offset, first_variant_offset, sample_count)?
        } else {
            Vec::new()
        };

        let variant_records =
            parse_variant_records(&mmap, first_variant_offset, variant_count, sample_count, compression_type)?;
        let chromosome_boundary_indices = build_chromosome_boundary_indices(&variant_records);

        Ok(Self {
            bgen_path: bgen_path.to_path_buf(),
            mmap,
            sample_count,
            variant_count,
            contains_embedded_samples,
            sample_identifiers,
            compression_type,
            trusted_no_missing_diploid,
            variant_records,
            chromosome_boundary_indices,
            prepared_sample_selection: Mutex::new(None),
            profiling: ReaderProfiling::default(),
        })
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn variant_count(&self) -> usize {
        self.variant_count
    }

    pub fn contains_embedded_samples(&self) -> bool {
        self.contains_embedded_samples
    }

    pub fn sample_identifiers(&self) -> Vec<String> {
        self.sample_identifiers.clone()
    }

    pub fn chromosome_boundary_indices(&self) -> Vec<usize> {
        self.chromosome_boundary_indices.clone()
    }

    pub fn prepare_sample_selection(&self, sample_indices: &[i64]) -> Result<(), BgenError> {
        let sample_selection_start_time = Instant::now();
        let sample_selection = Arc::new(build_sample_selection(self.sample_count, sample_indices)?);
        self.profiling.record_sample_selection_prepare(elapsed_nanoseconds(sample_selection_start_time));
        let mut prepared_sample_selection = self
            .prepared_sample_selection
            .lock()
            .map_err(|_| BgenError::InvalidFormat("Prepared BGEN sample selection mutex was poisoned.".to_string()))?;
        *prepared_sample_selection = Some(sample_selection);
        Ok(())
    }

    pub fn clear_prepared_sample_selection(&self) -> Result<(), BgenError> {
        let mut prepared_sample_selection = self
            .prepared_sample_selection
            .lock()
            .map_err(|_| BgenError::InvalidFormat("Prepared BGEN sample selection mutex was poisoned.".to_string()))?;
        *prepared_sample_selection = None;
        Ok(())
    }

    pub fn reset_profile(&self) {
        self.profiling.reset();
    }

    pub fn profile_snapshot(&self) -> ReaderProfileSnapshot {
        self.profiling.snapshot()
    }

    pub fn validate_trusted_no_missing_diploid(&self) -> Result<(), BgenError> {
        let mut thread_scratch = ThreadScratch::default();
        let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
        for variant_record in &self.variant_records {
            validate_variant_compatible_with_trusted_no_missing_diploid(
                &self.mmap,
                self.compression_type,
                variant_record,
                self.sample_count,
                &mut thread_scratch,
                &mut thread_local_profile_snapshot,
            )?;
        }
        Ok(())
    }

    pub fn variant_metadata_slice(
        &self,
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<VariantMetadataLists, BgenError> {
        let metadata_slice_start_time = Instant::now();
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;

        let selected_variant_records = &self.variant_records[variant_start..variant_stop];
        let chromosome_values =
            selected_variant_records.iter().map(|variant_record| variant_record.chromosome.clone()).collect();
        let variant_identifier_values = selected_variant_records
            .iter()
            .map(|variant_record| variant_record.resolved_variant_identifier.clone())
            .collect();
        let position_values = selected_variant_records.iter().map(|variant_record| variant_record.position).collect();
        let allele_one_values =
            selected_variant_records.iter().map(|variant_record| variant_record.counted_allele.clone()).collect();
        let allele_two_values =
            selected_variant_records.iter().map(|variant_record| variant_record.reference_allele.clone()).collect();

        let variant_metadata_lists =
            (chromosome_values, variant_identifier_values, position_values, allele_one_values, allele_two_values);
        self.profiling.record_metadata_slice(elapsed_nanoseconds(metadata_slice_start_time));
        Ok(variant_metadata_lists)
    }

    pub fn read_dosage_f32(
        &self,
        sample_indices: &[i64],
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<Vec<f32>, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let sample_selection_start_time = Instant::now();
        let sample_selection = build_sample_selection(self.sample_count, sample_indices)?;
        self.profiling.record_sample_selection_prepare(elapsed_nanoseconds(sample_selection_start_time));
        let selected_sample_count = sample_selection.selected_sample_count;
        let selected_variant_count = variant_stop - variant_start;
        let mut row_major_dosage_values = vec![0.0_f32; selected_sample_count * selected_variant_count];
        self.read_dosage_f32_into_address_with_selection(
            &sample_selection,
            variant_start,
            variant_stop,
            row_major_dosage_values.as_mut_ptr() as usize,
            row_major_dosage_values.len(),
        )?;
        Ok(row_major_dosage_values)
    }

    pub fn read_dosage_f32_prepared(&self, variant_start: usize, variant_stop: usize) -> Result<Vec<f32>, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let sample_selection = self.prepared_sample_selection_arc()?;
        let selected_sample_count = sample_selection.selected_sample_count;
        let selected_variant_count = variant_stop - variant_start;
        let mut row_major_dosage_values = vec![0.0_f32; selected_sample_count * selected_variant_count];
        self.read_dosage_f32_into_address_with_selection(
            sample_selection.as_ref(),
            variant_start,
            variant_stop,
            row_major_dosage_values.as_mut_ptr() as usize,
            row_major_dosage_values.len(),
        )?;
        Ok(row_major_dosage_values)
    }

    pub fn read_dosage_f32_into_address(
        &self,
        sample_indices: &[i64],
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<(), BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let sample_selection_start_time = Instant::now();
        let sample_selection = build_sample_selection(self.sample_count, sample_indices)?;
        self.profiling.record_sample_selection_prepare(elapsed_nanoseconds(sample_selection_start_time));
        self.read_dosage_f32_into_address_with_selection(
            &sample_selection,
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
        )
    }

    pub fn read_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<(), BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let sample_selection = self.prepared_sample_selection_arc()?;
        self.read_dosage_f32_into_address_with_selection(
            sample_selection.as_ref(),
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
        )
    }

    pub fn bgen_path(&self) -> &Path {
        &self.bgen_path
    }

    fn prepared_sample_selection_arc(&self) -> Result<Arc<SampleSelection>, BgenError> {
        let prepared_sample_selection = self
            .prepared_sample_selection
            .lock()
            .map_err(|_| BgenError::InvalidFormat("Prepared BGEN sample selection mutex was poisoned.".to_string()))?;
        prepared_sample_selection.clone().ok_or_else(|| {
            BgenError::Range("Prepared BGEN sample selection was requested before binding aligned samples.".to_string())
        })
    }

    fn read_dosage_f32_into_address_with_selection(
        &self,
        sample_selection: &SampleSelection,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<(), BgenError> {
        let selected_sample_count = sample_selection.selected_sample_count;
        let selected_variant_count = variant_stop - variant_start;
        let expected_output_value_count =
            selected_sample_count.checked_mul(selected_variant_count).ok_or_else(|| {
                BgenError::Range("Integer overflow while validating BGEN output buffer size.".to_string())
            })?;
        if output_value_count != expected_output_value_count {
            return Err(BgenError::Range(format!(
                "Output buffer shape mismatch for BGEN dosage read. Expected {expected_output_value_count} float32 values, observed {output_value_count}.",
            )));
        }
        if selected_sample_count == 0 || selected_variant_count == 0 {
            return Ok(());
        }

        let output_pointer = output_pointer_address;
        let profiling = &self.profiling;
        let profiling_enabled = profiling.enabled.load(Ordering::Relaxed);
        profiling.record_selected_sample_count(selected_sample_count);
        let decode_tile_variant_count = decode_tile_variant_count();
        self.variant_records[variant_start..variant_stop]
            .par_chunks(decode_tile_variant_count)
            .enumerate()
            .map_init(ThreadScratch::default, |thread_scratch, (tile_index, variant_record_chunk)| {
                decode_variant_dosage_tile_into_row_major_matrix(
                    &self.mmap,
                    self.compression_type,
                    self.sample_count,
                    sample_selection,
                    variant_record_chunk,
                    output_pointer,
                    selected_variant_count,
                    tile_index * decode_tile_variant_count,
                    profiling_enabled,
                    self.trusted_no_missing_diploid,
                    thread_scratch,
                )
            })
            .collect::<Result<Vec<ThreadLocalProfileSnapshot>, BgenError>>()?
            .into_iter()
            .for_each(|thread_local_snapshot| profiling.merge_thread_local_snapshot(&thread_local_snapshot));
        Ok(())
    }
}

#[derive(Debug)]
struct SampleSelection {
    selected_sample_count: usize,
    file_to_selected_index: Vec<usize>,
    is_identity: bool,
}

struct ThreadScratch {
    zlib_decompressor: Decompress,
    decompressed_probability_block: Vec<u8>,
    dosage_tile: Vec<f32>,
}

impl Default for ThreadScratch {
    fn default() -> Self {
        Self {
            zlib_decompressor: Decompress::new(true),
            decompressed_probability_block: Vec::new(),
            dosage_tile: Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_variant_dosage_tile_into_row_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_pointer_address: usize,
    selected_variant_count: usize,
    tile_variant_start_index: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<ThreadLocalProfileSnapshot, BgenError> {
    let tile_variant_count = variant_record_chunk.len();
    let tile_value_count = sample_selection
        .selected_sample_count
        .checked_mul(tile_variant_count)
        .ok_or_else(|| BgenError::Range("Integer overflow while allocating a BGEN dosage decode tile.".to_string()))?;
    if thread_scratch.dosage_tile.capacity() < tile_value_count {
        thread_scratch.dosage_tile.reserve(tile_value_count - thread_scratch.dosage_tile.capacity());
    }
    unsafe {
        // Every tile element is overwritten during decode before any reads occur.
        thread_scratch.dosage_tile.set_len(tile_value_count);
    }

    let tile_pointer_address = thread_scratch.dosage_tile.as_mut_ptr() as usize;
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    for (tile_variant_index, variant_record) in variant_record_chunk.iter().enumerate() {
        let variant_profile_snapshot = decode_variant_dosages_into_row_major_matrix(
            mmap,
            compression_type,
            sample_count,
            sample_selection,
            variant_record,
            tile_pointer_address,
            tile_variant_index,
            tile_variant_count,
            profiling_enabled,
            trusted_no_missing_diploid,
            thread_scratch,
        )?;
        thread_local_profile_snapshot.compressed_block_fetch_ns += variant_profile_snapshot.compressed_block_fetch_ns;
        thread_local_profile_snapshot.compressed_block_fetch_count +=
            variant_profile_snapshot.compressed_block_fetch_count;
        thread_local_profile_snapshot.compressed_byte_count += variant_profile_snapshot.compressed_byte_count;
        thread_local_profile_snapshot.decompression_ns += variant_profile_snapshot.decompression_ns;
        thread_local_profile_snapshot.decompression_count += variant_profile_snapshot.decompression_count;
        thread_local_profile_snapshot.uncompressed_byte_count += variant_profile_snapshot.uncompressed_byte_count;
        thread_local_profile_snapshot.zlib_stream_count += variant_profile_snapshot.zlib_stream_count;
        thread_local_profile_snapshot.probability_decode_ns += variant_profile_snapshot.probability_decode_ns;
        thread_local_profile_snapshot.probability_decode_count += variant_profile_snapshot.probability_decode_count;
        thread_local_profile_snapshot.variant_decode_count += variant_profile_snapshot.variant_decode_count;
        thread_local_profile_snapshot.output_write_ns += variant_profile_snapshot.output_write_ns;
        thread_local_profile_snapshot.output_write_count += variant_profile_snapshot.output_write_count;
        thread_local_profile_snapshot.output_byte_count += variant_profile_snapshot.output_byte_count;
    }
    thread_local_profile_snapshot.decode_tile_count += 1;

    let copy_tile_start_time = profiling_enabled.then(Instant::now);
    let output_pointer = output_pointer_address as *mut f32;
    for selected_sample_index in 0..sample_selection.selected_sample_count {
        let tile_row_start = selected_sample_index * tile_variant_count;
        let output_row_start = (selected_sample_index * selected_variant_count) + tile_variant_start_index;
        unsafe {
            // Each parallel worker owns a disjoint contiguous variant span in every output row.
            std::ptr::copy_nonoverlapping(
                thread_scratch.dosage_tile.as_ptr().add(tile_row_start),
                output_pointer.add(output_row_start),
                tile_variant_count,
            );
        }
    }
    if let Some(copy_tile_start_time) = copy_tile_start_time {
        thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(copy_tile_start_time);
        thread_local_profile_snapshot.output_write_count += 1;
        thread_local_profile_snapshot.output_byte_count +=
            u64::try_from(tile_value_count.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
                BgenError::Range("Integer overflow while profiling BGEN tile copy bytes.".to_string())
            })?)
            .unwrap_or(u64::MAX);
    }

    Ok(thread_local_profile_snapshot)
}

fn validate_variant_bounds(variant_start: usize, variant_stop: usize, variant_count: usize) -> Result<(), BgenError> {
    if variant_start > variant_stop || variant_stop > variant_count {
        return Err(BgenError::Range(format!(
            "Variant bounds must satisfy 0 <= start <= stop <= {variant_count}. Received start={variant_start}, stop={variant_stop}.",
        )));
    }
    Ok(())
}

fn build_sample_selection(sample_count: usize, sample_indices: &[i64]) -> Result<SampleSelection, BgenError> {
    let mut file_to_selected_index = vec![usize::MAX; sample_count];
    let mut is_identity = sample_indices.len() == sample_count;
    for (selected_index, raw_sample_index) in sample_indices.iter().enumerate() {
        let sample_index = usize::try_from(*raw_sample_index).map_err(|_| {
            BgenError::Range(format!("Sample indices must be non-negative. Observed sample index {raw_sample_index}.",))
        })?;
        if sample_index >= sample_count {
            return Err(BgenError::Range(format!(
                "Sample index {sample_index} is out of bounds for a BGEN file with {sample_count} samples.",
            )));
        }
        if file_to_selected_index[sample_index] != usize::MAX {
            return Err(BgenError::Range(format!(
                "Sample index {sample_index} was requested more than once in the same read.",
            )));
        }
        file_to_selected_index[sample_index] = selected_index;
        if sample_index != selected_index {
            is_identity = false;
        }
    }
    Ok(SampleSelection { selected_sample_count: sample_indices.len(), file_to_selected_index, is_identity })
}

fn parse_sample_identifier_block(
    mmap: &[u8],
    sample_block_offset: usize,
    first_variant_offset: usize,
    expected_sample_count: usize,
) -> Result<Vec<String>, BgenError> {
    let block_length = u32_to_usize(read_u32_at(mmap, sample_block_offset)?)?;
    let sample_block_stop = sample_block_offset + block_length;
    if sample_block_stop > first_variant_offset {
        return Err(BgenError::InvalidFormat(
            "Embedded BGEN sample block overlaps the first variant block.".to_string(),
        ));
    }

    let observed_sample_count = u32_to_usize(read_u32_at(mmap, sample_block_offset + 4)?)?;
    if observed_sample_count != expected_sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "Embedded BGEN sample block reports {observed_sample_count} samples, but the header reports {expected_sample_count}.",
        )));
    }

    let mut cursor = sample_block_offset + 8;
    let mut sample_identifiers = Vec::with_capacity(expected_sample_count);
    for _sample_index in 0..expected_sample_count {
        let identifier_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let identifier_bytes = read_exact_bytes(mmap, cursor, identifier_length)?;
        sample_identifiers.push(String::from_utf8_lossy(identifier_bytes).into_owned());
        cursor += identifier_length;
    }
    if cursor != sample_block_stop {
        return Err(BgenError::InvalidFormat(
            "Embedded BGEN sample block length does not match the encoded sample identifiers.".to_string(),
        ));
    }

    Ok(sample_identifiers)
}

fn parse_variant_records(
    mmap: &[u8],
    first_variant_offset: usize,
    variant_count: usize,
    sample_count: usize,
    compression_type: CompressionType,
) -> Result<Vec<VariantRecord>, BgenError> {
    let mut cursor = first_variant_offset;
    let mut variant_records = Vec::with_capacity(variant_count);

    for variant_index in 0..variant_count {
        let variant_identifier_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let variant_identifier =
            String::from_utf8_lossy(read_exact_bytes(mmap, cursor, variant_identifier_length)?).into_owned();
        cursor += variant_identifier_length;

        let rsid_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let rsid = String::from_utf8_lossy(read_exact_bytes(mmap, cursor, rsid_length)?).into_owned();
        cursor += rsid_length;

        let chromosome_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let chromosome = String::from_utf8_lossy(read_exact_bytes(mmap, cursor, chromosome_length)?).into_owned();
        cursor += chromosome_length;

        let position = i64::from(read_u32_at(mmap, cursor)?);
        cursor += 4;

        let allele_count = read_u16_at(mmap, cursor)?;
        cursor += 2;
        if allele_count != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Only diploid biallelic BGEN variants are supported. Variant index {variant_index} reports {allele_count} alleles.",
            )));
        }

        let mut allele_values = Vec::with_capacity(usize::from(allele_count));
        for _allele_index in 0..usize::from(allele_count) {
            let allele_length = u32_to_usize(read_u32_at(mmap, cursor)?)?;
            cursor += ALLELE_LENGTH_SIZE_IN_BYTES;
            let allele_value = String::from_utf8_lossy(read_exact_bytes(mmap, cursor, allele_length)?).into_owned();
            cursor += allele_length;
            allele_values.push(allele_value);
        }

        let genotype_block_offset = cursor;
        let total_block_length = u32_to_usize(read_u32_at(mmap, genotype_block_offset)?)?;
        let block_payload_offset = genotype_block_offset + 4;
        let (probability_payload_offset, probability_payload_length, declared_uncompressed_block_length) =
            match compression_type {
                CompressionType::None => (block_payload_offset, total_block_length, total_block_length),
                CompressionType::Zlib => {
                    let declared_uncompressed_block_length = u32_to_usize(read_u32_at(mmap, block_payload_offset)?)?;
                    let probability_payload_length = total_block_length.checked_sub(4).ok_or_else(|| {
                        BgenError::InvalidFormat(
                            "Compressed BGEN blocks must include a four-byte uncompressed length prefix.".to_string(),
                        )
                    })?;
                    (block_payload_offset + 4, probability_payload_length, declared_uncompressed_block_length)
                }
            };
        cursor += 4 + total_block_length;
        if cursor > mmap.len() {
            return Err(BgenError::InvalidFormat(format!(
                "Variant index {variant_index} points beyond the end of the BGEN file.",
            )));
        }

        if variant_index == 0 {
            validate_variant_probability_block(
                mmap,
                compression_type,
                &VariantRecord {
                    probability_payload_offset,
                    probability_payload_length,
                    declared_uncompressed_block_length,
                    chromosome: chromosome.clone(),
                    resolved_variant_identifier: if rsid.is_empty() {
                        variant_identifier.clone()
                    } else {
                        rsid.clone()
                    },
                    position,
                    counted_allele: allele_values[1].clone(),
                    reference_allele: allele_values[0].clone(),
                },
                sample_count,
                "first variant",
            )?;
        }

        let reference_allele = allele_values[0].clone();
        let counted_allele = allele_values[1].clone();
        let resolved_variant_identifier = if rsid.is_empty() { variant_identifier } else { rsid.clone() };

        variant_records.push(VariantRecord {
            probability_payload_offset,
            probability_payload_length,
            declared_uncompressed_block_length,
            chromosome,
            resolved_variant_identifier,
            position,
            counted_allele,
            reference_allele,
        });
    }

    Ok(variant_records)
}

fn build_chromosome_boundary_indices(variant_records: &[VariantRecord]) -> Vec<usize> {
    let mut chromosome_boundary_indices = Vec::with_capacity(variant_records.len().min(256) + 1);
    chromosome_boundary_indices.push(0);
    for variant_index in 1..variant_records.len() {
        if variant_records[variant_index].chromosome != variant_records[variant_index - 1].chromosome {
            chromosome_boundary_indices.push(variant_index);
        }
    }
    chromosome_boundary_indices.push(variant_records.len());
    chromosome_boundary_indices
}

fn validate_variant_probability_block(
    mmap: &[u8],
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    sample_count: usize,
    variant_label: &str,
) -> Result<(), BgenError> {
    let mut thread_scratch = ThreadScratch::default();
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let probability_block = read_probability_block(
        mmap,
        compression_type,
        variant_record,
        &mut thread_scratch,
        &mut thread_local_profile_snapshot,
        false,
    )?;
    let observed_sample_count = u32_to_usize(read_u32_at(probability_block, 0)?)?;
    if observed_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "The {variant_label} stores {observed_sample_count} samples in its probability block, but the file header reports {sample_count}.",
        )));
    }
    Ok(())
}

fn validate_variant_compatible_with_trusted_no_missing_diploid(
    mmap: &[u8],
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    sample_count: usize,
    thread_scratch: &mut ThreadScratch,
    thread_local_profile_snapshot: &mut ThreadLocalProfileSnapshot,
) -> Result<(), BgenError> {
    let probability_block = read_probability_block(
        mmap,
        compression_type,
        variant_record,
        thread_scratch,
        thread_local_profile_snapshot,
        false,
    )?;

    let mut cursor = 0;
    let stored_sample_count = u32_to_usize(read_u32_at(probability_block, cursor)?)?;
    cursor += 4;
    if stored_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' stores {stored_sample_count} samples in its probability block, but the file header reports {sample_count}.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let allele_count = read_u16_at(probability_block, cursor)?;
    cursor += 2;
    if allele_count != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' is not compatible with trusted_no_missing_diploid because it is not biallelic.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let minimum_ploidy = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let maximum_ploidy = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    if minimum_ploidy != 2 || maximum_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' is not compatible with trusted_no_missing_diploid because ploidy bounds are [{minimum_ploidy}, {maximum_ploidy}] instead of [2, 2].",
            variant_record.resolved_variant_identifier,
        )));
    }

    let sample_ploidy_and_missingness = read_exact_bytes(probability_block, cursor, sample_count)?;
    cursor += sample_count;
    if !all_samples_present_diploid(sample_ploidy_and_missingness) {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' is not compatible with trusted_no_missing_diploid because at least one sample is missing or non-diploid.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let phased_flag = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    if phased_flag != 0 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' is not compatible with trusted_no_missing_diploid because it is phased.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let probability_bit_count = read_u8_at(probability_block, cursor)?;
    if probability_bit_count != 8 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' is not compatible with trusted_no_missing_diploid because it uses {probability_bit_count} bits per probability instead of 8.",
            variant_record.resolved_variant_identifier,
        )));
    }

    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments, clippy::too_many_lines)]
fn decode_variant_dosages_into_row_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: usize,
    variant_index: usize,
    variant_count: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<ThreadLocalProfileSnapshot, BgenError> {
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let probability_block = read_probability_block(
        mmap,
        compression_type,
        variant_record,
        thread_scratch,
        &mut thread_local_profile_snapshot,
        profiling_enabled,
    )?;
    let block_bytes = probability_block;

    let mut cursor = 0;
    let stored_sample_count = u32_to_usize(read_u32_at(block_bytes, cursor)?)?;
    cursor += 4;
    if stored_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' stores {stored_sample_count} samples in its probability block, but the file header reports {sample_count}.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let allele_count = read_u16_at(block_bytes, cursor)?;
    cursor += 2;
    if allele_count != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' is not biallelic.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let minimum_ploidy = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    let maximum_ploidy = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    if minimum_ploidy != 2 || maximum_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' uses ploidy bounds [{minimum_ploidy}, {maximum_ploidy}], but the native Rust reader currently supports diploid BGEN variants only.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let sample_ploidy_and_missingness = read_exact_bytes(block_bytes, cursor, sample_count)?;
    cursor += sample_count;

    let phased_flag = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    let probability_bit_count = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    if !(1..=32).contains(&probability_bit_count) {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' uses {probability_bit_count} bits per probability, but BGEN Layout 2 requires a value between 1 and 32.",
            variant_record.resolved_variant_identifier,
        )));
    }

    if phased_flag == 0 && probability_bit_count == 8 {
        return decode_unphased_eight_bit_dosages_into_row_major_matrix(
            sample_ploidy_and_missingness,
            &block_bytes[cursor..],
            sample_selection,
            variant_record,
            output_pointer_address,
            variant_index,
            variant_count,
            profiling_enabled,
            trusted_no_missing_diploid,
            thread_local_profile_snapshot,
        );
    }

    let probability_scale_denominator =
        if probability_bit_count == 32 { f64::from(u32::MAX) } else { f64::from((1_u32 << probability_bit_count) - 1) };
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let mut bit_reader = PackedProbabilityReader::new(&block_bytes[cursor..]);
    let output_pointer = output_pointer_address as *mut f32;
    if sample_selection.is_identity {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().enumerate() {
            let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
            if observed_ploidy != 2 {
                return Err(BgenError::UnsupportedFormat(format!(
                    "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                    variant_record.resolved_variant_identifier,
                )));
            }
            let is_missing = (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0;

            let dosage_value = match phased_flag {
                0 => {
                    let homozygous_reference_probability =
                        f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                    let heterozygous_probability =
                        f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                    if is_missing {
                        f32::NAN
                    } else {
                        let dosage_value =
                            2.0_f64 - ((2.0 * homozygous_reference_probability) + heterozygous_probability);
                        dosage_value as f32
                    }
                }
                1 => {
                    let first_haplotype_reference_probability =
                        f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                    let second_haplotype_reference_probability =
                        f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                    if is_missing {
                        f32::NAN
                    } else {
                        let dosage_value =
                            2.0_f64 - (first_haplotype_reference_probability + second_haplotype_reference_probability);
                        dosage_value as f32
                    }
                }
                unsupported_flag => {
                    return Err(BgenError::InvalidFormat(format!(
                        "Variant '{}' uses phased flag {unsupported_flag}, but BGEN Layout 2 requires 0 or 1.",
                        variant_record.resolved_variant_identifier,
                    )));
                }
            };

            let output_offset = (file_sample_index * variant_count) + variant_index;
            unsafe {
                // Identity-aligned full-sample reads map file-order rows directly into output rows.
                output_pointer.add(output_offset).write(dosage_value);
            }
        }
        if let Some(output_write_start_time) = output_write_start_time {
            thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
            thread_local_profile_snapshot.output_write_count += 1;
            thread_local_profile_snapshot.output_byte_count +=
                u64::try_from(sample_ploidy_and_missingness.len().checked_mul(std::mem::size_of::<f32>()).ok_or_else(
                    || BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string()),
                )?)
                .unwrap_or(u64::MAX);
        }

        if let Some(probability_decode_start_time) = probability_decode_start_time {
            thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
            thread_local_profile_snapshot.probability_decode_count += 1;
        }
        thread_local_profile_snapshot.variant_decode_count += 1;

        return Ok(thread_local_profile_snapshot);
    }

    let output_write_start_time = profiling_enabled.then(Instant::now);
    for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().enumerate() {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }
        let is_missing = (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0;

        let dosage_value = match phased_flag {
            0 => {
                let homozygous_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                let heterozygous_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                if is_missing {
                    f32::NAN
                } else {
                    let dosage_value = 2.0_f64 - ((2.0 * homozygous_reference_probability) + heterozygous_probability);
                    dosage_value as f32
                }
            }
            1 => {
                let first_haplotype_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                let second_haplotype_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                if is_missing {
                    f32::NAN
                } else {
                    let dosage_value =
                        2.0_f64 - (first_haplotype_reference_probability + second_haplotype_reference_probability);
                    dosage_value as f32
                }
            }
            unsupported_flag => {
                return Err(BgenError::InvalidFormat(format!(
                    "Variant '{}' uses phased flag {unsupported_flag}, but BGEN Layout 2 requires 0 or 1.",
                    variant_record.resolved_variant_identifier,
                )));
            }
        };

        let selected_index = sample_selection.file_to_selected_index[file_sample_index];
        if selected_index != usize::MAX {
            let output_offset = (selected_index * variant_count) + variant_index;
            unsafe {
                // Each parallel worker owns one distinct variant column, so these writes do not overlap.
                output_pointer.add(output_offset).write(dosage_value);
            }
        }
    }
    if let Some(output_write_start_time) = output_write_start_time {
        thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
        thread_local_profile_snapshot.output_write_count += 1;
        thread_local_profile_snapshot.output_byte_count += u64::try_from(
            sample_selection
                .selected_sample_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string()))?,
        )
        .unwrap_or(u64::MAX);
    }

    if let Some(probability_decode_start_time) = probability_decode_start_time {
        thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
        thread_local_profile_snapshot.probability_decode_count += 1;
    }

    thread_local_profile_snapshot.variant_decode_count += 1;

    Ok(thread_local_profile_snapshot)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn decode_unphased_eight_bit_dosages_into_row_major_matrix(
    sample_ploidy_and_missingness: &[u8],
    packed_probability_bytes: &[u8],
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: usize,
    variant_index: usize,
    variant_count: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    mut thread_local_profile_snapshot: ThreadLocalProfileSnapshot,
) -> Result<ThreadLocalProfileSnapshot, BgenError> {
    let expected_probability_byte_count = sample_ploidy_and_missingness.len().checked_mul(2).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while decoding 8-bit BGEN probabilities.".to_string())
    })?;
    if packed_probability_bytes.len() < expected_probability_byte_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' ended before all 8-bit probabilities were decoded.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let dosage_lookup = unphased_eight_bit_dosage_lookup();
    let mut probability_offset = 0;
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let output_pointer = output_pointer_address as *mut f32;
    let all_samples_present = trusted_no_missing_diploid || all_samples_present_diploid(sample_ploidy_and_missingness);
    if sample_selection.is_identity && all_samples_present {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        for file_sample_index in 0..sample_ploidy_and_missingness.len() {
            let probability_offset = file_sample_index * 2;
            let packed_probability_index =
                read_packed_probability_index_eight_bit(packed_probability_bytes, probability_offset);
            let dosage_value = dosage_lookup[packed_probability_index];

            let output_offset = (file_sample_index * variant_count) + variant_index;
            unsafe {
                // Identity-aligned full-sample reads map file-order rows directly into output rows.
                output_pointer.add(output_offset).write(dosage_value);
            }
        }
        if let Some(output_write_start_time) = output_write_start_time {
            thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
            thread_local_profile_snapshot.output_write_count += 1;
            thread_local_profile_snapshot.output_byte_count +=
                u64::try_from(sample_ploidy_and_missingness.len().checked_mul(std::mem::size_of::<f32>()).ok_or_else(
                    || BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string()),
                )?)
                .unwrap_or(u64::MAX);
        }

        if let Some(probability_decode_start_time) = probability_decode_start_time {
            thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
            thread_local_profile_snapshot.probability_decode_count += 1;
        }
        thread_local_profile_snapshot.variant_decode_count += 1;

        return Ok(thread_local_profile_snapshot);
    }
    if sample_selection.is_identity {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().enumerate() {
            let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
            if observed_ploidy != 2 {
                return Err(BgenError::UnsupportedFormat(format!(
                    "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                    variant_record.resolved_variant_identifier,
                )));
            }

            let packed_probability_index = usize::from(packed_probability_bytes[probability_offset])
                | (usize::from(packed_probability_bytes[probability_offset + 1]) << 8);
            probability_offset += 2;

            let dosage_value = if (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0 {
                f32::NAN
            } else {
                dosage_lookup[packed_probability_index]
            };

            let output_offset = (file_sample_index * variant_count) + variant_index;
            unsafe {
                // Identity-aligned full-sample reads map file-order rows directly into output rows.
                output_pointer.add(output_offset).write(dosage_value);
            }
        }
        if let Some(output_write_start_time) = output_write_start_time {
            thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
            thread_local_profile_snapshot.output_write_count += 1;
            thread_local_profile_snapshot.output_byte_count +=
                u64::try_from(sample_ploidy_and_missingness.len().checked_mul(std::mem::size_of::<f32>()).ok_or_else(
                    || BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string()),
                )?)
                .unwrap_or(u64::MAX);
        }

        if let Some(probability_decode_start_time) = probability_decode_start_time {
            thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
            thread_local_profile_snapshot.probability_decode_count += 1;
        }
        thread_local_profile_snapshot.variant_decode_count += 1;

        return Ok(thread_local_profile_snapshot);
    }

    if all_samples_present {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        for file_sample_index in 0..sample_ploidy_and_missingness.len() {
            let probability_offset = file_sample_index * 2;
            let packed_probability_index =
                read_packed_probability_index_eight_bit(packed_probability_bytes, probability_offset);
            let dosage_value = dosage_lookup[packed_probability_index];

            let selected_index = sample_selection.file_to_selected_index[file_sample_index];
            if selected_index != usize::MAX {
                let output_offset = (selected_index * variant_count) + variant_index;
                unsafe {
                    // Each parallel worker owns one distinct variant column, so these writes do not overlap.
                    output_pointer.add(output_offset).write(dosage_value);
                }
            }
        }
        if let Some(output_write_start_time) = output_write_start_time {
            thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
            thread_local_profile_snapshot.output_write_count += 1;
            thread_local_profile_snapshot.output_byte_count += u64::try_from(
                sample_selection.selected_sample_count.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
                    BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string())
                })?,
            )
            .unwrap_or(u64::MAX);
        }

        if let Some(probability_decode_start_time) = probability_decode_start_time {
            thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
            thread_local_profile_snapshot.probability_decode_count += 1;
        }
        thread_local_profile_snapshot.variant_decode_count += 1;

        return Ok(thread_local_profile_snapshot);
    }

    let output_write_start_time = profiling_enabled.then(Instant::now);
    for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().enumerate() {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }

        let packed_probability_index = usize::from(packed_probability_bytes[probability_offset])
            | (usize::from(packed_probability_bytes[probability_offset + 1]) << 8);
        probability_offset += 2;

        let dosage_value = if (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0 {
            f32::NAN
        } else {
            dosage_lookup[packed_probability_index]
        };

        let selected_index = sample_selection.file_to_selected_index[file_sample_index];
        if selected_index != usize::MAX {
            let output_offset = (selected_index * variant_count) + variant_index;
            unsafe {
                // Each parallel worker owns one distinct variant column, so these writes do not overlap.
                output_pointer.add(output_offset).write(dosage_value);
            }
        }
    }
    if let Some(output_write_start_time) = output_write_start_time {
        thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
        thread_local_profile_snapshot.output_write_count += 1;
        thread_local_profile_snapshot.output_byte_count += u64::try_from(
            sample_selection
                .selected_sample_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string()))?,
        )
        .unwrap_or(u64::MAX);
    }

    if let Some(probability_decode_start_time) = probability_decode_start_time {
        thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
        thread_local_profile_snapshot.probability_decode_count += 1;
    }
    thread_local_profile_snapshot.variant_decode_count += 1;

    Ok(thread_local_profile_snapshot)
}

fn read_probability_block<'a>(
    mmap: &'a [u8],
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    thread_scratch: &'a mut ThreadScratch,
    thread_local_profile_snapshot: &mut ThreadLocalProfileSnapshot,
    profiling_enabled: bool,
) -> Result<&'a [u8], BgenError> {
    let compressed_block_fetch_start_time = profiling_enabled.then(Instant::now);
    match compression_type {
        CompressionType::None => {
            let block_payload = read_exact_bytes(
                mmap,
                variant_record.probability_payload_offset,
                variant_record.probability_payload_length,
            )?;
            if let Some(compressed_block_fetch_start_time) = compressed_block_fetch_start_time {
                thread_local_profile_snapshot.compressed_block_fetch_ns +=
                    elapsed_nanoseconds(compressed_block_fetch_start_time);
                thread_local_profile_snapshot.compressed_block_fetch_count += 1;
                thread_local_profile_snapshot.compressed_byte_count +=
                    u64::try_from(variant_record.probability_payload_length).unwrap_or(u64::MAX);
            }
            thread_local_profile_snapshot.uncompressed_byte_count +=
                u64::try_from(variant_record.declared_uncompressed_block_length).unwrap_or(u64::MAX);
            Ok(block_payload)
        }
        CompressionType::Zlib => {
            let compressed_payload = read_exact_bytes(
                mmap,
                variant_record.probability_payload_offset,
                variant_record.probability_payload_length,
            )?;
            if let Some(compressed_block_fetch_start_time) = compressed_block_fetch_start_time {
                thread_local_profile_snapshot.compressed_block_fetch_ns +=
                    elapsed_nanoseconds(compressed_block_fetch_start_time);
                thread_local_profile_snapshot.compressed_block_fetch_count += 1;
                thread_local_profile_snapshot.compressed_byte_count +=
                    u64::try_from(variant_record.probability_payload_length).unwrap_or(u64::MAX);
            }

            let decompression_start_time = profiling_enabled.then(Instant::now);
            decompress_zlib_block_into_scratch(
                compressed_payload,
                variant_record.declared_uncompressed_block_length,
                thread_scratch,
            )?;
            if let Some(decompression_start_time) = decompression_start_time {
                thread_local_profile_snapshot.decompression_ns += elapsed_nanoseconds(decompression_start_time);
                thread_local_profile_snapshot.decompression_count += 1;
            }
            thread_local_profile_snapshot.uncompressed_byte_count +=
                u64::try_from(variant_record.declared_uncompressed_block_length).unwrap_or(u64::MAX);
            thread_local_profile_snapshot.zlib_stream_count += 1;
            Ok(thread_scratch.decompressed_probability_block.as_slice())
        }
    }
}

fn decompress_zlib_block_into_scratch(
    compressed_payload: &[u8],
    expected_length: usize,
    thread_scratch: &mut ThreadScratch,
) -> Result<(), BgenError> {
    thread_scratch.decompressed_probability_block.clear();
    if thread_scratch.decompressed_probability_block.capacity() < expected_length {
        thread_scratch
            .decompressed_probability_block
            .reserve(expected_length - thread_scratch.decompressed_probability_block.capacity());
    }
    thread_scratch.zlib_decompressor.reset(true);
    let status = thread_scratch
        .zlib_decompressor
        .decompress_vec(compressed_payload, &mut thread_scratch.decompressed_probability_block, FlushDecompress::Finish)
        .map_err(std::io::Error::from)?;
    if status != Status::StreamEnd {
        return Err(BgenError::InvalidFormat(
            "Zlib-compressed BGEN block did not terminate at stream end.".to_string(),
        ));
    }
    if thread_scratch.decompressed_probability_block.len() != expected_length {
        return Err(BgenError::InvalidFormat(format!(
            "Zlib-compressed BGEN block expanded to {} bytes, but the header declared {expected_length} bytes.",
            thread_scratch.decompressed_probability_block.len(),
        )));
    }
    Ok(())
}

struct PackedProbabilityReader<'a> {
    packed_probability_bytes: &'a [u8],
    bit_offset: usize,
}

impl<'a> PackedProbabilityReader<'a> {
    fn new(packed_probability_bytes: &'a [u8]) -> Self {
        Self { packed_probability_bytes, bit_offset: 0 }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn read_probability(&mut self, bit_count: u8) -> Result<u32, BgenError> {
        let bit_count_usize = usize::from(bit_count);
        let byte_offset = self.bit_offset / 8;
        let bit_index_in_byte = self.bit_offset % 8;
        let last_required_bit = self.bit_offset + bit_count_usize;
        let last_required_byte = last_required_bit.div_ceil(8);
        if last_required_byte > self.packed_probability_bytes.len() {
            return Err(BgenError::InvalidFormat(
                "Packed BGEN probability stream ended before all probabilities were decoded.".to_string(),
            ));
        }

        let mut window = 0_u64;
        let bytes_to_copy = (self.packed_probability_bytes.len() - byte_offset).min(8);
        for copied_byte_index in 0..bytes_to_copy {
            window |=
                u64::from(self.packed_probability_bytes[byte_offset + copied_byte_index]) << (copied_byte_index * 8);
        }

        let mask = if bit_count == 32 { u64::from(u32::MAX) } else { (1_u64 << bit_count) - 1 };
        let probability_value = ((window >> bit_index_in_byte) & mask) as u32;
        self.bit_offset += bit_count_usize;
        Ok(probability_value)
    }
}

fn read_u8_at(buffer: &[u8], offset: usize) -> Result<u8, BgenError> {
    Ok(*read_exact_bytes(buffer, offset, 1)?
        .first()
        .ok_or_else(|| BgenError::InvalidFormat("Unexpected empty byte slice.".to_string()))?)
}

fn read_u16_at(buffer: &[u8], offset: usize) -> Result<u16, BgenError> {
    let bytes = read_exact_bytes(buffer, offset, 2)?;
    let byte_array: [u8; 2] = bytes
        .try_into()
        .map_err(|_| BgenError::InvalidFormat("Failed to decode a two-byte integer from the BGEN file.".to_string()))?;
    Ok(u16::from_le_bytes(byte_array))
}

fn read_u32_at(buffer: &[u8], offset: usize) -> Result<u32, BgenError> {
    let bytes = read_exact_bytes(buffer, offset, 4)?;
    let byte_array: [u8; 4] = bytes.try_into().map_err(|_| {
        BgenError::InvalidFormat("Failed to decode a four-byte integer from the BGEN file.".to_string())
    })?;
    Ok(u32::from_le_bytes(byte_array))
}

fn read_exact_bytes(buffer: &[u8], offset: usize, length: usize) -> Result<&[u8], BgenError> {
    let stop = offset
        .checked_add(length)
        .ok_or_else(|| BgenError::InvalidFormat("Integer overflow while slicing BGEN file bytes.".to_string()))?;
    buffer
        .get(offset..stop)
        .ok_or_else(|| BgenError::InvalidFormat("Unexpected end of file while reading BGEN bytes.".to_string()))
}

fn u32_to_usize(value: u32) -> Result<usize, BgenError> {
    usize::try_from(value).map_err(|_| {
        BgenError::InvalidFormat(format!(
            "BGEN integer value {value} does not fit into the native platform usize type.",
        ))
    })
}
