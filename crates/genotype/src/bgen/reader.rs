use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;

use crate::buffer::raw_pointer::{OutputBufferAddress, OutputValueCount, RowMajorDosageBuffer};
use crate::common::{ChunkSpec, ChunkStats, VariantMetadataColumns};
use crate::error::GenotypeResult;
use crate::preprocess;

use super::decode::{
    DosageTileDecodeResult, ThreadScratch, decode_tile_variant_count, decode_variant_dosage_tile_into_row_major_matrix,
    read_exact_bytes, read_u32_at, u32_to_usize,
};
use super::error::BgenError;
use super::format::CompressionType;
use super::metadata::VariantRecord;
use super::profile::{ReaderProfileSnapshot, ReaderProfiling, ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use super::sample_selection::{SampleSelection, build_sample_selection};
use super::{index, metadata, trusted};

mod variant_major;

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
    trusted_no_missing_diploid_validated: AtomicBool,
    variant_records: Vec<VariantRecord>,
    chromosome_boundary_indices: Vec<usize>,
    prepared_sample_selection: Mutex<Option<Arc<SampleSelection>>>,
    profiling: ReaderProfiling,
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
            index::parse_sample_identifier_block(&mmap, sample_block_offset, first_variant_offset, sample_count)?
        } else {
            Vec::new()
        };

        let variant_records =
            index::parse_variant_records(&mmap, first_variant_offset, variant_count, sample_count, compression_type)?;
        let chromosome_boundary_indices = metadata::build_chromosome_boundary_indices(&variant_records);

        Ok(Self {
            bgen_path: bgen_path.to_path_buf(),
            mmap,
            sample_count,
            variant_count,
            contains_embedded_samples,
            sample_identifiers,
            compression_type,
            trusted_no_missing_diploid,
            trusted_no_missing_diploid_validated: AtomicBool::new(false),
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

    pub fn plan_chromosome_homogeneous_chunks(
        &self,
        chunk_size: usize,
        variant_limit: Option<usize>,
        committed_chunk_identifiers: &BTreeSet<usize>,
    ) -> GenotypeResult<Vec<ChunkSpec>> {
        crate::planner::plan_chromosome_homogeneous_chunks(
            self.variant_count,
            chunk_size,
            variant_limit,
            &self.chromosome_boundary_indices,
            committed_chunk_identifiers,
        )
    }

    pub fn prepare_sample_selection(&self, sample_indices: &[usize]) -> Result<(), BgenError> {
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
        if self.trusted_no_missing_diploid && self.trusted_no_missing_diploid_validated.load(Ordering::Acquire) {
            return Ok(());
        }
        self.variant_records.par_iter().try_for_each_init(
            || (ThreadScratch::default(), ThreadLocalProfileSnapshot::default()),
            |(thread_scratch, thread_local_profile_snapshot), variant_record| {
                trusted::validate_variant_compatible_with_trusted_no_missing_diploid(
                    &self.mmap,
                    self.compression_type,
                    variant_record,
                    self.sample_count,
                    thread_scratch,
                    thread_local_profile_snapshot,
                )
            },
        )?;
        if self.trusted_no_missing_diploid {
            self.trusted_no_missing_diploid_validated.store(true, Ordering::Release);
        }
        Ok(())
    }

    pub fn mark_trusted_no_missing_diploid_validated(&self) -> Result<(), BgenError> {
        if !self.trusted_no_missing_diploid {
            return Err(BgenError::Range(
                "Trusted no-missing diploid validation cannot be marked on a non-trusted BGEN reader.".to_string(),
            ));
        }
        self.trusted_no_missing_diploid_validated.store(true, Ordering::Release);
        Ok(())
    }

    pub fn variant_metadata_slice(
        &self,
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<VariantMetadataColumns, BgenError> {
        let metadata_slice_start_time = Instant::now();
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;

        let selected_variant_records = &self.variant_records[variant_start..variant_stop];
        let variant_metadata_columns = metadata::build_variant_metadata_columns(selected_variant_records);
        self.profiling.record_metadata_slice(elapsed_nanoseconds(metadata_slice_start_time));
        Ok(variant_metadata_columns)
    }

    pub fn read_preprocessed_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: OutputBufferAddress,
        output_value_count: OutputValueCount,
    ) -> Result<ChunkStats, BgenError> {
        let sample_selection = self.prepared_sample_selection_arc()?;
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        if selected_variant_count == 0 {
            return Ok(preprocess::build_empty_chunk_stats(0, false));
        }
        let selected_sample_count = output_value_count.get().checked_div(selected_variant_count).ok_or_else(|| {
            BgenError::Range("Unable to resolve sample count for preprocessed BGEN dosage matrix.".to_string())
        })?;
        let mut output_buffer = unsafe {
            RowMajorDosageBuffer::from_pointer_address(
                output_pointer_address,
                output_value_count,
                "row-major BGEN dosage",
            )?
        };
        self.read_dosage_f32_into_address_with_selection_and_optional_stats(
            &sample_selection,
            variant_start,
            variant_stop,
            output_buffer.pointer_address(),
            output_value_count,
            false,
        )
        .map(|_| ())?;
        preprocess::preprocess_row_major_dosage_matrix(
            output_buffer.values_mut(),
            selected_sample_count,
            selected_variant_count,
        )
        .map_err(|error| BgenError::Range(error.to_string()))
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

    fn trusted_no_missing_diploid_decode_enabled(&self) -> bool {
        self.trusted_no_missing_diploid && self.trusted_no_missing_diploid_validated.load(Ordering::Acquire)
    }

    fn validate_packed8_probability_pair_preconditions(&self) -> Result<(), BgenError> {
        if self.trusted_no_missing_diploid_decode_enabled() {
            return Ok(());
        }
        Err(BgenError::UnsupportedFormat(
            "Packed8 BGEN probability-pair delivery requires trusted no-missing diploid validation.".to_string(),
        ))
    }

    fn read_dosage_f32_into_address_with_selection_and_optional_stats(
        &self,
        sample_selection: &SampleSelection,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: OutputBufferAddress,
        output_value_count: OutputValueCount,
        collect_dosage_totals: bool,
    ) -> Result<Option<Vec<f32>>, BgenError> {
        let selected_sample_count = sample_selection.selected_sample_count;
        let selected_variant_count = variant_stop - variant_start;
        let expected_output_value_count =
            selected_sample_count.checked_mul(selected_variant_count).ok_or_else(|| {
                BgenError::Range("Integer overflow while validating BGEN output buffer size.".to_string())
            })?;
        if output_value_count.get() != expected_output_value_count {
            return Err(BgenError::Range(format!(
                "Output buffer shape mismatch for BGEN dosage read. Expected {expected_output_value_count} float32 values, observed {}.",
                output_value_count.get(),
            )));
        }
        if selected_sample_count == 0 || selected_variant_count == 0 {
            return Ok(collect_dosage_totals.then(|| vec![0.0_f32; selected_variant_count]));
        }

        let output_pointer = output_pointer_address;
        let profiling = &self.profiling;
        let profiling_enabled = profiling.is_enabled();
        profiling.record_selected_sample_count(selected_sample_count);
        let decode_tile_variant_count = decode_tile_variant_count();
        let decode_results = self.variant_records[variant_start..variant_stop]
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
                    self.trusted_no_missing_diploid_decode_enabled(),
                    collect_dosage_totals,
                    thread_scratch,
                )
            })
            .collect::<Result<Vec<DosageTileDecodeResult>, BgenError>>()?;
        let mut selected_dosage_totals = collect_dosage_totals.then(|| Vec::with_capacity(selected_variant_count));
        for decode_result in decode_results {
            profiling.merge_thread_local_snapshot(&decode_result.profile_snapshot);
            if let Some(totals) = &mut selected_dosage_totals {
                totals.extend(decode_result.selected_dosage_totals);
            }
        }
        Ok(selected_dosage_totals)
    }
}

fn validate_variant_bounds(variant_start: usize, variant_stop: usize, variant_count: usize) -> Result<(), BgenError> {
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
    fn private_reader_optional_stats_collects_row_major_dosage_totals() {
        let path = temporary_bgen_path("optional-stats");
        write_single_variant_bgen(&path);
        let reader = BgenReaderCore::open(&path, false).expect("BGEN reader should open");

        let empty_selection = build_sample_selection(reader.sample_count, &[]).expect("empty selection should build");
        let mut empty_output = Vec::<f32>::new();
        let empty_totals = reader
            .read_dosage_f32_into_address_with_selection_and_optional_stats(
                &empty_selection,
                0,
                1,
                OutputBufferAddress::from_mut_ptr(empty_output.as_mut_ptr()),
                OutputValueCount::new(0),
                true,
            )
            .expect("empty selected samples should return totals")
            .expect("totals should be collected");
        assert_eq!(empty_totals, vec![0.0]);

        let sample_selection =
            build_sample_selection(reader.sample_count, &[0, 2]).expect("non-contiguous selection should build");
        let mut output = vec![f32::NAN; 2];
        let totals = reader
            .read_dosage_f32_into_address_with_selection_and_optional_stats(
                &sample_selection,
                0,
                1,
                OutputBufferAddress::from_mut_ptr(output.as_mut_ptr()),
                OutputValueCount::new(output.len()),
                true,
            )
            .expect("row-major read should collect totals")
            .expect("totals should be present");
        assert_eq!(output, vec![2.0, 1.0]);
        assert_eq!(totals, vec![3.0]);

        let _ = fs::remove_file(path);
    }
}
