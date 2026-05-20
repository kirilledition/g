use std::time::Instant;

use super::decode::{
    DosageTileDecodeResult, ThreadScratch, VariantDecodeResult, read_exact_bytes, read_probability_block, read_u8_at,
    read_u16_at, read_u32_at, u32_to_usize, unphased_eight_bit_dosage_lookup,
};
use super::metadata::VariantRecord;
use super::profile::{ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use super::sample_selection::SampleSelection;
use super::{BgenError, CompressionType};
use crate::genotype::preprocess;

pub(super) fn all_samples_present_diploid(sample_ploidy_and_missingness: &[u8]) -> bool {
    const PRESENT_DIPLOID_BYTE_GROUP: [u8; 16] = [2_u8; 16];
    let mut ploidy_chunks = sample_ploidy_and_missingness.chunks_exact(PRESENT_DIPLOID_BYTE_GROUP.len());
    for ploidy_chunk in &mut ploidy_chunks {
        if ploidy_chunk != PRESENT_DIPLOID_BYTE_GROUP {
            return false;
        }
    }
    ploidy_chunks.remainder().iter().all(|ploidy_byte| *ploidy_byte == 2)
}

pub(super) fn validate_variant_compatible_with_trusted_no_missing_diploid(
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

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_trusted_variant_major_dosage_tile(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_pointer_address: usize,
    selected_sample_count: usize,
    tile_variant_start_index: usize,
    profiling_enabled: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<DosageTileDecodeResult, BgenError> {
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let mut selected_dosage_totals = vec![0.0_f32; variant_record_chunk.len()];
    let mut selected_dosage_square_totals = vec![0.0_f32; variant_record_chunk.len()];
    let selected_observation_counts =
        vec![i32::try_from(selected_sample_count).unwrap_or(i32::MAX); variant_record_chunk.len()];
    let mut zero_counts = vec![0_i32; variant_record_chunk.len()];
    let mut nonzero_counts = vec![0_i32; variant_record_chunk.len()];
    let mut homozygous_reference_counts = vec![0_i32; variant_record_chunk.len()];
    let mut heterozygous_counts = vec![0_i32; variant_record_chunk.len()];
    let mut homozygous_alternate_counts = vec![0_i32; variant_record_chunk.len()];
    for (tile_variant_index, variant_record) in variant_record_chunk.iter().enumerate() {
        let variant_decode_result = decode_trusted_unphased_eight_bit_variant_into_variant_major_matrix(
            mmap,
            compression_type,
            sample_count,
            sample_selection,
            variant_record,
            output_pointer_address,
            tile_variant_start_index + tile_variant_index,
            selected_sample_count,
            profiling_enabled,
            thread_scratch,
        )?;
        let variant_profile_snapshot = variant_decode_result.profile_snapshot;
        selected_dosage_totals[tile_variant_index] = variant_decode_result.selected_dosage_total;
        selected_dosage_square_totals[tile_variant_index] = variant_decode_result.selected_dosage_square_total;
        zero_counts[tile_variant_index] = variant_decode_result.zero_count;
        nonzero_counts[tile_variant_index] = variant_decode_result.nonzero_count;
        homozygous_reference_counts[tile_variant_index] = variant_decode_result.homozygous_reference_count;
        heterozygous_counts[tile_variant_index] = variant_decode_result.heterozygous_count;
        homozygous_alternate_counts[tile_variant_index] = variant_decode_result.homozygous_alternate_count;
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
    Ok(DosageTileDecodeResult {
        profile_snapshot: thread_local_profile_snapshot,
        selected_dosage_totals,
        selected_dosage_square_totals,
        selected_observation_counts,
        has_missing_values: false,
        zero_counts,
        nonzero_counts,
        homozygous_reference_counts,
        heterozygous_counts,
        homozygous_alternate_counts,
    })
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn decode_trusted_unphased_eight_bit_variant_into_variant_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: usize,
    variant_index: usize,
    selected_sample_count: usize,
    profiling_enabled: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<VariantDecodeResult, BgenError> {
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let probability_block = read_probability_block(
        mmap,
        compression_type,
        variant_record,
        thread_scratch,
        &mut thread_local_profile_snapshot,
        profiling_enabled,
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
            "Variant '{}' is not biallelic.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let minimum_ploidy = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let maximum_ploidy = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    if minimum_ploidy != 2 || maximum_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' uses ploidy bounds [{minimum_ploidy}, {maximum_ploidy}], but variant-major trusted reads require diploid variants.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let sample_ploidy_and_missingness = read_exact_bytes(probability_block, cursor, sample_count)?;
    cursor += sample_count;
    if !all_samples_present_diploid(sample_ploidy_and_missingness) {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' contains missing or non-diploid samples, but variant-major trusted reads require no missing diploid genotypes.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let phased_flag = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let probability_bit_count = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    if phased_flag != 0 || probability_bit_count != 8 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' is not an unphased 8-bit BGEN variant.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let expected_probability_byte_count = sample_count.checked_mul(2).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while decoding 8-bit BGEN probabilities.".to_string())
    })?;
    let packed_probability_bytes = read_exact_bytes(probability_block, cursor, expected_probability_byte_count)?;
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let output_write_start_time = profiling_enabled.then(Instant::now);
    let dosage_lookup = unphased_eight_bit_dosage_lookup();
    let output_pointer = output_pointer_address as *mut f32;
    let variant_row_offset = variant_index.checked_mul(selected_sample_count).ok_or_else(|| {
        BgenError::Range("Integer overflow while locating variant-major BGEN output row.".to_string())
    })?;
    let mut selected_dosage_total = 0.0_f32;
    let mut selected_dosage_square_total = 0.0_f32;
    let mut zero_count = 0_i32;
    let mut nonzero_count = 0_i32;
    let mut homozygous_reference_count = 0_i32;
    let mut heterozygous_count = 0_i32;
    let mut homozygous_alternate_count = 0_i32;
    for (file_sample_index, probability_pair) in packed_probability_bytes.chunks_exact(2).enumerate() {
        let selected_index = if sample_selection.is_identity {
            file_sample_index
        } else {
            sample_selection.file_to_selected_index[file_sample_index]
        };
        if selected_index == usize::MAX {
            continue;
        }
        let packed_probability_index = usize::from(probability_pair[0]) | (usize::from(probability_pair[1]) << 8);
        let dosage_value = dosage_lookup[packed_probability_index];
        selected_dosage_total += dosage_value;
        selected_dosage_square_total += dosage_value * dosage_value;
        preprocess::increment_dosage_summary_counts(
            dosage_value,
            &mut zero_count,
            &mut nonzero_count,
            &mut homozygous_reference_count,
            &mut heterozygous_count,
            &mut homozygous_alternate_count,
        );
        unsafe {
            // Each parallel worker owns a distinct variant row in the variant-major output matrix.
            output_pointer.add(variant_row_offset + selected_index).write(dosage_value);
        }
    }
    if let Some(output_write_start_time) = output_write_start_time {
        thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
        thread_local_profile_snapshot.output_write_count += 1;
        thread_local_profile_snapshot.output_byte_count += u64::try_from(
            selected_sample_count
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
    Ok(VariantDecodeResult {
        profile_snapshot: thread_local_profile_snapshot,
        selected_dosage_total,
        selected_dosage_square_total,
        selected_observation_count: i32::try_from(selected_sample_count).unwrap_or(i32::MAX),
        has_missing_values: false,
        zero_count,
        nonzero_count,
        homozygous_reference_count,
        heterozygous_count,
        homozygous_alternate_count,
    })
}
