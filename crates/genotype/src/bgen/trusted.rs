use std::time::Instant;

use super::decode::{
    ThreadScratch, VariantDecodeResult, VariantMajorOutputMatrix, VariantMajorTileDecodeResult,
    VariantMajorTileStatsMut, packed_eight_bit_probability_index, read_eight_bit_probability_pair, read_exact_bytes,
    read_probability_block, read_u8_at, read_u16_at, read_u32_at, u32_to_usize, unphased_eight_bit_dosage_lookup,
    validate_variant_major_tile_stats_lengths,
};
use super::metadata::VariantRecord;
use super::profile::{ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use super::sample_selection::SampleSelection;
use super::simd;
use super::{BgenError, CompressionType};
use crate::buffer::raw_pointer::OutputBufferAddress;
use crate::preprocess;

fn selected_sample_count_to_i32(selected_sample_count: usize) -> Result<i32, BgenError> {
    i32::try_from(selected_sample_count).map_err(|_| {
        BgenError::Range(format!(
            "Selected sample count {selected_sample_count} exceeds the supported i32 statistics range.",
        ))
    })
}

#[derive(Clone, Copy)]
enum TrustedEightBitParseContext {
    Validation,
    VariantMajorDosage,
    VariantMajorPackedProbabilityPairs,
}

struct TrustedUnphasedEightBitBlock<'a> {
    packed_probability_bytes: &'a [u8],
}

fn trusted_ploidy_bounds_error_message(
    parse_context: TrustedEightBitParseContext,
    variant_identifier: &str,
    minimum_ploidy: u8,
    maximum_ploidy: u8,
) -> String {
    match parse_context {
        TrustedEightBitParseContext::Validation => format!(
            "Variant '{variant_identifier}' is not compatible with trusted_no_missing_diploid because ploidy bounds are [{minimum_ploidy}, {maximum_ploidy}] instead of [2, 2].",
        ),
        TrustedEightBitParseContext::VariantMajorDosage => format!(
            "Variant '{variant_identifier}' uses ploidy bounds [{minimum_ploidy}, {maximum_ploidy}], but variant-major trusted reads require diploid variants.",
        ),
        TrustedEightBitParseContext::VariantMajorPackedProbabilityPairs => format!(
            "Variant '{variant_identifier}' uses ploidy bounds [{minimum_ploidy}, {maximum_ploidy}], but packed8 trusted reads require diploid variants.",
        ),
    }
}

fn trusted_missingness_error_message(parse_context: TrustedEightBitParseContext, variant_identifier: &str) -> String {
    match parse_context {
        TrustedEightBitParseContext::Validation => format!(
            "Variant '{variant_identifier}' is not compatible with trusted_no_missing_diploid because at least one sample is missing or non-diploid.",
        ),
        TrustedEightBitParseContext::VariantMajorDosage => format!(
            "Variant '{variant_identifier}' contains missing or non-diploid samples, but variant-major trusted reads require every sample to be present diploid.",
        ),
        TrustedEightBitParseContext::VariantMajorPackedProbabilityPairs => format!(
            "Variant '{variant_identifier}' contains missing or non-diploid samples, but packed8 trusted reads require every sample to be present diploid.",
        ),
    }
}

fn validate_trusted_phase_and_bit_count(
    parse_context: TrustedEightBitParseContext,
    variant_identifier: &str,
    phased_flag: u8,
    probability_bit_count: u8,
) -> Result<(), BgenError> {
    match parse_context {
        TrustedEightBitParseContext::Validation => {
            if phased_flag != 0 {
                return Err(BgenError::UnsupportedFormat(format!(
                    "Variant '{variant_identifier}' is not compatible with trusted_no_missing_diploid because it is phased.",
                )));
            }
            if probability_bit_count != 8 {
                return Err(BgenError::UnsupportedFormat(format!(
                    "Variant '{variant_identifier}' is not compatible with trusted_no_missing_diploid because it uses {probability_bit_count} bits per probability instead of 8.",
                )));
            }
        }
        TrustedEightBitParseContext::VariantMajorDosage
        | TrustedEightBitParseContext::VariantMajorPackedProbabilityPairs => {
            if phased_flag != 0 || probability_bit_count != 8 {
                return Err(BgenError::UnsupportedFormat(format!(
                    "Variant '{variant_identifier}' is not an unphased 8-bit BGEN variant.",
                )));
            }
        }
    }

    Ok(())
}

fn parse_trusted_unphased_eight_bit_probability_block<'a>(
    probability_block: &'a [u8],
    sample_count: usize,
    variant_record: &VariantRecord,
    parse_context: TrustedEightBitParseContext,
    validate_sample_ploidy_and_missingness: bool,
) -> Result<TrustedUnphasedEightBitBlock<'a>, BgenError> {
    let variant_identifier = variant_record.resolved_variant_identifier.as_str();
    let mut cursor = 0;
    let stored_sample_count = u32_to_usize(read_u32_at(probability_block, cursor)?)?;
    cursor += 4;
    if stored_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{variant_identifier}' stores {stored_sample_count} samples in its probability block, but the file header reports {sample_count}.",
        )));
    }

    let allele_count = read_u16_at(probability_block, cursor)?;
    cursor += 2;
    if allele_count != 2 {
        return Err(BgenError::UnsupportedFormat(format!("Variant '{variant_identifier}' is not biallelic.")));
    }

    let minimum_ploidy = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let maximum_ploidy = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    if minimum_ploidy != 2 || maximum_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(trusted_ploidy_bounds_error_message(
            parse_context,
            variant_identifier,
            minimum_ploidy,
            maximum_ploidy,
        )));
    }

    let sample_ploidy_and_missingness = read_exact_bytes(probability_block, cursor, sample_count)?;
    cursor += sample_count;
    if validate_sample_ploidy_and_missingness
        && !simd::all_samples_present_diploid_simd_or_scalar(sample_ploidy_and_missingness)
    {
        return Err(BgenError::UnsupportedFormat(trusted_missingness_error_message(parse_context, variant_identifier)));
    }
    if !validate_sample_ploidy_and_missingness {
        debug_assert!(
            simd::all_samples_present_diploid_simd_or_scalar(sample_ploidy_and_missingness),
            "trusted no-missing diploid decode skipped a ploidy scan before validation"
        );
    }

    let phased_flag = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let probability_bit_count = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    validate_trusted_phase_and_bit_count(parse_context, variant_identifier, phased_flag, probability_bit_count)?;

    let expected_probability_byte_count = sample_count.checked_mul(2).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while decoding 8-bit BGEN probabilities.".to_string())
    })?;
    let packed_probability_bytes = read_exact_bytes(probability_block, cursor, expected_probability_byte_count)?;

    Ok(TrustedUnphasedEightBitBlock { packed_probability_bytes })
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

    parse_trusted_unphased_eight_bit_probability_block(
        probability_block,
        sample_count,
        variant_record,
        TrustedEightBitParseContext::Validation,
        true,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_trusted_variant_major_dosage_tile(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_pointer_address: OutputBufferAddress,
    selected_sample_count: usize,
    tile_variant_start_index: usize,
    profiling_enabled: bool,
    validate_sample_ploidy_and_missingness: bool,
    tile_stats: &mut VariantMajorTileStatsMut<'_>,
    thread_scratch: &mut ThreadScratch,
) -> Result<VariantMajorTileDecodeResult, BgenError> {
    validate_variant_major_tile_stats_lengths(tile_stats, variant_record_chunk.len())?;
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
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
            validate_sample_ploidy_and_missingness,
            thread_scratch,
        )?;
        tile_stats.dosage_sum[tile_variant_index] = variant_decode_result.selected_dosage_total;
        tile_stats.dosage_square_sum[tile_variant_index] = variant_decode_result.selected_dosage_square_total;
        tile_stats.observation_count[tile_variant_index] = variant_decode_result.selected_observation_count;
        tile_stats.zero_count[tile_variant_index] = variant_decode_result.zero_count;
        tile_stats.nonzero_count[tile_variant_index] = variant_decode_result.nonzero_count;
        tile_stats.homozygous_reference_count[tile_variant_index] = variant_decode_result.homozygous_reference_count;
        tile_stats.heterozygous_count[tile_variant_index] = variant_decode_result.heterozygous_count;
        tile_stats.homozygous_alternate_count[tile_variant_index] = variant_decode_result.homozygous_alternate_count;
        if profiling_enabled {
            thread_local_profile_snapshot.merge_from(&variant_decode_result.profile_snapshot);
        }
    }
    if profiling_enabled {
        thread_local_profile_snapshot.decode_tile_count += 1;
    }
    Ok(VariantMajorTileDecodeResult { profile_snapshot: thread_local_profile_snapshot, has_missing_values: false })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_trusted_variant_major_packed8_probability_pair_tile(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_pointer_address: OutputBufferAddress,
    selected_sample_count: usize,
    tile_variant_start_index: usize,
    profiling_enabled: bool,
    validate_sample_ploidy_and_missingness: bool,
    tile_stats: &mut VariantMajorTileStatsMut<'_>,
    thread_scratch: &mut ThreadScratch,
) -> Result<VariantMajorTileDecodeResult, BgenError> {
    validate_variant_major_tile_stats_lengths(tile_stats, variant_record_chunk.len())?;
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    for (tile_variant_index, variant_record) in variant_record_chunk.iter().enumerate() {
        let variant_decode_result = decode_trusted_unphased_eight_bit_variant_into_variant_major_probability_pairs(
            mmap,
            compression_type,
            sample_count,
            sample_selection,
            variant_record,
            output_pointer_address,
            tile_variant_start_index + tile_variant_index,
            selected_sample_count,
            profiling_enabled,
            validate_sample_ploidy_and_missingness,
            thread_scratch,
        )?;
        tile_stats.dosage_sum[tile_variant_index] = variant_decode_result.selected_dosage_total;
        tile_stats.dosage_square_sum[tile_variant_index] = variant_decode_result.selected_dosage_square_total;
        tile_stats.observation_count[tile_variant_index] = variant_decode_result.selected_observation_count;
        tile_stats.zero_count[tile_variant_index] = variant_decode_result.zero_count;
        tile_stats.nonzero_count[tile_variant_index] = variant_decode_result.nonzero_count;
        tile_stats.homozygous_reference_count[tile_variant_index] = variant_decode_result.homozygous_reference_count;
        tile_stats.heterozygous_count[tile_variant_index] = variant_decode_result.heterozygous_count;
        tile_stats.homozygous_alternate_count[tile_variant_index] = variant_decode_result.homozygous_alternate_count;
        if profiling_enabled {
            thread_local_profile_snapshot.merge_from(&variant_decode_result.profile_snapshot);
        }
    }
    if profiling_enabled {
        thread_local_profile_snapshot.decode_tile_count += 1;
    }
    Ok(VariantMajorTileDecodeResult { profile_snapshot: thread_local_profile_snapshot, has_missing_values: false })
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn decode_trusted_unphased_eight_bit_variant_into_variant_major_probability_pairs(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: OutputBufferAddress,
    variant_index: usize,
    selected_sample_count: usize,
    profiling_enabled: bool,
    validate_sample_ploidy_and_missingness: bool,
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

    let trusted_block = parse_trusted_unphased_eight_bit_probability_block(
        probability_block,
        sample_count,
        variant_record,
        TrustedEightBitParseContext::VariantMajorPackedProbabilityPairs,
        validate_sample_ploidy_and_missingness,
    )?;
    let packed_probability_bytes = trusted_block.packed_probability_bytes;
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let output_write_start_time = profiling_enabled.then(Instant::now);
    let selected_probability_byte_count = selected_sample_count.checked_mul(2).ok_or_else(|| {
        BgenError::Range("Integer overflow while sizing variant-major packed8 BGEN output row.".to_string())
    })?;
    let mut output_matrix = unsafe {
        VariantMajorOutputMatrix::<u8>::from_pointer_address(
            output_pointer_address,
            selected_probability_byte_count,
            "variant-major packed8 BGEN",
        )?
    };
    let output_row = output_matrix.row_mut(variant_index)?;

    let decode_summary = if sample_selection.is_identity {
        simd::copy_unphased_eight_bit_probability_pairs_and_summarize_simd_or_scalar(
            packed_probability_bytes,
            output_row,
        )
    } else if let Some(contiguous_file_index_start) = sample_selection.contiguous_file_index_start {
        let probability_offset = contiguous_file_index_start.checked_mul(2).ok_or_else(|| {
            BgenError::InvalidFormat("Integer overflow while indexing trusted BGEN probabilities.".to_string())
        })?;
        let selected_probability_bytes =
            read_exact_bytes(packed_probability_bytes, probability_offset, selected_probability_byte_count)?;
        simd::copy_unphased_eight_bit_probability_pairs_and_summarize_simd_or_scalar(
            selected_probability_bytes,
            output_row,
        )
    } else {
        let mut raw_integer_summary = simd::EightBitRawIntegerSummary::default();
        for (selected_index, file_sample_index) in sample_selection.selected_file_indices.iter().copied().enumerate() {
            let probability_offset = file_sample_index.checked_mul(2).ok_or_else(|| {
                BgenError::InvalidFormat("Integer overflow while indexing trusted BGEN probabilities.".to_string())
            })?;
            let probability_pair = read_eight_bit_probability_pair(packed_probability_bytes, probability_offset)?;
            let output_offset = selected_index.checked_mul(2).ok_or_else(|| {
                BgenError::Range("Integer overflow while writing selected packed8 probabilities.".to_string())
            })?;
            output_row[output_offset] = probability_pair[0];
            output_row[output_offset + 1] = probability_pair[1];
            raw_integer_summary.record_probability_pair(probability_pair);
        }
        raw_integer_summary.into_decode_summary()
    };
    if let Some(output_write_start_time) = output_write_start_time {
        thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
        thread_local_profile_snapshot.output_write_count += 1;
        thread_local_profile_snapshot.output_byte_count =
            u64::try_from(selected_probability_byte_count).unwrap_or(u64::MAX);
    }
    if let Some(probability_decode_start_time) = probability_decode_start_time {
        thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
        thread_local_profile_snapshot.probability_decode_count += 1;
    }
    if profiling_enabled {
        thread_local_profile_snapshot.variant_decode_count += 1;
    }
    Ok(VariantDecodeResult {
        profile_snapshot: thread_local_profile_snapshot,
        selected_dosage_total: decode_summary.selected_dosage_total,
        selected_dosage_square_total: decode_summary.selected_dosage_square_total,
        selected_observation_count: decode_summary.selected_observation_count,
        has_missing_values: false,
        zero_count: decode_summary.zero_count,
        nonzero_count: decode_summary.nonzero_count,
        homozygous_reference_count: decode_summary.homozygous_reference_count,
        heterozygous_count: decode_summary.heterozygous_count,
        homozygous_alternate_count: decode_summary.homozygous_alternate_count,
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
    output_pointer_address: OutputBufferAddress,
    variant_index: usize,
    selected_sample_count: usize,
    profiling_enabled: bool,
    validate_sample_ploidy_and_missingness: bool,
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

    let trusted_block = parse_trusted_unphased_eight_bit_probability_block(
        probability_block,
        sample_count,
        variant_record,
        TrustedEightBitParseContext::VariantMajorDosage,
        validate_sample_ploidy_and_missingness,
    )?;
    let packed_probability_bytes = trusted_block.packed_probability_bytes;
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let output_write_start_time = profiling_enabled.then(Instant::now);
    let mut output_matrix = unsafe {
        VariantMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            selected_sample_count,
            "variant-major BGEN",
        )?
    };
    let mut selected_dosage_total = 0.0_f32;
    let mut selected_dosage_square_total = 0.0_f32;
    let mut selected_observation_count = selected_sample_count_to_i32(selected_sample_count)?;
    let mut zero_count = 0_i32;
    let mut nonzero_count = 0_i32;
    let mut homozygous_reference_count = 0_i32;
    let mut heterozygous_count = 0_i32;
    let mut homozygous_alternate_count = 0_i32;
    if sample_selection.is_identity {
        let output_row = output_matrix.row_mut(variant_index)?;
        let decode_summary =
            simd::decode_unphased_eight_bit_identity_simd_or_scalar(packed_probability_bytes, output_row);
        selected_dosage_total = decode_summary.selected_dosage_total;
        selected_dosage_square_total = decode_summary.selected_dosage_square_total;
        selected_observation_count = decode_summary.selected_observation_count;
        zero_count = decode_summary.zero_count;
        nonzero_count = decode_summary.nonzero_count;
        homozygous_reference_count = decode_summary.homozygous_reference_count;
        heterozygous_count = decode_summary.heterozygous_count;
        homozygous_alternate_count = decode_summary.homozygous_alternate_count;
    } else if let Some(contiguous_file_index_start) = sample_selection.contiguous_file_index_start {
        let probability_offset = contiguous_file_index_start.checked_mul(2).ok_or_else(|| {
            BgenError::InvalidFormat("Integer overflow while indexing trusted BGEN probabilities.".to_string())
        })?;
        let selected_probability_byte_count = selected_sample_count.checked_mul(2).ok_or_else(|| {
            BgenError::InvalidFormat("Integer overflow while slicing selected trusted BGEN probabilities.".to_string())
        })?;
        let selected_probability_bytes =
            read_exact_bytes(packed_probability_bytes, probability_offset, selected_probability_byte_count)?;
        let output_row = output_matrix.row_mut(variant_index)?;
        let decode_summary =
            simd::decode_unphased_eight_bit_identity_simd_or_scalar(selected_probability_bytes, output_row);
        selected_dosage_total = decode_summary.selected_dosage_total;
        selected_dosage_square_total = decode_summary.selected_dosage_square_total;
        selected_observation_count = decode_summary.selected_observation_count;
        zero_count = decode_summary.zero_count;
        nonzero_count = decode_summary.nonzero_count;
        homozygous_reference_count = decode_summary.homozygous_reference_count;
        heterozygous_count = decode_summary.heterozygous_count;
        homozygous_alternate_count = decode_summary.homozygous_alternate_count;
    } else {
        let dosage_lookup = unphased_eight_bit_dosage_lookup();
        let output_row = output_matrix.row_mut(variant_index)?;
        for (selected_index, file_sample_index) in sample_selection.selected_file_indices.iter().copied().enumerate() {
            let probability_offset = file_sample_index.checked_mul(2).ok_or_else(|| {
                BgenError::InvalidFormat("Integer overflow while indexing trusted BGEN probabilities.".to_string())
            })?;
            let probability_pair = read_eight_bit_probability_pair(packed_probability_bytes, probability_offset)?;
            let packed_probability_index = packed_eight_bit_probability_index(probability_pair);
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
            output_row[selected_index] = dosage_value;
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
    if profiling_enabled {
        thread_local_profile_snapshot.variant_decode_count += 1;
    }
    Ok(VariantDecodeResult {
        profile_snapshot: thread_local_profile_snapshot,
        selected_dosage_total,
        selected_dosage_square_total,
        selected_observation_count,
        has_missing_values: false,
        zero_count,
        nonzero_count,
        homozygous_reference_count,
        heterozygous_count,
        homozygous_alternate_count,
    })
}

#[cfg(test)]
mod tests {
    use super::super::metadata::VariantRecord;
    use super::super::sample_selection::build_sample_selection;
    use super::*;

    fn trusted_probability_block(
        sample_count: u32,
        allele_count: u16,
        minimum_ploidy: u8,
        maximum_ploidy: u8,
        sample_ploidy_and_missingness: &[u8],
        phased_flag: u8,
        probability_bit_count: u8,
        probability_bytes: &[u8],
    ) -> Vec<u8> {
        let mut block = Vec::new();
        block.extend_from_slice(&sample_count.to_le_bytes());
        block.extend_from_slice(&allele_count.to_le_bytes());
        block.push(minimum_ploidy);
        block.push(maximum_ploidy);
        block.extend_from_slice(sample_ploidy_and_missingness);
        block.push(phased_flag);
        block.push(probability_bit_count);
        block.extend_from_slice(probability_bytes);
        block
    }

    fn valid_trusted_probability_block(probability_bytes: &[u8]) -> Vec<u8> {
        trusted_probability_block(3, 2, 2, 2, &[2, 2, 2], 0, 8, probability_bytes)
    }

    fn variant_record(offset: usize, length: usize, identifier: &str) -> VariantRecord {
        VariantRecord {
            probability_payload_offset: offset,
            probability_payload_length: length,
            declared_uncompressed_block_length: length,
            chromosome: "22".to_string(),
            resolved_variant_identifier: identifier.to_string(),
            position: 1,
            counted_allele: "A".to_string(),
            reference_allele: "G".to_string(),
        }
    }

    #[test]
    fn trusted_variant_major_decode_writes_selected_samples_and_counts() {
        let first_block = valid_trusted_probability_block(&[0, 0, 255, 0, 0, 255]);
        let second_block = valid_trusted_probability_block(&[0, 255, 255, 0, 0, 0]);
        let second_offset = first_block.len();
        let mut mmap = first_block.clone();
        mmap.extend_from_slice(&second_block);
        let variant_records = [
            variant_record(0, first_block.len(), "trusted-first"),
            variant_record(second_offset, second_block.len(), "trusted-second"),
        ];
        let sample_selection = build_sample_selection(3, &[1, 2]).expect("subset sample selection should build");
        let mut output = vec![f32::NAN; 4];
        let mut thread_scratch = ThreadScratch::default();
        let mut dosage_sum = vec![0.0_f32; 2];
        let mut dosage_square_sum = vec![0.0_f32; 2];
        let mut observation_count = vec![0_i32; 2];
        let mut zero_count = vec![0_i32; 2];
        let mut nonzero_count = vec![0_i32; 2];
        let mut homozygous_reference_count = vec![0_i32; 2];
        let mut heterozygous_count = vec![0_i32; 2];
        let mut homozygous_alternate_count = vec![0_i32; 2];

        let mut tile_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            zero_count: &mut zero_count,
            nonzero_count: &mut nonzero_count,
            homozygous_reference_count: &mut homozygous_reference_count,
            heterozygous_count: &mut heterozygous_count,
            homozygous_alternate_count: &mut homozygous_alternate_count,
        };
        let result = decode_trusted_variant_major_dosage_tile(
            &mmap,
            CompressionType::None,
            3,
            &sample_selection,
            &variant_records,
            OutputBufferAddress::from_mut_ptr(output.as_mut_ptr()),
            2,
            0,
            true,
            true,
            &mut tile_stats,
            &mut thread_scratch,
        )
        .expect("trusted variant-major tile should decode");
        let dosage_lookup = unphased_eight_bit_dosage_lookup();

        assert_eq!(observation_count, vec![2, 2]);
        assert!(!result.has_missing_values);
        assert_eq!(result.profile_snapshot.variant_decode_count, 2);
        assert_eq!(result.profile_snapshot.decode_tile_count, 1);
        assert!((output[0] - dosage_lookup[usize::from(255_u8)]).abs() < f32::EPSILON);
        assert!((output[1] - dosage_lookup[usize::from(0_u8) | (usize::from(255_u8) << 8)]).abs() < f32::EPSILON);
        assert!((output[2] - dosage_lookup[usize::from(255_u8)]).abs() < f32::EPSILON);
        assert!((output[3] - dosage_lookup[0]).abs() < f32::EPSILON);
        assert_eq!(nonzero_count.len(), 2);

        let mut disabled_output = vec![f32::NAN; 4];
        let mut disabled_dosage_sum = vec![0.0_f32; 2];
        let mut disabled_dosage_square_sum = vec![0.0_f32; 2];
        let mut disabled_observation_count = vec![0_i32; 2];
        let mut disabled_zero_count = vec![0_i32; 2];
        let mut disabled_nonzero_count = vec![0_i32; 2];
        let mut disabled_homozygous_reference_count = vec![0_i32; 2];
        let mut disabled_heterozygous_count = vec![0_i32; 2];
        let mut disabled_homozygous_alternate_count = vec![0_i32; 2];
        let mut disabled_tile_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut disabled_dosage_sum,
            dosage_square_sum: &mut disabled_dosage_square_sum,
            observation_count: &mut disabled_observation_count,
            zero_count: &mut disabled_zero_count,
            nonzero_count: &mut disabled_nonzero_count,
            homozygous_reference_count: &mut disabled_homozygous_reference_count,
            heterozygous_count: &mut disabled_heterozygous_count,
            homozygous_alternate_count: &mut disabled_homozygous_alternate_count,
        };
        let disabled_result = decode_trusted_variant_major_dosage_tile(
            &mmap,
            CompressionType::None,
            3,
            &sample_selection,
            &variant_records,
            OutputBufferAddress::from_mut_ptr(disabled_output.as_mut_ptr()),
            2,
            0,
            false,
            true,
            &mut disabled_tile_stats,
            &mut thread_scratch,
        )
        .expect("trusted variant-major tile should decode without profiling");
        assert_eq!(disabled_result.profile_snapshot, ThreadLocalProfileSnapshot::default());
        assert_eq!(disabled_observation_count, vec![2, 2]);
        assert_eq!(disabled_output, output);
    }

    #[test]
    fn trusted_variant_major_decode_covers_identity_and_noncontiguous_selected_samples() {
        let block = valid_trusted_probability_block(&[0, 0, 255, 0, 0, 255]);
        let variant_records = [variant_record(0, block.len(), "trusted-single")];
        let dosage_lookup = unphased_eight_bit_dosage_lookup();

        let identity_selection = build_sample_selection(3, &[0, 1, 2]).expect("identity sample selection should build");
        let mut identity_output = vec![f32::NAN; 3];
        let mut thread_scratch = ThreadScratch::default();
        let mut dosage_sum = vec![0.0_f32; 1];
        let mut dosage_square_sum = vec![0.0_f32; 1];
        let mut observation_count = vec![0_i32; 1];
        let mut zero_count = vec![0_i32; 1];
        let mut nonzero_count = vec![0_i32; 1];
        let mut homozygous_reference_count = vec![0_i32; 1];
        let mut heterozygous_count = vec![0_i32; 1];
        let mut homozygous_alternate_count = vec![0_i32; 1];
        let mut identity_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            zero_count: &mut zero_count,
            nonzero_count: &mut nonzero_count,
            homozygous_reference_count: &mut homozygous_reference_count,
            heterozygous_count: &mut heterozygous_count,
            homozygous_alternate_count: &mut homozygous_alternate_count,
        };
        decode_trusted_variant_major_dosage_tile(
            &block,
            CompressionType::None,
            3,
            &identity_selection,
            &variant_records,
            OutputBufferAddress::from_mut_ptr(identity_output.as_mut_ptr()),
            3,
            0,
            true,
            true,
            &mut identity_stats,
            &mut thread_scratch,
        )
        .expect("trusted identity decode should succeed");
        assert_eq!(observation_count, vec![3]);
        assert_eq!(identity_output, vec![2.0, 0.0, 1.0]);

        let noncontiguous_selection =
            build_sample_selection(3, &[2, 0]).expect("non-contiguous sample selection should build");
        let mut noncontiguous_output = vec![f32::NAN; 2];
        let mut dosage_sum = vec![0.0_f32; 1];
        let mut dosage_square_sum = vec![0.0_f32; 1];
        let mut observation_count = vec![0_i32; 1];
        let mut zero_count = vec![0_i32; 1];
        let mut nonzero_count = vec![0_i32; 1];
        let mut homozygous_reference_count = vec![0_i32; 1];
        let mut heterozygous_count = vec![0_i32; 1];
        let mut homozygous_alternate_count = vec![0_i32; 1];
        let mut noncontiguous_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            zero_count: &mut zero_count,
            nonzero_count: &mut nonzero_count,
            homozygous_reference_count: &mut homozygous_reference_count,
            heterozygous_count: &mut heterozygous_count,
            homozygous_alternate_count: &mut homozygous_alternate_count,
        };
        decode_trusted_variant_major_dosage_tile(
            &block,
            CompressionType::None,
            3,
            &noncontiguous_selection,
            &variant_records,
            OutputBufferAddress::from_mut_ptr(noncontiguous_output.as_mut_ptr()),
            2,
            0,
            true,
            true,
            &mut noncontiguous_stats,
            &mut thread_scratch,
        )
        .expect("trusted non-contiguous decode should succeed");
        assert_eq!(observation_count, vec![2]);
        assert!(
            (noncontiguous_output[0] - dosage_lookup[usize::from(0_u8) | (usize::from(255_u8) << 8)]).abs()
                < f32::EPSILON
        );
        assert!((noncontiguous_output[1] - dosage_lookup[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn trusted_packed8_variant_major_decode_covers_identity_and_contiguous_selected_samples() {
        let probability_bytes = [0_u8, 0, 255, 0, 0, 255, 128, 0];
        let block = trusted_probability_block(4, 2, 2, 2, &[2, 2, 2, 2], 0, 8, &probability_bytes);
        let variant_records = [variant_record(0, block.len(), "trusted-packed8")];

        let identity_selection = build_sample_selection(4, &[0, 1, 2, 3]).expect("identity selection should build");
        let mut identity_output = vec![0_u8; probability_bytes.len()];
        let mut thread_scratch = ThreadScratch::default();
        let mut dosage_sum = vec![0.0_f32; 1];
        let mut dosage_square_sum = vec![0.0_f32; 1];
        let mut observation_count = vec![0_i32; 1];
        let mut zero_count = vec![0_i32; 1];
        let mut nonzero_count = vec![0_i32; 1];
        let mut homozygous_reference_count = vec![0_i32; 1];
        let mut heterozygous_count = vec![0_i32; 1];
        let mut homozygous_alternate_count = vec![0_i32; 1];
        let mut identity_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            zero_count: &mut zero_count,
            nonzero_count: &mut nonzero_count,
            homozygous_reference_count: &mut homozygous_reference_count,
            heterozygous_count: &mut heterozygous_count,
            homozygous_alternate_count: &mut homozygous_alternate_count,
        };
        decode_trusted_variant_major_packed8_probability_pair_tile(
            &block,
            CompressionType::None,
            4,
            &identity_selection,
            &variant_records,
            OutputBufferAddress::from_mut_ptr(identity_output.as_mut_ptr()),
            4,
            0,
            true,
            true,
            &mut identity_stats,
            &mut thread_scratch,
        )
        .expect("trusted packed8 identity decode should succeed");

        assert_eq!(identity_output, probability_bytes);
        assert_eq!(observation_count, vec![4]);
        assert_eq!(zero_count, vec![1]);
        assert_eq!(nonzero_count, vec![3]);
        assert_eq!(homozygous_reference_count, vec![1]);
        assert_eq!(heterozygous_count, vec![2]);
        assert_eq!(homozygous_alternate_count, vec![1]);

        let contiguous_selection = build_sample_selection(4, &[1, 2]).expect("contiguous selection should build");
        let mut contiguous_output = vec![0_u8; 4];
        let mut dosage_sum = vec![0.0_f32; 1];
        let mut dosage_square_sum = vec![0.0_f32; 1];
        let mut observation_count = vec![0_i32; 1];
        let mut zero_count = vec![0_i32; 1];
        let mut nonzero_count = vec![0_i32; 1];
        let mut homozygous_reference_count = vec![0_i32; 1];
        let mut heterozygous_count = vec![0_i32; 1];
        let mut homozygous_alternate_count = vec![0_i32; 1];
        let mut contiguous_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            zero_count: &mut zero_count,
            nonzero_count: &mut nonzero_count,
            homozygous_reference_count: &mut homozygous_reference_count,
            heterozygous_count: &mut heterozygous_count,
            homozygous_alternate_count: &mut homozygous_alternate_count,
        };
        decode_trusted_variant_major_packed8_probability_pair_tile(
            &block,
            CompressionType::None,
            4,
            &contiguous_selection,
            &variant_records,
            OutputBufferAddress::from_mut_ptr(contiguous_output.as_mut_ptr()),
            2,
            0,
            true,
            true,
            &mut contiguous_stats,
            &mut thread_scratch,
        )
        .expect("trusted packed8 contiguous decode should succeed");

        assert_eq!(contiguous_output, vec![255, 0, 0, 255]);
        assert_eq!(observation_count, vec![2]);
        assert_eq!(zero_count, vec![1]);
        assert_eq!(nonzero_count, vec![1]);
        assert_eq!(homozygous_reference_count, vec![1]);
        assert_eq!(heterozygous_count, vec![1]);
        assert_eq!(homozygous_alternate_count, vec![0]);
        assert!((dosage_sum[0] - 1.0).abs() < f32::EPSILON);
        assert!((dosage_square_sum[0] - 1.0).abs() < f32::EPSILON);
    }
}
