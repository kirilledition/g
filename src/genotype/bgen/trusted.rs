use std::time::Instant;

use super::decode::{
    ThreadScratch, VariantDecodeResult, VariantMajorTileDecodeResult, VariantMajorTileStatsMut, read_exact_bytes,
    read_probability_block, read_u8_at, read_u16_at, read_u32_at, u32_to_usize, unphased_eight_bit_dosage_lookup,
    validate_variant_major_tile_stats_lengths,
};
use super::metadata::VariantRecord;
use super::profile::{ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use super::sample_selection::SampleSelection;
use super::simd;
use super::{BgenError, CompressionType};
use crate::genotype::preprocess;

pub(super) fn all_samples_present_diploid(sample_ploidy_and_missingness: &[u8]) -> bool {
    simd::all_samples_present_diploid_simd_or_scalar(sample_ploidy_and_missingness)
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
            thread_scratch,
        )?;
        let variant_profile_snapshot = variant_decode_result.profile_snapshot;
        tile_stats.dosage_sum[tile_variant_index] = variant_decode_result.selected_dosage_total;
        tile_stats.dosage_square_sum[tile_variant_index] = variant_decode_result.selected_dosage_square_total;
        tile_stats.observation_count[tile_variant_index] = variant_decode_result.selected_observation_count;
        tile_stats.zero_count[tile_variant_index] = variant_decode_result.zero_count;
        tile_stats.nonzero_count[tile_variant_index] = variant_decode_result.nonzero_count;
        tile_stats.homozygous_reference_count[tile_variant_index] = variant_decode_result.homozygous_reference_count;
        tile_stats.heterozygous_count[tile_variant_index] = variant_decode_result.heterozygous_count;
        tile_stats.homozygous_alternate_count[tile_variant_index] = variant_decode_result.homozygous_alternate_count;
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
    Ok(VariantMajorTileDecodeResult { profile_snapshot: thread_local_profile_snapshot, has_missing_values: false })
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

    read_exact_bytes(probability_block, cursor, sample_count)?;
    cursor += sample_count;

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
    let output_pointer = output_pointer_address as *mut f32;
    let variant_row_offset = variant_index.checked_mul(selected_sample_count).ok_or_else(|| {
        BgenError::Range("Integer overflow while locating variant-major BGEN output row.".to_string())
    })?;
    let mut selected_dosage_total = 0.0_f32;
    let mut selected_dosage_square_total = 0.0_f32;
    let mut selected_observation_count = i32::try_from(selected_sample_count).unwrap_or(i32::MAX);
    let mut zero_count = 0_i32;
    let mut nonzero_count = 0_i32;
    let mut homozygous_reference_count = 0_i32;
    let mut heterozygous_count = 0_i32;
    let mut homozygous_alternate_count = 0_i32;
    if sample_selection.is_identity {
        let output_row = unsafe {
            // Each parallel worker owns a distinct variant row in the variant-major output matrix.
            std::slice::from_raw_parts_mut(output_pointer.add(variant_row_offset), selected_sample_count)
        };
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
        let output_row = unsafe {
            // The contiguous selected sample run maps directly to a contiguous output row.
            std::slice::from_raw_parts_mut(output_pointer.add(variant_row_offset), selected_sample_count)
        };
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
        for (selected_index, file_sample_index) in sample_selection.selected_file_indices.iter().copied().enumerate() {
            let probability_offset = file_sample_index.checked_mul(2).ok_or_else(|| {
                BgenError::InvalidFormat("Integer overflow while indexing trusted BGEN probabilities.".to_string())
            })?;
            let probability_pair = read_exact_bytes(packed_probability_bytes, probability_offset, 2)?;
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
                // Selected sample order maps directly to the caller's output row order.
                output_pointer.add(variant_row_offset + selected_index).write(dosage_value);
            }
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

    fn validation_error_for(block: &[u8], expected_sample_count: usize) -> String {
        let mut thread_scratch = ThreadScratch::default();
        let mut profile_snapshot = ThreadLocalProfileSnapshot::default();
        validate_variant_compatible_with_trusted_no_missing_diploid(
            block,
            CompressionType::None,
            &variant_record(0, block.len(), "trusted-test"),
            expected_sample_count,
            &mut thread_scratch,
            &mut profile_snapshot,
        )
        .expect_err("trusted validation should reject malformed block")
        .to_string()
    }

    #[test]
    fn trusted_validation_accepts_contract_and_rejects_each_assumption() {
        assert!(all_samples_present_diploid(&[2; 17]));
        assert!(!all_samples_present_diploid(&[2, 2, 3]));
        assert!(!all_samples_present_diploid(&[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2]));

        let valid_block = valid_trusted_probability_block(&[0, 0, 255, 0, 0, 255]);
        let mut thread_scratch = ThreadScratch::default();
        let mut profile_snapshot = ThreadLocalProfileSnapshot::default();
        validate_variant_compatible_with_trusted_no_missing_diploid(
            &valid_block,
            CompressionType::None,
            &variant_record(0, valid_block.len(), "trusted-valid"),
            3,
            &mut thread_scratch,
            &mut profile_snapshot,
        )
        .expect("valid trusted probability block should pass");

        assert!(validation_error_for(&valid_block, 2).contains("file header reports"));
        assert!(
            validation_error_for(&trusted_probability_block(3, 3, 2, 2, &[2, 2, 2], 0, 8, &[]), 3)
                .contains("not biallelic")
        );
        assert!(
            validation_error_for(&trusted_probability_block(3, 2, 1, 2, &[2, 2, 2], 0, 8, &[]), 3)
                .contains("ploidy bounds")
        );
        assert!(
            validation_error_for(&trusted_probability_block(3, 2, 2, 2, &[2, 0x82, 2], 0, 8, &[]), 3)
                .contains("missing or non-diploid")
        );
        assert!(
            validation_error_for(&trusted_probability_block(3, 2, 2, 2, &[2, 2, 2], 1, 8, &[]), 3).contains("phased")
        );
        assert!(
            validation_error_for(&trusted_probability_block(3, 2, 2, 2, &[2, 2, 2], 0, 16, &[]), 3)
                .contains("bits per probability")
        );
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
            output.as_mut_ptr() as usize,
            2,
            0,
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
    }

    #[test]
    fn trusted_variant_major_decode_reports_contract_violations() {
        let sample_selection = build_sample_selection(3, &[0, 1, 2]).expect("identity sample selection should build");
        let mut output = vec![0.0_f32; 3];

        for (block, expected_message) in [
            (trusted_probability_block(2, 2, 2, 2, &[2, 2, 2], 0, 8, &[]), "file header reports"),
            (trusted_probability_block(3, 3, 2, 2, &[2, 2, 2], 0, 8, &[]), "not biallelic"),
            (trusted_probability_block(3, 2, 1, 2, &[2, 2, 2], 0, 8, &[]), "ploidy bounds"),
            (trusted_probability_block(3, 2, 2, 2, &[2, 2, 2], 1, 8, &[]), "unphased 8-bit"),
            (trusted_probability_block(3, 2, 2, 2, &[2, 2, 2], 0, 4, &[]), "unphased 8-bit"),
            (trusted_probability_block(3, 2, 2, 2, &[2, 2, 2], 0, 8, &[0, 0]), "Unexpected end"),
        ] {
            let mut thread_scratch = ThreadScratch::default();
            let mut dosage_sum = vec![0.0_f32; 1];
            let mut dosage_square_sum = vec![0.0_f32; 1];
            let mut observation_count = vec![0_i32; 1];
            let mut zero_count = vec![0_i32; 1];
            let mut nonzero_count = vec![0_i32; 1];
            let mut homozygous_reference_count = vec![0_i32; 1];
            let mut heterozygous_count = vec![0_i32; 1];
            let mut homozygous_alternate_count = vec![0_i32; 1];
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
            let error = decode_trusted_variant_major_dosage_tile(
                &block,
                CompressionType::None,
                3,
                &sample_selection,
                &[variant_record(0, block.len(), "trusted-invalid")],
                output.as_mut_ptr() as usize,
                3,
                0,
                false,
                &mut tile_stats,
                &mut thread_scratch,
            )
            .expect_err("invalid trusted block should fail")
            .to_string();
            assert!(error.contains(expected_message), "expected '{expected_message}' in '{error}'");
        }
    }
}
