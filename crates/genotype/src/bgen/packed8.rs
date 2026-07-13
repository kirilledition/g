use std::mem::MaybeUninit;

use super::decode::{
    ThreadScratch, VariantDecodeFailure, VariantMajorTileStatsMut, parse_layout_two_probability_block,
    read_eight_bit_probability_pair, read_exact_bytes, read_probability_block, selected_sample_count_to_i32,
    validate_layout_two_probability_values, validate_variant_major_tile_stats_lengths,
};
use super::metadata::VariantRecord;
use super::sample_selection::SampleSelection;
use super::simd;
use super::{BgenError, CompressionType};
use crate::common::{DosageSummary, Packed8Compatibility};

fn exact_eight_bit_probability_bytes(probability_bytes: &[u8], sample_count: usize) -> Result<&[u8], BgenError> {
    let expected_probability_byte_count = sample_count.checked_mul(2).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while decoding 8-bit BGEN probabilities.".to_string())
    })?;
    if probability_bytes.len() != expected_probability_byte_count {
        return Err(BgenError::InvalidFormat(format!(
            "contains {} probability bytes, but an unphased 8-bit diploid record with {sample_count} samples requires exactly {expected_probability_byte_count}.",
            probability_bytes.len(),
        )));
    }
    Ok(probability_bytes)
}

fn packed8_probability_bytes(probability_block: &[u8], sample_count: usize) -> Result<&[u8], BgenError> {
    let parsed_probability_block = parse_layout_two_probability_block(probability_block, sample_count)?;
    if parsed_probability_block.minimum_ploidy != 2 || parsed_probability_block.maximum_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "uses ploidy bounds [{}, {}], but packed8 reads require diploid variants.",
            parsed_probability_block.minimum_ploidy, parsed_probability_block.maximum_ploidy,
        )));
    }
    if parsed_probability_block.phased_flag != 0 || parsed_probability_block.probability_bit_count != 8 {
        return Err(BgenError::UnsupportedFormat("is not an unphased 8-bit BGEN variant.".to_string()));
    }
    debug_assert!(
        simd::all_samples_present_diploid_simd_or_scalar(parsed_probability_block.sample_ploidy_and_missingness),
        "packed8 decode requires a completed compatibility scan"
    );

    exact_eight_bit_probability_bytes(parsed_probability_block.probability_bytes, sample_count)
}

pub(super) fn validate_variant_compatible_with_packed8(
    mmap: &[u8],
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    sample_count: usize,
    thread_scratch: &mut ThreadScratch,
) -> Result<Packed8Compatibility, BgenError> {
    let probability_block = read_probability_block(mmap, compression_type, variant_record, thread_scratch)?;
    let parsed_probability_block = parse_layout_two_probability_block(probability_block, sample_count)?;
    let has_missing_samples = validate_layout_two_probability_values(&parsed_probability_block, sample_count)?;
    let requires_dosage = has_missing_samples
        || parsed_probability_block.phased_flag != 0
        || parsed_probability_block.probability_bit_count != 8;

    Ok(if requires_dosage { Packed8Compatibility::RequiresDosage } else { Packed8Compatibility::Compatible })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_variant_major_probability_pair_tile(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_values: &mut [MaybeUninit<u8>],
    tile_variant_start_index: usize,
    tile_stats: &mut VariantMajorTileStatsMut<'_>,
    thread_scratch: &mut ThreadScratch,
) -> Result<(), VariantDecodeFailure> {
    validate_variant_major_tile_stats_lengths(tile_stats, variant_record_chunk.len())
        .map_err(|source| VariantDecodeFailure { relative_variant_index: None, source })?;
    let selected_sample_count = sample_selection.selected_sample_count();
    selected_sample_count_to_i32(selected_sample_count)
        .map_err(|source| VariantDecodeFailure { relative_variant_index: None, source })?;
    let selected_probability_byte_count = selected_sample_count.checked_mul(2).ok_or_else(|| VariantDecodeFailure {
        relative_variant_index: None,
        source: BgenError::Range("Integer overflow while sizing packed8 BGEN output row.".to_string()),
    })?;
    let expected_output_value_count = variant_record_chunk
        .len()
        .checked_mul(selected_probability_byte_count)
        .ok_or_else(|| VariantDecodeFailure {
            relative_variant_index: None,
            source: BgenError::Range("Integer overflow while sizing packed8 BGEN output tile.".to_string()),
        })?;
    if output_values.len() != expected_output_value_count {
        return Err(VariantDecodeFailure {
            relative_variant_index: None,
            source: BgenError::Range(format!(
                "Packed8 BGEN output tile requires {expected_output_value_count} bytes, observed {}.",
                output_values.len(),
            )),
        });
    }
    if selected_probability_byte_count == 0 {
        return Ok(());
    }
    let collect_sparse_candidate_counts = tile_stats.sparse_candidate_counts.is_some();
    for ((tile_variant_index, variant_record), output_row) in
        variant_record_chunk.iter().enumerate().zip(output_values.chunks_exact_mut(selected_probability_byte_count))
    {
        let variant_decode_result = decode_unphased_eight_bit_variant_into_variant_major_probability_pairs(
            mmap,
            compression_type,
            sample_count,
            sample_selection,
            variant_record,
            output_row,
            collect_sparse_candidate_counts,
            thread_scratch,
        )
        .map_err(|source| VariantDecodeFailure {
            relative_variant_index: Some(tile_variant_start_index + tile_variant_index),
            source,
        })?;
        tile_stats.dosage_sum[tile_variant_index] = variant_decode_result.dosage_sum;
        tile_stats.dosage_square_sum[tile_variant_index] = variant_decode_result.dosage_square_sum;
        tile_stats.observation_count[tile_variant_index] = variant_decode_result.observation_count;
        if let Some(sparse_candidate_counts) = tile_stats.sparse_candidate_counts.as_mut() {
            sparse_candidate_counts.zero_count[tile_variant_index] = variant_decode_result.zero_count;
            sparse_candidate_counts.homozygous_alternate_count[tile_variant_index] =
                variant_decode_result.homozygous_alternate_count;
        }
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn decode_unphased_eight_bit_variant_into_variant_major_probability_pairs(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_row: &mut [MaybeUninit<u8>],
    collect_sparse_candidate_counts: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<DosageSummary, BgenError> {
    let probability_block = read_probability_block(mmap, compression_type, variant_record, thread_scratch)?;

    let packed_probability_bytes = packed8_probability_bytes(probability_block, sample_count)?;
    let selected_probability_byte_count = output_row.len();
    if !selected_probability_byte_count.is_multiple_of(2) {
        return Err(BgenError::Range(
            "Variant-major packed8 BGEN output row must contain two bytes per sample.".to_string(),
        ));
    }

    let decode_summary = if sample_selection.is_identity() {
        simd::copy_unphased_eight_bit_probability_pairs_and_summarize_simd_or_scalar(
            packed_probability_bytes,
            output_row,
            collect_sparse_candidate_counts,
        )
    } else if let Some(contiguous_file_index_start) = sample_selection.contiguous_file_index_start() {
        let probability_offset = contiguous_file_index_start.checked_mul(2).ok_or_else(|| {
            BgenError::InvalidFormat("Integer overflow while indexing packed8 BGEN probabilities.".to_string())
        })?;
        let selected_probability_bytes =
            read_exact_bytes(packed_probability_bytes, probability_offset, selected_probability_byte_count)?;
        simd::copy_unphased_eight_bit_probability_pairs_and_summarize_simd_or_scalar(
            selected_probability_bytes,
            output_row,
            collect_sparse_candidate_counts,
        )
    } else {
        let mut raw_integer_summary = simd::EightBitRawIntegerSummary::new(collect_sparse_candidate_counts);
        let selected_file_indices = sample_selection
            .indexed_file_indices()
            .expect("non-identity, non-contiguous sample selections store explicit file indices");
        for (selected_index, file_sample_index) in selected_file_indices.iter().copied().enumerate() {
            let probability_offset = file_sample_index.checked_mul(2).ok_or_else(|| {
                BgenError::InvalidFormat("Integer overflow while indexing packed8 BGEN probabilities.".to_string())
            })?;
            let probability_pair = read_eight_bit_probability_pair(packed_probability_bytes, probability_offset)?;
            let output_offset = selected_index.checked_mul(2).ok_or_else(|| {
                BgenError::Range("Integer overflow while writing selected packed8 probabilities.".to_string())
            })?;
            output_row[output_offset].write(probability_pair[0]);
            output_row[output_offset + 1].write(probability_pair[1]);
            raw_integer_summary.record_probability_pair(probability_pair);
        }
        raw_integer_summary.into_decode_summary()
    };
    Ok(decode_summary)
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
