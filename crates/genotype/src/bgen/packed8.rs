use std::mem::MaybeUninit;

use super::decode::{
    ThreadScratch, VariantDecodeFailure, VariantMajorTileStatsMut, parse_layout_two_probability_block,
    read_eight_bit_probability_pair, read_exact_bytes, read_probability_block, selected_sample_count_to_i32,
    validate_layout_two_probability_values, validate_variant_major_tile_stats_lengths,
};
use super::metadata::VariantRecord;
use super::sample_selection::SampleSelection;
use super::simd;
use super::source::BgenByteWindow;
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
    source_window: BgenByteWindow<'_>,
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    sample_count: usize,
    thread_scratch: &mut ThreadScratch,
) -> Result<Packed8Compatibility, BgenError> {
    let probability_block = read_probability_block(source_window, compression_type, variant_record, thread_scratch)?;
    let parsed_probability_block = parse_layout_two_probability_block(probability_block, sample_count)?;
    let has_missing_samples = validate_layout_two_probability_values(&parsed_probability_block, sample_count)?;
    let requires_dosage = has_missing_samples
        || parsed_probability_block.phased_flag != 0
        || parsed_probability_block.probability_bit_count != 8;

    Ok(if requires_dosage { Packed8Compatibility::RequiresDosage } else { Packed8Compatibility::Compatible })
}

// Keeping this hot, lightweight tile boundary flat avoids passing a request
// aggregate indirectly; the wrapper measured about 2.3% slower on chr22.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_variant_major_probability_pair_tile(
    source_window: BgenByteWindow<'_>,
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
    let collect_sparse_candidate_statistics = tile_stats.sparse_candidate_statistics.is_some();
    for ((tile_variant_index, variant_record), output_row) in
        variant_record_chunk.iter().enumerate().zip(output_values.chunks_exact_mut(selected_probability_byte_count))
    {
        let variant_decode_result = decode_unphased_eight_bit_variant_into_variant_major_probability_pairs(
            source_window,
            compression_type,
            sample_count,
            sample_selection,
            variant_record,
            output_row,
            collect_sparse_candidate_statistics,
            thread_scratch,
        )
        .map_err(|source| VariantDecodeFailure {
            relative_variant_index: Some(tile_variant_start_index + tile_variant_index),
            source,
        })?;
        tile_stats.dosage_sum[tile_variant_index] = variant_decode_result.dosage_sum;
        tile_stats.dosage_square_sum[tile_variant_index] = variant_decode_result.dosage_square_sum;
        tile_stats.observation_count[tile_variant_index] = variant_decode_result.observation_count;
        if let Some(sparse_candidate_statistics) = tile_stats.sparse_candidate_statistics.as_mut() {
            sparse_candidate_statistics[tile_variant_index].exact_dosage_sum =
                variant_decode_result.exact_dosage_sum.ok_or_else(|| VariantDecodeFailure {
                    relative_variant_index: Some(tile_variant_start_index + tile_variant_index),
                    source: BgenError::Range(
                        "Sparse candidate packed8 decoding did not produce an exact dosage sum.".to_string(),
                    ),
                })?;
            sparse_candidate_statistics[tile_variant_index].zero_count = variant_decode_result.zero_count;
            sparse_candidate_statistics[tile_variant_index].homozygous_alternate_count =
                variant_decode_result.homozygous_alternate_count;
        }
    }
    Ok(())
}

fn decode_unphased_eight_bit_variant_into_variant_major_probability_pairs(
    source_window: BgenByteWindow<'_>,
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_row: &mut [MaybeUninit<u8>],
    collect_sparse_candidate_statistics: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<DosageSummary, BgenError> {
    let probability_block = read_probability_block(source_window, compression_type, variant_record, thread_scratch)?;

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
            collect_sparse_candidate_statistics,
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
            collect_sparse_candidate_statistics,
        )
    } else {
        let mut raw_integer_summary = simd::EightBitRawIntegerSummary::new(collect_sparse_candidate_statistics);
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
    use super::super::decode::{VariantMajorTileDecodeRequest, decode_variant_major_dosage_tile};
    use super::super::metadata::VariantRecord;
    use super::super::sample_selection::build_sample_selection;
    use super::*;
    use crate::common::{ExactDosageSum, SparseCandidateSummary};

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

    fn variant_record(offset: usize, length: usize) -> VariantRecord {
        VariantRecord {
            probability_payload_offset: u64::try_from(offset).expect("test probability payload offset should fit u64"),
            probability_payload_length: u32::try_from(length).expect("test probability payload length should fit u32"),
            declared_uncompressed_block_length: u32::try_from(length)
                .expect("test probability block length should fit u32"),
        }
    }

    fn dosage_output_with_sentinels(value_count: usize) -> Vec<MaybeUninit<f32>> {
        vec![MaybeUninit::new(f32::NAN); value_count]
    }

    fn probability_output_with_sentinels(value_count: usize) -> Vec<MaybeUninit<u8>> {
        vec![MaybeUninit::new(0xA5); value_count]
    }

    fn initialized_f32_values(values: Vec<MaybeUninit<f32>>) -> Vec<f32> {
        values
            .into_iter()
            .map(|value| {
                // SAFETY: every test output slot is initialized with a NaN
                // sentinel before decode, independently of decoder writes.
                unsafe { value.assume_init() }
            })
            .collect()
    }

    fn initialized_u8_values(values: Vec<MaybeUninit<u8>>) -> Vec<u8> {
        values
            .into_iter()
            .map(|value| {
                // SAFETY: every test output slot is initialized with a byte
                // sentinel before decode, independently of decoder writes.
                unsafe { value.assume_init() }
            })
            .collect()
    }

    fn expect_decode_success(result: Result<(), VariantDecodeFailure>, message: &str) {
        if let Err(failure) = result {
            panic!("{message}: {}", failure.source);
        }
    }

    #[test]
    fn trusted_variant_major_decode_writes_selected_samples_and_counts() {
        let first_block = valid_trusted_probability_block(&[0, 0, 255, 0, 0, 255]);
        let second_block = valid_trusted_probability_block(&[0, 255, 255, 0, 0, 0]);
        let second_offset = first_block.len();
        let mut source_bytes = first_block.clone();
        source_bytes.extend_from_slice(&second_block);
        let variant_records = [variant_record(0, first_block.len()), variant_record(second_offset, second_block.len())];
        let sample_selection = build_sample_selection(3, &[1, 2]).expect("subset sample selection should build");
        let mut output = dosage_output_with_sentinels(4);
        let mut thread_scratch = ThreadScratch::default();
        let mut dosage_sum = vec![0.0_f32; 2];
        let mut dosage_square_sum = vec![0.0_f32; 2];
        let mut observation_count = vec![0_i32; 2];
        let mut sparse_candidate_statistics = vec![SparseCandidateSummary::default(); 2];

        let mut tile_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            sparse_candidate_statistics: Some(&mut sparse_candidate_statistics),
        };
        expect_decode_success(
            decode_variant_major_dosage_tile(
                VariantMajorTileDecodeRequest {
                    source_window: BgenByteWindow::from_bytes(&source_bytes),
                    compression_type: CompressionType::None,
                    sample_count: 3,
                    sample_selection: &sample_selection,
                    variant_records: &variant_records,
                    tile_variant_start_index: 0,
                },
                &mut output,
                &mut tile_stats,
                &mut thread_scratch,
            ),
            "trusted variant-major tile should decode",
        );
        let output = initialized_f32_values(output);

        assert_eq!(observation_count, vec![2, 2]);
        assert_eq!(output, vec![0.0, 1.0, 0.0, 2.0]);
        assert_eq!(
            sparse_candidate_statistics,
            vec![
                SparseCandidateSummary {
                    exact_dosage_sum: ExactDosageSum::new(255, 255),
                    zero_count: 1,
                    homozygous_alternate_count: 0,
                },
                SparseCandidateSummary {
                    exact_dosage_sum: ExactDosageSum::new(510, 255),
                    zero_count: 1,
                    homozygous_alternate_count: 1,
                },
            ]
        );
    }

    #[test]
    fn trusted_variant_major_decode_covers_identity_and_noncontiguous_selected_samples() {
        let block = trusted_probability_block(4, 2, 2, 2, &[2, 2, 2, 2], 0, 8, &[0, 0, 255, 0, 0, 255, 128, 0]);
        let variant_records = [variant_record(0, block.len())];

        let identity_selection =
            build_sample_selection(4, &[0, 1, 2, 3]).expect("identity sample selection should build");
        let mut identity_output = dosage_output_with_sentinels(4);
        let mut thread_scratch = ThreadScratch::default();
        let mut dosage_sum = vec![0.0_f32; 1];
        let mut dosage_square_sum = vec![0.0_f32; 1];
        let mut observation_count = vec![0_i32; 1];
        let mut sparse_candidate_statistics = vec![SparseCandidateSummary::default(); 1];
        let mut identity_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            sparse_candidate_statistics: Some(&mut sparse_candidate_statistics),
        };
        expect_decode_success(
            decode_variant_major_dosage_tile(
                VariantMajorTileDecodeRequest {
                    source_window: BgenByteWindow::from_bytes(&block),
                    compression_type: CompressionType::None,
                    sample_count: 4,
                    sample_selection: &identity_selection,
                    variant_records: &variant_records,
                    tile_variant_start_index: 0,
                },
                &mut identity_output,
                &mut identity_stats,
                &mut thread_scratch,
            ),
            "trusted identity decode should succeed",
        );
        let identity_output = initialized_f32_values(identity_output);
        assert_eq!(observation_count, vec![4]);
        assert_eq!(sparse_candidate_statistics[0].exact_dosage_sum, ExactDosageSum::new(1_019, 255));
        for (observed_value, expected_value) in identity_output.iter().zip([2.0, 0.0, 1.0, 254.0 / 255.0]) {
            assert!((observed_value - expected_value).abs() < 1.0e-6);
        }

        let noncontiguous_selection =
            build_sample_selection(4, &[3, 0]).expect("non-contiguous sample selection should build");
        let mut noncontiguous_output = dosage_output_with_sentinels(2);
        let mut dosage_sum = vec![0.0_f32; 1];
        let mut dosage_square_sum = vec![0.0_f32; 1];
        let mut observation_count = vec![0_i32; 1];
        let mut sparse_candidate_statistics = vec![SparseCandidateSummary::default(); 1];
        let mut noncontiguous_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            sparse_candidate_statistics: Some(&mut sparse_candidate_statistics),
        };
        expect_decode_success(
            decode_variant_major_dosage_tile(
                VariantMajorTileDecodeRequest {
                    source_window: BgenByteWindow::from_bytes(&block),
                    compression_type: CompressionType::None,
                    sample_count: 4,
                    sample_selection: &noncontiguous_selection,
                    variant_records: &variant_records,
                    tile_variant_start_index: 0,
                },
                &mut noncontiguous_output,
                &mut noncontiguous_stats,
                &mut thread_scratch,
            ),
            "trusted non-contiguous decode should succeed",
        );
        let noncontiguous_output = initialized_f32_values(noncontiguous_output);
        assert_eq!(observation_count, vec![2]);
        assert_eq!(sparse_candidate_statistics[0].exact_dosage_sum, ExactDosageSum::new(764, 255));
        for (observed_value, expected_value) in noncontiguous_output.iter().zip([254.0 / 255.0, 2.0]) {
            assert!((observed_value - expected_value).abs() < 1.0e-6);
        }
        assert!((dosage_sum[0] - (764.0 / 255.0)).abs() < 1.0e-6);
        assert!((dosage_square_sum[0] - ((254.0_f32.powi(2) + 510.0_f32.powi(2)) / 65_025.0)).abs() < 1.0e-6);
    }

    #[test]
    fn packed8_compatibility_validation_covers_valid_missing_corrupt_and_truncated_blocks() {
        let probability_bytes = [0_u8, 0, 255, 0, 0, 255, 128, 0];
        let block = trusted_probability_block(4, 2, 2, 2, &[2, 2, 2, 2], 0, 8, &probability_bytes);
        let variant_records = [variant_record(0, block.len())];
        let mut thread_scratch = ThreadScratch::default();
        assert_eq!(
            validate_variant_compatible_with_packed8(
                BgenByteWindow::from_bytes(&block),
                CompressionType::None,
                &variant_records[0],
                4,
                &mut thread_scratch,
            )
            .expect("valid packed8 block should pass compatibility validation"),
            Packed8Compatibility::Compatible
        );

        let missing_block = trusted_probability_block(1, 2, 2, 2, &[0x82], 0, 8, &[0, 0]);
        assert_eq!(
            validate_variant_compatible_with_packed8(
                BgenByteWindow::from_bytes(&missing_block),
                CompressionType::None,
                &variant_record(0, missing_block.len()),
                1,
                &mut thread_scratch,
            )
            .expect("missing packed8 block should remain valid dosage input"),
            Packed8Compatibility::RequiresDosage
        );

        let corrupt_block = trusted_probability_block(1, 2, 2, 2, &[2], 0, 8, &[255, 1]);
        let corrupt_error = validate_variant_compatible_with_packed8(
            BgenByteWindow::from_bytes(&corrupt_block),
            CompressionType::None,
            &variant_record(0, corrupt_block.len()),
            1,
            &mut thread_scratch,
        )
        .expect_err("probability pairs above the scale should fail validation");
        assert!(corrupt_error.to_string().contains("sum above"));

        let mut truncated_block = trusted_probability_block(1, 2, 2, 2, &[2], 0, 8, &[255, 0]);
        truncated_block.pop();
        let truncation_error = validate_variant_compatible_with_packed8(
            BgenByteWindow::from_bytes(&truncated_block),
            CompressionType::None,
            &variant_record(0, truncated_block.len()),
            1,
            &mut thread_scratch,
        )
        .expect_err("truncated packed8 block should fail validation");
        assert!(truncation_error.to_string().contains("requires exactly"));
    }

    #[test]
    fn trusted_packed8_variant_major_decode_covers_identity_and_contiguous_selected_samples() {
        let probability_bytes = [0_u8, 0, 255, 0, 0, 255, 128, 0];
        let block = trusted_probability_block(4, 2, 2, 2, &[2, 2, 2, 2], 0, 8, &probability_bytes);
        let variant_records = [variant_record(0, block.len())];
        let mut thread_scratch = ThreadScratch::default();
        let identity_selection = build_sample_selection(4, &[0, 1, 2, 3]).expect("identity selection should build");
        let mut identity_output = probability_output_with_sentinels(probability_bytes.len());
        let mut dosage_sum = vec![0.0_f32; 1];
        let mut dosage_square_sum = vec![0.0_f32; 1];
        let mut observation_count = vec![0_i32; 1];
        let mut sparse_candidate_statistics = vec![SparseCandidateSummary::default(); 1];
        let mut identity_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            sparse_candidate_statistics: Some(&mut sparse_candidate_statistics),
        };
        expect_decode_success(
            decode_variant_major_probability_pair_tile(
                BgenByteWindow::from_bytes(&block),
                CompressionType::None,
                4,
                &identity_selection,
                &variant_records,
                &mut identity_output,
                0,
                &mut identity_stats,
                &mut thread_scratch,
            ),
            "trusted packed8 identity decode should succeed",
        );
        let identity_output = initialized_u8_values(identity_output);

        assert_eq!(identity_output, probability_bytes);
        assert_eq!(observation_count, vec![4]);
        assert_eq!(
            sparse_candidate_statistics,
            vec![SparseCandidateSummary {
                exact_dosage_sum: ExactDosageSum::new(1_019, 255),
                zero_count: 1,
                homozygous_alternate_count: 1,
            }]
        );

        let contiguous_selection = build_sample_selection(4, &[1, 2]).expect("contiguous selection should build");
        let mut contiguous_output = probability_output_with_sentinels(4);
        let mut dosage_sum = vec![0.0_f32; 1];
        let mut dosage_square_sum = vec![0.0_f32; 1];
        let mut observation_count = vec![0_i32; 1];
        let mut sparse_candidate_statistics = vec![SparseCandidateSummary::default(); 1];
        let mut contiguous_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            sparse_candidate_statistics: Some(&mut sparse_candidate_statistics),
        };
        expect_decode_success(
            decode_variant_major_probability_pair_tile(
                BgenByteWindow::from_bytes(&block),
                CompressionType::None,
                4,
                &contiguous_selection,
                &variant_records,
                &mut contiguous_output,
                0,
                &mut contiguous_stats,
                &mut thread_scratch,
            ),
            "trusted packed8 contiguous decode should succeed",
        );
        let contiguous_output = initialized_u8_values(contiguous_output);

        assert_eq!(contiguous_output, vec![255, 0, 0, 255]);
        assert_eq!(observation_count, vec![2]);
        assert_eq!(
            sparse_candidate_statistics,
            vec![SparseCandidateSummary {
                exact_dosage_sum: ExactDosageSum::new(255, 255),
                zero_count: 1,
                homozygous_alternate_count: 0,
            }]
        );
        assert!((dosage_sum[0] - 1.0).abs() < f32::EPSILON);
        assert!((dosage_square_sum[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn trusted_packed8_variant_major_decode_covers_indexed_fractional_samples() {
        let probability_bytes = [0_u8, 0, 255, 0, 0, 255, 128, 0];
        let block = trusted_probability_block(4, 2, 2, 2, &[2, 2, 2, 2], 0, 8, &probability_bytes);
        let variant_records = [variant_record(0, block.len())];
        let mut thread_scratch = ThreadScratch::default();
        let indexed_selection = build_sample_selection(4, &[3, 0, 2]).expect("indexed sample selection should build");
        let mut indexed_output = probability_output_with_sentinels(6);
        let mut dosage_sum = vec![0.0_f32; 1];
        let mut dosage_square_sum = vec![0.0_f32; 1];
        let mut observation_count = vec![0_i32; 1];
        let mut sparse_candidate_statistics = vec![SparseCandidateSummary::default(); 1];
        let mut indexed_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            sparse_candidate_statistics: Some(&mut sparse_candidate_statistics),
        };
        expect_decode_success(
            decode_variant_major_probability_pair_tile(
                BgenByteWindow::from_bytes(&block),
                CompressionType::None,
                4,
                &indexed_selection,
                &variant_records,
                &mut indexed_output,
                0,
                &mut indexed_stats,
                &mut thread_scratch,
            ),
            "trusted packed8 indexed decode should succeed",
        );
        let indexed_output = initialized_u8_values(indexed_output);

        assert_eq!(indexed_output, vec![128, 0, 0, 0, 0, 255]);
        assert_eq!(observation_count, vec![3]);
        assert_eq!(
            sparse_candidate_statistics,
            vec![SparseCandidateSummary {
                exact_dosage_sum: ExactDosageSum::new(1_019, 255),
                zero_count: 0,
                homozygous_alternate_count: 1,
            }]
        );
        assert!((dosage_sum[0] - (1019.0 / 255.0)).abs() < 1.0e-6);
        assert!(
            (dosage_square_sum[0] - ((254.0_f32.powi(2) + 510.0_f32.powi(2) + 255.0_f32.powi(2)) / 65_025.0)).abs()
                < 1.0e-6
        );
    }
}
