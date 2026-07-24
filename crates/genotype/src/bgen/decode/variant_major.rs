use std::mem::MaybeUninit;

use super::super::metadata::VariantRecord;
use super::super::sample_selection::SampleSelection;
use super::super::simd;
use super::super::{BgenError, CompressionType};
use super::matrix::{
    ThreadScratch, VariantMajorTileStatsMut, exact_eight_bit_probability_pairs, packed_eight_bit_probability_index,
    read_eight_bit_probability_pair, selected_sample_count_to_i32, unphased_eight_bit_dosage_lookup,
};
use super::probability::{
    PackedProbabilityReader, ParsedLayoutTwoProbabilityBlock, layout_two_probability_byte_count,
    parse_layout_two_probability_block, read_exact_bytes, read_probability_block, validate_diploid_sample_flags,
    validate_stored_probability_pair,
};
use super::{VariantDecodeFailure, VariantMajorTileDecodeRequest};
use crate::common::{DosageSummary, ExactDosageSum};
use crate::preprocess;

pub(in crate::bgen) fn decode_variant_major_dosage_tile(
    request: VariantMajorTileDecodeRequest<'_>,
    output_values: &mut [MaybeUninit<f32>],
    tile_stats: &mut VariantMajorTileStatsMut<'_>,
    thread_scratch: &mut ThreadScratch,
) -> Result<(), VariantDecodeFailure> {
    let VariantMajorTileDecodeRequest {
        source_window,
        compression_type,
        sample_count,
        sample_selection,
        variant_records,
        tile_variant_start_index,
    } = request;
    validate_variant_major_tile_stats_lengths(tile_stats, variant_records.len())
        .map_err(|source| VariantDecodeFailure { relative_variant_index: None, source })?;
    let selected_sample_count = sample_selection.selected_sample_count();
    let expected_output_value_count =
        variant_records.len().checked_mul(selected_sample_count).ok_or_else(|| VariantDecodeFailure {
            relative_variant_index: None,
            source: BgenError::Range("Integer overflow while validating a variant-major BGEN output tile.".to_string()),
        })?;
    if output_values.len() != expected_output_value_count {
        return Err(VariantDecodeFailure {
            relative_variant_index: None,
            source: BgenError::Range(format!(
                "Variant-major BGEN output tile contains {} values, expected {expected_output_value_count}.",
                output_values.len()
            )),
        });
    }
    if selected_sample_count == 0 {
        return Ok(());
    }
    let collect_sparse_candidate_statistics = tile_stats.sparse_candidate_statistics.is_some();
    for ((tile_variant_index, variant_record), output_row) in
        variant_records.iter().enumerate().zip(output_values.chunks_exact_mut(selected_sample_count))
    {
        let variant_decode_result = decode_variant_dosages_into_variant_major_row(
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
            sparse_candidate_statistics[tile_variant_index].exact_dosage_sum = variant_decode_result
                .exact_dosage_sum
                .ok_or_else(|| VariantDecodeFailure {
                relative_variant_index: Some(tile_variant_start_index + tile_variant_index),
                source: BgenError::Range("Sparse candidate decoding did not produce an exact dosage sum.".to_string()),
            })?;
            sparse_candidate_statistics[tile_variant_index].zero_count = variant_decode_result.zero_count;
            sparse_candidate_statistics[tile_variant_index].homozygous_alternate_count =
                variant_decode_result.homozygous_alternate_count;
        }
    }
    Ok(())
}

pub(in crate::bgen) fn validate_variant_major_tile_stats_lengths(
    tile_stats: &VariantMajorTileStatsMut<'_>,
    variant_count: usize,
) -> Result<(), BgenError> {
    if tile_stats.dosage_sum.len() == variant_count
        && tile_stats.dosage_square_sum.len() == variant_count
        && tile_stats.observation_count.len() == variant_count
        && tile_stats.sparse_candidate_statistics.as_ref().is_none_or(|statistics| statistics.len() == variant_count)
    {
        return Ok(());
    }
    Err(BgenError::Range(format!("Variant-major tile stats shape mismatch for {variant_count} variants.")))
}

fn decode_variant_dosages_into_variant_major_row(
    source_window: super::super::source::BgenByteWindow<'_>,
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_row: &mut [MaybeUninit<f32>],
    collect_sparse_candidate_statistics: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<DosageSummary, BgenError> {
    let probability_block = read_probability_block(source_window, compression_type, variant_record, thread_scratch)?;
    let parsed_probability_block = parse_layout_two_probability_block(probability_block, sample_count)?;
    if parsed_probability_block.minimum_ploidy != 2 || parsed_probability_block.maximum_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "uses ploidy bounds [{}, {}], but variant-major reads currently support diploid BGEN variants only.",
            parsed_probability_block.minimum_ploidy, parsed_probability_block.maximum_ploidy,
        )));
    }
    let sample_ploidy_and_missingness = parsed_probability_block.sample_ploidy_and_missingness;
    let phased_flag = parsed_probability_block.phased_flag;
    let probability_bit_count = parsed_probability_block.probability_bit_count;
    if phased_flag > 1 {
        return Err(BgenError::InvalidFormat(format!(
            "uses phased flag {phased_flag}, but BGEN Layout 2 requires 0 or 1.",
        )));
    }
    if !(1..=32).contains(&probability_bit_count) {
        return Err(BgenError::InvalidFormat(format!(
            "uses {probability_bit_count} bits per probability, but BGEN Layout 2 requires a value between 1 and 32.",
        )));
    }

    if phased_flag == 0 && probability_bit_count == 8 {
        return decode_unphased_eight_bit_dosages_into_variant_major_row(
            sample_ploidy_and_missingness,
            parsed_probability_block.probability_bytes,
            sample_selection,
            output_row,
            collect_sparse_candidate_statistics,
        );
    }

    decode_generic_variant_dosages_into_variant_major_row(
        &parsed_probability_block,
        sample_count,
        sample_selection,
        output_row,
        collect_sparse_candidate_statistics,
    )
}

fn decode_generic_variant_dosages_into_variant_major_row(
    parsed_probability_block: &ParsedLayoutTwoProbabilityBlock<'_>,
    sample_count: usize,
    sample_selection: &SampleSelection,
    output_row: &mut [MaybeUninit<f32>],
    collect_sparse_candidate_statistics: bool,
) -> Result<DosageSummary, BgenError> {
    let sample_ploidy_and_missingness = parsed_probability_block.sample_ploidy_and_missingness;
    let phased_flag = parsed_probability_block.phased_flag;
    let probability_bit_count = parsed_probability_block.probability_bit_count;
    let maximum_probability_value =
        if probability_bit_count == 32 { u32::MAX } else { (1_u32 << probability_bit_count) - 1 };
    let probability_scale_denominator = f64::from(maximum_probability_value);
    let expected_probability_byte_count = layout_two_probability_byte_count(sample_count, probability_bit_count)?;
    if parsed_probability_block.probability_bytes.len() != expected_probability_byte_count {
        return Err(BgenError::InvalidFormat(format!(
            "contains {} probability bytes, but its encoding requires exactly {expected_probability_byte_count}.",
            parsed_probability_block.probability_bytes.len(),
        )));
    }
    let mut bit_reader = PackedProbabilityReader::new(parsed_probability_block.probability_bytes);
    let selected_sample_count = output_row.len();
    selected_sample_count_to_i32(selected_sample_count)?;
    let mut dosage_sum = 0.0_f32;
    let mut dosage_square_sum = 0.0_f32;
    let mut observation_count = 0_i32;
    let mut has_missing_values = false;
    let mut exact_dosage_numerator = collect_sparse_candidate_statistics.then_some(0_u64);
    let mut zero_count = 0_i32;
    let mut homozygous_alternate_count = 0_i32;
    if collect_sparse_candidate_statistics {
        validate_exact_dosage_sum_capacity(selected_sample_count, maximum_probability_value)?;
    }

    for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().copied().enumerate() {
        let is_missing = validate_diploid_sample_flags(ploidy_and_missingness, file_sample_index)?;
        let first_probability = bit_reader.read_probability(probability_bit_count)?;
        let second_probability = bit_reader.read_probability(probability_bit_count)?;
        validate_stored_probability_pair(
            first_probability,
            second_probability,
            u64::from(maximum_probability_value),
            phased_flag,
            is_missing,
            file_sample_index,
        )?;
        let first_probability_scaled = f64::from(first_probability) / probability_scale_denominator;
        let second_probability_scaled = f64::from(second_probability) / probability_scale_denominator;
        let dosage_value_f64 = if phased_flag == 0 {
            2.0_f64 - ((2.0 * first_probability_scaled) + second_probability_scaled)
        } else {
            2.0_f64 - (first_probability_scaled + second_probability_scaled)
        };
        // A BGEN dosage is bounded to [0, 2]; f32 is the engine's documented
        // genotype storage contract.
        #[allow(clippy::cast_possible_truncation)]
        let dosage_value = dosage_value_f64 as f32;

        let Some(selected_index) = sample_selection.selected_index(file_sample_index) else {
            continue;
        };

        let output_value = if is_missing { f32::NAN } else { dosage_value };
        output_row[selected_index].write(output_value);
        if is_missing {
            has_missing_values = true;
            continue;
        }
        dosage_sum += dosage_value;
        dosage_square_sum += dosage_value * dosage_value;
        observation_count += 1;
        if let Some(exact_dosage_numerator) = exact_dosage_numerator.as_mut() {
            let dosage_numerator = quantized_dosage_numerator(
                first_probability,
                second_probability,
                maximum_probability_value,
                phased_flag,
            );
            *exact_dosage_numerator += dosage_numerator;
            preprocess::increment_exact_sparse_candidate_counts(
                dosage_numerator,
                maximum_probability_value,
                &mut zero_count,
                &mut homozygous_alternate_count,
            );
        }
    }

    if !bit_reader.has_only_zero_padding() {
        return Err(BgenError::InvalidFormat(
            "contains nonzero padding bits after its stored probabilities.".to_string(),
        ));
    }

    impute_variant_major_row_if_needed(output_row, dosage_sum, observation_count, has_missing_values);
    Ok(DosageSummary {
        dosage_sum,
        dosage_square_sum,
        observation_count,
        zero_count,
        homozygous_alternate_count,
        exact_dosage_sum: exact_dosage_numerator
            .map(|numerator| ExactDosageSum::new(numerator, maximum_probability_value)),
    })
}

fn decode_unphased_eight_bit_dosages_into_variant_major_row(
    sample_ploidy_and_missingness: &[u8],
    packed_probability_bytes: &[u8],
    sample_selection: &SampleSelection,
    output_row: &mut [MaybeUninit<f32>],
    collect_sparse_candidate_statistics: bool,
) -> Result<DosageSummary, BgenError> {
    let selected_sample_count = output_row.len();
    let expected_probability_byte_count = sample_ploidy_and_missingness.len().checked_mul(2).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while decoding 8-bit BGEN probabilities.".to_string())
    })?;
    if packed_probability_bytes.len() != expected_probability_byte_count {
        return Err(BgenError::InvalidFormat(format!(
            "contains {} probability bytes, but an 8-bit diploid record requires exactly {expected_probability_byte_count}.",
            packed_probability_bytes.len(),
        )));
    }

    selected_sample_count_to_i32(selected_sample_count)?;
    if collect_sparse_candidate_statistics {
        validate_exact_dosage_sum_capacity(selected_sample_count, u32::from(u8::MAX))?;
    }
    let all_samples_present = simd::all_samples_present_diploid_simd_or_scalar(sample_ploidy_and_missingness);

    if all_samples_present
        && !sample_selection.is_identity()
        && !simd::all_unphased_eight_bit_probability_pairs_valid_simd_or_scalar(
            &packed_probability_bytes[..expected_probability_byte_count],
        )
    {
        return Err(BgenError::InvalidFormat(
            "contains an 8-bit probability pair whose values sum above 255.".to_string(),
        ));
    }

    if sample_selection.is_identity() && all_samples_present {
        let decode_summary = simd::decode_unphased_eight_bit_identity_simd_or_scalar(
            &packed_probability_bytes[..expected_probability_byte_count],
            output_row,
            collect_sparse_candidate_statistics,
        )
        .ok_or_else(|| {
            BgenError::InvalidFormat("contains an 8-bit probability pair whose values sum above 255.".to_string())
        })?;
        return Ok(decode_summary);
    }

    if all_samples_present {
        return decode_all_present_unphased_eight_bit_subset(
            &packed_probability_bytes[..expected_probability_byte_count],
            sample_selection,
            output_row,
            collect_sparse_candidate_statistics,
        );
    }

    let mut dosage_sum = 0.0_f32;
    let mut dosage_square_sum = 0.0_f32;
    let mut observation_count = 0_i32;
    let mut has_missing_values = false;
    let mut exact_dosage_numerator = collect_sparse_candidate_statistics.then_some(0_u64);
    let mut zero_count = 0_i32;
    let mut homozygous_alternate_count = 0_i32;
    let probability_pairs =
        exact_eight_bit_probability_pairs(&packed_probability_bytes[..expected_probability_byte_count]);
    let dosage_lookup = unphased_eight_bit_dosage_lookup();
    for (file_sample_index, (ploidy_and_missingness, probability_pair)) in
        sample_ploidy_and_missingness.iter().zip(probability_pairs.iter().copied()).enumerate()
    {
        let is_missing = validate_diploid_sample_flags(*ploidy_and_missingness, file_sample_index)?;
        validate_stored_probability_pair(
            u32::from(probability_pair[0]),
            u32::from(probability_pair[1]),
            u64::from(u8::MAX),
            0,
            is_missing,
            file_sample_index,
        )?;

        let Some(selected_index) = sample_selection.selected_index(file_sample_index) else {
            continue;
        };

        let packed_probability_index = packed_eight_bit_probability_index(probability_pair);
        let dosage_value = dosage_lookup[packed_probability_index];
        let output_value = if is_missing { f32::NAN } else { dosage_value };
        output_row[selected_index].write(output_value);
        if is_missing {
            has_missing_values = true;
            continue;
        }
        dosage_sum += dosage_value;
        dosage_square_sum += dosage_value * dosage_value;
        observation_count += 1;
        if let Some(exact_dosage_numerator) = exact_dosage_numerator.as_mut() {
            let dosage_numerator =
                quantized_dosage_numerator(u32::from(probability_pair[0]), u32::from(probability_pair[1]), 255, 0);
            *exact_dosage_numerator += dosage_numerator;
            preprocess::increment_exact_sparse_candidate_counts(
                dosage_numerator,
                u32::from(u8::MAX),
                &mut zero_count,
                &mut homozygous_alternate_count,
            );
        }
    }

    impute_variant_major_row_if_needed(output_row, dosage_sum, observation_count, has_missing_values);
    Ok(DosageSummary {
        dosage_sum,
        dosage_square_sum,
        observation_count,
        zero_count,
        homozygous_alternate_count,
        exact_dosage_sum: exact_dosage_numerator.map(|numerator| ExactDosageSum::new(numerator, u32::from(u8::MAX))),
    })
}

fn decode_all_present_unphased_eight_bit_subset(
    packed_probability_bytes: &[u8],
    sample_selection: &SampleSelection,
    output_row: &mut [MaybeUninit<f32>],
    collect_sparse_candidate_statistics: bool,
) -> Result<DosageSummary, BgenError> {
    if let Some(contiguous_file_index_start) = sample_selection.contiguous_file_index_start() {
        let probability_offset = contiguous_file_index_start.checked_mul(2).ok_or_else(|| {
            BgenError::InvalidFormat("Integer overflow while indexing 8-bit BGEN probabilities.".to_string())
        })?;
        let selected_probability_byte_count = output_row.len().checked_mul(2).ok_or_else(|| {
            BgenError::InvalidFormat("Integer overflow while slicing selected 8-bit BGEN probabilities.".to_string())
        })?;
        let selected_probability_bytes =
            read_exact_bytes(packed_probability_bytes, probability_offset, selected_probability_byte_count)?;
        return simd::decode_unphased_eight_bit_identity_simd_or_scalar(
            selected_probability_bytes,
            output_row,
            collect_sparse_candidate_statistics,
        )
        .ok_or_else(|| {
            BgenError::InvalidFormat("contains an 8-bit probability pair whose values sum above 255.".to_string())
        });
    }

    let dosage_lookup = unphased_eight_bit_dosage_lookup();
    let selected_file_indices = sample_selection
        .indexed_file_indices()
        .expect("non-identity, non-contiguous sample selections store explicit file indices");
    let mut dosage_sum = 0.0_f32;
    let mut dosage_square_sum = 0.0_f32;
    let mut observation_count = 0_i32;
    let mut exact_dosage_numerator = collect_sparse_candidate_statistics.then_some(0_u64);
    let mut zero_count = 0_i32;
    let mut homozygous_alternate_count = 0_i32;
    for (selected_index, file_sample_index) in selected_file_indices.iter().copied().enumerate() {
        let probability_offset = file_sample_index.checked_mul(2).ok_or_else(|| {
            BgenError::InvalidFormat("Integer overflow while indexing 8-bit BGEN probabilities.".to_string())
        })?;
        let probability_pair = read_eight_bit_probability_pair(packed_probability_bytes, probability_offset)?;
        validate_stored_probability_pair(
            u32::from(probability_pair[0]),
            u32::from(probability_pair[1]),
            u64::from(u8::MAX),
            0,
            false,
            file_sample_index,
        )?;
        let dosage_value = dosage_lookup[packed_eight_bit_probability_index(probability_pair)];
        output_row[selected_index].write(dosage_value);
        dosage_sum += dosage_value;
        dosage_square_sum += dosage_value * dosage_value;
        observation_count += 1;
        if let Some(exact_dosage_numerator) = exact_dosage_numerator.as_mut() {
            let dosage_numerator =
                quantized_dosage_numerator(u32::from(probability_pair[0]), u32::from(probability_pair[1]), 255, 0);
            *exact_dosage_numerator += dosage_numerator;
            preprocess::increment_exact_sparse_candidate_counts(
                dosage_numerator,
                u32::from(u8::MAX),
                &mut zero_count,
                &mut homozygous_alternate_count,
            );
        }
    }
    Ok(DosageSummary {
        dosage_sum,
        dosage_square_sum,
        observation_count,
        zero_count,
        homozygous_alternate_count,
        exact_dosage_sum: exact_dosage_numerator.map(|numerator| ExactDosageSum::new(numerator, u32::from(u8::MAX))),
    })
}

fn quantized_dosage_numerator(
    first_probability: u32,
    second_probability: u32,
    maximum_probability_value: u32,
    phased_flag: u8,
) -> u64 {
    let maximum_probability_value = u64::from(maximum_probability_value);
    let stored_allele_zero_numerator = if phased_flag == 0 {
        (2 * u64::from(first_probability)) + u64::from(second_probability)
    } else {
        u64::from(first_probability) + u64::from(second_probability)
    };
    (2 * maximum_probability_value) - stored_allele_zero_numerator
}

fn validate_exact_dosage_sum_capacity(
    selected_sample_count: usize,
    probability_denominator: u32,
) -> Result<(), BgenError> {
    let selected_sample_count = u64::try_from(selected_sample_count).map_err(|_| {
        BgenError::Range(format!(
            "Selected sample count {selected_sample_count} exceeds the supported u64 statistics range.",
        ))
    })?;
    selected_sample_count
        .checked_mul(2)
        .and_then(|diploid_allele_count| diploid_allele_count.checked_mul(u64::from(probability_denominator)))
        .ok_or_else(|| BgenError::Range("Exact BGEN dosage-sum bound overflowed u64.".to_string()))?;
    Ok(())
}

fn impute_variant_major_row_if_needed(
    output_row: &mut [MaybeUninit<f32>],
    dosage_sum: f32,
    observation_count: i32,
    has_missing_values: bool,
) {
    if !has_missing_values {
        return;
    }
    // Observation counts are exact up to the enforced i32 sample limit; f32
    // division matches the dosage accumulator and output representation.
    #[allow(clippy::cast_precision_loss)]
    let imputed_dosage_value = dosage_sum / observation_count.max(1) as f32;
    let initialized_output_row = unsafe {
        // Every selected sample position is written exactly once before this
        // function is called. Decode failures return before reaching this point.
        std::slice::from_raw_parts_mut(output_row.as_mut_ptr().cast::<f32>(), output_row.len())
    };
    for output_value in initialized_output_row {
        if output_value.is_nan() {
            *output_value = imputed_dosage_value;
        }
    }
}
