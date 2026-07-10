use super::super::metadata::VariantRecord;
use super::super::sample_selection::SampleSelection;
use super::super::simd;
use super::super::{BgenError, CompressionType};
use super::matrix::{
    MISSING_SAMPLE_FLAG_MASK, PLOIDY_MASK, ThreadScratch, VariantDecodeResult, VariantMajorOutputMatrix,
    VariantMajorTileStatsMut, exact_eight_bit_probability_pairs, packed_eight_bit_probability_index,
    read_eight_bit_probability_pair, selected_sample_count_to_i32, unphased_eight_bit_dosage_lookup,
};
use super::probability::{
    PackedProbabilityReader, read_exact_bytes, read_probability_block, read_u8_at, read_u16_at, read_u32_at,
    u32_to_usize,
};
use crate::buffer::OutputBufferAddress;
use crate::preprocess;

#[allow(clippy::too_many_arguments)]
pub(in crate::bgen) fn decode_variant_major_dosage_tile(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_pointer_address: OutputBufferAddress,
    selected_sample_count: usize,
    tile_variant_start_index: usize,
    trusted_no_missing_diploid: bool,
    tile_stats: &mut VariantMajorTileStatsMut<'_>,
    thread_scratch: &mut ThreadScratch,
) -> Result<(), BgenError> {
    validate_variant_major_tile_stats_lengths(tile_stats, variant_record_chunk.len())?;
    for (tile_variant_index, variant_record) in variant_record_chunk.iter().enumerate() {
        let variant_decode_result = decode_variant_dosages_into_variant_major_matrix(
            mmap,
            compression_type,
            sample_count,
            sample_selection,
            variant_record,
            output_pointer_address,
            tile_variant_start_index + tile_variant_index,
            selected_sample_count,
            trusted_no_missing_diploid,
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
        && tile_stats.zero_count.len() == variant_count
        && tile_stats.nonzero_count.len() == variant_count
        && tile_stats.homozygous_reference_count.len() == variant_count
        && tile_stats.heterozygous_count.len() == variant_count
        && tile_stats.homozygous_alternate_count.len() == variant_count
    {
        return Ok(());
    }
    Err(BgenError::Range(format!("Variant-major tile stats shape mismatch for {variant_count} variants.")))
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn decode_variant_dosages_into_variant_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: OutputBufferAddress,
    variant_index: usize,
    selected_sample_count: usize,
    trusted_no_missing_diploid: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<VariantDecodeResult, BgenError> {
    let probability_block = read_probability_block(mmap, compression_type, variant_record, thread_scratch)?;

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
            "Variant '{}' uses ploidy bounds [{minimum_ploidy}, {maximum_ploidy}], but variant-major reads currently support diploid BGEN variants only.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let sample_ploidy_and_missingness = read_exact_bytes(probability_block, cursor, sample_count)?;
    cursor += sample_count;

    let phased_flag = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let probability_bit_count = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    if !(1..=32).contains(&probability_bit_count) {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' uses {probability_bit_count} bits per probability, but BGEN Layout 2 requires a value between 1 and 32.",
            variant_record.resolved_variant_identifier,
        )));
    }

    if phased_flag == 0 && probability_bit_count == 8 {
        return decode_unphased_eight_bit_dosages_into_variant_major_matrix(
            sample_ploidy_and_missingness,
            &probability_block[cursor..],
            sample_selection,
            variant_record,
            output_pointer_address,
            variant_index,
            selected_sample_count,
            trusted_no_missing_diploid,
        );
    }

    let probability_scale_denominator =
        if probability_bit_count == 32 { f64::from(u32::MAX) } else { f64::from((1_u32 << probability_bit_count) - 1) };
    let mut bit_reader = PackedProbabilityReader::new(&probability_block[cursor..]);
    selected_sample_count_to_i32(selected_sample_count)?;
    let mut output_matrix = unsafe {
        VariantMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            selected_sample_count,
            "variant-major BGEN",
        )?
    };
    let output_row = output_matrix.row_mut(variant_index)?;
    let all_samples_present =
        trusted_no_missing_diploid || simd::all_samples_present_diploid_simd_or_scalar(sample_ploidy_and_missingness);
    let mut selected_dosage_total = 0.0_f32;
    let mut selected_dosage_square_total = 0.0_f32;
    let mut selected_observation_count = 0_i32;
    let mut has_missing_values = false;
    let mut zero_count = 0_i32;
    let mut nonzero_count = 0_i32;
    let mut homozygous_reference_count = 0_i32;
    let mut heterozygous_count = 0_i32;
    let mut homozygous_alternate_count = 0_i32;

    for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().enumerate() {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }

        let dosage_value = match phased_flag {
            0 => {
                let homozygous_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                let heterozygous_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                2.0_f64 - ((2.0 * homozygous_reference_probability) + heterozygous_probability)
            }
            1 => {
                let first_haplotype_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                let second_haplotype_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                2.0_f64 - (first_haplotype_reference_probability + second_haplotype_reference_probability)
            }
            unsupported_flag => {
                return Err(BgenError::InvalidFormat(format!(
                    "Variant '{}' uses phased flag {unsupported_flag}, but BGEN Layout 2 requires 0 or 1.",
                    variant_record.resolved_variant_identifier,
                )));
            }
        } as f32;

        let selected_index = if sample_selection.is_identity {
            file_sample_index
        } else {
            sample_selection.file_to_selected_index[file_sample_index]
        };
        if selected_index == usize::MAX {
            continue;
        }

        let is_missing = !all_samples_present && (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0;
        let output_value = if is_missing { f32::NAN } else { dosage_value };
        output_row[selected_index] = output_value;
        if is_missing {
            has_missing_values = true;
            continue;
        }
        selected_dosage_total += dosage_value;
        selected_dosage_square_total += dosage_value * dosage_value;
        selected_observation_count += 1;
        preprocess::increment_dosage_summary_counts(
            dosage_value,
            &mut zero_count,
            &mut nonzero_count,
            &mut homozygous_reference_count,
            &mut heterozygous_count,
            &mut homozygous_alternate_count,
        );
    }

    impute_variant_major_row_if_needed(
        output_row,
        selected_dosage_total,
        selected_observation_count,
        has_missing_values,
    );
    Ok(VariantDecodeResult {
        selected_dosage_total,
        selected_dosage_square_total,
        selected_observation_count,
        zero_count,
        nonzero_count,
        homozygous_reference_count,
        heterozygous_count,
        homozygous_alternate_count,
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn decode_unphased_eight_bit_dosages_into_variant_major_matrix(
    sample_ploidy_and_missingness: &[u8],
    packed_probability_bytes: &[u8],
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: OutputBufferAddress,
    variant_index: usize,
    selected_sample_count: usize,
    trusted_no_missing_diploid: bool,
) -> Result<VariantDecodeResult, BgenError> {
    let expected_probability_byte_count = sample_ploidy_and_missingness.len().checked_mul(2).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while decoding 8-bit BGEN probabilities.".to_string())
    })?;
    if packed_probability_bytes.len() < expected_probability_byte_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' ended before all 8-bit probabilities were decoded.",
            variant_record.resolved_variant_identifier,
        )));
    }

    selected_sample_count_to_i32(selected_sample_count)?;
    let mut output_matrix = unsafe {
        VariantMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            selected_sample_count,
            "variant-major BGEN",
        )?
    };
    let output_row = output_matrix.row_mut(variant_index)?;
    let all_samples_present =
        trusted_no_missing_diploid || simd::all_samples_present_diploid_simd_or_scalar(sample_ploidy_and_missingness);

    if sample_selection.is_identity && all_samples_present {
        let decode_summary = simd::decode_unphased_eight_bit_identity_simd_or_scalar(
            &packed_probability_bytes[..expected_probability_byte_count],
            output_row,
        );
        return Ok(VariantDecodeResult {
            selected_dosage_total: decode_summary.selected_dosage_total,
            selected_dosage_square_total: decode_summary.selected_dosage_square_total,
            selected_observation_count: decode_summary.selected_observation_count,
            zero_count: decode_summary.zero_count,
            nonzero_count: decode_summary.nonzero_count,
            homozygous_reference_count: decode_summary.homozygous_reference_count,
            heterozygous_count: decode_summary.heterozygous_count,
            homozygous_alternate_count: decode_summary.homozygous_alternate_count,
        });
    }

    let mut selected_dosage_total = 0.0_f32;
    let mut selected_dosage_square_total = 0.0_f32;
    let mut selected_observation_count = 0_i32;
    let mut has_missing_values = false;
    let mut zero_count = 0_i32;
    let mut nonzero_count = 0_i32;
    let mut homozygous_reference_count = 0_i32;
    let mut heterozygous_count = 0_i32;
    let mut homozygous_alternate_count = 0_i32;

    let probability_pairs =
        exact_eight_bit_probability_pairs(&packed_probability_bytes[..expected_probability_byte_count]);
    if !sample_selection.is_identity && all_samples_present {
        if let Some(contiguous_file_index_start) = sample_selection.contiguous_file_index_start {
            let probability_offset = contiguous_file_index_start.checked_mul(2).ok_or_else(|| {
                BgenError::InvalidFormat("Integer overflow while indexing 8-bit BGEN probabilities.".to_string())
            })?;
            let selected_probability_byte_count = selected_sample_count.checked_mul(2).ok_or_else(|| {
                BgenError::InvalidFormat(
                    "Integer overflow while slicing selected 8-bit BGEN probabilities.".to_string(),
                )
            })?;
            let selected_probability_bytes = read_exact_bytes(
                &packed_probability_bytes[..expected_probability_byte_count],
                probability_offset,
                selected_probability_byte_count,
            )?;
            let decode_summary =
                simd::decode_unphased_eight_bit_identity_simd_or_scalar(selected_probability_bytes, output_row);
            return Ok(VariantDecodeResult {
                selected_dosage_total: decode_summary.selected_dosage_total,
                selected_dosage_square_total: decode_summary.selected_dosage_square_total,
                selected_observation_count: decode_summary.selected_observation_count,
                zero_count: decode_summary.zero_count,
                nonzero_count: decode_summary.nonzero_count,
                homozygous_reference_count: decode_summary.homozygous_reference_count,
                heterozygous_count: decode_summary.heterozygous_count,
                homozygous_alternate_count: decode_summary.homozygous_alternate_count,
            });
        }

        let dosage_lookup = unphased_eight_bit_dosage_lookup();
        for (selected_index, file_sample_index) in sample_selection.selected_file_indices.iter().copied().enumerate() {
            let probability_offset = file_sample_index.checked_mul(2).ok_or_else(|| {
                BgenError::InvalidFormat("Integer overflow while indexing 8-bit BGEN probabilities.".to_string())
            })?;
            let probability_pair = read_eight_bit_probability_pair(
                &packed_probability_bytes[..expected_probability_byte_count],
                probability_offset,
            )?;
            let packed_probability_index = packed_eight_bit_probability_index(probability_pair);
            let dosage_value = dosage_lookup[packed_probability_index];
            output_row[selected_index] = dosage_value;
            selected_dosage_total += dosage_value;
            selected_dosage_square_total += dosage_value * dosage_value;
            selected_observation_count += 1;
            preprocess::increment_dosage_summary_counts(
                dosage_value,
                &mut zero_count,
                &mut nonzero_count,
                &mut homozygous_reference_count,
                &mut heterozygous_count,
                &mut homozygous_alternate_count,
            );
        }
        return Ok(VariantDecodeResult {
            selected_dosage_total,
            selected_dosage_square_total,
            selected_observation_count,
            zero_count,
            nonzero_count,
            homozygous_reference_count,
            heterozygous_count,
            homozygous_alternate_count,
        });
    }

    let dosage_lookup = unphased_eight_bit_dosage_lookup();
    for (file_sample_index, (ploidy_and_missingness, probability_pair)) in
        sample_ploidy_and_missingness.iter().zip(probability_pairs.iter().copied()).enumerate()
    {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }

        let selected_index = if sample_selection.is_identity {
            file_sample_index
        } else {
            sample_selection.file_to_selected_index[file_sample_index]
        };
        if selected_index == usize::MAX {
            continue;
        }

        let packed_probability_index = packed_eight_bit_probability_index(probability_pair);
        let dosage_value = dosage_lookup[packed_probability_index];
        let is_missing = !all_samples_present && (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0;
        let output_value = if is_missing { f32::NAN } else { dosage_value };
        output_row[selected_index] = output_value;
        if is_missing {
            has_missing_values = true;
            continue;
        }
        selected_dosage_total += dosage_value;
        selected_dosage_square_total += dosage_value * dosage_value;
        selected_observation_count += 1;
        preprocess::increment_dosage_summary_counts(
            dosage_value,
            &mut zero_count,
            &mut nonzero_count,
            &mut homozygous_reference_count,
            &mut heterozygous_count,
            &mut homozygous_alternate_count,
        );
    }

    impute_variant_major_row_if_needed(
        output_row,
        selected_dosage_total,
        selected_observation_count,
        has_missing_values,
    );
    Ok(VariantDecodeResult {
        selected_dosage_total,
        selected_dosage_square_total,
        selected_observation_count,
        zero_count,
        nonzero_count,
        homozygous_reference_count,
        heterozygous_count,
        homozygous_alternate_count,
    })
}

#[allow(clippy::cast_precision_loss)]
fn impute_variant_major_row_if_needed(
    output_row: &mut [f32],
    selected_dosage_total: f32,
    selected_observation_count: i32,
    has_missing_values: bool,
) {
    if !has_missing_values {
        return;
    }
    let imputed_dosage_value = selected_dosage_total / selected_observation_count.max(1) as f32;
    for output_value in output_row {
        if output_value.is_nan() {
            *output_value = imputed_dosage_value;
        }
    }
}
