use std::time::Instant;

use super::super::metadata::VariantRecord;
use super::super::profile::{ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use super::super::sample_selection::SampleSelection;
use super::super::simd;
use super::super::{BgenError, CompressionType};
use super::matrix::{
    DosageTileDecodeResult, MISSING_SAMPLE_FLAG_MASK, PLOIDY_MASK, RowMajorOutputColumnMut, RowMajorOutputMatrix,
    ThreadScratch, VariantDecodeResult, build_variant_decode_result, exact_eight_bit_probability_pairs,
    packed_eight_bit_probability_index, record_variant_decode_if_enabled, unphased_eight_bit_dosage_lookup,
};
use super::probability::{
    PackedProbabilityReader, read_exact_bytes, read_probability_block, read_u8_at, read_u16_at, read_u32_at,
    u32_to_usize,
};
use crate::buffer::raw_pointer::OutputBufferAddress;

#[allow(clippy::too_many_arguments)]
pub(in crate::bgen) fn decode_variant_dosage_tile_into_row_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_pointer_address: OutputBufferAddress,
    selected_variant_count: usize,
    tile_variant_start_index: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    collect_dosage_totals: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<DosageTileDecodeResult, BgenError> {
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

    let tile_pointer_address = OutputBufferAddress::from_mut_ptr(thread_scratch.dosage_tile.as_mut_ptr());
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let mut selected_dosage_totals = if collect_dosage_totals { vec![0.0_f32; tile_variant_count] } else { Vec::new() };
    for (tile_variant_index, variant_record) in variant_record_chunk.iter().enumerate() {
        let variant_decode_result = decode_variant_dosages_into_row_major_matrix(
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
            collect_dosage_totals,
            thread_scratch,
        )?;
        if collect_dosage_totals {
            selected_dosage_totals[tile_variant_index] = variant_decode_result.selected_dosage_total;
        }
        if profiling_enabled {
            thread_local_profile_snapshot.merge_from(&variant_decode_result.profile_snapshot);
        }
    }
    if profiling_enabled {
        thread_local_profile_snapshot.decode_tile_count += 1;
    }

    let copy_tile_start_time = profiling_enabled.then(Instant::now);
    let mut output_matrix = unsafe {
        RowMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            selected_variant_count,
            "row-major BGEN dosage",
        )?
    };
    for selected_sample_index in 0..sample_selection.selected_sample_count {
        let tile_row_start = selected_sample_index * tile_variant_count;
        let tile_row_stop = tile_row_start + tile_variant_count;
        let output_row_range =
            output_matrix.row_range_mut(selected_sample_index, tile_variant_start_index, tile_variant_count)?;
        output_row_range.copy_from_slice(&thread_scratch.dosage_tile[tile_row_start..tile_row_stop]);
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

    Ok(DosageTileDecodeResult { profile_snapshot: thread_local_profile_snapshot, selected_dosage_totals })
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn decode_variant_dosages_into_row_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: OutputBufferAddress,
    variant_index: usize,
    variant_count: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    collect_dosage_totals: bool,
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
            collect_dosage_totals,
            thread_local_profile_snapshot,
        );
    }

    let probability_scale_denominator =
        if probability_bit_count == 32 { f64::from(u32::MAX) } else { f64::from((1_u32 << probability_bit_count) - 1) };
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let mut bit_reader = PackedProbabilityReader::new(&block_bytes[cursor..]);
    let mut output_matrix = unsafe {
        RowMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            variant_count,
            "row-major BGEN dosage",
        )?
    };
    let mut output_column = output_matrix.column_mut(variant_index)?;
    let mut selected_dosage_total = 0.0_f32;
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

            unsafe {
                // Identity-aligned full-sample reads write one validated variant column across file-order rows.
                output_column.write_unchecked(file_sample_index, dosage_value);
            }
            if collect_dosage_totals && !dosage_value.is_nan() {
                selected_dosage_total += dosage_value;
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
        record_variant_decode_if_enabled(&mut thread_local_profile_snapshot, profiling_enabled);

        return Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total));
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
            unsafe {
                // Selected indices are built by sample selection and map to valid output rows.
                output_column.write_unchecked(selected_index, dosage_value);
            }
            if collect_dosage_totals && !dosage_value.is_nan() {
                selected_dosage_total += dosage_value;
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

    record_variant_decode_if_enabled(&mut thread_local_profile_snapshot, profiling_enabled);

    Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn decode_unphased_eight_bit_dosages_into_row_major_matrix(
    sample_ploidy_and_missingness: &[u8],
    packed_probability_bytes: &[u8],
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: OutputBufferAddress,
    variant_index: usize,
    variant_count: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    collect_dosage_totals: bool,
    mut thread_local_profile_snapshot: ThreadLocalProfileSnapshot,
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

    let dosage_lookup = unphased_eight_bit_dosage_lookup();
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let mut output_matrix = unsafe {
        RowMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            variant_count,
            "row-major BGEN dosage",
        )?
    };
    let mut output_column = output_matrix.column_mut(variant_index)?;
    let probability_pairs =
        exact_eight_bit_probability_pairs(&packed_probability_bytes[..expected_probability_byte_count]);
    let all_samples_present =
        trusted_no_missing_diploid || simd::all_samples_present_diploid_simd_or_scalar(sample_ploidy_and_missingness);
    if trusted_no_missing_diploid {
        debug_assert!(
            simd::all_samples_present_diploid_simd_or_scalar(sample_ploidy_and_missingness),
            "trusted row-major decode skipped a ploidy scan before validation"
        );
    }
    if all_samples_present {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        let selected_dosage_total = decode_all_present_unphased_eight_bit_row_major_selection(
            probability_pairs,
            sample_selection,
            &mut output_column,
            dosage_lookup,
            collect_dosage_totals,
        )?;
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
        record_variant_decode_if_enabled(&mut thread_local_profile_snapshot, profiling_enabled);

        return Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total));
    }

    let mut selected_dosage_total = 0.0_f32;
    if sample_selection.is_identity {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        for (file_sample_index, (ploidy_and_missingness, probability_pair)) in
            sample_ploidy_and_missingness.iter().zip(probability_pairs.iter().copied()).enumerate()
        {
            let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
            if observed_ploidy != 2 {
                return Err(BgenError::UnsupportedFormat(format!(
                    "Variant '{}' contains a non-diploid sample in an identity-aligned full-sample read. Observed ploidy {observed_ploidy}.",
                    variant_record.resolved_variant_identifier,
                )));
            }

            let packed_probability_index = packed_eight_bit_probability_index(probability_pair);

            let dosage_value = if (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0 {
                f32::NAN
            } else {
                dosage_lookup[packed_probability_index]
            };
            unsafe {
                // Identity-aligned full-sample reads write file-order rows in the validated output column.
                output_column.write_unchecked(file_sample_index, dosage_value);
            }
            if collect_dosage_totals && !dosage_value.is_nan() {
                selected_dosage_total += dosage_value;
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
        record_variant_decode_if_enabled(&mut thread_local_profile_snapshot, profiling_enabled);

        return Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total));
    }

    let output_write_start_time = profiling_enabled.then(Instant::now);
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

        let packed_probability_index = packed_eight_bit_probability_index(probability_pair);

        let dosage_value = if (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0 {
            f32::NAN
        } else {
            dosage_lookup[packed_probability_index]
        };

        let selected_index = sample_selection.file_to_selected_index[file_sample_index];
        if selected_index != usize::MAX {
            unsafe {
                // Selected indices are built by sample selection and map to valid output rows.
                output_column.write_unchecked(selected_index, dosage_value);
            }
            if collect_dosage_totals && !dosage_value.is_nan() {
                selected_dosage_total += dosage_value;
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
    record_variant_decode_if_enabled(&mut thread_local_profile_snapshot, profiling_enabled);

    Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total))
}

fn decode_all_present_unphased_eight_bit_row_major_selection(
    probability_pairs: &[[u8; 2]],
    sample_selection: &SampleSelection,
    output_column: &mut RowMajorOutputColumnMut<'_, f32>,
    dosage_lookup: &[f32],
    collect_dosage_totals: bool,
) -> Result<f32, BgenError> {
    let mut selected_dosage_total = 0.0_f32;
    if sample_selection.is_identity {
        for (file_sample_index, probability_pair) in probability_pairs.iter().copied().enumerate() {
            let dosage_value = decode_unphased_eight_bit_row_major_probability_pair(probability_pair, dosage_lookup);
            unsafe {
                // Identity-aligned full-sample reads write file-order rows in the validated output column.
                output_column.write_unchecked(file_sample_index, dosage_value);
            }
            if collect_dosage_totals {
                selected_dosage_total += dosage_value;
            }
        }
        return Ok(selected_dosage_total);
    }

    if let Some(contiguous_file_index_start) = sample_selection.contiguous_file_index_start {
        selected_dosage_total = decode_contiguous_all_present_unphased_eight_bit_row_major_selection(
            probability_pairs,
            contiguous_file_index_start,
            sample_selection.selected_sample_count,
            output_column,
            dosage_lookup,
            collect_dosage_totals,
        )?;
        return Ok(selected_dosage_total);
    }

    if row_major_selection_prefers_sparse_indices(sample_selection, probability_pairs.len()) {
        selected_dosage_total = decode_sparse_all_present_unphased_eight_bit_row_major_selection(
            probability_pairs,
            sample_selection,
            output_column,
            dosage_lookup,
            collect_dosage_totals,
        )?;
    } else {
        selected_dosage_total = decode_dense_all_present_unphased_eight_bit_row_major_selection(
            probability_pairs,
            sample_selection,
            output_column,
            dosage_lookup,
            collect_dosage_totals,
        );
    }
    Ok(selected_dosage_total)
}

fn decode_unphased_eight_bit_row_major_probability_pair(probability_pair: [u8; 2], dosage_lookup: &[f32]) -> f32 {
    dosage_lookup[packed_eight_bit_probability_index(probability_pair)]
}

fn decode_contiguous_all_present_unphased_eight_bit_row_major_selection(
    probability_pairs: &[[u8; 2]],
    contiguous_file_index_start: usize,
    selected_sample_count: usize,
    output_column: &mut RowMajorOutputColumnMut<'_, f32>,
    dosage_lookup: &[f32],
    collect_dosage_totals: bool,
) -> Result<f32, BgenError> {
    let contiguous_file_index_stop = contiguous_file_index_start
        .checked_add(selected_sample_count)
        .ok_or_else(|| BgenError::Range("Integer overflow while slicing contiguous BGEN samples.".to_string()))?;
    let selected_probability_pairs =
        probability_pairs.get(contiguous_file_index_start..contiguous_file_index_stop).ok_or_else(|| {
            BgenError::Range("Contiguous BGEN sample selection exceeds decoded probability pairs.".to_string())
        })?;
    let mut selected_dosage_total = 0.0_f32;
    for (selected_index, probability_pair) in selected_probability_pairs.iter().copied().enumerate() {
        let dosage_value = decode_unphased_eight_bit_row_major_probability_pair(probability_pair, dosage_lookup);
        unsafe {
            // Contiguous selections write selected-order rows to the validated output column.
            output_column.write_unchecked(selected_index, dosage_value);
        }
        if collect_dosage_totals {
            selected_dosage_total += dosage_value;
        }
    }
    Ok(selected_dosage_total)
}

fn decode_sparse_all_present_unphased_eight_bit_row_major_selection(
    probability_pairs: &[[u8; 2]],
    sample_selection: &SampleSelection,
    output_column: &mut RowMajorOutputColumnMut<'_, f32>,
    dosage_lookup: &[f32],
    collect_dosage_totals: bool,
) -> Result<f32, BgenError> {
    let mut selected_dosage_total = 0.0_f32;
    for (selected_index, file_sample_index) in sample_selection.selected_file_indices.iter().copied().enumerate() {
        let probability_pair = probability_pairs.get(file_sample_index).copied().ok_or_else(|| {
            BgenError::Range("Sparse BGEN sample selection exceeds decoded probability pairs.".to_string())
        })?;
        let dosage_value = decode_unphased_eight_bit_row_major_probability_pair(probability_pair, dosage_lookup);
        unsafe {
            // Sparse selected indices are validated by sample selection and map to output rows.
            output_column.write_unchecked(selected_index, dosage_value);
        }
        if collect_dosage_totals {
            selected_dosage_total += dosage_value;
        }
    }
    Ok(selected_dosage_total)
}

fn decode_dense_all_present_unphased_eight_bit_row_major_selection(
    probability_pairs: &[[u8; 2]],
    sample_selection: &SampleSelection,
    output_column: &mut RowMajorOutputColumnMut<'_, f32>,
    dosage_lookup: &[f32],
    collect_dosage_totals: bool,
) -> f32 {
    let mut selected_dosage_total = 0.0_f32;
    for (file_sample_index, probability_pair) in probability_pairs.iter().copied().enumerate() {
        let selected_index = sample_selection.file_to_selected_index[file_sample_index];
        if selected_index == usize::MAX {
            continue;
        }
        let dosage_value = decode_unphased_eight_bit_row_major_probability_pair(probability_pair, dosage_lookup);
        unsafe {
            // Dense-mask selected indices are validated by sample selection and map to output rows.
            output_column.write_unchecked(selected_index, dosage_value);
        }
        if collect_dosage_totals {
            selected_dosage_total += dosage_value;
        }
    }
    selected_dosage_total
}

fn row_major_selection_prefers_sparse_indices(sample_selection: &SampleSelection, file_sample_count: usize) -> bool {
    sample_selection.selected_sample_count.saturating_mul(2) <= file_sample_count
}
