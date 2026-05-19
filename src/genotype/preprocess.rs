//! Host-side genotype preprocessing shared by native readers.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unreadable_literal)]

use crate::genotype::common::{ChunkStats, GenotypeError};

const NONZERO_DOSAGE_THRESHOLD: f32 = 1.0e-4;
const HETEROZYGOUS_DOSAGE_THRESHOLD: f32 = 0.5;
const HOMOZYGOUS_ALTERNATE_DOSAGE_THRESHOLD: f32 = 1.5;
const SPARSE_ZERO_DENSITY_THRESHOLD: f32 = 0.5;
const RARE_SPARSE_FIRTH_MINOR_ALLELE_COUNT_THRESHOLD: f32 = 50.0;

pub fn preprocess_row_major_dosage_matrix(
    dosage_values: &mut [f32],
    selected_sample_count: usize,
    selected_variant_count: usize,
) -> Result<ChunkStats, GenotypeError> {
    let expected_value_count = selected_sample_count.checked_mul(selected_variant_count).ok_or_else(|| {
        GenotypeError::InvalidInput("Integer overflow while validating genotype matrix shape.".to_string())
    })?;
    if dosage_values.len() != expected_value_count {
        return Err(GenotypeError::InvalidInput(format!(
            "Genotype matrix value count mismatch: expected {expected_value_count}, observed {}.",
            dosage_values.len(),
        )));
    }

    let mut dosage_sum = vec![0.0_f32; selected_variant_count];
    let mut dosage_square_sum = vec![0.0_f32; selected_variant_count];
    let mut observation_count = vec![0_i32; selected_variant_count];
    let mut zero_count = vec![0_i32; selected_variant_count];
    let mut nonzero_count = vec![0_i32; selected_variant_count];
    let mut homozygous_reference_count = vec![0_i32; selected_variant_count];
    let mut heterozygous_count = vec![0_i32; selected_variant_count];
    let mut homozygous_alternate_count = vec![0_i32; selected_variant_count];
    let mut has_missing_values = false;

    for sample_index in 0..selected_sample_count {
        let row_offset = sample_index
            .checked_mul(selected_variant_count)
            .ok_or_else(|| GenotypeError::InvalidInput("Integer overflow while scanning genotype rows.".to_string()))?;
        for variant_index in 0..selected_variant_count {
            let dosage_value = dosage_values[row_offset + variant_index];
            if dosage_value.is_nan() {
                has_missing_values = true;
                continue;
            }
            dosage_sum[variant_index] += dosage_value;
            dosage_square_sum[variant_index] += dosage_value * dosage_value;
            observation_count[variant_index] += 1;
            increment_dosage_summary_counts(
                dosage_value,
                &mut zero_count[variant_index],
                &mut nonzero_count[variant_index],
                &mut homozygous_reference_count[variant_index],
                &mut heterozygous_count[variant_index],
                &mut homozygous_alternate_count[variant_index],
            );
        }
    }

    let stats = build_chunk_stats_from_summaries(
        dosage_sum,
        dosage_square_sum,
        observation_count,
        zero_count,
        nonzero_count,
        homozygous_reference_count,
        heterozygous_count,
        homozygous_alternate_count,
        has_missing_values,
        selected_sample_count,
    );

    if has_missing_values {
        for sample_index in 0..selected_sample_count {
            let row_offset = sample_index.checked_mul(selected_variant_count).ok_or_else(|| {
                GenotypeError::InvalidInput("Integer overflow while imputing genotype rows.".to_string())
            })?;
            for variant_index in 0..selected_variant_count {
                let output_value = &mut dosage_values[row_offset + variant_index];
                if output_value.is_nan() {
                    *output_value =
                        stats.dosage_sum[variant_index] / stats.observation_count[variant_index].max(1) as f32;
                }
            }
        }
    }

    Ok(stats)
}

pub fn summarize_variant_major_dosage_matrix(
    dosage_values: &[f32],
    selected_sample_count: usize,
    selected_variant_count: usize,
) -> Result<ChunkStats, GenotypeError> {
    let expected_value_count = selected_sample_count.checked_mul(selected_variant_count).ok_or_else(|| {
        GenotypeError::InvalidInput("Integer overflow while validating variant-major genotype shape.".to_string())
    })?;
    if dosage_values.len() != expected_value_count {
        return Err(GenotypeError::InvalidInput(format!(
            "Variant-major genotype value count mismatch: expected {expected_value_count}, observed {}.",
            dosage_values.len(),
        )));
    }

    let mut dosage_sum = vec![0.0_f32; selected_variant_count];
    let mut dosage_square_sum = vec![0.0_f32; selected_variant_count];
    let mut observation_count = vec![0_i32; selected_variant_count];
    let mut zero_count = vec![0_i32; selected_variant_count];
    let mut nonzero_count = vec![0_i32; selected_variant_count];
    let mut homozygous_reference_count = vec![0_i32; selected_variant_count];
    let mut heterozygous_count = vec![0_i32; selected_variant_count];
    let mut homozygous_alternate_count = vec![0_i32; selected_variant_count];
    let mut has_missing_values = false;

    for variant_index in 0..selected_variant_count {
        let row_offset = variant_index.checked_mul(selected_sample_count).ok_or_else(|| {
            GenotypeError::InvalidInput("Integer overflow while scanning variant-major rows.".to_string())
        })?;
        for sample_index in 0..selected_sample_count {
            let dosage_value = dosage_values[row_offset + sample_index];
            if dosage_value.is_nan() {
                has_missing_values = true;
                continue;
            }
            dosage_sum[variant_index] += dosage_value;
            dosage_square_sum[variant_index] += dosage_value * dosage_value;
            observation_count[variant_index] += 1;
            increment_dosage_summary_counts(
                dosage_value,
                &mut zero_count[variant_index],
                &mut nonzero_count[variant_index],
                &mut homozygous_reference_count[variant_index],
                &mut heterozygous_count[variant_index],
                &mut homozygous_alternate_count[variant_index],
            );
        }
    }

    Ok(build_chunk_stats_from_summaries(
        dosage_sum,
        dosage_square_sum,
        observation_count,
        zero_count,
        nonzero_count,
        homozygous_reference_count,
        heterozygous_count,
        homozygous_alternate_count,
        has_missing_values,
        selected_sample_count,
    ))
}

#[must_use]
pub fn build_empty_chunk_stats(selected_variant_count: usize, has_missing_values: bool) -> ChunkStats {
    ChunkStats {
        allele_one_frequency: vec![0.0_f32; selected_variant_count],
        observation_count: vec![0_i32; selected_variant_count],
        has_missing_values,
        dosage_sum: vec![0.0_f32; selected_variant_count],
        dosage_square_sum: vec![0.0_f32; selected_variant_count],
        imputed_dosage_square_sum: vec![0.0_f32; selected_variant_count],
        dosage_variance_numerator: vec![0.0_f32; selected_variant_count],
        info_score: vec![None; selected_variant_count],
        allele_count: vec![0.0_f32; selected_variant_count],
        minor_allele_count: vec![0.0_f32; selected_variant_count],
        zero_count: vec![0_i32; selected_variant_count],
        nonzero_count: vec![0_i32; selected_variant_count],
        homozygous_reference_count: vec![0_i32; selected_variant_count],
        heterozygous_count: vec![0_i32; selected_variant_count],
        homozygous_alternate_count: vec![0_i32; selected_variant_count],
        is_sparse_candidate: vec![false; selected_variant_count],
        is_rare_sparse_firth_candidate: vec![false; selected_variant_count],
    }
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn build_chunk_stats_from_summaries(
    dosage_sum: Vec<f32>,
    dosage_square_sum: Vec<f32>,
    observation_count: Vec<i32>,
    zero_count: Vec<i32>,
    nonzero_count: Vec<i32>,
    homozygous_reference_count: Vec<i32>,
    heterozygous_count: Vec<i32>,
    homozygous_alternate_count: Vec<i32>,
    has_missing_values: bool,
    selected_sample_count: usize,
) -> ChunkStats {
    let selected_variant_count = observation_count.len();
    let mut allele_one_frequency = Vec::with_capacity(selected_variant_count);
    let mut imputed_dosage_square_sum = Vec::with_capacity(selected_variant_count);
    let mut dosage_variance_numerator = Vec::with_capacity(selected_variant_count);
    let mut info_score = Vec::with_capacity(selected_variant_count);
    let mut minor_allele_count = Vec::with_capacity(selected_variant_count);
    let mut is_sparse_candidate = Vec::with_capacity(selected_variant_count);
    let mut is_rare_sparse_firth_candidate = Vec::with_capacity(selected_variant_count);

    for variant_index in 0..selected_variant_count {
        let count = observation_count[variant_index];
        if count <= 0 {
            allele_one_frequency.push(0.0);
            imputed_dosage_square_sum.push(0.0);
            dosage_variance_numerator.push(0.0);
            info_score.push(None);
            minor_allele_count.push(0.0);
            is_sparse_candidate.push(false);
            is_rare_sparse_firth_candidate.push(false);
            continue;
        }

        let count_float = count as f32;
        let dosage_mean = dosage_sum[variant_index] / count_float;
        let missing_count = i32::try_from(selected_sample_count).unwrap_or(i32::MAX).saturating_sub(count).max(0);
        let current_imputed_dosage_square_sum =
            dosage_square_sum[variant_index] + (missing_count as f32 * dosage_mean * dosage_mean);
        let allele_frequency = dosage_mean / 2.0;
        let variance_numerator =
            (dosage_square_sum[variant_index] - (dosage_sum[variant_index] * dosage_mean)).max(0.0);
        let expected_variance_numerator = count_float * 2.0 * allele_frequency * (1.0 - allele_frequency);
        let current_info_score = if expected_variance_numerator > 0.0 {
            Some((variance_numerator / expected_variance_numerator).clamp(0.0, 1.0))
        } else {
            None
        };
        let allele_count = dosage_sum[variant_index];
        let reference_allele_count = (2.0 * count_float) - allele_count;
        let current_minor_allele_count = allele_count.min(reference_allele_count);
        let zero_density = zero_count[variant_index] as f32 / count_float;
        let current_sparse_candidate = zero_density >= SPARSE_ZERO_DENSITY_THRESHOLD;

        allele_one_frequency.push(allele_frequency);
        imputed_dosage_square_sum.push(current_imputed_dosage_square_sum);
        dosage_variance_numerator.push(variance_numerator);
        info_score.push(current_info_score);
        minor_allele_count.push(current_minor_allele_count);
        is_sparse_candidate.push(current_sparse_candidate);
        is_rare_sparse_firth_candidate.push(
            current_sparse_candidate && current_minor_allele_count < RARE_SPARSE_FIRTH_MINOR_ALLELE_COUNT_THRESHOLD,
        );
    }

    ChunkStats {
        allele_one_frequency,
        observation_count,
        has_missing_values,
        dosage_sum: dosage_sum.clone(),
        dosage_square_sum,
        imputed_dosage_square_sum,
        dosage_variance_numerator,
        info_score,
        allele_count: dosage_sum,
        minor_allele_count,
        zero_count,
        nonzero_count,
        homozygous_reference_count,
        heterozygous_count,
        homozygous_alternate_count,
        is_sparse_candidate,
        is_rare_sparse_firth_candidate,
    }
}

pub fn increment_dosage_summary_counts(
    dosage_value: f32,
    zero_count: &mut i32,
    nonzero_count: &mut i32,
    homozygous_reference_count: &mut i32,
    heterozygous_count: &mut i32,
    homozygous_alternate_count: &mut i32,
) {
    if dosage_value > NONZERO_DOSAGE_THRESHOLD {
        *nonzero_count += 1;
    } else {
        *zero_count += 1;
    }
    if dosage_value < HETEROZYGOUS_DOSAGE_THRESHOLD {
        *homozygous_reference_count += 1;
    } else if dosage_value < HOMOZYGOUS_ALTERNATE_DOSAGE_THRESHOLD {
        *heterozygous_count += 1;
    } else {
        *homozygous_alternate_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{preprocess_row_major_dosage_matrix, summarize_variant_major_dosage_matrix};

    #[test]
    fn preprocess_imputes_missing_values_and_computes_stats() {
        let mut dosage_values = vec![0.0, f32::NAN, 2.0, 1.0, 2.0, f32::NAN];

        let stats = preprocess_row_major_dosage_matrix(&mut dosage_values, 3, 2).expect("preprocess should succeed");

        assert_eq!(dosage_values, vec![0.0, 1.0, 2.0, 1.0, 2.0, 1.0]);
        assert_eq!(stats.observation_count, vec![3, 1]);
        assert_eq!(stats.allele_one_frequency, vec![2.0 / 3.0, 0.5]);
        assert_eq!(stats.dosage_square_sum, vec![8.0, 1.0]);
        assert_eq!(stats.imputed_dosage_square_sum, vec![8.0, 3.0]);
        assert_eq!(stats.zero_count, vec![1, 0]);
        assert_eq!(stats.nonzero_count, vec![2, 1]);
        assert_eq!(stats.homozygous_reference_count, vec![1, 0]);
        assert_eq!(stats.heterozygous_count, vec![0, 1]);
        assert_eq!(stats.homozygous_alternate_count, vec![2, 0]);
        assert!(stats.has_missing_values);
    }

    #[test]
    fn preprocess_handles_all_missing_column_as_zero_mean() {
        let mut dosage_values = vec![f32::NAN, 1.0, f32::NAN, 2.0];

        let stats = preprocess_row_major_dosage_matrix(&mut dosage_values, 2, 2).expect("preprocess should succeed");

        assert_eq!(dosage_values, vec![0.0, 1.0, 0.0, 2.0]);
        assert_eq!(stats.observation_count, vec![0, 2]);
        assert_eq!(stats.allele_one_frequency, vec![0.0, 0.75]);
        assert_eq!(stats.dosage_square_sum, vec![0.0, 5.0]);
        assert_eq!(stats.imputed_dosage_square_sum, vec![0.0, 5.0]);
        assert_eq!(stats.info_score[0], None);
        assert!(stats.has_missing_values);
    }

    #[test]
    fn summarize_variant_major_computes_info_and_sparse_flags() {
        let dosage_values = vec![0.0, 0.0, 1.0, 2.0, 0.0, 2.0];

        let stats = summarize_variant_major_dosage_matrix(&dosage_values, 3, 2).expect("stats should compute");

        assert_eq!(stats.observation_count, vec![3, 3]);
        assert_eq!(stats.allele_count, vec![1.0, 4.0]);
        assert_eq!(stats.dosage_square_sum, vec![1.0, 8.0]);
        assert_eq!(stats.imputed_dosage_square_sum, vec![1.0, 8.0]);
        assert_eq!(stats.minor_allele_count, vec![1.0, 2.0]);
        assert_eq!(stats.zero_count, vec![2, 1]);
        assert_eq!(stats.nonzero_count, vec![1, 2]);
        assert!(stats.is_sparse_candidate[0]);
        assert!(stats.is_rare_sparse_firth_candidate[0]);
        assert_eq!(stats.info_score, vec![Some(0.799_999_95), Some(1.0)]);
    }
}
