//! Host-side genotype preprocessing shared by native readers.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unreadable_literal)]

use g_genotype_contracts::{ChunkOutputStatistics, NullableFloat32Column};

use crate::common::{ChunkComputeStatistics, ChunkStatisticsPolicy, ChunkStats};
use crate::error::{GenotypeError, GenotypeResult};

const ZERO_DOSAGE_UPPER_BOUND: f32 = 1.0e-4;
const HOMOZYGOUS_ALTERNATE_DOSAGE_THRESHOLD: f32 = 1.5;
const SPARSE_ZERO_DENSITY_THRESHOLD: f32 = 0.5;
const RARE_SPARSE_FIRTH_MINOR_ALLELE_COUNT_THRESHOLD: f32 = 50.0;

#[cfg(test)]
pub(crate) fn preprocess_row_major_dosage_matrix(
    dosage_values: &mut [f32],
    selected_sample_count: usize,
    selected_variant_count: usize,
) -> GenotypeResult<ChunkStats> {
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
        homozygous_alternate_count,
        selected_sample_count,
    )?;

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

#[must_use]
pub(crate) fn build_empty_chunk_stats(
    selected_variant_count: usize,
    statistics_policy: ChunkStatisticsPolicy,
) -> ChunkStats {
    ChunkStats {
        output: ChunkOutputStatistics {
            allele_one_frequency: vec![0.0_f32; selected_variant_count],
            observation_count: vec![0_i32; selected_variant_count],
            info_score: NullableFloat32Column {
                values: vec![0.0_f32; selected_variant_count],
                validity_bytes: vec![0_u8; selected_variant_count.div_ceil(8)],
            },
        },
        compute: ChunkComputeStatistics {
            genotype_mean: vec![0.0_f32; selected_variant_count],
            imputed_dosage_square_sum: statistics_policy
                .retain_imputed_dosage_square_sum
                .then(|| vec![0.0_f32; selected_variant_count]),
            sparse_candidate_mask: statistics_policy
                .collect_sparse_candidate_mask
                .then(|| vec![false; selected_variant_count]),
        },
    }
}

pub(crate) fn build_chunk_stats_from_summaries(
    mut dosage_sum: Vec<f32>,
    mut dosage_square_sum: Vec<f32>,
    observation_count: Vec<i32>,
    zero_count: Option<Vec<i32>>,
    homozygous_alternate_count: Option<Vec<i32>>,
    selected_sample_count: usize,
    statistics_policy: ChunkStatisticsPolicy,
) -> GenotypeResult<ChunkStats> {
    let selected_variant_count = observation_count.len();
    let selected_sample_count_i32 = i32::try_from(selected_sample_count).map_err(|_| {
        GenotypeError::InvalidInput(format!(
            "Selected sample count {selected_sample_count} exceeds the supported i32 statistics range.",
        ))
    })?;
    let mut allele_one_frequency = Vec::with_capacity(selected_variant_count);
    let mut info_score = NullableFloat32Column {
        values: Vec::with_capacity(selected_variant_count),
        validity_bytes: Vec::with_capacity(selected_variant_count.div_ceil(8)),
    };
    let mut sparse_candidate_mask =
        statistics_policy.collect_sparse_candidate_mask.then(|| Vec::with_capacity(selected_variant_count));
    let sparse_candidate_counts = zero_count.as_ref().zip(homozygous_alternate_count.as_ref());
    if statistics_policy.collect_sparse_candidate_mask != sparse_candidate_counts.is_some() {
        return Err(GenotypeError::InvalidInput(
            "Sparse candidate count buffers do not match the requested statistics policy.".to_string(),
        ));
    }

    for variant_index in 0..selected_variant_count {
        let count = observation_count[variant_index];
        if count <= 0 {
            allele_one_frequency.push(0.0);
            dosage_sum[variant_index] = 0.0;
            if statistics_policy.retain_imputed_dosage_square_sum {
                dosage_square_sum[variant_index] = 0.0;
            }
            info_score.push(0.0, false);
            if let Some(sparse_candidate_mask) = sparse_candidate_mask.as_mut() {
                sparse_candidate_mask.push(false);
            }
            continue;
        }

        let count_float = count as f32;
        let dosage_mean = dosage_sum[variant_index] / count_float;
        let observed_dosage_square_sum = dosage_square_sum[variant_index];
        if statistics_policy.retain_imputed_dosage_square_sum {
            let missing_count = selected_sample_count_i32.checked_sub(count).ok_or_else(|| {
                GenotypeError::InvalidInput(format!(
                    "Variant observation count {count} exceeds selected sample count {selected_sample_count_i32}."
                ))
            })?;
            dosage_square_sum[variant_index] =
                observed_dosage_square_sum + (missing_count as f32 * dosage_mean * dosage_mean);
        }
        let allele_frequency = dosage_mean / 2.0;
        let variance_numerator = (observed_dosage_square_sum - (dosage_sum[variant_index] * dosage_mean)).max(0.0);
        // INFO is currently defined on observed genotype calls. Missing calls are
        // mean-imputed for downstream dosage sums, but not for the expected
        // Hardy-Weinberg variance denominator.
        let expected_variance_numerator = count_float * 2.0 * allele_frequency * (1.0 - allele_frequency);
        if expected_variance_numerator > 0.0 {
            info_score.push((variance_numerator / expected_variance_numerator).clamp(0.0, 1.0), true);
        } else {
            info_score.push(0.0, false);
        }
        allele_one_frequency.push(allele_frequency);
        if let (Some(sparse_candidate_mask), Some((zero_count, homozygous_alternate_count))) =
            (sparse_candidate_mask.as_mut(), sparse_candidate_counts)
        {
            let allele_count = dosage_sum[variant_index];
            let reference_allele_count = (2.0 * count_float) - allele_count;
            let current_minor_allele_count = allele_count.min(reference_allele_count);
            let regenie_flipped_zero_count = if allele_count > reference_allele_count {
                homozygous_alternate_count[variant_index]
            } else {
                zero_count[variant_index]
            };
            let zero_density = regenie_flipped_zero_count as f32 / count_float;
            sparse_candidate_mask.push(
                zero_density >= SPARSE_ZERO_DENSITY_THRESHOLD
                    && current_minor_allele_count < RARE_SPARSE_FIRTH_MINOR_ALLELE_COUNT_THRESHOLD,
            );
        }
        dosage_sum[variant_index] = dosage_mean;
    }

    Ok(ChunkStats {
        output: ChunkOutputStatistics { allele_one_frequency, observation_count, info_score },
        compute: ChunkComputeStatistics {
            genotype_mean: dosage_sum,
            imputed_dosage_square_sum: statistics_policy.retain_imputed_dosage_square_sum.then_some(dosage_square_sum),
            sparse_candidate_mask,
        },
    })
}

pub(crate) fn increment_sparse_candidate_counts(
    dosage_value: f32,
    zero_count: &mut i32,
    homozygous_alternate_count: &mut i32,
) {
    if dosage_value <= ZERO_DOSAGE_UPPER_BOUND {
        *zero_count += 1;
    }
    if dosage_value >= HOMOZYGOUS_ALTERNATE_DOSAGE_THRESHOLD {
        *homozygous_alternate_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        preprocess_row_major_dosage_matrix, summarize_variant_major_dosage_matrix, summarize_variant_major_row_scalar,
        summarize_variant_major_row_simd_or_scalar,
    };

    const SUMMARY_SAMPLE_COUNTS: [usize; 10] = [0, 1, 7, 8, 15, 16, 17, 31, 32, 33];

    fn assert_row_summaries_close(
        left: super::VariantMajorRowSummary,
        right: super::VariantMajorRowSummary,
        sample_count: usize,
    ) {
        let tolerance = sample_count as f32 * 1.0e-6;
        assert!((left.dosage_sum - right.dosage_sum).abs() <= tolerance);
        assert!((left.dosage_square_sum - right.dosage_square_sum).abs() <= tolerance * 4.0);
        assert_eq!(left.observation_count, right.observation_count);
        assert_eq!(left.zero_count, right.zero_count);
        assert_eq!(left.nonzero_count, right.nonzero_count);
        assert_eq!(left.homozygous_reference_count, right.homozygous_reference_count);
        assert_eq!(left.heterozygous_count, right.heterozygous_count);
        assert_eq!(left.homozygous_alternate_count, right.homozygous_alternate_count);
        assert_eq!(left.has_missing_values, right.has_missing_values);
    }

    fn deterministic_dosage_values(sample_count: usize) -> Vec<f32> {
        let mut dosage_values = Vec::with_capacity(sample_count);
        for sample_index in 0..sample_count {
            let raw_value = ((sample_index * 37) + 11) % 511;
            dosage_values.push(raw_value as f32 / 255.0_f32);
        }
        dosage_values
    }

    fn dosage_patterns(sample_count: usize) -> [Vec<f32>; 5] {
        [
            vec![0.0_f32; sample_count],
            vec![2.0_f32; sample_count],
            (0..sample_count).map(|sample_index| if sample_index % 2 == 0 { 0.0_f32 } else { 2.0_f32 }).collect(),
            (0..sample_count)
                .map(|sample_index| match sample_index % 5 {
                    0 => f32::NAN,
                    1 => 0.0_f32,
                    2 => 0.499_f32,
                    3 => 1.499_f32,
                    _ => 1.5_f32,
                })
                .collect(),
            deterministic_dosage_values(sample_count),
        ]
    }

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
        assert_eq!(stats.allele_count.as_ref(), [1.0, 4.0]);
        assert!(std::sync::Arc::ptr_eq(&stats.dosage_sum, &stats.allele_count));
        assert_eq!(stats.dosage_square_sum, vec![1.0, 8.0]);
        assert_eq!(stats.imputed_dosage_square_sum, vec![1.0, 8.0]);
        assert_eq!(stats.minor_allele_count, vec![1.0, 2.0]);
        assert_eq!(stats.zero_count, vec![2, 1]);
        assert_eq!(stats.nonzero_count, vec![1, 2]);
        assert!(stats.is_sparse_candidate[0]);
        assert!(stats.is_rare_sparse_firth_candidate[0]);
        assert!(stats.is_sparse_candidate[1]);
        assert!(stats.is_rare_sparse_firth_candidate[1]);
        assert_eq!(stats.info_score, vec![Some(0.799_999_95), Some(1.0)]);
    }

    #[test]
    fn summarize_variant_major_info_score_uses_observed_count_with_missing_values() {
        let dosage_values = vec![0.0, 1.0, f32::NAN, f32::NAN];

        let stats = summarize_variant_major_dosage_matrix(&dosage_values, 4, 1).expect("stats should compute");

        assert_eq!(stats.observation_count, vec![2]);
        assert!(stats.has_missing_values);
        assert_eq!(stats.imputed_dosage_square_sum, vec![1.5]);
        let info_score = stats.info_score[0].expect("partly observed variant should have an INFO score");
        assert!((info_score - (2.0 / 3.0)).abs() <= 1.0e-6);
        assert!((info_score - (1.0 / 3.0)).abs() > 1.0e-6);
    }

    #[test]
    fn variant_major_row_summary_simd_matches_scalar() {
        for sample_count in SUMMARY_SAMPLE_COUNTS {
            for dosage_values in dosage_patterns(sample_count) {
                let scalar_summary = summarize_variant_major_row_scalar(&dosage_values);
                let simd_summary = summarize_variant_major_row_simd_or_scalar(&dosage_values);
                assert_row_summaries_close(simd_summary, scalar_summary, sample_count);
            }
        }
    }
}
