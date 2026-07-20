//! Host-side genotype preprocessing shared by native readers.

use g_genotype_contracts::{ChunkOutputStatistics, NullableFloat32Column};

use crate::common::{
    ChunkComputeStatistics, ChunkStatisticsPolicy, ChunkStats, EIGHT_BIT_PROBABILITY_SCALE_RECIPROCAL,
    EIGHT_BIT_PROBABILITY_SCALE_SQUARE_RECIPROCAL, Packed8RawStatistics,
};
use crate::error::{GenotypeError, GenotypeResult};

const ZERO_DOSAGE_UPPER_BOUND: f32 = 1.0e-4;
const HOMOZYGOUS_ALTERNATE_DOSAGE_THRESHOLD: f32 = 1.5;
const SPARSE_ZERO_DENSITY_THRESHOLD: f32 = 0.5;
const RARE_SPARSE_FIRTH_MINOR_ALLELE_COUNT_THRESHOLD: f32 = 50.0;
const MAXIMUM_PACKED8_RAW_DOSAGE: u64 = 510;
const MAXIMUM_PACKED8_RAW_DOSAGE_SQUARE: u64 = MAXIMUM_PACKED8_RAW_DOSAGE * MAXIMUM_PACKED8_RAW_DOSAGE;

struct OutputVariantStatistics {
    allele_one_frequency: f32,
    info_score: f32,
    info_score_is_valid: bool,
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

// This consumes one decode's statistics buffers. Moving the optional Vec
// handles preserves that ownership boundary without copying their storage.
#[allow(clippy::needless_pass_by_value)]
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
    validate_summary_length("dosage sum", dosage_sum.len(), selected_variant_count)?;
    validate_summary_length("dosage square sum", dosage_square_sum.len(), selected_variant_count)?;
    let selected_sample_count_i32 = checked_selected_sample_count(selected_sample_count)?;
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
    if let Some((zero_count, homozygous_alternate_count)) = sparse_candidate_counts {
        validate_summary_length("zero count", zero_count.len(), selected_variant_count)?;
        validate_summary_length(
            "homozygous alternate count",
            homozygous_alternate_count.len(),
            selected_variant_count,
        )?;
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

        // Association statistics are intentionally computed in the f32 output
        // domain after the sample count has been validated within i32 bounds.
        #[allow(clippy::cast_precision_loss)]
        let count_float = count as f32;
        let dosage_mean = dosage_sum[variant_index] / count_float;
        let observed_dosage_square_sum = dosage_square_sum[variant_index];
        if statistics_policy.retain_imputed_dosage_square_sum {
            let missing_count = selected_sample_count_i32.checked_sub(count).ok_or_else(|| {
                GenotypeError::InvalidInput(format!(
                    "Variant observation count {count} exceeds selected sample count {selected_sample_count_i32}."
                ))
            })?;
            // Both counts share the enforced i32 bound and the surrounding
            // dosage-square calculation intentionally uses f32.
            #[allow(clippy::cast_precision_loss)]
            let missing_count_float = missing_count as f32;
            dosage_square_sum[variant_index] =
                observed_dosage_square_sum + (missing_count_float * dosage_mean * dosage_mean);
        }
        let output_statistics = calculate_output_variant_statistics(
            dosage_sum[variant_index],
            observed_dosage_square_sum,
            count_float,
            dosage_mean,
        );
        allele_one_frequency.push(output_statistics.allele_one_frequency);
        info_score.push(output_statistics.info_score, output_statistics.info_score_is_valid);
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
            // This exact bounded count is converted only to compute a density
            // in the f32 statistics domain.
            #[allow(clippy::cast_precision_loss)]
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

impl Packed8RawStatistics {
    /// Validate device status and build exact output-facing packed8 statistics.
    ///
    /// # Errors
    ///
    /// Returns an error when a device row failed, column lengths differ, the
    /// selected sample count exceeds the output domain, or integer summaries
    /// exceed the bounds implied by the selected sample count.
    pub fn into_output_statistics(self) -> GenotypeResult<ChunkOutputStatistics> {
        let Self { dosage_sums, dosage_square_sums, statuses, selected_sample_count } = self;
        let variant_count = statuses.len();
        validate_summary_length("packed8 dosage sum", dosage_sums.len(), variant_count)?;
        validate_summary_length("packed8 dosage square sum", dosage_square_sums.len(), variant_count)?;
        if let Some((variant_index, status)) = statuses.iter().copied().enumerate().find(|(_, status)| *status != 0) {
            return Err(GenotypeError::InvalidInput(format!(
                "Compressed packed8 device row {variant_index} failed with status 0x{status:08x}."
            )));
        }
        let output_observation_count = checked_selected_sample_count(selected_sample_count)?;
        let bound_sample_count = u64::try_from(selected_sample_count).map_err(|_| {
            GenotypeError::InvalidInput(format!(
                "Selected sample count {selected_sample_count} exceeds the supported u64 statistics range."
            ))
        })?;
        let maximum_dosage_sum = bound_sample_count
            .checked_mul(MAXIMUM_PACKED8_RAW_DOSAGE)
            .ok_or_else(|| GenotypeError::InvalidInput("Packed8 dosage-sum bound overflowed u64.".to_string()))?;
        let maximum_dosage_square_sum =
            bound_sample_count.checked_mul(MAXIMUM_PACKED8_RAW_DOSAGE_SQUARE).ok_or_else(|| {
                GenotypeError::InvalidInput("Packed8 dosage-square-sum bound overflowed u64.".to_string())
            })?;
        let mut allele_one_frequency = Vec::with_capacity(variant_count);
        let observation_count = vec![output_observation_count; variant_count];
        let mut info_score = NullableFloat32Column {
            values: Vec::with_capacity(variant_count),
            validity_bytes: Vec::with_capacity(variant_count.div_ceil(8)),
        };
        for (variant_index, (raw_dosage_sum, raw_dosage_square_sum)) in
            dosage_sums.into_iter().zip(dosage_square_sums).enumerate()
        {
            if raw_dosage_sum > maximum_dosage_sum || raw_dosage_square_sum > maximum_dosage_square_sum {
                return Err(GenotypeError::InvalidInput(format!(
                    "Compressed packed8 device row {variant_index} returned summaries outside the selected-sample bounds."
                )));
            }
            if output_observation_count == 0 {
                allele_one_frequency.push(0.0);
                info_score.push(0.0, false);
                continue;
            }
            // Device summaries are exact bounded integers; conversion happens
            // once at the documented f32 association-output boundary.
            #[allow(clippy::cast_precision_loss)]
            let dosage_sum = raw_dosage_sum as f32 * EIGHT_BIT_PROBABILITY_SCALE_RECIPROCAL;
            #[allow(clippy::cast_precision_loss)]
            let dosage_square_sum = raw_dosage_square_sum as f32 * EIGHT_BIT_PROBABILITY_SCALE_SQUARE_RECIPROCAL;
            #[allow(clippy::cast_precision_loss)]
            let count_float = output_observation_count as f32;
            let dosage_mean = dosage_sum / count_float;
            let output_statistics =
                calculate_output_variant_statistics(dosage_sum, dosage_square_sum, count_float, dosage_mean);
            allele_one_frequency.push(output_statistics.allele_one_frequency);
            info_score.push(output_statistics.info_score, output_statistics.info_score_is_valid);
        }
        Ok(ChunkOutputStatistics { allele_one_frequency, observation_count, info_score })
    }
}

#[inline]
fn calculate_output_variant_statistics(
    dosage_sum: f32,
    observed_dosage_square_sum: f32,
    count_float: f32,
    dosage_mean: f32,
) -> OutputVariantStatistics {
    let allele_one_frequency = dosage_mean / 2.0;
    let variance_numerator = (observed_dosage_square_sum - (dosage_sum * dosage_mean)).max(0.0);
    // INFO is defined on observed genotype calls. Missing calls are mean
    // imputed for association input, not the Hardy-Weinberg denominator.
    let expected_variance_numerator = count_float * 2.0 * allele_one_frequency * (1.0 - allele_one_frequency);
    let (info_score, info_score_is_valid) = if expected_variance_numerator > 0.0 {
        ((variance_numerator / expected_variance_numerator).clamp(0.0, 1.0), true)
    } else {
        (0.0, false)
    };
    OutputVariantStatistics { allele_one_frequency, info_score, info_score_is_valid }
}

fn checked_selected_sample_count(selected_sample_count: usize) -> GenotypeResult<i32> {
    i32::try_from(selected_sample_count).map_err(|_| {
        GenotypeError::InvalidInput(format!(
            "Selected sample count {selected_sample_count} exceeds the supported i32 statistics range."
        ))
    })
}

fn validate_summary_length(name: &str, observed: usize, expected: usize) -> GenotypeResult<()> {
    if observed != expected {
        return Err(GenotypeError::InvalidInput(format!("{name} contains {observed} values, expected {expected}.")));
    }
    Ok(())
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
    use crate::common::{ChunkStatisticsPolicy, Packed8RawStatistics};

    use super::build_chunk_stats_from_summaries;

    const COMPLETE_STATISTICS_POLICY: ChunkStatisticsPolicy =
        ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: true, collect_sparse_candidate_mask: true };

    #[test]
    fn chunk_stats_compute_means_info_and_sparse_candidates() {
        let statistics = build_chunk_stats_from_summaries(
            vec![1.0, 4.0],
            vec![1.0, 8.0],
            vec![3, 3],
            Some(vec![2, 1]),
            Some(vec![0, 2]),
            3,
            COMPLETE_STATISTICS_POLICY,
        )
        .expect("variant-major summaries should build chunk statistics");

        assert!((statistics.output.allele_one_frequency[0] - (1.0 / 6.0)).abs() < 1.0e-6);
        assert!((statistics.output.allele_one_frequency[1] - (2.0 / 3.0)).abs() < 1.0e-6);
        assert_eq!(statistics.output.observation_count, vec![3, 3]);
        assert_eq!(statistics.output.info_score.validity_bytes, vec![0b11]);
        assert!((statistics.output.info_score.values[0] - 0.8).abs() < 1.0e-6);
        assert!((statistics.output.info_score.values[1] - 1.0).abs() < 1.0e-6);
        assert!((statistics.compute.genotype_mean[0] - (1.0 / 3.0)).abs() < 1.0e-6);
        assert!((statistics.compute.genotype_mean[1] - (4.0 / 3.0)).abs() < 1.0e-6);
        assert_eq!(statistics.compute.imputed_dosage_square_sum, Some(vec![1.0, 8.0]));
        assert_eq!(statistics.compute.sparse_candidate_mask, Some(vec![true, true]));
    }

    #[test]
    fn chunk_stats_handle_unobserved_variants() {
        let statistics = build_chunk_stats_from_summaries(
            vec![0.0],
            vec![0.0],
            vec![0],
            Some(vec![0]),
            Some(vec![0]),
            4,
            COMPLETE_STATISTICS_POLICY,
        )
        .expect("unobserved variant summaries should build chunk statistics");

        assert_eq!(statistics.output.allele_one_frequency, vec![0.0]);
        assert_eq!(statistics.output.info_score.values, vec![0.0]);
        assert_eq!(statistics.output.info_score.validity_bytes, vec![0]);
        assert_eq!(statistics.compute.genotype_mean, vec![0.0]);
        assert_eq!(statistics.compute.imputed_dosage_square_sum, Some(vec![0.0]));
        assert_eq!(statistics.compute.sparse_candidate_mask, Some(vec![false]));
    }

    #[test]
    fn chunk_stats_use_observed_count_for_info_and_impute_square_sum() {
        let statistics = build_chunk_stats_from_summaries(
            vec![1.0],
            vec![1.0],
            vec![2],
            None,
            None,
            4,
            ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: true, collect_sparse_candidate_mask: false },
        )
        .expect("partly observed variant summaries should build chunk statistics");

        assert_eq!(statistics.compute.genotype_mean, vec![0.5]);
        assert_eq!(statistics.compute.imputed_dosage_square_sum, Some(vec![1.5]));
        assert!((statistics.output.info_score.values[0] - (2.0 / 3.0)).abs() < 1.0e-6);
        assert_eq!(statistics.output.info_score.validity_bytes, vec![1]);
    }

    #[test]
    fn chunk_stats_reject_mismatched_summary_shapes_and_policies() {
        let shape_error = build_chunk_stats_from_summaries(
            vec![],
            vec![0.0],
            vec![0],
            None,
            None,
            1,
            ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: false },
        )
        .expect_err("mismatched dosage sums should fail");
        assert!(shape_error.to_string().contains("dosage sum contains 0 values, expected 1"));

        let policy_error = build_chunk_stats_from_summaries(
            vec![0.0],
            vec![0.0],
            vec![1],
            Some(vec![1]),
            Some(vec![0]),
            1,
            ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: false },
        )
        .expect_err("unrequested sparse summaries should fail");
        assert!(policy_error.to_string().contains("do not match the requested statistics policy"));
    }

    #[test]
    fn packed8_raw_statistics_validate_device_status_and_bounds() {
        let output = Packed8RawStatistics {
            dosage_sums: vec![510],
            dosage_square_sums: vec![260_100],
            statuses: vec![0],
            selected_sample_count: 1,
        }
        .into_output_statistics()
        .expect("bounded packed8 summaries should convert");
        assert_eq!(output.allele_one_frequency, vec![1.0]);
        assert_eq!(output.observation_count, vec![1]);
        assert_eq!(output.info_score.validity_bytes, vec![0]);

        let status_error = Packed8RawStatistics {
            dosage_sums: vec![0],
            dosage_square_sums: vec![0],
            statuses: vec![7],
            selected_sample_count: 1,
        }
        .into_output_statistics()
        .expect_err("nonzero device status should fail");
        assert!(status_error.to_string().contains("status 0x00000007"));

        let bound_error = Packed8RawStatistics {
            dosage_sums: vec![511],
            dosage_square_sums: vec![0],
            statuses: vec![0],
            selected_sample_count: 1,
        }
        .into_output_statistics()
        .expect_err("out-of-range device summary should fail");
        assert!(bound_error.to_string().contains("outside the selected-sample bounds"));
    }
}
