//! Host-side genotype preprocessing shared by native readers.

use g_genotype_contracts::{ChunkOutputStatistics, NullableFloat32Column};

use crate::common::{
    ChunkComputeStatistics, ChunkStatisticsPolicy, ChunkStats, EIGHT_BIT_PROBABILITY_SCALE_RECIPROCAL,
    EIGHT_BIT_PROBABILITY_SCALE_SQUARE_RECIPROCAL, Packed8RawStatistics, SparseCandidateSummary,
};
use crate::error::{GenotypeError, GenotypeResult};

const ZERO_DOSAGE_THRESHOLD_DENOMINATOR: u128 = 10_000;
const HOMOZYGOUS_ALTERNATE_DOSAGE_NUMERATOR: u128 = 3;
const HOMOZYGOUS_ALTERNATE_DOSAGE_DENOMINATOR: u128 = 2;
const RARE_SPARSE_FIRTH_MINOR_ALLELE_COUNT_THRESHOLD: u128 = 50;
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
    sparse_candidate_statistics: Option<Vec<SparseCandidateSummary>>,
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
    if statistics_policy.collect_sparse_candidate_mask != sparse_candidate_statistics.is_some() {
        return Err(GenotypeError::InvalidInput(
            "Exact sparse candidate summaries do not match the requested statistics policy.".to_string(),
        ));
    }
    if let Some(sparse_candidate_statistics) = sparse_candidate_statistics.as_ref() {
        validate_summary_length(
            "exact sparse candidate summary",
            sparse_candidate_statistics.len(),
            selected_variant_count,
        )?;
    }

    for variant_index in 0..selected_variant_count {
        let count = observation_count[variant_index];
        if count < 0 || count > selected_sample_count_i32 {
            return Err(GenotypeError::InvalidInput(format!(
                "Variant observation count {count} is outside 0..={selected_sample_count_i32}."
            )));
        }
        if count == 0 {
            if let Some(sparse_candidate_statistics) = sparse_candidate_statistics.as_ref() {
                validate_sparse_candidate_summary(sparse_candidate_statistics[variant_index], count)?;
            }
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
            let missing_count = selected_sample_count_i32 - count;
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
        if let (Some(sparse_candidate_mask), Some(sparse_candidate_statistics)) =
            (sparse_candidate_mask.as_mut(), sparse_candidate_statistics.as_ref())
        {
            sparse_candidate_mask.push(classify_sparse_candidate(sparse_candidate_statistics[variant_index], count)?);
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

fn classify_sparse_candidate(
    sparse_candidate_summary: SparseCandidateSummary,
    observation_count: i32,
) -> GenotypeResult<bool> {
    validate_sparse_candidate_summary(sparse_candidate_summary, observation_count)?;
    if observation_count == 0 {
        return Ok(false);
    }

    let observation_count = u128::from(u32::try_from(observation_count).map_err(|_| {
        GenotypeError::InvalidInput("Sparse candidate observation count must be nonnegative.".to_string())
    })?);
    let dosage_numerator = u128::from(sparse_candidate_summary.exact_dosage_sum.numerator);
    let probability_denominator = u128::from(sparse_candidate_summary.exact_dosage_sum.probability_denominator.get());
    let allele_count_denominator = observation_count * probability_denominator;
    let diploid_allele_count_numerator = 2 * allele_count_denominator;
    let allele_is_flipped = dosage_numerator > allele_count_denominator;
    let regenie_zero_count = if allele_is_flipped {
        sparse_candidate_summary.homozygous_alternate_count
    } else {
        sparse_candidate_summary.zero_count
    };
    let regenie_zero_count =
        u128::from(u32::try_from(regenie_zero_count).map_err(|_| {
            GenotypeError::InvalidInput("Sparse candidate zero count must be nonnegative.".to_string())
        })?);
    let minor_allele_count_numerator = dosage_numerator.min(diploid_allele_count_numerator - dosage_numerator);

    Ok((2 * regenie_zero_count >= observation_count)
        && (minor_allele_count_numerator < RARE_SPARSE_FIRTH_MINOR_ALLELE_COUNT_THRESHOLD * probability_denominator))
}

fn validate_sparse_candidate_summary(
    sparse_candidate_summary: SparseCandidateSummary,
    observation_count: i32,
) -> GenotypeResult<()> {
    if sparse_candidate_summary.zero_count < 0
        || sparse_candidate_summary.zero_count > observation_count
        || sparse_candidate_summary.homozygous_alternate_count < 0
        || sparse_candidate_summary.homozygous_alternate_count > observation_count
    {
        return Err(GenotypeError::InvalidInput(format!(
            "Sparse candidate counts ({}, {}) are outside 0..={observation_count}.",
            sparse_candidate_summary.zero_count, sparse_candidate_summary.homozygous_alternate_count,
        )));
    }
    let probability_denominator = u128::from(sparse_candidate_summary.exact_dosage_sum.probability_denominator.get());
    let maximum_dosage_numerator = u128::from(u32::try_from(observation_count).map_err(|_| {
        GenotypeError::InvalidInput("Sparse candidate observation count must be nonnegative.".to_string())
    })?) * 2
        * probability_denominator;
    if u128::from(sparse_candidate_summary.exact_dosage_sum.numerator) > maximum_dosage_numerator {
        return Err(GenotypeError::InvalidInput(format!(
            "Exact dosage numerator {} exceeds the diploid bound {maximum_dosage_numerator}.",
            sparse_candidate_summary.exact_dosage_sum.numerator,
        )));
    }
    Ok(())
}

pub(crate) fn increment_exact_sparse_candidate_counts(
    dosage_numerator: u64,
    probability_denominator: u32,
    zero_count: &mut i32,
    homozygous_alternate_count: &mut i32,
) {
    let dosage_numerator = u128::from(dosage_numerator);
    let probability_denominator = u128::from(probability_denominator);
    if dosage_numerator * ZERO_DOSAGE_THRESHOLD_DENOMINATOR <= probability_denominator {
        *zero_count += 1;
    }
    if dosage_numerator * HOMOZYGOUS_ALTERNATE_DOSAGE_DENOMINATOR
        >= probability_denominator * HOMOZYGOUS_ALTERNATE_DOSAGE_NUMERATOR
    {
        *homozygous_alternate_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::common::{ChunkStatisticsPolicy, ExactDosageSum, Packed8RawStatistics, SparseCandidateSummary};

    use super::build_chunk_stats_from_summaries;

    const COMPLETE_STATISTICS_POLICY: ChunkStatisticsPolicy =
        ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: true, collect_sparse_candidate_mask: true };

    fn sparse_candidate_from_exact_summary(
        dosage_sum: f32,
        dosage_square_sum: f32,
        observation_count: i32,
        sparse_candidate_summary: SparseCandidateSummary,
        selected_sample_count: usize,
    ) -> bool {
        let statistics = build_chunk_stats_from_summaries(
            vec![dosage_sum],
            vec![dosage_square_sum],
            vec![observation_count],
            Some(vec![sparse_candidate_summary]),
            selected_sample_count,
            COMPLETE_STATISTICS_POLICY,
        )
        .expect("exact sparse candidate summary should classify");
        statistics.compute.sparse_candidate_mask.expect("sparse mask was requested")[0]
    }

    #[test]
    fn exact_sparse_candidate_boundaries_match_upstream_classification() {
        let minor_allele_count_49 = SparseCandidateSummary {
            exact_dosage_sum: ExactDosageSum::new(49, 1),
            zero_count: 151,
            homozygous_alternate_count: 0,
        };
        let minor_allele_count_50 = SparseCandidateSummary {
            exact_dosage_sum: ExactDosageSum::new(50, 1),
            zero_count: 150,
            homozygous_alternate_count: 0,
        };
        let exact_sparse_density = SparseCandidateSummary {
            exact_dosage_sum: ExactDosageSum::new(40, 1),
            zero_count: 40,
            homozygous_alternate_count: 0,
        };
        let below_sparse_density = SparseCandidateSummary {
            exact_dosage_sum: ExactDosageSum::new(41, 1),
            zero_count: 39,
            homozygous_alternate_count: 0,
        };
        let flipped_minor_allele_count_49 = SparseCandidateSummary {
            exact_dosage_sum: ExactDosageSum::new(351, 1),
            zero_count: 0,
            homozygous_alternate_count: 151,
        };

        assert!(sparse_candidate_from_exact_summary(49.0, 49.0, 200, minor_allele_count_49, 200));
        assert!(!sparse_candidate_from_exact_summary(50.0, 50.0, 200, minor_allele_count_50, 200));
        assert!(sparse_candidate_from_exact_summary(40.0, 40.0, 80, exact_sparse_density, 80));
        assert!(!sparse_candidate_from_exact_summary(41.0, 41.0, 80, below_sparse_density, 80));
        assert!(sparse_candidate_from_exact_summary(351.0, 653.0, 200, flipped_minor_allele_count_49, 200,));
    }

    #[test]
    fn exact_sparse_candidate_density_uses_observed_nonmissing_denominator() {
        let statistics = build_chunk_stats_from_summaries(
            vec![49.0],
            vec![49.0],
            vec![99],
            Some(vec![SparseCandidateSummary {
                exact_dosage_sum: ExactDosageSum::new(49, 1),
                zero_count: 50,
                homozygous_alternate_count: 0,
            }]),
            200,
            COMPLETE_STATISTICS_POLICY,
        )
        .expect("partly observed exact dosage summary should classify");

        assert_eq!(statistics.output.observation_count, vec![99]);
        assert_eq!(statistics.compute.sparse_candidate_mask, Some(vec![true]));
    }

    #[test]
    fn chunk_stats_compute_means_info_and_sparse_candidates() {
        let statistics = build_chunk_stats_from_summaries(
            vec![1.0, 4.0],
            vec![1.0, 8.0],
            vec![3, 3],
            Some(vec![
                SparseCandidateSummary {
                    exact_dosage_sum: ExactDosageSum::new(1, 1),
                    zero_count: 2,
                    homozygous_alternate_count: 0,
                },
                SparseCandidateSummary {
                    exact_dosage_sum: ExactDosageSum::new(4, 1),
                    zero_count: 1,
                    homozygous_alternate_count: 2,
                },
            ]),
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
            Some(vec![SparseCandidateSummary::default()]),
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
            1,
            ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: false },
        )
        .expect_err("mismatched dosage sums should fail");
        assert!(shape_error.to_string().contains("dosage sum contains 0 values, expected 1"));

        let policy_error = build_chunk_stats_from_summaries(
            vec![0.0],
            vec![0.0],
            vec![1],
            Some(vec![SparseCandidateSummary {
                exact_dosage_sum: ExactDosageSum::new(0, 1),
                zero_count: 1,
                homozygous_alternate_count: 0,
            }]),
            1,
            ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: false },
        )
        .expect_err("unrequested sparse summaries should fail");
        assert!(policy_error.to_string().contains("do not match the requested statistics policy"));
    }

    #[test]
    fn exact_sparse_classification_keeps_rs5753646_and_rs80694_mac_fifty_dense() {
        let observation_count = 2_504_i32;
        let observation_count_float = 2_504.0_f32;
        let historical_dosage_sum = 4_958.000_5_f32;
        let historical_minor_allele_count = (2.0_f32 * observation_count_float) - historical_dosage_sum;
        assert!((historical_minor_allele_count - 49.999_51).abs() < 1.0e-5);
        assert!(historical_minor_allele_count < 50.0);

        let exact_allele_count =
            u64::from(u32::try_from((2 * observation_count) - 50).expect("test allele count should fit u32")) * 255;
        let statistics = build_chunk_stats_from_summaries(
            vec![historical_dosage_sum],
            vec![historical_dosage_sum * historical_dosage_sum / observation_count_float],
            vec![observation_count],
            Some(vec![SparseCandidateSummary {
                exact_dosage_sum: ExactDosageSum::new(exact_allele_count, 255),
                zero_count: 0,
                homozygous_alternate_count: observation_count / 2,
            }]),
            usize::try_from(observation_count).expect("test sample count should fit usize"),
            ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: true },
        )
        .expect("exact MAC-fifty summary should preprocess");

        assert_eq!(statistics.compute.sparse_candidate_mask, Some(vec![false]));
        assert_eq!(statistics.compute.genotype_mean, vec![historical_dosage_sum / observation_count_float]);
    }

    #[test]
    fn exact_sparse_classification_validates_counts_and_u64_bounds() {
        let maximum_observation_count = i32::MAX;
        let maximum_denominator = u32::MAX;
        let maximum_numerator =
            u64::from(u32::try_from(maximum_observation_count).expect("positive i32 should fit u32"))
                * 2
                * u64::from(maximum_denominator);
        let maximum_summary = SparseCandidateSummary {
            exact_dosage_sum: ExactDosageSum::new(maximum_numerator, maximum_denominator),
            zero_count: 0,
            homozygous_alternate_count: 0,
        };
        let maximum_statistics = build_chunk_stats_from_summaries(
            vec![0.0],
            vec![0.0],
            vec![maximum_observation_count],
            Some(vec![maximum_summary]),
            usize::try_from(maximum_observation_count).expect("positive i32 should fit usize"),
            ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: true },
        )
        .expect("the i32 observation bound should fit exact u64 dosage accumulation");
        assert_eq!(maximum_statistics.compute.sparse_candidate_mask, Some(vec![false]));

        let numerator_error = build_chunk_stats_from_summaries(
            vec![0.0],
            vec![0.0],
            vec![maximum_observation_count],
            Some(vec![SparseCandidateSummary {
                exact_dosage_sum: ExactDosageSum::new(maximum_numerator + 1, maximum_denominator),
                ..maximum_summary
            }]),
            usize::try_from(maximum_observation_count).expect("positive i32 should fit usize"),
            ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: true },
        )
        .expect_err("an exact numerator above the diploid bound should fail");
        assert!(numerator_error.to_string().contains("exceeds the diploid bound"));

        let count_error = build_chunk_stats_from_summaries(
            vec![0.0],
            vec![0.0],
            vec![1],
            Some(vec![SparseCandidateSummary {
                exact_dosage_sum: ExactDosageSum::new(0, 255),
                zero_count: 2,
                homozygous_alternate_count: 0,
            }]),
            1,
            ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: true },
        )
        .expect_err("a sparse count above the observation count should fail");
        assert!(count_error.to_string().contains("outside 0..=1"));

        let observation_error = build_chunk_stats_from_summaries(
            vec![0.0],
            vec![0.0],
            vec![2],
            None,
            1,
            ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: false },
        )
        .expect_err("an observation count above the selected count should fail");
        assert!(observation_error.to_string().contains("outside 0..=1"));
    }

    #[test]
    fn exact_sparse_counts_respect_quantized_dosage_thresholds() {
        let mut zero_count = 0_i32;
        let mut homozygous_alternate_count = 0_i32;

        super::increment_exact_sparse_candidate_counts(1, 10_000, &mut zero_count, &mut homozygous_alternate_count);
        super::increment_exact_sparse_candidate_counts(2, 10_000, &mut zero_count, &mut homozygous_alternate_count);
        assert_eq!(zero_count, 1);
        assert_eq!(homozygous_alternate_count, 0);

        super::increment_exact_sparse_candidate_counts(3, 2, &mut zero_count, &mut homozygous_alternate_count);
        super::increment_exact_sparse_candidate_counts(2, 2, &mut zero_count, &mut homozygous_alternate_count);
        assert_eq!(zero_count, 1);
        assert_eq!(homozygous_alternate_count, 1);

        let maximum_denominator = u32::MAX;
        let largest_zero_numerator = u64::from(maximum_denominator) / 10_000;
        super::increment_exact_sparse_candidate_counts(
            largest_zero_numerator,
            maximum_denominator,
            &mut zero_count,
            &mut homozygous_alternate_count,
        );
        super::increment_exact_sparse_candidate_counts(
            largest_zero_numerator + 1,
            maximum_denominator,
            &mut zero_count,
            &mut homozygous_alternate_count,
        );
        assert_eq!(zero_count, 2);
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
