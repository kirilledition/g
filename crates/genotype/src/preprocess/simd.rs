//! SIMD row-summary kernels for genotype preprocessing.

use super::{HOMOZYGOUS_ALTERNATE_DOSAGE_THRESHOLD, ZERO_DOSAGE_UPPER_BOUND};
use crate::common::DosageSummary;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const AVX2_DOSAGE_LANE_COUNT: usize = 8;

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m256, _CMP_LE_OQ, _CMP_LT_OQ, _CMP_ORD_Q, _mm256_add_ps, _mm256_and_ps, _mm256_cmp_ps, _mm256_loadu_ps,
    _mm256_movemask_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, _CMP_LE_OQ, _CMP_LT_OQ, _CMP_ORD_Q, _mm256_add_ps, _mm256_and_ps, _mm256_cmp_ps, _mm256_loadu_ps,
    _mm256_movemask_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn summarize_variant_major_row_avx2(
    dosage_values: &[f32],
    collect_sparse_candidate_counts: bool,
) -> DosageSummary {
    let zero_dosage_upper_bound = _mm256_set1_ps(ZERO_DOSAGE_UPPER_BOUND);
    let homozygous_alternate_threshold = _mm256_set1_ps(HOMOZYGOUS_ALTERNATE_DOSAGE_THRESHOLD);
    let mut dosage_sum_vector = _mm256_setzero_ps();
    let mut dosage_square_sum_vector = _mm256_setzero_ps();
    let mut row_summary = DosageSummary::default();
    let mut dosage_index = 0_usize;

    while dosage_index + AVX2_DOSAGE_LANE_COUNT <= dosage_values.len() {
        let dosage_pointer = unsafe { dosage_values.as_ptr().add(dosage_index) };
        let dosage_vector = unsafe { _mm256_loadu_ps(dosage_pointer) };
        let observed_mask = _mm256_cmp_ps(dosage_vector, dosage_vector, _CMP_ORD_Q);
        let observed_dosage_vector = _mm256_and_ps(dosage_vector, observed_mask);
        dosage_sum_vector = _mm256_add_ps(dosage_sum_vector, observed_dosage_vector);
        dosage_square_sum_vector =
            _mm256_add_ps(dosage_square_sum_vector, _mm256_mul_ps(observed_dosage_vector, observed_dosage_vector));

        let observed_count = i32::try_from(_mm256_movemask_ps(observed_mask).count_ones())
            .expect("AVX2 observed lane count should fit i32");
        row_summary.observation_count += observed_count;

        if collect_sparse_candidate_counts {
            let zero_mask =
                _mm256_and_ps(observed_mask, _mm256_cmp_ps(dosage_vector, zero_dosage_upper_bound, _CMP_LE_OQ));
            row_summary.zero_count +=
                i32::try_from(_mm256_movemask_ps(zero_mask).count_ones()).expect("AVX2 zero lane count should fit i32");

            let less_than_homozygous_alternate_mask =
                _mm256_and_ps(observed_mask, _mm256_cmp_ps(dosage_vector, homozygous_alternate_threshold, _CMP_LT_OQ));
            let less_than_homozygous_alternate_count =
                i32::try_from(_mm256_movemask_ps(less_than_homozygous_alternate_mask).count_ones())
                    .expect("AVX2 threshold lane count should fit i32");
            row_summary.homozygous_alternate_count += observed_count - less_than_homozygous_alternate_count;
        }

        dosage_index += AVX2_DOSAGE_LANE_COUNT;
    }

    row_summary.dosage_sum = unsafe { horizontal_sum_avx2(dosage_sum_vector) };
    row_summary.dosage_square_sum = unsafe { horizontal_sum_avx2(dosage_square_sum_vector) };
    for &dosage_value in &dosage_values[dosage_index..] {
        if dosage_value.is_nan() {
            continue;
        }
        row_summary.record_observed_dosage(dosage_value, collect_sparse_candidate_counts);
    }

    row_summary
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(values: __m256) -> f32 {
    let mut lanes = [0.0_f32; AVX2_DOSAGE_LANE_COUNT];
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), values);
    }
    lanes.into_iter().sum()
}
