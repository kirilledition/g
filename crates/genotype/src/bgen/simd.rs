use std::mem::MaybeUninit;

use crate::common::DosageSummary;
#[cfg(test)]
use crate::preprocess;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const AVX2_SAMPLE_COUNT: usize = 8;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const AVX2_PACKED8_SAMPLE_COUNT: usize = 16;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const AVX2_ACCUMULATION_VECTOR_LIMIT: usize = 4096;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const AVX2_PLOIDY_BYTE_COUNT: usize = 32;
const EIGHT_BIT_PROBABILITY_SCALE_RECIPROCAL: f32 = 1.0_f32 / 255.0_f32;
const EIGHT_BIT_PROBABILITY_SCALE_SQUARE_RECIPROCAL: f32 = 1.0_f32 / (255.0_f32 * 255.0_f32);
const PRESENT_DIPLOID_BYTE_GROUP: [u8; 16] = [2_u8; 16];

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bgen_avx2_enabled() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn bgen_avx2_enabled() -> bool {
    false
}

impl DosageSummary {
    #[cfg(test)]
    fn record_dosage(&mut self, dosage_value: f32) {
        self.dosage_sum += dosage_value;
        self.dosage_square_sum += dosage_value * dosage_value;
        self.observation_count += 1;
        preprocess::increment_dosage_summary_counts(
            dosage_value,
            &mut self.zero_count,
            &mut self.nonzero_count,
            &mut self.homozygous_reference_count,
            &mut self.heterozygous_count,
            &mut self.homozygous_alternate_count,
        );
    }

    #[cfg(test)]
    fn record_raw_dosage_integer_from_f32_accumulation(&mut self, raw_dosage_integer: i32) {
        let dosage_value = raw_dosage_value(raw_dosage_integer);
        self.dosage_sum += dosage_value;
        self.dosage_square_sum += dosage_value * dosage_value;
        self.observation_count += 1;
        if raw_dosage_integer >= 1 {
            self.nonzero_count += 1;
        } else {
            self.zero_count += 1;
        }
        if raw_dosage_integer <= 127 {
            self.homozygous_reference_count += 1;
        } else if raw_dosage_integer <= 382 {
            self.heterozygous_count += 1;
        } else {
            self.homozygous_alternate_count += 1;
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct EightBitRawIntegerSummary {
    raw_dosage_total: i64,
    raw_dosage_square_total: u64,
    selected_observation_count: i32,
    zero_count: i32,
    homozygous_alternate_count: i32,
    collect_sparse_candidate_counts: bool,
}

impl EightBitRawIntegerSummary {
    pub(super) fn new(collect_sparse_candidate_counts: bool) -> Self {
        Self { collect_sparse_candidate_counts, ..Self::default() }
    }

    pub(super) fn record_probability_pair(
        &mut self,
        [homozygous_reference_probability_byte, heterozygous_probability_byte]: [u8; 2],
    ) {
        self.record_raw_dosage_integer(raw_dosage_integer(
            homozygous_reference_probability_byte,
            heterozygous_probability_byte,
        ));
    }

    fn record_raw_dosage_integer(&mut self, raw_dosage_integer: i32) {
        let raw_dosage_integer_i64 = i64::from(raw_dosage_integer);
        self.raw_dosage_total += raw_dosage_integer_i64;
        self.raw_dosage_square_total +=
            u64::try_from(raw_dosage_integer_i64 * raw_dosage_integer_i64).expect("raw dosage square should fit u64");
        self.selected_observation_count += 1;
        if self.collect_sparse_candidate_counts {
            if raw_dosage_integer == 0 {
                self.zero_count += 1;
            }
            if raw_dosage_integer >= 383 {
                self.homozygous_alternate_count += 1;
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    pub(super) fn into_decode_summary(self) -> DosageSummary {
        DosageSummary {
            dosage_sum: self.raw_dosage_total as f32 * EIGHT_BIT_PROBABILITY_SCALE_RECIPROCAL,
            dosage_square_sum: self.raw_dosage_square_total as f32 * EIGHT_BIT_PROBABILITY_SCALE_SQUARE_RECIPROCAL,
            observation_count: self.selected_observation_count,
            zero_count: self.zero_count,
            homozygous_alternate_count: self.homozygous_alternate_count,
        }
    }
}

pub(super) fn decode_unphased_eight_bit_identity_simd_or_scalar(
    packed_probability_bytes: &[u8],
    output_values: &mut [MaybeUninit<f32>],
    collect_sparse_candidate_counts: bool,
) -> Option<DosageSummary> {
    assert_eq!(packed_probability_bytes.len(), output_values.len() * 2);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if bgen_avx2_enabled() {
            // Full-sample packed8 benchmarks selected raw-integer AVX2 over lookup-gather AVX2.
            return unsafe {
                decode_unphased_eight_bit_identity_raw_avx2(
                    packed_probability_bytes,
                    output_values,
                    collect_sparse_candidate_counts,
                )
            };
        }
    }

    decode_unphased_eight_bit_identity_raw_scalar_integer_stats(
        packed_probability_bytes,
        output_values,
        collect_sparse_candidate_counts,
    )
}

pub(super) fn copy_unphased_eight_bit_probability_pairs_and_summarize_simd_or_scalar(
    packed_probability_bytes: &[u8],
    output_probability_bytes: &mut [MaybeUninit<u8>],
    collect_sparse_candidate_counts: bool,
) -> DosageSummary {
    assert_eq!(packed_probability_bytes.len(), output_probability_bytes.len());
    assert_eq!(packed_probability_bytes.len() % 2, 0);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if bgen_avx2_enabled() {
            return unsafe {
                copy_unphased_eight_bit_probability_pairs_and_summarize_raw_avx2(
                    packed_probability_bytes,
                    output_probability_bytes,
                    collect_sparse_candidate_counts,
                )
            };
        }
    }

    copy_unphased_eight_bit_probability_pairs_and_summarize_raw_scalar_integer_stats(
        packed_probability_bytes,
        output_probability_bytes,
        collect_sparse_candidate_counts,
    )
}

pub(super) fn all_samples_present_diploid_simd_or_scalar(sample_ploidy_and_missingness: &[u8]) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if bgen_avx2_enabled() {
            return unsafe { all_samples_present_diploid_avx2(sample_ploidy_and_missingness) };
        }
    }

    all_samples_present_diploid_scalar(sample_ploidy_and_missingness)
}

pub(super) fn all_unphased_eight_bit_probability_pairs_valid_simd_or_scalar(packed_probability_bytes: &[u8]) -> bool {
    if !packed_probability_bytes.len().is_multiple_of(2) {
        return false;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if bgen_avx2_enabled() {
            return unsafe { all_unphased_eight_bit_probability_pairs_valid_avx2(packed_probability_bytes) };
        }
    }
    all_unphased_eight_bit_probability_pairs_valid_scalar(packed_probability_bytes)
}

fn all_samples_present_diploid_scalar(sample_ploidy_and_missingness: &[u8]) -> bool {
    let mut ploidy_chunks = sample_ploidy_and_missingness.chunks_exact(PRESENT_DIPLOID_BYTE_GROUP.len());
    for ploidy_chunk in &mut ploidy_chunks {
        if ploidy_chunk != PRESENT_DIPLOID_BYTE_GROUP {
            return false;
        }
    }
    ploidy_chunks.remainder().iter().all(|ploidy_byte| *ploidy_byte == 2)
}

fn all_unphased_eight_bit_probability_pairs_valid_scalar(packed_probability_bytes: &[u8]) -> bool {
    packed_probability_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .all(|probability_pair| u8::try_from(u16::from(probability_pair[0]) + u16::from(probability_pair[1])).is_ok())
}

fn decode_unphased_eight_bit_identity_raw_scalar_integer_stats(
    packed_probability_bytes: &[u8],
    output_values: &mut [MaybeUninit<f32>],
    collect_sparse_candidate_counts: bool,
) -> Option<DosageSummary> {
    let mut raw_integer_summary = EightBitRawIntegerSummary::new(collect_sparse_candidate_counts);
    if !decode_unphased_eight_bit_identity_raw_scalar_integer_stats_from(
        packed_probability_bytes,
        output_values,
        0,
        &mut raw_integer_summary,
    ) {
        return None;
    }
    Some(raw_integer_summary.into_decode_summary())
}

fn copy_unphased_eight_bit_probability_pairs_and_summarize_raw_scalar_integer_stats(
    packed_probability_bytes: &[u8],
    output_probability_bytes: &mut [MaybeUninit<u8>],
    collect_sparse_candidate_counts: bool,
) -> DosageSummary {
    let mut raw_integer_summary = EightBitRawIntegerSummary::new(collect_sparse_candidate_counts);
    copy_unphased_eight_bit_probability_pairs_and_summarize_raw_scalar_integer_stats_from(
        packed_probability_bytes,
        output_probability_bytes,
        0,
        &mut raw_integer_summary,
    );
    raw_integer_summary.into_decode_summary()
}

fn copy_unphased_eight_bit_probability_pairs_and_summarize_raw_scalar_integer_stats_from(
    packed_probability_bytes: &[u8],
    output_probability_bytes: &mut [MaybeUninit<u8>],
    start_sample_index: usize,
    raw_integer_summary: &mut EightBitRawIntegerSummary,
) {
    let byte_start_index = start_sample_index * 2;
    let (probability_pairs, _) = packed_probability_bytes[byte_start_index..].as_chunks::<2>();
    let (output_probability_pairs, _) = output_probability_bytes[byte_start_index..].as_chunks_mut::<2>();
    for ([homozygous_reference_probability_byte, heterozygous_probability_byte], output_probability_pair) in
        probability_pairs.iter().copied().zip(output_probability_pairs)
    {
        output_probability_pair[0].write(homozygous_reference_probability_byte);
        output_probability_pair[1].write(heterozygous_probability_byte);
        raw_integer_summary.record_raw_dosage_integer(raw_dosage_integer(
            homozygous_reference_probability_byte,
            heterozygous_probability_byte,
        ));
    }
}

fn decode_unphased_eight_bit_identity_raw_scalar_integer_stats_from(
    packed_probability_bytes: &[u8],
    output_values: &mut [MaybeUninit<f32>],
    start_sample_index: usize,
    raw_integer_summary: &mut EightBitRawIntegerSummary,
) -> bool {
    let (probability_pairs, _) = packed_probability_bytes[start_sample_index * 2..].as_chunks::<2>();
    for ([homozygous_reference_probability_byte, heterozygous_probability_byte], output_value) in
        probability_pairs.iter().copied().zip(&mut output_values[start_sample_index..])
    {
        if u8::try_from(u16::from(homozygous_reference_probability_byte) + u16::from(heterozygous_probability_byte))
            .is_err()
        {
            return false;
        }
        let raw_dosage_integer =
            raw_dosage_integer(homozygous_reference_probability_byte, heterozygous_probability_byte);
        output_value.write(raw_dosage_value(raw_dosage_integer));
        raw_integer_summary.record_raw_dosage_integer(raw_dosage_integer);
    }
    true
}

#[cfg(test)]
fn decode_unphased_eight_bit_identity_lookup_scalar(
    packed_probability_bytes: &[u8],
    dosage_lookup: &[f32],
    output_values: &mut [f32],
) -> DosageSummary {
    let mut decode_summary = DosageSummary::default();
    decode_unphased_eight_bit_identity_lookup_scalar_from(
        packed_probability_bytes,
        dosage_lookup,
        output_values,
        0,
        &mut decode_summary,
    );
    decode_summary
}

#[cfg(test)]
fn decode_unphased_eight_bit_identity_lookup_scalar_from(
    packed_probability_bytes: &[u8],
    dosage_lookup: &[f32],
    output_values: &mut [f32],
    start_sample_index: usize,
    decode_summary: &mut DosageSummary,
) {
    let (probability_pairs, _) = packed_probability_bytes[start_sample_index * 2..].as_chunks::<2>();
    for (relative_sample_index, [homozygous_reference_probability_byte, heterozygous_probability_byte]) in
        probability_pairs.iter().copied().take(output_values.len().saturating_sub(start_sample_index)).enumerate()
    {
        let output_index = start_sample_index + relative_sample_index;
        let packed_probability_index =
            usize::from(homozygous_reference_probability_byte) | (usize::from(heterozygous_probability_byte) << 8);
        let dosage_value = dosage_lookup[packed_probability_index];
        output_values[output_index] = dosage_value;
        decode_summary.record_dosage(dosage_value);
    }
}

#[cfg(test)]
fn decode_unphased_eight_bit_identity_raw_scalar(
    packed_probability_bytes: &[u8],
    output_values: &mut [f32],
) -> DosageSummary {
    let mut decode_summary = DosageSummary::default();
    decode_unphased_eight_bit_identity_raw_scalar_from(packed_probability_bytes, output_values, 0, &mut decode_summary);
    decode_summary
}

#[cfg(test)]
fn decode_unphased_eight_bit_identity_raw_scalar_from(
    packed_probability_bytes: &[u8],
    output_values: &mut [f32],
    start_sample_index: usize,
    decode_summary: &mut DosageSummary,
) {
    let (probability_pairs, _) = packed_probability_bytes[start_sample_index * 2..].as_chunks::<2>();
    for (relative_sample_index, [homozygous_reference_probability_byte, heterozygous_probability_byte]) in
        probability_pairs.iter().copied().take(output_values.len().saturating_sub(start_sample_index)).enumerate()
    {
        let output_index = start_sample_index + relative_sample_index;
        let raw_dosage_integer =
            raw_dosage_integer(homozygous_reference_probability_byte, heterozygous_probability_byte);
        let dosage_value = raw_dosage_value(raw_dosage_integer);
        output_values[output_index] = dosage_value;
        decode_summary.record_raw_dosage_integer_from_f32_accumulation(raw_dosage_integer);
    }
}

fn raw_dosage_integer(homozygous_reference_probability_byte: u8, heterozygous_probability_byte: u8) -> i32 {
    510_i32 - (2_i32 * i32::from(homozygous_reference_probability_byte)) - i32::from(heterozygous_probability_byte)
}

#[allow(clippy::cast_precision_loss)]
fn raw_dosage_value(raw_dosage_integer: i32) -> f32 {
    raw_dosage_integer as f32 * EIGHT_BIT_PROBABILITY_SCALE_RECIPROCAL
}

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m256i, _mm_loadu_si128, _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi256_ps,
    _mm256_cmpeq_epi8, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpgt_epi16, _mm256_cmpgt_epi32,
    _mm256_cvtepi32_ps, _mm256_cvtepu16_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_movemask_epi8, _mm256_movemask_ps, _mm256_mul_ps, _mm256_mullo_epi32, _mm256_set1_epi8, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_srli_epi16, _mm256_srli_epi32,
    _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm_loadu_si128, _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi256_ps,
    _mm256_cmpeq_epi8, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpgt_epi16, _mm256_cmpgt_epi32,
    _mm256_cvtepi32_ps, _mm256_cvtepu16_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_movemask_epi8, _mm256_movemask_ps, _mm256_mul_ps, _mm256_mullo_epi32, _mm256_set1_epi8, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_srli_epi16, _mm256_srli_epi32,
    _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn all_samples_present_diploid_avx2(sample_ploidy_and_missingness: &[u8]) -> bool {
    let expected_ploidy_and_missingness = _mm256_set1_epi8(2_i8);
    let mut byte_index = 0_usize;
    while byte_index + AVX2_PLOIDY_BYTE_COUNT <= sample_ploidy_and_missingness.len() {
        let ploidy_pointer = unsafe { sample_ploidy_and_missingness.as_ptr().add(byte_index).cast::<__m256i>() };
        let ploidy_values = unsafe { _mm256_loadu_si256(ploidy_pointer) };
        let comparison_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(ploidy_values, expected_ploidy_and_missingness));
        if comparison_mask != -1 {
            return false;
        }
        byte_index += AVX2_PLOIDY_BYTE_COUNT;
    }

    all_samples_present_diploid_scalar(&sample_ploidy_and_missingness[byte_index..])
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn all_unphased_eight_bit_probability_pairs_valid_avx2(packed_probability_bytes: &[u8]) -> bool {
    let probability_byte_mask = _mm256_set1_epi16(0x00FF);
    let maximum_probability_sum = _mm256_set1_epi16(i16::from(u8::MAX));
    let mut byte_index = 0_usize;
    while byte_index + AVX2_PLOIDY_BYTE_COUNT <= packed_probability_bytes.len() {
        let probability_pointer = unsafe { packed_probability_bytes.as_ptr().add(byte_index).cast::<__m256i>() };
        let probability_pairs = unsafe { _mm256_loadu_si256(probability_pointer) };
        let first_probabilities = _mm256_and_si256(probability_pairs, probability_byte_mask);
        let second_probabilities = _mm256_srli_epi16(probability_pairs, 8);
        let probability_sums = _mm256_add_epi16(first_probabilities, second_probabilities);
        if _mm256_movemask_epi8(_mm256_cmpgt_epi16(probability_sums, maximum_probability_sum)) != 0 {
            return false;
        }
        byte_index += AVX2_PLOIDY_BYTE_COUNT;
    }
    all_unphased_eight_bit_probability_pairs_valid_scalar(&packed_probability_bytes[byte_index..])
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx2_mask_count(comparison_mask: i32) -> i32 {
    i32::try_from(comparison_mask.count_ones()).expect("AVX2 comparison mask count should fit i32")
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn record_raw_dosage_accumulators_avx2(
    raw_integer_summary: &mut EightBitRawIntegerSummary,
    raw_sum_accumulator: __m256i,
    raw_square_sum_accumulator: __m256i,
) {
    let mut raw_sum_lanes = [0_i32; AVX2_SAMPLE_COUNT];
    let mut raw_square_sum_lanes = [0_i32; AVX2_SAMPLE_COUNT];
    unsafe {
        _mm256_storeu_si256(raw_sum_lanes.as_mut_ptr().cast::<__m256i>(), raw_sum_accumulator);
        _mm256_storeu_si256(raw_square_sum_lanes.as_mut_ptr().cast::<__m256i>(), raw_square_sum_accumulator);
    }
    for lane_index in 0..AVX2_SAMPLE_COUNT {
        raw_integer_summary.raw_dosage_total += i64::from(raw_sum_lanes[lane_index]);
        raw_integer_summary.raw_dosage_square_total +=
            u64::try_from(raw_square_sum_lanes[lane_index]).expect("raw dosage square sum should fit u64");
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn decode_unphased_eight_bit_identity_raw_avx2(
    packed_probability_bytes: &[u8],
    output_values: &mut [MaybeUninit<f32>],
    collect_sparse_candidate_counts: bool,
) -> Option<DosageSummary> {
    let mut raw_integer_summary = EightBitRawIntegerSummary::new(collect_sparse_candidate_counts);
    let probability_byte_mask = _mm256_set1_epi32(0xFF);
    let raw_dosage_base = _mm256_set1_epi32(510);
    let maximum_probability_sum = _mm256_set1_epi32(i32::from(u8::MAX));
    let zero = _mm256_setzero_si256();
    let homozygous_alternate_lower_bound = _mm256_set1_epi32(382);
    let probability_scale_reciprocal = _mm256_set1_ps(EIGHT_BIT_PROBABILITY_SCALE_RECIPROCAL);
    let mut raw_sum_accumulator = _mm256_setzero_si256();
    let mut raw_square_sum_accumulator = _mm256_setzero_si256();
    let mut accumulated_vector_count = 0_usize;
    let mut sample_index = 0_usize;
    while sample_index + AVX2_SAMPLE_COUNT <= output_values.len() {
        let probability_pointer = unsafe { packed_probability_bytes.as_ptr().add(sample_index * 2).cast::<__m128i>() };
        let probability_words = unsafe { _mm_loadu_si128(probability_pointer) };
        let probability_indices = _mm256_cvtepu16_epi32(probability_words);
        let homozygous_reference_probability_bytes = _mm256_and_si256(probability_indices, probability_byte_mask);
        let heterozygous_probability_bytes = _mm256_srli_epi32(probability_indices, 8);
        let probability_sums = _mm256_add_epi32(homozygous_reference_probability_bytes, heterozygous_probability_bytes);
        if _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(probability_sums, maximum_probability_sum))) != 0 {
            return None;
        }
        let doubled_homozygous_reference_probability_bytes =
            _mm256_slli_epi32(homozygous_reference_probability_bytes, 1);
        let raw_dosage_integers = _mm256_sub_epi32(
            _mm256_sub_epi32(raw_dosage_base, doubled_homozygous_reference_probability_bytes),
            heterozygous_probability_bytes,
        );
        let dosage_values = _mm256_mul_ps(_mm256_cvtepi32_ps(raw_dosage_integers), probability_scale_reciprocal);
        unsafe {
            _mm256_storeu_ps(output_values.as_mut_ptr().cast::<f32>().add(sample_index), dosage_values);
        }

        raw_sum_accumulator = _mm256_add_epi32(raw_sum_accumulator, raw_dosage_integers);
        raw_square_sum_accumulator =
            _mm256_add_epi32(raw_square_sum_accumulator, _mm256_mullo_epi32(raw_dosage_integers, raw_dosage_integers));
        accumulated_vector_count += 1;

        if collect_sparse_candidate_counts {
            raw_integer_summary.zero_count +=
                avx2_mask_count(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(raw_dosage_integers, zero))));
            raw_integer_summary.homozygous_alternate_count += avx2_mask_count(_mm256_movemask_ps(_mm256_castsi256_ps(
                _mm256_cmpgt_epi32(raw_dosage_integers, homozygous_alternate_lower_bound),
            )));
        }

        if accumulated_vector_count == AVX2_ACCUMULATION_VECTOR_LIMIT {
            unsafe {
                record_raw_dosage_accumulators_avx2(
                    &mut raw_integer_summary,
                    raw_sum_accumulator,
                    raw_square_sum_accumulator,
                );
            }
            raw_sum_accumulator = _mm256_setzero_si256();
            raw_square_sum_accumulator = _mm256_setzero_si256();
            accumulated_vector_count = 0;
        }

        sample_index += AVX2_SAMPLE_COUNT;
    }

    if accumulated_vector_count > 0 {
        unsafe {
            record_raw_dosage_accumulators_avx2(
                &mut raw_integer_summary,
                raw_sum_accumulator,
                raw_square_sum_accumulator,
            );
        }
    }
    raw_integer_summary.selected_observation_count +=
        i32::try_from(sample_index).expect("selected sample count should fit i32");
    if !decode_unphased_eight_bit_identity_raw_scalar_integer_stats_from(
        packed_probability_bytes,
        output_values,
        sample_index,
        &mut raw_integer_summary,
    ) {
        return None;
    }
    Some(raw_integer_summary.into_decode_summary())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn copy_unphased_eight_bit_probability_pairs_and_summarize_raw_avx2(
    packed_probability_bytes: &[u8],
    output_probability_bytes: &mut [MaybeUninit<u8>],
    collect_sparse_candidate_counts: bool,
) -> DosageSummary {
    let mut raw_integer_summary = EightBitRawIntegerSummary::new(collect_sparse_candidate_counts);
    // Probability pairs are little-endian bytes [reference, heterozygous], so
    // 0x0102 applies the raw-dosage weights [2, 1] to every pair.
    let raw_dosage_probability_multipliers = _mm256_set1_epi16(0x0102);
    let raw_dosage_base = _mm256_set1_epi16(510);
    let adjacent_lane_sum_multiplier = _mm256_set1_epi16(1);
    let zero = _mm256_setzero_si256();
    let homozygous_alternate_lower_bound = _mm256_set1_epi16(382);
    let mut raw_sum_accumulator = _mm256_setzero_si256();
    let mut raw_square_sum_accumulator = _mm256_setzero_si256();
    let mut accumulated_vector_count = 0_usize;
    let sample_count = packed_probability_bytes.len() / 2;
    let mut sample_index = 0_usize;
    while sample_index + AVX2_PACKED8_SAMPLE_COUNT <= sample_count {
        let probability_pointer = unsafe { packed_probability_bytes.as_ptr().add(sample_index * 2).cast::<__m256i>() };
        let probability_words = unsafe { _mm256_loadu_si256(probability_pointer) };
        let output_probability_pointer =
            unsafe { output_probability_bytes.as_mut_ptr().cast::<u8>().add(sample_index * 2).cast::<__m256i>() };
        unsafe {
            _mm256_storeu_si256(output_probability_pointer, probability_words);
        }

        let weighted_probability_bytes = _mm256_maddubs_epi16(probability_words, raw_dosage_probability_multipliers);
        let raw_dosage_integers = _mm256_sub_epi16(raw_dosage_base, weighted_probability_bytes);

        raw_sum_accumulator =
            _mm256_add_epi32(raw_sum_accumulator, _mm256_madd_epi16(raw_dosage_integers, adjacent_lane_sum_multiplier));
        raw_square_sum_accumulator =
            _mm256_add_epi32(raw_square_sum_accumulator, _mm256_madd_epi16(raw_dosage_integers, raw_dosage_integers));
        accumulated_vector_count += 1;

        if collect_sparse_candidate_counts {
            raw_integer_summary.zero_count +=
                avx2_mask_count(_mm256_movemask_epi8(_mm256_cmpeq_epi16(raw_dosage_integers, zero))) / 2;
            raw_integer_summary.homozygous_alternate_count += avx2_mask_count(_mm256_movemask_epi8(
                _mm256_cmpgt_epi16(raw_dosage_integers, homozygous_alternate_lower_bound),
            )) / 2;
        }

        if accumulated_vector_count == AVX2_ACCUMULATION_VECTOR_LIMIT {
            unsafe {
                record_raw_dosage_accumulators_avx2(
                    &mut raw_integer_summary,
                    raw_sum_accumulator,
                    raw_square_sum_accumulator,
                );
            }
            raw_sum_accumulator = _mm256_setzero_si256();
            raw_square_sum_accumulator = _mm256_setzero_si256();
            accumulated_vector_count = 0;
        }

        sample_index += AVX2_PACKED8_SAMPLE_COUNT;
    }

    if accumulated_vector_count > 0 {
        unsafe {
            record_raw_dosage_accumulators_avx2(
                &mut raw_integer_summary,
                raw_sum_accumulator,
                raw_square_sum_accumulator,
            );
        }
    }
    raw_integer_summary.selected_observation_count +=
        i32::try_from(sample_index).expect("selected sample count should fit i32");
    copy_unphased_eight_bit_probability_pairs_and_summarize_raw_scalar_integer_stats_from(
        packed_probability_bytes,
        output_probability_bytes,
        sample_index,
        &mut raw_integer_summary,
    );
    raw_integer_summary.into_decode_summary()
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn decode_unphased_eight_bit_identity_raw_avx2_scalar_stats(
    packed_probability_bytes: &[u8],
    output_values: &mut [f32],
) -> DosageSummary {
    let mut raw_integer_summary = EightBitRawIntegerSummary::default();
    let probability_byte_mask = _mm256_set1_epi32(0xFF);
    let raw_dosage_base = _mm256_set1_epi32(510);
    let probability_scale_reciprocal = _mm256_set1_ps(EIGHT_BIT_PROBABILITY_SCALE_RECIPROCAL);
    let mut sample_index = 0_usize;
    while sample_index + AVX2_SAMPLE_COUNT <= output_values.len() {
        let probability_pointer = unsafe { packed_probability_bytes.as_ptr().add(sample_index * 2).cast::<__m128i>() };
        let probability_words = unsafe { _mm_loadu_si128(probability_pointer) };
        let probability_indices = _mm256_cvtepu16_epi32(probability_words);
        let homozygous_reference_probability_bytes = _mm256_and_si256(probability_indices, probability_byte_mask);
        let heterozygous_probability_bytes = _mm256_srli_epi32(probability_indices, 8);
        let doubled_homozygous_reference_probability_bytes =
            _mm256_slli_epi32(homozygous_reference_probability_bytes, 1);
        let raw_dosage_integers = _mm256_sub_epi32(
            _mm256_sub_epi32(raw_dosage_base, doubled_homozygous_reference_probability_bytes),
            heterozygous_probability_bytes,
        );
        let dosage_values = _mm256_mul_ps(_mm256_cvtepi32_ps(raw_dosage_integers), probability_scale_reciprocal);
        unsafe {
            _mm256_storeu_ps(output_values.as_mut_ptr().add(sample_index), dosage_values);
        }

        let mut raw_dosage_chunk = [0_i32; AVX2_SAMPLE_COUNT];
        unsafe {
            _mm256_storeu_si256(raw_dosage_chunk.as_mut_ptr().cast::<__m256i>(), raw_dosage_integers);
        }
        for raw_dosage_integer in raw_dosage_chunk {
            raw_integer_summary.record_raw_dosage_integer(raw_dosage_integer);
        }

        sample_index += AVX2_SAMPLE_COUNT;
    }

    decode_unphased_eight_bit_identity_raw_scalar_integer_stats_from(
        packed_probability_bytes,
        output_values,
        sample_index,
        &mut raw_integer_summary,
    );
    raw_integer_summary.into_decode_summary()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRUSTED_IDENTITY_SAMPLE_COUNTS: [usize; 10] = [0, 1, 7, 8, 15, 16, 17, 31, 32, 33];
    const LOOKUP_RAW_MAX_DELTA_TOLERANCE: f32 = 1.0e-6;

    fn dosage_lookup() -> &'static [f32] {
        super::super::decode::unphased_eight_bit_dosage_lookup()
    }

    fn probability_bytes() -> Vec<u8> {
        vec![0, 0, 255, 0, 0, 255, 128, 0, 0, 128, 64, 64, 255, 255, 3, 252, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }

    fn probability_pair_for_raw_dosage(raw_dosage_integer: i32) -> [u8; 2] {
        let reference_probability_units = 510_i32 - raw_dosage_integer;
        [
            u8::try_from(reference_probability_units / 2).expect("homozygous reference probability should fit u8"),
            u8::try_from(reference_probability_units % 2).expect("heterozygous probability should fit u8"),
        ]
    }

    fn all_dosage_zero_probability_bytes(sample_count: usize) -> Vec<u8> {
        let mut probabilities = Vec::with_capacity(sample_count * 2);
        for _ in 0..sample_count {
            probabilities.extend_from_slice(&[255, 0]);
        }
        probabilities
    }

    fn all_dosage_two_probability_bytes(sample_count: usize) -> Vec<u8> {
        vec![0; sample_count * 2]
    }

    fn alternating_raw_probability_bytes(sample_count: usize) -> Vec<u8> {
        let raw_dosage_integers = [0, 1, 127, 128, 255, 382, 383, 510];
        let mut probabilities = Vec::with_capacity(sample_count * 2);
        for sample_index in 0..sample_count {
            probabilities.extend_from_slice(&probability_pair_for_raw_dosage(
                raw_dosage_integers[sample_index % raw_dosage_integers.len()],
            ));
        }
        probabilities
    }

    fn rare_variant_like_probability_bytes(sample_count: usize) -> Vec<u8> {
        let raw_dosage_integers = [0, 0, 0, 0, 0, 1, 2, 255, 510];
        let mut probabilities = Vec::with_capacity(sample_count * 2);
        for sample_index in 0..sample_count {
            probabilities.extend_from_slice(&probability_pair_for_raw_dosage(
                raw_dosage_integers[sample_index % raw_dosage_integers.len()],
            ));
        }
        probabilities
    }

    fn deterministic_random_valid_probability_bytes(sample_count: usize) -> Vec<u8> {
        let mut generator_state = 0x9E37_79B9_u32;
        let mut probabilities = Vec::with_capacity(sample_count * 2);
        for _ in 0..sample_count {
            generator_state = generator_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let homozygous_reference_probability_byte =
                u8::try_from(generator_state & 0xFF).expect("generator low byte should fit u8");
            generator_state = generator_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let heterozygous_limit = 255_u16 - u16::from(homozygous_reference_probability_byte);
            let heterozygous_probability_byte = u8::try_from(
                u16::try_from(generator_state & 0xFF).expect("generator low byte should fit u16")
                    % (heterozygous_limit + 1),
            )
            .expect("heterozygous probability should fit u8");
            probabilities.push(homozygous_reference_probability_byte);
            probabilities.push(heterozygous_probability_byte);
        }
        probabilities
    }

    fn expected_raw_summary(probabilities: &[u8]) -> DosageSummary {
        let mut decode_summary = DosageSummary::default();
        let (probability_pairs, _) = probabilities.as_chunks::<2>();
        for [homozygous_reference_probability_byte, heterozygous_probability_byte] in probability_pairs.iter().copied()
        {
            decode_summary.record_raw_dosage_integer_from_f32_accumulation(raw_dosage_integer(
                homozygous_reference_probability_byte,
                heterozygous_probability_byte,
            ));
        }
        decode_summary
    }

    fn expected_raw_integer_summary(probabilities: &[u8]) -> DosageSummary {
        let mut raw_integer_summary = EightBitRawIntegerSummary::default();
        let (probability_pairs, _) = probabilities.as_chunks::<2>();
        for [homozygous_reference_probability_byte, heterozygous_probability_byte] in probability_pairs.iter().copied()
        {
            raw_integer_summary.record_raw_dosage_integer(raw_dosage_integer(
                homozygous_reference_probability_byte,
                heterozygous_probability_byte,
            ));
        }
        raw_integer_summary.into_decode_summary()
    }

    fn max_absolute_delta(left_values: &[f32], right_values: &[f32]) -> f32 {
        left_values
            .iter()
            .zip(right_values)
            .map(|(left_value, right_value)| (left_value - right_value).abs())
            .fold(0.0_f32, f32::max)
    }

    fn probability_patterns(sample_count: usize) -> [Vec<u8>; 5] {
        [
            all_dosage_zero_probability_bytes(sample_count),
            all_dosage_two_probability_bytes(sample_count),
            alternating_raw_probability_bytes(sample_count),
            rare_variant_like_probability_bytes(sample_count),
            deterministic_random_valid_probability_bytes(sample_count),
        ]
    }

    #[test]
    fn trusted_identity_wrapper_uses_selected_decode_path() {
        let probabilities = probability_bytes();
        let sample_count = probabilities.len() / 2;
        let mut expected_output = vec![0.0_f32; sample_count];
        let mut wrapper_output = vec![0.0_f32; sample_count];

        let expected_summary =
            decode_unphased_eight_bit_identity_raw_scalar_integer_stats(&probabilities, &mut expected_output);
        let wrapper_summary = decode_unphased_eight_bit_identity_simd_or_scalar(&probabilities, &mut wrapper_output);

        assert_eq!(wrapper_output, expected_output);
        assert_eq!(wrapper_summary, expected_summary);
    }

    #[test]
    fn packed8_copy_summary_wrapper_copies_bytes_and_matches_identity_stats() {
        for sample_count in TRUSTED_IDENTITY_SAMPLE_COUNTS {
            for probabilities in probability_patterns(sample_count) {
                let mut expected_output = vec![0.0_f32; sample_count];
                let mut copied_probabilities = vec![0_u8; probabilities.len()];

                let expected_summary =
                    decode_unphased_eight_bit_identity_raw_scalar_integer_stats(&probabilities, &mut expected_output);
                let copied_summary = copy_unphased_eight_bit_probability_pairs_and_summarize_simd_or_scalar(
                    &probabilities,
                    &mut copied_probabilities,
                );

                assert_eq!(copied_probabilities, probabilities);
                assert_eq!(copied_summary, expected_summary);
            }
        }
    }

    #[test]
    fn trusted_identity_raw_scalar_matches_lookup_with_bounded_delta_and_expected_counts() {
        let lookup = dosage_lookup();
        for sample_count in TRUSTED_IDENTITY_SAMPLE_COUNTS {
            for probabilities in probability_patterns(sample_count) {
                let mut lookup_output = vec![0.0_f32; sample_count];
                let mut raw_output = vec![0.0_f32; sample_count];
                let sample_count_float =
                    f32::from(u16::try_from(sample_count).expect("test sample count should fit u16"));

                let lookup_summary =
                    decode_unphased_eight_bit_identity_lookup_scalar(&probabilities, lookup, &mut lookup_output);
                let raw_summary = decode_unphased_eight_bit_identity_raw_scalar(&probabilities, &mut raw_output);
                let expected_summary = expected_raw_summary(&probabilities);

                assert!(
                    max_absolute_delta(&lookup_output, &raw_output) <= LOOKUP_RAW_MAX_DELTA_TOLERANCE,
                    "lookup and raw output delta exceeded tolerance for {sample_count} samples"
                );
                assert!(
                    (lookup_summary.selected_dosage_total - raw_summary.selected_dosage_total).abs()
                        <= sample_count_float * LOOKUP_RAW_MAX_DELTA_TOLERANCE,
                    "lookup and raw dosage sums diverged for {sample_count} samples"
                );
                assert!(
                    (lookup_summary.selected_dosage_square_total - raw_summary.selected_dosage_square_total).abs()
                        <= sample_count_float * LOOKUP_RAW_MAX_DELTA_TOLERANCE * 4.0,
                    "lookup and raw dosage square sums diverged for {sample_count} samples"
                );
                assert_eq!(lookup_summary.selected_observation_count, expected_summary.selected_observation_count);
                assert_eq!(lookup_summary.zero_count, expected_summary.zero_count);
                assert_eq!(lookup_summary.nonzero_count, expected_summary.nonzero_count);
                assert_eq!(lookup_summary.homozygous_reference_count, expected_summary.homozygous_reference_count);
                assert_eq!(lookup_summary.heterozygous_count, expected_summary.heterozygous_count);
                assert_eq!(lookup_summary.homozygous_alternate_count, expected_summary.homozygous_alternate_count);
                assert_eq!(raw_summary, expected_summary);
            }
        }
    }

    #[test]
    fn trusted_identity_integer_stats_match_expected_raw_summaries() {
        for sample_count in TRUSTED_IDENTITY_SAMPLE_COUNTS {
            for probabilities in probability_patterns(sample_count) {
                let mut output = vec![0.0_f32; sample_count];

                let integer_summary =
                    decode_unphased_eight_bit_identity_raw_scalar_integer_stats(&probabilities, &mut output);
                let expected_summary = expected_raw_integer_summary(&probabilities);

                assert_eq!(integer_summary, expected_summary);
                assert!(output.iter().all(|dosage_value| (0.0..=2.0).contains(dosage_value)));
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn trusted_identity_raw_avx2_outputs_match_raw_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        for sample_count in TRUSTED_IDENTITY_SAMPLE_COUNTS {
            for probabilities in probability_patterns(sample_count) {
                let mut scalar_output = vec![0.0_f32; sample_count];
                let mut avx2_output = vec![0.0_f32; sample_count];

                let scalar_summary = decode_unphased_eight_bit_identity_raw_scalar(&probabilities, &mut scalar_output);
                let avx2_summary =
                    unsafe { decode_unphased_eight_bit_identity_raw_avx2(&probabilities, &mut avx2_output) };
                let expected_integer_summary = expected_raw_integer_summary(&probabilities);

                assert_eq!(avx2_output, scalar_output);
                assert_eq!(avx2_summary.selected_observation_count, scalar_summary.selected_observation_count);
                assert_eq!(avx2_summary.zero_count, scalar_summary.zero_count);
                assert_eq!(avx2_summary.nonzero_count, scalar_summary.nonzero_count);
                assert_eq!(avx2_summary.homozygous_reference_count, scalar_summary.homozygous_reference_count);
                assert_eq!(avx2_summary.heterozygous_count, scalar_summary.heterozygous_count);
                assert_eq!(avx2_summary.homozygous_alternate_count, scalar_summary.homozygous_alternate_count);
                assert_eq!(avx2_summary, expected_integer_summary);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn packed8_copy_summary_avx2_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        for sample_count in TRUSTED_IDENTITY_SAMPLE_COUNTS {
            for probabilities in probability_patterns(sample_count) {
                let mut scalar_output = vec![0_u8; probabilities.len()];
                let mut avx2_output = vec![0_u8; probabilities.len()];

                let scalar_summary = copy_unphased_eight_bit_probability_pairs_and_summarize_raw_scalar_integer_stats(
                    &probabilities,
                    &mut scalar_output,
                );
                let avx2_summary = unsafe {
                    copy_unphased_eight_bit_probability_pairs_and_summarize_raw_avx2(&probabilities, &mut avx2_output)
                };

                assert_eq!(avx2_output, scalar_output);
                assert_eq!(avx2_output, probabilities);
                assert_eq!(avx2_summary, scalar_summary);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn trusted_identity_raw_avx2_integer_stats_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        for sample_count in TRUSTED_IDENTITY_SAMPLE_COUNTS {
            for probabilities in probability_patterns(sample_count) {
                let mut scalar_output = vec![0.0_f32; sample_count];
                let mut avx2_output = vec![0.0_f32; sample_count];

                let scalar_summary =
                    decode_unphased_eight_bit_identity_raw_scalar_integer_stats(&probabilities, &mut scalar_output);
                let avx2_summary =
                    unsafe { decode_unphased_eight_bit_identity_raw_avx2(&probabilities, &mut avx2_output) };

                assert_eq!(avx2_output, scalar_output);
                assert_eq!(avx2_summary, scalar_summary);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn trusted_identity_raw_avx2_vector_stats_match_scalar_stats_baseline_after_periodic_reduction() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let sample_count = (AVX2_ACCUMULATION_VECTOR_LIMIT + 2) * AVX2_SAMPLE_COUNT + 3;
        let probabilities = deterministic_random_valid_probability_bytes(sample_count);
        let mut baseline_output = vec![0.0_f32; sample_count];
        let mut avx2_output = vec![0.0_f32; sample_count];

        let baseline_summary =
            unsafe { decode_unphased_eight_bit_identity_raw_avx2_scalar_stats(&probabilities, &mut baseline_output) };
        let avx2_summary = unsafe { decode_unphased_eight_bit_identity_raw_avx2(&probabilities, &mut avx2_output) };

        assert_eq!(avx2_output, baseline_output);
        assert_eq!(avx2_summary, baseline_summary);
    }
}
