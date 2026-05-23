use std::env;

use crate::genotype::preprocess;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const AVX2_SAMPLE_COUNT: usize = 8;
const EIGHT_BIT_PROBABILITY_SCALE_RECIPROCAL: f32 = 1.0_f32 / 255.0_f32;
const EIGHT_BIT_PROBABILITY_SCALE_SQUARE_RECIPROCAL: f32 = 1.0_f32 / (255.0_f32 * 255.0_f32);
const BGEN_SIMD_MODE_ENVIRONMENT_VARIABLE: &str = "G_BGEN_SIMD";
const TRUSTED_IDENTITY_MODE_ENVIRONMENT_VARIABLE: &str = "G_BGEN_TRUSTED_IDENTITY_MODE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TrustedIdentityDecodeMode {
    Auto,
    Lookup,
    RawScalar,
    RawAvx2,
}

impl TrustedIdentityDecodeMode {
    fn parse(mode_name: &str) -> Option<Self> {
        match mode_name {
            "auto" => Some(Self::Auto),
            "lookup" => Some(Self::Lookup),
            "raw_scalar" | "scalar" => Some(Self::RawScalar),
            "raw_avx2" | "avx2" => Some(Self::RawAvx2),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct TrustedEightBitIdentityDecodeSummary {
    pub(super) selected_dosage_total: f32,
    pub(super) selected_dosage_square_total: f32,
    pub(super) selected_observation_count: i32,
    pub(super) zero_count: i32,
    pub(super) nonzero_count: i32,
    pub(super) homozygous_reference_count: i32,
    pub(super) heterozygous_count: i32,
    pub(super) homozygous_alternate_count: i32,
}

pub(super) fn trusted_identity_decode_mode_from_environment() -> TrustedIdentityDecodeMode {
    let mode_name = match env::var(BGEN_SIMD_MODE_ENVIRONMENT_VARIABLE) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => match env::var(TRUSTED_IDENTITY_MODE_ENVIRONMENT_VARIABLE) {
            Ok(value) => value,
            Err(env::VarError::NotPresent) => return TrustedIdentityDecodeMode::Auto,
            Err(env::VarError::NotUnicode(_)) => {
                panic!("{TRUSTED_IDENTITY_MODE_ENVIRONMENT_VARIABLE} must contain valid UTF-8")
            }
        },
        Err(env::VarError::NotUnicode(_)) => panic!("{BGEN_SIMD_MODE_ENVIRONMENT_VARIABLE} must contain valid UTF-8"),
    };

    TrustedIdentityDecodeMode::parse(&mode_name).unwrap_or_else(|| {
        panic!(
            "{BGEN_SIMD_MODE_ENVIRONMENT_VARIABLE} must be one of auto, lookup, raw_scalar, or raw_avx2; received '{mode_name}'"
        )
    })
}

impl TrustedEightBitIdentityDecodeSummary {
    fn record_dosage(&mut self, dosage_value: f32) {
        self.selected_dosage_total += dosage_value;
        self.selected_dosage_square_total += dosage_value * dosage_value;
        self.selected_observation_count += 1;
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
        self.selected_dosage_total += dosage_value;
        self.selected_dosage_square_total += dosage_value * dosage_value;
        self.selected_observation_count += 1;
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
struct TrustedEightBitRawIntegerSummary {
    raw_dosage_total: i64,
    raw_dosage_square_total: u64,
    selected_observation_count: i32,
    zero_count: i32,
    nonzero_count: i32,
    homozygous_reference_count: i32,
    heterozygous_count: i32,
    homozygous_alternate_count: i32,
}

impl TrustedEightBitRawIntegerSummary {
    fn record_raw_dosage_integer(&mut self, raw_dosage_integer: i32) {
        let raw_dosage_integer_i64 = i64::from(raw_dosage_integer);
        self.raw_dosage_total += raw_dosage_integer_i64;
        self.raw_dosage_square_total +=
            u64::try_from(raw_dosage_integer_i64 * raw_dosage_integer_i64).expect("raw dosage square should fit u64");
        self.selected_observation_count += 1;
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

    #[allow(clippy::cast_precision_loss)]
    fn into_decode_summary(self) -> TrustedEightBitIdentityDecodeSummary {
        TrustedEightBitIdentityDecodeSummary {
            selected_dosage_total: self.raw_dosage_total as f32 * EIGHT_BIT_PROBABILITY_SCALE_RECIPROCAL,
            selected_dosage_square_total: self.raw_dosage_square_total as f32
                * EIGHT_BIT_PROBABILITY_SCALE_SQUARE_RECIPROCAL,
            selected_observation_count: self.selected_observation_count,
            zero_count: self.zero_count,
            nonzero_count: self.nonzero_count,
            homozygous_reference_count: self.homozygous_reference_count,
            heterozygous_count: self.heterozygous_count,
            homozygous_alternate_count: self.homozygous_alternate_count,
        }
    }
}

pub(super) fn decode_trusted_unphased_eight_bit_identity_simd_or_scalar(
    packed_probability_bytes: &[u8],
    dosage_lookup: &[f32],
    output_values: &mut [f32],
    decode_mode: TrustedIdentityDecodeMode,
) -> TrustedEightBitIdentityDecodeSummary {
    debug_assert_eq!(packed_probability_bytes.len(), output_values.len() * 2);

    match decode_mode {
        TrustedIdentityDecodeMode::Lookup => {
            return decode_trusted_unphased_eight_bit_identity_lookup_scalar(
                packed_probability_bytes,
                dosage_lookup,
                output_values,
            );
        }
        TrustedIdentityDecodeMode::RawScalar => {
            return decode_trusted_unphased_eight_bit_identity_raw_scalar_integer_stats(
                packed_probability_bytes,
                output_values,
            );
        }
        TrustedIdentityDecodeMode::RawAvx2 => {
            return decode_trusted_unphased_eight_bit_identity_raw_avx2_checked(
                packed_probability_bytes,
                output_values,
            );
        }
        TrustedIdentityDecodeMode::Auto => {}
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // Benchmarks on the trusted full-sample path selected raw-integer AVX2 over lookup-gather AVX2.
            return unsafe {
                decode_trusted_unphased_eight_bit_identity_raw_avx2(packed_probability_bytes, output_values)
            };
        }
    }

    decode_trusted_unphased_eight_bit_identity_lookup_scalar(packed_probability_bytes, dosage_lookup, output_values)
}

fn decode_trusted_unphased_eight_bit_identity_raw_scalar_integer_stats(
    packed_probability_bytes: &[u8],
    output_values: &mut [f32],
) -> TrustedEightBitIdentityDecodeSummary {
    let mut raw_integer_summary = TrustedEightBitRawIntegerSummary::default();
    decode_trusted_unphased_eight_bit_identity_raw_scalar_integer_stats_from(
        packed_probability_bytes,
        output_values,
        0,
        &mut raw_integer_summary,
    );
    raw_integer_summary.into_decode_summary()
}

fn decode_trusted_unphased_eight_bit_identity_raw_scalar_integer_stats_from(
    packed_probability_bytes: &[u8],
    output_values: &mut [f32],
    start_sample_index: usize,
    raw_integer_summary: &mut TrustedEightBitRawIntegerSummary,
) {
    for (relative_sample_index, probability_pair) in packed_probability_bytes[start_sample_index * 2..]
        .chunks_exact(2)
        .take(output_values.len().saturating_sub(start_sample_index))
        .enumerate()
    {
        let output_index = start_sample_index + relative_sample_index;
        let raw_dosage_integer = raw_dosage_integer(probability_pair[0], probability_pair[1]);
        output_values[output_index] = raw_dosage_value(raw_dosage_integer);
        raw_integer_summary.record_raw_dosage_integer(raw_dosage_integer);
    }
}

fn decode_trusted_unphased_eight_bit_identity_lookup_scalar(
    packed_probability_bytes: &[u8],
    dosage_lookup: &[f32],
    output_values: &mut [f32],
) -> TrustedEightBitIdentityDecodeSummary {
    let mut decode_summary = TrustedEightBitIdentityDecodeSummary::default();
    decode_trusted_unphased_eight_bit_identity_lookup_scalar_from(
        packed_probability_bytes,
        dosage_lookup,
        output_values,
        0,
        &mut decode_summary,
    );
    decode_summary
}

fn decode_trusted_unphased_eight_bit_identity_lookup_scalar_from(
    packed_probability_bytes: &[u8],
    dosage_lookup: &[f32],
    output_values: &mut [f32],
    start_sample_index: usize,
    decode_summary: &mut TrustedEightBitIdentityDecodeSummary,
) {
    for (relative_sample_index, probability_pair) in packed_probability_bytes[start_sample_index * 2..]
        .chunks_exact(2)
        .take(output_values.len().saturating_sub(start_sample_index))
        .enumerate()
    {
        let output_index = start_sample_index + relative_sample_index;
        let packed_probability_index = usize::from(probability_pair[0]) | (usize::from(probability_pair[1]) << 8);
        let dosage_value = dosage_lookup[packed_probability_index];
        output_values[output_index] = dosage_value;
        decode_summary.record_dosage(dosage_value);
    }
}

#[cfg(test)]
fn decode_trusted_unphased_eight_bit_identity_raw_scalar(
    packed_probability_bytes: &[u8],
    output_values: &mut [f32],
) -> TrustedEightBitIdentityDecodeSummary {
    let mut decode_summary = TrustedEightBitIdentityDecodeSummary::default();
    decode_trusted_unphased_eight_bit_identity_raw_scalar_from(
        packed_probability_bytes,
        output_values,
        0,
        &mut decode_summary,
    );
    decode_summary
}

#[cfg(test)]
fn decode_trusted_unphased_eight_bit_identity_raw_scalar_from(
    packed_probability_bytes: &[u8],
    output_values: &mut [f32],
    start_sample_index: usize,
    decode_summary: &mut TrustedEightBitIdentityDecodeSummary,
) {
    for (relative_sample_index, probability_pair) in packed_probability_bytes[start_sample_index * 2..]
        .chunks_exact(2)
        .take(output_values.len().saturating_sub(start_sample_index))
        .enumerate()
    {
        let output_index = start_sample_index + relative_sample_index;
        let raw_dosage_integer = raw_dosage_integer(probability_pair[0], probability_pair[1]);
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

pub fn benchmark_decode_trusted_unphased_eight_bit_identity_mode(
    mode_name: &str,
    packed_probability_bytes: &[u8],
    dosage_lookup: &[f32],
    output_values: &mut [f32],
) -> u64 {
    let decode_mode = TrustedIdentityDecodeMode::parse(mode_name)
        .unwrap_or_else(|| panic!("trusted identity benchmark mode '{mode_name}' is not supported"));
    let decode_summary = decode_trusted_unphased_eight_bit_identity_simd_or_scalar(
        packed_probability_bytes,
        dosage_lookup,
        output_values,
        decode_mode,
    );
    decode_summary_checksum(decode_summary)
}

fn decode_summary_checksum(decode_summary: TrustedEightBitIdentityDecodeSummary) -> u64 {
    u64::from(decode_summary.selected_dosage_total.to_bits())
        ^ (u64::from(decode_summary.selected_dosage_square_total.to_bits()) << 1)
        ^ u64::try_from(decode_summary.selected_observation_count).unwrap_or_default()
        ^ (u64::try_from(decode_summary.zero_count).unwrap_or_default() << 8)
        ^ (u64::try_from(decode_summary.nonzero_count).unwrap_or_default() << 16)
        ^ (u64::try_from(decode_summary.homozygous_reference_count).unwrap_or_default() << 24)
        ^ (u64::try_from(decode_summary.heterozygous_count).unwrap_or_default() << 32)
        ^ (u64::try_from(decode_summary.homozygous_alternate_count).unwrap_or_default() << 40)
}

fn decode_trusted_unphased_eight_bit_identity_raw_avx2_checked(
    packed_probability_bytes: &[u8],
    output_values: &mut [f32],
) -> TrustedEightBitIdentityDecodeSummary {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        assert!(std::arch::is_x86_feature_detected!("avx2"), "raw_avx2 mode requires AVX2 support");
        unsafe { decode_trusted_unphased_eight_bit_identity_raw_avx2(packed_probability_bytes, output_values) }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = packed_probability_bytes;
        let _ = output_values;
        panic!("raw_avx2 mode requires an x86 or x86_64 target");
    }
}

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m256i, _mm_loadu_si128, _mm256_and_si256, _mm256_cvtepi32_ps, _mm256_cvtepu16_epi32, _mm256_mul_ps,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_slli_epi32, _mm256_srli_epi32, _mm256_storeu_ps, _mm256_storeu_si256,
    _mm256_sub_epi32,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm_loadu_si128, _mm256_and_si256, _mm256_cvtepi32_ps, _mm256_cvtepu16_epi32, _mm256_mul_ps,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_slli_epi32, _mm256_srli_epi32, _mm256_storeu_ps, _mm256_storeu_si256,
    _mm256_sub_epi32,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn decode_trusted_unphased_eight_bit_identity_raw_avx2(
    packed_probability_bytes: &[u8],
    output_values: &mut [f32],
) -> TrustedEightBitIdentityDecodeSummary {
    let mut raw_integer_summary = TrustedEightBitRawIntegerSummary::default();
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

    decode_trusted_unphased_eight_bit_identity_raw_scalar_integer_stats_from(
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
            let homozygous_reference_probability_byte = (generator_state & 0xFF) as u8;
            generator_state = generator_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let heterozygous_limit = 255_u16 - u16::from(homozygous_reference_probability_byte);
            let heterozygous_probability_byte =
                u8::try_from((generator_state & 0xFF) as u16 % (heterozygous_limit + 1))
                    .expect("heterozygous probability should fit u8");
            probabilities.push(homozygous_reference_probability_byte);
            probabilities.push(heterozygous_probability_byte);
        }
        probabilities
    }

    fn expected_raw_summary(probabilities: &[u8]) -> TrustedEightBitIdentityDecodeSummary {
        let mut decode_summary = TrustedEightBitIdentityDecodeSummary::default();
        for probability_pair in probabilities.chunks_exact(2) {
            decode_summary.record_raw_dosage_integer_from_f32_accumulation(raw_dosage_integer(
                probability_pair[0],
                probability_pair[1],
            ));
        }
        decode_summary
    }

    fn expected_raw_integer_summary(probabilities: &[u8]) -> TrustedEightBitIdentityDecodeSummary {
        let mut raw_integer_summary = TrustedEightBitRawIntegerSummary::default();
        for probability_pair in probabilities.chunks_exact(2) {
            raw_integer_summary.record_raw_dosage_integer(raw_dosage_integer(probability_pair[0], probability_pair[1]));
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
        let lookup = dosage_lookup();
        let probabilities = probability_bytes();
        let sample_count = probabilities.len() / 2;
        let mut expected_output = vec![0.0_f32; sample_count];
        let mut wrapper_output = vec![0.0_f32; sample_count];

        let expected_summary = if cfg!(any(target_arch = "x86", target_arch = "x86_64"))
            && std::arch::is_x86_feature_detected!("avx2")
        {
            decode_trusted_unphased_eight_bit_identity_raw_scalar_integer_stats(&probabilities, &mut expected_output)
        } else {
            decode_trusted_unphased_eight_bit_identity_lookup_scalar(&probabilities, lookup, &mut expected_output)
        };
        let wrapper_summary = decode_trusted_unphased_eight_bit_identity_simd_or_scalar(
            &probabilities,
            lookup,
            &mut wrapper_output,
            TrustedIdentityDecodeMode::Auto,
        );

        assert_eq!(wrapper_output, expected_output);
        assert_eq!(wrapper_summary, expected_summary);
    }

    #[test]
    fn trusted_identity_explicit_modes_select_expected_decode_paths() {
        let lookup = dosage_lookup();
        let probabilities = probability_bytes();
        let sample_count = probabilities.len() / 2;
        let mut lookup_output = vec![0.0_f32; sample_count];
        let mut raw_scalar_output = vec![0.0_f32; sample_count];
        let mut explicit_lookup_output = vec![0.0_f32; sample_count];
        let mut explicit_raw_scalar_output = vec![0.0_f32; sample_count];

        let lookup_summary =
            decode_trusted_unphased_eight_bit_identity_lookup_scalar(&probabilities, lookup, &mut lookup_output);
        let raw_scalar_summary =
            decode_trusted_unphased_eight_bit_identity_raw_scalar_integer_stats(&probabilities, &mut raw_scalar_output);
        let explicit_lookup_summary = decode_trusted_unphased_eight_bit_identity_simd_or_scalar(
            &probabilities,
            lookup,
            &mut explicit_lookup_output,
            TrustedIdentityDecodeMode::Lookup,
        );
        let explicit_raw_scalar_summary = decode_trusted_unphased_eight_bit_identity_simd_or_scalar(
            &probabilities,
            lookup,
            &mut explicit_raw_scalar_output,
            TrustedIdentityDecodeMode::RawScalar,
        );

        assert_eq!(explicit_lookup_output, lookup_output);
        assert_eq!(explicit_lookup_summary, lookup_summary);
        assert_eq!(explicit_raw_scalar_output, raw_scalar_output);
        assert_eq!(explicit_raw_scalar_summary, raw_scalar_summary);
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

                let lookup_summary = decode_trusted_unphased_eight_bit_identity_lookup_scalar(
                    &probabilities,
                    lookup,
                    &mut lookup_output,
                );
                let raw_summary =
                    decode_trusted_unphased_eight_bit_identity_raw_scalar(&probabilities, &mut raw_output);
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
                    decode_trusted_unphased_eight_bit_identity_raw_scalar_integer_stats(&probabilities, &mut output);
                let expected_summary = expected_raw_integer_summary(&probabilities);

                assert_eq!(integer_summary, expected_summary);
                assert!(output.iter().all(|dosage_value| (-1.0..=2.0).contains(dosage_value)));
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

                let scalar_summary =
                    decode_trusted_unphased_eight_bit_identity_raw_scalar(&probabilities, &mut scalar_output);
                let avx2_summary =
                    unsafe { decode_trusted_unphased_eight_bit_identity_raw_avx2(&probabilities, &mut avx2_output) };
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
    fn trusted_identity_raw_avx2_integer_stats_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        for sample_count in TRUSTED_IDENTITY_SAMPLE_COUNTS {
            for probabilities in probability_patterns(sample_count) {
                let mut scalar_output = vec![0.0_f32; sample_count];
                let mut avx2_output = vec![0.0_f32; sample_count];

                let scalar_summary = decode_trusted_unphased_eight_bit_identity_raw_scalar_integer_stats(
                    &probabilities,
                    &mut scalar_output,
                );
                let avx2_summary =
                    unsafe { decode_trusted_unphased_eight_bit_identity_raw_avx2(&probabilities, &mut avx2_output) };

                assert_eq!(avx2_output, scalar_output);
                assert_eq!(avx2_summary, scalar_summary);
            }
        }
    }
}
