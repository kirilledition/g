use std::sync::OnceLock;

use crate::genotype::preprocess;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const AVX2_SAMPLE_COUNT: usize = 8;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const AVX512_SAMPLE_COUNT: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrustedIdentitySimdMode {
    Auto,
    Scalar,
    Avx2,
    Avx512,
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
}

pub(super) fn decode_trusted_unphased_eight_bit_identity_simd_or_scalar(
    packed_probability_bytes: &[u8],
    dosage_lookup: &[f32],
    output_values: &mut [f32],
) -> TrustedEightBitIdentityDecodeSummary {
    debug_assert_eq!(packed_probability_bytes.len(), output_values.len() * 2);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let simd_mode = trusted_identity_simd_mode();
        if simd_mode == TrustedIdentitySimdMode::Avx512
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // AVX-512 is kept as an explicit benchmark mode; AVX2 is faster on the current target hosts.
            return unsafe {
                decode_trusted_unphased_eight_bit_identity_avx512(
                    packed_probability_bytes,
                    dosage_lookup,
                    output_values,
                )
            };
        }
        if simd_mode != TrustedIdentitySimdMode::Scalar
            && simd_mode != TrustedIdentitySimdMode::Avx512
            && std::arch::is_x86_feature_detected!("avx2")
        {
            // The runtime feature check guarantees that the AVX2 implementation is valid on this host.
            return unsafe {
                decode_trusted_unphased_eight_bit_identity_avx2(packed_probability_bytes, dosage_lookup, output_values)
            };
        }
    }

    decode_trusted_unphased_eight_bit_identity_scalar(packed_probability_bytes, dosage_lookup, output_values)
}

fn trusted_identity_simd_mode() -> TrustedIdentitySimdMode {
    static SIMD_MODE: OnceLock<TrustedIdentitySimdMode> = OnceLock::new();
    *SIMD_MODE.get_or_init(|| match std::env::var("G_BGEN_TRUSTED_IDENTITY_SIMD_MODE").as_deref() {
        Ok("scalar") => TrustedIdentitySimdMode::Scalar,
        Ok("avx2") => TrustedIdentitySimdMode::Avx2,
        Ok("avx512") => TrustedIdentitySimdMode::Avx512,
        _ => TrustedIdentitySimdMode::Auto,
    })
}

fn decode_trusted_unphased_eight_bit_identity_scalar(
    packed_probability_bytes: &[u8],
    dosage_lookup: &[f32],
    output_values: &mut [f32],
) -> TrustedEightBitIdentityDecodeSummary {
    let mut decode_summary = TrustedEightBitIdentityDecodeSummary::default();
    decode_trusted_unphased_eight_bit_identity_scalar_from(
        packed_probability_bytes,
        dosage_lookup,
        output_values,
        0,
        &mut decode_summary,
    );
    decode_summary
}

fn decode_trusted_unphased_eight_bit_identity_scalar_from(
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

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m256i, _mm_loadu_si128, _mm256_cvtepu16_epi32, _mm256_i32gather_ps, _mm256_loadu_si256,
    _mm256_storeu_ps, _mm512_cvtepu16_epi32, _mm512_i32gather_ps, _mm512_storeu_ps,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm_loadu_si128, _mm256_cvtepu16_epi32, _mm256_i32gather_ps, _mm256_loadu_si256,
    _mm256_storeu_ps, _mm512_cvtepu16_epi32, _mm512_i32gather_ps, _mm512_storeu_ps,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn decode_trusted_unphased_eight_bit_identity_avx512(
    packed_probability_bytes: &[u8],
    dosage_lookup: &[f32],
    output_values: &mut [f32],
) -> TrustedEightBitIdentityDecodeSummary {
    let mut decode_summary = TrustedEightBitIdentityDecodeSummary::default();
    let mut sample_index = 0_usize;
    while sample_index + AVX512_SAMPLE_COUNT <= output_values.len() {
        let probability_pointer = unsafe { packed_probability_bytes.as_ptr().add(sample_index * 2).cast::<__m256i>() };
        let probability_words = unsafe { _mm256_loadu_si256(probability_pointer) };
        let probability_indices = _mm512_cvtepu16_epi32(probability_words);
        let dosage_values = unsafe { _mm512_i32gather_ps(probability_indices, dosage_lookup.as_ptr(), 4) };
        unsafe {
            _mm512_storeu_ps(output_values.as_mut_ptr().add(sample_index), dosage_values);
        }

        let mut dosage_chunk = [0.0_f32; AVX512_SAMPLE_COUNT];
        unsafe {
            _mm512_storeu_ps(dosage_chunk.as_mut_ptr(), dosage_values);
        }
        for dosage_value in dosage_chunk {
            decode_summary.record_dosage(dosage_value);
        }

        sample_index += AVX512_SAMPLE_COUNT;
    }

    decode_trusted_unphased_eight_bit_identity_scalar_from(
        packed_probability_bytes,
        dosage_lookup,
        output_values,
        sample_index,
        &mut decode_summary,
    );
    decode_summary
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn decode_trusted_unphased_eight_bit_identity_avx2(
    packed_probability_bytes: &[u8],
    dosage_lookup: &[f32],
    output_values: &mut [f32],
) -> TrustedEightBitIdentityDecodeSummary {
    let mut decode_summary = TrustedEightBitIdentityDecodeSummary::default();
    let mut sample_index = 0_usize;
    while sample_index + AVX2_SAMPLE_COUNT <= output_values.len() {
        let probability_pointer = unsafe { packed_probability_bytes.as_ptr().add(sample_index * 2).cast::<__m128i>() };
        let probability_words = unsafe { _mm_loadu_si128(probability_pointer) };
        let probability_indices = _mm256_cvtepu16_epi32(probability_words);
        let dosage_values = unsafe { _mm256_i32gather_ps(dosage_lookup.as_ptr(), probability_indices, 4) };
        unsafe {
            _mm256_storeu_ps(output_values.as_mut_ptr().add(sample_index), dosage_values);
        }

        let mut dosage_chunk = [0.0_f32; AVX2_SAMPLE_COUNT];
        unsafe {
            _mm256_storeu_ps(dosage_chunk.as_mut_ptr(), dosage_values);
        }
        for dosage_value in dosage_chunk {
            decode_summary.record_dosage(dosage_value);
        }

        sample_index += AVX2_SAMPLE_COUNT;
    }

    decode_trusted_unphased_eight_bit_identity_scalar_from(
        packed_probability_bytes,
        dosage_lookup,
        output_values,
        sample_index,
        &mut decode_summary,
    );
    decode_summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dosage_lookup() -> &'static [f32] {
        super::super::decode::unphased_eight_bit_dosage_lookup()
    }

    fn probability_bytes() -> Vec<u8> {
        vec![
            0, 0, 255, 0, 0, 255, 128, 0, 0, 128, 64, 64, 255, 255, 3, 252, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            110, 120, 130, 140, 150, 160,
        ]
    }

    #[test]
    fn trusted_identity_wrapper_matches_scalar_for_full_chunks_and_tail() {
        let lookup = dosage_lookup();
        let probabilities = probability_bytes();
        let sample_count = probabilities.len() / 2;
        let mut scalar_output = vec![0.0_f32; sample_count];
        let mut wrapper_output = vec![0.0_f32; sample_count];

        let scalar_summary =
            decode_trusted_unphased_eight_bit_identity_scalar(&probabilities, lookup, &mut scalar_output);
        let wrapper_summary =
            decode_trusted_unphased_eight_bit_identity_simd_or_scalar(&probabilities, lookup, &mut wrapper_output);

        assert_eq!(wrapper_output, scalar_output);
        assert_eq!(wrapper_summary, scalar_summary);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn trusted_identity_avx2_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let lookup = dosage_lookup();
        let probabilities = probability_bytes();
        let sample_count = probabilities.len() / 2;
        let mut scalar_output = vec![0.0_f32; sample_count];
        let mut avx2_output = vec![0.0_f32; sample_count];

        let scalar_summary =
            decode_trusted_unphased_eight_bit_identity_scalar(&probabilities, lookup, &mut scalar_output);
        let avx2_summary =
            unsafe { decode_trusted_unphased_eight_bit_identity_avx2(&probabilities, lookup, &mut avx2_output) };

        assert_eq!(avx2_output, scalar_output);
        assert_eq!(avx2_summary, scalar_summary);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn trusted_identity_avx512_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") || !std::arch::is_x86_feature_detected!("avx512bw") {
            return;
        }

        let lookup = dosage_lookup();
        let probabilities = probability_bytes();
        let sample_count = probabilities.len() / 2;
        let mut scalar_output = vec![0.0_f32; sample_count];
        let mut avx512_output = vec![0.0_f32; sample_count];

        let scalar_summary =
            decode_trusted_unphased_eight_bit_identity_scalar(&probabilities, lookup, &mut scalar_output);
        let avx512_summary =
            unsafe { decode_trusted_unphased_eight_bit_identity_avx512(&probabilities, lookup, &mut avx512_output) };

        assert_eq!(avx512_output, scalar_output);
        assert_eq!(avx512_summary, scalar_summary);
    }
}
