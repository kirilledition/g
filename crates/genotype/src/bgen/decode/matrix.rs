use std::ptr::NonNull;
use std::sync::OnceLock;

use super::super::BgenError;
use super::probability::read_exact_bytes;

pub(in crate::bgen) struct VariantMajorTileStatsMut<'a> {
    pub(in crate::bgen) dosage_sum: &'a mut [f32],
    pub(in crate::bgen) dosage_square_sum: &'a mut [f32],
    pub(in crate::bgen) observation_count: &'a mut [i32],
    pub(in crate::bgen) sparse_candidate_counts: Option<VariantMajorSparseCandidateCountsMut<'a>>,
}

pub(in crate::bgen) struct VariantMajorSparseCandidateCountsMut<'a> {
    pub(in crate::bgen) zero_count: &'a mut [i32],
    pub(in crate::bgen) homozygous_alternate_count: &'a mut [i32],
}

pub(in crate::bgen) fn selected_sample_count_to_i32(selected_sample_count: usize) -> Result<i32, BgenError> {
    i32::try_from(selected_sample_count).map_err(|_| {
        BgenError::Range(format!(
            "Selected sample count {selected_sample_count} exceeds the supported i32 statistics range.",
        ))
    })
}

pub(in crate::bgen) fn unphased_eight_bit_dosage_lookup() -> &'static [f32] {
    static UNPHASED_EIGHT_BIT_DOSAGE_LOOKUP: OnceLock<Vec<f32>> = OnceLock::new();
    UNPHASED_EIGHT_BIT_DOSAGE_LOOKUP.get_or_init(|| {
        let reciprocal_scale = 1.0_f32 / 255.0_f32;
        let mut dosage_lookup = Vec::with_capacity(usize::from(u16::MAX) + 1);
        for packed_probability_index in 0..=u16::MAX {
            let homozygous_reference_probability = f32::from(
                u8::try_from(packed_probability_index & 0x00FF).expect("low packed probability byte should fit u8"),
            ) * reciprocal_scale;
            let heterozygous_probability = f32::from(
                u8::try_from((packed_probability_index & 0xFF00) >> 8)
                    .expect("high packed probability byte should fit u8"),
            ) * reciprocal_scale;
            dosage_lookup.push(2.0_f32 - ((2.0_f32 * homozygous_reference_probability) + heterozygous_probability));
        }
        dosage_lookup
    })
}

pub(super) fn exact_eight_bit_probability_pairs(packed_probability_bytes: &[u8]) -> &[[u8; 2]] {
    let (probability_pairs, []) = packed_probability_bytes.as_chunks::<2>() else {
        unreachable!("8-bit BGEN probability byte slices are built from two bytes per sample");
    };
    probability_pairs
}

pub(in crate::bgen) fn packed_eight_bit_probability_index(
    [homozygous_reference_probability_byte, heterozygous_probability_byte]: [u8; 2],
) -> usize {
    usize::from(homozygous_reference_probability_byte) | (usize::from(heterozygous_probability_byte) << 8)
}

pub(in crate::bgen) fn read_eight_bit_probability_pair(buffer: &[u8], offset: usize) -> Result<[u8; 2], BgenError> {
    let probability_bytes = read_exact_bytes(buffer, offset, 2)?;
    let ([probability_pair], []) = probability_bytes.as_chunks::<2>() else {
        unreachable!("selected 8-bit BGEN probability reads request exactly two bytes");
    };
    Ok(*probability_pair)
}

pub(in crate::bgen) struct ThreadScratch {
    pub(super) zlib_decompressor: NonNull<libdeflate_sys::libdeflate_decompressor>,
    pub(super) zstandard_decompressor: Option<zstd::bulk::Decompressor<'static>>,
    pub(super) decompressed_probability_block: Vec<u8>,
}

impl Default for ThreadScratch {
    fn default() -> Self {
        // SAFETY: libdeflate owns the returned allocation until the matching
        // free call in `Drop`.
        let zlib_decompressor = unsafe { libdeflate_sys::libdeflate_alloc_decompressor() };
        Self {
            zlib_decompressor: NonNull::new(zlib_decompressor)
                .expect("libdeflate could not allocate a zlib decompressor"),
            zstandard_decompressor: None,
            decompressed_probability_block: Vec::new(),
        }
    }
}

impl Drop for ThreadScratch {
    fn drop(&mut self) {
        // SAFETY: the pointer was allocated by libdeflate, remains live, and
        // this destructor runs exactly once.
        unsafe { libdeflate_sys::libdeflate_free_decompressor(self.zlib_decompressor.as_ptr()) };
    }
}

// SAFETY: ownership of the native decompressor moves with the scratch state,
// which is borrowed exclusively before invoking libdeflate.
unsafe impl Send for ThreadScratch {}
