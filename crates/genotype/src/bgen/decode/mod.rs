use std::sync::atomic::{AtomicUsize, Ordering};

use super::BgenError;

mod matrix;
mod probability;
#[cfg(test)]
mod row_major;
mod variant_major;

#[cfg(test)]
pub(super) use matrix::DosageTileDecodeResult;
pub(super) use matrix::{
    ThreadScratch, VariantMajorOutputMatrix, VariantMajorSparseCandidateCountsMut, VariantMajorTileStatsMut,
    packed_eight_bit_probability_index, read_eight_bit_probability_pair, selected_sample_count_to_i32,
    unphased_eight_bit_dosage_lookup,
};
pub(super) use probability::{
    read_exact_bytes, read_probability_block, read_u8_at, read_u16_at, read_u32_at, u32_to_usize,
};
#[cfg(test)]
pub(super) use row_major::decode_variant_dosage_tile_into_row_major_matrix;
pub(super) use variant_major::{decode_variant_major_dosage_tile, validate_variant_major_tile_stats_lengths};

pub(super) struct VariantDecodeFailure {
    pub(super) relative_variant_index: Option<usize>,
    pub(super) source: BgenError,
}

const DEFAULT_DECODE_TILE_VARIANT_COUNT: usize = 64;
static DECODE_TILE_VARIANT_COUNT: AtomicUsize = AtomicUsize::new(DEFAULT_DECODE_TILE_VARIANT_COUNT);

pub(super) fn decode_tile_variant_count() -> usize {
    DECODE_TILE_VARIANT_COUNT.load(Ordering::Relaxed)
}

#[allow(clippy::missing_errors_doc)]
pub fn set_decode_tile_variant_count(tile_variant_count: usize) -> Result<(), BgenError> {
    if tile_variant_count == 0 {
        return Err(BgenError::Range("BGEN decode tile variant count must be positive.".to_string()));
    }
    DECODE_TILE_VARIANT_COUNT.store(tile_variant_count, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
mod tests;
