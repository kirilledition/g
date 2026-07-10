mod config;
mod matrix;
mod probability;
mod row_major;
mod variant_major;

pub(super) use config::decode_tile_variant_count;
pub use config::set_decode_tile_variant_count;
pub(super) use matrix::{
    DosageTileDecodeResult, ThreadScratch, VariantDecodeResult, VariantMajorOutputMatrix, VariantMajorTileDecodeResult,
    VariantMajorTileStatsMut, packed_eight_bit_probability_index, read_eight_bit_probability_pair,
    unphased_eight_bit_dosage_lookup,
};
pub(super) use probability::{
    read_exact_bytes, read_probability_block, read_u8_at, read_u16_at, read_u32_at, u32_to_usize,
};
pub(super) use row_major::decode_variant_dosage_tile_into_row_major_matrix;
pub(super) use variant_major::{decode_variant_major_dosage_tile, validate_variant_major_tile_stats_lengths};

#[cfg(test)]
mod tests;
