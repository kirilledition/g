use super::BgenError;

mod matrix;
mod probability;
mod variant_major;

pub(super) use matrix::{
    ThreadScratch, VariantMajorSparseCandidateCountsMut, VariantMajorTileStatsMut, read_eight_bit_probability_pair,
    selected_sample_count_to_i32, with_worker_thread_scratch,
};
#[cfg(test)]
pub(super) use matrix::{packed_eight_bit_probability_index, unphased_eight_bit_dosage_lookup};
pub(super) use probability::{
    parse_layout_two_probability_block, read_exact_bytes, read_probability_block, read_u16_at, read_u32_at,
    u32_to_usize, validate_layout_two_probability_values,
};
pub(super) use variant_major::{decode_variant_major_dosage_tile, validate_variant_major_tile_stats_lengths};

pub(super) struct VariantDecodeFailure {
    pub(super) relative_variant_index: Option<usize>,
    pub(super) source: BgenError,
}

#[cfg(test)]
mod tests;
