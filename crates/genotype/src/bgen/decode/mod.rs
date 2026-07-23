use super::metadata::VariantRecord;
use super::sample_selection::SampleSelection;
use super::{BgenError, CompressionType};

mod matrix;
mod probability;
mod variant_major;

pub(super) use matrix::{
    ThreadScratch, VariantMajorSparseCandidateCountsMut, VariantMajorTileStatsMut, read_eight_bit_probability_pair,
    selected_sample_count_to_i32, with_worker_thread_scratch,
};
pub(super) use probability::{
    parse_layout_two_probability_block, read_exact_bytes, read_probability_block, read_u32_at, u32_to_usize,
    validate_layout_two_probability_values,
};
pub(super) use variant_major::{decode_variant_major_dosage_tile, validate_variant_major_tile_stats_lengths};

#[derive(Clone, Copy)]
pub(super) struct VariantMajorTileDecodeRequest<'request> {
    pub(super) source_window: super::source::BgenByteWindow<'request>,
    pub(super) compression_type: CompressionType,
    pub(super) sample_count: usize,
    pub(super) sample_selection: &'request SampleSelection,
    pub(super) variant_records: &'request [VariantRecord],
    pub(super) tile_variant_start_index: usize,
}

pub(super) struct VariantDecodeFailure {
    pub(super) relative_variant_index: Option<usize>,
    pub(super) source: BgenError,
}

#[cfg(test)]
mod tests;
