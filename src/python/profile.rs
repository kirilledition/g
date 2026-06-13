use std::collections::HashMap;

use crate::genotype::bgen::ReaderProfileSnapshot;

pub(super) fn build_profile_snapshot_dict(profile_snapshot: &ReaderProfileSnapshot) -> HashMap<String, u64> {
    HashMap::from([
        ("sample_selection_prepare_ns".to_string(), profile_snapshot.sample_selection_prepare_ns),
        ("sample_selection_prepare_count".to_string(), profile_snapshot.sample_selection_prepare_count),
        ("compressed_block_fetch_ns".to_string(), profile_snapshot.compressed_block_fetch_ns),
        ("compressed_block_fetch_count".to_string(), profile_snapshot.compressed_block_fetch_count),
        ("compressed_byte_count".to_string(), profile_snapshot.compressed_byte_count),
        ("decompression_ns".to_string(), profile_snapshot.decompression_ns),
        ("decompression_count".to_string(), profile_snapshot.decompression_count),
        ("uncompressed_byte_count".to_string(), profile_snapshot.uncompressed_byte_count),
        ("zlib_stream_count".to_string(), profile_snapshot.zlib_stream_count),
        ("probability_decode_ns".to_string(), profile_snapshot.probability_decode_ns),
        ("probability_decode_count".to_string(), profile_snapshot.probability_decode_count),
        ("variant_decode_count".to_string(), profile_snapshot.variant_decode_count),
        ("output_write_ns".to_string(), profile_snapshot.output_write_ns),
        ("output_write_count".to_string(), profile_snapshot.output_write_count),
        ("output_byte_count".to_string(), profile_snapshot.output_byte_count),
        ("decode_tile_count".to_string(), profile_snapshot.decode_tile_count),
        ("selected_sample_count".to_string(), profile_snapshot.selected_sample_count),
        ("metadata_slice_ns".to_string(), profile_snapshot.metadata_slice_ns),
        ("metadata_slice_count".to_string(), profile_snapshot.metadata_slice_count),
    ])
}
