"""Genotype transfer and chunk metadata helpers for callback workers."""

from __future__ import annotations

from g.engine.callbacks import _legacy

cast_statistic_array_for_native_writer = _legacy.cast_statistic_array_for_native_writer
narrow_public_statistic_array_on_device = _legacy.narrow_public_statistic_array_on_device
select_active_trait_rows_on_device = _legacy.select_active_trait_rows_on_device
get_chunk_stats_compute_arrays = _legacy.get_chunk_stats_compute_arrays
get_linear_chunk_stats_arrays = _legacy.get_linear_chunk_stats_arrays
get_binary_chunk_stats_arrays = _legacy.get_binary_chunk_stats_arrays
put_compute_array_on_device = _legacy.put_compute_array_on_device
put_genotype_matrix_on_device = _legacy.put_genotype_matrix_on_device
put_chunk_array_on_device = _legacy.put_chunk_array_on_device
block_compute_result_for_timing = _legacy.block_compute_result_for_timing
build_chunk_timing_identity = _legacy.build_chunk_timing_identity
record_stage_duration_with_optional_chunk = _legacy.record_stage_duration_with_optional_chunk
record_transfer_metadata_for_array = _legacy.record_transfer_metadata_for_array
build_projected_variant_major_dosage_chunk_stats = _legacy.build_projected_variant_major_dosage_chunk_stats

__all__ = [
    "cast_statistic_array_for_native_writer",
    "narrow_public_statistic_array_on_device",
    "select_active_trait_rows_on_device",
    "get_chunk_stats_compute_arrays",
    "get_linear_chunk_stats_arrays",
    "get_binary_chunk_stats_arrays",
    "put_compute_array_on_device",
    "put_genotype_matrix_on_device",
    "put_chunk_array_on_device",
    "block_compute_result_for_timing",
    "build_chunk_timing_identity",
    "record_stage_duration_with_optional_chunk",
    "record_transfer_metadata_for_array",
    "build_projected_variant_major_dosage_chunk_stats",
]
