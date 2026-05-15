"""High-level orchestration and chunk helpers for REGENIE step 2 association runs."""

from g.engine.chromosome_chunks import (
    split_dosage_genotype_chunk_by_absolute_variant_slices,
    split_dosage_genotype_chunk_by_chromosome,
    split_dosage_genotype_chunk_with_reader_metadata,
)
from g.engine.dispatch import iter_regenie2_binary_output_frames, iter_regenie2_linear_output_frames
from g.engine.payloads import build_chunk_payload, build_chunk_write_payload_batch
from g.engine.profiling import profiled_regenie2_binary_chunk_step, profiled_regenie2_linear_chunk_step
from g.engine.types import Regenie2ChunkAccumulator, Regenie2ChunkPayload, Regenie2ChunkPayloadBatch

ChunkAccumulator = Regenie2ChunkAccumulator
ChunkPayload = Regenie2ChunkPayload
ChunkWritePayload = Regenie2ChunkPayloadBatch

__all__ = [
    "ChunkAccumulator",
    "ChunkPayload",
    "ChunkWritePayload",
    "Regenie2ChunkAccumulator",
    "Regenie2ChunkPayload",
    "Regenie2ChunkPayloadBatch",
    "build_chunk_payload",
    "build_chunk_write_payload_batch",
    "iter_regenie2_binary_output_frames",
    "iter_regenie2_linear_output_frames",
    "profiled_regenie2_binary_chunk_step",
    "profiled_regenie2_linear_chunk_step",
    "split_dosage_genotype_chunk_by_absolute_variant_slices",
    "split_dosage_genotype_chunk_by_chromosome",
    "split_dosage_genotype_chunk_with_reader_metadata",
]
