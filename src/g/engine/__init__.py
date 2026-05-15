"""High-level orchestration and chunk helpers for REGENIE step 2 association runs."""

from g.engine.chromosome_chunks import (
    split_dosage_genotype_chunk_by_absolute_variant_slices,
    split_dosage_genotype_chunk_by_chromosome,
    split_dosage_genotype_chunk_with_reader_metadata,
)
from g.engine.dispatch import iter_regenie2_binary_output_frames, iter_regenie2_linear_output_frames
from g.engine.payloads import build_chunk_payload
from g.engine.profiling import profiled_regenie2_binary_chunk_step, profiled_regenie2_linear_chunk_step
from g.engine.rust_pipeline import (
    BinaryRegenie2RustPipelineCallback,
    LinearRegenie2RustPipelineCallback,
    run_regenie2_binary_bgen_rust_pipeline,
    run_regenie2_linear_bgen_rust_pipeline,
)
from g.engine.types import Regenie2ChunkAccumulator, Regenie2ChunkPayload

ChunkAccumulator = Regenie2ChunkAccumulator
ChunkPayload = Regenie2ChunkPayload

__all__ = [
    "BinaryRegenie2RustPipelineCallback",
    "ChunkAccumulator",
    "ChunkPayload",
    "LinearRegenie2RustPipelineCallback",
    "Regenie2ChunkAccumulator",
    "Regenie2ChunkPayload",
    "build_chunk_payload",
    "iter_regenie2_binary_output_frames",
    "iter_regenie2_linear_output_frames",
    "profiled_regenie2_binary_chunk_step",
    "profiled_regenie2_linear_chunk_step",
    "run_regenie2_binary_bgen_rust_pipeline",
    "run_regenie2_linear_bgen_rust_pipeline",
    "split_dosage_genotype_chunk_by_absolute_variant_slices",
    "split_dosage_genotype_chunk_by_chromosome",
    "split_dosage_genotype_chunk_with_reader_metadata",
]
