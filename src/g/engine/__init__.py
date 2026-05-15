
"""High-level orchestration and chunk helpers for REGENIE step 2 association runs."""

from g.engine.dispatch import iter_regenie2_binary_output_frames, iter_regenie2_linear_output_frames
from g.engine.profiling import profiled_regenie2_binary_chunk_step, profiled_regenie2_linear_chunk_step
from g.engine.chromosome_chunks import (
    split_dosage_genotype_chunk_by_absolute_variant_slices,
    split_dosage_genotype_chunk_by_chromosome,
    split_dosage_genotype_chunk_with_reader_metadata,
)
from g.models import Regenie2ChunkAccumulator

__all__ = [
    "Regenie2ChunkAccumulator",
    "iter_regenie2_linear_output_frames",
    "iter_regenie2_binary_output_frames",
    "split_dosage_genotype_chunk_by_chromosome",
    "split_dosage_genotype_chunk_by_absolute_variant_slices",
    "split_dosage_genotype_chunk_with_reader_metadata",
    "profiled_regenie2_linear_chunk_step",
    "profiled_regenie2_binary_chunk_step",
]
