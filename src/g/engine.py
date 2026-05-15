from g.engine.types import Regenie2ChunkAccumulator, Regenie2ChunkPayload, Regenie2ChunkPayloadBatch
"""Compatibility exports for REGENIE step 2 orchestration and payload helpers."""

from __future__ import annotations

import typing
import jax

from g.engine import chromosome_chunks, dispatch, types as engine_types

if typing.TYPE_CHECKING:
    import collections.abc
ChunkAccumulator = engine_types.Regenie2ChunkAccumulator
ChunkPayload = engine_types.Regenie2ChunkPayload
ChunkWritePayload = engine_types.Regenie2ChunkPayloadBatch

split_dosage_genotype_chunk_by_chromosome = chromosome_chunks.split_dosage_genotype_chunk_by_chromosome
split_dosage_genotype_chunk_by_absolute_variant_slices = (
    chromosome_chunks.split_dosage_genotype_chunk_by_absolute_variant_slices
)
split_dosage_genotype_chunk_with_reader_metadata = chromosome_chunks.split_dosage_genotype_chunk_with_reader_metadata
iter_regenie2_linear_output_frames = dispatch.iter_regenie2_linear_output_frames
iter_regenie2_binary_output_frames = dispatch.iter_regenie2_binary_output_frames


def build_chunk_payload(chunk_accumulator: ChunkAccumulator) -> ChunkPayload:
    """Build one host-side REGENIE step 2 payload from a device-resident accumulator."""
    host_values = jax.device_get(
        {
            "allele_one_frequency": chunk_accumulator.allele_one_frequency,
            "observation_count": chunk_accumulator.observation_count,
            "beta": chunk_accumulator.beta,
            "standard_error": chunk_accumulator.standard_error,
            "chi_squared": chunk_accumulator.chi_squared,
            "log10_p_value": chunk_accumulator.log10_p_value,
            "extra_code": chunk_accumulator.extra_code,
        }
    )
    return engine_types.Regenie2ChunkPayload(
        chunk_identifier=chunk_accumulator.metadata.variant_start_index,
        variant_start_index=chunk_accumulator.metadata.variant_start_index,
        variant_stop_index=chunk_accumulator.metadata.variant_stop_index,
        chromosome=chunk_accumulator.metadata.chromosome,
        position=chunk_accumulator.metadata.position,
        variant_identifier=chunk_accumulator.metadata.variant_identifiers,
        allele_zero=chunk_accumulator.metadata.allele_two,
        allele_one=chunk_accumulator.metadata.allele_one,
        allele_one_frequency=host_values["allele_one_frequency"],
        observation_count=host_values["observation_count"],
        beta=host_values["beta"],
        standard_error=host_values["standard_error"],
        chi_squared=host_values["chi_squared"],
        log10_p_value=host_values["log10_p_value"],
        extra_code=host_values["extra_code"],
    )


def build_chunk_write_payload_batch(
    chunk_accumulators: collections.abc.Sequence[ChunkAccumulator],
) -> ChunkWritePayload:
    from g.io import output

    return output.build_chunk_write_payload_batch(chunk_accumulators)
