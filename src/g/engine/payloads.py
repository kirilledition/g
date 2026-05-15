"""Host-side payload builders for REGENIE step 2 output persistence."""

from __future__ import annotations

import jax

from g.engine import types as engine_types


def build_chunk_payload(chunk_accumulator: engine_types.Regenie2ChunkAccumulator) -> engine_types.Regenie2ChunkPayload:
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
