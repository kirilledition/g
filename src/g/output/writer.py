"""Output writer payload helpers."""

from __future__ import annotations

import numpy as np

from g.output import payload


def build_output_batch_from_chunk_payload(
    chunk_payload: payload.Regenie2ChunkPayload,
) -> payload.AssociationOutputBatch:
    """Build a one-chunk output batch from a single chunk payload."""
    return payload.AssociationOutputBatch(
        first_chunk_identifier=chunk_payload.chunk_identifier,
        last_chunk_identifier=chunk_payload.chunk_identifier,
        chunk_identifier=np.full(len(chunk_payload.position), chunk_payload.chunk_identifier, dtype=np.int64),
        variant_start_index=np.full(len(chunk_payload.position), chunk_payload.variant_start_index, dtype=np.int64),
        variant_stop_index=np.full(len(chunk_payload.position), chunk_payload.variant_stop_index, dtype=np.int64),
        chromosome=tuple(chunk_payload.chromosome.tolist()),
        position=chunk_payload.position,
        variant_identifier=tuple(chunk_payload.variant_identifier.tolist()),
        allele_zero=tuple(chunk_payload.allele_zero.tolist()),
        allele_one=tuple(chunk_payload.allele_one.tolist()),
        allele_one_frequency=chunk_payload.allele_one_frequency,
        observation_count=chunk_payload.observation_count,
        beta=chunk_payload.beta,
        standard_error=chunk_payload.standard_error,
        chi_squared=chunk_payload.chi_squared,
        log10_p_value=chunk_payload.log10_p_value,
        extra_code=chunk_payload.extra_code,
    )
