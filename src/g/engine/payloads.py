"""Host-side payload builders for REGENIE step 2 output persistence."""

from __future__ import annotations

import typing

import jax
import numpy as np
import numpy.typing as npt

from g.engine import types as engine_types

if typing.TYPE_CHECKING:
    import collections.abc


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


def build_chunk_write_payload_batch(
    chunk_accumulators: collections.abc.Sequence[engine_types.Regenie2ChunkAccumulator],
) -> engine_types.Regenie2ChunkPayloadBatch:
    """Build one flat host-side payload batch for Rust persistence."""
    if not chunk_accumulators:
        raise ValueError("Chunk payload batches require at least one accumulator.")
    host_value_lists = jax.device_get(
        {
            "allele_one_frequency": [
                chunk_accumulator.allele_one_frequency for chunk_accumulator in chunk_accumulators
            ],
            "observation_count": [chunk_accumulator.observation_count for chunk_accumulator in chunk_accumulators],
            "beta": [chunk_accumulator.beta for chunk_accumulator in chunk_accumulators],
            "standard_error": [chunk_accumulator.standard_error for chunk_accumulator in chunk_accumulators],
            "chi_squared": [chunk_accumulator.chi_squared for chunk_accumulator in chunk_accumulators],
            "log10_p_value": [chunk_accumulator.log10_p_value for chunk_accumulator in chunk_accumulators],
            "extra_code": [chunk_accumulator.extra_code for chunk_accumulator in chunk_accumulators],
        }
    )
    row_counts = np.asarray(
        [len(chunk_accumulator.metadata.position) for chunk_accumulator in chunk_accumulators],
        dtype=np.int64,
    )
    chunk_identifier = np.concatenate(
        [
            np.full(int(row_count), chunk_accumulator.metadata.variant_start_index, dtype=np.int64)
            for chunk_accumulator, row_count in zip(chunk_accumulators, row_counts, strict=True)
        ]
    )
    variant_start_index = np.concatenate(
        [
            np.full(int(row_count), chunk_accumulator.metadata.variant_start_index, dtype=np.int64)
            for chunk_accumulator, row_count in zip(chunk_accumulators, row_counts, strict=True)
        ]
    )
    variant_stop_index = np.concatenate(
        [
            np.full(int(row_count), chunk_accumulator.metadata.variant_stop_index, dtype=np.int64)
            for chunk_accumulator, row_count in zip(chunk_accumulators, row_counts, strict=True)
        ]
    )
    extra_code_value_list = typing.cast("list[npt.NDArray[np.int32] | None]", host_value_lists["extra_code"])
    extra_code: npt.NDArray[np.int32] | None
    if any(extra_code_value is None for extra_code_value in extra_code_value_list):
        extra_code = None
    else:
        extra_code = np.concatenate(typing.cast("list[npt.NDArray[np.int32]]", extra_code_value_list))
    return engine_types.Regenie2ChunkPayloadBatch(
        first_chunk_identifier=chunk_accumulators[0].metadata.variant_start_index,
        last_chunk_identifier=chunk_accumulators[-1].metadata.variant_start_index,
        chunk_identifier=chunk_identifier,
        variant_start_index=variant_start_index,
        variant_stop_index=variant_stop_index,
        chromosome=tuple(
            np.concatenate([chunk_accumulator.metadata.chromosome for chunk_accumulator in chunk_accumulators]).tolist()
        ),
        position=np.concatenate([chunk_accumulator.metadata.position for chunk_accumulator in chunk_accumulators]),
        variant_identifier=tuple(
            np.concatenate(
                [chunk_accumulator.metadata.variant_identifiers for chunk_accumulator in chunk_accumulators]
            ).tolist()
        ),
        allele_zero=tuple(
            np.concatenate([chunk_accumulator.metadata.allele_two for chunk_accumulator in chunk_accumulators]).tolist()
        ),
        allele_one=tuple(
            np.concatenate([chunk_accumulator.metadata.allele_one for chunk_accumulator in chunk_accumulators]).tolist()
        ),
        allele_one_frequency=np.concatenate(host_value_lists["allele_one_frequency"]),
        observation_count=np.concatenate(host_value_lists["observation_count"]),
        beta=np.concatenate(host_value_lists["beta"]),
        standard_error=np.concatenate(host_value_lists["standard_error"]),
        chi_squared=np.concatenate(host_value_lists["chi_squared"]),
        log10_p_value=np.concatenate(host_value_lists["log10_p_value"]),
        extra_code=extra_code,
    )
