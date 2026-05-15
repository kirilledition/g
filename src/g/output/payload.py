"""Output payload dataclasses and batch construction."""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from g.output import schema

import jax
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Regenie2ChunkPayload:
    """Host-side REGENIE step 2 chunk payload ready for Rust persistence."""

    chunk_identifier: int
    variant_start_index: int
    variant_stop_index: int
    chromosome: npt.NDArray[np.str_]
    position: npt.NDArray[np.int64]
    variant_identifier: npt.NDArray[np.str_]
    allele_zero: npt.NDArray[np.str_]
    allele_one: npt.NDArray[np.str_]
    allele_one_frequency: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    beta: npt.NDArray[np.float32]
    standard_error: npt.NDArray[np.float32]
    chi_squared: npt.NDArray[np.float32]
    log10_p_value: npt.NDArray[np.float32]
    extra_code: npt.NDArray[np.int32] | None


@dataclass(frozen=True)
class AssociationOutputBatch:
    """Flat host-side REGENIE step 2 payload batch ready for Rust persistence."""

    first_chunk_identifier: int
    last_chunk_identifier: int
    chunk_identifier: npt.NDArray[np.int64]
    variant_start_index: npt.NDArray[np.int64]
    variant_stop_index: npt.NDArray[np.int64]
    chromosome: tuple[str, ...]
    position: npt.NDArray[np.int64]
    variant_identifier: tuple[str, ...]
    allele_zero: tuple[str, ...]
    allele_one: tuple[str, ...]
    allele_one_frequency: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    beta: npt.NDArray[np.float32]
    standard_error: npt.NDArray[np.float32]
    chi_squared: npt.NDArray[np.float32]
    log10_p_value: npt.NDArray[np.float32]
    extra_code: npt.NDArray[np.int32] | None


def build_output_batch(
    chunk_results: typing.Sequence[schema.AssociationChunkResult],
) -> AssociationOutputBatch:
    """Build one flat host-side payload batch for Rust persistence."""
    if not chunk_results:
        message = "Chunk payload batches require at least one accumulator."
        raise ValueError(message)
    host_value_lists = jax.device_get(
        {
            "allele_one_frequency": [chunk_result.allele_one_frequency for chunk_result in chunk_results],
            "observation_count": [chunk_result.observation_count for chunk_result in chunk_results],
            "beta": [chunk_result.beta for chunk_result in chunk_results],
            "standard_error": [chunk_result.standard_error for chunk_result in chunk_results],
            "chi_squared": [chunk_result.chi_squared for chunk_result in chunk_results],
            "log10_p_value": [chunk_result.log10_p_value for chunk_result in chunk_results],
            "extra_code": [chunk_result.extra_code for chunk_result in chunk_results],
        }
    )
    row_counts = np.asarray([len(chunk_result.metadata.position) for chunk_result in chunk_results], dtype=np.int64)
    chunk_identifier = np.concatenate(
        [
            np.full(int(row_count), chunk_result.metadata.variant_start_index, dtype=np.int64)
            for chunk_result, row_count in zip(chunk_results, row_counts, strict=True)
        ]
    )
    variant_start_index = np.concatenate(
        [
            np.full(int(row_count), chunk_result.metadata.variant_start_index, dtype=np.int64)
            for chunk_result, row_count in zip(chunk_results, row_counts, strict=True)
        ]
    )
    variant_stop_index = np.concatenate(
        [
            np.full(int(row_count), chunk_result.metadata.variant_stop_index, dtype=np.int64)
            for chunk_result, row_count in zip(chunk_results, row_counts, strict=True)
        ]
    )
    extra_code_value_list = typing.cast("list[npt.NDArray[np.int32] | None]", host_value_lists["extra_code"])
    if any(extra_code_value is None for extra_code_value in extra_code_value_list):
        extra_code: npt.NDArray[np.int32] | None = None
    else:
        extra_code = np.concatenate(typing.cast("list[npt.NDArray[np.int32]]", extra_code_value_list))
    return AssociationOutputBatch(
        first_chunk_identifier=chunk_results[0].metadata.variant_start_index,
        last_chunk_identifier=chunk_results[-1].metadata.variant_start_index,
        chunk_identifier=chunk_identifier,
        variant_start_index=variant_start_index,
        variant_stop_index=variant_stop_index,
        chromosome=tuple(np.concatenate([chunk_result.metadata.chromosome for chunk_result in chunk_results]).tolist()),
        position=np.concatenate([chunk_result.metadata.position for chunk_result in chunk_results]),
        variant_identifier=tuple(
            np.concatenate([chunk_result.metadata.variant_identifiers for chunk_result in chunk_results]).tolist()
        ),
        allele_zero=tuple(
            np.concatenate([chunk_result.metadata.allele_two for chunk_result in chunk_results]).tolist()
        ),
        allele_one=tuple(
            np.concatenate([chunk_result.metadata.allele_one for chunk_result in chunk_results]).tolist()
        ),
        allele_one_frequency=np.concatenate(host_value_lists["allele_one_frequency"]),
        observation_count=np.concatenate(host_value_lists["observation_count"]),
        beta=np.concatenate(host_value_lists["beta"]),
        standard_error=np.concatenate(host_value_lists["standard_error"]),
        chi_squared=np.concatenate(host_value_lists["chi_squared"]),
        log10_p_value=np.concatenate(host_value_lists["log10_p_value"]),
        extra_code=extra_code,
    )
