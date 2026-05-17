"""Structured engine payload types for REGENIE step 2 orchestration."""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import jax
    import numpy as np
    import numpy.typing as npt


@dataclass(frozen=True)
class Regenie2ChunkAccumulator:
    """Device-resident association results for one variant chunk.

    Attributes:
        metadata: Variant metadata for the accumulated chunk.
        allele_one_frequency: Counted allele frequency per variant.
        observation_count: Non-missing observation count per variant.
        beta: Coefficient estimate per variant.
        standard_error: Standard error per variant.
        chi_squared: Chi-squared statistic per variant.
        log10_p_value: Negative base-10 p-value per variant.
        extra_code: Optional method/status code per variant.

    """

    metadata: typing.Any
    allele_one_frequency: jax.Array | npt.NDArray[np.float32]
    observation_count: jax.Array | npt.NDArray[np.int32]
    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array | None


@dataclass(frozen=True)
class Regenie2ChunkPayload:
    """Host-side output payload for one variant chunk.

    Attributes:
        chunk_identifier: Stable chunk identifier.
        variant_start_index: Inclusive absolute variant start index.
        variant_stop_index: Exclusive absolute variant stop index.
        chromosome: Chromosome identifier per variant.
        position: Genomic position per variant.
        variant_identifier: Variant identifier per variant.
        allele_zero: Reference allele per variant.
        allele_one: Counted allele per variant.
        allele_one_frequency: Counted allele frequency per variant.
        observation_count: Non-missing observation count per variant.
        beta: Coefficient estimate per variant.
        standard_error: Standard error per variant.
        chi_squared: Chi-squared statistic per variant.
        log10_p_value: Negative base-10 p-value per variant.
        extra_code: Optional method/status code per variant.

    """

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
