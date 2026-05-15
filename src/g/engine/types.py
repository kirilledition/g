from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import numpy as np

from g.models import VariantMetadata

if typing.TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class Regenie2ChunkAccumulator:
    metadata: VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array
    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array | None


@dataclass(frozen=True)
class Regenie2ChunkPayload:
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
class Regenie2ChunkPayloadBatch:
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
