"""High-level orchestration for Phase 1 association runs."""

from __future__ import annotations

import contextlib
import itertools
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
import polars as pl

from g import models, types
from g.compute import linear, logistic, regenie2_linear
from g.io import regenie, source

if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path

    import numpy.typing as npt


@dataclass(frozen=True)
class LinearChunkAccumulator:
    """Accumulator for linear regression chunk results (JAX arrays, device memory).

    Attributes:
        metadata: Variant metadata for the chunk.
        allele_one_frequency: Allele frequencies per variant.
        observation_count: Observation counts per variant.
        linear_result: Linear regression results.

    """

    metadata: models.VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array
    linear_result: models.LinearAssociationChunkResult


@dataclass(frozen=True)
class LogisticChunkAccumulator:
    """Accumulator for logistic regression chunk results (JAX arrays, device memory).

    Attributes:
        metadata: Variant metadata for the chunk.
        allele_one_frequency: Allele frequencies per variant.
        observation_count: Observation counts per variant.
        logistic_result: Logistic regression results.

    """

    metadata: models.VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array
    logistic_result: models.LogisticAssociationChunkResult


@dataclass(frozen=True)
class Regenie2LinearChunkAccumulator:
    """Accumulator for REGENIE step 2 linear chunk results (JAX arrays, device memory).

    Attributes:
        metadata: Variant metadata for the chunk.
        allele_one_frequency: Allele frequencies per variant.
        observation_count: Observation counts per variant.
        regenie2_linear_result: REGENIE step 2 linear association results.

    """

    metadata: models.VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array
    regenie2_linear_result: models.Regenie2LinearChunkResult


@dataclass(frozen=True)
class LinearChunkPayload:
    """Host-side linear association payload ready for persistence."""

    chunk_identifier: int
    variant_start_index: int
    variant_stop_index: int
    chromosome: npt.NDArray[np.str_]
    position: npt.NDArray[np.int64]
    variant_identifier: npt.NDArray[np.str_]
    allele_one: npt.NDArray[np.str_]
    allele_two: npt.NDArray[np.str_]
    allele_one_frequency: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    beta: npt.NDArray[np.float32]
    standard_error: npt.NDArray[np.float32]
    t_statistic: npt.NDArray[np.float32]
    p_value: npt.NDArray[np.float32]
    is_valid: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class LogisticChunkPayload:
    """Host-side logistic association payload ready for persistence."""

    chunk_identifier: int
    variant_start_index: int
    variant_stop_index: int
    chromosome: npt.NDArray[np.str_]
    position: npt.NDArray[np.int64]
    variant_identifier: npt.NDArray[np.str_]
    allele_one: npt.NDArray[np.str_]
    allele_two: npt.NDArray[np.str_]
    allele_one_frequency: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    beta: npt.NDArray[np.float32]
    standard_error: npt.NDArray[np.float32]
    z_statistic: npt.NDArray[np.float32]
    p_value: npt.NDArray[np.float32]
    firth_flag: npt.NDArray[np.str_]
    error_code: npt.NDArray[np.str_]
    converged: npt.NDArray[np.bool_]
    iteration_count: npt.NDArray[np.int32]
    is_valid: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Regenie2LinearChunkPayload:
    """Host-side REGENIE step 2 linear association payload ready for persistence."""

    chunk_identifier: int
    variant_start_index: int
    variant_stop_index: int
    chromosome: npt.NDArray[np.str_]
    position: npt.NDArray[np.int64]
    variant_identifier: npt.NDArray[np.str_]
    allele_one: npt.NDArray[np.str_]
    allele_two: npt.NDArray[np.str_]
    allele_one_frequency: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    beta: npt.NDArray[np.float32]
    standard_error: npt.NDArray[np.float32]
    chi_squared: npt.NDArray[np.float32]
    log10_p_value: npt.NDArray[np.float32]
    is_valid: npt.NDArray[np.bool_]


ChunkPayload = LinearChunkPayload | LogisticChunkPayload | Regenie2LinearChunkPayload


open_genotype_reader = source.open_genotype_reader
load_aligned_sample_data_from_source = source.load_aligned_sample_data_from_source
iter_genotype_chunks_from_source = source.iter_genotype_chunks_from_source
iter_linear_genotype_chunks_from_source = source.iter_linear_genotype_chunks_from_source
build_plink_source_config = source.build_plink_source_config
prepare_linear_association_state = linear.prepare_linear_association_state
compute_linear_association_chunk = linear.compute_linear_association_chunk
prepare_no_missing_logistic_constants = logistic.prepare_no_missing_logistic_constants
load_prediction_source = regenie.load_prediction_source
prepare_regenie2_linear_state = regenie2_linear.prepare_regenie2_linear_state
compute_regenie2_linear_chunk = regenie2_linear.compute_regenie2_linear_chunk


def split_linear_genotype_chunk_by_chromosome(
    genotype_chunk: models.LinearGenotypeChunk,
) -> tuple[models.LinearGenotypeChunk, ...]:
    """Split a linear genotype chunk into chromosome-homogeneous subchunks."""
    chromosome_values = genotype_chunk.metadata.chromosome
    if chromosome_values.size == 0:
        return (genotype_chunk,)
    if np.all(chromosome_values == chromosome_values[0]):
        return (genotype_chunk,)

    chromosome_start_indices = [0]
    for variant_index in range(1, len(chromosome_values)):
        if chromosome_values[variant_index] != chromosome_values[variant_index - 1]:
            chromosome_start_indices.append(variant_index)
    chromosome_start_indices.append(len(chromosome_values))

    chromosome_subchunks: list[models.LinearGenotypeChunk] = []
    for start_index, stop_index in itertools.pairwise(chromosome_start_indices):
        chromosome_subchunks.append(
            models.LinearGenotypeChunk(
                genotypes=genotype_chunk.genotypes[:, start_index:stop_index],
                metadata=models.VariantMetadata(
                    variant_start_index=genotype_chunk.metadata.variant_start_index + start_index,
                    variant_stop_index=genotype_chunk.metadata.variant_start_index + stop_index,
                    chromosome=genotype_chunk.metadata.chromosome[start_index:stop_index],
                    variant_identifiers=genotype_chunk.metadata.variant_identifiers[start_index:stop_index],
                    position=genotype_chunk.metadata.position[start_index:stop_index],
                    allele_one=genotype_chunk.metadata.allele_one[start_index:stop_index],
                    allele_two=genotype_chunk.metadata.allele_two[start_index:stop_index],
                ),
                allele_one_frequency=genotype_chunk.allele_one_frequency[start_index:stop_index],
                observation_count=genotype_chunk.observation_count[start_index:stop_index],
            )
        )
    return tuple(chromosome_subchunks)


def format_logistic_method_codes(method_code_values: np.ndarray) -> np.ndarray:
    """Convert logistic method codes to PLINK-style FIRTH flags."""
    return np.where(method_code_values == logistic.LogisticMethod.FIRTH, "Y", "N")


def format_logistic_error_codes(error_code_values: np.ndarray) -> np.ndarray:
    """Convert logistic error codes to PLINK-style error labels."""
    return np.where(
        error_code_values == logistic.LogisticErrorCode.FIRTH_CONVERGE_FAIL,
        "FIRTH_CONVERGE_FAIL",
        np.where(
            error_code_values == logistic.LogisticErrorCode.LOGISTIC_CONVERGE_FAIL,
            "LOGISTIC_CONVERGE_FAIL",
            np.where(
                error_code_values == logistic.LogisticErrorCode.UNFINISHED,
                "UNFINISHED",
                ".",
            ),
        ),
    )


def build_linear_output_frame(
    metadata: models.VariantMetadata,
    allele_one_frequency: jax.Array,
    observation_count: jax.Array,
    linear_result: models.LinearAssociationChunkResult,
) -> pl.DataFrame:
    """Build a tabular linear association result frame."""
    host_values = jax.device_get(
        {
            "allele_one_frequency": allele_one_frequency,
            "observation_count": observation_count,
            "beta": linear_result.beta,
            "standard_error": linear_result.standard_error,
            "test_statistic": linear_result.test_statistic,
            "p_value": linear_result.p_value,
            "valid_mask": linear_result.valid_mask,
        }
    )
    return pl.DataFrame(
        {
            "chromosome": metadata.chromosome,
            "position": metadata.position,
            "variant_identifier": metadata.variant_identifiers,
            "allele_one": metadata.allele_one,
            "allele_two": metadata.allele_two,
            "allele_one_frequency": host_values["allele_one_frequency"],
            "observation_count": host_values["observation_count"],
            "beta": host_values["beta"],
            "standard_error": host_values["standard_error"],
            "t_statistic": host_values["test_statistic"],
            "p_value": host_values["p_value"],
            "is_valid": host_values["valid_mask"],
        }
    )


def build_linear_chunk_payload(
    metadata: models.VariantMetadata,
    allele_one_frequency: jax.Array,
    observation_count: jax.Array,
    linear_result: models.LinearAssociationChunkResult,
) -> LinearChunkPayload:
    """Build a host-side linear payload for background persistence."""
    host_values = jax.device_get(
        {
            "allele_one_frequency": allele_one_frequency,
            "observation_count": observation_count,
            "beta": linear_result.beta,
            "standard_error": linear_result.standard_error,
            "test_statistic": linear_result.test_statistic,
            "p_value": linear_result.p_value,
            "valid_mask": linear_result.valid_mask,
        }
    )
    return LinearChunkPayload(
        chunk_identifier=metadata.variant_start_index,
        variant_start_index=metadata.variant_start_index,
        variant_stop_index=metadata.variant_stop_index,
        chromosome=metadata.chromosome,
        position=metadata.position,
        variant_identifier=metadata.variant_identifiers,
        allele_one=metadata.allele_one,
        allele_two=metadata.allele_two,
        allele_one_frequency=host_values["allele_one_frequency"],
        observation_count=host_values["observation_count"],
        beta=host_values["beta"],
        standard_error=host_values["standard_error"],
        t_statistic=host_values["test_statistic"],
        p_value=host_values["p_value"],
        is_valid=host_values["valid_mask"],
    )


def build_logistic_output_frame(
    metadata: models.VariantMetadata,
    allele_one_frequency: jax.Array,
    observation_count: jax.Array,
    logistic_result: models.LogisticAssociationChunkResult,
) -> pl.DataFrame:
    """Build a tabular logistic association result frame."""
    host_values = jax.device_get(
        {
            "allele_one_frequency": allele_one_frequency,
            "observation_count": observation_count,
            "beta": logistic_result.beta,
            "standard_error": logistic_result.standard_error,
            "test_statistic": logistic_result.test_statistic,
            "p_value": logistic_result.p_value,
            "method_code": logistic_result.method_code,
            "error_code": logistic_result.error_code,
            "converged_mask": logistic_result.converged_mask,
            "iteration_count": logistic_result.iteration_count,
            "valid_mask": logistic_result.valid_mask,
        }
    )
    return pl.DataFrame(
        {
            "chromosome": metadata.chromosome,
            "position": metadata.position,
            "variant_identifier": metadata.variant_identifiers,
            "allele_one": metadata.allele_one,
            "allele_two": metadata.allele_two,
            "allele_one_frequency": host_values["allele_one_frequency"],
            "observation_count": host_values["observation_count"],
            "beta": host_values["beta"],
            "standard_error": host_values["standard_error"],
            "z_statistic": host_values["test_statistic"],
            "p_value": host_values["p_value"],
            "firth_flag": format_logistic_method_codes(host_values["method_code"]),
            "error_code": format_logistic_error_codes(host_values["error_code"]),
            "converged": host_values["converged_mask"],
            "iteration_count": host_values["iteration_count"],
            "is_valid": host_values["valid_mask"],
        }
    )


def build_logistic_chunk_payload(
    metadata: models.VariantMetadata,
    allele_one_frequency: jax.Array,
    observation_count: jax.Array,
    logistic_result: models.LogisticAssociationChunkResult,
) -> LogisticChunkPayload:
    """Build a host-side logistic payload for background persistence."""
    host_values = jax.device_get(
        {
            "allele_one_frequency": allele_one_frequency,
            "observation_count": observation_count,
            "beta": logistic_result.beta,
            "standard_error": logistic_result.standard_error,
            "test_statistic": logistic_result.test_statistic,
            "p_value": logistic_result.p_value,
            "method_code": logistic_result.method_code,
            "error_code": logistic_result.error_code,
            "converged_mask": logistic_result.converged_mask,
            "iteration_count": logistic_result.iteration_count,
            "valid_mask": logistic_result.valid_mask,
        }
    )
    return LogisticChunkPayload(
        chunk_identifier=metadata.variant_start_index,
        variant_start_index=metadata.variant_start_index,
        variant_stop_index=metadata.variant_stop_index,
        chromosome=metadata.chromosome,
        position=metadata.position,
        variant_identifier=metadata.variant_identifiers,
        allele_one=metadata.allele_one,
        allele_two=metadata.allele_two,
        allele_one_frequency=host_values["allele_one_frequency"],
        observation_count=host_values["observation_count"],
        beta=host_values["beta"],
        standard_error=host_values["standard_error"],
        z_statistic=host_values["test_statistic"],
        p_value=host_values["p_value"],
        firth_flag=format_logistic_method_codes(host_values["method_code"]),
        error_code=format_logistic_error_codes(host_values["error_code"]),
        converged=host_values["converged_mask"],
        iteration_count=host_values["iteration_count"],
        is_valid=host_values["valid_mask"],
    )


def build_regenie2_linear_chunk_payload(
    metadata: models.VariantMetadata,
    allele_one_frequency: jax.Array,
    observation_count: jax.Array,
    regenie2_linear_result: models.Regenie2LinearChunkResult,
) -> Regenie2LinearChunkPayload:
    """Build a host-side REGENIE step 2 linear payload for background persistence."""
    host_values = jax.device_get(
        {
            "allele_one_frequency": allele_one_frequency,
            "observation_count": observation_count,
            "beta": regenie2_linear_result.beta,
            "standard_error": regenie2_linear_result.standard_error,
            "chi_squared": regenie2_linear_result.chi_squared,
            "log10_p_value": regenie2_linear_result.log10_p_value,
            "valid_mask": regenie2_linear_result.valid_mask,
        }
    )
    return Regenie2LinearChunkPayload(
        chunk_identifier=metadata.variant_start_index,
        variant_start_index=metadata.variant_start_index,
        variant_stop_index=metadata.variant_stop_index,
        chromosome=metadata.chromosome,
        position=metadata.position,
        variant_identifier=metadata.variant_identifiers,
        allele_one=metadata.allele_one,
        allele_two=metadata.allele_two,
        allele_one_frequency=host_values["allele_one_frequency"],
        observation_count=host_values["observation_count"],
        beta=host_values["beta"],
        standard_error=host_values["standard_error"],
        chi_squared=host_values["chi_squared"],
        log10_p_value=host_values["log10_p_value"],
        is_valid=host_values["valid_mask"],
    )


def build_regenie2_linear_chunk_payload_batch(
    chunk_accumulators: collections.abc.Sequence[Regenie2LinearChunkAccumulator],
) -> list[Regenie2LinearChunkPayload]:
    """Build host-side REGENIE step 2 payloads with one device transfer for the batch."""
    if not chunk_accumulators:
        return []

    host_lists = jax.device_get(
        {
            "allele_one_frequency": [
                chunk_accumulator.allele_one_frequency for chunk_accumulator in chunk_accumulators
            ],
            "observation_count": [chunk_accumulator.observation_count for chunk_accumulator in chunk_accumulators],
            "beta": [chunk_accumulator.regenie2_linear_result.beta for chunk_accumulator in chunk_accumulators],
            "standard_error": [
                chunk_accumulator.regenie2_linear_result.standard_error for chunk_accumulator in chunk_accumulators
            ],
            "chi_squared": [
                chunk_accumulator.regenie2_linear_result.chi_squared for chunk_accumulator in chunk_accumulators
            ],
            "log10_p_value": [
                chunk_accumulator.regenie2_linear_result.log10_p_value for chunk_accumulator in chunk_accumulators
            ],
            "valid_mask": [
                chunk_accumulator.regenie2_linear_result.valid_mask for chunk_accumulator in chunk_accumulators
            ],
        }
    )
    host_values = {key: np.concatenate(value) for key, value in host_lists.items()}

    payloads: list[Regenie2LinearChunkPayload] = []
    variant_offset = 0
    for chunk_accumulator in chunk_accumulators:
        metadata = chunk_accumulator.metadata
        variant_count = len(metadata.position)
        next_variant_offset = variant_offset + variant_count
        payloads.append(
            Regenie2LinearChunkPayload(
                chunk_identifier=metadata.variant_start_index,
                variant_start_index=metadata.variant_start_index,
                variant_stop_index=metadata.variant_stop_index,
                chromosome=metadata.chromosome,
                position=metadata.position,
                variant_identifier=metadata.variant_identifiers,
                allele_one=metadata.allele_one,
                allele_two=metadata.allele_two,
                allele_one_frequency=np.ascontiguousarray(
                    host_values["allele_one_frequency"][variant_offset:next_variant_offset]
                ),
                observation_count=np.ascontiguousarray(
                    host_values["observation_count"][variant_offset:next_variant_offset]
                ),
                beta=np.ascontiguousarray(host_values["beta"][variant_offset:next_variant_offset]),
                standard_error=np.ascontiguousarray(host_values["standard_error"][variant_offset:next_variant_offset]),
                chi_squared=np.ascontiguousarray(host_values["chi_squared"][variant_offset:next_variant_offset]),
                log10_p_value=np.ascontiguousarray(host_values["log10_p_value"][variant_offset:next_variant_offset]),
                is_valid=np.ascontiguousarray(host_values["valid_mask"][variant_offset:next_variant_offset]),
            )
        )
        variant_offset = next_variant_offset
    return payloads


def build_chunk_payload(
    chunk_accumulator: LinearChunkAccumulator | LogisticChunkAccumulator | Regenie2LinearChunkAccumulator,
) -> ChunkPayload:
    """Build a host-side chunk payload from a device-resident accumulator."""
    if isinstance(chunk_accumulator, LinearChunkAccumulator):
        return build_linear_chunk_payload(
            metadata=chunk_accumulator.metadata,
            allele_one_frequency=chunk_accumulator.allele_one_frequency,
            observation_count=chunk_accumulator.observation_count,
            linear_result=chunk_accumulator.linear_result,
        )
    if isinstance(chunk_accumulator, Regenie2LinearChunkAccumulator):
        return build_regenie2_linear_chunk_payload(
            metadata=chunk_accumulator.metadata,
            allele_one_frequency=chunk_accumulator.allele_one_frequency,
            observation_count=chunk_accumulator.observation_count,
            regenie2_linear_result=chunk_accumulator.regenie2_linear_result,
        )
    return build_logistic_chunk_payload(
        metadata=chunk_accumulator.metadata,
        allele_one_frequency=chunk_accumulator.allele_one_frequency,
        observation_count=chunk_accumulator.observation_count,
        logistic_result=chunk_accumulator.logistic_result,
    )


def build_chunk_payload_batch(
    chunk_accumulators: collections.abc.Sequence[
        LinearChunkAccumulator | LogisticChunkAccumulator | Regenie2LinearChunkAccumulator
    ],
) -> list[ChunkPayload]:
    """Build host-side payloads from a same-mode accumulator batch."""
    if not chunk_accumulators:
        return []
    first_chunk_accumulator = chunk_accumulators[0]
    if isinstance(first_chunk_accumulator, Regenie2LinearChunkAccumulator):
        assert all(
            isinstance(chunk_accumulator, Regenie2LinearChunkAccumulator) for chunk_accumulator in chunk_accumulators
        )
        regenie_chunk_accumulators = typing.cast(
            "collections.abc.Sequence[Regenie2LinearChunkAccumulator]",
            chunk_accumulators,
        )
        return typing.cast(
            "list[ChunkPayload]",
            build_regenie2_linear_chunk_payload_batch(regenie_chunk_accumulators),
        )
    return [build_chunk_payload(chunk_accumulator) for chunk_accumulator in chunk_accumulators]


def concatenate_linear_results(
    accumulators: list[LinearChunkAccumulator],
) -> pl.DataFrame:
    """Concatenate linear chunk results and build a single DataFrame.

    Args:
        accumulators: List of chunk accumulators with JAX arrays.

    Returns:
        Single Polars DataFrame with all results.

    """
    if not accumulators:
        return pl.DataFrame()

    # Concatenate metadata (these are numpy arrays already)
    all_chromosomes = np.concatenate([acc.metadata.chromosome for acc in accumulators])
    all_positions = np.concatenate([acc.metadata.position for acc in accumulators])
    all_variant_identifiers = np.concatenate([acc.metadata.variant_identifiers for acc in accumulators])
    all_allele_one = np.concatenate([acc.metadata.allele_one for acc in accumulators])
    all_allele_two = np.concatenate([acc.metadata.allele_two for acc in accumulators])

    # Fetch lists of arrays from device, then concatenate on host
    host_lists = jax.device_get(
        {
            "allele_one_frequency": [acc.allele_one_frequency for acc in accumulators],
            "observation_count": [acc.observation_count for acc in accumulators],
            "beta": [acc.linear_result.beta for acc in accumulators],
            "standard_error": [acc.linear_result.standard_error for acc in accumulators],
            "test_statistic": [acc.linear_result.test_statistic for acc in accumulators],
            "p_value": [acc.linear_result.p_value for acc in accumulators],
            "valid_mask": [acc.linear_result.valid_mask for acc in accumulators],
        }
    )
    host_values = {key: np.concatenate(value) for key, value in host_lists.items()}

    return pl.DataFrame(
        {
            "chromosome": all_chromosomes,
            "position": all_positions,
            "variant_identifier": all_variant_identifiers,
            "allele_one": all_allele_one,
            "allele_two": all_allele_two,
            "allele_one_frequency": host_values["allele_one_frequency"],
            "observation_count": host_values["observation_count"],
            "beta": host_values["beta"],
            "standard_error": host_values["standard_error"],
            "t_statistic": host_values["test_statistic"],
            "p_value": host_values["p_value"],
            "is_valid": host_values["valid_mask"],
        }
    )


def concatenate_logistic_results(
    accumulators: list[LogisticChunkAccumulator],
) -> pl.DataFrame:
    """Concatenate logistic chunk results and build a single DataFrame.

    Args:
        accumulators: List of chunk accumulators with JAX arrays.

    Returns:
        Single Polars DataFrame with all results.

    """
    if not accumulators:
        return pl.DataFrame()

    # Concatenate metadata on the host; these values are already NumPy arrays.
    all_chromosomes = np.concatenate([acc.metadata.chromosome for acc in accumulators])
    all_positions = np.concatenate([acc.metadata.position for acc in accumulators])
    all_variant_identifiers = np.concatenate([acc.metadata.variant_identifiers for acc in accumulators])
    all_allele_one = np.concatenate([acc.metadata.allele_one for acc in accumulators])
    all_allele_two = np.concatenate([acc.metadata.allele_two for acc in accumulators])

    # Fetch lists of arrays from device, then concatenate on host
    host_lists = jax.device_get(
        {
            "allele_one_frequency": [acc.allele_one_frequency for acc in accumulators],
            "observation_count": [acc.observation_count for acc in accumulators],
            "beta": [acc.logistic_result.beta for acc in accumulators],
            "standard_error": [acc.logistic_result.standard_error for acc in accumulators],
            "test_statistic": [acc.logistic_result.test_statistic for acc in accumulators],
            "p_value": [acc.logistic_result.p_value for acc in accumulators],
            "method_code": [acc.logistic_result.method_code for acc in accumulators],
            "error_code": [acc.logistic_result.error_code for acc in accumulators],
            "converged_mask": [acc.logistic_result.converged_mask for acc in accumulators],
            "iteration_count": [acc.logistic_result.iteration_count for acc in accumulators],
            "valid_mask": [acc.logistic_result.valid_mask for acc in accumulators],
        }
    )
    host_values = {key: np.concatenate(value) for key, value in host_lists.items()}

    return pl.DataFrame(
        {
            "chromosome": all_chromosomes,
            "position": all_positions,
            "variant_identifier": all_variant_identifiers,
            "allele_one": all_allele_one,
            "allele_two": all_allele_two,
            "allele_one_frequency": host_values["allele_one_frequency"],
            "observation_count": host_values["observation_count"],
            "beta": host_values["beta"],
            "standard_error": host_values["standard_error"],
            "z_statistic": host_values["test_statistic"],
            "p_value": host_values["p_value"],
            "firth_flag": format_logistic_method_codes(host_values["method_code"]),
            "error_code": format_logistic_error_codes(host_values["error_code"]),
            "converged": host_values["converged_mask"],
            "iteration_count": host_values["iteration_count"],
            "is_valid": host_values["valid_mask"],
        }
    )


def concatenate_regenie2_linear_results(
    accumulators: list[Regenie2LinearChunkAccumulator],
) -> pl.DataFrame:
    """Concatenate REGENIE step 2 linear chunk results into one DataFrame."""
    if not accumulators:
        return pl.DataFrame()

    all_chromosomes = np.concatenate([accumulator.metadata.chromosome for accumulator in accumulators])
    all_positions = np.concatenate([accumulator.metadata.position for accumulator in accumulators])
    all_variant_identifiers = np.concatenate([accumulator.metadata.variant_identifiers for accumulator in accumulators])
    all_allele_one = np.concatenate([accumulator.metadata.allele_one for accumulator in accumulators])
    all_allele_two = np.concatenate([accumulator.metadata.allele_two for accumulator in accumulators])

    host_lists = jax.device_get(
        {
            "allele_one_frequency": [accumulator.allele_one_frequency for accumulator in accumulators],
            "observation_count": [accumulator.observation_count for accumulator in accumulators],
            "beta": [accumulator.regenie2_linear_result.beta for accumulator in accumulators],
            "standard_error": [accumulator.regenie2_linear_result.standard_error for accumulator in accumulators],
            "chi_squared": [accumulator.regenie2_linear_result.chi_squared for accumulator in accumulators],
            "log10_p_value": [accumulator.regenie2_linear_result.log10_p_value for accumulator in accumulators],
            "valid_mask": [accumulator.regenie2_linear_result.valid_mask for accumulator in accumulators],
        }
    )
    host_values = {key: np.concatenate(value) for key, value in host_lists.items()}

    return pl.DataFrame(
        {
            "chromosome": all_chromosomes,
            "position": all_positions,
            "variant_identifier": all_variant_identifiers,
            "allele_one": all_allele_one,
            "allele_two": all_allele_two,
            "allele_one_frequency": host_values["allele_one_frequency"],
            "observation_count": host_values["observation_count"],
            "beta": host_values["beta"],
            "standard_error": host_values["standard_error"],
            "chi_squared": host_values["chi_squared"],
            "log10_p_value": host_values["log10_p_value"],
            "is_valid": host_values["valid_mask"],
        }
    )


def compute_logistic_association_with_missing_exclusion(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_chunk: models.GenotypeChunk,
    max_iterations: int,
    tolerance: float,
    no_missing_constants: logistic.NoMissingLogisticConstants | None = None,
) -> models.LogisticAssociationEvaluation:
    """Compute logistic regression while excluding missing genotype rows per variant."""
    if not genotype_chunk.has_missing_values:
        with jax.profiler.TraceAnnotation("logistic.standard_no_missing"):
            return models.LogisticAssociationEvaluation(
                logistic_result=logistic.compute_logistic_association_chunk(
                    covariate_matrix=covariate_matrix,
                    phenotype_vector=phenotype_vector,
                    genotype_matrix=genotype_chunk.genotypes,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    no_missing_constants=no_missing_constants,
                ),
                allele_one_frequency=genotype_chunk.allele_one_frequency,
                observation_count=genotype_chunk.observation_count,
            )

    with jax.profiler.TraceAnnotation("logistic.prepare_missing_exclusion"):
        observation_mask = ~jnp.transpose(genotype_chunk.missing_mask)
        sanitized_genotype_matrix = jnp.asarray(
            jnp.where(genotype_chunk.missing_mask, 0.0, genotype_chunk.genotypes),
            dtype=jnp.float32,
        )
        observation_count = jnp.sum(observation_mask, axis=1, dtype=jnp.int32)
        allele_one_frequency = jnp.where(
            observation_count > 0,
            jnp.sum(jnp.transpose(sanitized_genotype_matrix), axis=1) / (2.0 * observation_count),
            0.0,
        )
    with jax.profiler.TraceAnnotation("logistic.compute_missing_exclusion"):
        return models.LogisticAssociationEvaluation(
            logistic_result=logistic.compute_logistic_association_chunk_with_mask(
                covariate_matrix=covariate_matrix,
                phenotype_vector=phenotype_vector,
                genotype_matrix=genotype_chunk.genotypes,
                observation_mask=observation_mask,
                max_iterations=max_iterations,
                tolerance=tolerance,
            ),
            allele_one_frequency=allele_one_frequency,
            observation_count=observation_count,
        )


def iter_linear_output_frames(
    *,
    genotype_source_config: source.GenotypeSourceConfig | None = None,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    prefetch_chunks: int = 0,
    committed_chunk_identifiers: set[int] | None = None,
    bed_prefix: Path | None = None,
) -> collections.abc.Iterator[LinearChunkAccumulator]:
    """Yield linear association chunk accumulators (JAX arrays, device memory)."""
    resolved_genotype_source_config = genotype_source_config
    if resolved_genotype_source_config is None:
        if bed_prefix is None:
            message = "Either genotype_source_config or bed_prefix must be provided."
            raise ValueError(message)
        resolved_genotype_source_config = build_plink_source_config(bed_prefix)
    genotype_reader = None
    if resolved_genotype_source_config.source_format == types.GenotypeSourceFormat.BGEN:
        genotype_reader = open_genotype_reader(resolved_genotype_source_config)
    reader_context = genotype_reader if genotype_reader is not None else contextlib.nullcontext()

    with reader_context:
        with jax.profiler.TraceAnnotation("linear.load_aligned_sample_data"):
            aligned_sample_data = load_aligned_sample_data_from_source(
                genotype_source_config=resolved_genotype_source_config,
                phenotype_path=phenotype_path,
                phenotype_name=phenotype_name,
                covariate_path=covariate_path,
                covariate_names=covariate_names,
                is_binary_trait=False,
                genotype_reader=genotype_reader,
            )
        with jax.profiler.TraceAnnotation("linear.prepare_state"):
            linear_association_state = prepare_linear_association_state(
                covariate_matrix=aligned_sample_data.covariate_matrix,
                phenotype_vector=aligned_sample_data.phenotype_vector,
            )

        chunk_iterator = iter_linear_genotype_chunks_from_source(
            genotype_source_config=resolved_genotype_source_config,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            prefetch_chunks=prefetch_chunks,
            genotype_reader=genotype_reader,
        )
        committed_identifier_set = committed_chunk_identifiers or set()
        for chunk_number, current_chunk in enumerate(chunk_iterator):
            chunk_identifier = current_chunk.metadata.variant_start_index
            if chunk_identifier in committed_identifier_set:
                continue
            with jax.profiler.StepTraceAnnotation("linear_chunk", step_num=chunk_number):
                with jax.profiler.TraceAnnotation("linear.compute"):
                    linear_result = compute_linear_association_chunk(
                        linear_association_state=linear_association_state,
                        genotype_matrix=current_chunk.genotypes,
                    )
                with jax.profiler.TraceAnnotation("linear.accumulate"):
                    yield LinearChunkAccumulator(
                        metadata=current_chunk.metadata,
                        allele_one_frequency=current_chunk.allele_one_frequency,
                        observation_count=current_chunk.observation_count,
                        linear_result=linear_result,
                    )


def iter_logistic_output_frames(
    *,
    genotype_source_config: source.GenotypeSourceConfig | None = None,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    max_iterations: int,
    tolerance: float,
    prefetch_chunks: int = 0,
    committed_chunk_identifiers: set[int] | None = None,
    bed_prefix: Path | None = None,
) -> collections.abc.Iterator[LogisticChunkAccumulator]:
    """Yield logistic association chunk accumulators (JAX arrays, device memory)."""
    resolved_genotype_source_config = genotype_source_config
    if resolved_genotype_source_config is None:
        if bed_prefix is None:
            message = "Either genotype_source_config or bed_prefix must be provided."
            raise ValueError(message)
        resolved_genotype_source_config = build_plink_source_config(bed_prefix)
    genotype_reader = None
    if resolved_genotype_source_config.source_format == types.GenotypeSourceFormat.BGEN:
        genotype_reader = open_genotype_reader(resolved_genotype_source_config)
    reader_context = genotype_reader if genotype_reader is not None else contextlib.nullcontext()

    with reader_context:
        with jax.profiler.TraceAnnotation("logistic.load_aligned_sample_data"):
            aligned_sample_data = load_aligned_sample_data_from_source(
                genotype_source_config=resolved_genotype_source_config,
                phenotype_path=phenotype_path,
                phenotype_name=phenotype_name,
                covariate_path=covariate_path,
                covariate_names=covariate_names,
                is_binary_trait=True,
                genotype_reader=genotype_reader,
            )
        with jax.profiler.TraceAnnotation("logistic.prepare_no_missing_constants"):
            no_missing_constants = prepare_no_missing_logistic_constants(
                covariate_matrix=aligned_sample_data.covariate_matrix,
                phenotype_vector=aligned_sample_data.phenotype_vector,
            )
        chunk_iterator = iter_genotype_chunks_from_source(
            genotype_source_config=resolved_genotype_source_config,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            prefetch_chunks=prefetch_chunks,
            genotype_reader=genotype_reader,
        )
        committed_identifier_set = committed_chunk_identifiers or set()
        for chunk_number, current_chunk in enumerate(chunk_iterator):
            chunk_identifier = current_chunk.metadata.variant_start_index
            if chunk_identifier in committed_identifier_set:
                continue
            with jax.profiler.StepTraceAnnotation("logistic_chunk", step_num=chunk_number):
                with jax.profiler.TraceAnnotation("logistic.compute"):
                    logistic_evaluation = compute_logistic_association_with_missing_exclusion(
                        covariate_matrix=aligned_sample_data.covariate_matrix,
                        phenotype_vector=aligned_sample_data.phenotype_vector,
                        genotype_chunk=current_chunk,
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                        no_missing_constants=no_missing_constants,
                    )
                with jax.profiler.TraceAnnotation("logistic.accumulate"):
                    yield LogisticChunkAccumulator(
                        metadata=current_chunk.metadata,
                        allele_one_frequency=logistic_evaluation.allele_one_frequency,
                        observation_count=logistic_evaluation.observation_count,
                        logistic_result=logistic_evaluation.logistic_result,
                    )


def iter_regenie2_linear_output_frames(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    prefetch_chunks: int = 0,
    committed_chunk_identifiers: set[int] | None = None,
) -> collections.abc.Iterator[Regenie2LinearChunkAccumulator]:
    """Yield REGENIE step 2 linear chunk accumulators."""
    if genotype_source_config.source_format != types.GenotypeSourceFormat.BGEN:
        message = "REGENIE step 2 linear association requires a BGEN genotype source."
        raise ValueError(message)

    genotype_reader = open_genotype_reader(genotype_source_config)
    committed_identifier_set = committed_chunk_identifiers or set()

    with genotype_reader:
        with jax.profiler.TraceAnnotation("regenie2_linear.load_aligned_sample_data"):
            aligned_sample_data = load_aligned_sample_data_from_source(
                genotype_source_config=genotype_source_config,
                phenotype_path=phenotype_path,
                phenotype_name=phenotype_name,
                covariate_path=covariate_path,
                covariate_names=covariate_names,
                is_binary_trait=False,
                genotype_reader=genotype_reader,
            )
        with jax.profiler.TraceAnnotation("regenie2_linear.prepare_state"):
            regenie2_linear_state = prepare_regenie2_linear_state(
                covariate_matrix=aligned_sample_data.covariate_matrix,
                phenotype_vector=aligned_sample_data.phenotype_vector,
            )
        with jax.profiler.TraceAnnotation("regenie2_linear.load_prediction_source"):
            prediction_source = load_prediction_source(prediction_list_path, phenotype_name)

        chunk_iterator = iter_linear_genotype_chunks_from_source(
            genotype_source_config=genotype_source_config,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            prefetch_chunks=prefetch_chunks,
            genotype_reader=genotype_reader,
        )

        current_chromosome: str | None = None
        current_loco_predictions: jax.Array | None = None
        chunk_number = 0
        for source_chunk in chunk_iterator:
            for current_chunk in split_linear_genotype_chunk_by_chromosome(source_chunk):
                chunk_identifier = current_chunk.metadata.variant_start_index
                if chunk_identifier in committed_identifier_set:
                    continue

                chromosome = str(current_chunk.metadata.chromosome[0])
                if chromosome != current_chromosome:
                    current_loco_predictions = prediction_source.get_chromosome_predictions(
                        chromosome=chromosome,
                        sample_family_identifiers=aligned_sample_data.family_identifiers,
                        sample_individual_identifiers=aligned_sample_data.individual_identifiers,
                    )
                    current_chromosome = chromosome

                assert current_loco_predictions is not None
                with jax.profiler.StepTraceAnnotation("regenie2_linear_chunk", step_num=chunk_number):
                    with jax.profiler.TraceAnnotation("regenie2_linear.compute"):
                        regenie2_linear_result = compute_regenie2_linear_chunk(
                            state=regenie2_linear_state,
                            genotype_matrix=current_chunk.genotypes,
                            loco_predictions=current_loco_predictions,
                        )
                    with jax.profiler.TraceAnnotation("regenie2_linear.accumulate"):
                        yield Regenie2LinearChunkAccumulator(
                            metadata=current_chunk.metadata,
                            allele_one_frequency=current_chunk.allele_one_frequency,
                            observation_count=current_chunk.observation_count,
                            regenie2_linear_result=regenie2_linear_result,
                        )
                chunk_number += 1


def run_linear_association(
    *,
    genotype_source_config: source.GenotypeSourceConfig | None = None,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    bed_prefix: Path | None = None,
) -> pl.DataFrame:
    """Run additive linear regression for all requested variants.

    Accumulates chunk results in device memory, then performs a single
    host synchronization and builds one Polars DataFrame at the end.

    """
    accumulators = list(
        iter_linear_output_frames(
            genotype_source_config=genotype_source_config,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            bed_prefix=bed_prefix,
        )
    )
    return concatenate_linear_results(accumulators)


def run_logistic_association(
    *,
    genotype_source_config: source.GenotypeSourceConfig | None = None,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    max_iterations: int,
    tolerance: float,
    bed_prefix: Path | None = None,
) -> pl.DataFrame:
    """Run additive logistic regression for all requested variants.

    Accumulates chunk results in device memory, then performs a single
    host synchronization and builds one Polars DataFrame at the end.

    """
    accumulators = list(
        iter_logistic_output_frames(
            genotype_source_config=genotype_source_config,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            max_iterations=max_iterations,
            tolerance=tolerance,
            bed_prefix=bed_prefix,
        )
    )
    return concatenate_logistic_results(accumulators)


def write_frame_iterator_to_tsv(
    frame_iterator: collections.abc.Iterator[LinearChunkAccumulator]
    | collections.abc.Iterator[LogisticChunkAccumulator]
    | collections.abc.Iterator[Regenie2LinearChunkAccumulator],
    output_path: Path,
) -> None:
    """Write accumulated chunk results to a TSV file.

    Accumulates all chunk results (keeping arrays in device memory),
    then performs a single host synchronization to build the final
    DataFrame and write to disk.

    Args:
        frame_iterator: Iterator yielding chunk accumulators.
        output_path: Path to write the TSV file.

    """
    # Accumulate all chunks first (stays in device memory)
    accumulators = list(frame_iterator)

    if not accumulators:
        # Write empty DataFrame with headers
        pl.DataFrame().write_csv(output_path, separator="\t")
        return

    # Determine type and concatenate (single host sync)
    first_acc = accumulators[0]
    if isinstance(first_acc, LinearChunkAccumulator):
        result_frame = concatenate_linear_results(
            [acc for acc in accumulators if isinstance(acc, LinearChunkAccumulator)]
        )
    elif isinstance(first_acc, Regenie2LinearChunkAccumulator):
        result_frame = concatenate_regenie2_linear_results(
            [acc for acc in accumulators if isinstance(acc, Regenie2LinearChunkAccumulator)]
        )
    else:
        result_frame = concatenate_logistic_results(
            [acc for acc in accumulators if isinstance(acc, LogisticChunkAccumulator)]
        )

    # Write the complete DataFrame
    result_frame.write_csv(output_path, separator="\t")
