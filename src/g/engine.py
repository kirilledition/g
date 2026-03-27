"""High-level orchestration for Phase 1 association runs."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
import polars as pl

from g.compute.linear import compute_linear_association_chunk, prepare_linear_association_state
from g.compute.logistic import (
    LOGISTIC_ERROR_FIRTH_CONVERGE_FAIL,
    LOGISTIC_ERROR_LOGISTIC_CONVERGE_FAIL,
    LOGISTIC_ERROR_UNFINISHED,
    LOGISTIC_METHOD_FIRTH,
    NoMissingLogisticConstants,
    compute_logistic_association_chunk,
    compute_logistic_association_chunk_with_mask,
    prepare_no_missing_logistic_constants,
)
from g.io.plink import iter_genotype_chunks
from g.io.tabular import load_aligned_sample_data
from g.models import (
    GenotypeChunk,
    LinearAssociationChunkResult,
    LogisticAssociationChunkResult,
    LogisticAssociationEvaluation,
    VariantMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass
class LinearChunkAccumulator:
    """Accumulator for linear regression chunk results (JAX arrays, device memory).

    Attributes:
        metadata: Variant metadata for the chunk.
        allele_one_frequency: Allele frequencies per variant.
        observation_count: Observation counts per variant.
        linear_result: Linear regression results.

    """

    metadata: VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array
    linear_result: LinearAssociationChunkResult


@dataclass
class LogisticChunkAccumulator:
    """Accumulator for logistic regression chunk results (JAX arrays, device memory).

    Attributes:
        metadata: Variant metadata for the chunk.
        allele_one_frequency: Allele frequencies per variant.
        observation_count: Observation counts per variant.
        logistic_result: Logistic regression results.

    """

    metadata: VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array
    logistic_result: LogisticAssociationChunkResult


def format_logistic_method_codes(method_code_values: np.ndarray) -> np.ndarray:
    """Convert logistic method codes to PLINK-style FIRTH flags."""
    return np.where(method_code_values == LOGISTIC_METHOD_FIRTH, "Y", "N")


def format_logistic_error_codes(error_code_values: np.ndarray) -> np.ndarray:
    """Convert logistic error codes to PLINK-style error labels."""
    return np.where(
        error_code_values == LOGISTIC_ERROR_FIRTH_CONVERGE_FAIL,
        "FIRTH_CONVERGE_FAIL",
        np.where(
            error_code_values == LOGISTIC_ERROR_LOGISTIC_CONVERGE_FAIL,
            "LOGISTIC_CONVERGE_FAIL",
            np.where(
                error_code_values == LOGISTIC_ERROR_UNFINISHED,
                "UNFINISHED",
                ".",
            ),
        ),
    )


def build_linear_output_frame(
    metadata: VariantMetadata,
    allele_one_frequency: jax.Array,
    observation_count: jax.Array,
    linear_result: LinearAssociationChunkResult,
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


def build_logistic_output_frame(
    metadata: VariantMetadata,
    allele_one_frequency: jax.Array,
    observation_count: jax.Array,
    logistic_result: LogisticAssociationChunkResult,
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

    # Concatenate JAX arrays on device, then do ONE device_get
    all_allele_one_frequency = jnp.concatenate([acc.allele_one_frequency for acc in accumulators])
    all_observation_count = jnp.concatenate([acc.observation_count for acc in accumulators])
    all_beta = jnp.concatenate([acc.linear_result.beta for acc in accumulators])
    all_standard_error = jnp.concatenate([acc.linear_result.standard_error for acc in accumulators])
    all_test_statistic = jnp.concatenate([acc.linear_result.test_statistic for acc in accumulators])
    all_p_value = jnp.concatenate([acc.linear_result.p_value for acc in accumulators])
    all_valid_mask = jnp.concatenate([acc.linear_result.valid_mask for acc in accumulators])

    # Single host synchronization
    host_values = jax.device_get(
        {
            "allele_one_frequency": all_allele_one_frequency,
            "observation_count": all_observation_count,
            "beta": all_beta,
            "standard_error": all_standard_error,
            "test_statistic": all_test_statistic,
            "p_value": all_p_value,
            "valid_mask": all_valid_mask,
        }
    )

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

    # TODO: Isnt it faster to do that in polars???
    # Concatenate metadata (these are numpy arrays already)
    all_chromosomes = np.concatenate([acc.metadata.chromosome for acc in accumulators])
    all_positions = np.concatenate([acc.metadata.position for acc in accumulators])
    all_variant_identifiers = np.concatenate([acc.metadata.variant_identifiers for acc in accumulators])
    all_allele_one = np.concatenate([acc.metadata.allele_one for acc in accumulators])
    all_allele_two = np.concatenate([acc.metadata.allele_two for acc in accumulators])

    # Concatenate JAX arrays on device, then do ONE device_get
    all_allele_one_frequency = jnp.concatenate([acc.allele_one_frequency for acc in accumulators])
    all_observation_count = jnp.concatenate([acc.observation_count for acc in accumulators])
    all_beta = jnp.concatenate([acc.logistic_result.beta for acc in accumulators])
    all_standard_error = jnp.concatenate([acc.logistic_result.standard_error for acc in accumulators])
    all_test_statistic = jnp.concatenate([acc.logistic_result.test_statistic for acc in accumulators])
    all_p_value = jnp.concatenate([acc.logistic_result.p_value for acc in accumulators])
    all_method_code = jnp.concatenate([acc.logistic_result.method_code for acc in accumulators])
    all_error_code = jnp.concatenate([acc.logistic_result.error_code for acc in accumulators])
    all_converged_mask = jnp.concatenate([acc.logistic_result.converged_mask for acc in accumulators])
    all_iteration_count = jnp.concatenate([acc.logistic_result.iteration_count for acc in accumulators])
    all_valid_mask = jnp.concatenate([acc.logistic_result.valid_mask for acc in accumulators])

    # Single host synchronization
    host_values = jax.device_get(
        {
            "allele_one_frequency": all_allele_one_frequency,
            "observation_count": all_observation_count,
            "beta": all_beta,
            "standard_error": all_standard_error,
            "test_statistic": all_test_statistic,
            "p_value": all_p_value,
            "method_code": all_method_code,
            "error_code": all_error_code,
            "converged_mask": all_converged_mask,
            "iteration_count": all_iteration_count,
            "valid_mask": all_valid_mask,
        }
    )

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


def compute_logistic_association_with_missing_exclusion(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_chunk: GenotypeChunk,
    max_iterations: int,
    tolerance: float,
    no_missing_constants: NoMissingLogisticConstants | None = None,
) -> LogisticAssociationEvaluation:
    """Compute logistic regression while excluding missing genotype rows per variant."""
    if not genotype_chunk.has_missing_values:
        with jax.profiler.TraceAnnotation("logistic.standard_no_missing"):
            return LogisticAssociationEvaluation(
                logistic_result=compute_logistic_association_chunk(
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
        observation_count = jnp.sum(observation_mask, axis=1, dtype=jnp.int64)
        allele_one_frequency = jnp.where(
            observation_count > 0,
            jnp.sum(jnp.transpose(sanitized_genotype_matrix), axis=1) / (2.0 * observation_count),
            0.0,
        )
    with jax.profiler.TraceAnnotation("logistic.compute_missing_exclusion"):
        return LogisticAssociationEvaluation(
            logistic_result=compute_logistic_association_chunk_with_mask(
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
    bed_prefix: Path,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
) -> Iterator[LinearChunkAccumulator]:
    """Yield linear association chunk accumulators (JAX arrays, device memory)."""
    with jax.profiler.TraceAnnotation("linear.load_aligned_sample_data"):
        aligned_sample_data = load_aligned_sample_data(
            bed_prefix=bed_prefix,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            is_binary_trait=False,
        )
    with jax.profiler.TraceAnnotation("linear.prepare_state"):
        linear_association_state = prepare_linear_association_state(
            covariate_matrix=aligned_sample_data.covariate_matrix,
            phenotype_vector=aligned_sample_data.phenotype_vector,
        )

    chunk_iterator = iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=aligned_sample_data.sample_indices,
        expected_individual_identifiers=aligned_sample_data.individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    )
    current_chunk = next(chunk_iterator, None)
    for chunk_index, next_chunk in enumerate(
        itertools.chain(chunk_iterator, [None]),
    ):
        if current_chunk is None:
            break
        with jax.profiler.StepTraceAnnotation("linear_chunk", step_num=chunk_index):
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
        current_chunk = next_chunk


def iter_logistic_output_frames(
    bed_prefix: Path,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    max_iterations: int,
    tolerance: float,
) -> Iterator[LogisticChunkAccumulator]:
    """Yield logistic association chunk accumulators (JAX arrays, device memory)."""
    with jax.profiler.TraceAnnotation("logistic.load_aligned_sample_data"):
        aligned_sample_data = load_aligned_sample_data(
            bed_prefix=bed_prefix,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            is_binary_trait=True,
        )
    with jax.profiler.TraceAnnotation("logistic.prepare_no_missing_constants"):
        no_missing_constants = prepare_no_missing_logistic_constants(
            covariate_matrix=aligned_sample_data.covariate_matrix,
            phenotype_vector=aligned_sample_data.phenotype_vector,
        )
    chunk_iterator = iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=aligned_sample_data.sample_indices,
        expected_individual_identifiers=aligned_sample_data.individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    )
    current_chunk = next(chunk_iterator, None)
    for chunk_index, next_chunk in enumerate(
        itertools.chain(chunk_iterator, [None]),
    ):
        if current_chunk is None:
            break
        with jax.profiler.StepTraceAnnotation("logistic_chunk", step_num=chunk_index):
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
        current_chunk = next_chunk


def run_linear_association(
    bed_prefix: Path,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
) -> pl.DataFrame:
    """Run additive linear regression for all requested variants.

    Accumulates chunk results in device memory, then performs a single
    host synchronization and builds one Polars DataFrame at the end.

    """
    accumulators = list(
        iter_linear_output_frames(
            bed_prefix=bed_prefix,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
    )
    return concatenate_linear_results(accumulators)


def run_logistic_association(
    bed_prefix: Path,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    max_iterations: int,
    tolerance: float,
) -> pl.DataFrame:
    """Run additive logistic regression for all requested variants.

    Accumulates chunk results in device memory, then performs a single
    host synchronization and builds one Polars DataFrame at the end.

    """
    accumulators = list(
        iter_logistic_output_frames(
            bed_prefix=bed_prefix,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    )
    return concatenate_logistic_results(accumulators)


def write_frame_iterator_to_tsv(
    frame_iterator: Iterator[LinearChunkAccumulator] | Iterator[LogisticChunkAccumulator],
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
    else:
        result_frame = concatenate_logistic_results(
            [acc for acc in accumulators if isinstance(acc, LogisticChunkAccumulator)]
        )

    # Write the complete DataFrame
    result_frame.write_csv(output_path, separator="\t")
