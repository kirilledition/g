"""High-level orchestration for Phase 1 association runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax.scipy.special import betainc

from g.compute.linear import compute_linear_association_chunk, prepare_linear_association_state
from g.compute.logistic import (
    LOGISTIC_ERROR_FIRTH_CONVERGE_FAIL,
    LOGISTIC_ERROR_LOGISTIC_CONVERGE_FAIL,
    LOGISTIC_ERROR_UNFINISHED,
    LOGISTIC_METHOD_FIRTH,
    compute_logistic_association_chunk,
    compute_logistic_association_chunk_with_mask,
)
from g.io.plink import iter_genotype_chunks
from g.io.tabular import load_aligned_sample_data
from g.models import LogisticAssociationEvaluation

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


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
    metadata,
    allele_one_frequency,
    observation_count,
    linear_result,
    sample_count: int,
    covariate_parameter_count: int,
) -> pl.DataFrame:
    """Build a tabular linear association result frame."""
    host_values = jax.device_get(
        {
            "allele_one_frequency": allele_one_frequency,
            "observation_count": observation_count,
            "beta": linear_result.beta,
            "standard_error": linear_result.standard_error,
            "test_statistic": linear_result.test_statistic,
            "valid_mask": linear_result.valid_mask,
        }
    )
    degrees_of_freedom = sample_count - covariate_parameter_count - 1
    absolute_test_statistic = jnp.abs(jnp.asarray(host_values["test_statistic"]))
    degrees_of_freedom_value = jnp.asarray(degrees_of_freedom, dtype=absolute_test_statistic.dtype)
    beta_inc_argument = degrees_of_freedom_value / (
        degrees_of_freedom_value + absolute_test_statistic * absolute_test_statistic
    )
    p_value = jax.device_get(
        betainc(0.5 * degrees_of_freedom_value, 0.5, beta_inc_argument),
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
            "p_value": p_value,
            "is_valid": host_values["valid_mask"],
        }
    )


def build_logistic_output_frame(
    metadata,
    allele_one_frequency,
    observation_count,
    logistic_result,
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


def compute_logistic_association_with_missing_exclusion(
    covariate_matrix,
    phenotype_vector,
    genotype_chunk,
    covariate_only_coefficients,
    max_iterations: int,
    tolerance: float,
) -> LogisticAssociationEvaluation:
    """Compute logistic regression while excluding missing genotype rows per variant."""
    if not bool(jax.device_get(jnp.any(genotype_chunk.missing_mask))):
        return LogisticAssociationEvaluation(
            logistic_result=compute_logistic_association_chunk(
                covariate_matrix=covariate_matrix,
                phenotype_vector=phenotype_vector,
                genotype_matrix=genotype_chunk.genotypes,
                covariate_only_coefficients=covariate_only_coefficients,
                max_iterations=max_iterations,
                tolerance=tolerance,
            ),
            allele_one_frequency=genotype_chunk.allele_one_frequency,
            observation_count=genotype_chunk.observation_count,
        )

    observation_mask = ~genotype_chunk.missing_mask.T
    sanitized_genotype_matrix = jnp.where(genotype_chunk.missing_mask, 0.0, genotype_chunk.genotypes)
    observation_count = jnp.sum(observation_mask, axis=1, dtype=jnp.int64)
    allele_one_frequency = jnp.where(
        observation_count > 0,
        jnp.sum(sanitized_genotype_matrix.T, axis=1) / (2.0 * observation_count),
        0.0,
    )
    return LogisticAssociationEvaluation(
        logistic_result=compute_logistic_association_chunk_with_mask(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            genotype_matrix=genotype_chunk.genotypes,
            observation_mask=observation_mask,
            covariate_only_coefficients=covariate_only_coefficients,
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
) -> Iterator[pl.DataFrame]:
    """Yield linear association result frames chunk by chunk."""
    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    linear_association_state = prepare_linear_association_state(
        covariate_matrix=aligned_sample_data.covariate_matrix,
        phenotype_vector=aligned_sample_data.phenotype_vector,
    )

    for genotype_chunk in iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=aligned_sample_data.sample_indices,
        expected_individual_identifiers=aligned_sample_data.individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    ):
        linear_result = compute_linear_association_chunk(
            linear_association_state=linear_association_state,
            genotype_matrix=genotype_chunk.genotypes,
        )
        yield build_linear_output_frame(
            metadata=genotype_chunk.metadata,
            allele_one_frequency=genotype_chunk.allele_one_frequency,
            observation_count=genotype_chunk.observation_count,
            linear_result=linear_result,
            sample_count=aligned_sample_data.covariate_matrix.shape[0],
            covariate_parameter_count=aligned_sample_data.covariate_matrix.shape[1],
        )


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
) -> Iterator[pl.DataFrame]:
    """Yield logistic association result frames chunk by chunk."""
    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=True,
    )
    covariate_only_coefficients = jnp.zeros(
        (aligned_sample_data.covariate_matrix.shape[1],),
        dtype=aligned_sample_data.covariate_matrix.dtype,
    )

    for genotype_chunk in iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=aligned_sample_data.sample_indices,
        expected_individual_identifiers=aligned_sample_data.individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    ):
        logistic_evaluation = compute_logistic_association_with_missing_exclusion(
            covariate_matrix=aligned_sample_data.covariate_matrix,
            phenotype_vector=aligned_sample_data.phenotype_vector,
            genotype_chunk=genotype_chunk,
            covariate_only_coefficients=covariate_only_coefficients,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        yield build_logistic_output_frame(
            metadata=genotype_chunk.metadata,
            allele_one_frequency=logistic_evaluation.allele_one_frequency,
            observation_count=logistic_evaluation.observation_count,
            logistic_result=logistic_evaluation.logistic_result,
        )


def run_linear_association(
    bed_prefix: Path,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
) -> pl.DataFrame:
    """Run additive linear regression for all requested variants."""
    output_frames = list(
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
    return pl.concat(output_frames, how="vertical") if output_frames else pl.DataFrame()


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
    """Run additive logistic regression for all requested variants."""
    output_frames = list(
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
    return pl.concat(output_frames, how="vertical") if output_frames else pl.DataFrame()


def write_frame_iterator_to_tsv(frame_iterator: Iterator[pl.DataFrame], output_path: Path) -> None:
    """Write chunked result frames to a TSV file incrementally."""
    with output_path.open("w", encoding="utf-8") as output_handle:
        for frame_index, output_frame in enumerate(frame_iterator):
            output_frame.write_csv(
                output_handle,
                separator="\t",
                include_header=frame_index == 0,
            )
