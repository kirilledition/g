"""High-level orchestration for Phase 1 association runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl

from g.compute.linear import compute_linear_association_chunk, finalize_linear_p_values
from g.compute.logistic import compute_logistic_association_chunk, fit_covariate_only_logistic_regression
from g.io.plink import iter_genotype_chunks
from g.io.tabular import load_aligned_sample_data
from g.models import LogisticAssociationChunkResult

if TYPE_CHECKING:
    from pathlib import Path


def build_linear_output_frame(
    metadata,
    allele_one_frequency,
    observation_count,
    linear_result,
) -> pl.DataFrame:
    """Build a tabular linear association result frame."""
    return pl.DataFrame(
        {
            "chromosome": metadata.chromosome,
            "position": metadata.position,
            "variant_identifier": metadata.variant_identifiers,
            "allele_one": metadata.allele_one,
            "allele_two": metadata.allele_two,
            "allele_one_frequency": np.asarray(allele_one_frequency),
            "observation_count": np.asarray(observation_count),
            "beta": np.asarray(linear_result.beta),
            "standard_error": np.asarray(linear_result.standard_error),
            "t_statistic": np.asarray(linear_result.test_statistic),
            "p_value": np.asarray(linear_result.p_value),
            "is_valid": np.asarray(linear_result.valid_mask),
        }
    )


def build_logistic_output_frame(
    metadata,
    allele_one_frequency,
    observation_count,
    logistic_result,
) -> pl.DataFrame:
    """Build a tabular logistic association result frame."""
    return pl.DataFrame(
        {
            "chromosome": metadata.chromosome,
            "position": metadata.position,
            "variant_identifier": metadata.variant_identifiers,
            "allele_one": metadata.allele_one,
            "allele_two": metadata.allele_two,
            "allele_one_frequency": np.asarray(allele_one_frequency),
            "observation_count": np.asarray(observation_count),
            "beta": np.asarray(logistic_result.beta),
            "standard_error": np.asarray(logistic_result.standard_error),
            "z_statistic": np.asarray(logistic_result.test_statistic),
            "p_value": np.asarray(logistic_result.p_value),
            "converged": np.asarray(logistic_result.converged_mask),
            "iteration_count": np.asarray(logistic_result.iteration_count),
            "is_valid": np.asarray(logistic_result.valid_mask),
        }
    )


def compute_logistic_association_with_missing_exclusion(
    covariate_matrix,
    phenotype_vector,
    genotype_chunk,
    covariate_only_coefficients,
    max_iterations: int,
    tolerance: float,
) -> tuple[LogisticAssociationChunkResult, np.ndarray, np.ndarray]:
    """Compute logistic regression while excluding missing genotype rows per variant."""
    if not np.asarray(genotype_chunk.missing_mask).any():
        return (
            compute_logistic_association_chunk(
                covariate_matrix=covariate_matrix,
                phenotype_vector=phenotype_vector,
                genotype_matrix=genotype_chunk.genotypes,
                covariate_only_coefficients=covariate_only_coefficients,
                max_iterations=max_iterations,
                tolerance=tolerance,
            ),
            np.asarray(genotype_chunk.allele_one_frequency),
            np.asarray(genotype_chunk.observation_count),
        )

    genotype_matrix = np.asarray(genotype_chunk.genotypes)
    missing_mask = np.asarray(genotype_chunk.missing_mask)
    covariate_matrix_numpy = np.asarray(covariate_matrix)
    phenotype_vector_numpy = np.asarray(phenotype_vector)

    beta_values: list[float] = []
    standard_error_values: list[float] = []
    z_statistic_values: list[float] = []
    p_value_values: list[float] = []
    converged_values: list[bool] = []
    valid_values: list[bool] = []
    iteration_values: list[int] = []
    allele_frequency_values: list[float] = []
    observation_count_values: list[int] = []

    for variant_index in range(genotype_matrix.shape[1]):
        variant_nonmissing_mask = ~missing_mask[:, variant_index]
        filtered_covariates = covariate_matrix_numpy[variant_nonmissing_mask, :]
        filtered_phenotype = phenotype_vector_numpy[variant_nonmissing_mask]
        filtered_genotype = genotype_matrix[variant_nonmissing_mask, variant_index][:, None]
        filtered_observation_count = int(filtered_genotype.shape[0])
        observation_count_values.append(filtered_observation_count)
        allele_frequency_values.append(float(filtered_genotype[:, 0].mean() / 2.0))

        filtered_covariate_only_coefficients = fit_covariate_only_logistic_regression(
            covariate_matrix=jnp.asarray(filtered_covariates),
            phenotype_vector=jnp.asarray(filtered_phenotype),
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        variant_result = compute_logistic_association_chunk(
            covariate_matrix=jnp.asarray(filtered_covariates),
            phenotype_vector=jnp.asarray(filtered_phenotype),
            genotype_matrix=jnp.asarray(filtered_genotype),
            covariate_only_coefficients=filtered_covariate_only_coefficients,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        beta_values.append(float(np.asarray(variant_result.beta)[0]))
        standard_error_values.append(float(np.asarray(variant_result.standard_error)[0]))
        z_statistic_values.append(float(np.asarray(variant_result.test_statistic)[0]))
        p_value_values.append(float(np.asarray(variant_result.p_value)[0]))
        converged_values.append(bool(np.asarray(variant_result.converged_mask)[0]))
        valid_values.append(bool(np.asarray(variant_result.valid_mask)[0]))
        iteration_values.append(int(np.asarray(variant_result.iteration_count)[0]))

    return (
        LogisticAssociationChunkResult(
            beta=jnp.asarray(beta_values),
            standard_error=jnp.asarray(standard_error_values),
            test_statistic=jnp.asarray(z_statistic_values),
            p_value=jnp.asarray(p_value_values),
            converged_mask=jnp.asarray(converged_values),
            valid_mask=jnp.asarray(valid_values),
            iteration_count=jnp.asarray(iteration_values),
        ),
        np.asarray(allele_frequency_values),
        np.asarray(observation_count_values),
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
    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    output_frames: list[pl.DataFrame] = []

    for genotype_chunk in iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=aligned_sample_data.sample_indices,
        expected_individual_identifiers=aligned_sample_data.individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    ):
        linear_result = compute_linear_association_chunk(
            covariate_matrix=aligned_sample_data.covariate_matrix,
            phenotype_vector=aligned_sample_data.phenotype_vector,
            genotype_matrix=genotype_chunk.genotypes,
        )
        linear_result = finalize_linear_p_values(
            linear_result=linear_result,
            sample_count=aligned_sample_data.covariate_matrix.shape[0],
            covariate_parameter_count=aligned_sample_data.covariate_matrix.shape[1],
        )
        output_frames.append(
            build_linear_output_frame(
                metadata=genotype_chunk.metadata,
                allele_one_frequency=genotype_chunk.allele_one_frequency,
                observation_count=genotype_chunk.observation_count,
                linear_result=linear_result,
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
    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=True,
    )
    covariate_only_coefficients = fit_covariate_only_logistic_regression(
        covariate_matrix=aligned_sample_data.covariate_matrix,
        phenotype_vector=aligned_sample_data.phenotype_vector,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    output_frames: list[pl.DataFrame] = []
    for genotype_chunk in iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=aligned_sample_data.sample_indices,
        expected_individual_identifiers=aligned_sample_data.individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    ):
        logistic_result, allele_one_frequency, observation_count = compute_logistic_association_with_missing_exclusion(
            covariate_matrix=aligned_sample_data.covariate_matrix,
            phenotype_vector=aligned_sample_data.phenotype_vector,
            genotype_chunk=genotype_chunk,
            covariate_only_coefficients=covariate_only_coefficients,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        output_frames.append(
            build_logistic_output_frame(
                metadata=genotype_chunk.metadata,
                allele_one_frequency=allele_one_frequency,
                observation_count=observation_count,
                logistic_result=logistic_result,
            )
        )

    return pl.concat(output_frames, how="vertical") if output_frames else pl.DataFrame()
