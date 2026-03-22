"""High-level orchestration for Phase 1 association runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from g.compute.linear import compute_linear_association_chunk, finalize_linear_p_values
from g.compute.logistic import compute_logistic_association_chunk, fit_covariate_only_logistic_regression
from g.io.plink import iter_genotype_chunks
from g.io.tabular import load_aligned_sample_data

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
        logistic_result = compute_logistic_association_chunk(
            covariate_matrix=aligned_sample_data.covariate_matrix,
            phenotype_vector=aligned_sample_data.phenotype_vector,
            genotype_matrix=genotype_chunk.genotypes,
            covariate_only_coefficients=covariate_only_coefficients,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        output_frames.append(
            build_logistic_output_frame(
                metadata=genotype_chunk.metadata,
                allele_one_frequency=genotype_chunk.allele_one_frequency,
                observation_count=genotype_chunk.observation_count,
                logistic_result=logistic_result,
            )
        )

    return pl.concat(output_frames, how="vertical") if output_frames else pl.DataFrame()
