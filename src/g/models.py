"""Typed data models for Phase 1 GWAS execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class AlignedSampleData(NamedTuple):
    """Aligned sample, phenotype, and covariate inputs.

    Attributes:
        sample_indices: BED/FAM row indices after alignment and filtering.
        family_identifiers: Ordered family identifiers.
        individual_identifiers: Ordered individual identifiers.
        phenotype_name: Selected phenotype column name.
        phenotype_vector: Phenotype values as a 1D JAX array.
        covariate_names: Ordered covariate column names, including intercept.
        covariate_matrix: Covariate design matrix as a 2D JAX array.
        is_binary_trait: Whether the phenotype is binary.

    """

    sample_indices: npt.NDArray[np.int64]
    family_identifiers: npt.NDArray[np.str_]
    individual_identifiers: npt.NDArray[np.str_]
    phenotype_name: str
    phenotype_vector: jax.Array
    covariate_names: tuple[str, ...]
    covariate_matrix: jax.Array
    is_binary_trait: bool


class VariantMetadata(NamedTuple):
    """Metadata describing a contiguous block of variants."""

    chromosome: npt.NDArray[np.str_]
    variant_identifiers: npt.NDArray[np.str_]
    position: npt.NDArray[np.int64]
    allele_one: npt.NDArray[np.str_]
    allele_two: npt.NDArray[np.str_]


class GenotypeChunk(NamedTuple):
    """Genotype matrix and metadata for a chunk of variants."""

    genotypes: jax.Array
    missing_mask: jax.Array
    has_missing_values: bool
    metadata: VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array


class PreprocessedGenotypeChunkData(NamedTuple):
    """Preprocessed genotype arrays before metadata attachment."""

    genotypes: jax.Array
    missing_mask: jax.Array
    has_missing_values: bool
    allele_one_frequency: jax.Array
    observation_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class LinearAssociationChunkResult:
    """Association outputs for a linear-regression chunk.

    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        test_statistic: t-statistics.
        p_value: Two-tailed p-values.
        valid_mask: Boolean mask for valid statistics.

    """

    beta: jax.Array
    standard_error: jax.Array
    test_statistic: jax.Array
    p_value: jax.Array
    valid_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class LinearAssociationState:
    """Precomputed covariate-only state for linear association chunks.

    Attributes:
        covariate_matrix: Covariate design matrix.
        covariate_matrix_transpose: Transpose of the covariate design matrix.
        covariate_crossproduct_cholesky_factor: Lower-triangular Cholesky factor of X'X.
        phenotype_residual: Residuals after covariate adjustment.
        phenotype_residual_sum_squares: Sum of squared residuals.

    """

    covariate_matrix: jax.Array
    covariate_matrix_transpose: jax.Array
    covariate_crossproduct_cholesky_factor: jax.Array
    phenotype_residual: jax.Array
    phenotype_residual_sum_squares: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class LogisticAssociationChunkResult:
    """Association outputs for a logistic-regression chunk.

    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        test_statistic: Z-statistics (Wald tests).
        p_value: Two-tailed p-values.
        method_code: Method indicator encoded by `g.compute.logistic.LogisticMethod`.
        error_code: Error status code encoded by `g.compute.logistic.LogisticErrorCode`.
        converged_mask: Boolean mask for convergence.
        valid_mask: Boolean mask for valid statistics.
        iteration_count: IRLS iterations performed.

    """

    beta: jax.Array
    standard_error: jax.Array
    test_statistic: jax.Array
    p_value: jax.Array
    method_code: jax.Array
    error_code: jax.Array
    converged_mask: jax.Array
    valid_mask: jax.Array
    iteration_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class LogisticAssociationEvaluation:
    """Logistic association result and per-variant summary values."""

    logistic_result: LogisticAssociationChunkResult
    allele_one_frequency: jax.Array
    observation_count: jax.Array
