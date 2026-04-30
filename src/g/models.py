"""Typed data models for GWAS execution."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass(frozen=True)
class AlignedSampleData:
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


@dataclass(frozen=True)
class VariantMetadata:
    """Metadata describing a contiguous block of variants.

    Attributes:
        variant_start_index: Start index of the variant slice.
        variant_stop_index: Stop index of the variant slice.
        chromosome: Chromosome identifiers per variant.
        variant_identifiers: Variant identifiers per variant.
        position: Genomic positions per variant.
        allele_one: First allele per variant.
        allele_two: Second allele per variant.

    """

    variant_start_index: int
    variant_stop_index: int
    chromosome: npt.NDArray[np.str_]
    variant_identifiers: npt.NDArray[np.str_]
    position: npt.NDArray[np.int64]
    allele_one: npt.NDArray[np.str_]
    allele_two: npt.NDArray[np.str_]


@dataclass(frozen=True)
class GenotypeChunk:
    """Genotype matrix and metadata for a chunk of variants.

    Attributes:
        genotypes: Mean-imputed genotype matrix.
        missing_mask: Boolean mask indicating missing values.
        has_missing_values: Whether the chunk contains any missing values.
        metadata: Variant metadata for the chunk.
        allele_one_frequency: Allele frequencies per variant.
        observation_count: Observation counts per variant.

    """

    genotypes: jax.Array
    missing_mask: jax.Array
    has_missing_values: bool
    metadata: VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array


@dataclass(frozen=True)
class DosageGenotypeChunk:
    """Dosage genotype chunk without missingness bookkeeping.

    Attributes:
        genotypes: Mean-imputed dosage matrix.
        metadata: Variant metadata for the chunk.
        allele_one_frequency: Allele frequencies per variant.
        observation_count: Observation counts per variant.

    """

    genotypes: jax.Array
    metadata: VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array


@dataclass(frozen=True)
class PreprocessedGenotypeChunkData:
    """Preprocessed genotype arrays before metadata attachment.

    Attributes:
        genotypes: Mean-imputed genotype matrix.
        missing_mask: Boolean mask indicating missing values.
        has_missing_values: Whether the chunk contains any missing values.
        allele_one_frequency: Allele frequencies per variant.
        observation_count: Observation counts per variant.

    """

    genotypes: jax.Array
    missing_mask: jax.Array
    has_missing_values: bool
    allele_one_frequency: jax.Array
    observation_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2LinearState:
    """Precomputed state for REGENIE step 2 linear association.

    This state is computed once before processing genotype chunks and contains
    the covariate projection matrix and covariate-residualized phenotype.
    The LOCO predictions are applied per-chromosome during chunk processing.

    Attributes:
        covariate_matrix: Covariate design matrix including intercept.
        covariate_matrix_transpose: Transpose of the covariate design matrix.
        covariate_crossproduct_cholesky_factor: Lower-triangular Cholesky factor of X'X.
        phenotype_residual: Phenotype residualized against covariates (before LOCO adjustment).
        sample_count: Number of samples.
        degrees_of_freedom: Residual degrees of freedom for one-variant tests.

    """

    covariate_matrix: jax.Array
    covariate_matrix_transpose: jax.Array
    covariate_crossproduct_cholesky_factor: jax.Array
    phenotype_residual: jax.Array
    sample_count: jax.Array
    degrees_of_freedom: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2LinearChromosomeState:
    """Chromosome-specific REGENIE step 2 linear state.

    Attributes:
        covariate_matrix_transpose: Transpose of the covariate design matrix.
        covariate_crossproduct_cholesky_factor: Lower-triangular Cholesky factor of X'X.
        adjusted_residual: Covariate-residualized phenotype after LOCO subtraction.
        adjusted_residual_sum_squares: Sum of squares of ``adjusted_residual``.
        degrees_of_freedom: Residual degrees of freedom for one-variant tests.

    """

    covariate_matrix_transpose: jax.Array
    covariate_crossproduct_cholesky_factor: jax.Array
    adjusted_residual: jax.Array
    adjusted_residual_sum_squares: jax.Array
    degrees_of_freedom: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2LinearChunkResult:
    """Association outputs for a REGENIE step 2 linear chunk.

    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        chi_squared: Chi-squared statistics (1 df).
        log10_p_value: Negative log10 p-values.
        valid_mask: Boolean mask for valid statistics.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    valid_mask: jax.Array
