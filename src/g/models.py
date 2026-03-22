"""Typed data models for Phase 1 GWAS execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import jax
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
    metadata: VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array


class LinearAssociationChunkResult(NamedTuple):
    """Association outputs for a linear-regression chunk."""

    beta: jax.Array
    standard_error: jax.Array
    test_statistic: jax.Array
    p_value: jax.Array
    valid_mask: jax.Array


class LogisticAssociationChunkResult(NamedTuple):
    """Association outputs for a logistic-regression chunk."""

    beta: jax.Array
    standard_error: jax.Array
    test_statistic: jax.Array
    p_value: jax.Array
    converged_mask: jax.Array
    valid_mask: jax.Array
    iteration_count: jax.Array
