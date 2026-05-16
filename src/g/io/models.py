"""I/O payload models for genotype, sample, and variant data."""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import jax
    import numpy as np
    import numpy.typing as npt


@dataclass(frozen=True)
class AlignedSampleData:
    """Aligned sample, phenotype, and covariate inputs.

    Attributes:
        sample_indices: Genotype row indices after alignment and filtering.
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
