"""Shared genotype preprocessing and chunk-construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from g.models import GenotypeChunk, LinearGenotypeChunk, PreprocessedGenotypeChunkData, VariantMetadata

if TYPE_CHECKING:
    import numpy as np


@jax.tree_util.register_dataclass
@dataclass
class PreprocessedGenotypeArrays:
    """Device-resident genotype preprocessing outputs.

    Attributes:
        genotypes: Mean-imputed genotype matrix.
        missing_mask: Boolean mask of missing genotype calls.
        allele_one_frequency: Allele-one frequency per variant.
        observation_count: Non-missing observation count per variant.

    """

    genotypes: jax.Array
    missing_mask: jax.Array
    allele_one_frequency: jax.Array
    observation_count: jax.Array


@jax.jit
def preprocess_genotype_matrix_arrays(genotype_matrix: jax.Array) -> PreprocessedGenotypeArrays:
    """Preprocess a raw genotype matrix into device-resident summary arrays."""
    genotype_matrix_float32 = jnp.asarray(genotype_matrix, dtype=jnp.float32)
    missing_mask = jnp.isnan(genotype_matrix_float32)
    observed_genotype_total = jnp.where(missing_mask, 0.0, genotype_matrix_float32).sum(axis=0)
    observation_count = jnp.sum(~missing_mask, axis=0, dtype=jnp.int32)
    column_means = observed_genotype_total / jnp.maximum(observation_count, 1)
    sanitized_column_means = jnp.where(observation_count > 0, column_means, 0.0)
    imputed_matrix = jnp.where(missing_mask, sanitized_column_means[None, :], genotype_matrix_float32)
    return PreprocessedGenotypeArrays(
        genotypes=imputed_matrix,
        missing_mask=missing_mask,
        allele_one_frequency=sanitized_column_means / 2.0,
        observation_count=observation_count,
    )


def preprocess_genotype_matrix(
    genotype_matrix: jax.Array,
    *,
    include_missing_value_flag: bool = True,
) -> PreprocessedGenotypeChunkData:
    """Preprocess a raw genotype matrix with Phase 1 semantics.

    Args:
        genotype_matrix: Raw genotype matrix with NaNs for missing values.
        include_missing_value_flag: Whether to materialize a host boolean that
            records if the chunk contains missing values.

    Returns:
        Mean-imputed genotype arrays and summary values.

    """
    preprocessed_arrays = preprocess_genotype_matrix_arrays(genotype_matrix)
    has_missing_values = (
        bool(jax.device_get(jnp.any(preprocessed_arrays.missing_mask))) if include_missing_value_flag else False
    )
    return PreprocessedGenotypeChunkData(
        genotypes=preprocessed_arrays.genotypes,
        missing_mask=preprocessed_arrays.missing_mask,
        has_missing_values=has_missing_values,
        allele_one_frequency=preprocessed_arrays.allele_one_frequency,
        observation_count=preprocessed_arrays.observation_count,
    )


def build_genotype_chunk(
    preprocessed_chunk_data: PreprocessedGenotypeChunkData,
    chromosome_values: np.ndarray,
    variant_identifier_values: np.ndarray,
    position_values: np.ndarray,
    allele_one_values: np.ndarray,
    allele_two_values: np.ndarray,
    variant_start: int,
    variant_stop: int,
) -> GenotypeChunk:
    """Build one genotype chunk from preprocessed arrays and metadata.

    Args:
        preprocessed_chunk_data: Preprocessed genotype arrays and summary values.
        chromosome_values: Full chromosome value array.
        variant_identifier_values: Full variant identifier array.
        position_values: Full position array.
        allele_one_values: Full allele-one array.
        allele_two_values: Full allele-two array.
        variant_start: Inclusive variant start index.
        variant_stop: Exclusive variant stop index.

    Returns:
        Mean-imputed genotype chunk with metadata and summary statistics.

    """
    variant_metadata = VariantMetadata(
        variant_start_index=variant_start,
        variant_stop_index=variant_stop,
        chromosome=chromosome_values[variant_start:variant_stop],
        variant_identifiers=variant_identifier_values[variant_start:variant_stop],
        position=position_values[variant_start:variant_stop],
        allele_one=allele_one_values[variant_start:variant_stop],
        allele_two=allele_two_values[variant_start:variant_stop],
    )
    return GenotypeChunk(
        genotypes=preprocessed_chunk_data.genotypes,
        missing_mask=preprocessed_chunk_data.missing_mask,
        has_missing_values=preprocessed_chunk_data.has_missing_values,
        metadata=variant_metadata,
        allele_one_frequency=preprocessed_chunk_data.allele_one_frequency,
        observation_count=preprocessed_chunk_data.observation_count,
    )


def build_linear_genotype_chunk(
    preprocessed_genotype_arrays: PreprocessedGenotypeArrays,
    chromosome_values: np.ndarray,
    variant_identifier_values: np.ndarray,
    position_values: np.ndarray,
    allele_one_values: np.ndarray,
    allele_two_values: np.ndarray,
    variant_start: int,
    variant_stop: int,
) -> LinearGenotypeChunk:
    """Build one linear-regression genotype chunk without missingness fields."""
    return LinearGenotypeChunk(
        genotypes=preprocessed_genotype_arrays.genotypes,
        metadata=VariantMetadata(
            variant_start_index=variant_start,
            variant_stop_index=variant_stop,
            chromosome=chromosome_values[variant_start:variant_stop],
            variant_identifiers=variant_identifier_values[variant_start:variant_stop],
            position=position_values[variant_start:variant_stop],
            allele_one=allele_one_values[variant_start:variant_stop],
            allele_two=allele_two_values[variant_start:variant_stop],
        ),
        allele_one_frequency=preprocessed_genotype_arrays.allele_one_frequency,
        observation_count=preprocessed_genotype_arrays.observation_count,
    )
