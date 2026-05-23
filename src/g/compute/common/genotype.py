"""Genotype coding helpers shared by REGENIE compute kernels."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

ALLELE_COUNT_MULTIPLIER = 2.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RegenieGenotypeFlipResult:
    """REGENIE-style genotype coding for score and correction lanes.

    Attributes:
        genotype_matrix_by_variant: Candidate genotypes coded to the minor allele when REGENIE would flip.
        flip_mask: Per-candidate flag indicating that beta must be flipped back for A1 output.

    """

    genotype_matrix_by_variant: jax.Array
    flip_mask: jax.Array


def convert_sample_major_to_variant_major(genotype_matrix: jax.Array) -> jax.Array:
    """Convert sample-major dosages to the canonical variant-major compute layout.

    Args:
        genotype_matrix: Sample-major dosage matrix.

    Returns:
        Variant-major dosage matrix.

    """
    return jnp.asarray(genotype_matrix, dtype=jnp.float32).T


def normalize_high_frequency_diploid_genotypes_sample_major(genotype_matrix: jax.Array) -> jax.Array:
    """Shift high-frequency diploid dosages to avoid float32 cancellation.

    The model includes an intercept, so subtracting a per-variant constant does
    not change the residualized genotype or score statistic. It does keep rare
    reference-allele carriers near zero before float32 matrix products.

    Args:
        genotype_matrix: Sample-major dosage matrix.

    Returns:
        Shifted sample-major dosage matrix.

    """
    genotype_matrix_compute = jnp.asarray(genotype_matrix, dtype=jnp.float32)
    genotype_mean = jnp.mean(genotype_matrix_compute, axis=0)
    genotype_offset = jnp.where(genotype_mean > 1.0, ALLELE_COUNT_MULTIPLIER, 0.0)
    return genotype_matrix_compute - genotype_offset[None, :]


def normalize_high_frequency_diploid_genotypes_variant_major(genotype_matrix_by_variant: jax.Array) -> jax.Array:
    """Shift high-frequency diploid dosages to avoid float32 cancellation.

    Args:
        genotype_matrix_by_variant: Variant-major dosage matrix.

    Returns:
        Shifted variant-major dosage matrix.

    """
    genotype_matrix_by_variant_compute = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    genotype_mean = jnp.mean(genotype_matrix_by_variant_compute, axis=1)
    genotype_offset = jnp.where(genotype_mean > 1.0, ALLELE_COUNT_MULTIPLIER, 0.0)
    return genotype_matrix_by_variant_compute - genotype_offset[:, None]


def build_regenie_flipped_genotypes(
    genotype_matrix_by_variant: jax.Array,
) -> RegenieGenotypeFlipResult:
    """Code variant-major genotypes the way REGENIE does before testing.

    Args:
        genotype_matrix_by_variant: Variant-major dosage matrix.

    Returns:
        Flipped genotype matrix and per-variant flip mask.

    """
    allele_count = jnp.sum(genotype_matrix_by_variant, axis=1)
    flip_threshold = jnp.asarray(genotype_matrix_by_variant.shape[1], dtype=genotype_matrix_by_variant.dtype)
    flip_mask = allele_count > flip_threshold
    flipped_genotype_matrix_by_variant = jnp.where(
        flip_mask[:, None],
        ALLELE_COUNT_MULTIPLIER - genotype_matrix_by_variant,
        genotype_matrix_by_variant,
    )
    return RegenieGenotypeFlipResult(
        genotype_matrix_by_variant=flipped_genotype_matrix_by_variant,
        flip_mask=flip_mask,
    )
