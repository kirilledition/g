"""Genotype coding helpers shared by REGENIE compute kernels."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g import types
from g.compute.common import dtype as compute_dtype

ALLELE_COUNT_MULTIPLIER = 2.0
EIGHT_BIT_PROBABILITY_DENOMINATOR = 255.0
PACKED8_DIPLOID_NUMERATOR = 510.0


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


def convert_sample_major_to_variant_major(
    genotype_matrix: jax.Array,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> jax.Array:
    """Convert sample-major dosages to the canonical variant-major compute layout.

    Args:
        genotype_matrix: Sample-major dosage matrix.
        score_dtype: Floating-point dtype for score-test computation.

    Returns:
        Variant-major dosage matrix.

    """
    return jnp.asarray(genotype_matrix, dtype=compute_dtype.resolve_jax_dtype(score_dtype)).T


def decode_packed8_probability_pairs_to_variant_major_dosage(
    packed_probability_pairs_by_variant: jax.Array,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> jax.Array:
    """Decode trusted unphased 8-bit BGEN probability pairs to variant-major dosage.

    Args:
        packed_probability_pairs_by_variant: Variant-major uint8 probability pairs.
        score_dtype: Floating-point dtype for the decoded dosage matrix.

    Returns:
        Variant-major dosage matrix decoded on the active JAX device.

    """
    compute_type = compute_dtype.resolve_jax_dtype(score_dtype)
    probability_values = jnp.asarray(packed_probability_pairs_by_variant, dtype=compute_type)
    homozygous_reference_probability_byte = probability_values[:, :, 0]
    heterozygous_probability_byte = probability_values[:, :, 1]
    return (
        PACKED8_DIPLOID_NUMERATOR
        - (ALLELE_COUNT_MULTIPLIER * homozygous_reference_probability_byte)
        - heterozygous_probability_byte
    ) / EIGHT_BIT_PROBABILITY_DENOMINATOR


def normalize_high_frequency_diploid_genotypes_sample_major(
    genotype_matrix: jax.Array,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> jax.Array:
    """Shift high-frequency diploid dosages to avoid score-kernel cancellation.

    The model includes an intercept, so subtracting a per-variant constant does
    not change the residualized genotype or score statistic. It does keep rare
    reference-allele carriers near zero before float32 matrix products.

    Args:
        genotype_matrix: Sample-major dosage matrix.
        score_dtype: Floating-point dtype for score-test computation.

    Returns:
        Shifted sample-major dosage matrix.

    """
    genotype_matrix_compute = jnp.asarray(genotype_matrix, dtype=compute_dtype.resolve_jax_dtype(score_dtype))
    genotype_mean = jnp.mean(genotype_matrix_compute, axis=0)
    genotype_offset = jnp.where(genotype_mean > 1.0, ALLELE_COUNT_MULTIPLIER, 0.0)
    return genotype_matrix_compute - genotype_offset[None, :]


def normalize_high_frequency_diploid_genotypes_variant_major(
    genotype_matrix_by_variant: jax.Array,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> jax.Array:
    """Shift high-frequency diploid dosages to avoid score-kernel cancellation.

    Args:
        genotype_matrix_by_variant: Variant-major dosage matrix.
        score_dtype: Floating-point dtype for score-test computation.

    Returns:
        Shifted variant-major dosage matrix.

    """
    genotype_matrix_by_variant_compute = jnp.asarray(
        genotype_matrix_by_variant,
        dtype=compute_dtype.resolve_jax_dtype(score_dtype),
    )
    genotype_mean = jnp.mean(genotype_matrix_by_variant_compute, axis=1)
    genotype_offset = jnp.where(genotype_mean > 1.0, ALLELE_COUNT_MULTIPLIER, 0.0)
    return genotype_matrix_by_variant_compute - genotype_offset[:, None]


def build_regenie_flipped_genotypes(
    genotype_matrix_by_variant: jax.Array,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> RegenieGenotypeFlipResult:
    """Code variant-major genotypes the way REGENIE does before testing.

    Args:
        genotype_matrix_by_variant: Variant-major dosage matrix.
        dosage_sum: Optional native per-variant dosage sum.
        observation_count: Optional native per-variant observed genotype count.

    Returns:
        Flipped genotype matrix and per-variant flip mask.

    """
    if dosage_sum is None or observation_count is None:
        genotype_mean = jnp.mean(genotype_matrix_by_variant, axis=1)
    else:
        dosage_sum_compute = jnp.asarray(dosage_sum, dtype=genotype_matrix_by_variant.dtype)
        observation_count_compute = jnp.asarray(observation_count, dtype=genotype_matrix_by_variant.dtype)
        genotype_mean = dosage_sum_compute / jnp.maximum(observation_count_compute, 1.0)
    flip_mask = genotype_mean > 1.0
    flipped_genotype_matrix_by_variant = jnp.where(
        flip_mask[:, None],
        ALLELE_COUNT_MULTIPLIER - genotype_matrix_by_variant,
        genotype_matrix_by_variant,
    )
    return RegenieGenotypeFlipResult(
        genotype_matrix_by_variant=flipped_genotype_matrix_by_variant,
        flip_mask=flip_mask,
    )
