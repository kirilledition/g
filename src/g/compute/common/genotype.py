"""Genotype coding helpers shared by REGENIE compute kernels."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import dtype as compute_dtype

if typing.TYPE_CHECKING:
    from g import types

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


def decode_packed8_probability_pairs_to_variant_major_dosage(
    packed_probability_pairs_by_variant: jax.Array,
    score_dtype: types.FloatingPointDtype,
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


def compute_diploid_genotype_mean(
    genotype_matrix_by_variant: jax.Array,
    native_genotype_mean: jax.Array | None,
) -> jax.Array:
    """Compute per-variant genotype means from native statistics when available.

    Args:
        genotype_matrix_by_variant: Variant-major dosage matrix.
        native_genotype_mean: Optional native per-variant genotype mean.

    Returns:
        Per-variant genotype means in the genotype matrix dtype.

    """
    if native_genotype_mean is None:
        return jnp.mean(genotype_matrix_by_variant, axis=1)
    return jnp.asarray(native_genotype_mean, dtype=genotype_matrix_by_variant.dtype)


def build_regenie_flipped_genotypes(
    genotype_matrix_by_variant: jax.Array,
    native_genotype_mean: jax.Array | None,
) -> RegenieGenotypeFlipResult:
    """Code variant-major genotypes the way REGENIE does before testing.

    Args:
        genotype_matrix_by_variant: Variant-major dosage matrix.
        native_genotype_mean: Optional native per-variant genotype mean.

    Returns:
        Flipped genotype matrix and per-variant flip mask.

    """
    genotype_mean = compute_diploid_genotype_mean(
        genotype_matrix_by_variant,
        native_genotype_mean,
    )
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
