"""Binary score-test kernels for REGENIE step 2."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.compute.common import dtype as compute_dtype
from g.compute.common import genotype, pvalue
from g.compute.common import result as association_result
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import state as regenie2_binary_state

if typing.TYPE_CHECKING:
    from g import types
    from g.compute.regenie2_binary import result as regenie2_binary_result


def compute_positive_variance_mask(
    variance: jax.Array,
    reference_sum_squares: jax.Array,
    kernel_config: regenie2_binary_config.BinaryScoreConfig,
) -> jax.Array:
    """Return a stable positive-variance mask after covariate projection.

    Args:
        variance: Residualized score-test variance.
        reference_sum_squares: Pre-projection weighted genotype sum of squares.
        kernel_config: Binary-kernel numerical policy.

    Returns:
        Boolean mask for numerically usable score-test variance.

    """
    variance_floor = jnp.maximum(
        kernel_config.numerical.minimum_variance,
        reference_sum_squares * kernel_config.numerical.relative_variance_tolerance,
    )
    return variance > variance_floor


def compute_multi_binary_score_test_chunk_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    firth_candidate_p_threshold: float | None,
    kernel_config: regenie2_binary_config.BinaryScoreConfig,
    native_genotype_mean: jax.Array | None,
    score_dtype: types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Compute batched binary score tests for trait-major states and variant-major genotypes.

    Args:
        chromosome_state: Trait-major chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        firth_candidate_p_threshold: Firth candidate threshold, or ``None`` for score-only execution.
        kernel_config: Binary-kernel numerical policy.
        native_genotype_mean: Optional native per-variant genotype mean.
        score_dtype: Floating-point dtype for score-test computation.

    Returns:
        Trait-major score-test result for the chunk.

    """
    raw_genotype_matrix_by_variant = jnp.asarray(
        genotype_matrix_by_variant,
        dtype=compute_dtype.resolve_jax_dtype(score_dtype),
    )
    genotype_mean = genotype.compute_diploid_genotype_mean(
        raw_genotype_matrix_by_variant,
        native_genotype_mean,
    )
    trait_count = chromosome_state.null_logistic_converged.shape[0]
    covariate_count = chromosome_state.score_projection_sum.shape[1]
    variant_count = raw_genotype_matrix_by_variant.shape[0]
    genotype_flip_mask = genotype_mean > 1.0
    genotype_flip_mask_by_trait_variant = genotype_flip_mask[None, :]
    genotype_matrix_by_variant_squared = raw_genotype_matrix_by_variant * raw_genotype_matrix_by_variant
    stacked_product_by_variant = raw_genotype_matrix_by_variant @ chromosome_state.score_right_hand_matrix.T
    projection_row_count = trait_count * covariate_count
    weighted_genotype_sum_start = projection_row_count
    score_start = weighted_genotype_sum_start + trait_count
    bernoulli_weight = chromosome_state.score_right_hand_matrix[weighted_genotype_sum_start:score_start, :]
    projection_coordinates = jnp.reshape(
        stacked_product_by_variant[:, :projection_row_count],
        (variant_count, trait_count, covariate_count),
    )
    projection_coordinates = jnp.transpose(projection_coordinates, (1, 0, 2))
    weighted_genotype_sum = jnp.transpose(
        stacked_product_by_variant[:, weighted_genotype_sum_start:score_start],
        (1, 0),
    )
    score = jnp.transpose(
        stacked_product_by_variant[:, score_start:],
        (1, 0),
    )
    projection_coordinates = jnp.where(
        genotype_flip_mask_by_trait_variant[:, :, None],
        genotype.ALLELE_COUNT_MULTIPLIER * chromosome_state.score_projection_sum[:, None, :] - projection_coordinates,
        projection_coordinates,
    )
    weighted_genotype_sum_squares = jnp.einsum(
        "vs,ts->tv",
        genotype_matrix_by_variant_squared,
        bernoulli_weight,
    )
    weighted_genotype_sum_squares = jnp.where(
        genotype_flip_mask_by_trait_variant,
        (
            genotype.ALLELE_COUNT_MULTIPLIER
            * genotype.ALLELE_COUNT_MULTIPLIER
            * chromosome_state.bernoulli_weight_sum[:, None]
        )
        - (2.0 * genotype.ALLELE_COUNT_MULTIPLIER * weighted_genotype_sum)
        + weighted_genotype_sum_squares,
        weighted_genotype_sum_squares,
    )
    projection_sum_squares = jnp.einsum("tvc,tvc->tv", projection_coordinates, projection_coordinates)
    variance = jnp.maximum(weighted_genotype_sum_squares - projection_sum_squares, 0.0)
    score = jnp.where(
        genotype_flip_mask_by_trait_variant,
        genotype.ALLELE_COUNT_MULTIPLIER * chromosome_state.score_residual_sum[:, None] - score,
        score,
    )
    null_logistic_converged = chromosome_state.null_logistic_converged[:, None]
    positive_variance_mask = compute_positive_variance_mask(variance, weighted_genotype_sum_squares, kernel_config)
    statistic_mask = positive_variance_mask & null_logistic_converged
    inverse_variance = jnp.where(statistic_mask, jnp.reciprocal(variance), 0.0)
    beta = jnp.where(
        statistic_mask,
        jnp.where(genotype_flip_mask_by_trait_variant, -score * inverse_variance, score * inverse_variance),
        jnp.nan,
    )
    standard_error = jnp.where(statistic_mask, jnp.sqrt(inverse_variance), jnp.nan)
    chi_squared = jnp.where(
        statistic_mask,
        score * score * inverse_variance,
        jnp.nan,
    )
    log10_p_value = jnp.where(
        statistic_mask,
        pvalue.chi_squared_to_log10_p_value(chi_squared),
        jnp.nan,
    )
    valid_mask = null_logistic_converged & jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    correction_code = regenie2_binary_correction.build_correction_code(
        log10_p_value,
        valid_mask,
        firth_candidate_p_threshold,
    )
    return association_result.AssociationResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        correction_code=correction_code,
    )
