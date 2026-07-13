"""Binary score-test kernels for REGENIE step 2."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.compute.common import genotype, linalg, pvalue
from g.compute.common import result as association_result
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import state as regenie2_binary_state

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import result as regenie2_binary_result

SCORE_STATIC_ARGNAMES = (
    "firth_candidate_p_threshold",
    "minimum_variance",
    "relative_variance_tolerance",
)


def compute_multi_binary_score_test_chunk_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    firth_candidate_p_threshold: float | None,
    minimum_variance: float,
    relative_variance_tolerance: float,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Compute batched binary score tests for trait-major states and variant-major genotypes.

    Args:
        chromosome_state: Trait-major chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        firth_candidate_p_threshold: Firth candidate threshold, or ``None`` for score-only execution.
        minimum_variance: Absolute variance floor.
        relative_variance_tolerance: Relative variance floor multiplier.
        native_genotype_mean: Optional native per-variant genotype mean.

    Returns:
        Trait-major score-test result for the chunk.

    """
    raw_genotype_matrix_by_variant = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
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
    positive_variance_mask = linalg.compute_positive_residual_variance_mask(
        variance,
        weighted_genotype_sum_squares,
        minimum_variance,
        relative_variance_tolerance,
    )
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


compute_multi_binary_score_test_variant_major = jax.jit(
    compute_multi_binary_score_test_chunk_variant_major,
    static_argnames=SCORE_STATIC_ARGNAMES,
)

compute_multi_binary_score_test_variant_major_donating_inputs = jax.jit(
    compute_multi_binary_score_test_chunk_variant_major,
    static_argnames=SCORE_STATIC_ARGNAMES,
    donate_argnames=("native_genotype_mean",),
)


def compute_multi_binary_score_test_packed8_core(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    firth_candidate_p_threshold: float | None,
    minimum_variance: float,
    relative_variance_tolerance: float,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Decode packed8 genotypes and compute binary score statistics."""
    return compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
            packed_probability_pairs_by_variant
        ),
        firth_candidate_p_threshold=firth_candidate_p_threshold,
        minimum_variance=minimum_variance,
        relative_variance_tolerance=relative_variance_tolerance,
        native_genotype_mean=native_genotype_mean,
    )


compute_multi_binary_score_test_packed8_donating_inputs = jax.jit(
    compute_multi_binary_score_test_packed8_core,
    static_argnames=SCORE_STATIC_ARGNAMES,
    donate_argnames=("native_genotype_mean",),
)
