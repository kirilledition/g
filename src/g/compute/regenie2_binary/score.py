"""Binary score-test kernels for REGENIE step 2."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g import types
from g.compute.common import dtype as compute_dtype
from g.compute.common import genotype, pvalue
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state


def compute_positive_variance_mask(
    variance: jax.Array,
    reference_sum_squares: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
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


def compute_binary_score_test_chunk_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult:
    """Compute the binary score test from canonical variant-major genotypes.

    Args:
        chromosome_state: Chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        correction_plan: Binary fallback/correction policy.
        kernel_config: Binary-kernel numerical policy.
        dosage_sum: Optional native per-variant dosage sum.
        observation_count: Optional native per-variant observed genotype count.
        score_dtype: Floating-point dtype for score-test computation.

    Returns:
        Uncorrected score-test result for the chunk.

    """
    multi_chromosome_state = regenie2_binary_state.build_multi_binary_chromosome_state_from_single(chromosome_state)
    multi_result = compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=multi_chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )
    return regenie2_binary_result.squeeze_single_binary_score_result(multi_result)


def compute_multi_binary_score_test_chunk_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Compute batched binary score tests for trait-major states and variant-major genotypes.

    Args:
        chromosome_state: Trait-major chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        correction_plan: Binary fallback/correction policy.
        kernel_config: Binary-kernel numerical policy.
        dosage_sum: Optional native per-variant dosage sum.
        observation_count: Optional native per-variant observed genotype count.
        score_dtype: Floating-point dtype for score-test computation.

    Returns:
        Trait-major score-test result for the chunk.

    """
    raw_genotype_matrix_by_variant = jnp.asarray(
        genotype_matrix_by_variant,
        dtype=compute_dtype.resolve_jax_dtype(score_dtype),
    )
    genotype_mean = compute_genotype_mean(
        raw_genotype_matrix_by_variant,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
    )
    trait_count = chromosome_state.score_residual.shape[0]
    covariate_count = chromosome_state.score_projection_matrix.shape[1]
    variant_count = raw_genotype_matrix_by_variant.shape[0]
    genotype_flip_mask = genotype_mean > 1.0
    genotype_flip_mask_by_trait_variant = genotype_flip_mask[None, :]
    genotype_matrix_by_variant_squared = raw_genotype_matrix_by_variant * raw_genotype_matrix_by_variant
    stacked_product_by_variant = raw_genotype_matrix_by_variant @ chromosome_state.score_right_hand_matrix.T
    projection_row_count = trait_count * covariate_count
    weighted_genotype_sum_start = projection_row_count
    score_start = weighted_genotype_sum_start + trait_count
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
        chromosome_state.bernoulli_weight,
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
    extra_code = regenie2_binary_correction.build_extra_code(log10_p_value, valid_mask, correction_plan)
    return regenie2_binary_result.build_multi_binary_score_test_chunk_result(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        valid_mask=valid_mask,
    )


def compute_genotype_mean(
    genotype_matrix_by_variant: jax.Array,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> jax.Array:
    """Compute per-variant genotype means from native stats when available."""
    if dosage_sum is None or observation_count is None:
        return jnp.mean(genotype_matrix_by_variant, axis=1)
    dosage_sum_compute = jnp.asarray(dosage_sum, dtype=genotype_matrix_by_variant.dtype)
    observation_count_compute = jnp.asarray(observation_count, dtype=genotype_matrix_by_variant.dtype)
    return dosage_sum_compute / jnp.maximum(observation_count_compute, 1.0)
