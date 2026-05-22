"""Binary score-test kernels for REGENIE step 2."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.compute.common import genotype, pvalue
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary import types as regenie2_binary_types

if typing.TYPE_CHECKING:
    from g import types

MINIMUM_VARIANCE = 1.0e-8
RELATIVE_VARIANCE_TOLERANCE = 1.0e-6


def compute_positive_variance_mask(variance: jax.Array, reference_sum_squares: jax.Array) -> jax.Array:
    """Return a stable positive-variance mask after covariate projection.

    Args:
        variance: Residualized score-test variance.
        reference_sum_squares: Pre-projection weighted genotype sum of squares.

    Returns:
        Boolean mask for numerically usable score-test variance.

    """
    variance_floor = jnp.maximum(MINIMUM_VARIANCE, reference_sum_squares * RELATIVE_VARIANCE_TOLERANCE)
    return variance > variance_floor


def compute_binary_score_test_chunk_variant_major(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute the binary score test from canonical variant-major genotypes.

    Args:
        chromosome_state: Chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        correction_plan: Binary fallback/correction policy.

    Returns:
        Uncorrected score-test result for the chunk.

    """
    raw_genotype_matrix_by_variant = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    genotype_flip_result = genotype.build_regenie_flipped_genotypes(raw_genotype_matrix_by_variant)
    genotype_matrix_by_variant_float32 = genotype_flip_result.genotype_matrix_by_variant
    weighted_genotype_matrix_by_variant = (
        genotype_matrix_by_variant_float32 * chromosome_state.square_root_weight[None, :]
    )
    projection_coordinates = (
        weighted_genotype_matrix_by_variant @ chromosome_state.weighted_genotype_projection_matrix.T
    )
    weighted_genotype_sum_squares = jnp.einsum(
        "ij,ij->i",
        weighted_genotype_matrix_by_variant,
        weighted_genotype_matrix_by_variant,
    )
    projection_sum_squares = jnp.einsum("ij,ij->i", projection_coordinates, projection_coordinates)
    variance = jnp.maximum(weighted_genotype_sum_squares - projection_sum_squares, 0.0)
    score = genotype_matrix_by_variant_float32 @ chromosome_state.score_residual
    null_logistic_converged = chromosome_state.null_logistic_converged
    positive_variance_mask = compute_positive_variance_mask(variance, weighted_genotype_sum_squares)
    statistic_mask = positive_variance_mask & null_logistic_converged
    inverse_variance = jnp.where(statistic_mask, jnp.reciprocal(variance), 0.0)
    beta = jnp.where(
        statistic_mask,
        jnp.where(genotype_flip_result.flip_mask, -score * inverse_variance, score * inverse_variance),
        jnp.nan,
    )
    standard_error = jnp.where(statistic_mask, jnp.sqrt(inverse_variance), jnp.nan)
    chi_squared = jnp.where(
        null_logistic_converged,
        jnp.where(positive_variance_mask, score * score * inverse_variance, 0.0),
        jnp.nan,
    )
    log10_p_value = jnp.where(
        null_logistic_converged,
        pvalue.chi_squared_to_log10_p_value(chi_squared),
        jnp.nan,
    )
    valid_mask = null_logistic_converged & jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    extra_code = regenie2_binary_candidate_planning.build_extra_code(log10_p_value, valid_mask, correction_plan)
    return regenie2_binary_types.Regenie2BinaryChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        valid_mask=valid_mask,
        firth_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_failure_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_convergence_reason_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_correction_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_sparse_correction_mask=jnp.zeros_like(extra_code, dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
    )


def compute_multi_binary_score_test_chunk_variant_major(
    chromosome_state: regenie2_binary_types.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Compute batched binary score tests for trait-major states and variant-major genotypes.

    Args:
        chromosome_state: Trait-major chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        correction_plan: Binary fallback/correction policy.

    Returns:
        Trait-major score-test result for the chunk.

    """
    raw_genotype_matrix_by_variant = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    genotype_flip_result = genotype.build_regenie_flipped_genotypes(raw_genotype_matrix_by_variant)
    genotype_matrix_by_variant_float32 = genotype_flip_result.genotype_matrix_by_variant
    weighted_genotype_matrix_by_trait_variant_sample = (
        genotype_matrix_by_variant_float32[None, :, :] * chromosome_state.square_root_weight[:, None, :]
    )
    projection_coordinates = jnp.einsum(
        "tvs,tcs->tvc",
        weighted_genotype_matrix_by_trait_variant_sample,
        chromosome_state.weighted_genotype_projection_matrix,
    )
    weighted_genotype_sum_squares = jnp.einsum(
        "tvs,tvs->tv",
        weighted_genotype_matrix_by_trait_variant_sample,
        weighted_genotype_matrix_by_trait_variant_sample,
    )
    projection_sum_squares = jnp.einsum("tvc,tvc->tv", projection_coordinates, projection_coordinates)
    variance = jnp.maximum(weighted_genotype_sum_squares - projection_sum_squares, 0.0)
    score = jnp.einsum("vs,ts->tv", genotype_matrix_by_variant_float32, chromosome_state.score_residual)
    null_logistic_converged = chromosome_state.null_logistic_converged[:, None]
    positive_variance_mask = compute_positive_variance_mask(variance, weighted_genotype_sum_squares)
    statistic_mask = positive_variance_mask & null_logistic_converged
    inverse_variance = jnp.where(statistic_mask, jnp.reciprocal(variance), 0.0)
    beta = jnp.where(
        statistic_mask,
        jnp.where(genotype_flip_result.flip_mask[None, :], -score * inverse_variance, score * inverse_variance),
        jnp.nan,
    )
    standard_error = jnp.where(statistic_mask, jnp.sqrt(inverse_variance), jnp.nan)
    chi_squared = jnp.where(
        null_logistic_converged,
        jnp.where(positive_variance_mask, score * score * inverse_variance, 0.0),
        jnp.nan,
    )
    log10_p_value = jnp.where(
        null_logistic_converged,
        pvalue.chi_squared_to_log10_p_value(chi_squared),
        jnp.nan,
    )
    valid_mask = null_logistic_converged & jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    extra_code = regenie2_binary_candidate_planning.build_extra_code(log10_p_value, valid_mask, correction_plan)
    return regenie2_binary_types.Regenie2MultiBinaryChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        valid_mask=valid_mask,
        firth_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_failure_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_convergence_reason_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_correction_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_sparse_correction_mask=jnp.zeros_like(extra_code, dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
    )
