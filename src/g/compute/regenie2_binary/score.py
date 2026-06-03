"""Binary score-test kernels for REGENIE step 2."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.compute.common import genotype, pvalue
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state

if typing.TYPE_CHECKING:
    from g import types


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
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult:
    """Compute the binary score test from canonical variant-major genotypes.

    Args:
        chromosome_state: Chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        correction_plan: Binary fallback/correction policy.
        kernel_config: Binary-kernel numerical policy.
        dosage_sum: Optional native per-variant dosage sum.
        observation_count: Optional native per-variant observed genotype count.

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
    )
    return regenie2_binary_result.squeeze_single_binary_score_result(multi_result)


def compute_multi_binary_score_test_chunk_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Compute batched binary score tests for trait-major states and variant-major genotypes.

    Args:
        chromosome_state: Trait-major chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        correction_plan: Binary fallback/correction policy.
        kernel_config: Binary-kernel numerical policy.
        dosage_sum: Optional native per-variant dosage sum.
        observation_count: Optional native per-variant observed genotype count.

    Returns:
        Trait-major score-test result for the chunk.

    """
    raw_genotype_matrix_by_variant = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    genotype_flip_result = genotype.build_regenie_flipped_genotypes(
        raw_genotype_matrix_by_variant,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
    )
    genotype_matrix_by_variant_float32 = genotype_flip_result.genotype_matrix_by_variant
    genotype_matrix_by_variant_squared = genotype_matrix_by_variant_float32 * genotype_matrix_by_variant_float32
    bernoulli_weight = chromosome_state.square_root_weight * chromosome_state.square_root_weight
    weighted_projection_matrix = (
        chromosome_state.weighted_genotype_projection_matrix * chromosome_state.square_root_weight[:, None, :]
    )
    projection_coordinates = jnp.einsum(
        "vs,tcs->tvc",
        genotype_matrix_by_variant_float32,
        weighted_projection_matrix,
    )
    weighted_genotype_sum_squares = jnp.einsum(
        "vs,ts->tv",
        genotype_matrix_by_variant_squared,
        bernoulli_weight,
    )
    projection_sum_squares = jnp.einsum("tvc,tvc->tv", projection_coordinates, projection_coordinates)
    variance = jnp.maximum(weighted_genotype_sum_squares - projection_sum_squares, 0.0)
    score = jnp.einsum("vs,ts->tv", genotype_matrix_by_variant_float32, chromosome_state.score_residual)
    null_logistic_converged = chromosome_state.null_logistic_converged[:, None]
    positive_variance_mask = compute_positive_variance_mask(variance, weighted_genotype_sum_squares, kernel_config)
    statistic_mask = positive_variance_mask & null_logistic_converged
    inverse_variance = jnp.where(statistic_mask, jnp.reciprocal(variance), 0.0)
    beta = jnp.where(
        statistic_mask,
        jnp.where(genotype_flip_result.flip_mask[None, :], -score * inverse_variance, score * inverse_variance),
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
