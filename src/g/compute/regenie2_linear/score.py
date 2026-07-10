"""Linear score-test kernels for REGENIE step 2."""

from __future__ import annotations

import functools
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g import types
from g.compute.common import dtype as compute_dtype
from g.compute.common import genotype, pvalue

if typing.TYPE_CHECKING:
    from g.compute.regenie2_linear import state as regenie2_linear_state


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiLinearChunkResult:
    """Trait-major association outputs for a multi-trait linear chunk.

    Attributes:
        beta: Estimated effect sizes with shape ``traits x variants``.
        standard_error: Standard errors with shape ``traits x variants``.
        chi_squared: Chi-squared statistics with shape ``traits x variants``.
        log10_p_value: Negative log10 p-values with shape ``traits x variants``.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array


def compute_positive_residual_variance_mask(
    variance: jax.Array,
    reference_sum_squares: jax.Array,
    *,
    minimum_variance: float,
    relative_variance_tolerance: float,
) -> jax.Array:
    """Return a stable positive residual-variance mask after covariate projection."""
    variance_floor = jnp.maximum(
        minimum_variance,
        reference_sum_squares * relative_variance_tolerance,
    )
    return variance > variance_floor


def compute_normalized_genotype_sum_squares_from_stats(
    *,
    genotype_dosage_sum: jax.Array,
    genotype_observation_count: jax.Array,
    genotype_imputed_dosage_square_sum: jax.Array,
    sample_count: int,
    score_dtype: types.FloatingPointDtype,
) -> jax.Array:
    """Compute shifted genotype sum of squares from native chunk statistics."""
    jax_dtype = compute_dtype.resolve_jax_dtype(score_dtype)
    dosage_sum_compute = jnp.asarray(genotype_dosage_sum, dtype=jax_dtype)
    observation_count_compute = jnp.asarray(genotype_observation_count, dtype=jax_dtype)
    imputed_dosage_square_sum_compute = jnp.asarray(genotype_imputed_dosage_square_sum, dtype=jax_dtype)
    sample_count_compute = jnp.asarray(sample_count, dtype=jax_dtype)
    genotype_mean = dosage_sum_compute / jnp.maximum(observation_count_compute, 1.0)
    imputed_dosage_sum_compute = genotype_mean * sample_count_compute
    genotype_offset = jnp.where(genotype_mean > 1.0, genotype.ALLELE_COUNT_MULTIPLIER, 0.0)
    shifted_sum_squares = (
        imputed_dosage_square_sum_compute
        - 2.0 * genotype_offset * imputed_dosage_sum_compute
        + sample_count_compute * genotype_offset * genotype_offset
    )
    return shifted_sum_squares


@functools.partial(
    jax.jit,
    static_argnames=("score_dtype", "linear_minimum_variance", "linear_relative_variance_tolerance"),
)
def compute_regenie2_linear_chunk_trait_major_variant_major(
    *,
    chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    genotype_dosage_sum: jax.Array | None,
    genotype_observation_count: jax.Array | None,
    genotype_imputed_dosage_square_sum: jax.Array | None,
    score_dtype: types.FloatingPointDtype,
    linear_minimum_variance: float,
    linear_relative_variance_tolerance: float,
) -> Regenie2MultiLinearChunkResult:
    """Compute linear score-test statistics for trait-major residuals and variant-major genotypes."""
    if genotype_dosage_sum is None or genotype_observation_count is None:
        normalized_genotype_matrix_by_variant = genotype.normalize_high_frequency_diploid_genotypes_variant_major(
            genotype_matrix_by_variant,
            score_dtype,
        )
    else:
        normalized_genotype_matrix_by_variant = (
            genotype.normalize_high_frequency_diploid_genotypes_variant_major_from_stats(
                genotype_matrix_by_variant,
                genotype_dosage_sum,
                genotype_observation_count,
                score_dtype,
            )
        )
    if genotype_dosage_sum is None or genotype_observation_count is None or genotype_imputed_dosage_square_sum is None:
        genotype_sum_squares_compute = jnp.einsum(
            "ij,ij->i",
            normalized_genotype_matrix_by_variant,
            normalized_genotype_matrix_by_variant,
        )
    else:
        genotype_sum_squares_compute = compute_normalized_genotype_sum_squares_from_stats(
            genotype_dosage_sum=genotype_dosage_sum,
            genotype_observation_count=genotype_observation_count,
            genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
            sample_count=genotype_matrix_by_variant.shape[1],
            score_dtype=score_dtype,
        )
    covariate_count = chromosome_state.whitened_covariate_transpose.shape[0]
    stacked_projection_product = chromosome_state.score_left_hand_matrix @ normalized_genotype_matrix_by_variant.T
    covariate_projection_coordinates = stacked_projection_product[:covariate_count, :]
    raw_covariance_with_phenotype = stacked_projection_product[covariate_count:, :]
    covariance_with_phenotype = raw_covariance_with_phenotype - (
        chromosome_state.adjusted_residual_projection_coordinate_matrix @ covariate_projection_coordinates
    )

    projection_sum_squares = jnp.einsum(
        "ij,ij->j",
        covariate_projection_coordinates,
        covariate_projection_coordinates,
    )
    genotype_residual_sum_squares = jnp.maximum(genotype_sum_squares_compute - projection_sum_squares, 0.0)
    positive_genotype_residual_mask = compute_positive_residual_variance_mask(
        genotype_residual_sum_squares,
        genotype_sum_squares_compute,
        minimum_variance=linear_minimum_variance,
        relative_variance_tolerance=linear_relative_variance_tolerance,
    )
    genotype_residual_sum_squares_inverse = jnp.where(
        positive_genotype_residual_mask,
        jnp.reciprocal(genotype_residual_sum_squares),
        0.0,
    )
    covariance_squared = covariance_with_phenotype * covariance_with_phenotype
    beta = jnp.where(
        positive_genotype_residual_mask[None, :],
        covariance_with_phenotype * genotype_residual_sum_squares_inverse[None, :],
        jnp.nan,
    )
    null_mean_squared_error = chromosome_state.adjusted_residual_sum_squares / chromosome_state.degrees_of_freedom
    positive_null_mean_squared_error_mask = null_mean_squared_error > 0.0
    standard_error = jnp.where(
        positive_genotype_residual_mask[None, :] & positive_null_mean_squared_error_mask[:, None],
        jnp.sqrt(null_mean_squared_error[:, None] * genotype_residual_sum_squares_inverse[None, :]),
        jnp.nan,
    )
    valid_statistic_mask = positive_genotype_residual_mask[None, :] & positive_null_mean_squared_error_mask[:, None]
    chi_squared = jnp.where(
        valid_statistic_mask,
        covariance_squared * genotype_residual_sum_squares_inverse[None, :] / null_mean_squared_error[:, None],
        jnp.nan,
    )
    log10_p_value = jnp.where(
        valid_statistic_mask,
        pvalue.chi_squared_to_log10_p_value(chi_squared),
        jnp.nan,
    )
    return Regenie2MultiLinearChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
    )
