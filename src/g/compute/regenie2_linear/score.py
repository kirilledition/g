"""Linear score-test kernels for REGENIE step 2."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute.common import genotype, pvalue
from g.compute.regenie2_linear import types as regenie2_linear_types


def compute_regenie2_linear_chunk_trait_major_variant_major(
    *,
    whitened_covariate_transpose: jax.Array,
    adjusted_residual_matrix: jax.Array,
    adjusted_residual_projection_coordinate_matrix: jax.Array,
    adjusted_residual_sum_squares: jax.Array,
    degrees_of_freedom: jax.Array,
    genotype_matrix_by_variant: jax.Array,
) -> regenie2_linear_types.Regenie2MultiLinearChunkResult:
    """Compute linear score-test statistics for trait-major residuals and variant-major genotypes."""
    normalized_genotype_matrix_by_variant = genotype.normalize_high_frequency_diploid_genotypes_variant_major(
        genotype_matrix_by_variant
    )
    genotype_sum_squares_compute = jnp.einsum(
        "ij,ij->i",
        normalized_genotype_matrix_by_variant,
        normalized_genotype_matrix_by_variant,
    )
    covariate_projection_coordinates = whitened_covariate_transpose @ normalized_genotype_matrix_by_variant.T
    raw_covariance_with_phenotype = adjusted_residual_matrix @ normalized_genotype_matrix_by_variant.T
    covariance_with_phenotype = raw_covariance_with_phenotype - (
        adjusted_residual_projection_coordinate_matrix @ covariate_projection_coordinates
    )

    projection_sum_squares = jnp.einsum(
        "ij,ij->j",
        covariate_projection_coordinates,
        covariate_projection_coordinates,
    )
    genotype_residual_sum_squares = jnp.maximum(genotype_sum_squares_compute - projection_sum_squares, 0.0)
    positive_genotype_residual_mask = genotype_residual_sum_squares > 0.0
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
    null_mean_squared_error = adjusted_residual_sum_squares / degrees_of_freedom
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
    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    return regenie2_linear_types.Regenie2MultiLinearChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        valid_mask=valid_mask,
    )


def squeeze_single_trait_linear_result(
    result: regenie2_linear_types.Regenie2MultiLinearChunkResult,
) -> regenie2_linear_types.Regenie2LinearChunkResult:
    """Remove the trait axis from a single-trait linear result."""
    return regenie2_linear_types.Regenie2LinearChunkResult(
        beta=result.beta[0],
        standard_error=result.standard_error[0],
        chi_squared=result.chi_squared[0],
        log10_p_value=result.log10_p_value[0],
        valid_mask=result.valid_mask[0],
    )
