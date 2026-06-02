"""Linear score-test kernels for REGENIE step 2."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute.common import genotype, pvalue
from g.compute.regenie2_linear import result as regenie2_linear_result


def compute_normalized_genotype_sum_squares_from_stats(
    *,
    genotype_dosage_sum: jax.Array,
    genotype_imputed_dosage_square_sum: jax.Array,
    sample_count: int,
) -> jax.Array:
    """Compute shifted genotype sum of squares from native chunk statistics."""
    dosage_sum_compute = jnp.asarray(genotype_dosage_sum, dtype=jnp.float32)
    imputed_dosage_square_sum_compute = jnp.asarray(genotype_imputed_dosage_square_sum, dtype=jnp.float32)
    sample_count_compute = jnp.asarray(sample_count, dtype=jnp.float32)
    genotype_mean = dosage_sum_compute / sample_count_compute
    genotype_offset = jnp.where(genotype_mean > 1.0, genotype.ALLELE_COUNT_MULTIPLIER, 0.0)
    return (
        imputed_dosage_square_sum_compute
        - 2.0 * genotype_offset * dosage_sum_compute
        + sample_count_compute * genotype_offset * genotype_offset
    )


def compute_regenie2_linear_chunk_trait_major_variant_major(
    *,
    whitened_covariate_transpose: jax.Array,
    adjusted_residual_matrix: jax.Array,
    adjusted_residual_projection_coordinate_matrix: jax.Array,
    adjusted_residual_sum_squares: jax.Array,
    degrees_of_freedom: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    genotype_dosage_sum: jax.Array | None = None,
    genotype_imputed_dosage_square_sum: jax.Array | None = None,
) -> regenie2_linear_result.Regenie2MultiLinearChunkResult:
    """Compute linear score-test statistics for trait-major residuals and variant-major genotypes."""
    normalized_genotype_matrix_by_variant = genotype.normalize_high_frequency_diploid_genotypes_variant_major(
        genotype_matrix_by_variant
    )
    if genotype_dosage_sum is None or genotype_imputed_dosage_square_sum is None:
        genotype_sum_squares_compute = jnp.einsum(
            "ij,ij->i",
            normalized_genotype_matrix_by_variant,
            normalized_genotype_matrix_by_variant,
        )
    else:
        genotype_sum_squares_compute = compute_normalized_genotype_sum_squares_from_stats(
            genotype_dosage_sum=genotype_dosage_sum,
            genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
            sample_count=genotype_matrix_by_variant.shape[1],
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
    return regenie2_linear_result.Regenie2MultiLinearChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        valid_mask=valid_mask,
    )
