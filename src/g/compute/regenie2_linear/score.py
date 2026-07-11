"""Linear score-test kernels for REGENIE step 2."""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp

from g.compute.common import dtype as compute_dtype
from g.compute.common import genotype, pvalue
from g.compute.common import result as association_result

if typing.TYPE_CHECKING:
    from g import types
    from g.compute.regenie2_linear import state as regenie2_linear_state


type Regenie2MultiLinearChunkResult = association_result.AssociationResult[jax.Array, None]


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


@functools.partial(
    jax.jit,
    static_argnames=("score_dtype", "linear_minimum_variance", "linear_relative_variance_tolerance"),
)
def compute_regenie2_linear_chunk_trait_major_variant_major(
    *,
    chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    native_genotype_mean: jax.Array | None,
    genotype_imputed_dosage_square_sum: jax.Array | None,
    score_dtype: types.FloatingPointDtype,
    linear_minimum_variance: float,
    linear_relative_variance_tolerance: float,
) -> Regenie2MultiLinearChunkResult:
    """Compute linear score-test statistics for trait-major residuals and variant-major genotypes."""
    jax_dtype = compute_dtype.resolve_jax_dtype(score_dtype)
    genotype_matrix_by_variant_compute = jnp.asarray(genotype_matrix_by_variant, dtype=jax_dtype)
    genotype_mean = genotype.compute_diploid_genotype_mean(
        genotype_matrix_by_variant_compute,
        native_genotype_mean,
    )
    genotype_offset = jnp.where(genotype_mean > 1.0, genotype.ALLELE_COUNT_MULTIPLIER, 0.0)
    normalized_genotype_matrix_by_variant = genotype_matrix_by_variant_compute - genotype_offset[:, None]
    if native_genotype_mean is None or genotype_imputed_dosage_square_sum is None:
        genotype_sum_squares_compute = jnp.einsum(
            "ij,ij->i",
            normalized_genotype_matrix_by_variant,
            normalized_genotype_matrix_by_variant,
        )
    else:
        sample_count_compute = jnp.asarray(genotype_matrix_by_variant_compute.shape[1], dtype=jax_dtype)
        imputed_dosage_sum_compute = genotype_mean * sample_count_compute
        imputed_dosage_square_sum_compute = jnp.asarray(
            genotype_imputed_dosage_square_sum,
            dtype=jax_dtype,
        )
        genotype_sum_squares_compute = (
            imputed_dosage_square_sum_compute
            - 2.0 * genotype_offset * imputed_dosage_sum_compute
            + sample_count_compute * genotype_offset * genotype_offset
        )
    covariate_count = chromosome_state.adjusted_residual_projection_coordinate_matrix.shape[1]
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
    return association_result.AssociationResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        correction_code=None,
    )
