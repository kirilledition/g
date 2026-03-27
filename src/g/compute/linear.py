"""Linear-regression kernels for additive association testing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc

from g import jax_setup  # noqa: F401
from g.models import LinearAssociationChunkResult, LinearAssociationState


def prepare_linear_association_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> LinearAssociationState:
    """Precompute covariate-only linear regression terms.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Continuous phenotype vector.

    Returns:
        Reusable linear-regression state for genotype chunks.

    """
    covariate_crossproduct = covariate_matrix.T @ covariate_matrix
    covariate_crossproduct_inverse = jnp.linalg.inv(covariate_crossproduct)
    phenotype_projection = covariate_crossproduct_inverse @ (covariate_matrix.T @ phenotype_vector)
    phenotype_residual = phenotype_vector - covariate_matrix @ phenotype_projection
    phenotype_residual_sum_squares = jnp.dot(phenotype_residual, phenotype_residual)
    return LinearAssociationState(
        covariate_matrix=covariate_matrix,
        covariate_crossproduct_inverse=covariate_crossproduct_inverse,
        phenotype_residual=phenotype_residual,
        phenotype_residual_sum_squares=phenotype_residual_sum_squares,
    )


@jax.jit
def compute_linear_association_chunk(
    linear_association_state: LinearAssociationState,
    genotype_matrix: jax.Array,
) -> LinearAssociationChunkResult:
    """Compute linear association statistics for a chunk of variants.

    Args:
        linear_association_state: Precomputed covariate-only linear state.
        genotype_matrix: Mean-imputed genotype matrix.

    Returns:
        Chunk-level linear association statistics.

    """
    covariate_matrix = linear_association_state.covariate_matrix
    sample_count = covariate_matrix.shape[0]
    covariate_parameter_count = covariate_matrix.shape[1]
    degrees_of_freedom = sample_count - covariate_parameter_count - 1

    # ⚡ Bolt Optimization: Use geometric properties of orthogonal projections
    # to compute the sum of squares and covariance without ever materializing
    # the massive N x M genotype_residual matrix in device memory.
    covariate_genotype_product = covariate_matrix.T @ genotype_matrix
    genotype_projection = linear_association_state.covariate_crossproduct_inverse @ covariate_genotype_product
    genotype_residual_sum_squares = jnp.sum(genotype_matrix * genotype_matrix, axis=0) - jnp.sum(
        genotype_projection * covariate_genotype_product, axis=0
    )
    covariance_with_phenotype = genotype_matrix.T @ linear_association_state.phenotype_residual

    safe_denominator = jnp.where(genotype_residual_sum_squares > 0.0, genotype_residual_sum_squares, jnp.nan)
    beta = covariance_with_phenotype / safe_denominator
    residual_sum_squares = linear_association_state.phenotype_residual_sum_squares - beta * covariance_with_phenotype
    residual_sum_squares = jnp.maximum(residual_sum_squares, 0.0)
    residual_variance = residual_sum_squares / degrees_of_freedom
    standard_error = jnp.sqrt(residual_variance / safe_denominator)
    test_statistic = beta / standard_error
    absolute_test_statistic = jnp.abs(test_statistic)
    degrees_of_freedom_value = jnp.asarray(degrees_of_freedom, dtype=beta.dtype)
    beta_inc_argument = degrees_of_freedom_value / (
        degrees_of_freedom_value + absolute_test_statistic * absolute_test_statistic
    )
    p_value = betainc(0.5 * degrees_of_freedom_value, 0.5, beta_inc_argument)
    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)

    return LinearAssociationChunkResult(
        beta=beta,
        standard_error=standard_error,
        test_statistic=test_statistic,
        p_value=p_value,
        valid_mask=valid_mask,
    )
