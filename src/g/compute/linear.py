"""Linear-regression kernels for additive association testing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from g import jax_setup  # noqa: F401
from g.models import LinearAssociationChunkResult


@jax.jit
def compute_linear_association_chunk(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix: jax.Array,
) -> LinearAssociationChunkResult:
    """Compute linear association statistics for a chunk of variants.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Continuous phenotype vector.
        genotype_matrix: Mean-imputed genotype matrix.

    Returns:
        Chunk-level linear association statistics.

    """
    sample_count = covariate_matrix.shape[0]
    covariate_parameter_count = covariate_matrix.shape[1]
    degrees_of_freedom = sample_count - covariate_parameter_count - 1

    covariate_crossproduct = covariate_matrix.T @ covariate_matrix
    phenotype_projection = jnp.linalg.solve(covariate_crossproduct, covariate_matrix.T @ phenotype_vector)
    phenotype_residual = phenotype_vector - covariate_matrix @ phenotype_projection
    phenotype_residual_sum_squares = jnp.dot(phenotype_residual, phenotype_residual)

    genotype_projection = jnp.linalg.solve(covariate_crossproduct, covariate_matrix.T @ genotype_matrix)
    genotype_residual = genotype_matrix - covariate_matrix @ genotype_projection
    genotype_residual_sum_squares = jnp.sum(genotype_residual * genotype_residual, axis=0)
    covariance_with_phenotype = genotype_residual.T @ phenotype_residual

    safe_denominator = jnp.where(genotype_residual_sum_squares > 0.0, genotype_residual_sum_squares, jnp.nan)
    beta = covariance_with_phenotype / safe_denominator
    residual_sum_squares = phenotype_residual_sum_squares - beta * covariance_with_phenotype
    residual_sum_squares = jnp.maximum(residual_sum_squares, 0.0)
    residual_variance = residual_sum_squares / degrees_of_freedom
    standard_error = jnp.sqrt(residual_variance / safe_denominator)
    test_statistic = beta / standard_error
    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)

    placeholder_p_values = jnp.full(beta.shape, jnp.nan, dtype=beta.dtype)
    return LinearAssociationChunkResult(
        beta=beta,
        standard_error=standard_error,
        test_statistic=test_statistic,
        p_value=placeholder_p_values,
        valid_mask=valid_mask,
    )


def finalize_linear_p_values(
    linear_result: LinearAssociationChunkResult,
    sample_count: int,
    covariate_parameter_count: int,
) -> LinearAssociationChunkResult:
    """Attach Student-t p-values to a linear association result.

    Args:
        linear_result: JAX-computed linear association result.
        sample_count: Number of aligned samples.
        covariate_parameter_count: Number of covariate columns including intercept.

    Returns:
        Linear association result with populated p-values.

    """
    degrees_of_freedom = sample_count - covariate_parameter_count - 1
    test_statistic = np.asarray(linear_result.test_statistic)
    p_value = stats.t.sf(np.abs(test_statistic), df=degrees_of_freedom) * 2.0
    return LinearAssociationChunkResult(
        beta=linear_result.beta,
        standard_error=linear_result.standard_error,
        test_statistic=linear_result.test_statistic,
        p_value=jnp.asarray(p_value),
        valid_mask=linear_result.valid_mask,
    )
