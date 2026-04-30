"""REGENIE step 2 linear association kernel with LOCO adjustment."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.stats

from g import models


def solve_positive_definite_system(
    cholesky_factor: jax.Array,
    right_hand_side: jax.Array,
) -> jax.Array:
    """Solve a positive-definite linear system from its Cholesky factor.

    Args:
        cholesky_factor: Lower-triangular Cholesky factor.
        right_hand_side: Right-hand side vector or matrix.

    Returns:
        Solution to the linear system.

    """
    forward_substitution = jax.lax.linalg.triangular_solve(
        cholesky_factor,
        right_hand_side,
        left_side=True,
        lower=True,
    )
    return jax.lax.linalg.triangular_solve(
        cholesky_factor,
        forward_substitution,
        left_side=True,
        lower=True,
        transpose_a=True,
    )


def prepare_regenie2_linear_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> models.Regenie2LinearState:
    """Prepare covariate projection and phenotype residual for REGENIE step 2.

    Residualizes the phenotype against covariates but does NOT subtract
    LOCO predictions (that happens per-chromosome in the chunk function).

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Continuous phenotype vector.

    Returns:
        Reusable state for REGENIE step 2 linear chunk computation.

    """
    covariate_matrix_float32 = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_vector_float32 = jnp.asarray(phenotype_vector, dtype=jnp.float32)
    sample_count = covariate_matrix_float32.shape[0]
    covariate_parameter_count = covariate_matrix_float32.shape[1]
    degrees_of_freedom = sample_count - covariate_parameter_count - 1

    covariate_matrix_transpose = covariate_matrix_float32.T
    covariate_crossproduct = covariate_matrix_transpose @ covariate_matrix_float32
    covariate_crossproduct_cholesky_factor = jnp.linalg.cholesky(covariate_crossproduct)

    phenotype_projection = solve_positive_definite_system(
        covariate_crossproduct_cholesky_factor,
        covariate_matrix_transpose @ phenotype_vector_float32,
    )
    phenotype_residual = phenotype_vector_float32 - covariate_matrix_float32 @ phenotype_projection

    return models.Regenie2LinearState(
        covariate_matrix=covariate_matrix_float32,
        covariate_matrix_transpose=covariate_matrix_transpose,
        covariate_crossproduct_cholesky_factor=covariate_crossproduct_cholesky_factor,
        phenotype_residual=phenotype_residual,
        sample_count=jnp.asarray(sample_count, dtype=jnp.int32),
        degrees_of_freedom=jnp.asarray(degrees_of_freedom, dtype=jnp.float32),
    )


def chi_squared_to_log10_p_value(chi_squared: jax.Array) -> jax.Array:
    """Convert chi-squared statistics to negative log10 p-values.

    Uses the exact relationship ``chi2(df=1) = Z^2`` so the survival function
    can be evaluated through the normal tail in log-space. This stays finite
    for the large statistics that would underflow through ``chi2.logsf``.

    Args:
        chi_squared: Chi-squared statistics (1 df).

    Returns:
        Negative log10 p-values (-log10(p)).

    """
    safe_chi_squared = jnp.maximum(jnp.asarray(chi_squared, dtype=jnp.float32), 0.0)
    log_p_value = jnp.log(2.0) + jax.scipy.stats.norm.logsf(jnp.sqrt(safe_chi_squared))
    return -log_p_value / jnp.log(10.0)


@jax.jit
def prepare_regenie2_linear_chromosome_state(
    state: models.Regenie2LinearState,
    loco_predictions: jax.Array,
) -> models.Regenie2LinearChromosomeState:
    """Prepare chromosome-specific residual state reused across chunks."""
    loco_predictions_float32 = jnp.asarray(loco_predictions, dtype=jnp.float32)
    adjusted_residual = state.phenotype_residual - loco_predictions_float32
    adjusted_residual_sum_squares = jnp.dot(adjusted_residual, adjusted_residual)
    return models.Regenie2LinearChromosomeState(
        covariate_matrix_transpose=state.covariate_matrix_transpose,
        covariate_crossproduct_cholesky_factor=state.covariate_crossproduct_cholesky_factor,
        adjusted_residual=adjusted_residual,
        adjusted_residual_sum_squares=adjusted_residual_sum_squares,
        degrees_of_freedom=state.degrees_of_freedom,
    )


@jax.jit
def compute_regenie2_linear_chunk_from_chromosome_state(
    chromosome_state: models.Regenie2LinearChromosomeState,
    genotype_matrix: jax.Array,
) -> models.Regenie2LinearChunkResult:
    """Compute REGENIE step 2 linear association using chromosome-cached state."""
    covariate_matrix_transpose = chromosome_state.covariate_matrix_transpose
    covariate_genotype_crossproduct = covariate_matrix_transpose @ genotype_matrix
    genotype_projection = solve_positive_definite_system(
        chromosome_state.covariate_crossproduct_cholesky_factor,
        covariate_genotype_crossproduct,
    )

    genotype_sum_squares = jnp.einsum("ij,ij->j", genotype_matrix, genotype_matrix)
    projection_sum_squares = jnp.einsum("ij,ij->j", covariate_genotype_crossproduct, genotype_projection)
    genotype_residual_sum_squares = jnp.maximum(genotype_sum_squares - projection_sum_squares, 0.0)

    covariance_with_phenotype = genotype_matrix.T @ chromosome_state.adjusted_residual
    covariance_squared = covariance_with_phenotype * covariance_with_phenotype

    positive_genotype_residual_mask = genotype_residual_sum_squares > 0.0
    genotype_residual_sum_squares_inverse = jnp.where(
        positive_genotype_residual_mask,
        jnp.reciprocal(genotype_residual_sum_squares),
        0.0,
    )

    beta = jnp.where(
        positive_genotype_residual_mask,
        covariance_with_phenotype * genotype_residual_sum_squares_inverse,
        jnp.nan,
    )

    residual_sum_squares_after = (
        chromosome_state.adjusted_residual_sum_squares
        - covariance_squared * genotype_residual_sum_squares_inverse
    )
    residual_sum_squares_after = jnp.maximum(residual_sum_squares_after, 0.0)
    positive_residual_sum_squares_mask = residual_sum_squares_after > 0.0

    standard_error = jnp.where(
        positive_genotype_residual_mask & positive_residual_sum_squares_mask,
        jnp.sqrt(
            residual_sum_squares_after
            * genotype_residual_sum_squares_inverse
            / chromosome_state.degrees_of_freedom
        ),
        jnp.nan,
    )

    chi_squared = jnp.where(
        positive_genotype_residual_mask & positive_residual_sum_squares_mask,
        covariance_squared
        * genotype_residual_sum_squares_inverse
        * chromosome_state.degrees_of_freedom
        / residual_sum_squares_after,
        0.0,
    )

    log10_p_value = chi_squared_to_log10_p_value(chi_squared)

    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)

    return models.Regenie2LinearChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        valid_mask=valid_mask,
    )


def compute_regenie2_linear_chunk(
    state: models.Regenie2LinearState,
    genotype_matrix: jax.Array,
    loco_predictions: jax.Array,
) -> models.Regenie2LinearChunkResult:
    """Compute REGENIE step 2 linear association for a genotype chunk.

    This implements the REGENIE step 2 score test for quantitative traits:
    1. Subtract LOCO predictions from the covariate-residualized phenotype
    2. Residualize genotypes against covariates
    3. Compute score test statistics

    The test statistic follows a chi-squared distribution with 1 degree of freedom.

    Args:
        state: Precomputed covariate state from prepare_regenie2_linear_state.
        genotype_matrix: Mean-imputed genotype dosage matrix (samples x variants).
        loco_predictions: LOCO predictions for this chromosome (samples,).

    Returns:
        Association statistics for the chunk.

    Mathematical formulation:
        adjusted_residual = phenotype_residual - loco_predictions
        For each variant g:
            genotype_residual = g - X @ (X'X)^-1 @ X' @ g
            beta = (genotype_residual' @ adjusted_residual) / (genotype_residual' @ genotype_residual)
            variance = sigma_e^2 / (genotype_residual' @ genotype_residual)
            chi_squared = beta^2 / variance
            log10_p_value = -log10(chi2_to_p(chi_squared, df=1))

    """
    chromosome_state = prepare_regenie2_linear_chromosome_state(state, loco_predictions)
    return compute_regenie2_linear_chunk_from_chromosome_state(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
    )
