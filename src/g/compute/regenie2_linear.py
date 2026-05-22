"""REGENIE step 2 linear association kernel with LOCO adjustment."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute import regenie2_linear_score, regenie2_linear_types
from g.compute.common import linalg


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
    return linalg.solve_positive_definite_system(cholesky_factor, right_hand_side)


def prepare_regenie2_linear_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> regenie2_linear_types.Regenie2LinearState:
    """Prepare covariate projection and phenotype residual for REGENIE step 2.

    Residualizes the phenotype against covariates but does NOT subtract
    LOCO predictions (that happens per-chromosome in the chunk function).

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Continuous phenotype vector.

    Returns:
        Reusable state for REGENIE step 2 linear chunk computation.

    """
    covariate_matrix_compute = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_vector_compute = jnp.asarray(phenotype_vector, dtype=jnp.float32)
    sample_count = covariate_matrix_compute.shape[0]
    covariate_parameter_count = covariate_matrix_compute.shape[1]
    degrees_of_freedom = sample_count - covariate_parameter_count

    covariate_matrix_transpose = covariate_matrix_compute.T
    covariate_crossproduct = covariate_matrix_transpose @ covariate_matrix_compute
    covariate_crossproduct_cholesky_factor = jnp.linalg.cholesky(covariate_crossproduct)
    whitened_covariate_transpose = jax.lax.linalg.triangular_solve(
        covariate_crossproduct_cholesky_factor,
        covariate_matrix_transpose,
        left_side=True,
        lower=True,
    )

    phenotype_projection = solve_positive_definite_system(
        covariate_crossproduct_cholesky_factor,
        covariate_matrix_transpose @ phenotype_vector_compute,
    )
    phenotype_residual = phenotype_vector_compute - covariate_matrix_compute @ phenotype_projection

    return regenie2_linear_types.Regenie2LinearState(
        covariate_matrix=covariate_matrix_compute,
        covariate_matrix_transpose=covariate_matrix_transpose,
        covariate_crossproduct_cholesky_factor=covariate_crossproduct_cholesky_factor,
        whitened_covariate_transpose=whitened_covariate_transpose,
        phenotype_residual=phenotype_residual,
        sample_count=jnp.asarray(sample_count, dtype=jnp.int32),
        degrees_of_freedom=jnp.asarray(degrees_of_freedom, dtype=jnp.float32),
    )


def prepare_regenie2_multi_linear_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
) -> regenie2_linear_types.Regenie2MultiLinearState:
    """Prepare shared covariate projection and trait-major phenotype residuals.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_matrix: Trait-major continuous phenotype matrix.

    Returns:
        Reusable state for multi-trait REGENIE step 2 linear computation.

    """
    covariate_matrix_compute = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_matrix_compute = jnp.asarray(phenotype_matrix, dtype=jnp.float32)
    sample_count = covariate_matrix_compute.shape[0]
    covariate_parameter_count = covariate_matrix_compute.shape[1]
    degrees_of_freedom = sample_count - covariate_parameter_count

    covariate_matrix_transpose = covariate_matrix_compute.T
    covariate_crossproduct = covariate_matrix_transpose @ covariate_matrix_compute
    covariate_crossproduct_cholesky_factor = jnp.linalg.cholesky(covariate_crossproduct)
    whitened_covariate_transpose = jax.lax.linalg.triangular_solve(
        covariate_crossproduct_cholesky_factor,
        covariate_matrix_transpose,
        left_side=True,
        lower=True,
    )
    phenotype_projection_matrix = solve_positive_definite_system(
        covariate_crossproduct_cholesky_factor,
        covariate_matrix_transpose @ phenotype_matrix_compute.T,
    )
    phenotype_residual_matrix = phenotype_matrix_compute - (covariate_matrix_compute @ phenotype_projection_matrix).T

    return regenie2_linear_types.Regenie2MultiLinearState(
        covariate_matrix=covariate_matrix_compute,
        covariate_matrix_transpose=covariate_matrix_transpose,
        covariate_crossproduct_cholesky_factor=covariate_crossproduct_cholesky_factor,
        whitened_covariate_transpose=whitened_covariate_transpose,
        phenotype_residual_matrix=phenotype_residual_matrix,
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
    return regenie2_linear_score.chi_squared_to_log10_p_value(chi_squared)


def normalize_high_frequency_diploid_genotypes_sample_major(genotype_matrix: jax.Array) -> jax.Array:
    """Shift high-frequency diploid dosages to avoid float32 cancellation.

    The model includes an intercept, so subtracting a per-variant constant does
    not change the residualized genotype or score statistic. It does keep rare
    reference-allele carriers near zero before float32 matrix products.
    """
    return regenie2_linear_score.normalize_high_frequency_diploid_genotypes_sample_major(genotype_matrix)


def normalize_high_frequency_diploid_genotypes_variant_major(genotype_matrix_by_variant: jax.Array) -> jax.Array:
    """Shift high-frequency diploid dosages to avoid float32 cancellation."""
    return regenie2_linear_score.normalize_high_frequency_diploid_genotypes_variant_major(genotype_matrix_by_variant)


@jax.jit
def prepare_regenie2_linear_chromosome_state(
    state: regenie2_linear_types.Regenie2LinearState,
    loco_predictions: jax.Array,
) -> regenie2_linear_types.Regenie2LinearChromosomeState:
    """Prepare chromosome-specific residual state reused across chunks."""
    loco_predictions_compute = jnp.asarray(loco_predictions, dtype=jnp.float32)
    adjusted_residual = state.phenotype_residual - loco_predictions_compute
    adjusted_residual_projection_coordinates = state.whitened_covariate_transpose @ adjusted_residual
    raw_adjusted_residual_sum_squares = jnp.dot(adjusted_residual, adjusted_residual)
    adjusted_residual_projection_sum_squares = jnp.dot(
        adjusted_residual_projection_coordinates,
        adjusted_residual_projection_coordinates,
    )
    adjusted_residual_sum_squares = jnp.maximum(
        raw_adjusted_residual_sum_squares - adjusted_residual_projection_sum_squares,
        0.0,
    )
    stacked_score_matrix = jnp.concatenate(
        [state.whitened_covariate_transpose, adjusted_residual[None, :]],
        axis=0,
    )
    return regenie2_linear_types.Regenie2LinearChromosomeState(
        covariate_matrix_transpose=state.covariate_matrix_transpose,
        covariate_crossproduct_cholesky_factor=state.covariate_crossproduct_cholesky_factor,
        stacked_score_matrix=stacked_score_matrix,
        adjusted_residual=adjusted_residual,
        adjusted_residual_projection_coordinates=adjusted_residual_projection_coordinates,
        adjusted_residual_sum_squares=adjusted_residual_sum_squares,
        degrees_of_freedom=state.degrees_of_freedom,
    )


@jax.jit
def prepare_regenie2_multi_linear_chromosome_state(
    state: regenie2_linear_types.Regenie2MultiLinearState,
    loco_prediction_matrix: jax.Array,
) -> regenie2_linear_types.Regenie2MultiLinearChromosomeState:
    """Prepare chromosome-specific multi-trait residual state reused across chunks."""
    loco_prediction_matrix_compute = jnp.asarray(loco_prediction_matrix, dtype=jnp.float32)
    adjusted_residual_matrix = state.phenotype_residual_matrix - loco_prediction_matrix_compute
    adjusted_residual_projection_coordinate_matrix = adjusted_residual_matrix @ state.whitened_covariate_transpose.T
    raw_adjusted_residual_sum_squares = jnp.einsum("ij,ij->i", adjusted_residual_matrix, adjusted_residual_matrix)
    adjusted_residual_projection_sum_squares = jnp.einsum(
        "ij,ij->i",
        adjusted_residual_projection_coordinate_matrix,
        adjusted_residual_projection_coordinate_matrix,
    )
    adjusted_residual_sum_squares = jnp.maximum(
        raw_adjusted_residual_sum_squares - adjusted_residual_projection_sum_squares,
        0.0,
    )
    return regenie2_linear_types.Regenie2MultiLinearChromosomeState(
        covariate_matrix_transpose=state.covariate_matrix_transpose,
        covariate_crossproduct_cholesky_factor=state.covariate_crossproduct_cholesky_factor,
        whitened_covariate_transpose=state.whitened_covariate_transpose,
        adjusted_residual_matrix=adjusted_residual_matrix,
        adjusted_residual_projection_coordinate_matrix=adjusted_residual_projection_coordinate_matrix,
        adjusted_residual_sum_squares=adjusted_residual_sum_squares,
        degrees_of_freedom=state.degrees_of_freedom,
    )


@jax.jit
def compute_regenie2_linear_chunk_from_chromosome_state(
    chromosome_state: regenie2_linear_types.Regenie2LinearChromosomeState,
    genotype_matrix: jax.Array,
) -> regenie2_linear_types.Regenie2LinearChunkResult:
    """Compute REGENIE step 2 linear association using chromosome-cached state."""
    multi_result = compute_regenie2_linear_chunk_trait_major_variant_major(
        whitened_covariate_transpose=chromosome_state.stacked_score_matrix[:-1],
        adjusted_residual_matrix=chromosome_state.adjusted_residual[None, :],
        adjusted_residual_projection_coordinate_matrix=chromosome_state.adjusted_residual_projection_coordinates[
            None, :
        ],
        adjusted_residual_sum_squares=chromosome_state.adjusted_residual_sum_squares[None],
        degrees_of_freedom=chromosome_state.degrees_of_freedom,
        genotype_matrix_by_variant=jnp.asarray(genotype_matrix, dtype=jnp.float32).T,
    )
    return squeeze_single_trait_linear_result(multi_result)


@jax.jit
def compute_regenie2_multi_linear_chunk_from_chromosome_state(
    chromosome_state: regenie2_linear_types.Regenie2MultiLinearChromosomeState,
    genotype_matrix: jax.Array,
) -> regenie2_linear_types.Regenie2MultiLinearChunkResult:
    """Compute multi-trait quantitative REGENIE step 2 association."""
    return compute_regenie2_linear_chunk_trait_major_variant_major(
        whitened_covariate_transpose=chromosome_state.whitened_covariate_transpose,
        adjusted_residual_matrix=chromosome_state.adjusted_residual_matrix,
        adjusted_residual_projection_coordinate_matrix=chromosome_state.adjusted_residual_projection_coordinate_matrix,
        adjusted_residual_sum_squares=chromosome_state.adjusted_residual_sum_squares,
        degrees_of_freedom=chromosome_state.degrees_of_freedom,
        genotype_matrix_by_variant=jnp.asarray(genotype_matrix, dtype=jnp.float32).T,
    )


@jax.jit
def compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_linear_types.Regenie2LinearChromosomeState,
    genotype_matrix_by_variant: jax.Array,
) -> regenie2_linear_types.Regenie2LinearChunkResult:
    """Compute quantitative REGENIE step 2 association from variant-major genotypes."""
    multi_result = compute_regenie2_linear_chunk_trait_major_variant_major(
        whitened_covariate_transpose=chromosome_state.stacked_score_matrix[:-1],
        adjusted_residual_matrix=chromosome_state.adjusted_residual[None, :],
        adjusted_residual_projection_coordinate_matrix=chromosome_state.adjusted_residual_projection_coordinates[
            None, :
        ],
        adjusted_residual_sum_squares=chromosome_state.adjusted_residual_sum_squares[None],
        degrees_of_freedom=chromosome_state.degrees_of_freedom,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
    )
    return squeeze_single_trait_linear_result(multi_result)


@jax.jit
def compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_linear_types.Regenie2MultiLinearChromosomeState,
    genotype_matrix_by_variant: jax.Array,
) -> regenie2_linear_types.Regenie2MultiLinearChunkResult:
    """Compute multi-trait quantitative REGENIE step 2 from variant-major genotypes."""
    return compute_regenie2_linear_chunk_trait_major_variant_major(
        whitened_covariate_transpose=chromosome_state.whitened_covariate_transpose,
        adjusted_residual_matrix=chromosome_state.adjusted_residual_matrix,
        adjusted_residual_projection_coordinate_matrix=chromosome_state.adjusted_residual_projection_coordinate_matrix,
        adjusted_residual_sum_squares=chromosome_state.adjusted_residual_sum_squares,
        degrees_of_freedom=chromosome_state.degrees_of_freedom,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
    )


def compute_regenie2_linear_chunk_trait_major_variant_major(
    *,
    whitened_covariate_transpose: jax.Array,
    adjusted_residual_matrix: jax.Array,
    adjusted_residual_projection_coordinate_matrix: jax.Array,
    adjusted_residual_sum_squares: jax.Array,
    degrees_of_freedom: jax.Array,
    genotype_matrix_by_variant: jax.Array,
) -> regenie2_linear_types.Regenie2MultiLinearChunkResult:
    """Compute linear score-test statistics for trait-major residuals and variant-major genotypes.

    Args:
        whitened_covariate_transpose: Cholesky-whitened covariate transpose.
        adjusted_residual_matrix: Trait-major residuals after LOCO adjustment.
        adjusted_residual_projection_coordinate_matrix: Per-trait projection onto whitened covariates.
        adjusted_residual_sum_squares: Per-trait residual sums of squares after covariate projection.
        degrees_of_freedom: Null-model residual degrees of freedom.
        genotype_matrix_by_variant: Variant-major dosage matrix.

    Returns:
        Trait-major association statistics.

    """
    return regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major(
        whitened_covariate_transpose=whitened_covariate_transpose,
        adjusted_residual_matrix=adjusted_residual_matrix,
        adjusted_residual_projection_coordinate_matrix=adjusted_residual_projection_coordinate_matrix,
        adjusted_residual_sum_squares=adjusted_residual_sum_squares,
        degrees_of_freedom=degrees_of_freedom,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
    )


def squeeze_single_trait_linear_result(
    result: regenie2_linear_types.Regenie2MultiLinearChunkResult,
) -> regenie2_linear_types.Regenie2LinearChunkResult:
    """Remove the trait axis from a single-trait linear result.

    Args:
        result: Trait-major result with exactly one trait.

    Returns:
        Single-trait result using the legacy one-dimensional output shape.

    """
    return regenie2_linear_score.squeeze_single_trait_linear_result(result)


def compute_regenie2_linear_chunk(
    state: regenie2_linear_types.Regenie2LinearState,
    genotype_matrix: jax.Array,
    loco_predictions: jax.Array,
) -> regenie2_linear_types.Regenie2LinearChunkResult:
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
            variance = null_mean_squared_error / (genotype_residual' @ genotype_residual)
            chi_squared = beta^2 / variance
            log10_p_value = -log10(chi2_to_p(chi_squared, df=1))

    """
    chromosome_state = prepare_regenie2_linear_chromosome_state(state, loco_predictions)
    return compute_regenie2_linear_chunk_from_chromosome_state(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
    )
