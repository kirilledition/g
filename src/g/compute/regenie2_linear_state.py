"""Linear state preparation for REGENIE step 2."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute import regenie2_linear_types
from g.compute.common import linalg


def solve_positive_definite_system(
    cholesky_factor: jax.Array,
    right_hand_side: jax.Array,
) -> jax.Array:
    """Solve a positive-definite linear system from its Cholesky factor."""
    return linalg.solve_positive_definite_system(cholesky_factor, right_hand_side)


def prepare_regenie2_linear_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> regenie2_linear_types.Regenie2LinearState:
    """Prepare covariate projection and phenotype residual for REGENIE step 2."""
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
    """Prepare shared covariate projection and trait-major phenotype residuals."""
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
