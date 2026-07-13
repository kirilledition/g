"""Linear state preparation for REGENIE step 2."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import linalg


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiLinearState:
    """Precomputed state for multi-trait REGENIE step 2 linear association.

    Attributes:
        whitened_covariate_transpose: Cholesky-whitened covariate transpose.
        phenotype_residual_matrix: Trait-major phenotype residuals after covariate projection.
        degrees_of_freedom: Null-model residual degrees of freedom.

    """

    whitened_covariate_transpose: jax.Array
    phenotype_residual_matrix: jax.Array
    degrees_of_freedom: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiLinearChromosomeState:
    """Chromosome-specific multi-trait linear state.

    Attributes:
        adjusted_residual_projection_coordinate_matrix: Per-trait projection onto whitened covariates.
        score_left_hand_matrix: Stacked left-hand matrix multiplied by genotype chunks.
        adjusted_residual_sum_squares: Per-trait sums of squares after removing covariate projections.
        degrees_of_freedom: Null-model residual degrees of freedom.

    """

    adjusted_residual_projection_coordinate_matrix: jax.Array
    score_left_hand_matrix: jax.Array
    adjusted_residual_sum_squares: jax.Array
    degrees_of_freedom: jax.Array


def build_multi_linear_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
) -> Regenie2MultiLinearState:
    """Build shared covariate projection and trait-major phenotype residuals."""
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

    phenotype_projection_matrix = linalg.solve_positive_definite_system(
        covariate_crossproduct_cholesky_factor,
        covariate_matrix_transpose @ phenotype_matrix_compute.T,
    )
    phenotype_residual_matrix = phenotype_matrix_compute - (covariate_matrix_compute @ phenotype_projection_matrix).T

    return Regenie2MultiLinearState(
        whitened_covariate_transpose=whitened_covariate_transpose,
        phenotype_residual_matrix=phenotype_residual_matrix,
        degrees_of_freedom=jnp.asarray(degrees_of_freedom, dtype=jnp.float32),
    )


@jax.jit
def build_multi_linear_chromosome_state(
    state: Regenie2MultiLinearState,
    loco_prediction_matrix: jax.Array,
) -> Regenie2MultiLinearChromosomeState:
    """Build chromosome-specific trait-major residual state reused across chunks."""
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
    score_left_hand_matrix = jnp.concatenate(
        [
            state.whitened_covariate_transpose,
            adjusted_residual_matrix,
        ],
        axis=0,
    )
    return Regenie2MultiLinearChromosomeState(
        adjusted_residual_projection_coordinate_matrix=adjusted_residual_projection_coordinate_matrix,
        score_left_hand_matrix=score_left_hand_matrix,
        adjusted_residual_sum_squares=adjusted_residual_sum_squares,
        degrees_of_freedom=state.degrees_of_freedom,
    )
