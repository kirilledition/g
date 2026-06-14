"""Linear state preparation for REGENIE step 2."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import dtype as compute_dtype
from g.compute.common import linalg

if typing.TYPE_CHECKING:
    from g import types


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2LinearState:
    """Precomputed state for REGENIE step 2 linear association.

    Attributes:
        whitened_covariate_transpose: Cholesky-whitened covariate transpose.
        phenotype_residual: Phenotype residualized against covariates.
        degrees_of_freedom: Null-model residual degrees of freedom.

    """

    whitened_covariate_transpose: jax.Array
    phenotype_residual: jax.Array
    degrees_of_freedom: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2LinearChromosomeState:
    """Chromosome-specific REGENIE step 2 linear state.

    Attributes:
        whitened_covariate_transpose: Cholesky-whitened covariate transpose.
        adjusted_residual: Phenotype residual after covariate residualization and LOCO subtraction.
        adjusted_residual_projection_coordinates: Projection of adjusted residual onto whitened covariates.
        score_left_hand_matrix: Stacked left-hand matrix multiplied by genotype chunks.
        adjusted_residual_sum_squares: Sum of squares after removing the covariate projection.
        degrees_of_freedom: Null-model residual degrees of freedom.

    """

    whitened_covariate_transpose: jax.Array
    adjusted_residual: jax.Array
    adjusted_residual_projection_coordinates: jax.Array
    score_left_hand_matrix: jax.Array
    adjusted_residual_sum_squares: jax.Array
    degrees_of_freedom: jax.Array


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
        whitened_covariate_transpose: Cholesky-whitened covariate transpose.
        adjusted_residual_matrix: Trait-major residuals after covariate residualization and LOCO subtraction.
        adjusted_residual_projection_coordinate_matrix: Per-trait projection onto whitened covariates.
        score_left_hand_matrix: Stacked left-hand matrix multiplied by genotype chunks.
        adjusted_residual_sum_squares: Per-trait sums of squares after removing covariate projections.
        degrees_of_freedom: Null-model residual degrees of freedom.

    """

    whitened_covariate_transpose: jax.Array
    adjusted_residual_matrix: jax.Array
    adjusted_residual_projection_coordinate_matrix: jax.Array
    score_left_hand_matrix: jax.Array
    adjusted_residual_sum_squares: jax.Array
    degrees_of_freedom: jax.Array


def build_multi_linear_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    score_dtype: types.FloatingPointDtype,
) -> Regenie2MultiLinearState:
    """Build shared covariate projection and trait-major phenotype residuals."""
    jax_dtype = compute_dtype.resolve_jax_dtype(score_dtype)
    covariate_matrix_compute = jnp.asarray(covariate_matrix, dtype=jax_dtype)
    phenotype_matrix_compute = jnp.asarray(phenotype_matrix, dtype=jax_dtype)
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
        degrees_of_freedom=jnp.asarray(degrees_of_freedom, dtype=jax_dtype),
    )


def build_single_linear_state_from_multi(
    state: Regenie2MultiLinearState,
) -> Regenie2LinearState:
    """Build a single-trait linear state view from a trait-major state."""
    return Regenie2LinearState(
        whitened_covariate_transpose=state.whitened_covariate_transpose,
        phenotype_residual=state.phenotype_residual_matrix[0],
        degrees_of_freedom=state.degrees_of_freedom,
    )


def build_multi_linear_state_from_single(
    state: Regenie2LinearState,
) -> Regenie2MultiLinearState:
    """Build a trait-major linear state view from a single-trait state."""
    return Regenie2MultiLinearState(
        whitened_covariate_transpose=state.whitened_covariate_transpose,
        phenotype_residual_matrix=state.phenotype_residual[None, :],
        degrees_of_freedom=state.degrees_of_freedom,
    )


def build_single_linear_chromosome_state_from_multi(
    chromosome_state: Regenie2MultiLinearChromosomeState,
) -> Regenie2LinearChromosomeState:
    """Build a single-trait chromosome state view from a trait-major state."""
    adjusted_residual = chromosome_state.adjusted_residual_matrix[0]
    adjusted_residual_projection_coordinates = chromosome_state.adjusted_residual_projection_coordinate_matrix[0]
    return Regenie2LinearChromosomeState(
        whitened_covariate_transpose=chromosome_state.whitened_covariate_transpose,
        adjusted_residual=adjusted_residual,
        adjusted_residual_projection_coordinates=adjusted_residual_projection_coordinates,
        score_left_hand_matrix=jnp.concatenate(
            [
                chromosome_state.whitened_covariate_transpose,
                adjusted_residual[None, :],
            ],
            axis=0,
        ),
        adjusted_residual_sum_squares=chromosome_state.adjusted_residual_sum_squares[0],
        degrees_of_freedom=chromosome_state.degrees_of_freedom,
    )


def build_multi_linear_chromosome_state(
    state: Regenie2MultiLinearState,
    loco_prediction_matrix: jax.Array,
    score_dtype: types.FloatingPointDtype,
) -> Regenie2MultiLinearChromosomeState:
    """Build chromosome-specific trait-major residual state reused across chunks."""
    loco_prediction_matrix_compute = jnp.asarray(
        loco_prediction_matrix,
        dtype=compute_dtype.resolve_jax_dtype(score_dtype),
    )
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
        whitened_covariate_transpose=state.whitened_covariate_transpose,
        adjusted_residual_matrix=adjusted_residual_matrix,
        adjusted_residual_projection_coordinate_matrix=adjusted_residual_projection_coordinate_matrix,
        score_left_hand_matrix=score_left_hand_matrix,
        adjusted_residual_sum_squares=adjusted_residual_sum_squares,
        degrees_of_freedom=state.degrees_of_freedom,
    )
