"""Linear state preparation for REGENIE step 2."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiLinearState:
    """Precomputed state for multi-trait REGENIE step 2 linear association.

    Attributes:
        whitened_covariate_transpose: Float64 orthonormal covariate basis transpose.
        phenotype_residual_matrix: Float64 trait-major residuals after covariate projection.
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


def project_residuals_at_float64_resolution(
    values: jax.Array,
    orthonormal_covariate_transpose: jax.Array,
) -> jax.Array:
    """Remove covariate components and discard unresolved projection roundoff.

    Args:
        values: Float64 trait-major source values.
        orthonormal_covariate_transpose: Float64 orthonormal covariate basis transpose.

    Returns:
        Explicit residuals, with rows zeroed when their norm is no larger than
        the float64 projection resolution relative to their source norm.

    """
    projection_coordinates = values @ orthonormal_covariate_transpose.T
    residuals = values - projection_coordinates @ orthonormal_covariate_transpose
    source_norms = jnp.linalg.norm(values, axis=1)
    residual_norms = jnp.linalg.norm(residuals, axis=1)
    relative_resolution = max(orthonormal_covariate_transpose.shape) * jnp.finfo(jnp.float64).eps
    resolved_residual_mask = residual_norms > relative_resolution * source_norms
    return jnp.where(resolved_residual_mask[:, None], residuals, 0.0)


def build_multi_linear_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
) -> Regenie2MultiLinearState:
    """Build a stable shared covariate basis and phenotype residuals.

    Raises:
        ValueError: If the design is rank deficient or leaves no residual degrees of freedom.

    """
    covariate_matrix_compute = jnp.asarray(covariate_matrix, dtype=jnp.float64)
    phenotype_matrix_compute = jnp.asarray(phenotype_matrix, dtype=jnp.float64)
    sample_count = covariate_matrix_compute.shape[0]
    covariate_parameter_count = covariate_matrix_compute.shape[1]
    degrees_of_freedom = sample_count - covariate_parameter_count
    if degrees_of_freedom <= 0:
        raise ValueError("Covariate design must leave positive residual degrees of freedom.")

    # Native alignment supplies a leading intercept. Center only in its span;
    # a caller without that intercept must retain the original column space.
    leading_covariate = covariate_matrix_compute[:, 0]
    has_leading_intercept = jnp.all(leading_covariate == leading_covariate[0]) & (leading_covariate[0] != 0.0)
    covariate_offsets = (
        jnp.where(
            has_leading_intercept,
            jnp.mean(covariate_matrix_compute, axis=0),
            0.0,
        )
        .at[0]
        .set(0.0)
    )
    centered_covariate_matrix = covariate_matrix_compute - covariate_offsets[None, :]
    covariate_column_norms = jnp.linalg.norm(centered_covariate_matrix, axis=0)
    normalized_covariate_matrix = centered_covariate_matrix / jnp.where(
        covariate_column_norms > 0.0,
        covariate_column_norms,
        1.0,
    )
    orthonormal_covariate_matrix, triangular_factor = jnp.linalg.qr(normalized_covariate_matrix, mode="reduced")
    singular_values = jnp.linalg.svd(triangular_factor, compute_uv=False)
    rank_tolerance = max(sample_count, covariate_parameter_count) * jnp.finfo(jnp.float64).eps * singular_values[0]
    if not bool(jnp.all(jnp.isfinite(singular_values) & (singular_values > rank_tolerance))):
        raise ValueError("Covariate design must have full column rank after centering and scaling.")
    whitened_covariate_transpose = orthonormal_covariate_matrix.T
    phenotype_residual_matrix = project_residuals_at_float64_resolution(
        phenotype_matrix_compute,
        whitened_covariate_transpose,
    )

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
    loco_prediction_matrix_compute = jnp.asarray(loco_prediction_matrix, dtype=jnp.float64)
    adjusted_residual_matrix = state.phenotype_residual_matrix - loco_prediction_matrix_compute
    projected_adjusted_residual_matrix = project_residuals_at_float64_resolution(
        adjusted_residual_matrix,
        state.whitened_covariate_transpose,
    )
    # The explicit residual norm avoids cancellation when LOCO contains large
    # covariate components. Retain float64 until this preparation is complete.
    adjusted_residual_sum_squares = jnp.einsum(
        "ij,ij->i",
        projected_adjusted_residual_matrix,
        projected_adjusted_residual_matrix,
    )
    score_covariate_transpose = jnp.asarray(state.whitened_covariate_transpose, dtype=jnp.float32)
    score_adjusted_residual_matrix = jnp.asarray(projected_adjusted_residual_matrix, dtype=jnp.float32)
    adjusted_residual_projection_coordinate_matrix = score_adjusted_residual_matrix @ score_covariate_transpose.T
    score_left_hand_matrix = jnp.concatenate(
        [
            score_covariate_transpose,
            score_adjusted_residual_matrix,
        ],
        axis=0,
    )
    return Regenie2MultiLinearChromosomeState(
        adjusted_residual_projection_coordinate_matrix=adjusted_residual_projection_coordinate_matrix,
        score_left_hand_matrix=score_left_hand_matrix,
        adjusted_residual_sum_squares=jnp.asarray(adjusted_residual_sum_squares, dtype=jnp.float32),
        degrees_of_freedom=state.degrees_of_freedom,
    )
