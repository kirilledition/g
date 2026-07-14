"""Linear algebra helpers shared by REGENIE compute kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_positive_residual_variance_mask(
    variance: jax.Array,
    reference_sum_squares: jax.Array,
    minimum_variance: float,
    relative_variance_tolerance: float,
) -> jax.Array:
    """Return a stable positive-variance mask after covariate projection.

    Args:
        variance: Residualized score-test variance.
        reference_sum_squares: Pre-projection weighted genotype sum of squares.
        minimum_variance: Absolute variance floor.
        relative_variance_tolerance: Relative variance floor multiplier.

    Returns:
        Boolean mask for numerically usable residual variance.

    """
    variance_floor = jnp.maximum(
        minimum_variance,
        reference_sum_squares * relative_variance_tolerance,
    )
    return variance > variance_floor


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
