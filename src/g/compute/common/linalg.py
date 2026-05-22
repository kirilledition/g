"""Linear algebra helpers shared by REGENIE compute kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp


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


def solve_from_positive_definite_matrix(
    positive_definite_matrix: jax.Array,
    right_hand_side: jax.Array,
) -> jax.Array:
    """Solve a positive-definite system from its matrix form.

    Args:
        positive_definite_matrix: Symmetric positive-definite coefficient matrix.
        right_hand_side: Right-hand side vector or matrix.

    Returns:
        Solution to the linear system.

    """
    cholesky_factor = jnp.linalg.cholesky(positive_definite_matrix)
    return solve_positive_definite_system(cholesky_factor, right_hand_side)
