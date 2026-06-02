"""Covariate-only null logistic IRLS for REGENIE step 2 binary tests."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import linalg
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullLogisticFitState:
    """State for covariate-only null logistic IRLS.

    Attributes:
        coefficients: Current coefficient estimates.
        iteration_count: Number of IRLS updates performed.
        converged: Whether the coefficient update tolerance has been reached.

    """

    coefficients: jax.Array
    iteration_count: jax.Array
    converged: jax.Array


def fit_null_logistic_coefficients(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    maximum_iterations: int | None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> NullLogisticFitState:
    """Fit a covariate-only logistic null model with a fixed LOCO offset."""
    covariate_count = covariate_matrix.shape[1]
    resolved_maximum_iterations = (
        kernel_config.null_logistic.maximum_iterations if maximum_iterations is None else maximum_iterations
    )
    coefficient_tolerance = kernel_config.null_logistic.coefficient_tolerance
    jax_dtype = covariate_matrix.dtype

    def condition_function(state: NullLogisticFitState) -> jax.Array:
        return (state.iteration_count < resolved_maximum_iterations) & (~state.converged)

    def body_function(state: NullLogisticFitState) -> NullLogisticFitState:
        linear_predictor = covariate_matrix @ state.coefficients + loco_offset
        fitted_probability = regenie2_binary_logistic.compute_clipped_logistic_probability(
            linear_predictor,
            kernel_config,
        )
        weight_vector = jnp.maximum(
            fitted_probability * (1.0 - fitted_probability),
            kernel_config.numerical.minimum_variance,
        )
        score_vector = covariate_matrix.T @ (phenotype_vector - fitted_probability)
        information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
        cholesky_factor = jnp.linalg.cholesky(
            information_matrix + jnp.eye(covariate_count, dtype=jax_dtype) * kernel_config.numerical.minimum_variance
        )
        coefficient_delta = linalg.solve_positive_definite_system(cholesky_factor, score_vector)
        updated_iteration_count = state.iteration_count + jnp.asarray(1, dtype=jnp.int32)
        converged = (updated_iteration_count > 0) & (jnp.max(jnp.abs(coefficient_delta)) <= coefficient_tolerance)
        return NullLogisticFitState(
            coefficients=state.coefficients + coefficient_delta,
            iteration_count=updated_iteration_count,
            converged=converged,
        )

    initial_coefficients = jnp.zeros(covariate_count, dtype=jax_dtype)
    return jax.lax.while_loop(
        condition_function,
        body_function,
        NullLogisticFitState(
            coefficients=initial_coefficients,
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            converged=jnp.asarray(0, dtype=jnp.bool_),
        ),
    )
