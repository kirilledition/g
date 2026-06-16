"""Covariate-only null logistic IRLS for REGENIE step 2 binary tests."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import linalg

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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullLogisticFitLoopCarry:
    """Loop carry for covariate-only null logistic IRLS.

    Attributes:
        state: Current fit state.
        covariate_matrix: Covariate design matrix.
        phenotype_vector: Binary phenotype values.
        loco_offset: Per-sample LOCO offset.
        maximum_iterations: Maximum IRLS iterations.
        coefficient_tolerance: Coefficient-update convergence tolerance.
        minimum_probability: Logistic probability clipping floor.
        minimum_variance: Bernoulli and information-matrix variance floor.

    """

    state: NullLogisticFitState
    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    loco_offset: jax.Array
    maximum_iterations: jax.Array
    coefficient_tolerance: jax.Array
    minimum_probability: jax.Array
    minimum_variance: jax.Array


def should_continue_null_logistic_fit(carry: NullLogisticFitLoopCarry) -> jax.Array:
    """Return whether the null logistic IRLS loop should continue."""
    return (carry.state.iteration_count < carry.maximum_iterations) & (~carry.state.converged)


def run_null_logistic_fit_iteration(carry: NullLogisticFitLoopCarry) -> NullLogisticFitLoopCarry:
    """Run one covariate-only null logistic IRLS update."""
    covariate_count = carry.covariate_matrix.shape[1]
    linear_predictor = carry.covariate_matrix @ carry.state.coefficients + carry.loco_offset
    probability = jax.nn.sigmoid(linear_predictor)
    fitted_probability = jnp.clip(
        probability,
        carry.minimum_probability,
        1.0 - carry.minimum_probability,
    )
    weight_vector = jnp.maximum(
        fitted_probability * (1.0 - fitted_probability),
        carry.minimum_variance,
    )
    score_vector = carry.covariate_matrix.T @ (carry.phenotype_vector - fitted_probability)
    information_matrix = (carry.covariate_matrix.T * weight_vector) @ carry.covariate_matrix
    cholesky_factor = jnp.linalg.cholesky(
        information_matrix + jnp.eye(covariate_count, dtype=carry.covariate_matrix.dtype) * carry.minimum_variance
    )
    coefficient_delta = linalg.solve_positive_definite_system(cholesky_factor, score_vector)
    updated_iteration_count = carry.state.iteration_count + jnp.asarray(1, dtype=jnp.int32)
    converged = (updated_iteration_count > 0) & (jnp.max(jnp.abs(coefficient_delta)) <= carry.coefficient_tolerance)
    return NullLogisticFitLoopCarry(
        state=NullLogisticFitState(
            coefficients=carry.state.coefficients + coefficient_delta,
            iteration_count=updated_iteration_count,
            converged=converged,
        ),
        covariate_matrix=carry.covariate_matrix,
        phenotype_vector=carry.phenotype_vector,
        loco_offset=carry.loco_offset,
        maximum_iterations=carry.maximum_iterations,
        coefficient_tolerance=carry.coefficient_tolerance,
        minimum_probability=carry.minimum_probability,
        minimum_variance=carry.minimum_variance,
    )


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
    jax_dtype = covariate_matrix.dtype

    initial_coefficients = jnp.zeros(covariate_count, dtype=jax_dtype)
    final_carry = jax.lax.while_loop(
        should_continue_null_logistic_fit,
        run_null_logistic_fit_iteration,
        NullLogisticFitLoopCarry(
            state=NullLogisticFitState(
                coefficients=initial_coefficients,
                iteration_count=jnp.asarray(0, dtype=jnp.int32),
                converged=jnp.asarray(0, dtype=jnp.bool_),
            ),
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            loco_offset=loco_offset,
            maximum_iterations=jnp.asarray(resolved_maximum_iterations, dtype=jnp.int32),
            coefficient_tolerance=jnp.asarray(kernel_config.null_logistic.coefficient_tolerance, dtype=jax_dtype),
            minimum_probability=jnp.asarray(kernel_config.numerical.minimum_probability, dtype=jax_dtype),
            minimum_variance=jnp.asarray(kernel_config.numerical.minimum_variance, dtype=jax_dtype),
        ),
    )
    return final_carry.state
