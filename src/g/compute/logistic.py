"""Logistic-regression kernels for additive association testing."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from g import jax_setup  # noqa: F401
from g.models import LogisticAssociationChunkResult

MINIMUM_PROBABILITY = 1.0e-9
MINIMUM_WEIGHT = 1.0e-9


class LogisticState(NamedTuple):
    """State container for batched IRLS."""

    coefficients: jax.Array
    converged_mask: jax.Array
    iteration_count: jax.Array


def compute_information_matrix(design_matrix: jax.Array, weights: jax.Array) -> jax.Array:
    """Compute a batch of Fisher information matrices."""
    weighted_design = design_matrix * weights[:, :, None]
    return jnp.einsum("mnd,mne->mde", design_matrix, weighted_design)


@jax.jit
def fit_covariate_only_logistic_regression(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> jax.Array:
    """Fit a logistic model using covariates only.

    Args:
        covariate_matrix: Covariate design matrix.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        max_iterations: Maximum IRLS iterations.
        tolerance: Convergence tolerance for coefficient updates.

    Returns:
        Covariate-only coefficient estimates.

    """
    coefficient_count = covariate_matrix.shape[1]
    initial_coefficients = jnp.zeros((coefficient_count,), dtype=covariate_matrix.dtype)

    def condition_function(state: tuple[jax.Array, int]) -> jax.Array:
        coefficients, iteration_count = state
        return (iteration_count < max_iterations) & jnp.logical_not(has_converged(coefficients))

    def has_converged(coefficients: jax.Array) -> jax.Array:
        eta = covariate_matrix @ coefficients
        mu = jnp.clip(jax.nn.sigmoid(eta), MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)
        weights = jnp.clip(mu * (1.0 - mu), MINIMUM_WEIGHT)
        score = covariate_matrix.T @ (phenotype_vector - mu)
        information_matrix = covariate_matrix.T @ (covariate_matrix * weights[:, None])
        step = jnp.linalg.solve(information_matrix, score)
        return jnp.max(jnp.abs(step)) < tolerance

    def body_function(state: tuple[jax.Array, int]) -> tuple[jax.Array, int]:
        coefficients, iteration_count = state
        eta = covariate_matrix @ coefficients
        mu = jnp.clip(jax.nn.sigmoid(eta), MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)
        weights = jnp.clip(mu * (1.0 - mu), MINIMUM_WEIGHT)
        score = covariate_matrix.T @ (phenotype_vector - mu)
        information_matrix = covariate_matrix.T @ (covariate_matrix * weights[:, None])
        step = jnp.linalg.solve(information_matrix, score)
        return coefficients + step, iteration_count + 1

    fitted_coefficients, _ = jax.lax.while_loop(condition_function, body_function, (initial_coefficients, 0))
    return fitted_coefficients


@jax.jit
def compute_logistic_association_chunk(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix: jax.Array,
    covariate_only_coefficients: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> LogisticAssociationChunkResult:
    """Compute batched logistic association statistics for a chunk of variants.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        genotype_matrix: Mean-imputed genotype matrix.
        covariate_only_coefficients: Starting point from the covariate-only model.
        max_iterations: Maximum IRLS iterations.
        tolerance: Convergence tolerance.

    Returns:
        Chunk-level logistic association statistics.

    """
    sample_count = covariate_matrix.shape[0]
    variant_count = genotype_matrix.shape[1]
    covariate_count = covariate_matrix.shape[1]

    repeated_covariates = jnp.broadcast_to(covariate_matrix[None, :, :], (variant_count, sample_count, covariate_count))
    design_matrix = jnp.concatenate([repeated_covariates, genotype_matrix.T[:, :, None]], axis=2)
    phenotype_matrix = jnp.broadcast_to(phenotype_vector[None, :], (variant_count, sample_count))
    initial_coefficients = jnp.concatenate(
        [
            jnp.broadcast_to(covariate_only_coefficients[None, :], (variant_count, covariate_count)),
            jnp.zeros((variant_count, 1), dtype=covariate_matrix.dtype),
        ],
        axis=1,
    )

    def condition_function(state: LogisticState) -> jax.Array:
        return (jnp.max(state.iteration_count) < max_iterations) & jnp.any(~state.converged_mask)

    def body_function(state: LogisticState) -> LogisticState:
        eta = jnp.einsum("mnd,md->mn", design_matrix, state.coefficients)
        mu = jnp.clip(jax.nn.sigmoid(eta), MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)
        weights = jnp.clip(mu * (1.0 - mu), MINIMUM_WEIGHT)
        score = jnp.einsum("mnd,mn->md", design_matrix, phenotype_matrix - mu)
        information_matrix = compute_information_matrix(design_matrix, weights)
        step = jnp.linalg.solve(information_matrix, score[:, :, None]).squeeze(-1)
        step = jnp.where(state.converged_mask[:, None], 0.0, step)
        updated_coefficients = state.coefficients + step
        updated_converged_mask = state.converged_mask | (jnp.max(jnp.abs(step), axis=1) < tolerance)
        updated_iteration_count = state.iteration_count + (~state.converged_mask).astype(jnp.int32)
        return LogisticState(updated_coefficients, updated_converged_mask, updated_iteration_count)

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        LogisticState(
            coefficients=initial_coefficients,
            converged_mask=jnp.zeros((variant_count,), dtype=bool),
            iteration_count=jnp.zeros((variant_count,), dtype=jnp.int32),
        ),
    )

    eta = jnp.einsum("mnd,md->mn", design_matrix, final_state.coefficients)
    mu = jnp.clip(jax.nn.sigmoid(eta), MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)
    weights = jnp.clip(mu * (1.0 - mu), MINIMUM_WEIGHT)
    information_matrix = compute_information_matrix(design_matrix, weights)
    identity_matrix = jnp.broadcast_to(
        jnp.eye(
            design_matrix.shape[2],
            dtype=design_matrix.dtype,
        ),
        information_matrix.shape,
    )
    covariance_matrix = jnp.linalg.solve(information_matrix, identity_matrix)
    beta = final_state.coefficients[:, -1]
    standard_error = jnp.sqrt(covariance_matrix[:, -1, -1])
    test_statistic = beta / standard_error
    p_value = 2.0 * norm.sf(jnp.abs(test_statistic))
    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)

    return LogisticAssociationChunkResult(
        beta=beta,
        standard_error=standard_error,
        test_statistic=test_statistic,
        p_value=p_value,
        converged_mask=final_state.converged_mask,
        valid_mask=valid_mask,
        iteration_count=final_state.iteration_count,
    )
