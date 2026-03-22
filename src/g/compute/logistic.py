"""Logistic-regression kernels for additive association testing."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from g import jax_setup  # noqa: F401
from g.models import LogisticAssociationChunkResult

MINIMUM_PROBABILITY = 1.0e-12
MINIMUM_WEIGHT = 1.0e-12
INITIAL_RESPONSE_SCALE = 4.863891244002886
MAX_ITERATION_COUNT = 25


class LogisticState(NamedTuple):
    """State container for batched association IRLS."""

    coefficients: jax.Array
    converged_mask: jax.Array
    iteration_count: jax.Array
    previous_log_likelihood: jax.Array
    last_covariate_information_matrix: jax.Array
    last_cross_information_vector: jax.Array
    last_genotype_information: jax.Array


class CovariateOnlyLogisticState(NamedTuple):
    """State container for masked covariate-only IRLS."""

    coefficients: jax.Array
    converged_mask: jax.Array
    iteration_count: jax.Array
    previous_log_likelihood: jax.Array


def compute_covariate_information_matrix(
    covariate_matrix: jax.Array,
    effective_weights: jax.Array,
) -> jax.Array:
    """Compute batched Fisher information matrices for covariates."""
    return jnp.einsum("np,mn,nq->mpq", covariate_matrix, effective_weights, covariate_matrix)


def compute_covariate_score(
    covariate_matrix: jax.Array,
    residual_matrix: jax.Array,
) -> jax.Array:
    """Compute batched covariate score vectors."""
    return jnp.einsum("np,mn->mp", covariate_matrix, residual_matrix)


def assemble_full_information_matrix(
    covariate_information_matrix: jax.Array,
    cross_information_vector: jax.Array,
    genotype_information: jax.Array,
) -> jax.Array:
    """Assemble full batched Fisher information matrices."""
    genotype_information_column = genotype_information[:, None, None]
    top_block = jnp.concatenate([covariate_information_matrix, cross_information_vector[:, :, None]], axis=2)
    bottom_block = jnp.concatenate([cross_information_vector[:, None, :], genotype_information_column], axis=2)
    return jnp.concatenate([top_block, bottom_block], axis=1)


def compute_log_likelihood(probability_matrix: jax.Array, phenotype_matrix: jax.Array) -> jax.Array:
    """Compute batched logistic log-likelihood values."""
    clipped_probability_matrix = jnp.clip(probability_matrix, MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)
    return jnp.sum(
        phenotype_matrix * jnp.log(clipped_probability_matrix)
        + (1.0 - phenotype_matrix) * jnp.log1p(-clipped_probability_matrix),
        axis=1,
    )


def compute_covariate_only_probability_matrix(
    covariate_matrix: jax.Array,
    coefficients: jax.Array,
) -> jax.Array:
    """Compute batched covariate-only logistic probabilities."""
    linear_predictor = jnp.einsum("np,mp->mn", covariate_matrix, coefficients)
    return jnp.clip(jax.nn.sigmoid(linear_predictor), MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)


def compute_probability_matrix(
    covariate_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    coefficients: jax.Array,
) -> jax.Array:
    """Compute batched logistic probabilities for covariates plus genotype."""
    covariate_coefficients = coefficients[:, :-1]
    genotype_coefficients = coefficients[:, -1]
    linear_predictor = (
        jnp.einsum("np,mp->mn", covariate_matrix, covariate_coefficients)
        + genotype_matrix_by_variant * genotype_coefficients[:, None]
    )
    return jnp.clip(jax.nn.sigmoid(linear_predictor), MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)


def initialize_full_model_coefficients(
    covariate_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    phenotype_matrix: jax.Array,
    observation_mask: jax.Array,
) -> jax.Array:
    """Initialize full-model coefficients with pseudo-response regression."""
    pseudo_response = INITIAL_RESPONSE_SCALE * (phenotype_matrix - 0.5)
    masked_pseudo_response = jnp.where(observation_mask, pseudo_response, 0.0)
    observation_mask_float = observation_mask.astype(covariate_matrix.dtype)
    covariate_information_matrix = jnp.einsum(
        "np,mn,nq->mpq",
        covariate_matrix,
        observation_mask_float,
        covariate_matrix,
    )
    cross_information_vector = jnp.einsum(
        "np,mn->mp", covariate_matrix, observation_mask_float * genotype_matrix_by_variant
    )
    genotype_information = jnp.sum(
        observation_mask_float * genotype_matrix_by_variant * genotype_matrix_by_variant, axis=1
    )
    full_information_matrix = assemble_full_information_matrix(
        covariate_information_matrix=covariate_information_matrix,
        cross_information_vector=cross_information_vector,
        genotype_information=genotype_information,
    )
    covariate_score = compute_covariate_score(covariate_matrix, masked_pseudo_response)
    genotype_score = jnp.sum(genotype_matrix_by_variant * masked_pseudo_response, axis=1)
    full_score = jnp.concatenate([covariate_score, genotype_score[:, None]], axis=1)
    return jnp.linalg.solve(full_information_matrix, full_score[:, :, None]).squeeze(-1)


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
        tolerance: Relative log-likelihood convergence tolerance.

    Returns:
        Covariate-only coefficient estimates.

    """
    observation_mask = jnp.ones((1, phenotype_vector.shape[0]), dtype=bool)
    initial_coefficients = jnp.linalg.solve(
        covariate_matrix.T @ covariate_matrix,
        covariate_matrix.T @ (INITIAL_RESPONSE_SCALE * (phenotype_vector - 0.5)),
    )
    return fit_masked_covariate_only_logistic_regression(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        observation_mask=observation_mask,
        initial_coefficients=initial_coefficients,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )[0]


@jax.jit
def fit_masked_covariate_only_logistic_regression(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    observation_mask: jax.Array,
    initial_coefficients: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> jax.Array:
    """Fit covariate-only logistic models for per-variant observation masks.

    Args:
        covariate_matrix: Covariate design matrix.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        observation_mask: Per-variant sample inclusion mask.
        initial_coefficients: Shared initial covariate coefficients.
        max_iterations: Maximum IRLS iterations.
        tolerance: Relative log-likelihood convergence tolerance.

    Returns:
        Batched covariate-only coefficient estimates.

    """
    variant_count = observation_mask.shape[0]
    coefficient_count = covariate_matrix.shape[1]
    phenotype_matrix = jnp.broadcast_to(phenotype_vector[None, :], observation_mask.shape)
    iteration_limit = jnp.minimum(jnp.int32(max_iterations), jnp.int32(MAX_ITERATION_COUNT))
    broadcast_initial_coefficients = jnp.broadcast_to(initial_coefficients[None, :], (variant_count, coefficient_count))
    initial_probability_matrix = compute_covariate_only_probability_matrix(
        covariate_matrix, broadcast_initial_coefficients
    )
    masked_initial_probability_matrix = jnp.where(observation_mask, initial_probability_matrix, 0.5)
    initial_log_likelihood = compute_log_likelihood(masked_initial_probability_matrix, phenotype_matrix)

    def condition_function(state: CovariateOnlyLogisticState) -> jax.Array:
        return (jnp.max(state.iteration_count) < iteration_limit) & jnp.any(~state.converged_mask)

    def body_function(state: CovariateOnlyLogisticState) -> CovariateOnlyLogisticState:
        probability_matrix = compute_covariate_only_probability_matrix(covariate_matrix, state.coefficients)
        masked_residual = jnp.where(observation_mask, probability_matrix - phenotype_matrix, 0.0)
        effective_weights = jnp.where(
            observation_mask, jnp.clip(probability_matrix * (1.0 - probability_matrix), MINIMUM_WEIGHT), 0.0
        )
        score = compute_covariate_score(covariate_matrix, masked_residual)
        information_matrix = compute_covariate_information_matrix(covariate_matrix, effective_weights)
        step = jnp.linalg.solve(information_matrix, score[:, :, None]).squeeze(-1)
        step = jnp.where(state.converged_mask[:, None], 0.0, step)
        updated_coefficients = state.coefficients - step
        updated_probability_matrix = compute_covariate_only_probability_matrix(covariate_matrix, updated_coefficients)
        masked_updated_probability_matrix = jnp.where(observation_mask, updated_probability_matrix, 0.5)
        updated_log_likelihood = compute_log_likelihood(masked_updated_probability_matrix, phenotype_matrix)
        updated_converged_mask = state.converged_mask | (
            jnp.abs(updated_log_likelihood - state.previous_log_likelihood)
            < (tolerance * (0.05 + jnp.abs(updated_log_likelihood)))
        )
        updated_iteration_count = state.iteration_count + (~state.converged_mask).astype(jnp.int32)
        return CovariateOnlyLogisticState(
            coefficients=updated_coefficients,
            converged_mask=updated_converged_mask,
            iteration_count=updated_iteration_count,
            previous_log_likelihood=updated_log_likelihood,
        )

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        CovariateOnlyLogisticState(
            coefficients=broadcast_initial_coefficients,
            converged_mask=jnp.zeros((variant_count,), dtype=bool),
            iteration_count=jnp.zeros((variant_count,), dtype=jnp.int32),
            previous_log_likelihood=initial_log_likelihood,
        ),
    )
    return final_state.coefficients


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
        genotype_matrix: Genotype matrix.
        covariate_only_coefficients: Unused legacy argument retained for API stability.
        max_iterations: Maximum IRLS iterations.
        tolerance: Relative log-likelihood convergence tolerance.

    Returns:
        Chunk-level logistic association statistics.

    """
    observation_mask = jnp.ones(genotype_matrix.T.shape, dtype=bool)
    return compute_logistic_association_chunk_with_mask(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_matrix=genotype_matrix,
        observation_mask=observation_mask,
        covariate_only_coefficients=covariate_only_coefficients,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


@jax.jit
def compute_logistic_association_chunk_with_mask(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix: jax.Array,
    observation_mask: jax.Array,
    covariate_only_coefficients: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> LogisticAssociationChunkResult:
    """Compute batched logistic association statistics with per-variant masks.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        genotype_matrix: Mean-imputed genotype matrix.
        observation_mask: Per-variant sample inclusion mask.
        covariate_only_coefficients: Unused legacy argument retained for API stability.
        max_iterations: Maximum IRLS iterations.
        tolerance: Relative log-likelihood convergence tolerance.

    Returns:
        Chunk-level logistic association statistics.

    """
    genotype_matrix_by_variant = genotype_matrix.T
    variant_count = genotype_matrix_by_variant.shape[0]
    covariate_count = covariate_matrix.shape[1]
    phenotype_matrix = jnp.broadcast_to(phenotype_vector[None, :], observation_mask.shape)
    iteration_limit = jnp.minimum(jnp.int32(max_iterations), jnp.int32(MAX_ITERATION_COUNT))
    del covariate_only_coefficients
    initial_coefficients = initialize_full_model_coefficients(
        covariate_matrix=covariate_matrix,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        phenotype_matrix=phenotype_matrix,
        observation_mask=observation_mask,
    )
    initial_probability_matrix = compute_probability_matrix(
        covariate_matrix, genotype_matrix_by_variant, initial_coefficients
    )
    masked_initial_probability_matrix = jnp.where(observation_mask, initial_probability_matrix, 0.5)
    initial_log_likelihood = compute_log_likelihood(masked_initial_probability_matrix, phenotype_matrix)
    initial_covariate_information_matrix = jnp.broadcast_to(
        jnp.eye(covariate_count, dtype=covariate_matrix.dtype)[None, :, :],
        (variant_count, covariate_count, covariate_count),
    )
    initial_cross_information_vector = jnp.zeros((variant_count, covariate_count), dtype=covariate_matrix.dtype)
    initial_genotype_information = jnp.ones((variant_count,), dtype=covariate_matrix.dtype)

    def condition_function(state: LogisticState) -> jax.Array:
        return (jnp.max(state.iteration_count) < iteration_limit) & jnp.any(~state.converged_mask)

    def body_function(state: LogisticState) -> LogisticState:
        probability_matrix = compute_probability_matrix(
            covariate_matrix, genotype_matrix_by_variant, state.coefficients
        )
        masked_residual = jnp.where(observation_mask, probability_matrix - phenotype_matrix, 0.0)
        effective_weights = jnp.where(
            observation_mask, jnp.clip(probability_matrix * (1.0 - probability_matrix), MINIMUM_WEIGHT), 0.0
        )
        weighted_genotype_matrix = effective_weights * genotype_matrix_by_variant
        covariate_score = compute_covariate_score(covariate_matrix, masked_residual)
        genotype_score = jnp.sum(genotype_matrix_by_variant * masked_residual, axis=1)
        covariate_information_matrix = compute_covariate_information_matrix(covariate_matrix, effective_weights)
        cross_information_vector = jnp.einsum("np,mn->mp", covariate_matrix, weighted_genotype_matrix)
        genotype_information = jnp.sum(weighted_genotype_matrix * genotype_matrix_by_variant, axis=1)
        information_matrix = assemble_full_information_matrix(
            covariate_information_matrix=covariate_information_matrix,
            cross_information_vector=cross_information_vector,
            genotype_information=genotype_information,
        )
        score = jnp.concatenate([covariate_score, genotype_score[:, None]], axis=1)
        step = jnp.linalg.solve(information_matrix, score[:, :, None]).squeeze(-1)
        step = jnp.where(state.converged_mask[:, None], 0.0, step)
        updated_coefficients = state.coefficients - step
        updated_probability_matrix = compute_probability_matrix(
            covariate_matrix, genotype_matrix_by_variant, updated_coefficients
        )
        masked_updated_probability_matrix = jnp.where(observation_mask, updated_probability_matrix, 0.5)
        updated_log_likelihood = compute_log_likelihood(masked_updated_probability_matrix, phenotype_matrix)
        updated_converged_mask = state.converged_mask | (
            jnp.abs(updated_log_likelihood - state.previous_log_likelihood)
            < (tolerance * (0.05 + jnp.abs(updated_log_likelihood)))
        )
        updated_iteration_count = state.iteration_count + (~state.converged_mask).astype(jnp.int32)
        updated_covariate_information_matrix = jnp.where(
            state.converged_mask[:, None, None],
            state.last_covariate_information_matrix,
            covariate_information_matrix,
        )
        updated_cross_information_vector = jnp.where(
            state.converged_mask[:, None],
            state.last_cross_information_vector,
            cross_information_vector,
        )
        updated_genotype_information = jnp.where(
            state.converged_mask,
            state.last_genotype_information,
            genotype_information,
        )
        return LogisticState(
            coefficients=updated_coefficients,
            converged_mask=updated_converged_mask,
            iteration_count=updated_iteration_count,
            previous_log_likelihood=updated_log_likelihood,
            last_covariate_information_matrix=updated_covariate_information_matrix,
            last_cross_information_vector=updated_cross_information_vector,
            last_genotype_information=updated_genotype_information,
        )

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        LogisticState(
            coefficients=initial_coefficients,
            converged_mask=jnp.zeros((variant_count,), dtype=bool),
            iteration_count=jnp.zeros((variant_count,), dtype=jnp.int32),
            previous_log_likelihood=initial_log_likelihood,
            last_covariate_information_matrix=initial_covariate_information_matrix,
            last_cross_information_vector=initial_cross_information_vector,
            last_genotype_information=initial_genotype_information,
        ),
    )

    cross_information_solution = jnp.linalg.solve(
        final_state.last_covariate_information_matrix,
        final_state.last_cross_information_vector[:, :, None],
    ).squeeze(-1)
    schur_complement = final_state.last_genotype_information - jnp.sum(
        final_state.last_cross_information_vector * cross_information_solution,
        axis=1,
    )
    safe_schur_complement = jnp.where(schur_complement > 0.0, schur_complement, jnp.nan)
    beta = final_state.coefficients[:, -1]
    standard_error = jnp.sqrt(1.0 / safe_schur_complement)
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
