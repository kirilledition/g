"""Shared Firth utilities for REGENIE step 2 binary kernels."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g import types
from g.compute import regenie2_binary_firth_types, regenie2_binary_types

MINIMUM_PROBABILITY = 1.0e-6
BINARY_CASE_THRESHOLD = 0.5
FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS = 12


def compute_firth_penalized_log_likelihood_from_cholesky(
    probability_vector: jax.Array,
    phenotype_vector: jax.Array,
    information_cholesky_factor: jax.Array,
) -> jax.Array:
    """Compute Firth-penalized log-likelihood from a Cholesky factor."""
    clipped_probability = jnp.clip(probability_vector, MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)
    true_class_probability = jnp.where(phenotype_vector == 1.0, clipped_probability, 1.0 - clipped_probability)
    log_likelihood = jnp.sum(jnp.log(true_class_probability))
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(information_cholesky_factor)))
    cholesky_valid = jnp.all(jnp.isfinite(information_cholesky_factor))
    penalty_term = jnp.where(cholesky_valid, BINARY_CASE_THRESHOLD * log_determinant, -jnp.inf)
    return log_likelihood + penalty_term


def compute_firth_convergence_mask(
    *,
    current_penalized_log_likelihood: jax.Array,
    candidate_penalized_log_likelihood: jax.Array,
    coefficient_step: jax.Array,
    adjusted_score: jax.Array,
    kernel_config: regenie2_binary_types.BinaryKernelConfig,
) -> jax.Array:
    """Return whether an accepted Firth step satisfies convergence tolerances."""
    likelihood_delta = candidate_penalized_log_likelihood - current_penalized_log_likelihood
    finite_mask = (
        jnp.isfinite(current_penalized_log_likelihood)
        & jnp.isfinite(candidate_penalized_log_likelihood)
        & jnp.all(jnp.isfinite(coefficient_step))
        & jnp.all(jnp.isfinite(adjusted_score))
    )
    monotonic_mask = likelihood_delta >= -kernel_config.firth_likelihood_tolerance
    likelihood_tolerance_mask = jnp.abs(likelihood_delta) <= kernel_config.firth_likelihood_tolerance
    coefficient_tolerance_mask = jnp.max(jnp.abs(coefficient_step)) <= kernel_config.firth_coefficient_tolerance
    score_tolerance_mask = jnp.max(jnp.abs(adjusted_score)) <= kernel_config.firth_gradient_tolerance
    return finite_mask & monotonic_mask & likelihood_tolerance_mask & coefficient_tolerance_mask & score_tolerance_mask


def run_firth_step_halving(
    *,
    current_coefficients: jax.Array,
    current_penalized_log_likelihood: jax.Array,
    coefficient_step: jax.Array,
    evaluate_penalized_log_likelihood: typing.Callable[[jax.Array], jax.Array],
    kernel_config: regenie2_binary_types.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthBacktrackingResult:
    """Accept the first bounded Firth step that preserves penalized likelihood."""

    def condition_function(state: regenie2_binary_firth_types.FirthBacktrackingState) -> jax.Array:
        return (state.attempt_count < FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS) & (~state.accepted)

    def body_function(
        state: regenie2_binary_firth_types.FirthBacktrackingState,
    ) -> regenie2_binary_firth_types.FirthBacktrackingState:
        candidate_coefficients = current_coefficients + state.next_coefficient_step
        candidate_penalized_log_likelihood = evaluate_penalized_log_likelihood(candidate_coefficients)
        accepted = (
            jnp.isfinite(current_penalized_log_likelihood)
            & jnp.isfinite(candidate_penalized_log_likelihood)
            & jnp.all(jnp.isfinite(candidate_coefficients))
            & jnp.all(jnp.isfinite(state.next_coefficient_step))
            & (
                candidate_penalized_log_likelihood
                >= current_penalized_log_likelihood - kernel_config.firth_likelihood_tolerance
            )
        )
        return regenie2_binary_firth_types.FirthBacktrackingState(
            attempt_count=state.attempt_count + jnp.asarray(1, dtype=jnp.int32),
            next_coefficient_step=state.next_coefficient_step * BINARY_CASE_THRESHOLD,
            accepted_coefficient_step=jnp.where(
                accepted,
                state.next_coefficient_step,
                state.accepted_coefficient_step,
            ),
            accepted_coefficients=jnp.where(
                accepted,
                candidate_coefficients,
                state.accepted_coefficients,
            ),
            accepted_penalized_log_likelihood=jnp.where(
                accepted,
                candidate_penalized_log_likelihood,
                state.accepted_penalized_log_likelihood,
            ),
            accepted=accepted,
        )

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        regenie2_binary_firth_types.FirthBacktrackingState(
            attempt_count=jnp.asarray(0, dtype=jnp.int32),
            next_coefficient_step=coefficient_step,
            accepted_coefficient_step=jnp.zeros_like(coefficient_step),
            accepted_coefficients=current_coefficients,
            accepted_penalized_log_likelihood=current_penalized_log_likelihood,
            accepted=jnp.asarray(0, dtype=jnp.bool_),
        ),
    )
    exhausted = ~final_state.accepted
    return regenie2_binary_firth_types.FirthBacktrackingResult(
        coefficient_step=final_state.accepted_coefficient_step,
        coefficients=final_state.accepted_coefficients,
        penalized_log_likelihood=final_state.accepted_penalized_log_likelihood,
        accepted=final_state.accepted,
        exhausted=exhausted,
    )


def map_firth_reason_code_to_failure_code(reason_code: jax.Array) -> jax.Array:
    """Map internal Firth termination reasons to public failure labels."""
    return jnp.where(
        reason_code == regenie2_binary_firth_types.FirthConvergenceReason.MAX_ITERATIONS.value,
        types.FirthFailureCode.MAX_ITERATIONS.value,
        jnp.where(
            reason_code == regenie2_binary_firth_types.FirthConvergenceReason.INVALID_STATISTIC.value,
            types.FirthFailureCode.INVALID_STATISTIC.value,
            jnp.where(
                reason_code == regenie2_binary_firth_types.FirthConvergenceReason.NEGATIVE_LRT.value,
                types.FirthFailureCode.INVALID_STATISTIC.value,
                jnp.where(
                    (reason_code == regenie2_binary_firth_types.FirthConvergenceReason.STEP_HALVING_EXHAUSTED.value)
                    | (reason_code == regenie2_binary_firth_types.FirthConvergenceReason.STEP_SIZE_INCREASE.value),
                    types.FirthFailureCode.STEP_HALVING.value,
                    jnp.where(
                        (reason_code == regenie2_binary_firth_types.FirthConvergenceReason.NUMERICAL_FAILURE.value)
                        | (reason_code == regenie2_binary_firth_types.FirthConvergenceReason.NULL_FAILURE.value)
                        | (reason_code == regenie2_binary_firth_types.FirthConvergenceReason.PROBABILITY_FAILURE.value),
                        types.FirthFailureCode.NUMERICAL.value,
                        types.FirthFailureCode.NONE.value,
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
