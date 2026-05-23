"""Line-search helpers for binary Firth correction kernels."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config

FIRTH_STEP_HALVING_SCALE = 0.5


def compute_firth_convergence_mask(
    *,
    current_penalized_log_likelihood: jax.Array,
    candidate_penalized_log_likelihood: jax.Array,
    coefficient_step: jax.Array,
    adjusted_score: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
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
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthBacktrackingResult:
    """Accept the first bounded Firth step that preserves penalized likelihood."""

    def condition_function(state: regenie2_binary_firth_types.FirthBacktrackingState) -> jax.Array:
        return (state.attempt_count < kernel_config.firth_step_halving_maximum_attempts) & (~state.accepted)

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
            next_coefficient_step=state.next_coefficient_step * FIRTH_STEP_HALVING_SCALE,
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
