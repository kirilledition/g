"""Covariate-only null logistic IRLS for REGENIE step 2 binary tests."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import linalg

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config

NULL_LOGISTIC_MAXIMUM_PREDICTOR_STEP = 5.0
NULL_LOGISTIC_LINE_SEARCH_MAXIMUM_ATTEMPTS = 25
NULL_LOGISTIC_LINE_SEARCH_STEP_SCALE = 0.5
NULL_LOGISTIC_SUFFICIENT_DECREASE = 1.0e-4
NULL_LOGISTIC_ROUNDOFF_MULTIPLIER = 8.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullLogisticFitState:
    """State for covariate-only null logistic IRLS.

    Attributes:
        coefficients: Current coefficient estimates.
        iteration_count: Number of IRLS updates performed.
        converged: Whether the coefficient update tolerance has been reached.
        failed: Whether invalid numerics, line search, or the iteration budget prevented convergence.

    """

    coefficients: jax.Array
    iteration_count: jax.Array
    converged: jax.Array
    failed: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullLogisticLineSearchState:
    """Finite, sufficient-decrease search along a safeguarded Newton direction."""

    coefficients: jax.Array
    attempt_count: jax.Array
    step_scale: jax.Array
    accepted: jax.Array


def run_null_logistic_line_search(
    *,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    current_coefficients: jax.Array,
    current_linear_predictor: jax.Array,
    coefficient_step: jax.Array,
    score_vector: jax.Array,
) -> NullLogisticLineSearchState:
    """Halve a bounded Newton direction until its Bernoulli loss decreases."""
    scalar_dtype = current_coefficients.dtype
    label_sign = 1.0 - 2.0 * phenotype_vector
    current_sample_loss = jax.nn.softplus(label_sign * current_linear_predictor)
    predictor_step = covariate_matrix @ coefficient_step
    directional_improvement = score_vector @ coefficient_step
    roundoff_allowance = (
        NULL_LOGISTIC_ROUNDOFF_MULTIPLIER * jnp.finfo(scalar_dtype).eps * jnp.maximum(jnp.sum(current_sample_loss), 1.0)
    )

    def should_continue(state: NullLogisticLineSearchState) -> jax.Array:
        return (state.attempt_count < NULL_LOGISTIC_LINE_SEARCH_MAXIMUM_ATTEMPTS) & (~state.accepted)

    def run_iteration(state: NullLogisticLineSearchState) -> NullLogisticLineSearchState:
        candidate_coefficients = current_coefficients + state.step_scale * coefficient_step
        candidate_predictor = current_linear_predictor + state.step_scale * predictor_step
        candidate_sample_loss = jax.nn.softplus(label_sign * candidate_predictor)
        loss_difference = jnp.sum(candidate_sample_loss - current_sample_loss)
        required_decrease = NULL_LOGISTIC_SUFFICIENT_DECREASE * state.step_scale * directional_improvement
        accepted = (
            jnp.all(jnp.isfinite(candidate_coefficients))
            & jnp.all(jnp.isfinite(candidate_predictor))
            & jnp.isfinite(loss_difference)
            & jnp.isfinite(roundoff_allowance)
            & jnp.isfinite(directional_improvement)
            & (directional_improvement > 0.0)
            & (loss_difference <= roundoff_allowance - required_decrease)
            & jnp.any(candidate_coefficients != current_coefficients)
        )
        return NullLogisticLineSearchState(
            coefficients=jnp.where(accepted, candidate_coefficients, state.coefficients),
            attempt_count=state.attempt_count + jnp.asarray(1, dtype=jnp.int32),
            step_scale=state.step_scale * NULL_LOGISTIC_LINE_SEARCH_STEP_SCALE,
            accepted=accepted,
        )

    return jax.lax.while_loop(
        should_continue,
        run_iteration,
        NullLogisticLineSearchState(
            coefficients=current_coefficients,
            attempt_count=jnp.asarray(0, dtype=jnp.int32),
            step_scale=jnp.asarray(1.0, dtype=scalar_dtype),
            accepted=jnp.asarray(0, dtype=jnp.bool_),
        ),
    )


def fit_null_logistic_coefficients(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    kernel_config: regenie2_binary_config.BinaryScoreConfig,
) -> NullLogisticFitState:
    """Fit a logistic null with centered offsets and safeguarded Newton updates.

    The first covariate must be the intercept. Convergence requires a finite
    undamped Newton correction within the configured coefficient tolerance;
    step clipping or halving cannot itself establish convergence. Float64
    reductions keep the convergence criterion meaningful for large cohorts.
    """
    covariate_matrix = jnp.asarray(covariate_matrix, dtype=jnp.float64)
    phenotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float64)
    loco_offset = jnp.asarray(loco_offset, dtype=jnp.float64)
    covariate_count = covariate_matrix.shape[1]
    jax_dtype = covariate_matrix.dtype
    maximum_iterations = jnp.asarray(kernel_config.null_logistic.maximum_iterations, dtype=jnp.int32)
    coefficient_tolerance = jnp.asarray(kernel_config.null_logistic.coefficient_tolerance, dtype=jax_dtype)
    minimum_probability = jnp.asarray(kernel_config.numerical.minimum_probability, dtype=jax_dtype)
    minimum_variance = jnp.asarray(kernel_config.numerical.minimum_variance, dtype=jax_dtype)
    offset_mean = jnp.mean(loco_offset)
    centered_offset = loco_offset - offset_mean
    case_count = jnp.sum(phenotype_vector)
    control_count = jnp.sum(1.0 - phenotype_vector)
    valid_inputs = (
        jnp.all(jnp.isfinite(covariate_matrix))
        & jnp.all(jnp.isfinite(phenotype_vector))
        & jnp.all(jnp.isfinite(centered_offset))
        & jnp.all((phenotype_vector == 0.0) | (phenotype_vector == 1.0))
        & jnp.all(covariate_matrix[:, 0] == 1.0)
        & (case_count > 0.0)
        & (control_count > 0.0)
    )

    def condition_function(state: NullLogisticFitState) -> jax.Array:
        return (state.iteration_count < maximum_iterations) & (~state.converged) & (~state.failed)

    def body_function(state: NullLogisticFitState) -> NullLogisticFitState:
        linear_predictor = covariate_matrix @ state.coefficients + centered_offset
        probability = jax.nn.sigmoid(linear_predictor)
        fitted_probability = jnp.clip(probability, minimum_probability, 1.0 - minimum_probability)
        weight_vector = jnp.maximum(fitted_probability * (1.0 - fitted_probability), minimum_variance)
        score_vector = covariate_matrix.T @ (phenotype_vector - fitted_probability)
        information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
        cholesky_factor = jnp.linalg.cholesky(
            information_matrix + jnp.eye(covariate_count, dtype=jax_dtype) * minimum_variance
        )
        coefficient_delta = linalg.solve_positive_definite_system(cholesky_factor, score_vector)
        predictor_delta = covariate_matrix @ coefficient_delta
        updated_iteration_count = state.iteration_count + jnp.asarray(1, dtype=jnp.int32)
        valid_components = (
            jnp.all(jnp.isfinite(linear_predictor))
            & jnp.all(jnp.isfinite(score_vector))
            & jnp.all(jnp.isfinite(cholesky_factor))
            & jnp.all(jnp.isfinite(coefficient_delta))
            & jnp.all(jnp.isfinite(predictor_delta))
        )
        converged = valid_components & (jnp.max(jnp.abs(coefficient_delta)) <= coefficient_tolerance)

        def finish_iteration(_: None) -> NullLogisticFitState:
            return NullLogisticFitState(
                coefficients=jnp.where(converged, state.coefficients + coefficient_delta, state.coefficients),
                iteration_count=updated_iteration_count,
                converged=converged,
                failed=~valid_components,
            )

        def update_coefficients(_: None) -> NullLogisticFitState:
            step_divisor = jnp.maximum(jnp.max(jnp.abs(predictor_delta)) / NULL_LOGISTIC_MAXIMUM_PREDICTOR_STEP, 1.0)
            line_search_result = run_null_logistic_line_search(
                covariate_matrix=covariate_matrix,
                phenotype_vector=phenotype_vector,
                current_coefficients=state.coefficients,
                current_linear_predictor=linear_predictor,
                coefficient_step=coefficient_delta / step_divisor,
                score_vector=score_vector,
            )
            return NullLogisticFitState(
                coefficients=line_search_result.coefficients,
                iteration_count=updated_iteration_count,
                converged=jnp.asarray(0, dtype=jnp.bool_),
                failed=~line_search_result.accepted,
            )

        return jax.lax.cond(
            converged | (~valid_components),
            finish_iteration,
            update_coefficients,
            None,
        )

    initial_intercept = jnp.log(jnp.maximum(case_count, 1.0)) - jnp.log(jnp.maximum(control_count, 1.0))
    initial_coefficients = (
        jnp.zeros(covariate_count, dtype=jax_dtype).at[0].set(jnp.where(valid_inputs, initial_intercept, 0.0))
    )
    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        NullLogisticFitState(
            coefficients=initial_coefficients,
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            converged=jnp.asarray(0, dtype=jnp.bool_),
            failed=~valid_inputs,
        ),
    )
    return NullLogisticFitState(
        coefficients=final_state.coefficients.at[0].add(-jnp.where(jnp.isfinite(offset_mean), offset_mean, 0.0)),
        iteration_count=final_state.iteration_count,
        converged=final_state.converged,
        failed=~final_state.converged,
    )
