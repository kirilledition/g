"""Covariate-only null Firth solver for REGENIE step 2 binary tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute.common import linalg
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

FIRTH_DEVIANCE_LOG_DETERMINANT_MULTIPLIER = 0.5
NULL_FIRTH_MAXIMUM_CONSECUTIVE_SCORE_INCREASES = 25


def compute_null_firth_components(
    *,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    coefficients: jax.Array,
) -> regenie2_binary_firth_types.NullFirthComponents:
    """Compute REGENIE null Firth score and deviance quantities."""
    linear_predictor = covariate_matrix @ coefficients + loco_offset
    probability_vector = regenie2_binary_logistic.compute_regenie_logistic_probability(linear_predictor)
    weight_vector = probability_vector * (1.0 - probability_vector)
    information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
    information_cholesky_factor = jnp.linalg.cholesky(information_matrix)
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(information_cholesky_factor)))
    deviance = (
        regenie2_binary_logistic.compute_logistic_deviance(
            phenotype_vector,
            probability_vector,
            jnp.ones_like(phenotype_vector, dtype=jnp.bool_),
        )
        - log_determinant
    )
    projected_covariate_matrix = linalg.solve_positive_definite_system(
        information_cholesky_factor,
        covariate_matrix.T,
    ).T
    leverage_vector = weight_vector * jnp.einsum("ij,ij->i", projected_covariate_matrix, covariate_matrix)
    modified_score = covariate_matrix.T @ (
        phenotype_vector
        - probability_vector
        + leverage_vector * (regenie2_binary_config.BINARY_CASE_THRESHOLD - probability_vector)
    )
    valid = (
        jnp.all(jnp.isfinite(coefficients))
        & jnp.all(jnp.isfinite(probability_vector))
        & jnp.all(jnp.isfinite(weight_vector))
        & jnp.all(jnp.isfinite(information_cholesky_factor))
        & jnp.isfinite(deviance)
        & jnp.all(jnp.isfinite(leverage_vector))
        & jnp.all(jnp.isfinite(modified_score))
    )
    return regenie2_binary_firth_types.NullFirthComponents(
        information_cholesky_factor=information_cholesky_factor,
        deviance=deviance,
        modified_score=modified_score,
        valid=valid,
    )


def run_null_firth_line_search(
    *,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    current_coefficients: jax.Array,
    current_deviance: jax.Array,
    coefficient_step: jax.Array,
    maximum_attempts: int | jax.Array,
    step_halving_scale: float | jax.Array,
) -> regenie2_binary_firth_types.NullFirthLineSearchResult:
    """Accept the first null Firth step that decreases penalized deviance."""
    scalar_dtype = current_coefficients.dtype
    maximum_attempt_count = jnp.asarray(maximum_attempts, dtype=jnp.int32)
    step_scale = jnp.asarray(step_halving_scale, dtype=scalar_dtype)

    def should_continue(state: regenie2_binary_firth_types.NullFirthLineSearchState) -> jax.Array:
        return (state.attempt_count < maximum_attempt_count) & (~state.accepted) & state.valid

    def run_iteration(
        state: regenie2_binary_firth_types.NullFirthLineSearchState,
    ) -> regenie2_binary_firth_types.NullFirthLineSearchState:
        candidate_coefficients = state.accepted_coefficients + state.next_coefficient_step
        candidate_components = compute_null_firth_components(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            loco_offset=loco_offset,
            coefficients=candidate_coefficients,
        )
        accepted = candidate_components.valid & (candidate_components.deviance < state.accepted_deviance)
        return regenie2_binary_firth_types.NullFirthLineSearchState(
            attempt_count=state.attempt_count + jnp.asarray(1, dtype=jnp.int32),
            next_coefficient_step=state.next_coefficient_step * step_scale,
            accepted_coefficients=jnp.where(accepted, candidate_coefficients, state.accepted_coefficients),
            accepted_deviance=jnp.where(accepted, candidate_components.deviance, state.accepted_deviance),
            accepted=accepted,
            valid=state.valid & candidate_components.valid,
        )

    final_state = jax.lax.while_loop(
        should_continue,
        run_iteration,
        regenie2_binary_firth_types.NullFirthLineSearchState(
            attempt_count=jnp.asarray(0, dtype=jnp.int32),
            next_coefficient_step=coefficient_step,
            accepted_coefficients=current_coefficients,
            accepted_deviance=current_deviance,
            accepted=jnp.asarray(0, dtype=jnp.bool_),
            valid=jnp.asarray(1, dtype=jnp.bool_),
        ),
    )
    return regenie2_binary_firth_types.NullFirthLineSearchResult(
        coefficients=final_state.accepted_coefficients,
        deviance=final_state.accepted_deviance,
        accepted=final_state.accepted,
        valid=final_state.valid,
    )


def update_null_firth_score_history(
    *,
    state: regenie2_binary_firth_types.NullFirthScoreHistoryState,
    score_maximum: jax.Array,
    converged: bool | jax.Array,
    check_score_increase: bool | jax.Array,
) -> regenie2_binary_firth_types.NullFirthScoreHistoryState:
    """Apply REGENIE's consecutive score-increase transition.

    Args:
        state: Score history before evaluating the current iterate.
        score_maximum: Maximum absolute modified score at the current iterate.
        converged: Whether the current iterate satisfies the convergence rule.
        check_score_increase: Whether exceeding the consecutive-increase limit fails the attempt.

    Returns:
        Score history after accepting convergence or evaluating the increase heuristic.

    """
    converged_value = jnp.asarray(converged, dtype=jnp.bool_)
    check_score_increase_value = jnp.asarray(check_score_increase, dtype=jnp.bool_)
    increased_score_count = jnp.where(
        score_maximum > state.previous_score_maximum,
        state.score_increase_count + jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    evaluate_score_increase = ~converged_value
    score_increase_count = jnp.where(
        evaluate_score_increase,
        increased_score_count,
        state.score_increase_count,
    )
    return regenie2_binary_firth_types.NullFirthScoreHistoryState(
        previous_score_maximum=jnp.where(
            evaluate_score_increase,
            score_maximum,
            state.previous_score_maximum,
        ),
        score_increase_count=score_increase_count,
        failed=state.failed
        | (
            evaluate_score_increase
            & check_score_increase_value
            & (increased_score_count > NULL_FIRTH_MAXIMUM_CONSECUTIVE_SCORE_INCREASES)
        ),
    )


def fit_covariate_only_firth_null_model_once(
    *,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    maximum_iterations: int | jax.Array,
    maximum_step_size: float | jax.Array,
    tolerance: float | jax.Array,
    line_search_maximum_attempts: int | jax.Array,
    line_search_step_halving_scale: float | jax.Array,
    check_score_increase: bool | jax.Array,
) -> regenie2_binary_firth_types.NullFirthFitResult:
    """Run one REGENIE-style covariate-only null Firth attempt."""
    scalar_dtype = covariate_matrix.dtype
    tolerance_value = jnp.asarray(tolerance, dtype=scalar_dtype)
    maximum_step_size_value = jnp.asarray(maximum_step_size, dtype=scalar_dtype)
    maximum_iteration_count = jnp.asarray(maximum_iterations, dtype=jnp.int32)
    line_search_maximum_attempt_count = jnp.asarray(line_search_maximum_attempts, dtype=jnp.int32)
    line_search_step_scale = jnp.asarray(line_search_step_halving_scale, dtype=scalar_dtype)
    check_score_increase_value = jnp.asarray(check_score_increase, dtype=jnp.bool_)

    initial_components = compute_null_firth_components(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        loco_offset=loco_offset,
        coefficients=initial_coefficients,
    )

    def should_continue(state: regenie2_binary_firth_types.NullFirthNewtonRaphsonState) -> jax.Array:
        return (state.iteration_count < maximum_iteration_count) & (~state.converged) & (~state.failed)

    def run_iteration(
        state: regenie2_binary_firth_types.NullFirthNewtonRaphsonState,
    ) -> regenie2_binary_firth_types.NullFirthNewtonRaphsonState:
        components = compute_null_firth_components(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            loco_offset=loco_offset,
            coefficients=state.coefficients,
        )
        updated_iteration_count = state.iteration_count + jnp.asarray(1, dtype=jnp.int32)
        score_maximum = jnp.max(jnp.abs(components.modified_score))
        converged = components.valid & (score_maximum < tolerance_value) & (updated_iteration_count >= 2)
        score_history_state = update_null_firth_score_history(
            state=regenie2_binary_firth_types.NullFirthScoreHistoryState(
                previous_score_maximum=state.previous_score_maximum,
                score_increase_count=state.score_increase_count,
                failed=state.failed,
            ),
            score_maximum=score_maximum,
            converged=converged,
            check_score_increase=check_score_increase_value,
        )

        def finish_without_update(_: None) -> regenie2_binary_firth_types.NullFirthNewtonRaphsonState:
            return regenie2_binary_firth_types.NullFirthNewtonRaphsonState(
                coefficients=state.coefficients,
                deviance=components.deviance,
                converged=converged,
                failed=score_history_state.failed,
                iteration_count=updated_iteration_count,
                previous_score_maximum=score_history_state.previous_score_maximum,
                score_increase_count=score_history_state.score_increase_count,
            )

        def update_coefficients(_: None) -> regenie2_binary_firth_types.NullFirthNewtonRaphsonState:
            coefficient_step = linalg.solve_positive_definite_system(
                components.information_cholesky_factor,
                components.modified_score,
            )
            maximum_coefficient_step = jnp.max(jnp.abs(coefficient_step))
            step_scale = jnp.maximum(maximum_coefficient_step / maximum_step_size_value, 1.0)
            line_search_result = run_null_firth_line_search(
                covariate_matrix=covariate_matrix,
                phenotype_vector=phenotype_vector,
                loco_offset=loco_offset,
                current_coefficients=state.coefficients,
                current_deviance=components.deviance,
                coefficient_step=coefficient_step / step_scale,
                maximum_attempts=line_search_maximum_attempt_count,
                step_halving_scale=line_search_step_scale,
            )
            step_halving_failed = ~line_search_result.accepted
            numerical_failed = (
                (~components.valid) | (~jnp.all(jnp.isfinite(coefficient_step))) | (~line_search_result.valid)
            )
            failed = numerical_failed | score_history_state.failed | step_halving_failed
            return regenie2_binary_firth_types.NullFirthNewtonRaphsonState(
                coefficients=jnp.where(failed, state.coefficients, line_search_result.coefficients),
                deviance=line_search_result.deviance,
                converged=converged,
                failed=failed,
                iteration_count=updated_iteration_count,
                previous_score_maximum=score_history_state.previous_score_maximum,
                score_increase_count=score_history_state.score_increase_count,
            )

        return jax.lax.cond(converged, finish_without_update, update_coefficients, None)

    final_state = jax.lax.while_loop(
        should_continue,
        run_iteration,
        regenie2_binary_firth_types.NullFirthNewtonRaphsonState(
            coefficients=initial_coefficients,
            deviance=initial_components.deviance,
            converged=jnp.asarray(0, dtype=jnp.bool_),
            failed=~initial_components.valid,
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            previous_score_maximum=jnp.asarray(jnp.inf, dtype=scalar_dtype),
            score_increase_count=jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return regenie2_binary_firth_types.NullFirthFitResult(
        coefficients=final_state.coefficients,
        penalized_log_likelihood=jnp.where(
            final_state.converged,
            -FIRTH_DEVIANCE_LOG_DETERMINANT_MULTIPLIER * final_state.deviance,
            jnp.asarray(jnp.nan, dtype=scalar_dtype),
        ),
        converged=final_state.converged,
    )


def fit_covariate_only_firth_null_model(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.NullFirthFitResult:
    """Fit the covariate-only Firth null model and return diagnostics."""
    covariate_matrix_float64 = jnp.asarray(covariate_matrix, dtype=jnp.float64)
    phenotype_vector_float64 = jnp.asarray(phenotype_vector, dtype=jnp.float64)
    loco_offset_float64 = jnp.asarray(loco_offset, dtype=jnp.float64)
    initial_coefficients_float64 = jnp.asarray(initial_coefficients, dtype=jnp.float64)
    zero_start_coefficients = jnp.zeros_like(initial_coefficients_float64).at[0].set(-jnp.mean(loco_offset_float64))

    first_result = fit_covariate_only_firth_null_model_once(
        covariate_matrix=covariate_matrix_float64,
        phenotype_vector=phenotype_vector_float64,
        loco_offset=loco_offset_float64,
        initial_coefficients=initial_coefficients_float64,
        maximum_iterations=kernel_config.null_firth.maximum_iterations,
        maximum_step_size=kernel_config.null_firth.maximum_step_size,
        tolerance=kernel_config.null_firth.gradient_tolerance,
        line_search_maximum_attempts=kernel_config.null_firth.line_search_maximum_attempts,
        line_search_step_halving_scale=kernel_config.null_firth.step_halving_scale,
        check_score_increase=True,
    )
    fallback_maximum_iterations = (
        kernel_config.null_firth.maximum_iterations * kernel_config.null_firth.fallback_iteration_multiplier
    )
    fallback_maximum_step_size = (
        kernel_config.null_firth.maximum_step_size / kernel_config.null_firth.fallback_step_divisor
    )
    maximum_iteration_count = jnp.asarray(kernel_config.null_firth.maximum_iterations, dtype=jnp.int32)
    fallback_maximum_iteration_count = jnp.asarray(fallback_maximum_iterations, dtype=jnp.int32)
    maximum_step_size = jnp.asarray(kernel_config.null_firth.maximum_step_size, dtype=jnp.float64)
    fallback_maximum_step_size_value = jnp.asarray(fallback_maximum_step_size, dtype=jnp.float64)
    tolerance = jnp.asarray(kernel_config.null_firth.gradient_tolerance, dtype=jnp.float64)
    line_search_maximum_attempt_count = jnp.asarray(
        kernel_config.null_firth.line_search_maximum_attempts,
        dtype=jnp.int32,
    )
    line_search_step_scale = jnp.asarray(kernel_config.null_firth.step_halving_scale, dtype=jnp.float64)

    def run_zero_start_standard_attempt(_: None) -> regenie2_binary_firth_types.NullFirthFitResult:
        return fit_covariate_only_firth_null_model_once(
            covariate_matrix=covariate_matrix_float64,
            phenotype_vector=phenotype_vector_float64,
            loco_offset=loco_offset_float64,
            initial_coefficients=zero_start_coefficients,
            maximum_iterations=maximum_iteration_count,
            maximum_step_size=maximum_step_size,
            tolerance=tolerance,
            line_search_maximum_attempts=line_search_maximum_attempt_count,
            line_search_step_halving_scale=line_search_step_scale,
            check_score_increase=True,
        )

    def run_zero_start_extended_attempt(_: None) -> regenie2_binary_firth_types.NullFirthFitResult:
        return fit_covariate_only_firth_null_model_once(
            covariate_matrix=covariate_matrix_float64,
            phenotype_vector=phenotype_vector_float64,
            loco_offset=loco_offset_float64,
            initial_coefficients=zero_start_coefficients,
            maximum_iterations=fallback_maximum_iteration_count,
            maximum_step_size=fallback_maximum_step_size_value,
            tolerance=tolerance,
            line_search_maximum_attempts=line_search_maximum_attempt_count,
            line_search_step_halving_scale=line_search_step_scale,
            check_score_increase=True,
        )

    def run_initial_start_extended_attempt(_: None) -> regenie2_binary_firth_types.NullFirthFitResult:
        return fit_covariate_only_firth_null_model_once(
            covariate_matrix=covariate_matrix_float64,
            phenotype_vector=phenotype_vector_float64,
            loco_offset=loco_offset_float64,
            initial_coefficients=initial_coefficients_float64,
            maximum_iterations=fallback_maximum_iteration_count,
            maximum_step_size=fallback_maximum_step_size_value,
            tolerance=tolerance,
            line_search_maximum_attempts=line_search_maximum_attempt_count,
            line_search_step_halving_scale=line_search_step_scale,
            check_score_increase=False,
        )

    def should_continue(state: regenie2_binary_firth_types.NullFirthFallbackState) -> jax.Array:
        return (~state.selected_result.converged) & (state.next_attempt_index <= jnp.asarray(4, dtype=jnp.int32))

    def run_iteration(
        state: regenie2_binary_firth_types.NullFirthFallbackState,
    ) -> regenie2_binary_firth_types.NullFirthFallbackState:
        attempt_result = jax.lax.switch(
            state.next_attempt_index - jnp.asarray(2, dtype=jnp.int32),
            (
                run_zero_start_standard_attempt,
                run_zero_start_extended_attempt,
                run_initial_start_extended_attempt,
            ),
            None,
        )
        return regenie2_binary_firth_types.NullFirthFallbackState(
            selected_result=attempt_result,
            next_attempt_index=state.next_attempt_index + jnp.asarray(1, dtype=jnp.int32),
        )

    final_fallback_state = jax.lax.while_loop(
        should_continue,
        run_iteration,
        regenie2_binary_firth_types.NullFirthFallbackState(
            selected_result=first_result,
            next_attempt_index=jnp.asarray(2, dtype=jnp.int32),
        ),
    )
    selected_result = final_fallback_state.selected_result
    return regenie2_binary_firth_types.NullFirthFitResult(
        coefficients=selected_result.coefficients,
        penalized_log_likelihood=jnp.where(
            selected_result.converged,
            selected_result.penalized_log_likelihood,
            jnp.asarray(jnp.nan, dtype=jnp.float64),
        ),
        converged=selected_result.converged,
    )
