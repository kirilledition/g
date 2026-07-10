"""Covariate-only null Firth solver for REGENIE step 2 binary tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute.common import linalg
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

FIRTH_DEVIANCE_LOG_DETERMINANT_MULTIPLIER = 0.5


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


def should_continue_null_firth_line_search(
    carry: regenie2_binary_firth_types.NullFirthLineSearchLoopCarry,
) -> jax.Array:
    """Return whether null Firth line search should evaluate another step."""
    return (carry.state.attempt_count < carry.maximum_attempts) & (~carry.state.accepted) & carry.state.valid


def run_null_firth_line_search_iteration(
    carry: regenie2_binary_firth_types.NullFirthLineSearchLoopCarry,
) -> regenie2_binary_firth_types.NullFirthLineSearchLoopCarry:
    """Run one null Firth step-halving attempt."""
    candidate_coefficients = carry.current_coefficients + carry.state.next_coefficient_step
    candidate_components = compute_null_firth_components(
        covariate_matrix=carry.covariate_matrix,
        phenotype_vector=carry.phenotype_vector,
        loco_offset=carry.loco_offset,
        coefficients=candidate_coefficients,
    )
    accepted = candidate_components.valid & (candidate_components.deviance < carry.current_deviance)
    return regenie2_binary_firth_types.NullFirthLineSearchLoopCarry(
        state=regenie2_binary_firth_types.NullFirthLineSearchState(
            attempt_count=carry.state.attempt_count + jnp.asarray(1, dtype=jnp.int32),
            next_coefficient_step=carry.state.next_coefficient_step * carry.step_halving_scale,
            accepted_coefficients=jnp.where(
                accepted,
                candidate_coefficients,
                carry.state.accepted_coefficients,
            ),
            accepted_deviance=jnp.where(
                accepted,
                candidate_components.deviance,
                carry.state.accepted_deviance,
            ),
            accepted=accepted,
            valid=carry.state.valid & candidate_components.valid,
        ),
        covariate_matrix=carry.covariate_matrix,
        phenotype_vector=carry.phenotype_vector,
        loco_offset=carry.loco_offset,
        current_coefficients=carry.current_coefficients,
        current_deviance=carry.current_deviance,
        maximum_attempts=carry.maximum_attempts,
        step_halving_scale=carry.step_halving_scale,
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
    final_carry = jax.lax.while_loop(
        should_continue_null_firth_line_search,
        run_null_firth_line_search_iteration,
        regenie2_binary_firth_types.NullFirthLineSearchLoopCarry(
            state=regenie2_binary_firth_types.NullFirthLineSearchState(
                attempt_count=jnp.asarray(0, dtype=jnp.int32),
                next_coefficient_step=coefficient_step,
                accepted_coefficients=current_coefficients,
                accepted_deviance=current_deviance,
                accepted=jnp.asarray(0, dtype=jnp.bool_),
                valid=jnp.asarray(1, dtype=jnp.bool_),
            ),
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            loco_offset=loco_offset,
            current_coefficients=current_coefficients,
            current_deviance=current_deviance,
            maximum_attempts=jnp.asarray(maximum_attempts, dtype=jnp.int32),
            step_halving_scale=jnp.asarray(step_halving_scale, dtype=scalar_dtype),
        ),
    )
    return regenie2_binary_firth_types.NullFirthLineSearchResult(
        coefficients=final_carry.state.accepted_coefficients,
        deviance=final_carry.state.accepted_deviance,
        accepted=final_carry.state.accepted,
        valid=final_carry.state.valid,
    )


def should_continue_null_firth_newton_raphson(
    carry: regenie2_binary_firth_types.NullFirthNewtonRaphsonLoopCarry,
) -> jax.Array:
    """Return whether the null Firth Newton-Raphson loop should continue."""
    state = carry.state
    return (state.iteration_count < carry.maximum_iterations) & (~state.converged) & (~state.failed)


def run_null_firth_newton_raphson_iteration(
    carry: regenie2_binary_firth_types.NullFirthNewtonRaphsonLoopCarry,
) -> regenie2_binary_firth_types.NullFirthNewtonRaphsonLoopCarry:
    """Run one null Firth Newton-Raphson iteration."""
    state = carry.state
    components = compute_null_firth_components(
        covariate_matrix=carry.covariate_matrix,
        phenotype_vector=carry.phenotype_vector,
        loco_offset=carry.loco_offset,
        coefficients=state.coefficients,
    )
    updated_iteration_count = state.iteration_count + jnp.asarray(1, dtype=jnp.int32)
    score_maximum = jnp.max(jnp.abs(components.modified_score))
    converged = components.valid & (score_maximum < carry.tolerance) & (updated_iteration_count >= 2)
    score_increased = score_maximum > state.previous_score_maximum
    score_increase_count = jnp.where(
        score_increased,
        state.score_increase_count + jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    score_increase_failed = carry.check_score_increase & (score_increase_count > 25)
    coefficient_step = linalg.solve_positive_definite_system(
        components.information_cholesky_factor,
        components.modified_score,
    )
    maximum_coefficient_step = jnp.max(jnp.abs(coefficient_step))
    step_scale = jnp.maximum(maximum_coefficient_step / carry.maximum_step_size, 1.0)
    scaled_coefficient_step = coefficient_step / step_scale
    line_search_result = run_null_firth_line_search(
        covariate_matrix=carry.covariate_matrix,
        phenotype_vector=carry.phenotype_vector,
        loco_offset=carry.loco_offset,
        current_coefficients=state.coefficients,
        current_deviance=components.deviance,
        coefficient_step=scaled_coefficient_step,
        maximum_attempts=carry.line_search_maximum_attempts,
        step_halving_scale=carry.line_search_step_halving_scale,
    )
    step_halving_failed = (~converged) & (~line_search_result.accepted)
    numerical_failed = (
        (~components.valid) | (~jnp.all(jnp.isfinite(coefficient_step))) | ((~converged) & (~line_search_result.valid))
    )
    failed = numerical_failed | score_increase_failed | step_halving_failed
    return regenie2_binary_firth_types.NullFirthNewtonRaphsonLoopCarry(
        state=regenie2_binary_firth_types.NullFirthNewtonRaphsonState(
            coefficients=jnp.where(converged | failed, state.coefficients, line_search_result.coefficients),
            deviance=jnp.where(converged, components.deviance, line_search_result.deviance),
            converged=converged & (~failed),
            failed=failed,
            iteration_count=updated_iteration_count,
            previous_score_maximum=jnp.where(
                score_maximum < state.previous_score_maximum,
                score_maximum,
                state.previous_score_maximum,
            ),
            score_increase_count=score_increase_count,
        ),
        covariate_matrix=carry.covariate_matrix,
        phenotype_vector=carry.phenotype_vector,
        loco_offset=carry.loco_offset,
        maximum_iterations=carry.maximum_iterations,
        maximum_step_size=carry.maximum_step_size,
        tolerance=carry.tolerance,
        line_search_maximum_attempts=carry.line_search_maximum_attempts,
        line_search_step_halving_scale=carry.line_search_step_halving_scale,
        check_score_increase=carry.check_score_increase,
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

    initial_components = compute_null_firth_components(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        loco_offset=loco_offset,
        coefficients=initial_coefficients,
    )
    final_carry = jax.lax.while_loop(
        should_continue_null_firth_newton_raphson,
        run_null_firth_newton_raphson_iteration,
        regenie2_binary_firth_types.NullFirthNewtonRaphsonLoopCarry(
            state=regenie2_binary_firth_types.NullFirthNewtonRaphsonState(
                coefficients=initial_coefficients,
                deviance=initial_components.deviance,
                converged=jnp.asarray(0, dtype=jnp.bool_),
                failed=~initial_components.valid,
                iteration_count=jnp.asarray(0, dtype=jnp.int32),
                previous_score_maximum=jnp.asarray(jnp.inf, dtype=scalar_dtype),
                score_increase_count=jnp.asarray(0, dtype=jnp.int32),
            ),
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            loco_offset=loco_offset,
            maximum_iterations=jnp.asarray(maximum_iterations, dtype=jnp.int32),
            maximum_step_size=maximum_step_size_value,
            tolerance=tolerance_value,
            line_search_maximum_attempts=jnp.asarray(line_search_maximum_attempts, dtype=jnp.int32),
            line_search_step_halving_scale=jnp.asarray(line_search_step_halving_scale, dtype=scalar_dtype),
            check_score_increase=jnp.asarray(check_score_increase, dtype=jnp.bool_),
        ),
    )
    final_state = final_carry.state
    return regenie2_binary_firth_types.NullFirthFitResult(
        coefficients=final_state.coefficients,
        penalized_log_likelihood=jnp.where(
            final_state.converged,
            -FIRTH_DEVIANCE_LOG_DETERMINANT_MULTIPLIER * final_state.deviance,
            jnp.asarray(jnp.nan, dtype=scalar_dtype),
        ),
        converged=final_state.converged,
    )


def run_second_null_firth_attempt_from_parameters(
    parameters: regenie2_binary_firth_types.NullFirthFallbackParameters,
) -> regenie2_binary_firth_types.NullFirthFitResult:
    """Run the second null Firth fallback attempt."""
    return fit_covariate_only_firth_null_model_once(
        covariate_matrix=parameters.covariate_matrix,
        phenotype_vector=parameters.phenotype_vector,
        loco_offset=parameters.loco_offset,
        initial_coefficients=parameters.zero_start_coefficients,
        maximum_iterations=parameters.maximum_iterations,
        maximum_step_size=parameters.maximum_step_size,
        tolerance=parameters.tolerance,
        line_search_maximum_attempts=parameters.line_search_maximum_attempts,
        line_search_step_halving_scale=parameters.line_search_step_halving_scale,
        check_score_increase=True,
    )


def run_third_null_firth_attempt_from_parameters(
    parameters: regenie2_binary_firth_types.NullFirthFallbackParameters,
) -> regenie2_binary_firth_types.NullFirthFitResult:
    """Run the third null Firth fallback attempt."""
    return fit_covariate_only_firth_null_model_once(
        covariate_matrix=parameters.covariate_matrix,
        phenotype_vector=parameters.phenotype_vector,
        loco_offset=parameters.loco_offset,
        initial_coefficients=parameters.zero_start_coefficients,
        maximum_iterations=parameters.fallback_maximum_iterations,
        maximum_step_size=parameters.fallback_maximum_step_size,
        tolerance=parameters.tolerance,
        line_search_maximum_attempts=parameters.line_search_maximum_attempts,
        line_search_step_halving_scale=parameters.line_search_step_halving_scale,
        check_score_increase=True,
    )


def run_fourth_null_firth_attempt_from_parameters(
    parameters: regenie2_binary_firth_types.NullFirthFallbackParameters,
) -> regenie2_binary_firth_types.NullFirthFitResult:
    """Run the fourth null Firth fallback attempt."""
    return fit_covariate_only_firth_null_model_once(
        covariate_matrix=parameters.covariate_matrix,
        phenotype_vector=parameters.phenotype_vector,
        loco_offset=parameters.loco_offset,
        initial_coefficients=parameters.initial_coefficients,
        maximum_iterations=parameters.fallback_maximum_iterations,
        maximum_step_size=parameters.fallback_maximum_step_size,
        tolerance=parameters.tolerance,
        line_search_maximum_attempts=parameters.line_search_maximum_attempts,
        line_search_step_halving_scale=parameters.line_search_step_halving_scale,
        check_score_increase=False,
    )


def should_continue_null_firth_fallback_loop(
    carry: regenie2_binary_firth_types.NullFirthFallbackLoopCarry,
) -> jax.Array:
    """Return whether another null Firth fallback attempt is needed."""
    return (~carry.selected_result.converged) & (carry.next_attempt_index <= jnp.asarray(4, dtype=jnp.int32))


def run_null_firth_fallback_loop_iteration(
    carry: regenie2_binary_firth_types.NullFirthFallbackLoopCarry,
) -> regenie2_binary_firth_types.NullFirthFallbackLoopCarry:
    """Run one lazy null Firth fallback attempt."""
    attempt_result = jax.lax.switch(
        carry.next_attempt_index - jnp.asarray(2, dtype=jnp.int32),
        (
            run_second_null_firth_attempt_from_parameters,
            run_third_null_firth_attempt_from_parameters,
            run_fourth_null_firth_attempt_from_parameters,
        ),
        carry.parameters,
    )
    return regenie2_binary_firth_types.NullFirthFallbackLoopCarry(
        parameters=carry.parameters,
        selected_result=attempt_result,
        next_attempt_index=carry.next_attempt_index + jnp.asarray(1, dtype=jnp.int32),
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

    fallback_parameters = regenie2_binary_firth_types.NullFirthFallbackParameters(
        covariate_matrix=covariate_matrix_float64,
        phenotype_vector=phenotype_vector_float64,
        loco_offset=loco_offset_float64,
        initial_coefficients=initial_coefficients_float64,
        zero_start_coefficients=zero_start_coefficients,
        maximum_iterations=jnp.asarray(kernel_config.null_firth.maximum_iterations, dtype=jnp.int32),
        fallback_maximum_iterations=jnp.asarray(fallback_maximum_iterations, dtype=jnp.int32),
        maximum_step_size=jnp.asarray(kernel_config.null_firth.maximum_step_size, dtype=jnp.float64),
        fallback_maximum_step_size=jnp.asarray(fallback_maximum_step_size, dtype=jnp.float64),
        tolerance=jnp.asarray(kernel_config.null_firth.gradient_tolerance, dtype=jnp.float64),
        line_search_maximum_attempts=jnp.asarray(
            kernel_config.null_firth.line_search_maximum_attempts, dtype=jnp.int32
        ),
        line_search_step_halving_scale=jnp.asarray(kernel_config.null_firth.step_halving_scale, dtype=jnp.float64),
    )
    final_fallback_carry = jax.lax.while_loop(
        should_continue_null_firth_fallback_loop,
        run_null_firth_fallback_loop_iteration,
        regenie2_binary_firth_types.NullFirthFallbackLoopCarry(
            parameters=fallback_parameters,
            selected_result=first_result,
            next_attempt_index=jnp.asarray(2, dtype=jnp.int32),
        ),
    )
    selected_result = final_fallback_carry.selected_result
    return regenie2_binary_firth_types.NullFirthFitResult(
        coefficients=selected_result.coefficients,
        penalized_log_likelihood=jnp.where(
            selected_result.converged,
            selected_result.penalized_log_likelihood,
            jnp.asarray(jnp.nan, dtype=jnp.float64),
        ),
        converged=selected_result.converged,
    )
