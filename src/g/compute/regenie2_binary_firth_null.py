"""Covariate-only null Firth solver for REGENIE step 2 binary tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute import regenie2_binary_firth_types, regenie2_binary_types
from g.compute.common import linalg

BINARY_CASE_THRESHOLD = 0.5
FIRTH_NULL_MAXIMUM_ITERATIONS = 1000
FIRTH_NULL_FALLBACK_ITERATION_MULTIPLIER = 5
FIRTH_NULL_GRADIENT_TOLERANCE = 50.0e-6
FIRTH_NULL_MAXIMUM_STEP_SIZE = 25.0
FIRTH_NULL_FALLBACK_STEP_DIVISOR = 5.0
FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS = 25
REGENIE_LOGISTIC_MINIMUM_ETA = -30.0
REGENIE_LOGISTIC_MAXIMUM_ETA = 30.0
REGENIE_NUMERICAL_EPSILON = 10.0 * 2.220446049250313e-16


def compute_regenie_logistic_probability(linear_predictor: jax.Array) -> jax.Array:
    """Compute probabilities with REGENIE's glm-style endpoint clipping."""
    epsilon = jnp.asarray(REGENIE_NUMERICAL_EPSILON, dtype=linear_predictor.dtype)
    lower_probability = epsilon / (1.0 + epsilon)
    upper_probability = jnp.reciprocal(1.0 + epsilon)
    return jnp.where(
        linear_predictor > REGENIE_LOGISTIC_MAXIMUM_ETA,
        upper_probability,
        jnp.where(
            linear_predictor < REGENIE_LOGISTIC_MINIMUM_ETA,
            lower_probability,
            jax.nn.sigmoid(linear_predictor),
        ),
    )


def compute_logistic_deviance(
    phenotype_vector: jax.Array,
    probability_vector: jax.Array,
    active_sample_mask: jax.Array,
) -> jax.Array:
    """Compute REGENIE's Bernoulli deviance over active samples."""
    epsilon = jnp.asarray(REGENIE_NUMERICAL_EPSILON, dtype=probability_vector.dtype)
    clipped_probability = jnp.clip(
        probability_vector,
        epsilon / (1.0 + epsilon),
        jnp.reciprocal(1.0 + epsilon),
    )
    negative_log_likelihood = -jnp.where(
        phenotype_vector > BINARY_CASE_THRESHOLD,
        jnp.log(clipped_probability),
        jnp.log1p(-clipped_probability),
    )
    return 2.0 * jnp.sum(jnp.where(active_sample_mask, negative_log_likelihood, 0.0))


def compute_null_firth_components(
    *,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    coefficients: jax.Array,
) -> regenie2_binary_firth_types.NullFirthComponents:
    """Compute REGENIE null Firth score and deviance quantities."""
    linear_predictor = covariate_matrix @ coefficients + loco_offset
    probability_vector = compute_regenie_logistic_probability(linear_predictor)
    weight_vector = probability_vector * (1.0 - probability_vector)
    information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
    information_cholesky_factor = jnp.linalg.cholesky(information_matrix)
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(information_cholesky_factor)))
    deviance = (
        compute_logistic_deviance(
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
        phenotype_vector - probability_vector + leverage_vector * (BINARY_CASE_THRESHOLD - probability_vector)
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
        probability_vector=probability_vector,
        weight_vector=weight_vector,
        information_matrix=information_matrix,
        information_cholesky_factor=information_cholesky_factor,
        deviance=deviance,
        leverage_vector=leverage_vector,
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
) -> regenie2_binary_firth_types.NullFirthLineSearchResult:
    """Accept the first null Firth step that decreases penalized deviance."""

    def condition_function(state: regenie2_binary_firth_types.NullFirthLineSearchState) -> jax.Array:
        return (state.attempt_count < FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS) & (~state.accepted) & state.valid

    def body_function(
        state: regenie2_binary_firth_types.NullFirthLineSearchState,
    ) -> regenie2_binary_firth_types.NullFirthLineSearchState:
        candidate_coefficients = current_coefficients + state.next_coefficient_step
        candidate_components = compute_null_firth_components(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            loco_offset=loco_offset,
            coefficients=candidate_coefficients,
        )
        accepted = candidate_components.valid & (candidate_components.deviance < current_deviance)
        return regenie2_binary_firth_types.NullFirthLineSearchState(
            attempt_count=state.attempt_count + jnp.asarray(1, dtype=jnp.int32),
            next_coefficient_step=state.next_coefficient_step * BINARY_CASE_THRESHOLD,
            accepted_coefficients=jnp.where(accepted, candidate_coefficients, state.accepted_coefficients),
            accepted_deviance=jnp.where(accepted, candidate_components.deviance, state.accepted_deviance),
            accepted=accepted,
            valid=state.valid & candidate_components.valid,
        )

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
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


def fit_covariate_only_firth_null_model_once(
    *,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    maximum_iterations: int,
    maximum_step_size: float,
    tolerance: float,
    check_score_increase: bool,
) -> regenie2_binary_firth_types.NullFirthFitResult:
    """Run one REGENIE-style covariate-only null Firth attempt."""
    scalar_dtype = covariate_matrix.dtype
    tolerance_value = jnp.asarray(tolerance, dtype=scalar_dtype)
    maximum_step_size_value = jnp.asarray(maximum_step_size, dtype=scalar_dtype)

    def condition_function(state: regenie2_binary_firth_types.NullFirthNewtonRaphsonState) -> jax.Array:
        return (state.iteration_count < maximum_iterations) & (~state.converged) & (~state.failed)

    def body_function(
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
        score_increased = score_maximum > state.previous_score_maximum
        score_increase_count = jnp.where(
            score_increased,
            state.score_increase_count + jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        score_increase_failed = check_score_increase & (score_increase_count > 25)
        coefficient_step = linalg.solve_positive_definite_system(
            components.information_cholesky_factor,
            components.modified_score,
        )
        maximum_coefficient_step = jnp.max(jnp.abs(coefficient_step))
        step_scale = jnp.maximum(maximum_coefficient_step / maximum_step_size_value, 1.0)
        scaled_coefficient_step = coefficient_step / step_scale
        line_search_result = run_null_firth_line_search(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            loco_offset=loco_offset,
            current_coefficients=state.coefficients,
            current_deviance=components.deviance,
            coefficient_step=scaled_coefficient_step,
        )
        step_halving_failed = (~converged) & (~line_search_result.accepted)
        numerical_failed = (
            (~components.valid)
            | (~jnp.all(jnp.isfinite(coefficient_step)))
            | ((~converged) & (~line_search_result.valid))
        )
        failed = numerical_failed | score_increase_failed | step_halving_failed
        reason_code = jnp.where(
            score_increase_failed,
            regenie2_binary_firth_types.FirthConvergenceReason.STEP_SIZE_INCREASE.value,
            jnp.where(
                step_halving_failed,
                regenie2_binary_firth_types.FirthConvergenceReason.STEP_HALVING_EXHAUSTED.value,
                jnp.where(
                    numerical_failed,
                    regenie2_binary_firth_types.FirthConvergenceReason.NUMERICAL_FAILURE.value,
                    jnp.where(
                        converged,
                        regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
                        regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return regenie2_binary_firth_types.NullFirthNewtonRaphsonState(
            coefficients=jnp.where(converged | failed, state.coefficients, line_search_result.coefficients),
            deviance=jnp.where(converged, components.deviance, line_search_result.deviance),
            converged=converged & (~failed),
            failed=failed,
            iteration_count=updated_iteration_count,
            termination_reason_code=reason_code,
            previous_score_maximum=jnp.where(
                score_maximum < state.previous_score_maximum, score_maximum, state.previous_score_maximum
            ),
            score_increase_count=score_increase_count,
        )

    initial_components = compute_null_firth_components(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        loco_offset=loco_offset,
        coefficients=initial_coefficients,
    )
    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        regenie2_binary_firth_types.NullFirthNewtonRaphsonState(
            coefficients=initial_coefficients,
            deviance=initial_components.deviance,
            converged=jnp.asarray(0, dtype=jnp.bool_),
            failed=~initial_components.valid,
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            termination_reason_code=jnp.where(
                initial_components.valid,
                regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                regenie2_binary_firth_types.FirthConvergenceReason.NUMERICAL_FAILURE.value,
            ).astype(jnp.int32),
            previous_score_maximum=jnp.asarray(jnp.inf, dtype=scalar_dtype),
            score_increase_count=jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    max_iteration_failure = (~final_state.converged) & (~final_state.failed)
    convergence_reason_code = jnp.where(
        max_iteration_failure,
        regenie2_binary_firth_types.FirthConvergenceReason.MAX_ITERATIONS.value,
        final_state.termination_reason_code,
    ).astype(jnp.int32)
    return regenie2_binary_firth_types.NullFirthFitResult(
        coefficients=final_state.coefficients,
        penalized_log_likelihood=jnp.where(
            final_state.converged,
            -BINARY_CASE_THRESHOLD * final_state.deviance,
            jnp.asarray(jnp.nan, dtype=scalar_dtype),
        ),
        iteration_count=final_state.iteration_count,
        convergence_reason_code=convergence_reason_code,
        converged=final_state.converged,
    )


def fit_covariate_only_firth_null_model(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    kernel_config: regenie2_binary_types.BinaryKernelConfig,
) -> regenie2_binary_firth_types.NullFirthFitResult:
    """Fit the covariate-only Firth null model and return diagnostics."""
    del kernel_config

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
        maximum_iterations=FIRTH_NULL_MAXIMUM_ITERATIONS,
        maximum_step_size=FIRTH_NULL_MAXIMUM_STEP_SIZE,
        tolerance=FIRTH_NULL_GRADIENT_TOLERANCE,
        check_score_increase=True,
    )
    second_result = fit_covariate_only_firth_null_model_once(
        covariate_matrix=covariate_matrix_float64,
        phenotype_vector=phenotype_vector_float64,
        loco_offset=loco_offset_float64,
        initial_coefficients=zero_start_coefficients,
        maximum_iterations=FIRTH_NULL_MAXIMUM_ITERATIONS,
        maximum_step_size=FIRTH_NULL_MAXIMUM_STEP_SIZE,
        tolerance=FIRTH_NULL_GRADIENT_TOLERANCE,
        check_score_increase=True,
    )
    fallback_maximum_iterations = FIRTH_NULL_MAXIMUM_ITERATIONS * FIRTH_NULL_FALLBACK_ITERATION_MULTIPLIER
    fallback_maximum_step_size = FIRTH_NULL_MAXIMUM_STEP_SIZE / FIRTH_NULL_FALLBACK_STEP_DIVISOR
    third_result = fit_covariate_only_firth_null_model_once(
        covariate_matrix=covariate_matrix_float64,
        phenotype_vector=phenotype_vector_float64,
        loco_offset=loco_offset_float64,
        initial_coefficients=zero_start_coefficients,
        maximum_iterations=fallback_maximum_iterations,
        maximum_step_size=fallback_maximum_step_size,
        tolerance=FIRTH_NULL_GRADIENT_TOLERANCE,
        check_score_increase=True,
    )
    fourth_result = fit_covariate_only_firth_null_model_once(
        covariate_matrix=covariate_matrix_float64,
        phenotype_vector=phenotype_vector_float64,
        loco_offset=loco_offset_float64,
        initial_coefficients=initial_coefficients_float64,
        maximum_iterations=fallback_maximum_iterations,
        maximum_step_size=fallback_maximum_step_size,
        tolerance=FIRTH_NULL_GRADIENT_TOLERANCE,
        check_score_increase=False,
    )
    use_first_result = first_result.converged
    use_second_result = (~use_first_result) & second_result.converged
    use_third_result = (~use_first_result) & (~use_second_result) & third_result.converged
    selected_coefficients = jnp.where(
        use_first_result,
        first_result.coefficients,
        jnp.where(
            use_second_result,
            second_result.coefficients,
            jnp.where(use_third_result, third_result.coefficients, fourth_result.coefficients),
        ),
    )
    selected_penalized_log_likelihood = jnp.where(
        use_first_result,
        first_result.penalized_log_likelihood,
        jnp.where(
            use_second_result,
            second_result.penalized_log_likelihood,
            jnp.where(
                use_third_result,
                third_result.penalized_log_likelihood,
                fourth_result.penalized_log_likelihood,
            ),
        ),
    )
    selected_iteration_count = jnp.where(
        use_first_result,
        first_result.iteration_count,
        jnp.where(
            use_second_result,
            second_result.iteration_count,
            jnp.where(use_third_result, third_result.iteration_count, fourth_result.iteration_count),
        ),
    )
    selected_reason_code = jnp.where(
        use_first_result,
        first_result.convergence_reason_code,
        jnp.where(
            use_second_result,
            second_result.convergence_reason_code,
            jnp.where(use_third_result, third_result.convergence_reason_code, fourth_result.convergence_reason_code),
        ),
    ).astype(jnp.int32)
    selected_converged = (
        first_result.converged | second_result.converged | third_result.converged | fourth_result.converged
    )
    return regenie2_binary_firth_types.NullFirthFitResult(
        coefficients=selected_coefficients,
        penalized_log_likelihood=jnp.where(
            selected_converged,
            selected_penalized_log_likelihood,
            jnp.asarray(jnp.nan, dtype=jnp.float64),
        ),
        iteration_count=selected_iteration_count,
        convergence_reason_code=selected_reason_code,
        converged=selected_converged,
    )
