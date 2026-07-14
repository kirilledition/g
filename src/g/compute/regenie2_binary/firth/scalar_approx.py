"""Scalar approximate-Firth solver for REGENIE step 2 binary tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute.common import pvalue
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types


def build_scalar_approximate_firth_solver_parameters(
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.ScalarApproximateFirthSolverParameters:
    """Build explicit scalar approximate-Firth policy operands."""
    pseudo_maximum_iterations = min(
        kernel_config.approximate_firth.maximum_iterations // 2,
        kernel_config.approximate_firth.pseudo_maximum_iterations,
    )
    newton_raphson_maximum_iterations = kernel_config.approximate_firth.maximum_iterations // 2
    return regenie2_binary_firth_types.ScalarApproximateFirthSolverParameters(
        minimum_variance=jnp.asarray(kernel_config.numerical.minimum_variance, dtype=jnp.float64),
        tolerance=jnp.asarray(kernel_config.approximate_firth.gradient_tolerance, dtype=jnp.float64),
        maximum_step_size=jnp.asarray(kernel_config.approximate_firth.maximum_step_size, dtype=jnp.float64),
        pseudo_maximum_iterations=jnp.asarray(pseudo_maximum_iterations, dtype=jnp.int32),
        pseudo_inner_maximum_iterations=jnp.asarray(
            kernel_config.approximate_firth.pseudo_inner_maximum_iterations,
            dtype=jnp.int32,
        ),
        newton_raphson_maximum_iterations=jnp.asarray(newton_raphson_maximum_iterations, dtype=jnp.int32),
        newton_raphson_zero_start_iterations=jnp.asarray(
            kernel_config.approximate_firth.newton_raphson_zero_start_iterations,
            dtype=jnp.int32,
        ),
        line_search_maximum_attempts=jnp.asarray(
            kernel_config.approximate_firth.line_search_maximum_attempts,
            dtype=jnp.int32,
        ),
    )


def compute_scalar_firth_components_with_minimum_variance(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    beta: jax.Array,
    minimum_variance: jax.Array,
) -> regenie2_binary_firth_types.ScalarFirthComponents:
    """Compute scalar approximate-Firth quantities with explicit numeric policy."""
    linear_predictor = offset_vector + genotype_vector * beta
    probability_vector = regenie2_binary_logistic.compute_regenie_logistic_probability(linear_predictor)
    weight_vector = probability_vector * (1.0 - probability_vector)
    active_weight_vector = jnp.where(active_sample_mask, weight_vector, 0.0)
    genotype_information_diagonal = genotype_vector * genotype_vector * active_weight_vector
    genotype_information = jnp.sum(genotype_information_diagonal)
    penalized_deviance = (
        non_active_deviance
        + regenie2_binary_logistic.compute_logistic_deviance(phenotype_vector, probability_vector, active_sample_mask)
        - jnp.log(genotype_information)
    )
    score_adjustment_numerator = jnp.sum(
        jnp.where(
            active_sample_mask,
            genotype_vector
            * genotype_information_diagonal
            * (regenie2_binary_config.BINARY_CASE_THRESHOLD - probability_vector),
            0.0,
        )
    )
    score_adjustment = score_adjustment_numerator / genotype_information
    score = (
        jnp.sum(jnp.where(active_sample_mask, genotype_vector * (phenotype_vector - probability_vector), 0.0))
        + score_adjustment
    )
    valid = (
        jnp.isfinite(genotype_information)
        & (genotype_information > minimum_variance)
        & jnp.isfinite(penalized_deviance)
        & jnp.isfinite(score)
        & jnp.all(jnp.isfinite(probability_vector))
    )
    return regenie2_binary_firth_types.ScalarFirthComponents(
        genotype_information=genotype_information,
        score_adjustment=score_adjustment,
        penalized_deviance=penalized_deviance,
        score=score,
        valid=valid,
    )


def fit_scalar_pseudo_logistic_step(
    *,
    genotype_vector: jax.Array,
    active_sample_mask: jax.Array,
    offset_vector: jax.Array,
    phenotype_vector: jax.Array,
    score_adjustment: jax.Array,
    initial_score: jax.Array,
    initial_genotype_information: jax.Array,
    initial_beta: jax.Array,
    tolerance: jax.Array,
    maximum_iterations: int | jax.Array,
    maximum_step_size: jax.Array,
) -> regenie2_binary_firth_types.ScalarPseudoLogisticState:
    """Run REGENIE's inner pseudo-response scalar logistic update."""
    maximum_iteration_count = jnp.asarray(maximum_iterations, dtype=jnp.int32)

    def should_continue(state: regenie2_binary_firth_types.ScalarPseudoLogisticState) -> jax.Array:
        return (state.iteration_count < maximum_iteration_count) & (~state.converged) & (~state.failed)

    def run_iteration(
        state: regenie2_binary_firth_types.ScalarPseudoLogisticState,
    ) -> regenie2_binary_firth_types.ScalarPseudoLogisticState:
        step_size = state.score / state.genotype_information
        absolute_step_size = jnp.abs(step_size)
        step_increased = absolute_step_size > state.previous_step_size
        step_scale = jnp.maximum(absolute_step_size / maximum_step_size, 1.0)
        updated_beta = state.beta + step_size / step_scale
        probability_vector = regenie2_binary_logistic.compute_regenie_logistic_probability(
            offset_vector + genotype_vector * updated_beta
        )
        updated_score = (
            jnp.sum(
                jnp.where(
                    active_sample_mask,
                    genotype_vector * (phenotype_vector - probability_vector),
                    0.0,
                )
            )
            + score_adjustment
        )
        weight_vector = probability_vector * (1.0 - probability_vector)
        active_weight_vector = jnp.where(active_sample_mask, weight_vector, 0.0)
        updated_genotype_information = jnp.sum(genotype_vector * genotype_vector * active_weight_vector)
        probability_failed = jnp.any(active_sample_mask & (weight_vector == 0.0))
        numerical_failed = (
            (~jnp.isfinite(updated_beta))
            | (~jnp.isfinite(updated_score))
            | (~jnp.isfinite(updated_genotype_information))
            | (updated_genotype_information <= 0.0)
        )
        failed = step_increased | probability_failed | numerical_failed
        return regenie2_binary_firth_types.ScalarPseudoLogisticState(
            beta=jnp.where(failed, state.beta, updated_beta),
            score=jnp.where(failed, state.score, updated_score),
            genotype_information=jnp.where(failed, state.genotype_information, updated_genotype_information),
            previous_step_size=absolute_step_size,
            iteration_count=state.iteration_count + jnp.asarray(1, dtype=jnp.int32),
            converged=(jnp.abs(updated_score) < tolerance) & (~failed),
            failed=failed,
        )

    return jax.lax.while_loop(
        should_continue,
        run_iteration,
        regenie2_binary_firth_types.ScalarPseudoLogisticState(
            beta=initial_beta,
            score=initial_score,
            genotype_information=initial_genotype_information,
            previous_step_size=jnp.asarray(jnp.inf, dtype=initial_beta.dtype),
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            converged=jnp.asarray(0, dtype=jnp.bool_),
            failed=jnp.asarray(0, dtype=jnp.bool_),
        ),
    )


def fit_scalar_pseudo_firth_with_minimum_variance(
    *,
    deviance_null: jax.Array,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    initial_beta: jax.Array,
    maximum_iterations: int | jax.Array,
    tolerance: jax.Array,
    inner_maximum_iterations: int | jax.Array,
    maximum_step_size: jax.Array,
    minimum_variance: jax.Array,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run scalar pseudo-Firth with explicit numeric policy operands."""
    maximum_iteration_count = jnp.asarray(maximum_iterations, dtype=jnp.int32)
    inner_maximum_iteration_count = jnp.asarray(inner_maximum_iterations, dtype=jnp.int32)
    initial_components = compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=initial_beta,
        minimum_variance=minimum_variance,
    )

    def should_continue(state: regenie2_binary_firth_types.ScalarPseudoFirthState) -> jax.Array:
        return (state.outer_iteration_count < maximum_iteration_count) & (~state.converged) & (~state.failed)

    def run_iteration(
        state: regenie2_binary_firth_types.ScalarPseudoFirthState,
    ) -> regenie2_binary_firth_types.ScalarPseudoFirthState:
        components = state.components
        updated_outer_iteration_count = state.outer_iteration_count + jnp.asarray(1, dtype=jnp.int32)
        converged = (jnp.abs(components.score) < tolerance) & (updated_outer_iteration_count >= 2)
        beta_iteration_14 = jnp.where(
            updated_outer_iteration_count == 14,
            state.beta,
            state.beta_iteration_14,
        )
        slow_convergence_failure = (updated_outer_iteration_count == 15) & (
            jnp.abs(state.beta - beta_iteration_14) > 0.1
        )

        def finish_without_update(_: None) -> regenie2_binary_firth_types.ScalarPseudoFirthState:
            failed = (~components.valid) | slow_convergence_failure
            return regenie2_binary_firth_types.ScalarPseudoFirthState(
                beta=state.beta,
                components=components,
                outer_iteration_count=updated_outer_iteration_count,
                beta_iteration_14=beta_iteration_14,
                converged=converged & (~failed),
                failed=failed,
            )

        def update_beta(_: None) -> regenie2_binary_firth_types.ScalarPseudoFirthState:
            logistic_state = fit_scalar_pseudo_logistic_step(
                genotype_vector=genotype_vector,
                active_sample_mask=active_sample_mask,
                offset_vector=offset_vector,
                phenotype_vector=phenotype_vector,
                score_adjustment=components.score_adjustment,
                initial_score=components.score,
                initial_genotype_information=components.genotype_information,
                initial_beta=state.beta,
                tolerance=tolerance,
                maximum_iterations=inner_maximum_iteration_count,
                maximum_step_size=maximum_step_size,
            )
            failed = (~components.valid) | slow_convergence_failure | logistic_state.failed
            updated_beta = jnp.where(failed, state.beta, logistic_state.beta)
            updated_components = compute_scalar_firth_components_with_minimum_variance(
                phenotype_vector=phenotype_vector,
                genotype_vector=genotype_vector,
                offset_vector=offset_vector,
                active_sample_mask=active_sample_mask,
                non_active_deviance=non_active_deviance,
                beta=updated_beta,
                minimum_variance=minimum_variance,
            )
            return regenie2_binary_firth_types.ScalarPseudoFirthState(
                beta=updated_beta,
                components=updated_components,
                outer_iteration_count=updated_outer_iteration_count,
                beta_iteration_14=beta_iteration_14,
                converged=converged,
                failed=failed,
            )

        return jax.lax.cond(converged, finish_without_update, update_beta, None)

    final_state = jax.lax.while_loop(
        should_continue,
        run_iteration,
        regenie2_binary_firth_types.ScalarPseudoFirthState(
            beta=initial_beta,
            components=initial_components,
            outer_iteration_count=jnp.asarray(0, dtype=jnp.int32),
            beta_iteration_14=jnp.asarray(0.0, dtype=initial_beta.dtype),
            converged=jnp.asarray(0, dtype=jnp.bool_),
            failed=~initial_components.valid,
        ),
    )
    final_components = final_state.components
    maximum_iteration_failure = (~final_state.converged) & (~final_state.failed)
    chi_squared = deviance_null - final_components.penalized_deviance
    negative_lrt_failure = final_state.converged & (chi_squared < 0.0)
    failed = final_state.failed | maximum_iteration_failure | negative_lrt_failure | (~final_components.valid)
    standard_error = jnp.sqrt(jnp.reciprocal(final_components.genotype_information))
    log10_p_value = jnp.asarray(
        pvalue.chi_squared_to_log10_p_value(jnp.maximum(chi_squared, 0.0)),
        dtype=initial_beta.dtype,
    )
    valid = final_state.converged & (~failed) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    return regenie2_binary_firth_types.FirthVariantResult(
        beta=final_state.beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        valid_mask=valid,
    )


def run_scalar_line_search_with_minimum_variance(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    current_beta: jax.Array,
    current_penalized_deviance: jax.Array,
    current_genotype_information: jax.Array,
    current_score: jax.Array,
    current_valid: jax.Array,
    initial_step_size: jax.Array,
    maximum_attempts: int | jax.Array,
    minimum_variance: jax.Array,
) -> regenie2_binary_firth_types.ScalarLineSearchState:
    """Run scalar NR step-halving with explicit numeric policy operands."""
    maximum_attempt_count = jnp.asarray(maximum_attempts, dtype=jnp.int32)

    def should_continue(state: regenie2_binary_firth_types.ScalarLineSearchState) -> jax.Array:
        return (state.attempt_count < maximum_attempt_count) & (~state.accepted) & state.valid

    def run_iteration(
        state: regenie2_binary_firth_types.ScalarLineSearchState,
    ) -> regenie2_binary_firth_types.ScalarLineSearchState:
        adjusted_step_size = jnp.where(state.attempt_count > 0, state.step_size / 2.0, state.step_size)
        candidate_beta = state.beta + adjusted_step_size
        components = compute_scalar_firth_components_with_minimum_variance(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            beta=candidate_beta,
            minimum_variance=minimum_variance,
        )
        accepted = components.valid & (components.penalized_deviance < current_penalized_deviance)
        return regenie2_binary_firth_types.ScalarLineSearchState(
            beta=jnp.where(accepted, candidate_beta, state.beta),
            step_size=adjusted_step_size,
            penalized_deviance=jnp.where(
                accepted,
                components.penalized_deviance,
                state.penalized_deviance,
            ),
            genotype_information=jnp.where(
                accepted,
                components.genotype_information,
                state.genotype_information,
            ),
            score=jnp.where(accepted, components.score, state.score),
            attempt_count=state.attempt_count + jnp.asarray(1, dtype=jnp.int32),
            accepted=accepted,
            valid=state.valid & components.valid,
        )

    return jax.lax.while_loop(
        should_continue,
        run_iteration,
        regenie2_binary_firth_types.ScalarLineSearchState(
            beta=current_beta,
            step_size=initial_step_size,
            penalized_deviance=current_penalized_deviance,
            genotype_information=current_genotype_information,
            score=current_score,
            attempt_count=jnp.asarray(0, dtype=jnp.int32),
            accepted=jnp.asarray(0, dtype=jnp.bool_),
            valid=current_valid,
        ),
    )


def fit_scalar_newton_raphson_firth_with_minimum_variance(
    *,
    deviance_null: jax.Array,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    initial_beta: jax.Array,
    maximum_iterations: int | jax.Array,
    tolerance: jax.Array,
    maximum_step_size: jax.Array,
    line_search_maximum_attempts: int | jax.Array,
    minimum_variance: jax.Array,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run scalar Newton-Raphson approximate Firth with explicit policy."""
    maximum_iteration_count = jnp.asarray(maximum_iterations, dtype=jnp.int32)
    line_search_maximum_attempt_count = jnp.asarray(line_search_maximum_attempts, dtype=jnp.int32)
    initial_components = compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=initial_beta,
        minimum_variance=minimum_variance,
    )

    def should_continue(state: regenie2_binary_firth_types.ScalarNewtonRaphsonState) -> jax.Array:
        return (state.iteration_count < maximum_iteration_count) & (~state.converged) & (~state.failed)

    def run_iteration(
        state: regenie2_binary_firth_types.ScalarNewtonRaphsonState,
    ) -> regenie2_binary_firth_types.ScalarNewtonRaphsonState:
        updated_iteration_count = state.iteration_count + jnp.asarray(1, dtype=jnp.int32)
        converged = (jnp.abs(state.score) < tolerance) & (updated_iteration_count >= 2)

        def finish_without_update(_: None) -> regenie2_binary_firth_types.ScalarNewtonRaphsonState:
            return regenie2_binary_firth_types.ScalarNewtonRaphsonState(
                beta=state.beta,
                penalized_deviance=state.penalized_deviance,
                genotype_information=state.genotype_information,
                score=state.score,
                iteration_count=updated_iteration_count,
                converged=converged,
                failed=state.failed,
            )

        def update_beta(_: None) -> regenie2_binary_firth_types.ScalarNewtonRaphsonState:
            raw_step_size = state.score / state.genotype_information
            step_scale = jnp.maximum(jnp.abs(raw_step_size) / maximum_step_size, 1.0)
            line_search_state = run_scalar_line_search_with_minimum_variance(
                phenotype_vector=phenotype_vector,
                genotype_vector=genotype_vector,
                offset_vector=offset_vector,
                active_sample_mask=active_sample_mask,
                non_active_deviance=non_active_deviance,
                current_beta=state.beta,
                current_penalized_deviance=state.penalized_deviance,
                current_genotype_information=state.genotype_information,
                current_score=state.score,
                current_valid=~state.failed,
                initial_step_size=raw_step_size / step_scale,
                maximum_attempts=line_search_maximum_attempt_count,
                minimum_variance=minimum_variance,
            )
            line_search_failed = ~line_search_state.accepted
            failed = (~state.failed) & (line_search_failed | (~line_search_state.valid))
            return regenie2_binary_firth_types.ScalarNewtonRaphsonState(
                beta=line_search_state.beta,
                penalized_deviance=line_search_state.penalized_deviance,
                genotype_information=line_search_state.genotype_information,
                score=line_search_state.score,
                iteration_count=updated_iteration_count,
                converged=converged,
                failed=failed,
            )

        return jax.lax.cond(converged, finish_without_update, update_beta, None)

    final_state = jax.lax.while_loop(
        should_continue,
        run_iteration,
        regenie2_binary_firth_types.ScalarNewtonRaphsonState(
            beta=initial_beta,
            penalized_deviance=initial_components.penalized_deviance,
            genotype_information=initial_components.genotype_information,
            score=initial_components.score,
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            converged=jnp.asarray(0, dtype=jnp.bool_),
            failed=~initial_components.valid,
        ),
    )
    maximum_iteration_failure = (~final_state.converged) & (~final_state.failed)
    chi_squared = deviance_null - final_state.penalized_deviance
    negative_lrt_failure = final_state.converged & (chi_squared < 0.0)
    failed = final_state.failed | maximum_iteration_failure | negative_lrt_failure
    standard_error = jnp.sqrt(jnp.reciprocal(final_state.genotype_information))
    log10_p_value = jnp.asarray(
        pvalue.chi_squared_to_log10_p_value(jnp.maximum(chi_squared, 0.0)),
        dtype=initial_beta.dtype,
    )
    valid = final_state.converged & (~failed) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    return regenie2_binary_firth_types.FirthVariantResult(
        beta=final_state.beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        valid_mask=valid,
    )


def build_scalar_inactive_solver_result(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Build the scalar result for inactive or null-failed lanes."""
    scalar_dtype = operands.offset_vector.dtype
    missing_value = jnp.asarray(jnp.nan, dtype=scalar_dtype)
    return regenie2_binary_firth_types.FirthVariantResult(
        beta=missing_value,
        standard_error=missing_value,
        chi_squared=missing_value,
        log10_p_value=missing_value,
        valid_mask=jnp.asarray(0, dtype=jnp.bool_),
    )


def run_scalar_pseudo_firth_attempt(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run the scalar pseudo-Firth attempt."""
    parameters = operands.solver_parameters
    return fit_scalar_pseudo_firth_with_minimum_variance(
        deviance_null=operands.deviance_null,
        phenotype_vector=operands.phenotype_vector,
        genotype_vector=operands.genotype_vector,
        offset_vector=operands.offset_vector,
        active_sample_mask=operands.active_sample_mask,
        non_active_deviance=operands.non_active_deviance,
        initial_beta=operands.warm_start_beta,
        maximum_iterations=parameters.pseudo_maximum_iterations,
        tolerance=parameters.tolerance,
        inner_maximum_iterations=parameters.pseudo_inner_maximum_iterations,
        maximum_step_size=parameters.maximum_step_size,
        minimum_variance=parameters.minimum_variance,
    )


def run_scalar_zero_start_newton_raphson_attempt(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run the scalar zero-start Newton-Raphson fallback attempt."""
    parameters = operands.solver_parameters
    return fit_scalar_newton_raphson_firth_with_minimum_variance(
        deviance_null=operands.deviance_null,
        phenotype_vector=operands.phenotype_vector,
        genotype_vector=operands.genotype_vector,
        offset_vector=operands.offset_vector,
        active_sample_mask=operands.active_sample_mask,
        non_active_deviance=operands.non_active_deviance,
        initial_beta=jnp.asarray(0.0, dtype=operands.offset_vector.dtype),
        maximum_iterations=parameters.newton_raphson_zero_start_iterations,
        tolerance=parameters.tolerance,
        maximum_step_size=parameters.maximum_step_size,
        line_search_maximum_attempts=parameters.line_search_maximum_attempts,
        minimum_variance=parameters.minimum_variance,
    )


def run_scalar_warm_start_newton_raphson_attempt(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run the scalar warm-start Newton-Raphson fallback attempt."""
    parameters = operands.solver_parameters
    return fit_scalar_newton_raphson_firth_with_minimum_variance(
        deviance_null=operands.deviance_null,
        phenotype_vector=operands.phenotype_vector,
        genotype_vector=operands.genotype_vector,
        offset_vector=operands.offset_vector,
        active_sample_mask=operands.active_sample_mask,
        non_active_deviance=operands.non_active_deviance,
        initial_beta=operands.warm_start_beta,
        maximum_iterations=parameters.newton_raphson_maximum_iterations,
        tolerance=parameters.tolerance,
        maximum_step_size=parameters.maximum_step_size,
        line_search_maximum_attempts=parameters.line_search_maximum_attempts,
        minimum_variance=parameters.minimum_variance,
    )


def run_scalar_zero_start_then_maybe_warm_start(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run zero-start fallback and warm-start only if zero-start fails."""
    zero_start_result = run_scalar_zero_start_newton_raphson_attempt(operands)
    return jax.lax.cond(
        zero_start_result.valid_mask,
        lambda _: zero_start_result,
        run_scalar_warm_start_newton_raphson_attempt,
        operands,
    )


def run_scalar_fallback_cascade(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run scalar Newton-Raphson fallbacks after pseudo-Firth failure."""
    run_zero_start = operands.sparse_correction & (
        jnp.abs(operands.warm_start_beta) > jnp.asarray(0.0, dtype=operands.offset_vector.dtype)
    )
    return jax.lax.cond(
        run_zero_start,
        run_scalar_zero_start_then_maybe_warm_start,
        run_scalar_warm_start_newton_raphson_attempt,
        operands,
    )


def run_scalar_active_solver(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run pseudo-Firth and lazy Newton-Raphson fallbacks for an active lane."""
    pseudo_result = run_scalar_pseudo_firth_attempt(operands)
    return jax.lax.cond(
        pseudo_result.valid_mask,
        lambda _: pseudo_result,
        run_scalar_fallback_cascade,
        operands,
    )


def fit_single_variant_regenie_approximate_firth_with_solver_parameters(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    carrier_sample_mask: jax.Array,
    full_null_deviance: jax.Array,
    sparse_correction: jax.Array,
    warm_start_beta: jax.Array,
    skip_firth: jax.Array,
    null_failed: jax.Array,
    solver_parameters: regenie2_binary_firth_types.ScalarApproximateFirthSolverParameters,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Fit one scalar approximate-Firth candidate with explicit solver policy."""
    phenotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float64)
    genotype_vector = jnp.asarray(genotype_vector, dtype=jnp.float64)
    offset_vector = jnp.asarray(offset_vector, dtype=jnp.float64)
    full_null_deviance = jnp.asarray(full_null_deviance, dtype=jnp.float64)
    all_sample_mask = jnp.ones_like(phenotype_vector, dtype=jnp.bool_)
    active_sample_mask = jnp.where(sparse_correction, carrier_sample_mask, all_sample_mask)
    null_probability_vector = regenie2_binary_logistic.compute_regenie_logistic_probability(offset_vector)
    active_null_deviance = regenie2_binary_logistic.compute_logistic_deviance(
        phenotype_vector, null_probability_vector, active_sample_mask
    )
    non_active_deviance = jnp.where(sparse_correction, full_null_deviance - active_null_deviance, 0.0)
    return fit_single_variant_regenie_approximate_firth_with_active_samples_and_solver_parameters(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        full_null_deviance=full_null_deviance,
        non_active_deviance=non_active_deviance,
        sparse_correction=sparse_correction,
        warm_start_beta=warm_start_beta,
        skip_firth=skip_firth,
        null_failed=null_failed,
        solver_parameters=solver_parameters,
    )


def fit_compact_carrier_regenie_approximate_firth_with_solver_parameters(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_carrier_slot_mask: jax.Array,
    full_null_deviance: jax.Array,
    warm_start_beta: jax.Array,
    skip_firth: jax.Array,
    null_failed: jax.Array,
    solver_parameters: regenie2_binary_firth_types.ScalarApproximateFirthSolverParameters,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Fit one compact sparse lane with explicit solver policy."""
    phenotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float64)
    genotype_vector = jnp.asarray(genotype_vector, dtype=jnp.float64)
    offset_vector = jnp.asarray(offset_vector, dtype=jnp.float64)
    null_probability_vector = regenie2_binary_logistic.compute_regenie_logistic_probability(offset_vector)
    active_null_deviance = regenie2_binary_logistic.compute_logistic_deviance(
        phenotype_vector,
        null_probability_vector,
        active_carrier_slot_mask,
    )
    return fit_single_variant_regenie_approximate_firth_with_active_samples_and_solver_parameters(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_carrier_slot_mask,
        full_null_deviance=jnp.asarray(full_null_deviance, dtype=jnp.float64),
        non_active_deviance=jnp.asarray(full_null_deviance, dtype=jnp.float64) - active_null_deviance,
        sparse_correction=jnp.ones((), dtype=jnp.bool_),
        warm_start_beta=warm_start_beta,
        skip_firth=skip_firth,
        null_failed=null_failed,
        solver_parameters=solver_parameters,
    )


@jax.jit(inline=True)
def fit_single_variant_regenie_approximate_firth_with_active_samples_and_solver_parameters(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    full_null_deviance: jax.Array,
    non_active_deviance: jax.Array,
    sparse_correction: jax.Array,
    warm_start_beta: jax.Array,
    skip_firth: jax.Array,
    null_failed: jax.Array,
    solver_parameters: regenie2_binary_firth_types.ScalarApproximateFirthSolverParameters,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Fit one approximate-Firth candidate with explicit solver parameters."""
    phenotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float64)
    genotype_vector = jnp.asarray(genotype_vector, dtype=jnp.float64)
    offset_vector = jnp.asarray(offset_vector, dtype=jnp.float64)
    active_sample_mask = jnp.asarray(active_sample_mask, dtype=jnp.bool_)
    full_null_deviance = jnp.asarray(full_null_deviance, dtype=jnp.float64)
    non_active_deviance = jnp.asarray(non_active_deviance, dtype=jnp.float64)
    warm_start_beta = jnp.asarray(warm_start_beta, dtype=jnp.float64)
    sparse_correction = jnp.asarray(sparse_correction, dtype=jnp.bool_)
    null_probability_vector = regenie2_binary_logistic.compute_regenie_logistic_probability(offset_vector)
    null_weight_vector = null_probability_vector * (1.0 - null_probability_vector)
    null_genotype_information = jnp.sum(
        jnp.where(active_sample_mask, genotype_vector * genotype_vector * null_weight_vector, 0.0)
    )
    deviance_null = full_null_deviance - jnp.log(null_genotype_information)
    solver_active = (~skip_firth) & (~null_failed)
    dispatch_operands = regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        deviance_null=deviance_null,
        non_active_deviance=non_active_deviance,
        sparse_correction=sparse_correction,
        warm_start_beta=warm_start_beta,
        solver_parameters=solver_parameters,
    )
    return jax.lax.cond(
        solver_active,
        run_scalar_active_solver,
        build_scalar_inactive_solver_result,
        dispatch_operands,
    )
