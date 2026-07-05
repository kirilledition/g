"""Scalar approximate-Firth solver for REGENIE step 2 binary tests."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g import types
from g.compute.common import pvalue
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import state as regenie2_binary_state


def map_scalar_pseudo_firth_failure_to_reason_code(failure_code: jax.Array) -> jax.Array:
    """Map REGENIE scalar pseudo-Firth failure states to internal reason codes."""
    return jnp.where(
        failure_code == jnp.asarray(1, dtype=jnp.int32),
        regenie2_binary_firth_types.FirthConvergenceReason.MAX_ITERATIONS.value,
        jnp.where(
            failure_code == jnp.asarray(2, dtype=jnp.int32),
            regenie2_binary_firth_types.FirthConvergenceReason.STEP_SIZE_INCREASE.value,
            jnp.where(
                failure_code == jnp.asarray(3, dtype=jnp.int32),
                regenie2_binary_firth_types.FirthConvergenceReason.PROBABILITY_FAILURE.value,
                jnp.where(
                    failure_code == jnp.asarray(4, dtype=jnp.int32),
                    regenie2_binary_firth_types.FirthConvergenceReason.NEGATIVE_LRT.value,
                    regenie2_binary_firth_types.FirthConvergenceReason.NUMERICAL_FAILURE.value,
                ),
            ),
        ),
    ).astype(jnp.int32)


def residualize_and_scale_genotypes_for_approximate_firth(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
) -> jax.Array:
    """Build REGENIE's approximate-Firth residualized genotype vector."""
    weighted_genotype_matrix_by_variant = genotype_matrix_by_variant * chromosome_state.square_root_weight[None, :]
    projection_coordinates = (
        weighted_genotype_matrix_by_variant @ chromosome_state.weighted_genotype_projection_matrix.T
    )
    weighted_residual_matrix_by_variant = weighted_genotype_matrix_by_variant - (
        projection_coordinates @ chromosome_state.weighted_genotype_projection_matrix
    )
    return weighted_residual_matrix_by_variant / chromosome_state.square_root_weight[None, :]


def build_scalar_approximate_firth_solver_parameters(
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    scalar_dtype: typing.Any,
) -> regenie2_binary_firth_types.ScalarApproximateFirthSolverParameters:
    """Build explicit scalar approximate-Firth policy operands."""
    pseudo_maximum_iterations = min(
        kernel_config.approximate_firth.maximum_iterations // 2,
        kernel_config.approximate_firth.pseudo_maximum_iterations,
    )
    newton_raphson_maximum_iterations = kernel_config.approximate_firth.maximum_iterations // 2
    return regenie2_binary_firth_types.ScalarApproximateFirthSolverParameters(
        minimum_variance=jnp.asarray(kernel_config.numerical.minimum_variance, dtype=scalar_dtype),
        tolerance=jnp.asarray(kernel_config.approximate_firth.gradient_tolerance, dtype=scalar_dtype),
        maximum_step_size=jnp.asarray(kernel_config.approximate_firth.maximum_step_size, dtype=scalar_dtype),
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
    leverage_vector = genotype_information_diagonal / genotype_information
    adjusted_response = phenotype_vector + leverage_vector * (
        regenie2_binary_config.BINARY_CASE_THRESHOLD - probability_vector
    )
    score = jnp.sum(jnp.where(active_sample_mask, genotype_vector * (adjusted_response - probability_vector), 0.0))
    valid = (
        jnp.isfinite(genotype_information)
        & (genotype_information > minimum_variance)
        & jnp.isfinite(penalized_deviance)
        & jnp.isfinite(score)
        & jnp.all(jnp.isfinite(probability_vector))
        & jnp.all(jnp.isfinite(weight_vector))
    )
    return regenie2_binary_firth_types.ScalarFirthComponents(
        probability_vector=probability_vector,
        weight_vector=weight_vector,
        genotype_information=genotype_information,
        genotype_information_diagonal=genotype_information_diagonal,
        penalized_deviance=penalized_deviance,
        score=score,
        valid=valid,
    )


def should_continue_scalar_pseudo_logistic(
    carry: regenie2_binary_firth_types.ScalarPseudoLogisticLoopCarry,
) -> jax.Array:
    """Return whether the pseudo-logistic inner loop should continue."""
    state = carry.state
    return (state.iteration_count < carry.maximum_iterations) & (~state.converged) & (~state.failed)


def run_scalar_pseudo_logistic_iteration(
    carry: regenie2_binary_firth_types.ScalarPseudoLogisticLoopCarry,
) -> regenie2_binary_firth_types.ScalarPseudoLogisticLoopCarry:
    """Run one pseudo-response scalar logistic update."""
    state = carry.state
    step_size = state.score / state.genotype_information
    absolute_step_size = jnp.abs(step_size)
    step_increased = absolute_step_size > state.previous_step_size
    step_scale = jnp.maximum(absolute_step_size / carry.maximum_step_size, 1.0)
    updated_beta = state.beta + step_size / step_scale
    probability_vector = regenie2_binary_logistic.compute_regenie_logistic_probability(
        carry.offset_vector + carry.genotype_vector * updated_beta
    )
    updated_score = jnp.sum(
        jnp.where(
            carry.active_sample_mask,
            carry.genotype_vector * (carry.adjusted_response - probability_vector),
            0.0,
        )
    )
    weight_vector = probability_vector * (1.0 - probability_vector)
    active_weight_vector = jnp.where(carry.active_sample_mask, weight_vector, 0.0)
    updated_genotype_information = jnp.sum(carry.genotype_vector * carry.genotype_vector * active_weight_vector)
    probability_failed = jnp.any(carry.active_sample_mask & (weight_vector == 0.0))
    numerical_failed = (
        (~jnp.isfinite(updated_beta))
        | (~jnp.isfinite(updated_score))
        | (~jnp.isfinite(updated_genotype_information))
        | (updated_genotype_information <= 0.0)
    )
    failed = step_increased | probability_failed | numerical_failed
    failure_code = jnp.where(
        step_increased,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.where(
            probability_failed | numerical_failed,
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return regenie2_binary_firth_types.ScalarPseudoLogisticLoopCarry(
        state=regenie2_binary_firth_types.ScalarPseudoLogisticState(
            beta=jnp.where(failed, state.beta, updated_beta),
            score=jnp.where(failed, state.score, updated_score),
            genotype_information=jnp.where(failed, state.genotype_information, updated_genotype_information),
            step_size=step_size,
            previous_step_size=absolute_step_size,
            iteration_count=state.iteration_count + jnp.asarray(1, dtype=jnp.int32),
            converged=(jnp.abs(updated_score) < carry.tolerance) & (~failed),
            failed=failed,
            failure_code=failure_code,
        ),
        genotype_vector=carry.genotype_vector,
        active_sample_mask=carry.active_sample_mask,
        offset_vector=carry.offset_vector,
        adjusted_response=carry.adjusted_response,
        tolerance=carry.tolerance,
        maximum_iterations=carry.maximum_iterations,
        maximum_step_size=carry.maximum_step_size,
    )


def fit_scalar_pseudo_logistic_step(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    active_sample_mask: jax.Array,
    offset_vector: jax.Array,
    adjusted_response: jax.Array,
    initial_score: jax.Array,
    initial_genotype_information: jax.Array,
    initial_beta: jax.Array,
    tolerance: jax.Array,
    maximum_iterations: int | jax.Array,
    maximum_step_size: jax.Array,
) -> regenie2_binary_firth_types.ScalarPseudoLogisticState:
    """Run REGENIE's inner pseudo-response scalar logistic update."""
    del phenotype_vector
    final_carry = jax.lax.while_loop(
        should_continue_scalar_pseudo_logistic,
        run_scalar_pseudo_logistic_iteration,
        regenie2_binary_firth_types.ScalarPseudoLogisticLoopCarry(
            state=regenie2_binary_firth_types.ScalarPseudoLogisticState(
                beta=initial_beta,
                score=initial_score,
                genotype_information=initial_genotype_information,
                step_size=jnp.asarray(0.0, dtype=initial_beta.dtype),
                previous_step_size=jnp.asarray(jnp.inf, dtype=initial_beta.dtype),
                iteration_count=jnp.asarray(0, dtype=jnp.int32),
                converged=jnp.asarray(0, dtype=jnp.bool_),
                failed=jnp.asarray(0, dtype=jnp.bool_),
                failure_code=jnp.asarray(0, dtype=jnp.int32),
            ),
            genotype_vector=genotype_vector,
            active_sample_mask=active_sample_mask,
            offset_vector=offset_vector,
            adjusted_response=adjusted_response,
            tolerance=tolerance,
            maximum_iterations=jnp.asarray(maximum_iterations, dtype=jnp.int32),
            maximum_step_size=maximum_step_size,
        ),
    )
    return final_carry.state


def should_continue_scalar_pseudo_firth(
    carry: regenie2_binary_firth_types.ScalarPseudoFirthLoopCarry,
) -> jax.Array:
    """Return whether the scalar pseudo-Firth outer loop should continue."""
    state = carry.state
    return (state.outer_iteration_count < carry.maximum_iterations) & (~state.converged) & (~state.failed)


def run_scalar_pseudo_firth_iteration(
    carry: regenie2_binary_firth_types.ScalarPseudoFirthLoopCarry,
) -> regenie2_binary_firth_types.ScalarPseudoFirthLoopCarry:
    """Run one scalar pseudo-Firth outer iteration."""
    state = carry.state
    components = compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=carry.phenotype_vector,
        genotype_vector=carry.genotype_vector,
        offset_vector=carry.offset_vector,
        active_sample_mask=carry.active_sample_mask,
        non_active_deviance=carry.non_active_deviance,
        beta=state.beta,
        minimum_variance=carry.minimum_variance,
    )
    updated_outer_iteration_count = state.outer_iteration_count + jnp.asarray(1, dtype=jnp.int32)
    converged = (jnp.abs(components.score) < carry.tolerance) & (updated_outer_iteration_count >= 2)
    beta_iteration_14 = jnp.where(
        updated_outer_iteration_count == 14,
        state.beta,
        state.beta_iteration_14,
    )
    slow_convergence_failure = (updated_outer_iteration_count == 15) & (jnp.abs(state.beta - beta_iteration_14) > 0.1)
    leverage_vector = components.genotype_information_diagonal / components.genotype_information
    adjusted_response = carry.phenotype_vector + leverage_vector * (
        regenie2_binary_config.BINARY_CASE_THRESHOLD - components.probability_vector
    )
    logistic_state = fit_scalar_pseudo_logistic_step(
        phenotype_vector=carry.phenotype_vector,
        genotype_vector=carry.genotype_vector,
        active_sample_mask=carry.active_sample_mask,
        offset_vector=carry.offset_vector,
        adjusted_response=adjusted_response,
        initial_score=components.score,
        initial_genotype_information=components.genotype_information,
        initial_beta=state.beta,
        tolerance=carry.tolerance,
        maximum_iterations=carry.inner_maximum_iterations,
        maximum_step_size=carry.maximum_step_size,
    )
    failed = (~components.valid) | slow_convergence_failure | logistic_state.failed
    failure_code = jnp.where(
        ~components.valid,
        jnp.asarray(3, dtype=jnp.int32),
        jnp.where(
            slow_convergence_failure,
            jnp.asarray(1, dtype=jnp.int32),
            logistic_state.failure_code,
        ),
    )
    return regenie2_binary_firth_types.ScalarPseudoFirthLoopCarry(
        state=regenie2_binary_firth_types.ScalarPseudoFirthState(
            beta=jnp.where(converged | failed, state.beta, logistic_state.beta),
            penalized_deviance=components.penalized_deviance,
            genotype_information=components.genotype_information,
            score=components.score,
            outer_iteration_count=updated_outer_iteration_count,
            inner_iteration_count=state.inner_iteration_count + logistic_state.iteration_count,
            beta_iteration_14=beta_iteration_14,
            converged=converged & (~failed),
            failed=failed,
            failure_code=failure_code,
        ),
        phenotype_vector=carry.phenotype_vector,
        genotype_vector=carry.genotype_vector,
        offset_vector=carry.offset_vector,
        active_sample_mask=carry.active_sample_mask,
        non_active_deviance=carry.non_active_deviance,
        tolerance=carry.tolerance,
        maximum_iterations=carry.maximum_iterations,
        inner_maximum_iterations=carry.inner_maximum_iterations,
        maximum_step_size=carry.maximum_step_size,
        minimum_variance=carry.minimum_variance,
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
) -> regenie2_binary_firth_types.ScalarFirthAttemptResult:
    """Run scalar pseudo-Firth with explicit numeric policy operands."""
    initial_components = compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=initial_beta,
        minimum_variance=minimum_variance,
    )
    final_carry = jax.lax.while_loop(
        should_continue_scalar_pseudo_firth,
        run_scalar_pseudo_firth_iteration,
        regenie2_binary_firth_types.ScalarPseudoFirthLoopCarry(
            state=regenie2_binary_firth_types.ScalarPseudoFirthState(
                beta=initial_beta,
                penalized_deviance=initial_components.penalized_deviance,
                genotype_information=initial_components.genotype_information,
                score=initial_components.score,
                outer_iteration_count=jnp.asarray(0, dtype=jnp.int32),
                inner_iteration_count=jnp.asarray(0, dtype=jnp.int32),
                beta_iteration_14=jnp.asarray(0.0, dtype=initial_beta.dtype),
                converged=jnp.asarray(0, dtype=jnp.bool_),
                failed=~initial_components.valid,
                failure_code=jnp.where(
                    initial_components.valid,
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(3, dtype=jnp.int32),
                ),
            ),
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            tolerance=tolerance,
            maximum_iterations=jnp.asarray(maximum_iterations, dtype=jnp.int32),
            inner_maximum_iterations=jnp.asarray(inner_maximum_iterations, dtype=jnp.int32),
            maximum_step_size=maximum_step_size,
            minimum_variance=minimum_variance,
        ),
    )
    final_state = final_carry.state
    final_components = compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=final_state.beta,
        minimum_variance=minimum_variance,
    )
    maximum_iteration_failure = (~final_state.converged) & (~final_state.failed)
    chi_squared = deviance_null - final_components.penalized_deviance
    negative_lrt_failure = final_state.converged & (chi_squared < 0.0)
    failed = final_state.failed | maximum_iteration_failure | negative_lrt_failure | (~final_components.valid)
    failure_code = jnp.where(
        maximum_iteration_failure,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.where(
            negative_lrt_failure,
            jnp.asarray(4, dtype=jnp.int32),
            jnp.where(~final_components.valid, jnp.asarray(3, dtype=jnp.int32), final_state.failure_code),
        ),
    )
    standard_error = jnp.sqrt(jnp.reciprocal(final_components.genotype_information))
    log10_p_value = jnp.asarray(
        pvalue.chi_squared_to_log10_p_value(jnp.maximum(chi_squared, 0.0)),
        dtype=initial_beta.dtype,
    )
    valid = final_state.converged & (~failed) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    return regenie2_binary_firth_types.ScalarFirthAttemptResult(
        beta=final_state.beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        penalized_deviance=final_components.penalized_deviance,
        genotype_information=final_components.genotype_information,
        converged=final_state.converged & (~failed),
        valid=valid,
        iteration_count=final_state.outer_iteration_count,
        failure_reason_code=map_scalar_pseudo_firth_failure_to_reason_code(failure_code),
    )


def should_continue_scalar_line_search(
    carry: regenie2_binary_firth_types.ScalarLineSearchLoopCarry,
) -> jax.Array:
    """Return whether scalar line search should evaluate another step."""
    state = carry.state
    return (state.attempt_count < carry.maximum_attempts) & (~state.accepted) & state.valid


def run_scalar_line_search_iteration(
    carry: regenie2_binary_firth_types.ScalarLineSearchLoopCarry,
) -> regenie2_binary_firth_types.ScalarLineSearchLoopCarry:
    """Run one scalar Newton-Raphson step-halving attempt."""
    state = carry.state
    adjusted_step_size = jnp.where(state.attempt_count > 0, state.step_size / 2.0, state.step_size)
    candidate_beta = carry.current_beta + adjusted_step_size
    components = compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=carry.phenotype_vector,
        genotype_vector=carry.genotype_vector,
        offset_vector=carry.offset_vector,
        active_sample_mask=carry.active_sample_mask,
        non_active_deviance=carry.non_active_deviance,
        beta=candidate_beta,
        minimum_variance=carry.minimum_variance,
    )
    accepted = components.valid & (components.penalized_deviance < carry.current_penalized_deviance)
    return regenie2_binary_firth_types.ScalarLineSearchLoopCarry(
        state=regenie2_binary_firth_types.ScalarLineSearchState(
            beta=jnp.where(accepted, candidate_beta, state.beta),
            step_size=adjusted_step_size,
            penalized_deviance=jnp.where(accepted, components.penalized_deviance, state.penalized_deviance),
            genotype_information=jnp.where(accepted, components.genotype_information, state.genotype_information),
            genotype_information_diagonal=jnp.where(
                accepted,
                components.genotype_information_diagonal,
                state.genotype_information_diagonal,
            ),
            probability_vector=jnp.where(accepted, components.probability_vector, state.probability_vector),
            attempt_count=state.attempt_count + jnp.asarray(1, dtype=jnp.int32),
            accepted=accepted,
            valid=state.valid & components.valid,
        ),
        phenotype_vector=carry.phenotype_vector,
        genotype_vector=carry.genotype_vector,
        offset_vector=carry.offset_vector,
        active_sample_mask=carry.active_sample_mask,
        non_active_deviance=carry.non_active_deviance,
        current_beta=carry.current_beta,
        current_penalized_deviance=carry.current_penalized_deviance,
        maximum_attempts=carry.maximum_attempts,
        minimum_variance=carry.minimum_variance,
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
    initial_step_size: jax.Array,
    maximum_attempts: int | jax.Array,
    minimum_variance: jax.Array,
) -> regenie2_binary_firth_types.ScalarLineSearchState:
    """Run scalar NR step-halving with explicit numeric policy operands."""
    initial_components = compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=current_beta,
        minimum_variance=minimum_variance,
    )
    final_carry = jax.lax.while_loop(
        should_continue_scalar_line_search,
        run_scalar_line_search_iteration,
        regenie2_binary_firth_types.ScalarLineSearchLoopCarry(
            state=regenie2_binary_firth_types.ScalarLineSearchState(
                beta=current_beta,
                step_size=initial_step_size,
                penalized_deviance=current_penalized_deviance,
                genotype_information=initial_components.genotype_information,
                genotype_information_diagonal=initial_components.genotype_information_diagonal,
                probability_vector=initial_components.probability_vector,
                attempt_count=jnp.asarray(0, dtype=jnp.int32),
                accepted=jnp.asarray(0, dtype=jnp.bool_),
                valid=initial_components.valid,
            ),
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            current_beta=current_beta,
            current_penalized_deviance=current_penalized_deviance,
            maximum_attempts=jnp.asarray(maximum_attempts, dtype=jnp.int32),
            minimum_variance=minimum_variance,
        ),
    )
    return final_carry.state


def should_continue_scalar_newton_raphson(
    carry: regenie2_binary_firth_types.ScalarNewtonRaphsonLoopCarry,
) -> jax.Array:
    """Return whether scalar Newton-Raphson should continue."""
    state = carry.state
    return (state.iteration_count < carry.maximum_iterations) & (~state.converged) & (~state.failed)


def run_scalar_newton_raphson_iteration(
    carry: regenie2_binary_firth_types.ScalarNewtonRaphsonLoopCarry,
) -> regenie2_binary_firth_types.ScalarNewtonRaphsonLoopCarry:
    """Run one scalar Newton-Raphson approximate-Firth iteration."""
    state = carry.state
    leverage_vector = state.genotype_information_diagonal / state.genotype_information
    score = jnp.sum(
        jnp.where(
            carry.active_sample_mask,
            carry.genotype_vector
            * (carry.phenotype_vector - state.probability_vector + leverage_vector * (0.5 - state.probability_vector)),
            0.0,
        )
    )
    updated_iteration_count = state.iteration_count + jnp.asarray(1, dtype=jnp.int32)
    converged = (jnp.abs(score) < carry.tolerance) & (updated_iteration_count >= 2)
    raw_step_size = score / state.genotype_information
    step_scale = jnp.maximum(jnp.abs(raw_step_size) / carry.maximum_step_size, 1.0)
    step_size = raw_step_size / step_scale
    line_search_state = run_scalar_line_search_with_minimum_variance(
        phenotype_vector=carry.phenotype_vector,
        genotype_vector=carry.genotype_vector,
        offset_vector=carry.offset_vector,
        active_sample_mask=carry.active_sample_mask,
        non_active_deviance=carry.non_active_deviance,
        current_beta=state.beta,
        current_penalized_deviance=state.penalized_deviance,
        initial_step_size=step_size,
        maximum_attempts=carry.line_search_maximum_attempts,
        minimum_variance=carry.minimum_variance,
    )
    line_search_failed = (~converged) & (~line_search_state.accepted)
    updated_beta = jnp.where(converged | line_search_failed, state.beta, line_search_state.beta)
    updated_components = compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=carry.phenotype_vector,
        genotype_vector=carry.genotype_vector,
        offset_vector=carry.offset_vector,
        active_sample_mask=carry.active_sample_mask,
        non_active_deviance=carry.non_active_deviance,
        beta=updated_beta,
        minimum_variance=carry.minimum_variance,
    )
    failed = (~state.failed) & (line_search_failed | (~updated_components.valid) | (~line_search_state.valid))
    return regenie2_binary_firth_types.ScalarNewtonRaphsonLoopCarry(
        state=regenie2_binary_firth_types.ScalarNewtonRaphsonState(
            beta=updated_beta,
            penalized_deviance=jnp.where(converged, state.penalized_deviance, updated_components.penalized_deviance),
            genotype_information=updated_components.genotype_information,
            genotype_information_diagonal=updated_components.genotype_information_diagonal,
            probability_vector=updated_components.probability_vector,
            score=score,
            iteration_count=updated_iteration_count,
            converged=converged & (~failed),
            failed=failed,
            failure_reason_code=jnp.where(
                failed,
                regenie2_binary_firth_types.FirthConvergenceReason.PROBABILITY_FAILURE.value,
                regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
            ).astype(jnp.int32),
        ),
        phenotype_vector=carry.phenotype_vector,
        genotype_vector=carry.genotype_vector,
        offset_vector=carry.offset_vector,
        active_sample_mask=carry.active_sample_mask,
        non_active_deviance=carry.non_active_deviance,
        tolerance=carry.tolerance,
        maximum_iterations=carry.maximum_iterations,
        maximum_step_size=carry.maximum_step_size,
        line_search_maximum_attempts=carry.line_search_maximum_attempts,
        minimum_variance=carry.minimum_variance,
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
) -> regenie2_binary_firth_types.ScalarFirthAttemptResult:
    """Run scalar Newton-Raphson approximate Firth with explicit policy."""
    initial_components = compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=initial_beta,
        minimum_variance=minimum_variance,
    )
    final_carry = jax.lax.while_loop(
        should_continue_scalar_newton_raphson,
        run_scalar_newton_raphson_iteration,
        regenie2_binary_firth_types.ScalarNewtonRaphsonLoopCarry(
            state=regenie2_binary_firth_types.ScalarNewtonRaphsonState(
                beta=initial_beta,
                penalized_deviance=initial_components.penalized_deviance,
                genotype_information=initial_components.genotype_information,
                genotype_information_diagonal=initial_components.genotype_information_diagonal,
                probability_vector=initial_components.probability_vector,
                score=initial_components.score,
                iteration_count=jnp.asarray(0, dtype=jnp.int32),
                converged=jnp.asarray(0, dtype=jnp.bool_),
                failed=~initial_components.valid,
                failure_reason_code=jnp.where(
                    initial_components.valid,
                    regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                    regenie2_binary_firth_types.FirthConvergenceReason.PROBABILITY_FAILURE.value,
                ).astype(jnp.int32),
            ),
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            tolerance=tolerance,
            maximum_iterations=jnp.asarray(maximum_iterations, dtype=jnp.int32),
            maximum_step_size=maximum_step_size,
            line_search_maximum_attempts=jnp.asarray(line_search_maximum_attempts, dtype=jnp.int32),
            minimum_variance=minimum_variance,
        ),
    )
    final_state = final_carry.state
    final_components = compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=final_state.beta,
        minimum_variance=minimum_variance,
    )
    maximum_iteration_failure = (~final_state.converged) & (~final_state.failed)
    chi_squared = deviance_null - final_components.penalized_deviance
    negative_lrt_failure = final_state.converged & (chi_squared < 0.0)
    reason_code = jnp.where(
        maximum_iteration_failure,
        regenie2_binary_firth_types.FirthConvergenceReason.MAX_ITERATIONS.value,
        jnp.where(
            negative_lrt_failure,
            regenie2_binary_firth_types.FirthConvergenceReason.NEGATIVE_LRT.value,
            final_state.failure_reason_code,
        ),
    ).astype(jnp.int32)
    failed = final_state.failed | maximum_iteration_failure | negative_lrt_failure | (~final_components.valid)
    standard_error = jnp.sqrt(jnp.reciprocal(final_components.genotype_information))
    log10_p_value = jnp.asarray(
        pvalue.chi_squared_to_log10_p_value(jnp.maximum(chi_squared, 0.0)),
        dtype=initial_beta.dtype,
    )
    valid = final_state.converged & (~failed) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    return regenie2_binary_firth_types.ScalarFirthAttemptResult(
        beta=final_state.beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        penalized_deviance=final_components.penalized_deviance,
        genotype_information=final_components.genotype_information,
        converged=final_state.converged & (~failed),
        valid=valid,
        iteration_count=final_state.iteration_count,
        failure_reason_code=reason_code,
    )


def build_single_variant_regenie_approximate_firth_result(
    *,
    skip_firth: jax.Array,
    null_failed: jax.Array,
    sparse_correction: jax.Array,
    pseudo_result: regenie2_binary_firth_types.ScalarFirthAttemptResult,
    zero_start_result: regenie2_binary_firth_types.ScalarFirthAttemptResult,
    warm_start_result: regenie2_binary_firth_types.ScalarFirthAttemptResult,
    run_zero_start: jax.Array,
    run_warm_start: jax.Array,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Build the public scalar approximate-Firth result from attempted stages."""
    scalar_dtype = pseudo_result.beta.dtype
    use_zero_start = run_zero_start & zero_start_result.valid
    use_warm_start = (~pseudo_result.valid) & (~use_zero_start) & warm_start_result.valid
    selected_beta = jnp.where(
        pseudo_result.valid,
        pseudo_result.beta,
        jnp.where(use_zero_start, zero_start_result.beta, warm_start_result.beta),
    )
    selected_standard_error = jnp.where(
        pseudo_result.valid,
        pseudo_result.standard_error,
        jnp.where(use_zero_start, zero_start_result.standard_error, warm_start_result.standard_error),
    )
    selected_chi_squared = jnp.where(
        pseudo_result.valid,
        pseudo_result.chi_squared,
        jnp.where(use_zero_start, zero_start_result.chi_squared, warm_start_result.chi_squared),
    )
    selected_log10_p_value = jnp.where(
        pseudo_result.valid,
        pseudo_result.log10_p_value,
        jnp.where(use_zero_start, zero_start_result.log10_p_value, warm_start_result.log10_p_value),
    )
    selected_deviance = jnp.where(
        pseudo_result.valid,
        pseudo_result.penalized_deviance,
        jnp.where(use_zero_start, zero_start_result.penalized_deviance, warm_start_result.penalized_deviance),
    )
    selected_reason_code = jnp.where(
        pseudo_result.valid,
        regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
        jnp.where(
            use_zero_start,
            regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
            jnp.where(
                use_warm_start,
                regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
                warm_start_result.failure_reason_code,
            ),
        ),
    ).astype(jnp.int32)
    valid_mask = (~skip_firth) & (~null_failed) & (pseudo_result.valid | use_zero_start | use_warm_start)
    selected_reason_code = jnp.where(
        null_failed,
        regenie2_binary_firth_types.FirthConvergenceReason.NULL_FAILURE.value,
        selected_reason_code,
    )
    failure_code = regenie2_binary_firth_types.map_firth_reason_code_to_failure_code(selected_reason_code)
    correction_code = jnp.where(
        valid_mask & pseudo_result.valid,
        types.FirthCorrectionCode.PSEUDO_FIRTH.value,
        jnp.where(
            valid_mask & use_zero_start,
            types.FirthCorrectionCode.NEWTON_RAPHSON_ZERO_START.value,
            jnp.where(
                valid_mask & use_warm_start,
                types.FirthCorrectionCode.NEWTON_RAPHSON_WARM_START.value,
                types.FirthCorrectionCode.NONE.value,
            ),
        ),
    ).astype(jnp.int32)
    return regenie2_binary_firth_types.FirthVariantResult(
        beta=jnp.where(skip_firth, jnp.nan, selected_beta),
        standard_error=jnp.where(skip_firth, jnp.nan, selected_standard_error),
        chi_squared=jnp.where(skip_firth, jnp.nan, selected_chi_squared),
        log10_p_value=jnp.asarray(jnp.where(skip_firth, jnp.nan, selected_log10_p_value), dtype=scalar_dtype),
        penalized_log_likelihood=jnp.where(skip_firth, jnp.nan, -0.5 * selected_deviance),
        converged_mask=valid_mask,
        valid_mask=valid_mask,
        iteration_count=jnp.where(
            skip_firth,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
                null_failed,
                jnp.asarray(0, dtype=jnp.int32),
                pseudo_result.iteration_count,
            )
            + jnp.where(run_zero_start, zero_start_result.iteration_count, jnp.asarray(0, dtype=jnp.int32))
            + jnp.where(run_warm_start, warm_start_result.iteration_count, jnp.asarray(0, dtype=jnp.int32)),
        ),
        failure_code=jnp.where(skip_firth | valid_mask, types.FirthFailureCode.NONE.value, failure_code).astype(
            jnp.int32
        ),
        convergence_reason_code=jnp.where(
            skip_firth,
            regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
            selected_reason_code,
        ),
        correction_code=jnp.where(skip_firth, types.FirthCorrectionCode.NONE.value, correction_code),
        sparse_correction_mask=(~skip_firth) & sparse_correction,
        pseudo_firth_iteration_count=jnp.where(
            skip_firth,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(null_failed, jnp.asarray(0, dtype=jnp.int32), pseudo_result.iteration_count),
        ),
        nr_zero_start_iteration_count=jnp.where(
            (~skip_firth) & run_zero_start,
            zero_start_result.iteration_count,
            jnp.asarray(0, dtype=jnp.int32),
        ),
        nr_warm_start_iteration_count=jnp.where(
            (~skip_firth) & run_warm_start,
            warm_start_result.iteration_count,
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )


def build_skipped_scalar_firth_attempt(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.ScalarFirthAttemptResult:
    """Build a placeholder scalar Firth attempt for skipped branches."""
    scalar_dtype = operands.offset_vector.dtype
    return regenie2_binary_firth_types.ScalarFirthAttemptResult(
        beta=jnp.asarray(jnp.nan, dtype=scalar_dtype),
        standard_error=jnp.asarray(jnp.nan, dtype=scalar_dtype),
        chi_squared=jnp.asarray(jnp.nan, dtype=scalar_dtype),
        log10_p_value=jnp.asarray(jnp.nan, dtype=scalar_dtype),
        penalized_deviance=jnp.asarray(jnp.nan, dtype=scalar_dtype),
        genotype_information=jnp.asarray(jnp.nan, dtype=scalar_dtype),
        converged=jnp.asarray(0, dtype=jnp.bool_),
        valid=jnp.asarray(0, dtype=jnp.bool_),
        iteration_count=jnp.asarray(0, dtype=jnp.int32),
        failure_reason_code=jnp.asarray(
            regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
            dtype=jnp.int32,
        ),
    )


def run_scalar_pseudo_firth_attempt(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.ScalarFirthAttemptResult:
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
) -> regenie2_binary_firth_types.ScalarFirthAttemptResult:
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
) -> regenie2_binary_firth_types.ScalarFirthAttemptResult:
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


def build_single_variant_regenie_approximate_firth_result_from_operands(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthResultOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Build a scalar approximate-Firth result from branch operands."""
    dispatch_operands = operands.dispatch_operands
    return build_single_variant_regenie_approximate_firth_result(
        skip_firth=dispatch_operands.skip_firth,
        null_failed=dispatch_operands.null_failed,
        sparse_correction=dispatch_operands.sparse_correction,
        pseudo_result=operands.pseudo_result,
        zero_start_result=operands.zero_start_result,
        warm_start_result=operands.warm_start_result,
        run_zero_start=operands.run_zero_start,
        run_warm_start=operands.run_warm_start,
    )


def build_scalar_inactive_solver_result(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Build the scalar result for inactive or null-failed lanes."""
    skipped_result = build_skipped_scalar_firth_attempt(operands)
    false_scalar = jnp.asarray(0, dtype=jnp.bool_)
    return build_single_variant_regenie_approximate_firth_result_from_operands(
        regenie2_binary_firth_types.ScalarApproximateFirthResultOperands(
            dispatch_operands=operands,
            pseudo_result=skipped_result,
            zero_start_result=skipped_result,
            warm_start_result=skipped_result,
            run_zero_start=false_scalar,
            run_warm_start=false_scalar,
        )
    )


def build_scalar_pseudo_firth_success_result(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthFallbackOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Build the scalar result when pseudo-Firth succeeds."""
    skipped_result = build_skipped_scalar_firth_attempt(operands.dispatch_operands)
    false_scalar = jnp.asarray(0, dtype=jnp.bool_)
    return build_single_variant_regenie_approximate_firth_result_from_operands(
        regenie2_binary_firth_types.ScalarApproximateFirthResultOperands(
            dispatch_operands=operands.dispatch_operands,
            pseudo_result=operands.pseudo_result,
            zero_start_result=skipped_result,
            warm_start_result=skipped_result,
            run_zero_start=false_scalar,
            run_warm_start=false_scalar,
        )
    )


def run_scalar_warm_start_without_zero_start(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthFallbackOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run warm-start fallback without a zero-start attempt."""
    skipped_result = build_skipped_scalar_firth_attempt(operands.dispatch_operands)
    warm_start_result = run_scalar_warm_start_newton_raphson_attempt(operands.dispatch_operands)
    return build_single_variant_regenie_approximate_firth_result_from_operands(
        regenie2_binary_firth_types.ScalarApproximateFirthResultOperands(
            dispatch_operands=operands.dispatch_operands,
            pseudo_result=operands.pseudo_result,
            zero_start_result=skipped_result,
            warm_start_result=warm_start_result,
            run_zero_start=jnp.asarray(0, dtype=jnp.bool_),
            run_warm_start=jnp.asarray(1, dtype=jnp.bool_),
        )
    )


def build_scalar_zero_start_success_result(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthZeroStartOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Build the scalar result when zero-start fallback succeeds."""
    dispatch_operands = operands.fallback_operands.dispatch_operands
    skipped_result = build_skipped_scalar_firth_attempt(dispatch_operands)
    return build_single_variant_regenie_approximate_firth_result_from_operands(
        regenie2_binary_firth_types.ScalarApproximateFirthResultOperands(
            dispatch_operands=dispatch_operands,
            pseudo_result=operands.fallback_operands.pseudo_result,
            zero_start_result=operands.zero_start_result,
            warm_start_result=skipped_result,
            run_zero_start=jnp.asarray(1, dtype=jnp.bool_),
            run_warm_start=jnp.asarray(0, dtype=jnp.bool_),
        )
    )


def run_scalar_warm_start_after_zero_start_failure(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthZeroStartOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run warm-start fallback after zero-start fallback fails."""
    dispatch_operands = operands.fallback_operands.dispatch_operands
    warm_start_result = run_scalar_warm_start_newton_raphson_attempt(dispatch_operands)
    return build_single_variant_regenie_approximate_firth_result_from_operands(
        regenie2_binary_firth_types.ScalarApproximateFirthResultOperands(
            dispatch_operands=dispatch_operands,
            pseudo_result=operands.fallback_operands.pseudo_result,
            zero_start_result=operands.zero_start_result,
            warm_start_result=warm_start_result,
            run_zero_start=jnp.asarray(1, dtype=jnp.bool_),
            run_warm_start=jnp.asarray(1, dtype=jnp.bool_),
        )
    )


def run_scalar_zero_start_then_maybe_warm_start(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthFallbackOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run zero-start fallback and warm-start only if zero-start fails."""
    zero_start_result = run_scalar_zero_start_newton_raphson_attempt(operands.dispatch_operands)
    zero_start_operands = regenie2_binary_firth_types.ScalarApproximateFirthZeroStartOperands(
        fallback_operands=operands,
        zero_start_result=zero_start_result,
    )
    return jax.lax.cond(
        zero_start_result.valid,
        build_scalar_zero_start_success_result,
        run_scalar_warm_start_after_zero_start_failure,
        zero_start_operands,
    )


def run_scalar_fallback_cascade(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthFallbackOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run scalar Newton-Raphson fallbacks after pseudo-Firth failure."""
    dispatch_operands = operands.dispatch_operands
    run_zero_start = dispatch_operands.sparse_correction & (
        jnp.abs(dispatch_operands.warm_start_beta) > jnp.asarray(0.0, dtype=dispatch_operands.offset_vector.dtype)
    )
    return jax.lax.cond(
        run_zero_start,
        run_scalar_zero_start_then_maybe_warm_start,
        run_scalar_warm_start_without_zero_start,
        operands,
    )


def run_scalar_active_solver(
    operands: regenie2_binary_firth_types.ScalarApproximateFirthDispatchOperands,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Run pseudo-Firth and lazy Newton-Raphson fallbacks for an active lane."""
    pseudo_result = run_scalar_pseudo_firth_attempt(operands)
    fallback_operands = regenie2_binary_firth_types.ScalarApproximateFirthFallbackOperands(
        dispatch_operands=operands,
        pseudo_result=pseudo_result,
    )
    return jax.lax.cond(
        pseudo_result.valid,
        build_scalar_pseudo_firth_success_result,
        run_scalar_fallback_cascade,
        fallback_operands,
    )


def fit_single_variant_regenie_approximate_firth(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    carrier_sample_mask: jax.Array,
    sparse_correction: jax.Array,
    warm_start_beta: jax.Array,
    skip_firth: jax.Array,
    null_failed: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Fit one REGENIE-equivalent scalar approximate-Firth candidate."""
    scalar_dtype = offset_vector.dtype
    return fit_single_variant_regenie_approximate_firth_with_solver_parameters(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        carrier_sample_mask=carrier_sample_mask,
        sparse_correction=sparse_correction,
        warm_start_beta=warm_start_beta,
        skip_firth=skip_firth,
        null_failed=null_failed,
        solver_parameters=build_scalar_approximate_firth_solver_parameters(kernel_config, scalar_dtype),
    )


def fit_single_variant_regenie_approximate_firth_with_solver_parameters(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    carrier_sample_mask: jax.Array,
    sparse_correction: jax.Array,
    warm_start_beta: jax.Array,
    skip_firth: jax.Array,
    null_failed: jax.Array,
    solver_parameters: regenie2_binary_firth_types.ScalarApproximateFirthSolverParameters,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Fit one scalar approximate-Firth candidate with explicit solver policy."""
    scalar_dtype = offset_vector.dtype
    phenotype_vector = jnp.asarray(phenotype_vector, dtype=scalar_dtype)
    genotype_vector = jnp.asarray(genotype_vector, dtype=scalar_dtype)
    all_sample_mask = jnp.ones_like(phenotype_vector, dtype=jnp.bool_)
    active_sample_mask = jnp.where(sparse_correction, carrier_sample_mask, all_sample_mask)
    null_probability_vector = regenie2_binary_logistic.compute_regenie_logistic_probability(offset_vector)
    full_null_deviance = regenie2_binary_logistic.compute_logistic_deviance(
        phenotype_vector, null_probability_vector, all_sample_mask
    )
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
    scalar_dtype = offset_vector.dtype
    phenotype_vector = jnp.asarray(phenotype_vector, dtype=scalar_dtype)
    genotype_vector = jnp.asarray(genotype_vector, dtype=scalar_dtype)
    offset_vector = jnp.asarray(offset_vector, dtype=scalar_dtype)
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
        full_null_deviance=jnp.asarray(full_null_deviance, dtype=scalar_dtype),
        non_active_deviance=jnp.asarray(full_null_deviance, dtype=scalar_dtype) - active_null_deviance,
        sparse_correction=jnp.ones((), dtype=jnp.bool_),
        warm_start_beta=warm_start_beta,
        skip_firth=skip_firth,
        null_failed=null_failed,
        solver_parameters=solver_parameters,
    )


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
    scalar_dtype = offset_vector.dtype
    phenotype_vector = jnp.asarray(phenotype_vector, dtype=scalar_dtype)
    genotype_vector = jnp.asarray(genotype_vector, dtype=scalar_dtype)
    offset_vector = jnp.asarray(offset_vector, dtype=scalar_dtype)
    active_sample_mask = jnp.asarray(active_sample_mask, dtype=jnp.bool_)
    full_null_deviance = jnp.asarray(full_null_deviance, dtype=scalar_dtype)
    non_active_deviance = jnp.asarray(non_active_deviance, dtype=scalar_dtype)
    warm_start_beta = jnp.asarray(warm_start_beta, dtype=scalar_dtype)
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
        skip_firth=skip_firth,
        null_failed=null_failed,
        solver_parameters=solver_parameters,
    )
    return jax.lax.cond(
        solver_active,
        run_scalar_active_solver,
        build_scalar_inactive_solver_result,
        dispatch_operands,
    )
