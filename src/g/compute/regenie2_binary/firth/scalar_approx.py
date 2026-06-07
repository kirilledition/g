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


def compute_scalar_firth_components(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    beta: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.ScalarFirthComponents:
    """Compute REGENIE scalar approximate-Firth quantities at one beta."""
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
        & (genotype_information > kernel_config.numerical.minimum_variance)
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
    maximum_iterations: int,
    maximum_step_size: jax.Array,
) -> regenie2_binary_firth_types.ScalarPseudoLogisticState:
    """Run REGENIE's inner pseudo-response scalar logistic update."""

    def condition_function(state: regenie2_binary_firth_types.ScalarPseudoLogisticState) -> jax.Array:
        return (state.iteration_count < maximum_iterations) & (~state.converged) & (~state.failed)

    def body_function(
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
        updated_score = jnp.sum(
            jnp.where(active_sample_mask, genotype_vector * (adjusted_response - probability_vector), 0.0)
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
        failure_code = jnp.where(
            step_increased,
            jnp.asarray(2, dtype=jnp.int32),
            jnp.where(
                probability_failed | numerical_failed,
                jnp.asarray(3, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        return regenie2_binary_firth_types.ScalarPseudoLogisticState(
            beta=jnp.where(failed, state.beta, updated_beta),
            score=jnp.where(failed, state.score, updated_score),
            genotype_information=jnp.where(failed, state.genotype_information, updated_genotype_information),
            step_size=step_size,
            previous_step_size=absolute_step_size,
            iteration_count=state.iteration_count + jnp.asarray(1, dtype=jnp.int32),
            converged=(jnp.abs(updated_score) < tolerance) & (~failed),
            failed=failed,
            failure_code=failure_code,
        )

    return jax.lax.while_loop(
        condition_function,
        body_function,
        regenie2_binary_firth_types.ScalarPseudoLogisticState(
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
    )


def fit_scalar_pseudo_firth(
    *,
    deviance_null: jax.Array,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    initial_beta: jax.Array,
    maximum_iterations: int,
    tolerance: jax.Array,
    inner_maximum_iterations: int,
    maximum_step_size: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.ScalarFirthAttemptResult:
    """Run REGENIE's scalar pseudo-Firth approximate correction."""

    def condition_function(state: regenie2_binary_firth_types.ScalarPseudoFirthState) -> jax.Array:
        return (state.outer_iteration_count < maximum_iterations) & (~state.converged) & (~state.failed)

    def body_function(
        state: regenie2_binary_firth_types.ScalarPseudoFirthState,
    ) -> regenie2_binary_firth_types.ScalarPseudoFirthState:
        components = compute_scalar_firth_components(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            beta=state.beta,
            kernel_config=kernel_config,
        )
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
        leverage_vector = components.genotype_information_diagonal / components.genotype_information
        adjusted_response = phenotype_vector + leverage_vector * (
            regenie2_binary_config.BINARY_CASE_THRESHOLD - components.probability_vector
        )
        logistic_state = fit_scalar_pseudo_logistic_step(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            active_sample_mask=active_sample_mask,
            offset_vector=offset_vector,
            adjusted_response=adjusted_response,
            initial_score=components.score,
            initial_genotype_information=components.genotype_information,
            initial_beta=state.beta,
            tolerance=tolerance,
            maximum_iterations=inner_maximum_iterations,
            maximum_step_size=maximum_step_size,
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
        return regenie2_binary_firth_types.ScalarPseudoFirthState(
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
        )

    initial_components = compute_scalar_firth_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=initial_beta,
        kernel_config=kernel_config,
    )
    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        regenie2_binary_firth_types.ScalarPseudoFirthState(
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
    )
    final_components = compute_scalar_firth_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=final_state.beta,
        kernel_config=kernel_config,
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


def run_scalar_line_search(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    current_beta: jax.Array,
    current_penalized_deviance: jax.Array,
    initial_step_size: jax.Array,
    maximum_attempts: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.ScalarLineSearchState:
    """Run REGENIE scalar NR step-halving against penalized deviance."""

    def condition_function(state: regenie2_binary_firth_types.ScalarLineSearchState) -> jax.Array:
        return (state.attempt_count < maximum_attempts) & (~state.accepted) & state.valid

    def body_function(
        state: regenie2_binary_firth_types.ScalarLineSearchState,
    ) -> regenie2_binary_firth_types.ScalarLineSearchState:
        adjusted_step_size = jnp.where(state.attempt_count > 0, state.step_size / 2.0, state.step_size)
        candidate_beta = current_beta + adjusted_step_size
        components = compute_scalar_firth_components(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            beta=candidate_beta,
            kernel_config=kernel_config,
        )
        accepted = components.valid & (components.penalized_deviance < current_penalized_deviance)
        return regenie2_binary_firth_types.ScalarLineSearchState(
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
        )

    initial_components = compute_scalar_firth_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=current_beta,
        kernel_config=kernel_config,
    )
    return jax.lax.while_loop(
        condition_function,
        body_function,
        regenie2_binary_firth_types.ScalarLineSearchState(
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
    )


def fit_scalar_newton_raphson_firth(
    *,
    deviance_null: jax.Array,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    initial_beta: jax.Array,
    maximum_iterations: int,
    tolerance: jax.Array,
    maximum_step_size: jax.Array,
    line_search_maximum_attempts: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.ScalarFirthAttemptResult:
    """Run REGENIE's scalar Newton-Raphson approximate-Firth fallback."""

    def condition_function(state: regenie2_binary_firth_types.ScalarNewtonRaphsonState) -> jax.Array:
        return (state.iteration_count < maximum_iterations) & (~state.converged) & (~state.failed)

    def body_function(
        state: regenie2_binary_firth_types.ScalarNewtonRaphsonState,
    ) -> regenie2_binary_firth_types.ScalarNewtonRaphsonState:
        leverage_vector = state.genotype_information_diagonal / state.genotype_information
        score = jnp.sum(
            jnp.where(
                active_sample_mask,
                genotype_vector
                * (phenotype_vector - state.probability_vector + leverage_vector * (0.5 - state.probability_vector)),
                0.0,
            )
        )
        updated_iteration_count = state.iteration_count + jnp.asarray(1, dtype=jnp.int32)
        converged = (jnp.abs(score) < tolerance) & (updated_iteration_count >= 2)
        raw_step_size = score / state.genotype_information
        step_scale = jnp.maximum(jnp.abs(raw_step_size) / maximum_step_size, 1.0)
        step_size = raw_step_size / step_scale
        line_search_state = run_scalar_line_search(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            current_beta=state.beta,
            current_penalized_deviance=state.penalized_deviance,
            initial_step_size=step_size,
            maximum_attempts=line_search_maximum_attempts,
            kernel_config=kernel_config,
        )
        line_search_failed = (~converged) & (~line_search_state.accepted)
        updated_beta = jnp.where(converged | line_search_failed, state.beta, line_search_state.beta)
        updated_components = compute_scalar_firth_components(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            beta=updated_beta,
            kernel_config=kernel_config,
        )
        failed = (~state.failed) & (line_search_failed | (~updated_components.valid) | (~line_search_state.valid))
        return regenie2_binary_firth_types.ScalarNewtonRaphsonState(
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
        )

    initial_components = compute_scalar_firth_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=initial_beta,
        kernel_config=kernel_config,
    )
    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        regenie2_binary_firth_types.ScalarNewtonRaphsonState(
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
    )
    final_components = compute_scalar_firth_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=final_state.beta,
        kernel_config=kernel_config,
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
    return fit_single_variant_regenie_approximate_firth_with_active_samples(
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
        kernel_config=kernel_config,
    )


def fit_single_variant_regenie_approximate_firth_compact_carriers(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_carrier_slot_mask: jax.Array,
    full_null_deviance: jax.Array,
    warm_start_beta: jax.Array,
    skip_firth: jax.Array,
    null_failed: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Fit one sparse approximate-Firth lane over fixed-capacity carrier slots."""
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
    return fit_single_variant_regenie_approximate_firth_with_active_samples(
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
        kernel_config=kernel_config,
    )


def fit_single_variant_regenie_approximate_firth_with_active_samples(
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
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Fit one approximate-Firth candidate over caller-selected active samples."""
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
    tolerance = jnp.asarray(kernel_config.approximate_firth.gradient_tolerance, dtype=scalar_dtype)
    pseudo_maximum_iterations = min(
        kernel_config.approximate_firth.maximum_iterations // 2,
        kernel_config.approximate_firth.pseudo_maximum_iterations,
    )
    newton_maximum_iterations = kernel_config.approximate_firth.maximum_iterations // 2
    maximum_step_size = jnp.asarray(kernel_config.approximate_firth.maximum_step_size, dtype=scalar_dtype)
    pseudo_result = fit_scalar_pseudo_firth(
        deviance_null=deviance_null,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        initial_beta=warm_start_beta,
        maximum_iterations=pseudo_maximum_iterations,
        tolerance=tolerance,
        inner_maximum_iterations=kernel_config.approximate_firth.pseudo_inner_maximum_iterations,
        maximum_step_size=maximum_step_size,
        kernel_config=kernel_config,
    )
    run_zero_start = (
        (~pseudo_result.valid) & sparse_correction & (jnp.abs(warm_start_beta) > jnp.asarray(0.0, dtype=scalar_dtype))
    )
    zero_start_result = fit_scalar_newton_raphson_firth(
        deviance_null=deviance_null,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        initial_beta=jnp.asarray(0.0, dtype=scalar_dtype),
        maximum_iterations=kernel_config.approximate_firth.newton_raphson_zero_start_iterations,
        tolerance=tolerance,
        maximum_step_size=maximum_step_size,
        line_search_maximum_attempts=kernel_config.approximate_firth.line_search_maximum_attempts,
        kernel_config=kernel_config,
    )
    run_warm_start = (~pseudo_result.valid) & (~(run_zero_start & zero_start_result.valid))
    warm_start_result = fit_scalar_newton_raphson_firth(
        deviance_null=deviance_null,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        initial_beta=warm_start_beta,
        maximum_iterations=newton_maximum_iterations,
        tolerance=tolerance,
        maximum_step_size=maximum_step_size,
        line_search_maximum_attempts=kernel_config.approximate_firth.line_search_maximum_attempts,
        kernel_config=kernel_config,
    )
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
