"""Full-model Firth logistic solver for REGENIE step 2 binary tests."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g import types
from g.compute.common import linalg, pvalue
from g.compute.regenie2_binary.firth import common as regenie2_binary_firth_common
from g.compute.regenie2_binary.firth import line_search as regenie2_binary_firth_line_search
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import types as regenie2_binary_types

MINIMUM_PROBABILITY = 1.0e-6
MINIMUM_VARIANCE = 1.0e-8
BINARY_CASE_THRESHOLD = 0.5


def compute_logistic_probability(linear_predictor: jax.Array) -> jax.Array:
    """Compute clipped logistic probabilities."""
    probability = jax.nn.sigmoid(linear_predictor)
    return jnp.clip(probability, MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)


def build_full_model_information_matrix(
    *,
    covariate_information_matrix: jax.Array,
    cross_information_vector: jax.Array,
    genotype_information: jax.Array,
) -> jax.Array:
    """Build a full-model information matrix from block components."""
    top_block = jnp.concatenate(
        [
            covariate_information_matrix,
            cross_information_vector[:, None],
        ],
        axis=1,
    )
    bottom_block = jnp.concatenate(
        [
            cross_information_vector[None, :],
            genotype_information[None, None],
        ],
        axis=1,
    )
    return jnp.concatenate([top_block, bottom_block], axis=0)


def compute_information_components(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    probability_vector: jax.Array,
) -> regenie2_binary_firth_types.InformationComponents:
    """Compute full information components for one genotype lane."""
    weight_vector = jnp.maximum(probability_vector * (1.0 - probability_vector), MINIMUM_VARIANCE)
    weighted_genotype_vector = weight_vector * genotype_vector
    covariate_information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
    cross_information_vector = weighted_genotype_vector @ covariate_matrix
    genotype_information = jnp.dot(weighted_genotype_vector, genotype_vector)
    information_matrix = build_full_model_information_matrix(
        covariate_information_matrix=covariate_information_matrix,
        cross_information_vector=cross_information_vector,
        genotype_information=genotype_information,
    )
    return regenie2_binary_firth_types.InformationComponents(
        covariate_information_matrix=covariate_information_matrix,
        cross_information_vector=cross_information_vector,
        genotype_information=genotype_information,
        information_matrix=information_matrix,
    )


def compute_weighted_full_model_information_components(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    weight_vector: jax.Array,
) -> regenie2_binary_firth_types.InformationComponents:
    """Compute full-model information blocks for one explicit weight vector."""
    weighted_genotype_vector = weight_vector * genotype_vector
    covariate_information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
    cross_information_vector = weighted_genotype_vector @ covariate_matrix
    genotype_information = jnp.dot(weighted_genotype_vector, genotype_vector)
    return regenie2_binary_firth_types.InformationComponents(
        covariate_information_matrix=covariate_information_matrix,
        cross_information_vector=cross_information_vector,
        genotype_information=genotype_information,
        information_matrix=build_full_model_information_matrix(
            covariate_information_matrix=covariate_information_matrix,
            cross_information_vector=cross_information_vector,
            genotype_information=genotype_information,
        ),
    )


def compute_full_model_adjusted_weight_components(
    full_design_matrix: jax.Array,
    probability_vector: jax.Array,
    information_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> regenie2_binary_firth_types.AdjustedWeightComponents:
    """Compute leverage-adjusted Firth weights for one full model."""
    variance_vector = jnp.maximum(probability_vector * (1.0 - probability_vector), MINIMUM_VARIANCE)
    projected_design_matrix = linalg.solve_from_positive_definite_matrix(
        information_matrix,
        full_design_matrix.T,
    ).T
    leverage_vector = variance_vector * jnp.einsum("ij,ij->i", projected_design_matrix, full_design_matrix)
    adjusted_weight_vector = (phenotype_vector - probability_vector) + leverage_vector * (
        BINARY_CASE_THRESHOLD - probability_vector
    )
    second_weight_vector = (1.0 + leverage_vector) * variance_vector
    return regenie2_binary_firth_types.AdjustedWeightComponents(
        leverage_vector=leverage_vector,
        adjusted_weight_vector=adjusted_weight_vector,
        second_weight_vector=second_weight_vector,
    )


def compute_full_model_adjusted_weight_components_from_parts(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    probability_vector: jax.Array,
    information_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> regenie2_binary_firth_types.AdjustedWeightComponents:
    """Compute full-model Firth weights without materializing a full design matrix."""
    variance_vector = jnp.maximum(probability_vector * (1.0 - probability_vector), MINIMUM_VARIANCE)
    stacked_design_transpose = jnp.concatenate([covariate_matrix.T, genotype_vector[None, :]], axis=0)
    projected_design_transpose = linalg.solve_from_positive_definite_matrix(
        information_matrix,
        stacked_design_transpose,
    )
    projected_covariate_matrix = projected_design_transpose[:-1, :].T
    projected_genotype_vector = projected_design_transpose[-1, :]
    leverage_vector = variance_vector * (
        jnp.einsum("ij,ij->i", projected_covariate_matrix, covariate_matrix)
        + projected_genotype_vector * genotype_vector
    )
    adjusted_weight_vector = (phenotype_vector - probability_vector) + leverage_vector * (
        BINARY_CASE_THRESHOLD - probability_vector
    )
    second_weight_vector = (1.0 + leverage_vector) * variance_vector
    return regenie2_binary_firth_types.AdjustedWeightComponents(
        leverage_vector=leverage_vector,
        adjusted_weight_vector=adjusted_weight_vector,
        second_weight_vector=second_weight_vector,
    )


def compute_full_model_score_components(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    score_weight_vector: jax.Array,
) -> regenie2_binary_firth_types.FullModelScoreComponents:
    """Compute covariate and genotype score blocks without a full design matrix."""
    return regenie2_binary_firth_types.FullModelScoreComponents(
        covariate_score=covariate_matrix.T @ score_weight_vector,
        genotype_score=jnp.dot(genotype_vector, score_weight_vector),
    )


def compute_covariate_only_adjusted_weight_components(
    covariate_matrix: jax.Array,
    probability_vector: jax.Array,
    information_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> regenie2_binary_firth_types.AdjustedWeightComponents:
    """Compute leverage-adjusted Firth weights for the covariate-only null model."""
    variance_vector = jnp.maximum(probability_vector * (1.0 - probability_vector), MINIMUM_VARIANCE)
    projected_covariate_matrix = linalg.solve_from_positive_definite_matrix(
        information_matrix,
        covariate_matrix.T,
    ).T
    leverage_vector = variance_vector * jnp.einsum("ij,ij->i", projected_covariate_matrix, covariate_matrix)
    adjusted_weight_vector = (phenotype_vector - probability_vector) + leverage_vector * (
        BINARY_CASE_THRESHOLD - probability_vector
    )
    second_weight_vector = (1.0 + leverage_vector) * variance_vector
    return regenie2_binary_firth_types.AdjustedWeightComponents(
        leverage_vector=leverage_vector,
        adjusted_weight_vector=adjusted_weight_vector,
        second_weight_vector=second_weight_vector,
    )


def fit_single_variant_firth_logistic_regression(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth: jax.Array,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_types.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Fit one Firth logistic model for a candidate variant."""
    use_block_firth_math = kernel_config.use_block_firth_math
    if use_block_firth_math:
        coefficient_count = covariate_matrix.shape[1] + 1
    else:
        full_design_matrix = jnp.concatenate([covariate_matrix, genotype_vector[:, None]], axis=1)
        coefficient_count = full_design_matrix.shape[1]
    unit_genotype_vector = jnp.zeros((coefficient_count,), dtype=jnp.float32).at[-1].set(1.0)

    def compute_probability_vector(coefficients: jax.Array) -> jax.Array:
        linear_predictor = covariate_matrix @ coefficients[:-1] + genotype_vector * coefficients[-1] + loco_offset
        return compute_logistic_probability(linear_predictor)

    def compute_full_penalized_log_likelihood(coefficients: jax.Array) -> jax.Array:
        probability_vector = compute_probability_vector(coefficients)
        information_components = compute_information_components(
            covariate_matrix=covariate_matrix,
            genotype_vector=genotype_vector,
            probability_vector=probability_vector,
        )
        information_matrix = (
            information_components.information_matrix
            + jnp.eye(
                information_components.information_matrix.shape[0],
                dtype=jnp.float32,
            )
            * MINIMUM_VARIANCE
        )
        information_cholesky_factor = jnp.linalg.cholesky(information_matrix)
        return regenie2_binary_firth_common.compute_firth_penalized_log_likelihood_from_cholesky(
            probability_vector=probability_vector,
            phenotype_vector=phenotype_vector,
            information_cholesky_factor=information_cholesky_factor,
        )

    def condition_function(state: regenie2_binary_firth_types.FirthState) -> jax.Array:
        return (
            (state.iteration_count < kernel_config.firth_maximum_iterations)
            & (~state.converged)
            & (~state.failed)
            & (~skip_firth)
        )

    def body_function(
        state: regenie2_binary_firth_types.FirthState,
    ) -> regenie2_binary_firth_types.FirthState:
        probability_vector = compute_probability_vector(state.coefficients)
        information_components = compute_information_components(
            covariate_matrix=covariate_matrix,
            genotype_vector=genotype_vector,
            probability_vector=probability_vector,
        )
        information_matrix = (
            information_components.information_matrix
            + jnp.eye(
                information_components.information_matrix.shape[0],
                dtype=jnp.float32,
            )
            * MINIMUM_VARIANCE
        )
        information_cholesky_factor = jnp.linalg.cholesky(information_matrix)
        current_penalized_log_likelihood = (
            regenie2_binary_firth_common.compute_firth_penalized_log_likelihood_from_cholesky(
                probability_vector=probability_vector,
                phenotype_vector=phenotype_vector,
                information_cholesky_factor=information_cholesky_factor,
            )
        )
        current_failed = (~jnp.isfinite(current_penalized_log_likelihood)) | (
            ~jnp.all(jnp.isfinite(state.coefficients))
        )
        if use_block_firth_math:
            adjusted_weight_components = compute_full_model_adjusted_weight_components_from_parts(
                covariate_matrix=covariate_matrix,
                genotype_vector=genotype_vector,
                probability_vector=probability_vector,
                information_matrix=information_matrix,
                phenotype_vector=phenotype_vector,
            )
            adjusted_score_components = compute_full_model_score_components(
                covariate_matrix=covariate_matrix,
                genotype_vector=genotype_vector,
                score_weight_vector=adjusted_weight_components.adjusted_weight_vector,
            )
            adjusted_score = jnp.concatenate(
                [adjusted_score_components.covariate_score, adjusted_score_components.genotype_score[None]],
                axis=0,
            )
            second_hessian_components = compute_weighted_full_model_information_components(
                covariate_matrix=covariate_matrix,
                genotype_vector=genotype_vector,
                weight_vector=adjusted_weight_components.second_weight_vector,
            )
            second_hessian = second_hessian_components.information_matrix
        else:
            adjusted_weight_components = compute_full_model_adjusted_weight_components(
                full_design_matrix=full_design_matrix,
                probability_vector=probability_vector,
                information_matrix=information_matrix,
                phenotype_vector=phenotype_vector,
            )
            adjusted_score = full_design_matrix.T @ adjusted_weight_components.adjusted_weight_vector
            second_hessian = (
                full_design_matrix.T * adjusted_weight_components.second_weight_vector
            ) @ full_design_matrix
        second_hessian = second_hessian + jnp.eye(second_hessian.shape[0], dtype=jnp.float32) * MINIMUM_VARIANCE
        coefficient_step = linalg.solve_from_positive_definite_matrix(second_hessian, adjusted_score)
        current_failed = (
            current_failed | (~jnp.all(jnp.isfinite(adjusted_score))) | (~jnp.all(jnp.isfinite(coefficient_step)))
        )
        maximum_coefficient_step = jnp.max(jnp.abs(coefficient_step))
        step_scale = jnp.minimum(
            1.0, kernel_config.firth_maximum_step_size / jnp.maximum(maximum_coefficient_step, MINIMUM_VARIANCE)
        )
        scaled_coefficient_step = coefficient_step * step_scale
        backtracking_result = regenie2_binary_firth_line_search.run_firth_step_halving(
            current_coefficients=state.coefficients,
            current_penalized_log_likelihood=state.penalized_log_likelihood,
            coefficient_step=scaled_coefficient_step,
            evaluate_penalized_log_likelihood=compute_full_penalized_log_likelihood,
            kernel_config=kernel_config,
        )
        step_halving_failed = (~current_failed) & backtracking_result.exhausted
        updated_failed = current_failed | step_halving_failed
        updated_converged = regenie2_binary_firth_line_search.compute_firth_convergence_mask(
            current_penalized_log_likelihood=state.penalized_log_likelihood,
            candidate_penalized_log_likelihood=backtracking_result.penalized_log_likelihood,
            coefficient_step=backtracking_result.coefficient_step,
            adjusted_score=adjusted_score,
            kernel_config=kernel_config,
        ) & (~updated_failed)
        updated_reason_code = jnp.where(
            step_halving_failed,
            regenie2_binary_firth_types.FirthConvergenceReason.STEP_HALVING_EXHAUSTED.value,
            jnp.where(
                current_failed,
                regenie2_binary_firth_types.FirthConvergenceReason.NUMERICAL_FAILURE.value,
                jnp.where(
                    updated_converged,
                    regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
                    regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                ),
            ),
        ).astype(jnp.int32)
        return regenie2_binary_firth_types.FirthState(
            coefficients=jnp.where(updated_failed, state.coefficients, backtracking_result.coefficients),
            penalized_log_likelihood=jnp.where(
                updated_failed,
                state.penalized_log_likelihood,
                backtracking_result.penalized_log_likelihood,
            ),
            converged=updated_converged,
            failed=updated_failed,
            iteration_count=state.iteration_count + jnp.asarray(1, dtype=jnp.int32),
            termination_reason_code=updated_reason_code,
        )

    initial_probability_vector = compute_probability_vector(initial_coefficients)
    initial_information_components = compute_information_components(
        covariate_matrix=covariate_matrix,
        genotype_vector=genotype_vector,
        probability_vector=initial_probability_vector,
    )
    initial_information_matrix = (
        initial_information_components.information_matrix
        + jnp.eye(
            initial_information_components.information_matrix.shape[0],
            dtype=jnp.float32,
        )
        * MINIMUM_VARIANCE
    )
    initial_information_cholesky_factor = jnp.linalg.cholesky(initial_information_matrix)
    initial_penalized_log_likelihood = (
        regenie2_binary_firth_common.compute_firth_penalized_log_likelihood_from_cholesky(
            probability_vector=initial_probability_vector,
            phenotype_vector=phenotype_vector,
            information_cholesky_factor=initial_information_cholesky_factor,
        )
    )
    initial_full_failed = (~jnp.isfinite(initial_penalized_log_likelihood)) | (
        ~jnp.all(jnp.isfinite(initial_coefficients))
    )
    initial_null_failed = (~skip_firth) & (~jnp.isfinite(null_penalized_log_likelihood))
    initial_failed = initial_full_failed | initial_null_failed
    initial_reason_code = jnp.where(
        initial_null_failed,
        regenie2_binary_firth_types.FirthConvergenceReason.NULL_FAILURE.value,
        jnp.where(
            initial_full_failed,
            regenie2_binary_firth_types.FirthConvergenceReason.NUMERICAL_FAILURE.value,
            regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
        ),
    ).astype(jnp.int32)
    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        regenie2_binary_firth_types.FirthState(
            coefficients=initial_coefficients,
            penalized_log_likelihood=jnp.where(skip_firth, 0.0, initial_penalized_log_likelihood),
            converged=skip_firth,
            failed=initial_failed,
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            termination_reason_code=initial_reason_code,
        ),
    )
    final_probability_vector = compute_probability_vector(final_state.coefficients)
    final_information_components = compute_information_components(
        covariate_matrix=covariate_matrix,
        genotype_vector=genotype_vector,
        probability_vector=final_probability_vector,
    )
    final_information_matrix = (
        final_information_components.information_matrix
        + jnp.eye(
            final_information_components.information_matrix.shape[0],
            dtype=jnp.float32,
        )
        * MINIMUM_VARIANCE
    )
    final_information_cholesky_factor = jnp.linalg.cholesky(final_information_matrix)
    final_penalized_log_likelihood = regenie2_binary_firth_common.compute_firth_penalized_log_likelihood_from_cholesky(
        probability_vector=final_probability_vector,
        phenotype_vector=phenotype_vector,
        information_cholesky_factor=final_information_cholesky_factor,
    )
    if use_block_firth_math:
        final_adjusted_weight_components = compute_full_model_adjusted_weight_components_from_parts(
            covariate_matrix=covariate_matrix,
            genotype_vector=genotype_vector,
            probability_vector=final_probability_vector,
            information_matrix=final_information_matrix,
            phenotype_vector=phenotype_vector,
        )
        final_second_hessian_components = compute_weighted_full_model_information_components(
            covariate_matrix=covariate_matrix,
            genotype_vector=genotype_vector,
            weight_vector=final_adjusted_weight_components.second_weight_vector,
        )
        final_second_hessian = final_second_hessian_components.information_matrix
    else:
        final_adjusted_weight_components = compute_full_model_adjusted_weight_components(
            full_design_matrix=full_design_matrix,
            probability_vector=final_probability_vector,
            information_matrix=final_information_matrix,
            phenotype_vector=phenotype_vector,
        )
        final_second_hessian = (
            full_design_matrix.T * final_adjusted_weight_components.second_weight_vector
        ) @ full_design_matrix
    final_second_hessian = (
        final_second_hessian + jnp.eye(final_second_hessian.shape[0], dtype=jnp.float32) * MINIMUM_VARIANCE
    )
    genotype_variance = linalg.solve_from_positive_definite_matrix(final_second_hessian, unit_genotype_vector)[-1]
    beta = final_state.coefficients[-1]
    standard_error = jnp.sqrt(jnp.where(genotype_variance > 0.0, genotype_variance, jnp.nan))
    chi_squared = jnp.maximum(2.0 * (final_penalized_log_likelihood - null_penalized_log_likelihood), 0.0)
    log10_p_value = jnp.asarray(pvalue.chi_squared_to_log10_p_value(chi_squared), dtype=jnp.float64)
    valid_mask = (
        (~skip_firth)
        & final_state.converged
        & (~final_state.failed)
        & jnp.isfinite(beta)
        & jnp.isfinite(standard_error)
        & jnp.isfinite(chi_squared)
        & jnp.isfinite(log10_p_value)
        & (standard_error > 0.0)
    )
    maximum_iteration_failure_mask = (
        (~skip_firth)
        & (~final_state.converged)
        & (~final_state.failed)
        & (final_state.iteration_count >= kernel_config.firth_maximum_iterations)
    )
    invalid_statistic_failure_mask = (~skip_firth) & final_state.converged & (~final_state.failed) & (~valid_mask)
    convergence_reason_code = jnp.where(
        maximum_iteration_failure_mask,
        regenie2_binary_firth_types.FirthConvergenceReason.MAX_ITERATIONS.value,
        jnp.where(
            invalid_statistic_failure_mask,
            regenie2_binary_firth_types.FirthConvergenceReason.INVALID_STATISTIC.value,
            final_state.termination_reason_code,
        ),
    ).astype(jnp.int32)
    failure_code = regenie2_binary_firth_common.map_firth_reason_code_to_failure_code(convergence_reason_code)
    return regenie2_binary_firth_types.FirthVariantResult(
        beta=jnp.asarray(jnp.where(skip_firth, jnp.nan, beta), dtype=jnp.float64),
        standard_error=jnp.asarray(jnp.where(skip_firth, jnp.nan, standard_error), dtype=jnp.float64),
        chi_squared=jnp.asarray(jnp.where(skip_firth, jnp.nan, chi_squared), dtype=jnp.float64),
        log10_p_value=jnp.asarray(jnp.where(skip_firth, jnp.nan, log10_p_value), dtype=jnp.float64),
        penalized_log_likelihood=jnp.asarray(
            jnp.where(skip_firth, jnp.nan, final_penalized_log_likelihood), dtype=jnp.float64
        ),
        converged_mask=jnp.where(skip_firth, jnp.asarray(0, dtype=jnp.bool_), final_state.converged),
        valid_mask=valid_mask,
        iteration_count=jnp.where(skip_firth, jnp.asarray(0, dtype=jnp.int32), final_state.iteration_count),
        failure_code=jnp.where(skip_firth, types.FirthFailureCode.NONE.value, failure_code).astype(jnp.int32),
        convergence_reason_code=jnp.where(
            skip_firth,
            regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
            convergence_reason_code,
        ),
        correction_code=jnp.where(
            skip_firth | (~valid_mask),
            types.FirthCorrectionCode.NONE.value,
            types.FirthCorrectionCode.NEWTON_RAPHSON_WARM_START.value,
        ).astype(jnp.int32),
        sparse_correction_mask=jnp.asarray(0, dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.asarray(0, dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.asarray(0, dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.where(
            skip_firth,
            jnp.asarray(0, dtype=jnp.int32),
            final_state.iteration_count,
        ),
    )
