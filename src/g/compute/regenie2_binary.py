"""REGENIE step 2 binary score-test kernel with device-resident Firth fallback."""

from __future__ import annotations

import functools
import os
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g import models, types
from g.compute import regenie2_linear

MINIMUM_PROBABILITY = 1.0e-6
MINIMUM_VARIANCE = 1.0e-8
DEFAULT_MAXIMUM_NULL_ITERATIONS = 50
REGENIE_SCORE_CHISQ_THRESHOLD = 3.841458820694124
EXTRA_CODE_SCORE = 0
EXTRA_CODE_FIRTH = 1
EXTRA_CODE_SPA = 2
EXTRA_CODE_TEST_FAIL = 3
INITIAL_RESPONSE_SCALE = 4.863891244002886
BINARY_CASE_THRESHOLD = 0.5
ALLELE_COUNT_MULTIPLIER = 2.0
FIRTH_GRADIENT_TOLERANCE = 1.0e-4
FIRTH_COEFFICIENT_TOLERANCE = 1.0e-4
FIRTH_LIKELIHOOD_TOLERANCE = 1.0e-4
FIRTH_MAXIMUM_STEP_SIZE = 5.0
FIRTH_MAXIMUM_ITERATIONS = 50
DEFAULT_FIRTH_BATCH_SIZE = 64

BinaryScoreTestChunkComputeFunction = typing.Callable[
    [models.Regenie2BinaryChromosomeState, jax.Array, types.RegenieBinaryCorrection],
    models.Regenie2BinaryChunkResult,
]
BinaryChunkComputeFunction = typing.Callable[
    [models.Regenie2BinaryChromosomeState, jax.Array, types.RegenieBinaryCorrection],
    models.Regenie2BinaryChunkResult,
]


@functools.cache
def get_firth_batch_size() -> int:
    """Resolve the active fixed Firth batch size from the environment."""
    raw_value = os.environ.get("G_REGENIE2_BINARY_FIRTH_BATCH_SIZE")
    if raw_value is None:
        return DEFAULT_FIRTH_BATCH_SIZE
    parsed_value = int(raw_value)
    if parsed_value <= 0:
        message = "G_REGENIE2_BINARY_FIRTH_BATCH_SIZE must be positive."
        raise ValueError(message)
    return parsed_value


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthState:
    """State container for one Firth-regression lane.

    Attributes:
        coefficients: Current coefficient estimates.
        converged: Whether the solver converged.
        failed: Whether the solver hit an unrecoverable numerical failure.
        iteration_count: Number of update steps performed.
        previous_penalized_log_likelihood: Previous penalized log-likelihood.

    """

    coefficients: jax.Array
    converged: jax.Array
    failed: jax.Array
    iteration_count: jax.Array
    previous_penalized_log_likelihood: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class InformationComponents:
    """Information-matrix components for one genotype lane.

    Attributes:
        covariate_information_matrix: Covariate block of the information matrix.
        cross_information_vector: Cross-information terms with the genotype.
        genotype_information: Genotype information scalar.
        information_matrix: Full information matrix.

    """

    covariate_information_matrix: jax.Array
    cross_information_vector: jax.Array
    genotype_information: jax.Array
    information_matrix: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AdjustedWeightComponents:
    """Intermediate leverage-adjusted weights for Firth updates.

    Attributes:
        leverage_vector: Per-sample leverage values.
        adjusted_weight_vector: Adjusted score contribution per sample.
        second_weight_vector: Second-order Hessian weights.

    """

    leverage_vector: jax.Array
    adjusted_weight_vector: jax.Array
    second_weight_vector: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthVariantResult:
    """Firth outputs for one genotype lane.

    Attributes:
        beta: Corrected genotype effect.
        standard_error: Standard error of the corrected effect.
        chi_squared: Likelihood-ratio chi-squared statistic.
        log10_p_value: Negative log10 p-value.
        penalized_log_likelihood: Final penalized log-likelihood.
        converged_mask: Whether the lane converged.
        valid_mask: Whether corrected statistics are valid.
        iteration_count: Number of solver iterations performed.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    penalized_log_likelihood: jax.Array
    converged_mask: jax.Array
    valid_mask: jax.Array
    iteration_count: jax.Array


def prepare_regenie2_binary_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> models.Regenie2BinaryState:
    """Prepare reusable binary step 2 state.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.

    Returns:
        Reusable binary step 2 state.

    """
    covariate_matrix_float32 = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_vector_float32 = jnp.asarray(phenotype_vector, dtype=jnp.float32)
    return models.Regenie2BinaryState(
        covariate_matrix=covariate_matrix_float32,
        phenotype_vector=phenotype_vector_float32,
        sample_count=jnp.asarray(covariate_matrix_float32.shape[0], dtype=jnp.int32),
    )


def compute_logistic_probability(linear_predictor: jax.Array) -> jax.Array:
    """Compute clipped logistic probabilities."""
    probability = jax.nn.sigmoid(linear_predictor)
    return jnp.clip(probability, MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)


def solve_from_positive_definite_matrix(
    positive_definite_matrix: jax.Array,
    right_hand_side: jax.Array,
) -> jax.Array:
    """Solve a positive-definite system from its matrix form."""
    cholesky_factor = jnp.linalg.cholesky(positive_definite_matrix)
    return regenie2_linear.solve_positive_definite_system(cholesky_factor, right_hand_side)


def build_extra_code(
    chi_squared: jax.Array,
    valid_mask: jax.Array,
    correction: types.RegenieBinaryCorrection,
) -> jax.Array:
    """Select correction labels from score-test statistics."""
    candidate_mask = chi_squared >= REGENIE_SCORE_CHISQ_THRESHOLD
    correction_code = EXTRA_CODE_SPA if correction == types.RegenieBinaryCorrection.SPA else EXTRA_CODE_FIRTH
    return jnp.where(
        valid_mask,
        jnp.where(candidate_mask, correction_code, EXTRA_CODE_SCORE),
        EXTRA_CODE_TEST_FAIL,
    ).astype(jnp.int32)


@jax.jit
def fit_null_logistic_coefficients(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    maximum_iterations: int = DEFAULT_MAXIMUM_NULL_ITERATIONS,
) -> jax.Array:
    """Fit a covariate-only logistic null model with a fixed LOCO offset."""
    covariate_count = covariate_matrix.shape[1]

    def update_coefficients(
        iteration_index: int,
        coefficient_vector: jax.Array,
    ) -> jax.Array:
        del iteration_index
        linear_predictor = covariate_matrix @ coefficient_vector + loco_offset
        fitted_probability = compute_logistic_probability(linear_predictor)
        weight_vector = jnp.maximum(fitted_probability * (1.0 - fitted_probability), MINIMUM_VARIANCE)
        score_vector = covariate_matrix.T @ (phenotype_vector - fitted_probability)
        information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
        cholesky_factor = jnp.linalg.cholesky(
            information_matrix + jnp.eye(covariate_count, dtype=jnp.float32) * MINIMUM_VARIANCE
        )
        coefficient_delta = regenie2_linear.solve_positive_definite_system(cholesky_factor, score_vector)
        return coefficient_vector + coefficient_delta

    initial_coefficients = jnp.zeros(covariate_count, dtype=jnp.float32)
    return jax.lax.fori_loop(0, maximum_iterations, update_coefficients, initial_coefficients)


@jax.jit
def prepare_regenie2_binary_chromosome_state(
    state: models.Regenie2BinaryState,
    loco_offset: jax.Array,
) -> models.Regenie2BinaryChromosomeState:
    """Prepare chromosome-specific null logistic state reused across chunks."""
    loco_offset_float32 = jnp.asarray(loco_offset, dtype=jnp.float32)
    null_logistic_coefficients = fit_null_logistic_coefficients(
        state.covariate_matrix,
        state.phenotype_vector,
        loco_offset_float32,
    )
    fitted_probability = compute_logistic_probability(
        state.covariate_matrix @ null_logistic_coefficients + loco_offset_float32
    )
    bernoulli_variance = jnp.maximum(fitted_probability * (1.0 - fitted_probability), MINIMUM_VARIANCE)
    square_root_weight = jnp.sqrt(bernoulli_variance)
    score_residual = state.phenotype_vector - fitted_probability
    standardized_residual = score_residual / square_root_weight
    weighted_covariate_matrix = square_root_weight[:, None] * state.covariate_matrix
    weighted_covariate_transpose = weighted_covariate_matrix.T
    weighted_covariate_crossproduct = weighted_covariate_transpose @ weighted_covariate_matrix
    cholesky_factor = jnp.linalg.cholesky(
        weighted_covariate_crossproduct
        + jnp.eye(weighted_covariate_crossproduct.shape[0], dtype=jnp.float32) * MINIMUM_VARIANCE
    )
    weighted_genotype_projection_matrix = jax.lax.linalg.triangular_solve(
        cholesky_factor,
        weighted_covariate_transpose,
        left_side=True,
        lower=True,
    )
    null_firth_penalized_log_likelihood = fit_covariate_only_firth_null_model(
        covariate_matrix=state.covariate_matrix,
        phenotype_vector=state.phenotype_vector,
        loco_offset=loco_offset_float32,
        initial_coefficients=null_logistic_coefficients,
    )
    return models.Regenie2BinaryChromosomeState(
        covariate_matrix=state.covariate_matrix,
        phenotype_vector=state.phenotype_vector,
        null_logistic_coefficients=null_logistic_coefficients,
        fitted_probability=fitted_probability,
        score_residual=score_residual,
        loco_offset=loco_offset_float32,
        standardized_residual=standardized_residual,
        square_root_weight=square_root_weight,
        weighted_genotype_projection_matrix=weighted_genotype_projection_matrix,
        null_firth_penalized_log_likelihood=null_firth_penalized_log_likelihood,
    )


@functools.partial(jax.jit, static_argnames=("correction",))
def compute_regenie2_binary_score_test_chunk_from_chromosome_state(
    chromosome_state: models.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction: types.RegenieBinaryCorrection = types.RegenieBinaryCorrection.FIRTH_APPROXIMATE,
) -> models.Regenie2BinaryChunkResult:
    """Compute the uncorrected score-test result for one binary chunk."""
    genotype_matrix_float32 = jnp.asarray(genotype_matrix, dtype=jnp.float32)
    weighted_genotype_matrix = chromosome_state.square_root_weight[:, None] * genotype_matrix_float32
    projection_coordinates = chromosome_state.weighted_genotype_projection_matrix @ weighted_genotype_matrix
    weighted_genotype_sum_squares = jnp.einsum("ij,ij->j", weighted_genotype_matrix, weighted_genotype_matrix)
    projection_sum_squares = jnp.einsum("ij,ij->j", projection_coordinates, projection_coordinates)
    variance = jnp.maximum(weighted_genotype_sum_squares - projection_sum_squares, 0.0)
    score = genotype_matrix_float32.T @ chromosome_state.score_residual
    positive_variance_mask = variance > MINIMUM_VARIANCE
    inverse_variance = jnp.where(positive_variance_mask, jnp.reciprocal(variance), 0.0)
    beta = jnp.where(positive_variance_mask, score * inverse_variance, jnp.nan)
    standard_error = jnp.where(positive_variance_mask, jnp.sqrt(inverse_variance), jnp.nan)
    chi_squared = jnp.where(positive_variance_mask, score * score * inverse_variance, 0.0)
    log10_p_value = regenie2_linear.chi_squared_to_log10_p_value(chi_squared)
    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    extra_code = build_extra_code(chi_squared, valid_mask, correction)
    return models.Regenie2BinaryChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        valid_mask=valid_mask,
    )


compute_regenie2_binary_score_test_chunk = typing.cast(
    "BinaryScoreTestChunkComputeFunction",
    compute_regenie2_binary_score_test_chunk_from_chromosome_state,
)


def compute_information_components(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    probability_vector: jax.Array,
) -> InformationComponents:
    """Compute full information components for one genotype lane."""
    weight_vector = jnp.maximum(probability_vector * (1.0 - probability_vector), MINIMUM_VARIANCE)
    weighted_genotype_vector = weight_vector * genotype_vector
    covariate_information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
    cross_information_vector = weighted_genotype_vector @ covariate_matrix
    genotype_information = jnp.dot(weighted_genotype_vector, genotype_vector)
    top_block = jnp.concatenate([covariate_information_matrix, cross_information_vector[:, None]], axis=1)
    bottom_block = jnp.concatenate([cross_information_vector[None, :], genotype_information[None, None]], axis=1)
    information_matrix = jnp.concatenate([top_block, bottom_block], axis=0)
    return InformationComponents(
        covariate_information_matrix=covariate_information_matrix,
        cross_information_vector=cross_information_vector,
        genotype_information=genotype_information,
        information_matrix=information_matrix,
    )


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


def compute_full_model_adjusted_weight_components(
    full_design_matrix: jax.Array,
    probability_vector: jax.Array,
    information_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> AdjustedWeightComponents:
    """Compute leverage-adjusted Firth weights for one full model."""
    variance_vector = jnp.maximum(probability_vector * (1.0 - probability_vector), MINIMUM_VARIANCE)
    projected_design_matrix = solve_from_positive_definite_matrix(information_matrix, full_design_matrix.T).T
    leverage_vector = variance_vector * jnp.einsum("ij,ij->i", projected_design_matrix, full_design_matrix)
    adjusted_weight_vector = (phenotype_vector - probability_vector) + leverage_vector * (
        BINARY_CASE_THRESHOLD - probability_vector
    )
    second_weight_vector = (1.0 + leverage_vector) * variance_vector
    return AdjustedWeightComponents(
        leverage_vector=leverage_vector,
        adjusted_weight_vector=adjusted_weight_vector,
        second_weight_vector=second_weight_vector,
    )


def compute_covariate_only_adjusted_weight_components(
    covariate_matrix: jax.Array,
    probability_vector: jax.Array,
    information_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> AdjustedWeightComponents:
    """Compute leverage-adjusted Firth weights for the covariate-only null model."""
    variance_vector = jnp.maximum(probability_vector * (1.0 - probability_vector), MINIMUM_VARIANCE)
    projected_covariate_matrix = solve_from_positive_definite_matrix(information_matrix, covariate_matrix.T).T
    leverage_vector = variance_vector * jnp.einsum("ij,ij->i", projected_covariate_matrix, covariate_matrix)
    adjusted_weight_vector = (phenotype_vector - probability_vector) + leverage_vector * (
        BINARY_CASE_THRESHOLD - probability_vector
    )
    second_weight_vector = (1.0 + leverage_vector) * variance_vector
    return AdjustedWeightComponents(
        leverage_vector=leverage_vector,
        adjusted_weight_vector=adjusted_weight_vector,
        second_weight_vector=second_weight_vector,
    )


def fit_covariate_only_firth_null_model(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
) -> jax.Array:
    """Fit the covariate-only Firth null model and return its penalized log-likelihood."""

    def condition_function(state: FirthState) -> jax.Array:
        return (state.iteration_count < FIRTH_MAXIMUM_ITERATIONS) & (~state.converged) & (~state.failed)

    def body_function(state: FirthState) -> FirthState:
        linear_predictor = covariate_matrix @ state.coefficients + loco_offset
        probability_vector = compute_logistic_probability(linear_predictor)
        weight_vector = jnp.maximum(probability_vector * (1.0 - probability_vector), MINIMUM_VARIANCE)
        information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
        information_matrix = (
            information_matrix + jnp.eye(information_matrix.shape[0], dtype=jnp.float32) * MINIMUM_VARIANCE
        )
        information_cholesky_factor = jnp.linalg.cholesky(information_matrix)
        current_penalized_log_likelihood = compute_firth_penalized_log_likelihood_from_cholesky(
            probability_vector=probability_vector,
            phenotype_vector=phenotype_vector,
            information_cholesky_factor=information_cholesky_factor,
        )
        current_failed = (~jnp.isfinite(current_penalized_log_likelihood)) | (
            ~jnp.all(jnp.isfinite(state.coefficients))
        )
        adjusted_weight_components = compute_covariate_only_adjusted_weight_components(
            covariate_matrix=covariate_matrix,
            probability_vector=probability_vector,
            information_matrix=information_matrix,
            phenotype_vector=phenotype_vector,
        )
        adjusted_score = covariate_matrix.T @ adjusted_weight_components.adjusted_weight_vector
        second_hessian = (covariate_matrix.T * adjusted_weight_components.second_weight_vector) @ covariate_matrix
        second_hessian = second_hessian + jnp.eye(second_hessian.shape[0], dtype=jnp.float32) * MINIMUM_VARIANCE
        coefficient_step = solve_from_positive_definite_matrix(second_hessian, adjusted_score)
        maximum_coefficient_step = jnp.max(jnp.abs(coefficient_step))
        step_scale = jnp.minimum(1.0, FIRTH_MAXIMUM_STEP_SIZE / jnp.maximum(maximum_coefficient_step, MINIMUM_VARIANCE))
        scaled_coefficient_step = coefficient_step * step_scale
        updated_converged = (
            (state.iteration_count > 0)
            & (jnp.max(jnp.abs(scaled_coefficient_step)) <= FIRTH_COEFFICIENT_TOLERANCE)
            & (jnp.max(jnp.abs(adjusted_score)) <= FIRTH_GRADIENT_TOLERANCE)
            & (
                (current_penalized_log_likelihood - state.previous_penalized_log_likelihood)
                < FIRTH_LIKELIHOOD_TOLERANCE
            )
            & (~current_failed)
        )
        updated_coefficients = jnp.where(
            updated_converged | current_failed,
            state.coefficients,
            state.coefficients + scaled_coefficient_step,
        )
        return FirthState(
            coefficients=updated_coefficients,
            converged=updated_converged,
            failed=current_failed,
            iteration_count=state.iteration_count + jnp.asarray(1, dtype=jnp.int32),
            previous_penalized_log_likelihood=current_penalized_log_likelihood,
        )

    initial_probability_vector = compute_logistic_probability(covariate_matrix @ initial_coefficients + loco_offset)
    initial_weight_vector = jnp.maximum(
        initial_probability_vector * (1.0 - initial_probability_vector), MINIMUM_VARIANCE
    )
    initial_information_matrix = (covariate_matrix.T * initial_weight_vector) @ covariate_matrix
    initial_information_matrix = (
        initial_information_matrix + jnp.eye(initial_information_matrix.shape[0], dtype=jnp.float32) * MINIMUM_VARIANCE
    )
    initial_information_cholesky_factor = jnp.linalg.cholesky(initial_information_matrix)
    initial_penalized_log_likelihood = compute_firth_penalized_log_likelihood_from_cholesky(
        probability_vector=initial_probability_vector,
        phenotype_vector=phenotype_vector,
        information_cholesky_factor=initial_information_cholesky_factor,
    )
    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        FirthState(
            coefficients=initial_coefficients,
            converged=jnp.asarray(0, dtype=jnp.bool_),
            failed=jnp.asarray(0, dtype=jnp.bool_),
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            previous_penalized_log_likelihood=initial_penalized_log_likelihood,
        ),
    )
    final_probability_vector = compute_logistic_probability(covariate_matrix @ final_state.coefficients + loco_offset)
    final_weight_vector = jnp.maximum(final_probability_vector * (1.0 - final_probability_vector), MINIMUM_VARIANCE)
    final_information_matrix = (covariate_matrix.T * final_weight_vector) @ covariate_matrix
    final_information_matrix = (
        final_information_matrix + jnp.eye(final_information_matrix.shape[0], dtype=jnp.float32) * MINIMUM_VARIANCE
    )
    final_information_cholesky_factor = jnp.linalg.cholesky(final_information_matrix)
    return compute_firth_penalized_log_likelihood_from_cholesky(
        probability_vector=final_probability_vector,
        phenotype_vector=phenotype_vector,
        information_cholesky_factor=final_information_cholesky_factor,
    )


def fit_single_variant_firth_logistic_regression(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth: jax.Array,
    null_penalized_log_likelihood: jax.Array,
) -> FirthVariantResult:
    """Fit one Firth logistic model for a candidate variant."""
    full_design_matrix = jnp.concatenate([covariate_matrix, genotype_vector[:, None]], axis=1)
    unit_genotype_vector = jnp.zeros((full_design_matrix.shape[1],), dtype=jnp.float32).at[-1].set(1.0)

    def compute_probability_vector(coefficients: jax.Array) -> jax.Array:
        linear_predictor = covariate_matrix @ coefficients[:-1] + genotype_vector * coefficients[-1] + loco_offset
        return compute_logistic_probability(linear_predictor)

    def condition_function(state: FirthState) -> jax.Array:
        return (state.iteration_count < FIRTH_MAXIMUM_ITERATIONS) & (~state.converged) & (~state.failed) & (~skip_firth)

    def body_function(state: FirthState) -> FirthState:
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
        current_penalized_log_likelihood = compute_firth_penalized_log_likelihood_from_cholesky(
            probability_vector=probability_vector,
            phenotype_vector=phenotype_vector,
            information_cholesky_factor=information_cholesky_factor,
        )
        current_failed = (~jnp.isfinite(current_penalized_log_likelihood)) | (
            ~jnp.all(jnp.isfinite(state.coefficients))
        )
        adjusted_weight_components = compute_full_model_adjusted_weight_components(
            full_design_matrix=full_design_matrix,
            probability_vector=probability_vector,
            information_matrix=information_matrix,
            phenotype_vector=phenotype_vector,
        )
        adjusted_score = full_design_matrix.T @ adjusted_weight_components.adjusted_weight_vector
        second_hessian = (full_design_matrix.T * adjusted_weight_components.second_weight_vector) @ full_design_matrix
        second_hessian = second_hessian + jnp.eye(second_hessian.shape[0], dtype=jnp.float32) * MINIMUM_VARIANCE
        coefficient_step = solve_from_positive_definite_matrix(second_hessian, adjusted_score)
        maximum_coefficient_step = jnp.max(jnp.abs(coefficient_step))
        step_scale = jnp.minimum(1.0, FIRTH_MAXIMUM_STEP_SIZE / jnp.maximum(maximum_coefficient_step, MINIMUM_VARIANCE))
        scaled_coefficient_step = coefficient_step * step_scale
        updated_converged = (
            (state.iteration_count > 0)
            & (jnp.max(jnp.abs(scaled_coefficient_step)) <= FIRTH_COEFFICIENT_TOLERANCE)
            & (jnp.max(jnp.abs(adjusted_score)) <= FIRTH_GRADIENT_TOLERANCE)
            & (
                (current_penalized_log_likelihood - state.previous_penalized_log_likelihood)
                < FIRTH_LIKELIHOOD_TOLERANCE
            )
            & (~current_failed)
        )
        updated_coefficients = jnp.where(
            updated_converged | current_failed,
            state.coefficients,
            state.coefficients + scaled_coefficient_step,
        )
        return FirthState(
            coefficients=updated_coefficients,
            converged=updated_converged,
            failed=current_failed,
            iteration_count=state.iteration_count + jnp.asarray(1, dtype=jnp.int32),
            previous_penalized_log_likelihood=current_penalized_log_likelihood,
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
    initial_penalized_log_likelihood = compute_firth_penalized_log_likelihood_from_cholesky(
        probability_vector=initial_probability_vector,
        phenotype_vector=phenotype_vector,
        information_cholesky_factor=initial_information_cholesky_factor,
    )
    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        FirthState(
            coefficients=initial_coefficients,
            converged=skip_firth,
            failed=jnp.asarray(0, dtype=jnp.bool_),
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            previous_penalized_log_likelihood=jnp.where(skip_firth, 0.0, initial_penalized_log_likelihood),
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
    final_penalized_log_likelihood = compute_firth_penalized_log_likelihood_from_cholesky(
        probability_vector=final_probability_vector,
        phenotype_vector=phenotype_vector,
        information_cholesky_factor=final_information_cholesky_factor,
    )
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
    genotype_variance = solve_from_positive_definite_matrix(final_second_hessian, unit_genotype_vector)[-1]
    beta = final_state.coefficients[-1]
    standard_error = jnp.sqrt(jnp.where(genotype_variance > 0.0, genotype_variance, jnp.nan))
    chi_squared = jnp.maximum(2.0 * (final_penalized_log_likelihood - null_penalized_log_likelihood), 0.0)
    log10_p_value = regenie2_linear.chi_squared_to_log10_p_value(chi_squared)
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
    return FirthVariantResult(
        beta=jnp.where(skip_firth, jnp.nan, beta),
        standard_error=jnp.where(skip_firth, jnp.nan, standard_error),
        chi_squared=jnp.where(skip_firth, jnp.nan, chi_squared),
        log10_p_value=jnp.where(skip_firth, jnp.nan, log10_p_value),
        penalized_log_likelihood=jnp.where(skip_firth, jnp.nan, final_penalized_log_likelihood),
        converged_mask=jnp.where(skip_firth, jnp.asarray(0, dtype=jnp.bool_), final_state.converged),
        valid_mask=valid_mask,
        iteration_count=jnp.where(skip_firth, jnp.asarray(0, dtype=jnp.int32), final_state.iteration_count),
    )


def compute_firth_pre_dispatch_mask_without_mask(
    genotype_matrix_by_variant: jax.Array,
    phenotype_vector: jax.Array,
) -> jax.Array:
    """Identify variants with obvious case-control allele-count separation."""
    case_mask = phenotype_vector > BINARY_CASE_THRESHOLD
    control_mask = phenotype_vector < BINARY_CASE_THRESHOLD
    case_mask_float = case_mask.astype(genotype_matrix_by_variant.dtype)
    control_mask_float = control_mask.astype(genotype_matrix_by_variant.dtype)
    case_sample_count = jnp.sum(case_mask_float)
    control_sample_count = jnp.sum(control_mask_float)
    case_allele_count = genotype_matrix_by_variant @ case_mask_float
    control_allele_count = genotype_matrix_by_variant @ control_mask_float
    case_reference_allele_count = ALLELE_COUNT_MULTIPLIER * case_sample_count - case_allele_count
    control_reference_allele_count = ALLELE_COUNT_MULTIPLIER * control_sample_count - control_allele_count
    return (
        (case_allele_count <= 0.0)
        | (control_allele_count <= 0.0)
        | (case_reference_allele_count <= 0.0)
        | (control_reference_allele_count <= 0.0)
    )


def initialize_full_model_coefficients_without_mask(
    covariate_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    phenotype_vector: jax.Array,
) -> jax.Array:
    """Initialize full-model coefficients with a pseudo-response regression."""
    pseudo_response_vector = INITIAL_RESPONSE_SCALE * (phenotype_vector - BINARY_CASE_THRESHOLD)
    covariate_information_matrix = covariate_matrix.T @ covariate_matrix
    covariate_information_matrix = jnp.broadcast_to(
        covariate_information_matrix[None, :, :],
        (genotype_matrix_by_variant.shape[0], covariate_matrix.shape[1], covariate_matrix.shape[1]),
    )
    cross_information_vector = genotype_matrix_by_variant @ covariate_matrix
    genotype_information = jnp.einsum("ij,ij->i", genotype_matrix_by_variant, genotype_matrix_by_variant)
    covariate_score = jnp.broadcast_to(
        (covariate_matrix.T @ pseudo_response_vector)[None, :],
        (genotype_matrix_by_variant.shape[0], covariate_matrix.shape[1]),
    )
    genotype_score = genotype_matrix_by_variant @ pseudo_response_vector
    stacked_right_hand_side = jnp.stack([covariate_score, cross_information_vector], axis=-1)
    covariate_and_cross_solutions = jax.vmap(solve_from_positive_definite_matrix)(
        covariate_information_matrix,
        stacked_right_hand_side,
    )
    covariate_solution = covariate_and_cross_solutions[..., 0]
    cross_solution = covariate_and_cross_solutions[..., 1]
    schur_complement = genotype_information - jnp.einsum("ij,ij->i", cross_information_vector, cross_solution)
    genotype_coefficient = (
        genotype_score - jnp.einsum("ij,ij->i", cross_information_vector, covariate_solution)
    ) / schur_complement
    covariate_coefficients = covariate_solution - cross_solution * genotype_coefficient[:, None]
    return jnp.concatenate([covariate_coefficients, genotype_coefficient[:, None]], axis=1)


def build_device_firth_batch_plan(
    fallback_mask: jax.Array,
    variant_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Build fixed-shape Firth index batches on device."""
    firth_batch_size = get_firth_batch_size()
    max_batch_count = (variant_count + firth_batch_size - 1) // firth_batch_size
    padded_variant_count = max_batch_count * firth_batch_size
    fallback_index_vector = jnp.nonzero(fallback_mask, size=variant_count, fill_value=0)[0]
    fallback_count = jnp.sum(fallback_mask, dtype=jnp.int32)
    padded_index_vector = jnp.pad(
        fallback_index_vector,
        (0, padded_variant_count - variant_count),
        constant_values=0,
    )
    active_mask_vector = jnp.arange(padded_variant_count, dtype=jnp.int32) < fallback_count
    return (
        padded_index_vector.reshape((max_batch_count, firth_batch_size)),
        active_mask_vector.reshape((max_batch_count, firth_batch_size)),
    )


def compute_firth_variantwise(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth_mask: jax.Array,
    null_penalized_log_likelihood: jax.Array,
) -> FirthVariantResult:
    """Compute device-side Firth fits for a padded set of candidate lanes."""
    return jax.vmap(
        fit_single_variant_firth_logistic_regression,
        in_axes=(None, None, 0, None, 0, 0, None),
    )(
        covariate_matrix,
        phenotype_vector,
        genotype_matrix_by_variant,
        loco_offset,
        initial_coefficients,
        skip_firth_mask,
        null_penalized_log_likelihood,
    )


def build_empty_firth_variant_result(
    batch_size: int,
) -> FirthVariantResult:
    """Build a placeholder Firth result for skipped padded batches."""
    return FirthVariantResult(
        beta=jnp.full((batch_size,), jnp.nan, dtype=jnp.float32),
        standard_error=jnp.full((batch_size,), jnp.nan, dtype=jnp.float32),
        chi_squared=jnp.full((batch_size,), jnp.nan, dtype=jnp.float32),
        log10_p_value=jnp.full((batch_size,), jnp.nan, dtype=jnp.float32),
        penalized_log_likelihood=jnp.full((batch_size,), jnp.nan, dtype=jnp.float32),
        converged_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
        valid_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
        iteration_count=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


@jax.jit
def apply_device_candidate_corrections_firth(
    chromosome_state: models.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    result: models.Regenie2BinaryChunkResult,
) -> models.Regenie2BinaryChunkResult:
    """Apply fully device-resident Firth corrections to score-test candidates."""
    candidate_mask = result.extra_code == EXTRA_CODE_FIRTH
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)

    def no_candidate_corrections() -> models.Regenie2BinaryChunkResult:
        return result

    def apply_candidate_corrections() -> models.Regenie2BinaryChunkResult:
        firth_batch_size = get_firth_batch_size()
        genotype_matrix_float32 = jnp.asarray(genotype_matrix, dtype=jnp.float32)
        variant_count = genotype_matrix_float32.shape[1]
        fallback_index_matrix, fallback_active_mask_matrix = build_device_firth_batch_plan(
            candidate_mask, variant_count
        )
        flat_fallback_indices = fallback_index_matrix.reshape((-1,))
        flat_active_mask = fallback_active_mask_matrix.reshape((-1,))
        genotype_matrix_by_variant = jnp.take(genotype_matrix_float32, flat_fallback_indices, axis=1).T
        heuristic_firth_mask = (
            compute_firth_pre_dispatch_mask_without_mask(
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                phenotype_vector=chromosome_state.phenotype_vector,
            )
            & flat_active_mask
        )
        standard_initial_coefficients = jnp.broadcast_to(
            chromosome_state.null_logistic_coefficients[None, :],
            (genotype_matrix_by_variant.shape[0], chromosome_state.null_logistic_coefficients.shape[0]),
        )
        standard_initial_coefficients = jnp.concatenate(
            [
                standard_initial_coefficients,
                jnp.take(result.beta, flat_fallback_indices, axis=0)[:, None],
            ],
            axis=1,
        )
        heuristic_initial_coefficients = initialize_full_model_coefficients_without_mask(
            covariate_matrix=chromosome_state.covariate_matrix,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            phenotype_vector=chromosome_state.phenotype_vector,
        )
        initial_coefficients = jnp.where(
            heuristic_firth_mask[:, None],
            heuristic_initial_coefficients,
            standard_initial_coefficients,
        )
        batch_count = fallback_index_matrix.shape[0]
        active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
        genotype_batches = genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
        initial_coefficient_batches = initial_coefficients.reshape((batch_count, firth_batch_size, -1))
        active_mask_batches = flat_active_mask.reshape((batch_count, firth_batch_size))
        empty_firth_variant_result = build_empty_firth_variant_result(firth_batch_size)

        def compute_firth_batch(
            carry: None,
            batch_index: jax.Array,
        ) -> tuple[None, FirthVariantResult]:
            del carry

            def run_active_batch(_: None) -> FirthVariantResult:
                return compute_firth_variantwise(
                    covariate_matrix=chromosome_state.covariate_matrix,
                    phenotype_vector=chromosome_state.phenotype_vector,
                    genotype_matrix_by_variant=genotype_batches[batch_index],
                    loco_offset=chromosome_state.loco_offset,
                    initial_coefficients=initial_coefficient_batches[batch_index],
                    skip_firth_mask=~active_mask_batches[batch_index],
                    null_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood,
                )

            batch_result = jax.lax.cond(
                batch_index < active_batch_count,
                run_active_batch,
                lambda _: empty_firth_variant_result,
                operand=None,
            )
            return None, batch_result

        _, batched_firth_result = jax.lax.scan(
            compute_firth_batch,
            None,
            jnp.arange(batch_count, dtype=jnp.int32),
        )
        firth_result = FirthVariantResult(
            beta=batched_firth_result.beta.reshape((-1,)),
            standard_error=batched_firth_result.standard_error.reshape((-1,)),
            chi_squared=batched_firth_result.chi_squared.reshape((-1,)),
            log10_p_value=batched_firth_result.log10_p_value.reshape((-1,)),
            penalized_log_likelihood=batched_firth_result.penalized_log_likelihood.reshape((-1,)),
            converged_mask=batched_firth_result.converged_mask.reshape((-1,)),
            valid_mask=batched_firth_result.valid_mask.reshape((-1,)),
            iteration_count=batched_firth_result.iteration_count.reshape((-1,)),
        )
        active_flat_positions = jnp.nonzero(flat_active_mask, size=variant_count, fill_value=0)[0]
        active_fallback_indices = flat_fallback_indices[active_flat_positions]
        current_beta = jnp.take(result.beta, active_fallback_indices, axis=0)
        current_standard_error = jnp.take(result.standard_error, active_fallback_indices, axis=0)
        current_chi_squared = jnp.take(result.chi_squared, active_fallback_indices, axis=0)
        current_log10_p_value = jnp.take(result.log10_p_value, active_fallback_indices, axis=0)
        active_valid_mask = firth_result.valid_mask[active_flat_positions]
        merged_beta = jnp.where(active_valid_mask, firth_result.beta[active_flat_positions], current_beta)
        merged_standard_error = jnp.where(
            active_valid_mask,
            firth_result.standard_error[active_flat_positions],
            current_standard_error,
        )
        merged_chi_squared = jnp.where(
            active_valid_mask,
            firth_result.chi_squared[active_flat_positions],
            current_chi_squared,
        )
        merged_log10_p_value = jnp.where(
            active_valid_mask,
            firth_result.log10_p_value[active_flat_positions],
            current_log10_p_value,
        )
        merged_extra_code = jnp.where(active_valid_mask, EXTRA_CODE_FIRTH, EXTRA_CODE_TEST_FAIL).astype(jnp.int32)
        return models.Regenie2BinaryChunkResult(
            beta=result.beta.at[active_fallback_indices].set(merged_beta),
            standard_error=result.standard_error.at[active_fallback_indices].set(merged_standard_error),
            chi_squared=result.chi_squared.at[active_fallback_indices].set(merged_chi_squared),
            log10_p_value=result.log10_p_value.at[active_fallback_indices].set(merged_log10_p_value),
            extra_code=result.extra_code.at[active_fallback_indices].set(merged_extra_code),
            valid_mask=result.valid_mask.at[active_fallback_indices].set(active_valid_mask),
        )

    return jax.lax.cond(fallback_count > 0, apply_candidate_corrections, no_candidate_corrections)


def apply_device_candidate_corrections(
    chromosome_state: models.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    result: models.Regenie2BinaryChunkResult,
    correction: types.RegenieBinaryCorrection,
) -> models.Regenie2BinaryChunkResult:
    """Apply binary candidate corrections without leaving device memory."""
    if correction == types.RegenieBinaryCorrection.SPA:
        return result
    return apply_device_candidate_corrections_firth(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=result,
    )


@functools.partial(jax.jit, static_argnames=("correction",))
def compute_regenie2_binary_chunk_from_chromosome_state(
    chromosome_state: models.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction: types.RegenieBinaryCorrection = types.RegenieBinaryCorrection.FIRTH_APPROXIMATE,
) -> models.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association using cached null state."""
    score_test_result = compute_regenie2_binary_score_test_chunk(
        chromosome_state,
        genotype_matrix,
        correction,
    )
    return apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=score_test_result,
        correction=correction,
    )


def compute_regenie2_binary_chunk(
    state: models.Regenie2BinaryState,
    genotype_matrix: jax.Array,
    loco_offset: jax.Array,
    correction: types.RegenieBinaryCorrection = types.RegenieBinaryCorrection.FIRTH_APPROXIMATE,
) -> models.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association for a genotype chunk."""
    chromosome_state = prepare_regenie2_binary_chromosome_state(state, loco_offset)
    compute_regenie2_binary_chunk_from_state = typing.cast(
        "BinaryChunkComputeFunction",
        compute_regenie2_binary_chunk_from_chromosome_state,
    )
    return compute_regenie2_binary_chunk_from_state(
        chromosome_state,
        genotype_matrix,
        correction,
    )
