"""Logistic-regression kernels for additive association testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from g import jax_setup  # noqa: F401
from g.models import LogisticAssociationChunkResult

if TYPE_CHECKING:
    import numpy.typing as npt

MINIMUM_PROBABILITY = 1.0e-12
MINIMUM_WEIGHT = 1.0e-12
INITIAL_RESPONSE_SCALE = 4.863891244002886
MISSING_PROBABILITY_FILL = 0.5
LOG_LIKELIHOOD_CONVERGENCE_OFFSET = 0.05
BINARY_CASE_THRESHOLD = 0.5
ALLELE_COUNT_MULTIPLIER = 2.0
FIRTH_GRADIENT_TOLERANCE = 1.0e-4
FIRTH_COEFFICIENT_TOLERANCE = 1.0e-4
FIRTH_LIKELIHOOD_TOLERANCE = 1.0e-4
FIRTH_MAXIMUM_STEP_SIZE = 5.0
FIRTH_TOLERANCE_FLOOR = 1.0e-12
FIRTH_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
MAX_ITERATION_COUNT = 100
LOGISTIC_METHOD_STANDARD = 0
LOGISTIC_METHOD_FIRTH = 1
LOGISTIC_ERROR_NONE = 0
LOGISTIC_ERROR_UNFINISHED = 1
LOGISTIC_ERROR_LOGISTIC_CONVERGE_FAIL = 2
LOGISTIC_ERROR_FIRTH_CONVERGE_FAIL = 3


class LogisticState(NamedTuple):
    """State container for batched association IRLS."""

    coefficients: jax.Array
    converged_mask: jax.Array
    iteration_count: jax.Array
    previous_log_likelihood: jax.Array
    last_covariate_information_matrix: jax.Array
    last_cross_information_vector: jax.Array
    last_genotype_information: jax.Array


class CovariateOnlyLogisticState(NamedTuple):
    """State container for masked covariate-only IRLS."""

    coefficients: jax.Array
    converged_mask: jax.Array
    iteration_count: jax.Array
    previous_log_likelihood: jax.Array


class StandardLogisticChunkEvaluation(NamedTuple):
    """Standard-logistic outputs plus coefficient estimates."""

    logistic_result: LogisticAssociationChunkResult
    coefficients: jax.Array


class HostStandardLogisticChunkEvaluation(NamedTuple):
    """Host-resident standard-logistic outputs plus coefficient estimates."""

    logistic_result: HostLogisticAssociationChunkResult
    coefficients: npt.NDArray[np.float64]


class HostLogisticAssociationChunkResult(NamedTuple):
    """Host-resident logistic association outputs."""

    beta: npt.NDArray[np.float64]
    standard_error: npt.NDArray[np.float64]
    test_statistic: npt.NDArray[np.float64]
    p_value: npt.NDArray[np.float64]
    method_code: npt.NDArray[np.int32]
    error_code: npt.NDArray[np.int32]
    converged_mask: npt.NDArray[np.bool_]
    valid_mask: npt.NDArray[np.bool_]
    iteration_count: npt.NDArray[np.int32]


class FirthIndexBatch(NamedTuple):
    """Padded and active fallback indices for one Firth batch."""

    padded_index_vector: jax.Array
    active_index_array: npt.NDArray[np.int64]


class FirthState(NamedTuple):
    """State container for single-variant Firth regression."""

    coefficients: jax.Array
    converged: jax.Array
    failed: jax.Array
    iteration_count: jax.Array
    previous_penalized_log_likelihood: jax.Array


class InformationComponents(NamedTuple):
    """Per-variant information-matrix components."""

    covariate_information_matrix: jax.Array
    cross_information_vector: jax.Array
    genotype_information: jax.Array
    information_matrix: jax.Array


class FirthVariantResult(NamedTuple):
    """Single-variant Firth fit outputs."""

    beta: jax.Array
    standard_error: jax.Array
    test_statistic: jax.Array
    p_value: jax.Array
    error_code: jax.Array
    converged_mask: jax.Array
    valid_mask: jax.Array
    iteration_count: jax.Array


class AdjustedWeightComponents(NamedTuple):
    """Intermediate leverage and weight vectors for Firth updates."""

    leverage_vector: jax.Array
    adjusted_weight_vector: jax.Array
    second_weight_vector: jax.Array


def compute_covariate_information_matrix(
    covariate_matrix: jax.Array,
    effective_weights: jax.Array,
) -> jax.Array:
    """Compute batched Fisher information matrices for covariates."""
    return jnp.einsum("np,mn,nq->mpq", covariate_matrix, effective_weights, covariate_matrix)


def compute_covariate_score(
    covariate_matrix: jax.Array,
    residual_matrix: jax.Array,
) -> jax.Array:
    """Compute batched covariate score vectors."""
    return jnp.einsum("np,mn->mp", covariate_matrix, residual_matrix)


def assemble_full_information_matrix(
    covariate_information_matrix: jax.Array,
    cross_information_vector: jax.Array,
    genotype_information: jax.Array,
) -> jax.Array:
    """Assemble full batched Fisher information matrices."""
    genotype_information_column = genotype_information[:, None, None]
    top_block = jnp.concatenate([covariate_information_matrix, cross_information_vector[:, :, None]], axis=2)
    bottom_block = jnp.concatenate([cross_information_vector[:, None, :], genotype_information_column], axis=2)
    return jnp.concatenate([top_block, bottom_block], axis=1)


def compute_log_likelihood(probability_matrix: jax.Array, phenotype_matrix: jax.Array) -> jax.Array:
    """Compute batched logistic log-likelihood values."""
    clipped_probability_matrix = jnp.clip(probability_matrix, MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)
    return jnp.sum(
        phenotype_matrix * jnp.log(clipped_probability_matrix)
        + (1.0 - phenotype_matrix) * jnp.log1p(-clipped_probability_matrix),
        axis=1,
    )


def compute_covariate_only_probability_matrix(
    covariate_matrix: jax.Array,
    coefficients: jax.Array,
) -> jax.Array:
    """Compute batched covariate-only logistic probabilities."""
    linear_predictor = jnp.einsum("np,mp->mn", covariate_matrix, coefficients)
    return jnp.clip(jax.nn.sigmoid(linear_predictor), MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)


def compute_probability_matrix(
    covariate_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    coefficients: jax.Array,
) -> jax.Array:
    """Compute batched logistic probabilities for covariates plus genotype."""
    covariate_coefficients = coefficients[:, :-1]
    genotype_coefficients = coefficients[:, -1]
    linear_predictor = (
        jnp.einsum("np,mp->mn", covariate_matrix, covariate_coefficients)
        + genotype_matrix_by_variant * genotype_coefficients[:, None]
    )
    return jnp.clip(jax.nn.sigmoid(linear_predictor), MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)


def initialize_full_model_coefficients(
    covariate_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    phenotype_vector: jax.Array,
    observation_mask: jax.Array,
) -> jax.Array:
    """Initialize full-model coefficients with pseudo-response regression."""
    pseudo_response_vector = INITIAL_RESPONSE_SCALE * (phenotype_vector - BINARY_CASE_THRESHOLD)
    masked_pseudo_response = jnp.where(observation_mask, pseudo_response_vector[None, :], 0.0)
    observation_mask_float = observation_mask.astype(covariate_matrix.dtype)
    covariate_information_matrix = jnp.einsum(
        "np,mn,nq->mpq",
        covariate_matrix,
        observation_mask_float,
        covariate_matrix,
    )
    cross_information_vector = jnp.einsum(
        "np,mn->mp", covariate_matrix, observation_mask_float * genotype_matrix_by_variant
    )
    genotype_information = jnp.sum(
        observation_mask_float * genotype_matrix_by_variant * genotype_matrix_by_variant, axis=1
    )
    full_information_matrix = assemble_full_information_matrix(
        covariate_information_matrix=covariate_information_matrix,
        cross_information_vector=cross_information_vector,
        genotype_information=genotype_information,
    )
    covariate_score = compute_covariate_score(covariate_matrix, masked_pseudo_response)
    genotype_score = jnp.sum(genotype_matrix_by_variant * masked_pseudo_response, axis=1)
    full_score = jnp.concatenate([covariate_score, genotype_score[:, None]], axis=1)
    return jnp.squeeze(jnp.linalg.solve(full_information_matrix, full_score[:, :, None]), axis=-1)


def compute_information_components(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    probability_vector: jax.Array,
    observation_mask: jax.Array,
) -> InformationComponents:
    """Compute information-matrix components for one variant."""
    effective_weights = jnp.where(
        observation_mask,
        jnp.clip(probability_vector * (1.0 - probability_vector), MINIMUM_WEIGHT),
        0.0,
    )
    weighted_genotype_vector = effective_weights * genotype_vector
    covariate_information_matrix = jnp.einsum("np,n,nq->pq", covariate_matrix, effective_weights, covariate_matrix)
    cross_information_vector = jnp.einsum("np,n->p", covariate_matrix, weighted_genotype_vector)
    genotype_information = jnp.sum(weighted_genotype_vector * genotype_vector)
    information_matrix = assemble_full_information_matrix(
        covariate_information_matrix=covariate_information_matrix[None, :, :],
        cross_information_vector=cross_information_vector[None, :],
        genotype_information=genotype_information[None],
    )[0]
    return InformationComponents(
        covariate_information_matrix=covariate_information_matrix,
        cross_information_vector=cross_information_vector,
        genotype_information=genotype_information,
        information_matrix=information_matrix,
    )


def compute_firth_penalized_log_likelihood(
    probability_vector: jax.Array,
    phenotype_vector: jax.Array,
    observation_mask: jax.Array,
    information_matrix: jax.Array,
) -> jax.Array:
    """Compute the Firth-penalized log-likelihood for one variant."""
    masked_probability_vector = jnp.where(observation_mask, probability_vector, MISSING_PROBABILITY_FILL)
    masked_phenotype_vector = jnp.where(observation_mask, phenotype_vector, 0.0)
    log_likelihood = jnp.sum(
        masked_phenotype_vector * jnp.log(masked_probability_vector)
        + (observation_mask.astype(probability_vector.dtype) - masked_phenotype_vector)
        * jnp.log1p(-masked_probability_vector),
    )
    slogdet_result = jnp.linalg.slogdet(information_matrix)
    sign = jnp.asarray(slogdet_result.sign)
    log_absolute_determinant = jnp.asarray(slogdet_result.logabsdet)
    penalty_term = jnp.where(sign > 0.0, BINARY_CASE_THRESHOLD * log_absolute_determinant, -jnp.inf)
    return log_likelihood + penalty_term


def compute_firth_statistics(
    coefficients: jax.Array,
    covariance_matrix: jax.Array,
    converged: jax.Array,
    failed: jax.Array,
    iteration_count: jax.Array,
) -> FirthVariantResult:
    """Build association statistics from a fitted Firth model."""
    beta = coefficients[-1]
    genotype_variance = covariance_matrix[-1, -1]
    standard_error = jnp.sqrt(jnp.where(genotype_variance > 0.0, genotype_variance, jnp.nan))
    test_statistic = beta / standard_error
    p_value = 2.0 * norm.sf(jnp.abs(test_statistic))
    valid_mask = (
        jnp.isfinite(beta)
        & jnp.isfinite(standard_error)
        & jnp.isfinite(test_statistic)
        & jnp.isfinite(p_value)
        & (standard_error > 0.0)
        & (~failed)
    )
    error_code = jnp.where(
        failed | (~valid_mask),
        jnp.asarray(LOGISTIC_ERROR_FIRTH_CONVERGE_FAIL, dtype=jnp.int32),
        jnp.where(
            converged,
            jnp.asarray(LOGISTIC_ERROR_NONE, dtype=jnp.int32),
            jnp.asarray(LOGISTIC_ERROR_UNFINISHED, dtype=jnp.int32),
        ),
    )
    return FirthVariantResult(
        beta=beta,
        standard_error=standard_error,
        test_statistic=test_statistic,
        p_value=p_value,
        error_code=error_code,
        converged_mask=converged,
        valid_mask=valid_mask,
        iteration_count=iteration_count,
    )


@jax.jit
def fit_covariate_only_logistic_regression(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> jax.Array:
    """Fit a logistic model using covariates only.

    Args:
        covariate_matrix: Covariate design matrix.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        max_iterations: Maximum IRLS iterations.
        tolerance: Relative log-likelihood convergence tolerance.

    Returns:
        Covariate-only coefficient estimates.

    """
    observation_mask = jnp.ones((1, phenotype_vector.shape[0]), dtype=bool)
    initial_coefficients = jnp.linalg.solve(
        covariate_matrix.T @ covariate_matrix,
        covariate_matrix.T @ (INITIAL_RESPONSE_SCALE * (phenotype_vector - BINARY_CASE_THRESHOLD)),
    )
    return fit_masked_covariate_only_logistic_regression(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        observation_mask=observation_mask,
        initial_coefficients=initial_coefficients,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )[0]


@jax.jit
def fit_masked_covariate_only_logistic_regression(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    observation_mask: jax.Array,
    initial_coefficients: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> jax.Array:
    """Fit covariate-only logistic models for per-variant observation masks.

    Args:
        covariate_matrix: Covariate design matrix.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        observation_mask: Per-variant sample inclusion mask.
        initial_coefficients: Shared initial covariate coefficients.
        max_iterations: Maximum IRLS iterations.
        tolerance: Relative log-likelihood convergence tolerance.

    Returns:
        Batched covariate-only coefficient estimates.

    """
    variant_count = observation_mask.shape[0]
    coefficient_count = covariate_matrix.shape[1]
    phenotype_matrix = jnp.broadcast_to(phenotype_vector[None, :], observation_mask.shape)
    iteration_limit = jnp.minimum(jnp.int32(max_iterations), jnp.int32(MAX_ITERATION_COUNT))
    broadcast_initial_coefficients = jnp.broadcast_to(initial_coefficients[None, :], (variant_count, coefficient_count))
    initial_probability_matrix = compute_covariate_only_probability_matrix(
        covariate_matrix, broadcast_initial_coefficients
    )
    masked_initial_probability_matrix = jnp.where(
        observation_mask,
        initial_probability_matrix,
        MISSING_PROBABILITY_FILL,
    )
    initial_log_likelihood = compute_log_likelihood(masked_initial_probability_matrix, phenotype_matrix)

    def condition_function(state: CovariateOnlyLogisticState) -> jax.Array:
        return (jnp.max(state.iteration_count) < iteration_limit) & jnp.any(~state.converged_mask)

    def body_function(state: CovariateOnlyLogisticState) -> CovariateOnlyLogisticState:
        probability_matrix = compute_covariate_only_probability_matrix(covariate_matrix, state.coefficients)
        masked_residual = jnp.where(observation_mask, probability_matrix - phenotype_matrix, 0.0)
        effective_weights = jnp.where(
            observation_mask,
            jnp.clip(probability_matrix * (1.0 - probability_matrix), MINIMUM_WEIGHT),
            0.0,
        )
        score = compute_covariate_score(covariate_matrix, masked_residual)
        information_matrix = compute_covariate_information_matrix(covariate_matrix, effective_weights)
        step = jnp.squeeze(jnp.linalg.solve(information_matrix, score[:, :, None]), axis=-1)
        step = jnp.where(state.converged_mask[:, None], 0.0, step)
        updated_coefficients = state.coefficients - step
        updated_probability_matrix = compute_covariate_only_probability_matrix(
            covariate_matrix,
            updated_coefficients,
        )
        masked_updated_probability_matrix = jnp.where(
            observation_mask,
            updated_probability_matrix,
            MISSING_PROBABILITY_FILL,
        )
        updated_log_likelihood = compute_log_likelihood(masked_updated_probability_matrix, phenotype_matrix)
        updated_converged_mask = state.converged_mask | (
            jnp.abs(updated_log_likelihood - state.previous_log_likelihood)
            < (tolerance * (LOG_LIKELIHOOD_CONVERGENCE_OFFSET + jnp.abs(updated_log_likelihood)))
        )
        updated_iteration_count = state.iteration_count + (~state.converged_mask).astype(jnp.int32)
        return CovariateOnlyLogisticState(
            coefficients=updated_coefficients,
            converged_mask=updated_converged_mask,
            iteration_count=updated_iteration_count,
            previous_log_likelihood=updated_log_likelihood,
        )

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        CovariateOnlyLogisticState(
            coefficients=broadcast_initial_coefficients,
            converged_mask=jnp.zeros((variant_count,), dtype=bool),
            iteration_count=jnp.zeros((variant_count,), dtype=jnp.int32),
            previous_log_likelihood=initial_log_likelihood,
        ),
    )
    return final_state.coefficients


@jax.jit
def compute_firth_pre_dispatch_mask(
    phenotype_vector: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    observation_mask: jax.Array,
) -> jax.Array:
    """Identify variants with obvious allele-count separation before logistic IRLS."""
    phenotype_matrix = jnp.broadcast_to(phenotype_vector[None, :], observation_mask.shape)
    case_observation_mask = observation_mask & (phenotype_matrix > BINARY_CASE_THRESHOLD)
    control_observation_mask = observation_mask & (phenotype_matrix < BINARY_CASE_THRESHOLD)
    case_sample_count = jnp.sum(case_observation_mask, axis=1, dtype=genotype_matrix_by_variant.dtype)
    control_sample_count = jnp.sum(control_observation_mask, axis=1, dtype=genotype_matrix_by_variant.dtype)
    case_allele_count = jnp.sum(jnp.where(case_observation_mask, genotype_matrix_by_variant, 0.0), axis=1)
    control_allele_count = jnp.sum(jnp.where(control_observation_mask, genotype_matrix_by_variant, 0.0), axis=1)
    case_reference_allele_count = ALLELE_COUNT_MULTIPLIER * case_sample_count - case_allele_count
    control_reference_allele_count = ALLELE_COUNT_MULTIPLIER * control_sample_count - control_allele_count
    return (
        (case_allele_count <= 0.0)
        | (control_allele_count <= 0.0)
        | (case_reference_allele_count <= 0.0)
        | (control_reference_allele_count <= 0.0)
    )


@jax.jit
def compute_standard_logistic_association_chunk_with_mask(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix: jax.Array,
    observation_mask: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> StandardLogisticChunkEvaluation:
    """Compute standard logistic association statistics for a chunk of variants."""
    genotype_matrix_by_variant = genotype_matrix.T
    variant_count = genotype_matrix_by_variant.shape[0]
    covariate_count = covariate_matrix.shape[1]
    phenotype_matrix = jnp.broadcast_to(phenotype_vector[None, :], observation_mask.shape)
    iteration_limit = jnp.minimum(jnp.int32(max_iterations), jnp.int32(MAX_ITERATION_COUNT))
    initial_coefficients = initialize_full_model_coefficients(
        covariate_matrix=covariate_matrix,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        phenotype_vector=phenotype_vector,
        observation_mask=observation_mask,
    )
    initial_probability_matrix = compute_probability_matrix(
        covariate_matrix, genotype_matrix_by_variant, initial_coefficients
    )
    masked_initial_probability_matrix = jnp.where(
        observation_mask,
        initial_probability_matrix,
        MISSING_PROBABILITY_FILL,
    )
    initial_log_likelihood = compute_log_likelihood(masked_initial_probability_matrix, phenotype_matrix)
    initial_covariate_information_matrix = jnp.broadcast_to(
        jnp.eye(covariate_count, dtype=covariate_matrix.dtype)[None, :, :],
        (variant_count, covariate_count, covariate_count),
    )
    initial_cross_information_vector = jnp.zeros((variant_count, covariate_count), dtype=covariate_matrix.dtype)
    initial_genotype_information = jnp.ones((variant_count,), dtype=covariate_matrix.dtype)

    def condition_function(state: LogisticState) -> jax.Array:
        return (jnp.max(state.iteration_count) < iteration_limit) & jnp.any(~state.converged_mask)

    def body_function(state: LogisticState) -> LogisticState:
        probability_matrix = compute_probability_matrix(
            covariate_matrix, genotype_matrix_by_variant, state.coefficients
        )
        masked_residual = jnp.where(observation_mask, probability_matrix - phenotype_matrix, 0.0)
        effective_weights = jnp.where(
            observation_mask,
            jnp.clip(probability_matrix * (1.0 - probability_matrix), MINIMUM_WEIGHT),
            0.0,
        )
        weighted_genotype_matrix = effective_weights * genotype_matrix_by_variant
        covariate_score = compute_covariate_score(covariate_matrix, masked_residual)
        genotype_score = jnp.sum(genotype_matrix_by_variant * masked_residual, axis=1)
        covariate_information_matrix = compute_covariate_information_matrix(covariate_matrix, effective_weights)
        cross_information_vector = jnp.einsum("np,mn->mp", covariate_matrix, weighted_genotype_matrix)
        genotype_information = jnp.sum(weighted_genotype_matrix * genotype_matrix_by_variant, axis=1)
        information_matrix = assemble_full_information_matrix(
            covariate_information_matrix=covariate_information_matrix,
            cross_information_vector=cross_information_vector,
            genotype_information=genotype_information,
        )
        score = jnp.concatenate([covariate_score, genotype_score[:, None]], axis=1)
        step = jnp.squeeze(jnp.linalg.solve(information_matrix, score[:, :, None]), axis=-1)
        step = jnp.where(state.converged_mask[:, None], 0.0, step)
        updated_coefficients = state.coefficients - step
        updated_probability_matrix = compute_probability_matrix(
            covariate_matrix, genotype_matrix_by_variant, updated_coefficients
        )
        masked_updated_probability_matrix = jnp.where(
            observation_mask,
            updated_probability_matrix,
            MISSING_PROBABILITY_FILL,
        )
        updated_log_likelihood = compute_log_likelihood(masked_updated_probability_matrix, phenotype_matrix)
        updated_converged_mask = state.converged_mask | (
            jnp.abs(updated_log_likelihood - state.previous_log_likelihood)
            < (tolerance * (LOG_LIKELIHOOD_CONVERGENCE_OFFSET + jnp.abs(updated_log_likelihood)))
        )
        updated_iteration_count = state.iteration_count + (~state.converged_mask).astype(jnp.int32)
        updated_covariate_information_matrix = jnp.where(
            state.converged_mask[:, None, None],
            state.last_covariate_information_matrix,
            covariate_information_matrix,
        )
        updated_cross_information_vector = jnp.where(
            state.converged_mask[:, None],
            state.last_cross_information_vector,
            cross_information_vector,
        )
        updated_genotype_information = jnp.where(
            state.converged_mask,
            state.last_genotype_information,
            genotype_information,
        )
        return LogisticState(
            coefficients=updated_coefficients,
            converged_mask=updated_converged_mask,
            iteration_count=updated_iteration_count,
            previous_log_likelihood=updated_log_likelihood,
            last_covariate_information_matrix=updated_covariate_information_matrix,
            last_cross_information_vector=updated_cross_information_vector,
            last_genotype_information=updated_genotype_information,
        )

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        LogisticState(
            coefficients=initial_coefficients,
            converged_mask=jnp.zeros((variant_count,), dtype=bool),
            iteration_count=jnp.zeros((variant_count,), dtype=jnp.int32),
            previous_log_likelihood=initial_log_likelihood,
            last_covariate_information_matrix=initial_covariate_information_matrix,
            last_cross_information_vector=initial_cross_information_vector,
            last_genotype_information=initial_genotype_information,
        ),
    )

    cross_information_solution = jnp.linalg.solve(
        final_state.last_covariate_information_matrix,
        final_state.last_cross_information_vector[:, :, None],
    )
    cross_information_solution = jnp.squeeze(cross_information_solution, axis=-1)
    schur_complement = final_state.last_genotype_information - jnp.sum(
        final_state.last_cross_information_vector * cross_information_solution,
        axis=1,
    )
    safe_schur_complement = jnp.where(schur_complement > 0.0, schur_complement, jnp.nan)
    beta = final_state.coefficients[:, -1]
    standard_error = jnp.sqrt(1.0 / safe_schur_complement)
    test_statistic = beta / standard_error
    p_value = 2.0 * norm.sf(jnp.abs(test_statistic))
    valid_mask = (
        jnp.isfinite(beta)
        & jnp.isfinite(standard_error)
        & jnp.isfinite(test_statistic)
        & jnp.isfinite(p_value)
        & (standard_error > 0.0)
    )
    error_code = jnp.where(
        valid_mask & final_state.converged_mask,
        jnp.full((variant_count,), LOGISTIC_ERROR_NONE, dtype=jnp.int32),
        jnp.where(
            valid_mask,
            jnp.full((variant_count,), LOGISTIC_ERROR_UNFINISHED, dtype=jnp.int32),
            jnp.full((variant_count,), LOGISTIC_ERROR_LOGISTIC_CONVERGE_FAIL, dtype=jnp.int32),
        ),
    )
    return StandardLogisticChunkEvaluation(
        logistic_result=LogisticAssociationChunkResult(
            beta=beta,
            standard_error=standard_error,
            test_statistic=test_statistic,
            p_value=p_value,
            method_code=jnp.full((variant_count,), LOGISTIC_METHOD_STANDARD, dtype=jnp.int32),
            error_code=error_code,
            converged_mask=final_state.converged_mask,
            valid_mask=valid_mask,
            iteration_count=final_state.iteration_count,
        ),
        coefficients=final_state.coefficients,
    )


def fit_single_variant_firth_logistic_regression(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    observation_mask: jax.Array,
    initial_coefficients: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> FirthVariantResult:
    """Fit Firth logistic regression for one variant."""
    iteration_limit = jnp.minimum(jnp.int32(max_iterations), jnp.int32(MAX_ITERATION_COUNT))
    gradient_tolerance = jnp.asarray(FIRTH_GRADIENT_TOLERANCE, dtype=covariate_matrix.dtype)
    coefficient_tolerance = jnp.asarray(FIRTH_COEFFICIENT_TOLERANCE, dtype=covariate_matrix.dtype)
    likelihood_tolerance = jnp.asarray(FIRTH_LIKELIHOOD_TOLERANCE, dtype=covariate_matrix.dtype)
    maximum_step_size = jnp.asarray(FIRTH_MAXIMUM_STEP_SIZE, dtype=covariate_matrix.dtype)

    def compute_probability_vector(coefficients: jax.Array) -> jax.Array:
        covariate_coefficients = coefficients[:-1]
        genotype_coefficient = coefficients[-1]
        linear_predictor = covariate_matrix @ covariate_coefficients + genotype_vector * genotype_coefficient
        return jnp.clip(jax.nn.sigmoid(linear_predictor), MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)

    full_design_matrix = jnp.concatenate([covariate_matrix, genotype_vector[:, None]], axis=1)

    def compute_covariance_matrix_from_weights(weight_vector: jax.Array) -> jax.Array:
        weighted_design_matrix = full_design_matrix * weight_vector[:, None]
        hessian_matrix = full_design_matrix.T @ weighted_design_matrix
        return jnp.linalg.inv(hessian_matrix)

    def compute_hdiag_and_adjusted_weights(
        covariance_matrix: jax.Array,
        probability_vector: jax.Array,
    ) -> AdjustedWeightComponents:
        variance_vector = jnp.where(
            observation_mask,
            jnp.clip(probability_vector * (1.0 - probability_vector), MINIMUM_WEIGHT),
            0.0,
        )
        projected_design_matrix = full_design_matrix @ covariance_matrix
        leverage_vector = variance_vector * jnp.sum(projected_design_matrix * full_design_matrix, axis=1)
        adjusted_weight_vector = jnp.where(
            observation_mask,
            (phenotype_vector - probability_vector) + leverage_vector * (BINARY_CASE_THRESHOLD - probability_vector),
            0.0,
        )
        second_weight_vector = jnp.where(observation_mask, (1.0 + leverage_vector) * variance_vector, 0.0)
        return AdjustedWeightComponents(
            leverage_vector=leverage_vector,
            adjusted_weight_vector=adjusted_weight_vector,
            second_weight_vector=second_weight_vector,
        )

    initial_probability_vector = compute_probability_vector(initial_coefficients)
    initial_information_components = compute_information_components(
        covariate_matrix=covariate_matrix,
        genotype_vector=genotype_vector,
        probability_vector=initial_probability_vector,
        observation_mask=observation_mask,
    )
    initial_penalized_log_likelihood = compute_firth_penalized_log_likelihood(
        probability_vector=initial_probability_vector,
        phenotype_vector=phenotype_vector,
        observation_mask=observation_mask,
        information_matrix=initial_information_components.information_matrix,
    )

    def condition_function(state: FirthState) -> jax.Array:
        return (state.iteration_count < iteration_limit) & (~state.converged) & (~state.failed)

    def body_function(state: FirthState) -> FirthState:
        probability_vector = compute_probability_vector(state.coefficients)
        information_components = compute_information_components(
            covariate_matrix=covariate_matrix,
            genotype_vector=genotype_vector,
            probability_vector=probability_vector,
            observation_mask=observation_mask,
        )
        covariance_matrix = jnp.linalg.inv(information_components.information_matrix)
        adjusted_weight_components = compute_hdiag_and_adjusted_weights(
            covariance_matrix=covariance_matrix,
            probability_vector=probability_vector,
        )
        adjusted_score = full_design_matrix.T @ adjusted_weight_components.adjusted_weight_vector
        adjusted_score_maximum = jnp.max(jnp.abs(adjusted_score))
        second_covariance_matrix = compute_covariance_matrix_from_weights(
            adjusted_weight_components.second_weight_vector
        )
        coefficient_step = second_covariance_matrix @ adjusted_score
        maximum_coefficient_step = jnp.max(jnp.abs(coefficient_step))
        step_scale = jnp.minimum(1.0, maximum_step_size / jnp.maximum(maximum_coefficient_step, FIRTH_TOLERANCE_FLOOR))
        scaled_coefficient_step = coefficient_step * step_scale
        updated_coefficients = state.coefficients + scaled_coefficient_step
        updated_probability_vector = compute_probability_vector(updated_coefficients)
        updated_information_components = compute_information_components(
            covariate_matrix=covariate_matrix,
            genotype_vector=genotype_vector,
            probability_vector=updated_probability_vector,
            observation_mask=observation_mask,
        )
        updated_penalized_log_likelihood = compute_firth_penalized_log_likelihood(
            probability_vector=updated_probability_vector,
            phenotype_vector=phenotype_vector,
            observation_mask=observation_mask,
            information_matrix=updated_information_components.information_matrix,
        )
        updated_failed = (~jnp.isfinite(updated_penalized_log_likelihood)) | (
            ~jnp.all(jnp.isfinite(updated_coefficients))
        )
        updated_converged = (
            (state.iteration_count > 0)
            & (jnp.max(jnp.abs(scaled_coefficient_step)) <= coefficient_tolerance)
            & (adjusted_score_maximum < gradient_tolerance)
            & ((updated_penalized_log_likelihood - state.previous_penalized_log_likelihood) < likelihood_tolerance)
            & (~updated_failed)
        )
        return FirthState(
            coefficients=updated_coefficients,
            converged=updated_converged,
            failed=updated_failed,
            iteration_count=state.iteration_count + jnp.asarray(1, dtype=jnp.int32),
            previous_penalized_log_likelihood=updated_penalized_log_likelihood,
        )

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        FirthState(
            coefficients=initial_coefficients,
            converged=jnp.zeros((), dtype=bool),
            failed=~jnp.isfinite(initial_penalized_log_likelihood),
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            previous_penalized_log_likelihood=initial_penalized_log_likelihood,
        ),
    )
    final_probability_vector = compute_probability_vector(final_state.coefficients)
    final_information_components = compute_information_components(
        covariate_matrix=covariate_matrix,
        genotype_vector=genotype_vector,
        probability_vector=final_probability_vector,
        observation_mask=observation_mask,
    )
    final_covariance_matrix = jnp.linalg.inv(final_information_components.information_matrix)
    final_adjusted_weight_components = compute_hdiag_and_adjusted_weights(
        covariance_matrix=final_covariance_matrix,
        probability_vector=final_probability_vector,
    )
    final_second_covariance_matrix = compute_covariance_matrix_from_weights(
        final_adjusted_weight_components.second_weight_vector
    )
    return compute_firth_statistics(
        coefficients=final_state.coefficients,
        covariance_matrix=final_second_covariance_matrix,
        converged=final_state.converged,
        failed=final_state.failed,
        iteration_count=final_state.iteration_count,
    )


compute_firth_association_chunk_variantwise = jax.jit(
    jax.vmap(
        fit_single_variant_firth_logistic_regression,
        in_axes=(None, None, 1, 0, 0, None, None),
    )
)


def choose_firth_batch_size(active_variant_count: int) -> int:
    """Choose the smallest configured Firth batch bucket for a count."""
    for batch_size in FIRTH_BATCH_BUCKETS:
        if active_variant_count <= batch_size:
            return batch_size
    return FIRTH_BATCH_BUCKETS[-1]


def build_firth_padded_index_batches(
    fallback_index_vector: npt.NDArray[np.int64],
) -> list[FirthIndexBatch]:
    """Build padded fallback index batches using size buckets."""
    if fallback_index_vector.size == 0:
        return []
    padded_batches: list[FirthIndexBatch] = []
    batch_start = 0
    while batch_start < fallback_index_vector.size:
        remaining_variant_count = fallback_index_vector.size - batch_start
        batch_size = choose_firth_batch_size(min(remaining_variant_count, FIRTH_BATCH_BUCKETS[-1]))
        active_index_array = fallback_index_vector[batch_start : batch_start + batch_size]
        active_variant_count = active_index_array.size
        pad_index_value = int(active_index_array[0])
        padded_index_array = np.full((batch_size,), pad_index_value, dtype=np.int32)
        padded_index_array[:active_variant_count] = active_index_array.astype(np.int32, copy=False)
        padded_batches.append(
            FirthIndexBatch(
                padded_index_vector=jnp.asarray(padded_index_array),
                active_index_array=active_index_array,
            )
        )
        batch_start += batch_size
    return padded_batches


def transfer_standard_logistic_evaluation_to_host(
    standard_evaluation: StandardLogisticChunkEvaluation,
) -> HostStandardLogisticChunkEvaluation:
    """Copy a standard-logistic evaluation to host arrays once."""
    host_standard_evaluation = jax.device_get(standard_evaluation)
    return HostStandardLogisticChunkEvaluation(
        logistic_result=HostLogisticAssociationChunkResult(
            beta=host_standard_evaluation.logistic_result.beta,
            standard_error=host_standard_evaluation.logistic_result.standard_error,
            test_statistic=host_standard_evaluation.logistic_result.test_statistic,
            p_value=host_standard_evaluation.logistic_result.p_value,
            method_code=host_standard_evaluation.logistic_result.method_code,
            error_code=host_standard_evaluation.logistic_result.error_code,
            converged_mask=host_standard_evaluation.logistic_result.converged_mask,
            valid_mask=host_standard_evaluation.logistic_result.valid_mask,
            iteration_count=host_standard_evaluation.logistic_result.iteration_count,
        ),
        coefficients=host_standard_evaluation.coefficients,
    )


def compute_firth_association_chunk_with_mask(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix: jax.Array,
    observation_mask: jax.Array,
    initial_coefficients: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> LogisticAssociationChunkResult:
    """Compute Firth association statistics for a chunk of variants."""
    firth_result = compute_firth_association_chunk_variantwise(
        covariate_matrix,
        phenotype_vector,
        genotype_matrix,
        observation_mask,
        initial_coefficients,
        max_iterations,
        tolerance,
    )
    variant_count = genotype_matrix.shape[1]
    return LogisticAssociationChunkResult(
        beta=firth_result.beta,
        standard_error=firth_result.standard_error,
        test_statistic=firth_result.test_statistic,
        p_value=firth_result.p_value,
        method_code=jnp.full((variant_count,), LOGISTIC_METHOD_FIRTH, dtype=jnp.int32),
        error_code=firth_result.error_code,
        converged_mask=firth_result.converged_mask,
        valid_mask=firth_result.valid_mask,
        iteration_count=firth_result.iteration_count,
    )


def compute_logistic_association_chunk(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> LogisticAssociationChunkResult:
    """Compute batched logistic association statistics for a chunk of variants.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        genotype_matrix: Genotype matrix.
        max_iterations: Maximum IRLS iterations.
        tolerance: Relative log-likelihood convergence tolerance.

    Returns:
        Chunk-level logistic association statistics.

    """
    observation_mask = jnp.ones(genotype_matrix.T.shape, dtype=bool)
    return compute_logistic_association_chunk_with_mask(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_matrix=genotype_matrix,
        observation_mask=observation_mask,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def compute_logistic_association_chunk_with_mask(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix: jax.Array,
    observation_mask: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> LogisticAssociationChunkResult:
    """Compute batched logistic association statistics with per-variant masks.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        genotype_matrix: Mean-imputed genotype matrix.
        observation_mask: Per-variant sample inclusion mask.
        max_iterations: Maximum IRLS iterations.
        tolerance: Relative log-likelihood convergence tolerance.

    Returns:
        Chunk-level logistic association statistics.

    """
    genotype_matrix_by_variant = genotype_matrix.T
    heuristic_firth_mask = jax.device_get(
        compute_firth_pre_dispatch_mask(
            phenotype_vector=phenotype_vector,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            observation_mask=observation_mask,
        )
    )
    standard_evaluation = compute_standard_logistic_association_chunk_with_mask(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_matrix=genotype_matrix,
        observation_mask=observation_mask,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    standard_logistic_result = standard_evaluation.logistic_result
    standard_converged_mask = jax.device_get(standard_logistic_result.converged_mask)
    standard_valid_mask = jax.device_get(standard_logistic_result.valid_mask)
    fallback_mask = heuristic_firth_mask | ((~standard_converged_mask) | (~standard_valid_mask))
    if not fallback_mask.any():
        return standard_logistic_result

    host_standard_evaluation = transfer_standard_logistic_evaluation_to_host(standard_evaluation)
    host_standard_logistic_result = host_standard_evaluation.logistic_result
    beta_values = host_standard_logistic_result.beta.copy()
    standard_error_values = host_standard_logistic_result.standard_error.copy()
    test_statistic_values = host_standard_logistic_result.test_statistic.copy()
    p_value_values = host_standard_logistic_result.p_value.copy()
    method_code_values = host_standard_logistic_result.method_code.copy()
    error_code_values = host_standard_logistic_result.error_code.copy()
    converged_mask_values = host_standard_logistic_result.converged_mask.copy()
    valid_mask_values = host_standard_logistic_result.valid_mask.copy()
    iteration_count_values = host_standard_logistic_result.iteration_count.copy()

    firth_tolerance = max(tolerance, FIRTH_TOLERANCE_FLOOR)
    fallback_index_batches = build_firth_padded_index_batches(fallback_mask.nonzero()[0])
    for firth_index_batch in fallback_index_batches:
        active_variant_count = firth_index_batch.active_index_array.size
        batch_genotype_matrix = jnp.take(genotype_matrix, firth_index_batch.padded_index_vector, axis=1)
        batch_observation_mask = jnp.take(observation_mask, firth_index_batch.padded_index_vector, axis=0)
        batch_initial_coefficients = initialize_full_model_coefficients(
            covariate_matrix=covariate_matrix,
            genotype_matrix_by_variant=batch_genotype_matrix.T,
            phenotype_vector=phenotype_vector,
            observation_mask=batch_observation_mask,
        )
        batch_fallback_mask = fallback_mask[firth_index_batch.active_index_array]
        batch_heuristic_firth_mask = heuristic_firth_mask[firth_index_batch.active_index_array]
        batch_standard_fallback_mask = batch_fallback_mask & (~batch_heuristic_firth_mask)
        if batch_standard_fallback_mask.any():
            batch_initial_coefficients_host = jax.device_get(batch_initial_coefficients)
            standard_fallback_positions = np.nonzero(batch_standard_fallback_mask)[0]
            standard_fallback_indices = firth_index_batch.active_index_array[standard_fallback_positions]
            batch_initial_coefficients_host[standard_fallback_positions, :] = host_standard_evaluation.coefficients[
                standard_fallback_indices,
                :,
            ]
            batch_initial_coefficients = jnp.asarray(batch_initial_coefficients_host)
        firth_host_result = jax.device_get(
            compute_firth_association_chunk_with_mask(
                covariate_matrix=covariate_matrix,
                phenotype_vector=phenotype_vector,
                genotype_matrix=batch_genotype_matrix,
                observation_mask=batch_observation_mask,
                initial_coefficients=batch_initial_coefficients,
                max_iterations=max_iterations,
                tolerance=firth_tolerance,
            )
        )
        batch_fallback_indices = firth_index_batch.active_index_array
        beta_values[batch_fallback_indices] = firth_host_result.beta[:active_variant_count]
        standard_error_values[batch_fallback_indices] = firth_host_result.standard_error[:active_variant_count]
        test_statistic_values[batch_fallback_indices] = firth_host_result.test_statistic[:active_variant_count]
        p_value_values[batch_fallback_indices] = firth_host_result.p_value[:active_variant_count]
        method_code_values[batch_fallback_indices] = firth_host_result.method_code[:active_variant_count]
        error_code_values[batch_fallback_indices] = firth_host_result.error_code[:active_variant_count]
        converged_mask_values[batch_fallback_indices] = firth_host_result.converged_mask[:active_variant_count]
        valid_mask_values[batch_fallback_indices] = firth_host_result.valid_mask[:active_variant_count]
        iteration_count_values[batch_fallback_indices] = firth_host_result.iteration_count[:active_variant_count]

    return LogisticAssociationChunkResult(
        beta=jnp.asarray(beta_values),
        standard_error=jnp.asarray(standard_error_values),
        test_statistic=jnp.asarray(test_statistic_values),
        p_value=jnp.asarray(p_value_values),
        method_code=jnp.asarray(method_code_values),
        error_code=jnp.asarray(error_code_values),
        converged_mask=jnp.asarray(converged_mask_values),
        valid_mask=jnp.asarray(valid_mask_values),
        iteration_count=jnp.asarray(iteration_count_values),
    )
