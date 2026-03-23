"""Logistic-regression kernels for additive association testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.profiler
from jax.scipy.stats import norm

from g import jax_setup  # noqa: F401
from g.models import LogisticAssociationChunkResult

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
FIRTH_BATCH_SIZE = 64
MAX_ITERATION_COUNT = 100
LOGISTIC_METHOD_STANDARD = 0
LOGISTIC_METHOD_FIRTH = 1
LOGISTIC_ERROR_NONE = 0
LOGISTIC_ERROR_UNFINISHED = 1
LOGISTIC_ERROR_LOGISTIC_CONVERGE_FAIL = 2
LOGISTIC_ERROR_FIRTH_CONVERGE_FAIL = 3


@jax.tree_util.register_dataclass
@dataclass
class LogisticState:
    """State container for batched association IRLS.

    Attributes:
        coefficients: Current coefficient estimates (variants x parameters).
        converged_mask: Boolean convergence mask per variant.
        iteration_count: Iteration counter per variant.
        previous_log_likelihood: Previous log-likelihood per variant.
        last_covariate_information_matrix: Last covariate information matrix.
        last_cross_information_vector: Last cross-information vector.
        last_genotype_information: Last genotype information scalar per variant.

    """

    coefficients: jax.Array
    converged_mask: jax.Array
    iteration_count: jax.Array
    previous_log_likelihood: jax.Array
    last_covariate_information_matrix: jax.Array
    last_cross_information_vector: jax.Array
    last_genotype_information: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class CovariateOnlyLogisticState:
    """State container for masked covariate-only IRLS.

    Attributes:
        coefficients: Current coefficient estimates.
        converged_mask: Boolean convergence mask.
        iteration_count: Iteration counter.
        previous_log_likelihood: Previous log-likelihood.

    """

    coefficients: jax.Array
    converged_mask: jax.Array
    iteration_count: jax.Array
    previous_log_likelihood: jax.Array


class StandardLogisticChunkEvaluation(NamedTuple):
    """Standard-logistic outputs plus coefficient estimates."""

    logistic_result: LogisticAssociationChunkResult
    coefficients: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class NoMissingLogisticConstants:
    """Chunk-invariant constants for no-missing logistic paths.

    Attributes:
        covariate_information_matrix: Precomputed covariate information matrix.
        covariate_pseudo_response_score: Covariate pseudo-response score vector.
        pseudo_response_vector: Pseudo-response vector for initialization.
        case_mask: Boolean mask for case samples.
        control_mask: Boolean mask for control samples.
        case_sample_count: Number of case samples.
        control_sample_count: Number of control samples.

    """

    covariate_information_matrix: jax.Array
    covariate_pseudo_response_score: jax.Array
    pseudo_response_vector: jax.Array
    case_mask: jax.Array
    control_mask: jax.Array
    case_sample_count: jax.Array
    control_sample_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class FirthState:
    """State container for single-variant Firth regression.

    Attributes:
        coefficients: Current coefficient estimates.
        converged: Boolean convergence flag.
        failed: Boolean failure flag.
        iteration_count: Iteration counter.
        previous_penalized_log_likelihood: Previous penalized log-likelihood.

    """

    coefficients: jax.Array
    converged: jax.Array
    failed: jax.Array
    iteration_count: jax.Array
    previous_penalized_log_likelihood: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class InformationComponents:
    """Per-variant information-matrix components.

    Attributes:
        covariate_information_matrix: Covariate block of information matrix.
        cross_information_vector: Cross terms between covariates and genotype.
        genotype_information: Genotype variance component.
        information_matrix: Full information matrix.

    """

    covariate_information_matrix: jax.Array
    cross_information_vector: jax.Array
    genotype_information: jax.Array
    information_matrix: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class FirthVariantResult:
    """Single-variant Firth fit outputs.

    Attributes:
        beta: Estimated effect size.
        standard_error: Standard error of estimate.
        test_statistic: Z-statistic (Wald test).
        p_value: Two-tailed p-value.
        error_code: Error status code.
        converged_mask: Boolean convergence flag.
        valid_mask: Boolean valid flag.
        iteration_count: IRLS iterations performed.

    """

    beta: jax.Array
    standard_error: jax.Array
    test_statistic: jax.Array
    p_value: jax.Array
    error_code: jax.Array
    converged_mask: jax.Array
    valid_mask: jax.Array
    iteration_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class AdjustedWeightComponents:
    """Intermediate leverage and weight vectors for Firth updates.

    Attributes:
        leverage_vector: Diagonal leverage values.
        adjusted_weight_vector: Adjusted response weights.
        second_weight_vector: Second-order weight adjustment.

    """

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


def initialize_full_model_coefficients_without_mask(
    covariate_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    no_missing_constants: NoMissingLogisticConstants,
) -> jax.Array:
    """Initialize full-model coefficients for chunks with no missing genotypes."""
    covariate_information_matrix = jnp.broadcast_to(
        no_missing_constants.covariate_information_matrix[None, :, :],
        (genotype_matrix_by_variant.shape[0], covariate_matrix.shape[1], covariate_matrix.shape[1]),
    )
    cross_information_vector = jnp.einsum("np,mn->mp", covariate_matrix, genotype_matrix_by_variant)
    genotype_information = jnp.sum(genotype_matrix_by_variant * genotype_matrix_by_variant, axis=1)
    full_information_matrix = assemble_full_information_matrix(
        covariate_information_matrix=covariate_information_matrix,
        cross_information_vector=cross_information_vector,
        genotype_information=genotype_information,
    )
    covariate_score = jnp.broadcast_to(
        no_missing_constants.covariate_pseudo_response_score[None, :],
        (genotype_matrix_by_variant.shape[0], covariate_matrix.shape[1]),
    )
    genotype_score = genotype_matrix_by_variant @ no_missing_constants.pseudo_response_vector
    full_score = jnp.concatenate([covariate_score, genotype_score[:, None]], axis=1)
    return jnp.squeeze(jnp.linalg.solve(full_information_matrix, full_score[:, :, None]), axis=-1)


def prepare_no_missing_logistic_constants(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> NoMissingLogisticConstants:
    """Prepare chunk-invariant constants for no-missing logistic computation."""
    pseudo_response_vector = INITIAL_RESPONSE_SCALE * (phenotype_vector - BINARY_CASE_THRESHOLD)
    case_mask = phenotype_vector > BINARY_CASE_THRESHOLD
    control_mask = phenotype_vector < BINARY_CASE_THRESHOLD
    return NoMissingLogisticConstants(
        covariate_information_matrix=covariate_matrix.T @ covariate_matrix,
        covariate_pseudo_response_score=covariate_matrix.T @ pseudo_response_vector,
        pseudo_response_vector=pseudo_response_vector,
        case_mask=case_mask,
        control_mask=control_mask,
        case_sample_count=jnp.sum(case_mask, dtype=covariate_matrix.dtype),
        control_sample_count=jnp.sum(control_mask, dtype=covariate_matrix.dtype),
    )


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
    top_block = jnp.concatenate([covariate_information_matrix, cross_information_vector[:, None]], axis=1)
    bottom_block = jnp.concatenate([cross_information_vector[None, :], genotype_information[None, None]], axis=1)
    information_matrix = jnp.concatenate([top_block, bottom_block], axis=0)
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
    cholesky_factor = jnp.linalg.cholesky(information_matrix)
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky_factor)))
    cholesky_valid = jnp.all(jnp.isfinite(cholesky_factor))
    penalty_term = jnp.where(cholesky_valid, BINARY_CASE_THRESHOLD * log_determinant, -jnp.inf)
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
        masked_probability_matrix = jnp.where(observation_mask, probability_matrix, MISSING_PROBABILITY_FILL)
        current_log_likelihood = compute_log_likelihood(masked_probability_matrix, phenotype_matrix)
        updated_converged_mask = state.converged_mask | (
            jnp.abs(current_log_likelihood - state.previous_log_likelihood)
            < (tolerance * (LOG_LIKELIHOOD_CONVERGENCE_OFFSET + jnp.abs(current_log_likelihood)))
        )
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
        updated_iteration_count = state.iteration_count + (~state.converged_mask).astype(jnp.int32)
        return CovariateOnlyLogisticState(
            coefficients=updated_coefficients,
            converged_mask=updated_converged_mask,
            iteration_count=updated_iteration_count,
            previous_log_likelihood=current_log_likelihood,
        )

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        CovariateOnlyLogisticState(
            coefficients=broadcast_initial_coefficients,
            converged_mask=jnp.zeros((variant_count,), dtype=bool),
            iteration_count=jnp.zeros((variant_count,), dtype=jnp.int32),
            previous_log_likelihood=jnp.full_like(initial_log_likelihood, -jnp.inf),
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
def compute_firth_pre_dispatch_mask_without_mask(
    genotype_matrix_by_variant: jax.Array,
    no_missing_constants: NoMissingLogisticConstants,
) -> jax.Array:
    """Identify obvious allele-count separation for chunks with no missing genotypes."""
    case_allele_count = jnp.sum(
        jnp.where(no_missing_constants.case_mask[None, :], genotype_matrix_by_variant, 0.0), axis=1
    )
    control_allele_count = jnp.sum(
        jnp.where(no_missing_constants.control_mask[None, :], genotype_matrix_by_variant, 0.0), axis=1
    )
    case_reference_allele_count = ALLELE_COUNT_MULTIPLIER * no_missing_constants.case_sample_count - case_allele_count
    control_reference_allele_count = (
        ALLELE_COUNT_MULTIPLIER * no_missing_constants.control_sample_count - control_allele_count
    )
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
        masked_probability_matrix = jnp.where(observation_mask, probability_matrix, MISSING_PROBABILITY_FILL)
        current_log_likelihood = compute_log_likelihood(masked_probability_matrix, phenotype_matrix)
        updated_converged_mask = state.converged_mask | (
            jnp.abs(current_log_likelihood - state.previous_log_likelihood)
            < (tolerance * (LOG_LIKELIHOOD_CONVERGENCE_OFFSET + jnp.abs(current_log_likelihood)))
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
        covariate_information_solution = jnp.linalg.solve(covariate_information_matrix, covariate_score[:, :, None])
        covariate_information_solution = jnp.squeeze(covariate_information_solution, axis=-1)
        cross_solution = jnp.linalg.solve(covariate_information_matrix, cross_information_vector[:, :, None])
        cross_solution = jnp.squeeze(cross_solution, axis=-1)
        schur_complement = genotype_information - jnp.sum(cross_information_vector * cross_solution, axis=1)
        safe_schur_complement = jnp.where(schur_complement > 0.0, schur_complement, 1.0)
        adjusted_genotype_score = genotype_score - jnp.sum(
            cross_information_vector * covariate_information_solution, axis=1
        )
        genotype_step = adjusted_genotype_score / safe_schur_complement
        covariate_step = covariate_information_solution - cross_solution * genotype_step[:, None]
        step = jnp.concatenate([covariate_step, genotype_step[:, None]], axis=1)
        step = jnp.where(state.converged_mask[:, None], 0.0, step)
        updated_coefficients = state.coefficients - step
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
            previous_log_likelihood=current_log_likelihood,
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
            previous_log_likelihood=jnp.full_like(initial_log_likelihood, -jnp.inf),
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


@jax.jit
def compute_standard_logistic_association_chunk_without_mask(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix: jax.Array,
    no_missing_constants: NoMissingLogisticConstants,
    max_iterations: int,
    tolerance: float,
) -> StandardLogisticChunkEvaluation:
    """Compute standard logistic statistics for chunks with no missing genotypes."""
    genotype_matrix_by_variant = genotype_matrix.T
    variant_count = genotype_matrix_by_variant.shape[0]
    covariate_count = covariate_matrix.shape[1]
    phenotype_matrix = jnp.broadcast_to(phenotype_vector[None, :], genotype_matrix_by_variant.shape)
    iteration_limit = jnp.minimum(jnp.int32(max_iterations), jnp.int32(MAX_ITERATION_COUNT))
    initial_coefficients = initialize_full_model_coefficients_without_mask(
        covariate_matrix=covariate_matrix,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        no_missing_constants=no_missing_constants,
    )
    initial_probability_matrix = compute_probability_matrix(
        covariate_matrix, genotype_matrix_by_variant, initial_coefficients
    )
    initial_log_likelihood = compute_log_likelihood(initial_probability_matrix, phenotype_matrix)
    initial_covariate_information_matrix = jnp.broadcast_to(
        no_missing_constants.covariate_information_matrix[None, :, :],
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
        current_log_likelihood = compute_log_likelihood(probability_matrix, phenotype_matrix)
        updated_converged_mask = state.converged_mask | (
            jnp.abs(current_log_likelihood - state.previous_log_likelihood)
            < (tolerance * (LOG_LIKELIHOOD_CONVERGENCE_OFFSET + jnp.abs(current_log_likelihood)))
        )
        residual_matrix = probability_matrix - phenotype_matrix
        effective_weights = jnp.clip(probability_matrix * (1.0 - probability_matrix), MINIMUM_WEIGHT)
        weighted_genotype_matrix = effective_weights * genotype_matrix_by_variant
        covariate_score = compute_covariate_score(covariate_matrix, residual_matrix)
        genotype_score = jnp.sum(genotype_matrix_by_variant * residual_matrix, axis=1)
        covariate_information_matrix = compute_covariate_information_matrix(covariate_matrix, effective_weights)
        cross_information_vector = jnp.einsum("np,mn->mp", covariate_matrix, weighted_genotype_matrix)
        genotype_information = jnp.sum(weighted_genotype_matrix * genotype_matrix_by_variant, axis=1)
        covariate_information_solution = jnp.linalg.solve(covariate_information_matrix, covariate_score[:, :, None])
        covariate_information_solution = jnp.squeeze(covariate_information_solution, axis=-1)
        cross_solution = jnp.linalg.solve(covariate_information_matrix, cross_information_vector[:, :, None])
        cross_solution = jnp.squeeze(cross_solution, axis=-1)
        schur_complement = genotype_information - jnp.sum(cross_information_vector * cross_solution, axis=1)
        safe_schur_complement = jnp.where(schur_complement > 0.0, schur_complement, 1.0)
        adjusted_genotype_score = genotype_score - jnp.sum(
            cross_information_vector * covariate_information_solution, axis=1
        )
        genotype_step = adjusted_genotype_score / safe_schur_complement
        covariate_step = covariate_information_solution - cross_solution * genotype_step[:, None]
        step = jnp.concatenate([covariate_step, genotype_step[:, None]], axis=1)
        step = jnp.where(state.converged_mask[:, None], 0.0, step)
        updated_coefficients = state.coefficients - step
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
            previous_log_likelihood=current_log_likelihood,
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
            previous_log_likelihood=jnp.full_like(initial_log_likelihood, -jnp.inf),
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
    skip_firth: jax.Array,
    max_iterations: int,
    tolerance: float,
) -> FirthVariantResult:
    """Fit Firth logistic regression for one variant.

    Args:
        covariate_matrix: Covariate design matrix.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        genotype_vector: Genotype vector for this variant.
        observation_mask: Boolean mask for valid observations.
        initial_coefficients: Starting coefficient values.
        skip_firth: Scalar boolean, when True the solver returns immediately
            with dummy values (used for variants that don't need Firth fallback).
        max_iterations: Maximum IRLS iterations.
        tolerance: Relative log-likelihood convergence tolerance.

    Returns:
        Firth regression results.

    """
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

    def compute_hessian_from_weights(weight_vector: jax.Array) -> jax.Array:
        weighted_design_matrix = full_design_matrix * weight_vector[:, None]
        return full_design_matrix.T @ weighted_design_matrix

    def compute_hdiag_and_adjusted_weights(
        information_matrix: jax.Array,
        probability_vector: jax.Array,
    ) -> AdjustedWeightComponents:
        variance_vector = jnp.where(
            observation_mask,
            jnp.clip(probability_vector * (1.0 - probability_vector), MINIMUM_WEIGHT),
            0.0,
        )
        projected_design_matrix = jnp.linalg.solve(information_matrix, full_design_matrix.T).T
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
        return (state.iteration_count < iteration_limit) & (~state.converged) & (~state.failed) & (~skip_firth)

    def body_function(state: FirthState) -> FirthState:
        probability_vector = compute_probability_vector(state.coefficients)
        information_components = compute_information_components(
            covariate_matrix=covariate_matrix,
            genotype_vector=genotype_vector,
            probability_vector=probability_vector,
            observation_mask=observation_mask,
        )
        current_penalized_log_likelihood = compute_firth_penalized_log_likelihood(
            probability_vector=probability_vector,
            phenotype_vector=phenotype_vector,
            observation_mask=observation_mask,
            information_matrix=information_components.information_matrix,
        )
        current_failed = (~jnp.isfinite(current_penalized_log_likelihood)) | (
            ~jnp.all(jnp.isfinite(state.coefficients))
        )
        adjusted_weight_components = compute_hdiag_and_adjusted_weights(
            information_matrix=information_components.information_matrix,
            probability_vector=probability_vector,
        )
        adjusted_score = full_design_matrix.T @ adjusted_weight_components.adjusted_weight_vector
        adjusted_score_maximum = jnp.max(jnp.abs(adjusted_score))
        second_hessian = compute_hessian_from_weights(adjusted_weight_components.second_weight_vector)
        coefficient_step = jnp.linalg.solve(second_hessian, adjusted_score)
        maximum_coefficient_step = jnp.max(jnp.abs(coefficient_step))
        step_scale = jnp.minimum(1.0, maximum_step_size / jnp.maximum(maximum_coefficient_step, FIRTH_TOLERANCE_FLOOR))
        scaled_coefficient_step = coefficient_step * step_scale
        updated_converged = (
            (state.iteration_count > 0)
            & (jnp.max(jnp.abs(scaled_coefficient_step)) <= coefficient_tolerance)
            & (adjusted_score_maximum < gradient_tolerance)
            & ((current_penalized_log_likelihood - state.previous_penalized_log_likelihood) < likelihood_tolerance)
            & (~current_failed)
        )
        updated_coefficients = jnp.where(
            updated_converged | current_failed, state.coefficients, state.coefficients + scaled_coefficient_step
        )
        return FirthState(
            coefficients=updated_coefficients,
            converged=updated_converged,
            failed=current_failed,
            iteration_count=state.iteration_count + jnp.asarray(1, dtype=jnp.int32),
            previous_penalized_log_likelihood=current_penalized_log_likelihood,
        )

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        FirthState(
            coefficients=initial_coefficients,
            converged=skip_firth,
            failed=jnp.zeros((), dtype=bool),
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            previous_penalized_log_likelihood=jnp.where(skip_firth, 0.0, initial_penalized_log_likelihood),
        ),
    )

    # Compute dummy results when skip_firth=True
    dummy_beta = jnp.nan
    dummy_standard_error = jnp.nan
    dummy_test_statistic = jnp.nan
    dummy_p_value = jnp.nan

    final_probability_vector = compute_probability_vector(final_state.coefficients)
    final_information_components = compute_information_components(
        covariate_matrix=covariate_matrix,
        genotype_vector=genotype_vector,
        probability_vector=final_probability_vector,
        observation_mask=observation_mask,
    )
    final_adjusted_weight_components = compute_hdiag_and_adjusted_weights(
        information_matrix=final_information_components.information_matrix,
        probability_vector=final_probability_vector,
    )
    final_second_hessian = compute_hessian_from_weights(final_adjusted_weight_components.second_weight_vector)
    final_second_covariance_matrix = jnp.linalg.inv(final_second_hessian)
    computed_result = compute_firth_statistics(
        coefficients=final_state.coefficients,
        covariance_matrix=final_second_covariance_matrix,
        converged=final_state.converged,
        failed=final_state.failed,
        iteration_count=final_state.iteration_count,
    )

    # Use jnp.where to select between dummy and computed results based on skip_firth
    return FirthVariantResult(
        beta=jnp.where(skip_firth, dummy_beta, computed_result.beta),
        standard_error=jnp.where(skip_firth, dummy_standard_error, computed_result.standard_error),
        test_statistic=jnp.where(skip_firth, dummy_test_statistic, computed_result.test_statistic),
        p_value=jnp.where(skip_firth, dummy_p_value, computed_result.p_value),
        error_code=jnp.where(skip_firth, jnp.asarray(LOGISTIC_ERROR_NONE, dtype=jnp.int32), computed_result.error_code),
        converged_mask=~skip_firth | final_state.converged,
        valid_mask=jnp.where(skip_firth, jnp.zeros((), dtype=bool), computed_result.valid_mask),
        iteration_count=jnp.where(skip_firth, jnp.asarray(0, dtype=jnp.int32), computed_result.iteration_count),
    )


compute_firth_association_chunk_variantwise = jax.jit(
    jax.vmap(
        fit_single_variant_firth_logistic_regression,
        in_axes=(None, None, 1, 0, 0, 0, None, None),
    )
)


@jax.jit
def compute_logistic_association_chunk(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix: jax.Array,
    max_iterations: int,
    tolerance: float,
    no_missing_constants: NoMissingLogisticConstants | None = None,
) -> LogisticAssociationChunkResult:
    """Compute batched logistic association statistics for a chunk of variants.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        genotype_matrix: Genotype matrix.
        max_iterations: Maximum IRLS iterations.
        tolerance: Relative log-likelihood convergence tolerance.
        no_missing_constants: Optional precomputed constants for the no-missing path.

    Returns:
        Chunk-level logistic association statistics.

    """
    if no_missing_constants is None:
        no_missing_constants = prepare_no_missing_logistic_constants(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
        )

    genotype_matrix_by_variant = genotype_matrix.T
    variant_count = genotype_matrix_by_variant.shape[0]
    sample_count = genotype_matrix.shape[0]
    firth_tolerance = jnp.maximum(tolerance, FIRTH_TOLERANCE_FLOOR)

    def run_standard_logistic() -> StandardLogisticChunkEvaluation:
        return compute_standard_logistic_association_chunk_without_mask(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            genotype_matrix=genotype_matrix,
            no_missing_constants=no_missing_constants,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    def build_firth_result(
        standard_evaluation: StandardLogisticChunkEvaluation,
    ) -> LogisticAssociationChunkResult:
        """Build Firth result using jax.lax.cond to avoid host sync."""
        standard_result = standard_evaluation.logistic_result
        standard_coefficients = standard_evaluation.coefficients

        # Compute heuristic mask on device
        heuristic_mask = compute_firth_pre_dispatch_mask_without_mask(
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            no_missing_constants=no_missing_constants,
        )

        # Compute fallback mask on device
        fallback_mask = heuristic_mask | (~standard_result.converged_mask) | (~standard_result.valid_mask)

        # Initialize Firth coefficients: use heuristic init where needed, otherwise standard
        firth_initial_coefficients = jnp.where(
            heuristic_mask[:, None],
            initialize_full_model_coefficients_without_mask(
                covariate_matrix=covariate_matrix,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                no_missing_constants=no_missing_constants,
            ),
            standard_coefficients,
        )

        # Create observation mask for all variants (all ones for no-missing path)
        observation_mask = jnp.ones((variant_count, sample_count), dtype=bool)

        # Create skip_firth mask (inverse of fallback_mask)
        skip_firth_mask = ~fallback_mask

        def firth_branch(
            standard_result_inner: LogisticAssociationChunkResult,
        ) -> LogisticAssociationChunkResult:
            """Run Firth on all variants with skip flag, then merge results."""
            firth_result = compute_firth_association_chunk_variantwise(
                covariate_matrix,
                phenotype_vector,
                genotype_matrix,
                observation_mask,
                firth_initial_coefficients,
                skip_firth_mask,
                max_iterations,
                firth_tolerance,
            )

            # Merge: use Firth where fallback_mask is True, standard otherwise
            return LogisticAssociationChunkResult(
                beta=jnp.where(
                    fallback_mask, firth_result.beta, standard_result_inner.beta
                ),
                standard_error=jnp.where(
                    fallback_mask, firth_result.standard_error, standard_result_inner.standard_error
                ),
                test_statistic=jnp.where(
                    fallback_mask, firth_result.test_statistic, standard_result_inner.test_statistic
                ),
                p_value=jnp.where(
                    fallback_mask, firth_result.p_value, standard_result_inner.p_value
                ),
                method_code=jnp.where(
                    fallback_mask,
                    jnp.full((variant_count,), LOGISTIC_METHOD_FIRTH, dtype=jnp.int32),
                    standard_result_inner.method_code,
                ),
                error_code=jnp.where(
                    fallback_mask, firth_result.error_code, standard_result_inner.error_code
                ),
                converged_mask=jnp.where(
                    fallback_mask, firth_result.converged_mask, standard_result_inner.converged_mask
                ),
                valid_mask=jnp.where(
                    fallback_mask, firth_result.valid_mask, standard_result_inner.valid_mask
                ),
                iteration_count=jnp.where(
                    fallback_mask, firth_result.iteration_count, standard_result_inner.iteration_count
                ),
            )

        def no_firth_branch(
            standard_result_inner: LogisticAssociationChunkResult,
        ) -> LogisticAssociationChunkResult:
            """Return standard result unchanged."""
            return standard_result_inner

        # Use jax.lax.cond to branch on device without host sync
        return jax.lax.cond(
            jnp.any(fallback_mask),
            lambda: firth_branch(standard_result),
            lambda: no_firth_branch(standard_result),
        )

    # Run standard logistic (this is always needed as first step)
    standard_evaluation = run_standard_logistic()

    # Build final result with conditional Firth fallback (no host sync)
    return build_firth_result(standard_evaluation)


@jax.jit
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
    variant_count = genotype_matrix_by_variant.shape[0]
    firth_tolerance = jnp.maximum(tolerance, FIRTH_TOLERANCE_FLOOR)

    def run_standard_logistic_masked() -> StandardLogisticChunkEvaluation:
        return compute_standard_logistic_association_chunk_with_mask(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            genotype_matrix=genotype_matrix,
            observation_mask=observation_mask,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    def build_firth_result_masked(
        standard_evaluation: StandardLogisticChunkEvaluation,
    ) -> LogisticAssociationChunkResult:
        """Build Firth result using jax.lax.cond to avoid host sync."""
        standard_result = standard_evaluation.logistic_result
        standard_coefficients = standard_evaluation.coefficients

        # Compute heuristic mask on device
        heuristic_mask = compute_firth_pre_dispatch_mask(
            phenotype_vector=phenotype_vector,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            observation_mask=observation_mask,
        )

        # Compute fallback mask on device
        fallback_mask = heuristic_mask | (~standard_result.converged_mask) | (~standard_result.valid_mask)

        # Initialize Firth coefficients: use heuristic init where needed, otherwise standard
        firth_initial_coefficients = jnp.where(
            heuristic_mask[:, None],
            initialize_full_model_coefficients(
                covariate_matrix=covariate_matrix,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                phenotype_vector=phenotype_vector,
                observation_mask=observation_mask,
            ),
            standard_coefficients,
        )

        # Create skip_firth mask (inverse of fallback_mask)
        skip_firth_mask = ~fallback_mask

        def firth_branch_masked(
            standard_result_inner: LogisticAssociationChunkResult,
        ) -> LogisticAssociationChunkResult:
            """Run Firth on all variants with skip flag, then merge results."""
            firth_result = compute_firth_association_chunk_variantwise(
                covariate_matrix,
                phenotype_vector,
                genotype_matrix,
                observation_mask,
                firth_initial_coefficients,
                skip_firth_mask,
                max_iterations,
                firth_tolerance,
            )

            # Merge: use Firth where fallback_mask is True, standard otherwise
            return LogisticAssociationChunkResult(
                beta=jnp.where(
                    fallback_mask, firth_result.beta, standard_result_inner.beta
                ),
                standard_error=jnp.where(
                    fallback_mask, firth_result.standard_error, standard_result_inner.standard_error
                ),
                test_statistic=jnp.where(
                    fallback_mask, firth_result.test_statistic, standard_result_inner.test_statistic
                ),
                p_value=jnp.where(
                    fallback_mask, firth_result.p_value, standard_result_inner.p_value
                ),
                method_code=jnp.where(
                    fallback_mask,
                    jnp.full((variant_count,), LOGISTIC_METHOD_FIRTH, dtype=jnp.int32),
                    standard_result_inner.method_code,
                ),
                error_code=jnp.where(
                    fallback_mask, firth_result.error_code, standard_result_inner.error_code
                ),
                converged_mask=jnp.where(
                    fallback_mask, firth_result.converged_mask, standard_result_inner.converged_mask
                ),
                valid_mask=jnp.where(
                    fallback_mask, firth_result.valid_mask, standard_result_inner.valid_mask
                ),
                iteration_count=jnp.where(
                    fallback_mask, firth_result.iteration_count, standard_result_inner.iteration_count
                ),
            )

        def no_firth_branch_masked(
            standard_result_inner: LogisticAssociationChunkResult,
        ) -> LogisticAssociationChunkResult:
            """Return standard result unchanged."""
            return standard_result_inner

        # Use jax.lax.cond to branch on device without host sync
        return jax.lax.cond(
            jnp.any(fallback_mask),
            lambda: firth_branch_masked(standard_result),
            lambda: no_firth_branch_masked(standard_result),
        )

    # Run standard logistic with mask (this is always needed as first step)
    standard_evaluation = run_standard_logistic_masked()

    # Build final result with conditional Firth fallback (no host sync)
    return build_firth_result_masked(standard_evaluation)
