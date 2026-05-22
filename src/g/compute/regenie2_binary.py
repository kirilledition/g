"""REGENIE step 2 binary score-test kernel with device-resident Firth fallback."""

from __future__ import annotations

import enum
import functools
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g import types
from g.compute import regenie2_binary_candidate_planning, regenie2_binary_types
from g.compute.common import genotype, linalg, pvalue

jax.config.update("jax_enable_x64", val=True)

MINIMUM_PROBABILITY = 1.0e-6
MINIMUM_VARIANCE = 1.0e-8
RELATIVE_VARIANCE_TOLERANCE = 1.0e-6
DEFAULT_MAXIMUM_NULL_ITERATIONS = 50
NULL_LOGISTIC_COEFFICIENT_TOLERANCE = 1.0e-6
FIRTH_NULL_MAXIMUM_ITERATIONS = 1000
FIRTH_NULL_FALLBACK_ITERATION_MULTIPLIER = 5
FIRTH_NULL_GRADIENT_TOLERANCE = 50.0e-6
FIRTH_NULL_MAXIMUM_STEP_SIZE = 25.0
FIRTH_NULL_FALLBACK_STEP_DIVISOR = 5.0
INITIAL_RESPONSE_SCALE = 4.863891244002886
BINARY_CASE_THRESHOLD = 0.5
ALLELE_COUNT_MULTIPLIER = 2.0
FIRTH_GRADIENT_TOLERANCE = 2.5e-4
FIRTH_COEFFICIENT_TOLERANCE = 2.5e-4
FIRTH_LIKELIHOOD_TOLERANCE = 2.5e-4
FIRTH_MAXIMUM_STEP_SIZE = 5.0
FIRTH_MAXIMUM_ITERATIONS = 250
FIRTH_PSEUDO_MAXIMUM_ITERATIONS = 50
FIRTH_NEWTON_RAPHSON_ZERO_START_ITERATIONS = 100
FIRTH_PSEUDO_INNER_MAXIMUM_ITERATIONS = 25
FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS = 25
FIRTH_TOLERANCE = 2.5e-4
FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS = 12
REGENIE_LOGISTIC_MINIMUM_ETA = -30.0
REGENIE_LOGISTIC_MAXIMUM_ETA = 30.0
REGENIE_NUMERICAL_EPSILON = 10.0 * 2.220446049250313e-16
SPARSE_CARRIER_DOSAGE_THRESHOLD = 1.0e-4
DEFAULT_BINARY_KERNEL_CONFIG = regenie2_binary_types.BinaryKernelConfig(
    maximum_null_iterations=DEFAULT_MAXIMUM_NULL_ITERATIONS,
    null_logistic_coefficient_tolerance=NULL_LOGISTIC_COEFFICIENT_TOLERANCE,
    firth_batch_size=regenie2_binary_candidate_planning.DEFAULT_FIRTH_BATCH_SIZE,
    firth_candidate_capacity=regenie2_binary_candidate_planning.DEFAULT_FIRTH_CANDIDATE_CAPACITY,
    firth_maximum_iterations=FIRTH_MAXIMUM_ITERATIONS,
    firth_gradient_tolerance=FIRTH_GRADIENT_TOLERANCE,
    firth_coefficient_tolerance=FIRTH_COEFFICIENT_TOLERANCE,
    firth_likelihood_tolerance=FIRTH_LIKELIHOOD_TOLERANCE,
    firth_maximum_step_size=FIRTH_MAXIMUM_STEP_SIZE,
    use_block_firth_math=False,
)


class FirthConvergenceReason(enum.IntEnum):
    """Internal integer termination reasons for binary Firth fitting."""

    NONE = 0
    CONVERGED = 1
    NUMERICAL_FAILURE = 2
    MAX_ITERATIONS = 3
    INVALID_STATISTIC = 4
    STEP_HALVING_EXHAUSTED = 5
    NULL_FAILURE = 6
    NEGATIVE_LRT = 7
    PROBABILITY_FAILURE = 8
    STEP_SIZE_INCREASE = 9


BinaryScoreTestChunkComputeFunction = typing.Callable[
    [regenie2_binary_types.Regenie2BinaryChromosomeState, jax.Array, types.BinaryCorrectionPlan],
    regenie2_binary_types.Regenie2BinaryChunkResult,
]
BinaryChunkComputeFunction = typing.Callable[
    [
        regenie2_binary_types.Regenie2BinaryChromosomeState,
        jax.Array,
        types.BinaryCorrectionPlan,
        jax.Array | None,
        regenie2_binary_types.BinaryKernelConfig,
    ],
    regenie2_binary_types.Regenie2BinaryChunkResult,
]
BinaryVariantMajorChunkComputeFunction = typing.Callable[
    [
        regenie2_binary_types.Regenie2BinaryChromosomeState,
        jax.Array,
        types.BinaryCorrectionPlan,
        jax.Array | None,
        regenie2_binary_types.BinaryKernelConfig,
    ],
    regenie2_binary_types.Regenie2BinaryChunkResult,
]


RegenieGenotypeFlipResult = genotype.RegenieGenotypeFlipResult


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthState:
    """State container for one Firth-regression lane.

    Attributes:
        coefficients: Current coefficient estimates.
        penalized_log_likelihood: Current accepted penalized log-likelihood.
        converged: Whether the solver converged.
        failed: Whether the solver hit an unrecoverable numerical failure.
        iteration_count: Number of update steps performed.
        termination_reason_code: Internal termination-reason code.

    """

    coefficients: jax.Array
    penalized_log_likelihood: jax.Array
    converged: jax.Array
    failed: jax.Array
    iteration_count: jax.Array
    termination_reason_code: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthBacktrackingState:
    """Loop state for bounded Firth step-halving.

    Attributes:
        attempt_count: Number of candidate steps evaluated.
        next_coefficient_step: Candidate coefficient step for the next attempt.
        accepted_coefficient_step: Accepted coefficient step, or zeros before acceptance.
        accepted_coefficients: Accepted candidate coefficients, or current coefficients before acceptance.
        accepted_penalized_log_likelihood: Accepted candidate penalized log-likelihood.
        accepted: Whether a candidate step has been accepted.

    """

    attempt_count: jax.Array
    next_coefficient_step: jax.Array
    accepted_coefficient_step: jax.Array
    accepted_coefficients: jax.Array
    accepted_penalized_log_likelihood: jax.Array
    accepted: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthBacktrackingResult:
    """Result of bounded Firth step-halving.

    Attributes:
        coefficient_step: Accepted coefficient step, or zeros when exhausted.
        coefficients: Accepted candidate coefficients, or current coefficients when exhausted.
        penalized_log_likelihood: Accepted candidate penalized log-likelihood, or current value when exhausted.
        accepted: Whether any candidate step was accepted.
        exhausted: Whether step-halving attempts were exhausted.

    """

    coefficient_step: jax.Array
    coefficients: jax.Array
    penalized_log_likelihood: jax.Array
    accepted: jax.Array
    exhausted: jax.Array


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
class FullModelScoreComponents:
    """Score components for one full Firth model.

    Attributes:
        covariate_score: Covariate score vector.
        genotype_score: Genotype score scalar.

    """

    covariate_score: jax.Array
    genotype_score: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullLogisticFitState:
    """State for covariate-only null logistic IRLS.

    Attributes:
        coefficients: Current coefficient estimates.
        iteration_count: Number of IRLS updates performed.
        converged: Whether the coefficient update tolerance has been reached.

    """

    coefficients: jax.Array
    iteration_count: jax.Array
    converged: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthFitResult:
    """Result of the covariate-only Firth null fit.

    Attributes:
        coefficients: Final covariate coefficients, or the last attempted coefficients on failure.
        penalized_log_likelihood: Final trusted penalized log-likelihood, or NaN on failure.
        iteration_count: Number of solver iterations performed.
        convergence_reason_code: Internal termination-reason code.
        converged: Whether the null fit converged.

    """

    coefficients: jax.Array
    penalized_log_likelihood: jax.Array
    iteration_count: jax.Array
    convergence_reason_code: jax.Array
    converged: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthComponents:
    """Intermediate quantities for REGENIE-style null Firth Newton-Raphson."""

    probability_vector: jax.Array
    weight_vector: jax.Array
    information_matrix: jax.Array
    information_cholesky_factor: jax.Array
    deviance: jax.Array
    leverage_vector: jax.Array
    modified_score: jax.Array
    valid: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthNewtonRaphsonState:
    """Loop state for covariate-only null Firth Newton-Raphson."""

    coefficients: jax.Array
    deviance: jax.Array
    converged: jax.Array
    failed: jax.Array
    iteration_count: jax.Array
    termination_reason_code: jax.Array
    previous_score_maximum: jax.Array
    score_increase_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthLineSearchState:
    """Line-search state for covariate-only null Firth Newton-Raphson."""

    attempt_count: jax.Array
    next_coefficient_step: jax.Array
    accepted_coefficients: jax.Array
    accepted_deviance: jax.Array
    accepted: jax.Array
    valid: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthLineSearchResult:
    """Result of null Firth deviance-decreasing step-halving."""

    coefficients: jax.Array
    deviance: jax.Array
    accepted: jax.Array
    valid: jax.Array


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
        failure_code: Integer failure-reason code.
        convergence_reason_code: Internal termination-reason code.
        correction_code: Integer final correction branch code.
        sparse_correction_mask: Whether the lane used carrier-only sparse inputs.
        pseudo_firth_iteration_count: Iterations used by the scalar pseudo-Firth attempt.
        nr_zero_start_iteration_count: Iterations used by the zero-start Newton-Raphson fallback.
        nr_warm_start_iteration_count: Iterations used by the warm-start Newton-Raphson fallback.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    penalized_log_likelihood: jax.Array
    converged_mask: jax.Array
    valid_mask: jax.Array
    iteration_count: jax.Array
    failure_code: jax.Array
    convergence_reason_code: jax.Array
    correction_code: jax.Array
    sparse_correction_mask: jax.Array
    pseudo_firth_iteration_count: jax.Array
    nr_zero_start_iteration_count: jax.Array
    nr_warm_start_iteration_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarFirthComponents:
    """Scalar approximate-Firth quantities for one beta value.

    Attributes:
        probability_vector: Fitted probabilities for the active correction samples.
        weight_vector: Bernoulli weights for the active correction samples.
        genotype_information: Scalar genotype information.
        genotype_information_diagonal: Per-sample contributions to genotype information.
        penalized_deviance: REGENIE approximate penalized deviance.
        score: Scalar modified score.
        valid: Whether probabilities, weights, and information are finite and usable.

    """

    probability_vector: jax.Array
    weight_vector: jax.Array
    genotype_information: jax.Array
    genotype_information_diagonal: jax.Array
    penalized_deviance: jax.Array
    score: jax.Array
    valid: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarPseudoFirthState:
    """Loop state for REGENIE scalar pseudo-Firth."""

    beta: jax.Array
    penalized_deviance: jax.Array
    genotype_information: jax.Array
    score: jax.Array
    outer_iteration_count: jax.Array
    inner_iteration_count: jax.Array
    beta_iteration_14: jax.Array
    converged: jax.Array
    failed: jax.Array
    failure_code: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarPseudoLogisticState:
    """Inner pseudo-response logistic state for one scalar beta update."""

    beta: jax.Array
    score: jax.Array
    genotype_information: jax.Array
    step_size: jax.Array
    previous_step_size: jax.Array
    iteration_count: jax.Array
    converged: jax.Array
    failed: jax.Array
    failure_code: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarNewtonRaphsonState:
    """Loop state for REGENIE scalar Newton-Raphson Firth fallback."""

    beta: jax.Array
    penalized_deviance: jax.Array
    genotype_information: jax.Array
    genotype_information_diagonal: jax.Array
    probability_vector: jax.Array
    score: jax.Array
    iteration_count: jax.Array
    converged: jax.Array
    failed: jax.Array
    failure_reason_code: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarLineSearchState:
    """Line-search state for scalar Newton-Raphson Firth."""

    beta: jax.Array
    step_size: jax.Array
    penalized_deviance: jax.Array
    genotype_information: jax.Array
    genotype_information_diagonal: jax.Array
    probability_vector: jax.Array
    attempt_count: jax.Array
    accepted: jax.Array
    valid: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarFirthAttemptResult:
    """Result for one scalar approximate-Firth attempt."""

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    penalized_deviance: jax.Array
    genotype_information: jax.Array
    converged: jax.Array
    valid: jax.Array
    iteration_count: jax.Array
    failure_reason_code: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ApproximateFirthCandidateInputs:
    """Prepared scalar approximate-Firth inputs for one candidate."""

    phenotype_vector: jax.Array
    genotype_vector: jax.Array
    offset_vector: jax.Array
    active_sample_mask: jax.Array
    sparse_correction: jax.Array
    warm_start_beta: jax.Array


def prepare_regenie2_binary_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> regenie2_binary_types.Regenie2BinaryState:
    """Prepare reusable binary step 2 state.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.

    Returns:
        Reusable binary step 2 state.

    """
    covariate_matrix_float32 = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_vector_float32 = jnp.asarray(phenotype_vector, dtype=jnp.float32)
    return regenie2_binary_types.Regenie2BinaryState(
        covariate_matrix=covariate_matrix_float32,
        phenotype_vector=phenotype_vector_float32,
        sample_count=jnp.asarray(covariate_matrix_float32.shape[0], dtype=jnp.int32),
    )


def prepare_regenie2_multi_binary_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
) -> regenie2_binary_types.Regenie2MultiBinaryState:
    """Prepare reusable multi-trait binary step 2 state.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_matrix: Binary phenotype matrix in trait-major 0/1 encoding.

    Returns:
        Reusable multi-trait binary step 2 state.

    """
    covariate_matrix_float32 = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_matrix_float32 = jnp.asarray(phenotype_matrix, dtype=jnp.float32)
    return regenie2_binary_types.Regenie2MultiBinaryState(
        covariate_matrix=covariate_matrix_float32,
        phenotype_matrix=phenotype_matrix_float32,
        sample_count=jnp.asarray(covariate_matrix_float32.shape[0], dtype=jnp.int32),
    )


def compute_logistic_probability(linear_predictor: jax.Array) -> jax.Array:
    """Compute clipped logistic probabilities."""
    probability = jax.nn.sigmoid(linear_predictor)
    return jnp.clip(probability, MINIMUM_PROBABILITY, 1.0 - MINIMUM_PROBABILITY)


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


def solve_from_positive_definite_matrix(
    positive_definite_matrix: jax.Array,
    right_hand_side: jax.Array,
) -> jax.Array:
    """Solve a positive-definite system from its matrix form."""
    return linalg.solve_from_positive_definite_matrix(positive_definite_matrix, right_hand_side)


def compute_positive_variance_mask(variance: jax.Array, reference_sum_squares: jax.Array) -> jax.Array:
    """Return a stable positive-variance mask after covariate projection."""
    variance_floor = jnp.maximum(MINIMUM_VARIANCE, reference_sum_squares * RELATIVE_VARIANCE_TOLERANCE)
    return variance > variance_floor


@functools.partial(jax.jit, static_argnames=("maximum_iterations", "kernel_config"))
def fit_null_logistic_coefficients(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    maximum_iterations: int | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> NullLogisticFitState:
    """Fit a covariate-only logistic null model with a fixed LOCO offset."""
    covariate_count = covariate_matrix.shape[1]
    resolved_maximum_iterations = (
        kernel_config.maximum_null_iterations if maximum_iterations is None else maximum_iterations
    )
    coefficient_tolerance = kernel_config.null_logistic_coefficient_tolerance

    def condition_function(state: NullLogisticFitState) -> jax.Array:
        return (state.iteration_count < resolved_maximum_iterations) & (~state.converged)

    def body_function(state: NullLogisticFitState) -> NullLogisticFitState:
        linear_predictor = covariate_matrix @ state.coefficients + loco_offset
        fitted_probability = compute_logistic_probability(linear_predictor)
        weight_vector = jnp.maximum(fitted_probability * (1.0 - fitted_probability), MINIMUM_VARIANCE)
        score_vector = covariate_matrix.T @ (phenotype_vector - fitted_probability)
        information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
        cholesky_factor = jnp.linalg.cholesky(
            information_matrix + jnp.eye(covariate_count, dtype=jnp.float32) * MINIMUM_VARIANCE
        )
        coefficient_delta = linalg.solve_positive_definite_system(cholesky_factor, score_vector)
        updated_iteration_count = state.iteration_count + jnp.asarray(1, dtype=jnp.int32)
        converged = (updated_iteration_count > 0) & (jnp.max(jnp.abs(coefficient_delta)) <= coefficient_tolerance)
        return NullLogisticFitState(
            coefficients=state.coefficients + coefficient_delta,
            iteration_count=updated_iteration_count,
            converged=converged,
        )

    initial_coefficients = jnp.zeros(covariate_count, dtype=jnp.float32)
    return jax.lax.while_loop(
        condition_function,
        body_function,
        NullLogisticFitState(
            coefficients=initial_coefficients,
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            converged=jnp.asarray(0, dtype=jnp.bool_),
        ),
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def prepare_regenie2_binary_chromosome_state(
    state: regenie2_binary_types.Regenie2BinaryState,
    loco_offset: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChromosomeState:
    """Prepare chromosome-specific null logistic state reused across chunks."""
    loco_offset_float32 = jnp.asarray(loco_offset, dtype=jnp.float32)
    null_logistic_fit_state = fit_null_logistic_coefficients(
        state.covariate_matrix,
        state.phenotype_vector,
        loco_offset_float32,
        kernel_config=kernel_config,
    )
    null_logistic_coefficients = null_logistic_fit_state.coefficients
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
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        null_firth_coefficients = jnp.asarray(null_logistic_coefficients, dtype=jnp.float64)
        null_firth_offset = state.covariate_matrix.astype(jnp.float64) @ null_firth_coefficients + jnp.asarray(
            loco_offset_float32, dtype=jnp.float64
        )
        null_firth_result = NullFirthFitResult(
            coefficients=null_firth_coefficients,
            penalized_log_likelihood=jnp.asarray(0.0, dtype=jnp.float64),
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            convergence_reason_code=jnp.asarray(FirthConvergenceReason.NONE.value, dtype=jnp.int32),
            converged=jnp.asarray(1, dtype=jnp.bool_),
        )
    else:
        null_firth_result = fit_covariate_only_firth_null_model(
            covariate_matrix=state.covariate_matrix,
            phenotype_vector=state.phenotype_vector,
            loco_offset=loco_offset_float32,
            initial_coefficients=null_logistic_coefficients,
            kernel_config=kernel_config,
        )
        null_firth_offset = state.covariate_matrix.astype(jnp.float64) @ null_firth_result.coefficients + jnp.asarray(
            loco_offset_float32, dtype=jnp.float64
        )
    return regenie2_binary_types.Regenie2BinaryChromosomeState(
        covariate_matrix=state.covariate_matrix,
        phenotype_vector=state.phenotype_vector,
        null_logistic_coefficients=null_logistic_coefficients,
        null_firth_coefficients=null_firth_result.coefficients,
        null_firth_offset=null_firth_offset,
        fitted_probability=fitted_probability,
        score_residual=score_residual,
        loco_offset=loco_offset_float32,
        standardized_residual=standardized_residual,
        square_root_weight=square_root_weight,
        weighted_genotype_projection_matrix=weighted_genotype_projection_matrix,
        null_firth_penalized_log_likelihood=null_firth_result.penalized_log_likelihood,
        null_firth_iteration_count=null_firth_result.iteration_count,
        null_firth_convergence_reason_code=null_firth_result.convergence_reason_code,
        null_logistic_iteration_count=null_logistic_fit_state.iteration_count,
        null_logistic_converged=null_logistic_fit_state.converged,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def prepare_regenie2_multi_binary_chromosome_state(
    state: regenie2_binary_types.Regenie2MultiBinaryState,
    loco_offset_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2MultiBinaryChromosomeState:
    """Prepare chromosome-specific null logistic state for all requested binary traits."""
    loco_offset_matrix_float32 = jnp.asarray(loco_offset_matrix, dtype=jnp.float32)

    def prepare_one_trait(
        phenotype_vector: jax.Array,
        loco_offset: jax.Array,
    ) -> regenie2_binary_types.Regenie2BinaryChromosomeState:
        trait_state = regenie2_binary_types.Regenie2BinaryState(
            covariate_matrix=state.covariate_matrix,
            phenotype_vector=phenotype_vector,
            sample_count=state.sample_count,
        )
        return prepare_regenie2_binary_chromosome_state(trait_state, loco_offset, correction_plan, kernel_config)

    chromosome_states = jax.vmap(prepare_one_trait)(state.phenotype_matrix, loco_offset_matrix_float32)
    return regenie2_binary_types.Regenie2MultiBinaryChromosomeState(
        covariate_matrix=state.covariate_matrix,
        phenotype_matrix=state.phenotype_matrix,
        null_logistic_coefficients=chromosome_states.null_logistic_coefficients,
        null_firth_coefficients=chromosome_states.null_firth_coefficients,
        null_firth_offset_matrix=chromosome_states.null_firth_offset,
        fitted_probability=chromosome_states.fitted_probability,
        score_residual=chromosome_states.score_residual,
        loco_offset_matrix=chromosome_states.loco_offset,
        standardized_residual=chromosome_states.standardized_residual,
        square_root_weight=chromosome_states.square_root_weight,
        weighted_genotype_projection_matrix=chromosome_states.weighted_genotype_projection_matrix,
        null_firth_penalized_log_likelihood=chromosome_states.null_firth_penalized_log_likelihood,
        null_firth_iteration_count=chromosome_states.null_firth_iteration_count,
        null_firth_convergence_reason_code=chromosome_states.null_firth_convergence_reason_code,
        null_logistic_iteration_count=chromosome_states.null_logistic_iteration_count,
        null_logistic_converged=chromosome_states.null_logistic_converged,
    )


def compute_regenie2_binary_score_test_chunk_variant_major_core(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute the binary score test from canonical variant-major genotypes.

    Args:
        chromosome_state: Chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        correction_plan: Binary fallback/correction policy.

    Returns:
        Uncorrected score-test result for the chunk.

    """
    raw_genotype_matrix_by_variant = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    genotype_flip_result = build_regenie_flipped_genotypes(raw_genotype_matrix_by_variant)
    genotype_matrix_by_variant_float32 = genotype_flip_result.genotype_matrix_by_variant
    weighted_genotype_matrix_by_variant = (
        genotype_matrix_by_variant_float32 * chromosome_state.square_root_weight[None, :]
    )
    projection_coordinates = (
        weighted_genotype_matrix_by_variant @ chromosome_state.weighted_genotype_projection_matrix.T
    )
    weighted_genotype_sum_squares = jnp.einsum(
        "ij,ij->i",
        weighted_genotype_matrix_by_variant,
        weighted_genotype_matrix_by_variant,
    )
    projection_sum_squares = jnp.einsum("ij,ij->i", projection_coordinates, projection_coordinates)
    variance = jnp.maximum(weighted_genotype_sum_squares - projection_sum_squares, 0.0)
    score = genotype_matrix_by_variant_float32 @ chromosome_state.score_residual
    null_logistic_converged = chromosome_state.null_logistic_converged
    positive_variance_mask = compute_positive_variance_mask(variance, weighted_genotype_sum_squares)
    statistic_mask = positive_variance_mask & null_logistic_converged
    inverse_variance = jnp.where(statistic_mask, jnp.reciprocal(variance), 0.0)
    beta = jnp.where(
        statistic_mask,
        jnp.where(genotype_flip_result.flip_mask, -score * inverse_variance, score * inverse_variance),
        jnp.nan,
    )
    standard_error = jnp.where(statistic_mask, jnp.sqrt(inverse_variance), jnp.nan)
    chi_squared = jnp.where(
        null_logistic_converged,
        jnp.where(positive_variance_mask, score * score * inverse_variance, 0.0),
        jnp.nan,
    )
    log10_p_value = jnp.where(
        null_logistic_converged,
        pvalue.chi_squared_to_log10_p_value(chi_squared),
        jnp.nan,
    )
    valid_mask = null_logistic_converged & jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    extra_code = regenie2_binary_candidate_planning.build_extra_code(log10_p_value, valid_mask, correction_plan)
    return regenie2_binary_types.Regenie2BinaryChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        valid_mask=valid_mask,
        firth_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_failure_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_convergence_reason_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_correction_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_sparse_correction_mask=jnp.zeros_like(extra_code, dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
    )


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def compute_regenie2_binary_score_test_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute the uncorrected score-test result for one binary chunk."""
    return compute_regenie2_binary_score_test_chunk_variant_major_core(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(genotype_matrix, dtype=jnp.float32).T,
        correction_plan=correction_plan,
    )


compute_regenie2_binary_score_test_chunk = typing.cast(
    "BinaryScoreTestChunkComputeFunction",
    compute_regenie2_binary_score_test_chunk_from_chromosome_state,
)


def build_single_binary_chromosome_state_from_multi(
    chromosome_state: regenie2_binary_types.Regenie2MultiBinaryChromosomeState,
    trait_index: jax.Array,
) -> regenie2_binary_types.Regenie2BinaryChromosomeState:
    """Build a single-trait chromosome state view from a multi-trait state."""
    return regenie2_binary_types.Regenie2BinaryChromosomeState(
        covariate_matrix=chromosome_state.covariate_matrix,
        phenotype_vector=chromosome_state.phenotype_matrix[trait_index],
        null_logistic_coefficients=chromosome_state.null_logistic_coefficients[trait_index],
        null_firth_coefficients=chromosome_state.null_firth_coefficients[trait_index],
        null_firth_offset=chromosome_state.null_firth_offset_matrix[trait_index],
        fitted_probability=chromosome_state.fitted_probability[trait_index],
        score_residual=chromosome_state.score_residual[trait_index],
        loco_offset=chromosome_state.loco_offset_matrix[trait_index],
        standardized_residual=chromosome_state.standardized_residual[trait_index],
        square_root_weight=chromosome_state.square_root_weight[trait_index],
        weighted_genotype_projection_matrix=chromosome_state.weighted_genotype_projection_matrix[trait_index],
        null_firth_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood[trait_index],
        null_firth_iteration_count=chromosome_state.null_firth_iteration_count[trait_index],
        null_firth_convergence_reason_code=chromosome_state.null_firth_convergence_reason_code[trait_index],
        null_logistic_iteration_count=chromosome_state.null_logistic_iteration_count[trait_index],
        null_logistic_converged=chromosome_state.null_logistic_converged[trait_index],
    )


def build_multi_binary_chunk_result(
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Rewrap a vmapped single-trait binary result as a multi-trait result."""
    return regenie2_binary_types.Regenie2MultiBinaryChunkResult(
        beta=result.beta,
        standard_error=result.standard_error,
        chi_squared=result.chi_squared,
        log10_p_value=result.log10_p_value,
        extra_code=result.extra_code,
        valid_mask=result.valid_mask,
        firth_iteration_count=result.firth_iteration_count,
        firth_failure_code=result.firth_failure_code,
        firth_convergence_reason_code=result.firth_convergence_reason_code,
        firth_correction_code=result.firth_correction_code,
        firth_sparse_correction_mask=result.firth_sparse_correction_mask,
        pseudo_firth_iteration_count=result.pseudo_firth_iteration_count,
        nr_zero_start_iteration_count=result.nr_zero_start_iteration_count,
        nr_warm_start_iteration_count=result.nr_warm_start_iteration_count,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def compute_regenie2_multi_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_types.Regenie2MultiBinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary REGENIE step 2 association using one genotype chunk."""

    def compute_one_trait(trait_index: jax.Array) -> regenie2_binary_types.Regenie2BinaryChunkResult:
        single_chromosome_state = build_single_binary_chromosome_state_from_multi(chromosome_state, trait_index)
        return compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=single_chromosome_state,
            genotype_matrix=genotype_matrix,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=kernel_config,
        )

    trait_count = chromosome_state.phenotype_matrix.shape[0]
    return build_multi_binary_chunk_result(jax.vmap(compute_one_trait)(jnp.arange(trait_count, dtype=jnp.int32)))


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_types.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary association from variant-major genotypes."""

    def compute_one_trait(trait_index: jax.Array) -> regenie2_binary_types.Regenie2BinaryChunkResult:
        single_chromosome_state = build_single_binary_chromosome_state_from_multi(chromosome_state, trait_index)
        compute_variant_major_chunk = (
            regenie2_binary_variant_major.compute_regenie2_binary_chunk_from_chromosome_state_variant_major
        )
        return compute_variant_major_chunk(
            chromosome_state=single_chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=kernel_config,
        )

    trait_count = chromosome_state.phenotype_matrix.shape[0]
    return build_multi_binary_chunk_result(jax.vmap(compute_one_trait)(jnp.arange(trait_count, dtype=jnp.int32)))


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


def compute_weighted_full_model_information_components(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    weight_vector: jax.Array,
) -> InformationComponents:
    """Compute full-model information blocks for one explicit weight vector."""
    weighted_genotype_vector = weight_vector * genotype_vector
    covariate_information_matrix = (covariate_matrix.T * weight_vector) @ covariate_matrix
    cross_information_vector = weighted_genotype_vector @ covariate_matrix
    genotype_information = jnp.dot(weighted_genotype_vector, genotype_vector)
    return InformationComponents(
        covariate_information_matrix=covariate_information_matrix,
        cross_information_vector=cross_information_vector,
        genotype_information=genotype_information,
        information_matrix=build_full_model_information_matrix(
            covariate_information_matrix=covariate_information_matrix,
            cross_information_vector=cross_information_vector,
            genotype_information=genotype_information,
        ),
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


def compute_firth_convergence_mask(
    *,
    current_penalized_log_likelihood: jax.Array,
    candidate_penalized_log_likelihood: jax.Array,
    coefficient_step: jax.Array,
    adjusted_score: jax.Array,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> jax.Array:
    """Return whether an accepted Firth step satisfies convergence tolerances."""
    likelihood_delta = candidate_penalized_log_likelihood - current_penalized_log_likelihood
    finite_mask = (
        jnp.isfinite(current_penalized_log_likelihood)
        & jnp.isfinite(candidate_penalized_log_likelihood)
        & jnp.all(jnp.isfinite(coefficient_step))
        & jnp.all(jnp.isfinite(adjusted_score))
    )
    monotonic_mask = likelihood_delta >= -kernel_config.firth_likelihood_tolerance
    likelihood_tolerance_mask = jnp.abs(likelihood_delta) <= kernel_config.firth_likelihood_tolerance
    coefficient_tolerance_mask = jnp.max(jnp.abs(coefficient_step)) <= kernel_config.firth_coefficient_tolerance
    score_tolerance_mask = jnp.max(jnp.abs(adjusted_score)) <= kernel_config.firth_gradient_tolerance
    return finite_mask & monotonic_mask & likelihood_tolerance_mask & coefficient_tolerance_mask & score_tolerance_mask


def run_firth_step_halving(
    *,
    current_coefficients: jax.Array,
    current_penalized_log_likelihood: jax.Array,
    coefficient_step: jax.Array,
    evaluate_penalized_log_likelihood: typing.Callable[[jax.Array], jax.Array],
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> FirthBacktrackingResult:
    """Accept the first bounded Firth step that preserves penalized likelihood."""

    def condition_function(state: FirthBacktrackingState) -> jax.Array:
        return (state.attempt_count < FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS) & (~state.accepted)

    def body_function(state: FirthBacktrackingState) -> FirthBacktrackingState:
        candidate_coefficients = current_coefficients + state.next_coefficient_step
        candidate_penalized_log_likelihood = evaluate_penalized_log_likelihood(candidate_coefficients)
        accepted = (
            jnp.isfinite(current_penalized_log_likelihood)
            & jnp.isfinite(candidate_penalized_log_likelihood)
            & jnp.all(jnp.isfinite(candidate_coefficients))
            & jnp.all(jnp.isfinite(state.next_coefficient_step))
            & (
                candidate_penalized_log_likelihood
                >= current_penalized_log_likelihood - kernel_config.firth_likelihood_tolerance
            )
        )
        return FirthBacktrackingState(
            attempt_count=state.attempt_count + jnp.asarray(1, dtype=jnp.int32),
            next_coefficient_step=state.next_coefficient_step * BINARY_CASE_THRESHOLD,
            accepted_coefficient_step=jnp.where(
                accepted,
                state.next_coefficient_step,
                state.accepted_coefficient_step,
            ),
            accepted_coefficients=jnp.where(
                accepted,
                candidate_coefficients,
                state.accepted_coefficients,
            ),
            accepted_penalized_log_likelihood=jnp.where(
                accepted,
                candidate_penalized_log_likelihood,
                state.accepted_penalized_log_likelihood,
            ),
            accepted=accepted,
        )

    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        FirthBacktrackingState(
            attempt_count=jnp.asarray(0, dtype=jnp.int32),
            next_coefficient_step=coefficient_step,
            accepted_coefficient_step=jnp.zeros_like(coefficient_step),
            accepted_coefficients=current_coefficients,
            accepted_penalized_log_likelihood=current_penalized_log_likelihood,
            accepted=jnp.asarray(0, dtype=jnp.bool_),
        ),
    )
    exhausted = ~final_state.accepted
    return FirthBacktrackingResult(
        coefficient_step=final_state.accepted_coefficient_step,
        coefficients=final_state.accepted_coefficients,
        penalized_log_likelihood=final_state.accepted_penalized_log_likelihood,
        accepted=final_state.accepted,
        exhausted=exhausted,
    )


def map_firth_reason_code_to_failure_code(reason_code: jax.Array) -> jax.Array:
    """Map internal Firth termination reasons to public failure labels."""
    return jnp.where(
        reason_code == FirthConvergenceReason.MAX_ITERATIONS.value,
        types.FirthFailureCode.MAX_ITERATIONS.value,
        jnp.where(
            reason_code == FirthConvergenceReason.INVALID_STATISTIC.value,
            types.FirthFailureCode.INVALID_STATISTIC.value,
            jnp.where(
                reason_code == FirthConvergenceReason.NEGATIVE_LRT.value,
                types.FirthFailureCode.INVALID_STATISTIC.value,
                jnp.where(
                    (reason_code == FirthConvergenceReason.STEP_HALVING_EXHAUSTED.value)
                    | (reason_code == FirthConvergenceReason.STEP_SIZE_INCREASE.value),
                    types.FirthFailureCode.STEP_HALVING.value,
                    jnp.where(
                        (reason_code == FirthConvergenceReason.NUMERICAL_FAILURE.value)
                        | (reason_code == FirthConvergenceReason.NULL_FAILURE.value)
                        | (reason_code == FirthConvergenceReason.PROBABILITY_FAILURE.value),
                        types.FirthFailureCode.NUMERICAL.value,
                        types.FirthFailureCode.NONE.value,
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)


def map_scalar_pseudo_firth_failure_to_reason_code(failure_code: jax.Array) -> jax.Array:
    """Map REGENIE scalar pseudo-Firth failure states to internal reason codes."""
    return jnp.where(
        failure_code == jnp.asarray(1, dtype=jnp.int32),
        FirthConvergenceReason.MAX_ITERATIONS.value,
        jnp.where(
            failure_code == jnp.asarray(2, dtype=jnp.int32),
            FirthConvergenceReason.STEP_SIZE_INCREASE.value,
            jnp.where(
                failure_code == jnp.asarray(3, dtype=jnp.int32),
                FirthConvergenceReason.PROBABILITY_FAILURE.value,
                jnp.where(
                    failure_code == jnp.asarray(4, dtype=jnp.int32),
                    FirthConvergenceReason.NEGATIVE_LRT.value,
                    FirthConvergenceReason.NUMERICAL_FAILURE.value,
                ),
            ),
        ),
    ).astype(jnp.int32)


def compute_scalar_firth_components(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    beta: jax.Array,
) -> ScalarFirthComponents:
    """Compute REGENIE scalar approximate-Firth quantities at one beta."""
    linear_predictor = offset_vector + genotype_vector * beta
    probability_vector = compute_regenie_logistic_probability(linear_predictor)
    weight_vector = probability_vector * (1.0 - probability_vector)
    active_weight_vector = jnp.where(active_sample_mask, weight_vector, 0.0)
    genotype_information_diagonal = genotype_vector * genotype_vector * active_weight_vector
    genotype_information = jnp.sum(genotype_information_diagonal)
    penalized_deviance = (
        non_active_deviance
        + compute_logistic_deviance(phenotype_vector, probability_vector, active_sample_mask)
        - jnp.log(genotype_information)
    )
    leverage_vector = genotype_information_diagonal / genotype_information
    adjusted_response = phenotype_vector + leverage_vector * (BINARY_CASE_THRESHOLD - probability_vector)
    score = jnp.sum(jnp.where(active_sample_mask, genotype_vector * (adjusted_response - probability_vector), 0.0))
    valid = (
        jnp.isfinite(genotype_information)
        & (genotype_information > MINIMUM_VARIANCE)
        & jnp.isfinite(penalized_deviance)
        & jnp.isfinite(score)
        & jnp.all(jnp.isfinite(probability_vector))
        & jnp.all(jnp.isfinite(weight_vector))
    )
    return ScalarFirthComponents(
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
) -> ScalarPseudoLogisticState:
    """Run REGENIE's inner pseudo-response scalar logistic update."""

    def condition_function(state: ScalarPseudoLogisticState) -> jax.Array:
        return (state.iteration_count < FIRTH_PSEUDO_INNER_MAXIMUM_ITERATIONS) & (~state.converged) & (~state.failed)

    def body_function(state: ScalarPseudoLogisticState) -> ScalarPseudoLogisticState:
        step_size = state.score / state.genotype_information
        absolute_step_size = jnp.abs(step_size)
        step_increased = absolute_step_size > state.previous_step_size
        step_scale = jnp.maximum(absolute_step_size / FIRTH_MAXIMUM_STEP_SIZE, 1.0)
        updated_beta = state.beta + step_size / step_scale
        probability_vector = compute_regenie_logistic_probability(offset_vector + genotype_vector * updated_beta)
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
        return ScalarPseudoLogisticState(
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
        ScalarPseudoLogisticState(
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
) -> ScalarFirthAttemptResult:
    """Run REGENIE's scalar pseudo-Firth approximate correction."""

    def condition_function(state: ScalarPseudoFirthState) -> jax.Array:
        return (state.outer_iteration_count < maximum_iterations) & (~state.converged) & (~state.failed)

    def body_function(state: ScalarPseudoFirthState) -> ScalarPseudoFirthState:
        components = compute_scalar_firth_components(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            beta=state.beta,
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
        adjusted_response = phenotype_vector + leverage_vector * (BINARY_CASE_THRESHOLD - components.probability_vector)
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
        return ScalarPseudoFirthState(
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
    )
    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        ScalarPseudoFirthState(
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
    return ScalarFirthAttemptResult(
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
) -> ScalarLineSearchState:
    """Run REGENIE scalar NR step-halving against penalized deviance."""

    def condition_function(state: ScalarLineSearchState) -> jax.Array:
        return (state.attempt_count < FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS) & (~state.accepted) & state.valid

    def body_function(state: ScalarLineSearchState) -> ScalarLineSearchState:
        adjusted_step_size = jnp.where(state.attempt_count > 0, state.step_size / 2.0, state.step_size)
        candidate_beta = current_beta + adjusted_step_size
        components = compute_scalar_firth_components(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            beta=candidate_beta,
        )
        accepted = components.valid & (components.penalized_deviance < current_penalized_deviance)
        return ScalarLineSearchState(
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
    )
    return jax.lax.while_loop(
        condition_function,
        body_function,
        ScalarLineSearchState(
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
) -> ScalarFirthAttemptResult:
    """Run REGENIE's scalar Newton-Raphson approximate-Firth fallback."""

    def condition_function(state: ScalarNewtonRaphsonState) -> jax.Array:
        return (state.iteration_count < maximum_iterations) & (~state.converged) & (~state.failed)

    def body_function(state: ScalarNewtonRaphsonState) -> ScalarNewtonRaphsonState:
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
        )
        accepted_step_size = jnp.where(
            line_search_state.accepted,
            line_search_state.step_size,
            line_search_state.step_size + 1.0e-6,
        )
        updated_beta = jnp.where(converged, state.beta, state.beta + accepted_step_size)
        updated_components = compute_scalar_firth_components(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            beta=updated_beta,
        )
        failed = (~state.failed) & ((~updated_components.valid) | (~line_search_state.valid))
        return ScalarNewtonRaphsonState(
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
                FirthConvergenceReason.PROBABILITY_FAILURE.value,
                FirthConvergenceReason.NONE.value,
            ).astype(jnp.int32),
        )

    initial_components = compute_scalar_firth_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=initial_beta,
    )
    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        ScalarNewtonRaphsonState(
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
                FirthConvergenceReason.NONE.value,
                FirthConvergenceReason.PROBABILITY_FAILURE.value,
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
    )
    maximum_iteration_failure = (~final_state.converged) & (~final_state.failed)
    chi_squared = deviance_null - final_components.penalized_deviance
    negative_lrt_failure = final_state.converged & (chi_squared < 0.0)
    reason_code = jnp.where(
        maximum_iteration_failure,
        FirthConvergenceReason.MAX_ITERATIONS.value,
        jnp.where(
            negative_lrt_failure,
            FirthConvergenceReason.NEGATIVE_LRT.value,
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
    return ScalarFirthAttemptResult(
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
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> FirthVariantResult:
    """Fit one REGENIE-equivalent scalar approximate-Firth candidate."""
    scalar_dtype = offset_vector.dtype
    phenotype_vector = jnp.asarray(phenotype_vector, dtype=scalar_dtype)
    genotype_vector = jnp.asarray(genotype_vector, dtype=scalar_dtype)
    warm_start_beta = jnp.asarray(warm_start_beta, dtype=scalar_dtype)
    all_sample_mask = jnp.ones_like(phenotype_vector, dtype=jnp.bool_)
    active_sample_mask = jnp.where(sparse_correction, carrier_sample_mask, all_sample_mask)
    null_probability_vector = compute_regenie_logistic_probability(offset_vector)
    full_null_deviance = compute_logistic_deviance(phenotype_vector, null_probability_vector, all_sample_mask)
    active_null_deviance = compute_logistic_deviance(phenotype_vector, null_probability_vector, active_sample_mask)
    non_active_deviance = jnp.where(sparse_correction, full_null_deviance - active_null_deviance, 0.0)
    null_weight_vector = null_probability_vector * (1.0 - null_probability_vector)
    null_genotype_information = jnp.sum(
        jnp.where(active_sample_mask, genotype_vector * genotype_vector * null_weight_vector, 0.0)
    )
    deviance_null = full_null_deviance - jnp.log(null_genotype_information)
    tolerance = jnp.asarray(kernel_config.firth_gradient_tolerance, dtype=scalar_dtype)
    pseudo_maximum_iterations = min(kernel_config.firth_maximum_iterations // 2, FIRTH_PSEUDO_MAXIMUM_ITERATIONS)
    newton_maximum_iterations = kernel_config.firth_maximum_iterations // 2
    maximum_step_size = jnp.asarray(kernel_config.firth_maximum_step_size, dtype=scalar_dtype)
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
        maximum_iterations=FIRTH_NEWTON_RAPHSON_ZERO_START_ITERATIONS,
        tolerance=tolerance,
        maximum_step_size=maximum_step_size,
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
        FirthConvergenceReason.CONVERGED.value,
        jnp.where(
            use_zero_start,
            FirthConvergenceReason.CONVERGED.value,
            jnp.where(use_warm_start, FirthConvergenceReason.CONVERGED.value, warm_start_result.failure_reason_code),
        ),
    ).astype(jnp.int32)
    valid_mask = (~skip_firth) & (~null_failed) & (pseudo_result.valid | use_zero_start | use_warm_start)
    selected_reason_code = jnp.where(null_failed, FirthConvergenceReason.NULL_FAILURE.value, selected_reason_code)
    failure_code = map_firth_reason_code_to_failure_code(selected_reason_code)
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
    return FirthVariantResult(
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
            FirthConvergenceReason.NONE.value,
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


def compute_full_model_adjusted_weight_components_from_parts(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    probability_vector: jax.Array,
    information_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> AdjustedWeightComponents:
    """Compute full-model Firth weights without materializing a full design matrix."""
    variance_vector = jnp.maximum(probability_vector * (1.0 - probability_vector), MINIMUM_VARIANCE)
    stacked_design_transpose = jnp.concatenate([covariate_matrix.T, genotype_vector[None, :]], axis=0)
    projected_design_transpose = solve_from_positive_definite_matrix(information_matrix, stacked_design_transpose)
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
    return AdjustedWeightComponents(
        leverage_vector=leverage_vector,
        adjusted_weight_vector=adjusted_weight_vector,
        second_weight_vector=second_weight_vector,
    )


def compute_full_model_score_components(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    score_weight_vector: jax.Array,
) -> FullModelScoreComponents:
    """Compute covariate and genotype score blocks without a full design matrix."""
    return FullModelScoreComponents(
        covariate_score=covariate_matrix.T @ score_weight_vector,
        genotype_score=jnp.dot(genotype_vector, score_weight_vector),
    )


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


def compute_null_firth_components(
    *,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    coefficients: jax.Array,
) -> NullFirthComponents:
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
    return NullFirthComponents(
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
) -> NullFirthLineSearchResult:
    """Accept the first null Firth step that decreases penalized deviance."""

    def condition_function(state: NullFirthLineSearchState) -> jax.Array:
        return (state.attempt_count < FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS) & (~state.accepted) & state.valid

    def body_function(state: NullFirthLineSearchState) -> NullFirthLineSearchState:
        candidate_coefficients = current_coefficients + state.next_coefficient_step
        candidate_components = compute_null_firth_components(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            loco_offset=loco_offset,
            coefficients=candidate_coefficients,
        )
        accepted = candidate_components.valid & (candidate_components.deviance < current_deviance)
        return NullFirthLineSearchState(
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
        NullFirthLineSearchState(
            attempt_count=jnp.asarray(0, dtype=jnp.int32),
            next_coefficient_step=coefficient_step,
            accepted_coefficients=current_coefficients,
            accepted_deviance=current_deviance,
            accepted=jnp.asarray(0, dtype=jnp.bool_),
            valid=jnp.asarray(1, dtype=jnp.bool_),
        ),
    )
    return NullFirthLineSearchResult(
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
) -> NullFirthFitResult:
    """Run one REGENIE-style covariate-only null Firth attempt."""
    scalar_dtype = covariate_matrix.dtype
    tolerance_value = jnp.asarray(tolerance, dtype=scalar_dtype)
    maximum_step_size_value = jnp.asarray(maximum_step_size, dtype=scalar_dtype)

    def condition_function(state: NullFirthNewtonRaphsonState) -> jax.Array:
        return (state.iteration_count < maximum_iterations) & (~state.converged) & (~state.failed)

    def body_function(state: NullFirthNewtonRaphsonState) -> NullFirthNewtonRaphsonState:
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
            FirthConvergenceReason.STEP_SIZE_INCREASE.value,
            jnp.where(
                step_halving_failed,
                FirthConvergenceReason.STEP_HALVING_EXHAUSTED.value,
                jnp.where(
                    numerical_failed,
                    FirthConvergenceReason.NUMERICAL_FAILURE.value,
                    jnp.where(converged, FirthConvergenceReason.CONVERGED.value, FirthConvergenceReason.NONE.value),
                ),
            ),
        ).astype(jnp.int32)
        return NullFirthNewtonRaphsonState(
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
        NullFirthNewtonRaphsonState(
            coefficients=initial_coefficients,
            deviance=initial_components.deviance,
            converged=jnp.asarray(0, dtype=jnp.bool_),
            failed=~initial_components.valid,
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            termination_reason_code=jnp.where(
                initial_components.valid,
                FirthConvergenceReason.NONE.value,
                FirthConvergenceReason.NUMERICAL_FAILURE.value,
            ).astype(jnp.int32),
            previous_score_maximum=jnp.asarray(jnp.inf, dtype=scalar_dtype),
            score_increase_count=jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    max_iteration_failure = (~final_state.converged) & (~final_state.failed)
    convergence_reason_code = jnp.where(
        max_iteration_failure,
        FirthConvergenceReason.MAX_ITERATIONS.value,
        final_state.termination_reason_code,
    ).astype(jnp.int32)
    return NullFirthFitResult(
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
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> NullFirthFitResult:
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
    return NullFirthFitResult(
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


def fit_single_variant_firth_logistic_regression(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth: jax.Array,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> FirthVariantResult:
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
        return compute_firth_penalized_log_likelihood_from_cholesky(
            probability_vector=probability_vector,
            phenotype_vector=phenotype_vector,
            information_cholesky_factor=information_cholesky_factor,
        )

    def condition_function(state: FirthState) -> jax.Array:
        return (
            (state.iteration_count < kernel_config.firth_maximum_iterations)
            & (~state.converged)
            & (~state.failed)
            & (~skip_firth)
        )

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
        coefficient_step = solve_from_positive_definite_matrix(second_hessian, adjusted_score)
        current_failed = (
            current_failed | (~jnp.all(jnp.isfinite(adjusted_score))) | (~jnp.all(jnp.isfinite(coefficient_step)))
        )
        maximum_coefficient_step = jnp.max(jnp.abs(coefficient_step))
        step_scale = jnp.minimum(
            1.0, kernel_config.firth_maximum_step_size / jnp.maximum(maximum_coefficient_step, MINIMUM_VARIANCE)
        )
        scaled_coefficient_step = coefficient_step * step_scale
        backtracking_result = run_firth_step_halving(
            current_coefficients=state.coefficients,
            current_penalized_log_likelihood=state.penalized_log_likelihood,
            coefficient_step=scaled_coefficient_step,
            evaluate_penalized_log_likelihood=compute_full_penalized_log_likelihood,
            kernel_config=kernel_config,
        )
        step_halving_failed = (~current_failed) & backtracking_result.exhausted
        updated_failed = current_failed | step_halving_failed
        updated_converged = compute_firth_convergence_mask(
            current_penalized_log_likelihood=state.penalized_log_likelihood,
            candidate_penalized_log_likelihood=backtracking_result.penalized_log_likelihood,
            coefficient_step=backtracking_result.coefficient_step,
            adjusted_score=adjusted_score,
            kernel_config=kernel_config,
        ) & (~updated_failed)
        updated_reason_code = jnp.where(
            step_halving_failed,
            FirthConvergenceReason.STEP_HALVING_EXHAUSTED.value,
            jnp.where(
                current_failed,
                FirthConvergenceReason.NUMERICAL_FAILURE.value,
                jnp.where(updated_converged, FirthConvergenceReason.CONVERGED.value, FirthConvergenceReason.NONE.value),
            ),
        ).astype(jnp.int32)
        return FirthState(
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
    initial_penalized_log_likelihood = compute_firth_penalized_log_likelihood_from_cholesky(
        probability_vector=initial_probability_vector,
        phenotype_vector=phenotype_vector,
        information_cholesky_factor=initial_information_cholesky_factor,
    )
    initial_full_failed = (~jnp.isfinite(initial_penalized_log_likelihood)) | (
        ~jnp.all(jnp.isfinite(initial_coefficients))
    )
    initial_null_failed = (~skip_firth) & (~jnp.isfinite(null_penalized_log_likelihood))
    initial_failed = initial_full_failed | initial_null_failed
    initial_reason_code = jnp.where(
        initial_null_failed,
        FirthConvergenceReason.NULL_FAILURE.value,
        jnp.where(
            initial_full_failed,
            FirthConvergenceReason.NUMERICAL_FAILURE.value,
            FirthConvergenceReason.NONE.value,
        ),
    ).astype(jnp.int32)
    final_state = jax.lax.while_loop(
        condition_function,
        body_function,
        FirthState(
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
    final_penalized_log_likelihood = compute_firth_penalized_log_likelihood_from_cholesky(
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
    genotype_variance = solve_from_positive_definite_matrix(final_second_hessian, unit_genotype_vector)[-1]
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
        FirthConvergenceReason.MAX_ITERATIONS.value,
        jnp.where(
            invalid_statistic_failure_mask,
            FirthConvergenceReason.INVALID_STATISTIC.value,
            final_state.termination_reason_code,
        ),
    ).astype(jnp.int32)
    failure_code = map_firth_reason_code_to_failure_code(convergence_reason_code)
    return FirthVariantResult(
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
            FirthConvergenceReason.NONE.value,
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


def residualize_and_scale_genotypes_for_approximate_firth(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
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


def build_regenie_flipped_genotypes(
    genotype_matrix_by_variant: jax.Array,
) -> RegenieGenotypeFlipResult:
    """Code variant-major genotypes the way REGENIE does before testing."""
    return genotype.build_regenie_flipped_genotypes(genotype_matrix_by_variant)


def compute_firth_variantwise(
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> FirthVariantResult:
    """Compute device-side Firth fits for a padded set of candidate lanes."""
    scalar_offset_vector = jnp.asarray(null_firth_offset, dtype=jnp.float64)
    scalar_phenotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float64)

    def fit_variant(
        genotype_vector: jax.Array,
        raw_genotype_vector: jax.Array,
        variant_initial_coefficients: jax.Array,
        skip_firth: jax.Array,
        sparse_correction: jax.Array,
    ) -> FirthVariantResult:
        if not kernel_config.use_block_firth_math:
            return fit_single_variant_regenie_approximate_firth(
                phenotype_vector=scalar_phenotype_vector,
                genotype_vector=jnp.asarray(genotype_vector, dtype=jnp.float64),
                offset_vector=scalar_offset_vector,
                carrier_sample_mask=raw_genotype_vector > SPARSE_CARRIER_DOSAGE_THRESHOLD,
                sparse_correction=sparse_correction,
                warm_start_beta=jnp.asarray(0.0, dtype=jnp.float64),
                skip_firth=skip_firth,
                null_failed=~jnp.isfinite(null_penalized_log_likelihood),
                kernel_config=kernel_config,
            )
        return fit_single_variant_firth_logistic_regression(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            loco_offset=loco_offset,
            initial_coefficients=variant_initial_coefficients,
            skip_firth=skip_firth,
            null_penalized_log_likelihood=null_penalized_log_likelihood,
            kernel_config=kernel_config,
        )

    return jax.vmap(fit_variant, in_axes=(0, 0, 0, 0, 0))(
        genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant,
        initial_coefficients,
        skip_firth_mask,
        sparse_correction_mask,
    )


def build_empty_firth_variant_result(
    batch_size: int,
) -> FirthVariantResult:
    """Build a placeholder Firth result for skipped padded batches."""
    return FirthVariantResult(
        beta=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        standard_error=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        chi_squared=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        log10_p_value=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        penalized_log_likelihood=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        converged_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
        valid_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
        iteration_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        failure_code=jnp.zeros((batch_size,), dtype=jnp.int32),
        convergence_reason_code=jnp.zeros((batch_size,), dtype=jnp.int32),
        correction_code=jnp.zeros((batch_size,), dtype=jnp.int32),
        sparse_correction_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def apply_device_candidate_corrections_firth(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Apply fully device-resident Firth corrections to score-test candidates."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)

    def no_candidate_corrections() -> regenie2_binary_types.Regenie2BinaryChunkResult:
        return result

    def apply_candidate_corrections() -> regenie2_binary_types.Regenie2BinaryChunkResult:
        firth_batch_size = kernel_config.firth_batch_size
        kernel_candidate_capacity = kernel_config.firth_candidate_capacity
        genotype_matrix_float32 = jnp.asarray(genotype_matrix, dtype=jnp.float32)
        variant_count = genotype_matrix_float32.shape[1]

        def apply_candidate_corrections_with_capacity(
            candidate_capacity: int,
        ) -> regenie2_binary_types.Regenie2BinaryChunkResult:
            batch_plan = regenie2_binary_candidate_planning.build_device_firth_batch_plan(
                candidate_mask, candidate_capacity, firth_batch_size
            )
            flat_fallback_indices = batch_plan.fallback_index_matrix.reshape((-1,))
            flat_active_mask = batch_plan.fallback_active_mask_matrix.reshape((-1,))
            raw_genotype_matrix_by_variant = jnp.take(genotype_matrix_float32, flat_fallback_indices, axis=1).T
            genotype_flip_result = build_regenie_flipped_genotypes(raw_genotype_matrix_by_variant)
            firth_raw_genotype_matrix_by_variant = (
                raw_genotype_matrix_by_variant
                if kernel_config.use_block_firth_math
                else genotype_flip_result.genotype_matrix_by_variant
            )
            if kernel_config.use_block_firth_math:
                flat_genotype_flip_mask = jnp.zeros_like(flat_active_mask)
            else:
                flat_genotype_flip_mask = genotype_flip_result.flip_mask
            genotype_matrix_by_variant = jnp.where(
                kernel_config.use_block_firth_math,
                firth_raw_genotype_matrix_by_variant,
                residualize_and_scale_genotypes_for_approximate_firth(
                    chromosome_state,
                    firth_raw_genotype_matrix_by_variant,
                ),
            )
            if sparse_candidate_mask is None:
                flat_sparse_candidate_mask = jnp.zeros_like(flat_active_mask)
            else:
                flat_sparse_candidate_mask = (
                    jnp.take(jnp.asarray(sparse_candidate_mask, dtype=jnp.bool_), flat_fallback_indices, axis=0)
                    & flat_active_mask
                )
            heuristic_firth_mask = (
                compute_firth_pre_dispatch_mask_without_mask(
                    genotype_matrix_by_variant=firth_raw_genotype_matrix_by_variant,
                    phenotype_vector=chromosome_state.phenotype_vector,
                )
                | flat_sparse_candidate_mask
            ) & flat_active_mask
            ordered_candidate_inputs = regenie2_binary_candidate_planning.group_firth_candidate_batch_inputs(
                flat_fallback_indices=flat_fallback_indices,
                flat_active_mask=flat_active_mask,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                heuristic_firth_mask=heuristic_firth_mask,
            )
            flat_fallback_indices = ordered_candidate_inputs.flat_fallback_indices
            flat_active_mask = ordered_candidate_inputs.flat_active_mask
            genotype_matrix_by_variant = ordered_candidate_inputs.genotype_matrix_by_variant
            heuristic_firth_mask = ordered_candidate_inputs.heuristic_firth_mask
            raw_genotype_matrix_by_variant = jnp.take(genotype_matrix_float32, flat_fallback_indices, axis=1).T
            genotype_flip_result = build_regenie_flipped_genotypes(raw_genotype_matrix_by_variant)
            firth_raw_genotype_matrix_by_variant = (
                raw_genotype_matrix_by_variant
                if kernel_config.use_block_firth_math
                else genotype_flip_result.genotype_matrix_by_variant
            )
            if kernel_config.use_block_firth_math:
                flat_genotype_flip_mask = jnp.zeros_like(flat_active_mask)
            else:
                flat_genotype_flip_mask = genotype_flip_result.flip_mask
            flat_sparse_candidate_mask = (
                jnp.take(jnp.asarray(sparse_candidate_mask, dtype=jnp.bool_), flat_fallback_indices, axis=0)
                & flat_active_mask
                if sparse_candidate_mask is not None
                else jnp.zeros_like(flat_active_mask)
            )
            standard_initial_coefficients = jnp.broadcast_to(
                chromosome_state.null_logistic_coefficients[None, :],
                (genotype_matrix_by_variant.shape[0], chromosome_state.null_logistic_coefficients.shape[0]),
            )
            standard_initial_beta = (
                jnp.take(result.beta, flat_fallback_indices, axis=0)
                if kernel_config.use_block_firth_math
                else jnp.zeros_like(jnp.take(result.beta, flat_fallback_indices, axis=0))
            )
            standard_initial_coefficients = jnp.concatenate(
                [
                    standard_initial_coefficients,
                    standard_initial_beta[:, None],
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
            if not kernel_config.use_block_firth_math:
                initial_coefficients = standard_initial_coefficients
            batch_count = batch_plan.fallback_index_matrix.shape[0]
            active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
            genotype_batches = genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
            raw_genotype_batches = firth_raw_genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
            initial_coefficient_batches = initial_coefficients.reshape((batch_count, firth_batch_size, -1))
            active_mask_batches = flat_active_mask.reshape((batch_count, firth_batch_size))
            sparse_correction_mask_batches = flat_sparse_candidate_mask.reshape((batch_count, firth_batch_size))
            empty_firth_variant_result = build_empty_firth_variant_result(firth_batch_size)

            def compute_firth_batch(
                carry: None,
                batch_index: jax.Array,
            ) -> tuple[None, FirthVariantResult]:
                del carry

                def run_active_batch(_: None) -> FirthVariantResult:
                    return compute_firth_variantwise(
                        covariate_matrix=chromosome_state.covariate_matrix,
                        null_logistic_coefficients=chromosome_state.null_logistic_coefficients,
                        null_firth_offset=chromosome_state.null_firth_offset,
                        phenotype_vector=chromosome_state.phenotype_vector,
                        genotype_matrix_by_variant=genotype_batches[batch_index],
                        raw_genotype_matrix_by_variant=raw_genotype_batches[batch_index],
                        loco_offset=chromosome_state.loco_offset,
                        initial_coefficients=initial_coefficient_batches[batch_index],
                        skip_firth_mask=~active_mask_batches[batch_index],
                        sparse_correction_mask=sparse_correction_mask_batches[batch_index],
                        null_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood,
                        kernel_config=kernel_config,
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
                failure_code=batched_firth_result.failure_code.reshape((-1,)),
                convergence_reason_code=batched_firth_result.convergence_reason_code.reshape((-1,)),
                correction_code=batched_firth_result.correction_code.reshape((-1,)),
                sparse_correction_mask=batched_firth_result.sparse_correction_mask.reshape((-1,)),
                pseudo_firth_iteration_count=batched_firth_result.pseudo_firth_iteration_count.reshape((-1,)),
                nr_zero_start_iteration_count=batched_firth_result.nr_zero_start_iteration_count.reshape((-1,)),
                nr_warm_start_iteration_count=batched_firth_result.nr_warm_start_iteration_count.reshape((-1,)),
            )
            active_flat_positions = batch_plan.active_flat_position_vector
            active_fallback_indices = flat_fallback_indices[active_flat_positions]
            active_valid_mask = firth_result.valid_mask[active_flat_positions]
            active_firth_beta = jnp.where(
                flat_genotype_flip_mask[active_flat_positions],
                -firth_result.beta[active_flat_positions],
                firth_result.beta[active_flat_positions],
            )
            active_firth_chi_squared = firth_result.chi_squared[active_flat_positions]
            active_firth_standard_error = firth_result.standard_error[active_flat_positions]
            invalid_firth_statistic = jnp.full_like(active_firth_beta, jnp.nan)
            if correction_plan.firth_se:
                active_firth_standard_error = jnp.where(
                    active_firth_chi_squared > 0.0,
                    jnp.abs(active_firth_beta) / jnp.sqrt(active_firth_chi_squared),
                    active_firth_standard_error,
                )
            merged_beta = jnp.where(active_valid_mask, active_firth_beta, invalid_firth_statistic)
            merged_standard_error = jnp.where(
                active_valid_mask,
                active_firth_standard_error,
                invalid_firth_statistic,
            )
            merged_chi_squared = jnp.where(
                active_valid_mask,
                firth_result.chi_squared[active_flat_positions],
                invalid_firth_statistic,
            )
            merged_log10_p_value = jnp.where(
                active_valid_mask,
                firth_result.log10_p_value[active_flat_positions],
                invalid_firth_statistic,
            )
            merged_extra_code = jnp.where(
                active_valid_mask, types.BinaryExtraCode.FIRTH.value, types.BinaryExtraCode.TEST_FAIL.value
            ).astype(jnp.int32)
            return regenie2_binary_types.Regenie2BinaryChunkResult(
                beta=result.beta.at[active_fallback_indices].set(jnp.asarray(merged_beta, dtype=result.beta.dtype)),
                standard_error=result.standard_error.at[active_fallback_indices].set(
                    jnp.asarray(merged_standard_error, dtype=result.standard_error.dtype)
                ),
                chi_squared=result.chi_squared.at[active_fallback_indices].set(
                    jnp.asarray(merged_chi_squared, dtype=result.chi_squared.dtype)
                ),
                log10_p_value=result.log10_p_value.at[active_fallback_indices].set(
                    jnp.asarray(merged_log10_p_value, dtype=result.log10_p_value.dtype)
                ),
                extra_code=result.extra_code.at[active_fallback_indices].set(merged_extra_code),
                valid_mask=result.valid_mask.at[active_fallback_indices].set(active_valid_mask),
                firth_iteration_count=result.firth_iteration_count.at[active_fallback_indices].set(
                    firth_result.iteration_count[active_flat_positions]
                ),
                firth_failure_code=result.firth_failure_code.at[active_fallback_indices].set(
                    firth_result.failure_code[active_flat_positions]
                ),
                firth_convergence_reason_code=result.firth_convergence_reason_code.at[active_fallback_indices].set(
                    firth_result.convergence_reason_code[active_flat_positions]
                ),
                firth_correction_code=result.firth_correction_code.at[active_fallback_indices].set(
                    firth_result.correction_code[active_flat_positions]
                ),
                firth_sparse_correction_mask=result.firth_sparse_correction_mask.at[active_fallback_indices].set(
                    firth_result.sparse_correction_mask[active_flat_positions]
                ),
                pseudo_firth_iteration_count=result.pseudo_firth_iteration_count.at[active_fallback_indices].set(
                    firth_result.pseudo_firth_iteration_count[active_flat_positions]
                ),
                nr_zero_start_iteration_count=result.nr_zero_start_iteration_count.at[active_fallback_indices].set(
                    firth_result.nr_zero_start_iteration_count[active_flat_positions]
                ),
                nr_warm_start_iteration_count=result.nr_warm_start_iteration_count.at[active_fallback_indices].set(
                    firth_result.nr_warm_start_iteration_count[active_flat_positions]
                ),
            )

        bounded_candidate_capacity = min(kernel_candidate_capacity, variant_count)
        return jax.lax.cond(
            fallback_count <= bounded_candidate_capacity,
            lambda _: apply_candidate_corrections_with_capacity(bounded_candidate_capacity),
            lambda _: apply_candidate_corrections_with_capacity(variant_count),
            operand=None,
        )

    return jax.lax.cond(fallback_count > 0, apply_candidate_corrections, no_candidate_corrections)


def apply_device_candidate_corrections(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Apply binary candidate corrections without leaving device memory."""
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return result
    if correction_plan.method == types.BinaryFallbackMethod.FIRTH:
        message = "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx."
        raise NotImplementedError(message)
    if correction_plan.method == types.BinaryFallbackMethod.SPA:
        message = "SPA fallback is not implemented yet. Omit --spa for score-test-only output."
        raise NotImplementedError(message)
    return apply_device_candidate_corrections_firth(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def compute_regenie2_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association using cached null state."""
    score_test_result = compute_regenie2_binary_score_test_chunk(
        chromosome_state,
        genotype_matrix,
        correction_plan,
    )
    return apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=score_test_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )


def compute_regenie2_binary_chunk(
    state: regenie2_binary_types.Regenie2BinaryState,
    genotype_matrix: jax.Array,
    loco_offset: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association for a genotype chunk."""
    chromosome_state = prepare_regenie2_binary_chromosome_state(state, loco_offset, correction_plan, kernel_config)
    compute_regenie2_binary_chunk_from_state = typing.cast(
        "BinaryChunkComputeFunction",
        compute_regenie2_binary_chunk_from_chromosome_state,
    )
    return compute_regenie2_binary_chunk_from_state(
        chromosome_state,
        genotype_matrix,
        correction_plan,
        sparse_candidate_mask,
        kernel_config,
    )


from g.compute import regenie2_binary_variant_major  # noqa: E402
