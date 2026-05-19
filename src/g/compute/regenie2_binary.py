"""REGENIE step 2 binary score-test kernel with device-resident Firth fallback."""

from __future__ import annotations

import functools
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import g.compute.regenie2_binary_candidate_planning as regenie2_binary_candidate_planning
import g.compute.regenie2_binary_diagnostics as regenie2_binary_diagnostics
from g import types
from g.compute import regenie2_binary_types as regenie2_types
from g.compute import regenie2_linear

MINIMUM_PROBABILITY = 1.0e-6
MINIMUM_VARIANCE = 1.0e-8
DEFAULT_MAXIMUM_NULL_ITERATIONS = 50
NULL_LOGISTIC_COEFFICIENT_TOLERANCE = 1.0e-6
EXTRA_CODE_SCORE = regenie2_binary_diagnostics.EXTRA_CODE_SCORE
EXTRA_CODE_FIRTH = regenie2_binary_diagnostics.EXTRA_CODE_FIRTH
EXTRA_CODE_SPA = regenie2_binary_diagnostics.EXTRA_CODE_SPA
EXTRA_CODE_TEST_FAIL = regenie2_binary_diagnostics.EXTRA_CODE_TEST_FAIL
FIRTH_FAILURE_NONE = regenie2_binary_diagnostics.FIRTH_FAILURE_NONE
FIRTH_FAILURE_NUMERICAL = regenie2_binary_diagnostics.FIRTH_FAILURE_NUMERICAL
FIRTH_FAILURE_MAX_ITERATIONS = regenie2_binary_diagnostics.FIRTH_FAILURE_MAX_ITERATIONS
FIRTH_FAILURE_INVALID_STATISTIC = regenie2_binary_diagnostics.FIRTH_FAILURE_INVALID_STATISTIC
INITIAL_RESPONSE_SCALE = 4.863891244002886
BINARY_CASE_THRESHOLD = 0.5
ALLELE_COUNT_MULTIPLIER = 2.0
FIRTH_GRADIENT_TOLERANCE = 1.0e-4
FIRTH_COEFFICIENT_TOLERANCE = 1.0e-4
FIRTH_LIKELIHOOD_TOLERANCE = 1.0e-4
FIRTH_MAXIMUM_STEP_SIZE = 5.0
FIRTH_MAXIMUM_ITERATIONS = 50
DEFAULT_FIRTH_BATCH_SIZE = regenie2_binary_candidate_planning.DEFAULT_FIRTH_BATCH_SIZE
DEFAULT_FIRTH_CANDIDATE_CAPACITY = regenie2_binary_candidate_planning.DEFAULT_FIRTH_CANDIDATE_CAPACITY
configured_maximum_null_iterations = DEFAULT_MAXIMUM_NULL_ITERATIONS
configured_null_logistic_coefficient_tolerance = NULL_LOGISTIC_COEFFICIENT_TOLERANCE
configured_firth_gradient_tolerance = FIRTH_GRADIENT_TOLERANCE
configured_firth_coefficient_tolerance = FIRTH_COEFFICIENT_TOLERANCE
configured_firth_likelihood_tolerance = FIRTH_LIKELIHOOD_TOLERANCE
configured_firth_maximum_step_size = FIRTH_MAXIMUM_STEP_SIZE
configured_firth_maximum_iterations = FIRTH_MAXIMUM_ITERATIONS
configured_use_block_firth_math = False

BinaryScoreTestChunkComputeFunction = typing.Callable[
    [regenie2_types.Regenie2BinaryChromosomeState, jax.Array, types.BinaryCorrectionPlan],
    regenie2_types.Regenie2BinaryChunkResult,
]
BinaryChunkComputeFunction = typing.Callable[
    [regenie2_types.Regenie2BinaryChromosomeState, jax.Array, types.BinaryCorrectionPlan, jax.Array | None],
    regenie2_types.Regenie2BinaryChunkResult,
]
BinaryVariantMajorChunkComputeFunction = typing.Callable[
    [regenie2_types.Regenie2BinaryChromosomeState, jax.Array, types.BinaryCorrectionPlan, jax.Array | None],
    regenie2_types.Regenie2BinaryChunkResult,
]


get_firth_batch_size = regenie2_binary_candidate_planning.get_firth_batch_size
get_firth_candidate_capacity = regenie2_binary_candidate_planning.get_firth_candidate_capacity


def configure_binary_runtime(
    *,
    maximum_null_iterations: int,
    null_logistic_coefficient_tolerance: float,
    firth_maximum_iterations: int,
    firth_gradient_tolerance: float,
    firth_coefficient_tolerance: float,
    firth_likelihood_tolerance: float,
    firth_maximum_step_size: float,
    use_block_firth_math: bool,
) -> None:
    """Configure binary solver settings used when kernels are traced."""
    global configured_maximum_null_iterations, configured_null_logistic_coefficient_tolerance
    global configured_firth_gradient_tolerance, configured_firth_coefficient_tolerance
    global configured_firth_likelihood_tolerance, configured_firth_maximum_step_size
    global configured_firth_maximum_iterations, configured_use_block_firth_math
    configured_maximum_null_iterations = maximum_null_iterations
    configured_null_logistic_coefficient_tolerance = null_logistic_coefficient_tolerance
    configured_firth_maximum_iterations = firth_maximum_iterations
    configured_firth_gradient_tolerance = firth_gradient_tolerance
    configured_firth_coefficient_tolerance = firth_coefficient_tolerance
    configured_firth_likelihood_tolerance = firth_likelihood_tolerance
    configured_firth_maximum_step_size = firth_maximum_step_size
    configured_use_block_firth_math = use_block_firth_math
    get_maximum_null_iterations.cache_clear()
    get_null_logistic_coefficient_tolerance.cache_clear()
    get_firth_maximum_iterations.cache_clear()
    get_firth_gradient_tolerance.cache_clear()
    get_firth_coefficient_tolerance.cache_clear()
    get_firth_likelihood_tolerance.cache_clear()
    get_firth_maximum_step_size.cache_clear()
    get_use_block_firth_math.cache_clear()


@functools.cache
def get_maximum_null_iterations() -> int:
    """Return the configured null logistic iteration cap."""
    return configured_maximum_null_iterations


@functools.cache
def get_null_logistic_coefficient_tolerance() -> float:
    """Return the configured null logistic coefficient tolerance."""
    return configured_null_logistic_coefficient_tolerance


@functools.cache
def get_firth_gradient_tolerance() -> float:
    """Return the configured Firth gradient tolerance."""
    return configured_firth_gradient_tolerance


@functools.cache
def get_firth_coefficient_tolerance() -> float:
    """Return the configured Firth coefficient tolerance."""
    return configured_firth_coefficient_tolerance


@functools.cache
def get_firth_likelihood_tolerance() -> float:
    """Return the configured Firth likelihood tolerance."""
    return configured_firth_likelihood_tolerance


@functools.cache
def get_firth_maximum_step_size() -> float:
    """Return the configured Firth maximum step size."""
    return configured_firth_maximum_step_size


@functools.cache
def get_firth_maximum_iterations() -> int:
    """Return the configured Firth iteration cap."""
    return configured_firth_maximum_iterations


@functools.cache
def get_use_block_firth_math() -> bool:
    """Resolve whether experimental block Firth math is enabled."""
    return configured_use_block_firth_math


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


FirthBatchPlan = regenie2_binary_candidate_planning.FirthBatchPlan
FirthCandidateBatchInputs = regenie2_binary_candidate_planning.FirthCandidateBatchInputs


def prepare_regenie2_binary_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> regenie2_types.Regenie2BinaryState:
    """Prepare reusable binary step 2 state.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.

    Returns:
        Reusable binary step 2 state.

    """
    covariate_matrix_float32 = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_vector_float32 = jnp.asarray(phenotype_vector, dtype=jnp.float32)
    return regenie2_types.Regenie2BinaryState(
        covariate_matrix=covariate_matrix_float32,
        phenotype_vector=phenotype_vector_float32,
        sample_count=jnp.asarray(covariate_matrix_float32.shape[0], dtype=jnp.int32),
    )


def prepare_regenie2_multi_binary_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
) -> regenie2_types.Regenie2MultiBinaryState:
    """Prepare reusable multi-trait binary step 2 state.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_matrix: Binary phenotype matrix in trait-major 0/1 encoding.

    Returns:
        Reusable multi-trait binary step 2 state.

    """
    covariate_matrix_float32 = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_matrix_float32 = jnp.asarray(phenotype_matrix, dtype=jnp.float32)
    return regenie2_types.Regenie2MultiBinaryState(
        covariate_matrix=covariate_matrix_float32,
        phenotype_matrix=phenotype_matrix_float32,
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


build_extra_code = regenie2_binary_candidate_planning.build_extra_code


@jax.jit
def fit_null_logistic_coefficients(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    maximum_iterations: int | None = None,
) -> NullLogisticFitState:
    """Fit a covariate-only logistic null model with a fixed LOCO offset."""
    covariate_count = covariate_matrix.shape[1]
    resolved_maximum_iterations = get_maximum_null_iterations() if maximum_iterations is None else maximum_iterations
    coefficient_tolerance = get_null_logistic_coefficient_tolerance()

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
        coefficient_delta = regenie2_linear.solve_positive_definite_system(cholesky_factor, score_vector)
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


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def prepare_regenie2_binary_chromosome_state(
    state: regenie2_types.Regenie2BinaryState,
    loco_offset: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
) -> regenie2_types.Regenie2BinaryChromosomeState:
    """Prepare chromosome-specific null logistic state reused across chunks."""
    loco_offset_float32 = jnp.asarray(loco_offset, dtype=jnp.float32)
    null_logistic_fit_state = fit_null_logistic_coefficients(
        state.covariate_matrix,
        state.phenotype_vector,
        loco_offset_float32,
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
        null_firth_penalized_log_likelihood = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        null_firth_penalized_log_likelihood = fit_covariate_only_firth_null_model(
            covariate_matrix=state.covariate_matrix,
            phenotype_vector=state.phenotype_vector,
            loco_offset=loco_offset_float32,
            initial_coefficients=null_logistic_coefficients,
        )
    return regenie2_types.Regenie2BinaryChromosomeState(
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
        null_logistic_iteration_count=null_logistic_fit_state.iteration_count,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def prepare_regenie2_multi_binary_chromosome_state(
    state: regenie2_types.Regenie2MultiBinaryState,
    loco_offset_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
) -> regenie2_types.Regenie2MultiBinaryChromosomeState:
    """Prepare chromosome-specific null logistic state for all requested binary traits."""
    loco_offset_matrix_float32 = jnp.asarray(loco_offset_matrix, dtype=jnp.float32)

    def prepare_one_trait(
        phenotype_vector: jax.Array,
        loco_offset: jax.Array,
    ) -> regenie2_types.Regenie2BinaryChromosomeState:
        trait_state = regenie2_types.Regenie2BinaryState(
            covariate_matrix=state.covariate_matrix,
            phenotype_vector=phenotype_vector,
            sample_count=state.sample_count,
        )
        return prepare_regenie2_binary_chromosome_state(trait_state, loco_offset, correction_plan)

    chromosome_states = jax.vmap(prepare_one_trait)(state.phenotype_matrix, loco_offset_matrix_float32)
    return regenie2_types.Regenie2MultiBinaryChromosomeState(
        covariate_matrix=state.covariate_matrix,
        phenotype_matrix=state.phenotype_matrix,
        null_logistic_coefficients=chromosome_states.null_logistic_coefficients,
        fitted_probability=chromosome_states.fitted_probability,
        score_residual=chromosome_states.score_residual,
        loco_offset_matrix=chromosome_states.loco_offset,
        standardized_residual=chromosome_states.standardized_residual,
        square_root_weight=chromosome_states.square_root_weight,
        weighted_genotype_projection_matrix=chromosome_states.weighted_genotype_projection_matrix,
        null_firth_penalized_log_likelihood=chromosome_states.null_firth_penalized_log_likelihood,
        null_logistic_iteration_count=chromosome_states.null_logistic_iteration_count,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def compute_regenie2_binary_score_test_chunk_from_chromosome_state(
    chromosome_state: regenie2_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
) -> regenie2_types.Regenie2BinaryChunkResult:
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
    extra_code = build_extra_code(log10_p_value, valid_mask, correction_plan)
    return regenie2_types.Regenie2BinaryChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        valid_mask=valid_mask,
        firth_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_failure_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
    )


compute_regenie2_binary_score_test_chunk = typing.cast(
    "BinaryScoreTestChunkComputeFunction",
    compute_regenie2_binary_score_test_chunk_from_chromosome_state,
)


def build_single_binary_chromosome_state_from_multi(
    chromosome_state: regenie2_types.Regenie2MultiBinaryChromosomeState,
    trait_index: jax.Array,
) -> regenie2_types.Regenie2BinaryChromosomeState:
    """Build a single-trait chromosome state view from a multi-trait state."""
    return regenie2_types.Regenie2BinaryChromosomeState(
        covariate_matrix=chromosome_state.covariate_matrix,
        phenotype_vector=chromosome_state.phenotype_matrix[trait_index],
        null_logistic_coefficients=chromosome_state.null_logistic_coefficients[trait_index],
        fitted_probability=chromosome_state.fitted_probability[trait_index],
        score_residual=chromosome_state.score_residual[trait_index],
        loco_offset=chromosome_state.loco_offset_matrix[trait_index],
        standardized_residual=chromosome_state.standardized_residual[trait_index],
        square_root_weight=chromosome_state.square_root_weight[trait_index],
        weighted_genotype_projection_matrix=chromosome_state.weighted_genotype_projection_matrix[trait_index],
        null_firth_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood[trait_index],
        null_logistic_iteration_count=chromosome_state.null_logistic_iteration_count[trait_index],
    )


def build_multi_binary_chunk_result(
    result: regenie2_types.Regenie2BinaryChunkResult,
) -> regenie2_types.Regenie2MultiBinaryChunkResult:
    """Rewrap a vmapped single-trait binary result as a multi-trait result."""
    return regenie2_types.Regenie2MultiBinaryChunkResult(
        beta=result.beta,
        standard_error=result.standard_error,
        chi_squared=result.chi_squared,
        log10_p_value=result.log10_p_value,
        extra_code=result.extra_code,
        valid_mask=result.valid_mask,
        firth_iteration_count=result.firth_iteration_count,
        firth_failure_code=result.firth_failure_code,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def compute_regenie2_multi_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_types.Regenie2MultiBinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
) -> regenie2_types.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary REGENIE step 2 association using one genotype chunk."""

    def compute_one_trait(trait_index: jax.Array) -> regenie2_types.Regenie2BinaryChunkResult:
        single_chromosome_state = build_single_binary_chromosome_state_from_multi(chromosome_state, trait_index)
        return compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=single_chromosome_state,
            genotype_matrix=genotype_matrix,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
        )

    trait_count = chromosome_state.phenotype_matrix.shape[0]
    return build_multi_binary_chunk_result(jax.vmap(compute_one_trait)(jnp.arange(trait_count, dtype=jnp.int32)))


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_types.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
) -> regenie2_types.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary association from variant-major genotypes."""

    def compute_one_trait(trait_index: jax.Array) -> regenie2_types.Regenie2BinaryChunkResult:
        single_chromosome_state = build_single_binary_chromosome_state_from_multi(chromosome_state, trait_index)
        return compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
            chromosome_state=single_chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
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


def fit_covariate_only_firth_null_model(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
) -> jax.Array:
    """Fit the covariate-only Firth null model and return its penalized log-likelihood."""

    def condition_function(state: FirthState) -> jax.Array:
        return (state.iteration_count < get_firth_maximum_iterations()) & (~state.converged) & (~state.failed)

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
        step_scale = jnp.minimum(
            1.0, get_firth_maximum_step_size() / jnp.maximum(maximum_coefficient_step, MINIMUM_VARIANCE)
        )
        scaled_coefficient_step = coefficient_step * step_scale
        updated_converged = (
            (state.iteration_count > 0)
            & (jnp.max(jnp.abs(scaled_coefficient_step)) <= get_firth_coefficient_tolerance())
            & (jnp.max(jnp.abs(adjusted_score)) <= get_firth_gradient_tolerance())
            & (
                (current_penalized_log_likelihood - state.previous_penalized_log_likelihood)
                < get_firth_likelihood_tolerance()
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
    use_block_firth_math = get_use_block_firth_math()
    if use_block_firth_math:
        coefficient_count = covariate_matrix.shape[1] + 1
    else:
        full_design_matrix = jnp.concatenate([covariate_matrix, genotype_vector[:, None]], axis=1)
        coefficient_count = full_design_matrix.shape[1]
    unit_genotype_vector = jnp.zeros((coefficient_count,), dtype=jnp.float32).at[-1].set(1.0)

    def compute_probability_vector(coefficients: jax.Array) -> jax.Array:
        linear_predictor = covariate_matrix @ coefficients[:-1] + genotype_vector * coefficients[-1] + loco_offset
        return compute_logistic_probability(linear_predictor)

    def condition_function(state: FirthState) -> jax.Array:
        return (
            (state.iteration_count < get_firth_maximum_iterations())
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
        maximum_coefficient_step = jnp.max(jnp.abs(coefficient_step))
        step_scale = jnp.minimum(
            1.0, get_firth_maximum_step_size() / jnp.maximum(maximum_coefficient_step, MINIMUM_VARIANCE)
        )
        scaled_coefficient_step = coefficient_step * step_scale
        updated_converged = (
            (state.iteration_count > 0)
            & (jnp.max(jnp.abs(scaled_coefficient_step)) <= get_firth_coefficient_tolerance())
            & (jnp.max(jnp.abs(adjusted_score)) <= get_firth_gradient_tolerance())
            & (
                (current_penalized_log_likelihood - state.previous_penalized_log_likelihood)
                < get_firth_likelihood_tolerance()
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
    numerical_failure_mask = (~skip_firth) & final_state.failed
    maximum_iteration_failure_mask = (
        (~skip_firth)
        & (~final_state.converged)
        & (~final_state.failed)
        & (final_state.iteration_count >= get_firth_maximum_iterations())
    )
    invalid_statistic_failure_mask = (~skip_firth) & final_state.converged & (~final_state.failed) & (~valid_mask)
    failure_code = jnp.where(
        numerical_failure_mask,
        FIRTH_FAILURE_NUMERICAL,
        jnp.where(
            maximum_iteration_failure_mask,
            FIRTH_FAILURE_MAX_ITERATIONS,
            jnp.where(invalid_statistic_failure_mask, FIRTH_FAILURE_INVALID_STATISTIC, FIRTH_FAILURE_NONE),
        ),
    ).astype(jnp.int32)
    return FirthVariantResult(
        beta=jnp.where(skip_firth, jnp.nan, beta),
        standard_error=jnp.where(skip_firth, jnp.nan, standard_error),
        chi_squared=jnp.where(skip_firth, jnp.nan, chi_squared),
        log10_p_value=jnp.where(skip_firth, jnp.nan, log10_p_value),
        penalized_log_likelihood=jnp.where(skip_firth, jnp.nan, final_penalized_log_likelihood),
        converged_mask=jnp.where(skip_firth, jnp.asarray(0, dtype=jnp.bool_), final_state.converged),
        valid_mask=valid_mask,
        iteration_count=jnp.where(skip_firth, jnp.asarray(0, dtype=jnp.int32), final_state.iteration_count),
        failure_code=jnp.where(skip_firth, FIRTH_FAILURE_NONE, failure_code),
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


build_device_firth_batch_plan = regenie2_binary_candidate_planning.build_device_firth_batch_plan
group_firth_candidate_batch_inputs = regenie2_binary_candidate_planning.group_firth_candidate_batch_inputs


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
        failure_code=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def apply_device_candidate_corrections_firth(
    chromosome_state: regenie2_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    result: regenie2_types.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
) -> regenie2_types.Regenie2BinaryChunkResult:
    """Apply fully device-resident Firth corrections to score-test candidates."""
    candidate_mask = result.extra_code == EXTRA_CODE_FIRTH
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)

    def no_candidate_corrections() -> regenie2_types.Regenie2BinaryChunkResult:
        return result

    def apply_candidate_corrections() -> regenie2_types.Regenie2BinaryChunkResult:
        firth_batch_size = get_firth_batch_size()
        configured_candidate_capacity = get_firth_candidate_capacity()
        genotype_matrix_float32 = jnp.asarray(genotype_matrix, dtype=jnp.float32)
        variant_count = genotype_matrix_float32.shape[1]

        def apply_candidate_corrections_with_capacity(
            candidate_capacity: int,
        ) -> regenie2_types.Regenie2BinaryChunkResult:
            batch_plan = build_device_firth_batch_plan(candidate_mask, candidate_capacity)
            flat_fallback_indices = batch_plan.fallback_index_matrix.reshape((-1,))
            flat_active_mask = batch_plan.fallback_active_mask_matrix.reshape((-1,))
            genotype_matrix_by_variant = jnp.take(genotype_matrix_float32, flat_fallback_indices, axis=1).T
            if sparse_candidate_mask is None:
                flat_sparse_candidate_mask = jnp.zeros_like(flat_active_mask)
            else:
                flat_sparse_candidate_mask = (
                    jnp.take(jnp.asarray(sparse_candidate_mask, dtype=jnp.bool_), flat_fallback_indices, axis=0)
                    & flat_active_mask
                )
            heuristic_firth_mask = (
                compute_firth_pre_dispatch_mask_without_mask(
                    genotype_matrix_by_variant=genotype_matrix_by_variant,
                    phenotype_vector=chromosome_state.phenotype_vector,
                )
                | flat_sparse_candidate_mask
            ) & flat_active_mask
            ordered_candidate_inputs = group_firth_candidate_batch_inputs(
                flat_fallback_indices=flat_fallback_indices,
                flat_active_mask=flat_active_mask,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                heuristic_firth_mask=heuristic_firth_mask,
            )
            flat_fallback_indices = ordered_candidate_inputs.flat_fallback_indices
            flat_active_mask = ordered_candidate_inputs.flat_active_mask
            genotype_matrix_by_variant = ordered_candidate_inputs.genotype_matrix_by_variant
            heuristic_firth_mask = ordered_candidate_inputs.heuristic_firth_mask
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
            batch_count = batch_plan.fallback_index_matrix.shape[0]
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
                failure_code=batched_firth_result.failure_code.reshape((-1,)),
            )
            active_flat_positions = batch_plan.active_flat_position_vector
            active_fallback_indices = flat_fallback_indices[active_flat_positions]
            current_beta = jnp.take(result.beta, active_fallback_indices, axis=0)
            current_standard_error = jnp.take(result.standard_error, active_fallback_indices, axis=0)
            current_chi_squared = jnp.take(result.chi_squared, active_fallback_indices, axis=0)
            current_log10_p_value = jnp.take(result.log10_p_value, active_fallback_indices, axis=0)
            active_valid_mask = firth_result.valid_mask[active_flat_positions]
            active_firth_beta = firth_result.beta[active_flat_positions]
            active_firth_chi_squared = firth_result.chi_squared[active_flat_positions]
            active_firth_standard_error = firth_result.standard_error[active_flat_positions]
            if correction_plan.firth_se:
                active_firth_standard_error = jnp.where(
                    active_firth_chi_squared > 0.0,
                    jnp.abs(active_firth_beta) / jnp.sqrt(active_firth_chi_squared),
                    active_firth_standard_error,
                )
            merged_beta = jnp.where(active_valid_mask, firth_result.beta[active_flat_positions], current_beta)
            merged_standard_error = jnp.where(
                active_valid_mask,
                active_firth_standard_error,
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
            return regenie2_types.Regenie2BinaryChunkResult(
                beta=result.beta.at[active_fallback_indices].set(merged_beta),
                standard_error=result.standard_error.at[active_fallback_indices].set(merged_standard_error),
                chi_squared=result.chi_squared.at[active_fallback_indices].set(merged_chi_squared),
                log10_p_value=result.log10_p_value.at[active_fallback_indices].set(merged_log10_p_value),
                extra_code=result.extra_code.at[active_fallback_indices].set(merged_extra_code),
                valid_mask=result.valid_mask.at[active_fallback_indices].set(active_valid_mask),
                firth_iteration_count=result.firth_iteration_count.at[active_fallback_indices].set(
                    firth_result.iteration_count[active_flat_positions]
                ),
                firth_failure_code=result.firth_failure_code.at[active_fallback_indices].set(
                    firth_result.failure_code[active_flat_positions]
                ),
            )

        bounded_candidate_capacity = min(configured_candidate_capacity, variant_count)
        return jax.lax.cond(
            fallback_count <= bounded_candidate_capacity,
            lambda _: apply_candidate_corrections_with_capacity(bounded_candidate_capacity),
            lambda _: apply_candidate_corrections_with_capacity(variant_count),
            operand=None,
        )

    return jax.lax.cond(fallback_count > 0, apply_candidate_corrections, no_candidate_corrections)


def apply_device_candidate_corrections(
    chromosome_state: regenie2_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    result: regenie2_types.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
) -> regenie2_types.Regenie2BinaryChunkResult:
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
    )


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def compute_regenie2_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
) -> regenie2_types.Regenie2BinaryChunkResult:
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
    )


def compute_regenie2_binary_chunk(
    state: regenie2_types.Regenie2BinaryState,
    genotype_matrix: jax.Array,
    loco_offset: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
) -> regenie2_types.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association for a genotype chunk."""
    chromosome_state = prepare_regenie2_binary_chromosome_state(state, loco_offset, correction_plan)
    compute_regenie2_binary_chunk_from_state = typing.cast(
        "BinaryChunkComputeFunction",
        compute_regenie2_binary_chunk_from_chromosome_state,
    )
    return compute_regenie2_binary_chunk_from_state(
        chromosome_state,
        genotype_matrix,
        correction_plan,
        sparse_candidate_mask,
    )


import g.compute.regenie2_binary_variant_major_experimental as variant_major_experimental  # noqa: E402

compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major = (
    variant_major_experimental.compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major
)
compute_regenie2_binary_score_test_chunk_variant_major = (
    variant_major_experimental.compute_regenie2_binary_score_test_chunk_variant_major
)
apply_device_candidate_corrections_firth_variant_major = (
    variant_major_experimental.apply_device_candidate_corrections_firth_variant_major
)
apply_device_candidate_corrections_variant_major = (
    variant_major_experimental.apply_device_candidate_corrections_variant_major
)
compute_regenie2_binary_chunk_from_chromosome_state_variant_major = (
    variant_major_experimental.compute_regenie2_binary_chunk_from_chromosome_state_variant_major
)
