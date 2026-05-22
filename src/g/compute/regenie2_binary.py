"""REGENIE step 2 binary score-test kernel with device-resident Firth fallback."""

from __future__ import annotations

import functools
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g import types
from g.compute import (
    regenie2_binary_candidate_planning,
    regenie2_binary_config,
    regenie2_binary_firth_batch,
    regenie2_binary_firth_common,
    regenie2_binary_firth_full,
    regenie2_binary_firth_null,
    regenie2_binary_firth_scalar,
    regenie2_binary_firth_types,
    regenie2_binary_score,
    regenie2_binary_types,
)
from g.compute.common import genotype, linalg

jax.config.update("jax_enable_x64", val=True)

MINIMUM_PROBABILITY = regenie2_binary_config.MINIMUM_PROBABILITY
MINIMUM_VARIANCE = regenie2_binary_config.MINIMUM_VARIANCE
RELATIVE_VARIANCE_TOLERANCE = regenie2_binary_config.RELATIVE_VARIANCE_TOLERANCE
DEFAULT_MAXIMUM_NULL_ITERATIONS = regenie2_binary_config.DEFAULT_MAXIMUM_NULL_ITERATIONS
NULL_LOGISTIC_COEFFICIENT_TOLERANCE = regenie2_binary_config.NULL_LOGISTIC_COEFFICIENT_TOLERANCE
BINARY_CASE_THRESHOLD = regenie2_binary_config.BINARY_CASE_THRESHOLD
FIRTH_GRADIENT_TOLERANCE = regenie2_binary_config.FIRTH_GRADIENT_TOLERANCE
FIRTH_COEFFICIENT_TOLERANCE = regenie2_binary_config.FIRTH_COEFFICIENT_TOLERANCE
FIRTH_LIKELIHOOD_TOLERANCE = regenie2_binary_config.FIRTH_LIKELIHOOD_TOLERANCE
FIRTH_MAXIMUM_STEP_SIZE = regenie2_binary_config.FIRTH_MAXIMUM_STEP_SIZE
FIRTH_MAXIMUM_ITERATIONS = regenie2_binary_config.FIRTH_MAXIMUM_ITERATIONS
REGENIE_LOGISTIC_MINIMUM_ETA = regenie2_binary_config.REGENIE_LOGISTIC_MINIMUM_ETA
REGENIE_LOGISTIC_MAXIMUM_ETA = regenie2_binary_config.REGENIE_LOGISTIC_MAXIMUM_ETA
REGENIE_NUMERICAL_EPSILON = regenie2_binary_config.REGENIE_NUMERICAL_EPSILON
SPARSE_CARRIER_DOSAGE_THRESHOLD = regenie2_binary_firth_batch.SPARSE_CARRIER_DOSAGE_THRESHOLD
DEFAULT_BINARY_KERNEL_CONFIG = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG


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
FirthConvergenceReason = regenie2_binary_firth_types.FirthConvergenceReason
FirthState = regenie2_binary_firth_types.FirthState
FirthBacktrackingState = regenie2_binary_firth_types.FirthBacktrackingState
FirthBacktrackingResult = regenie2_binary_firth_types.FirthBacktrackingResult
InformationComponents = regenie2_binary_firth_types.InformationComponents
AdjustedWeightComponents = regenie2_binary_firth_types.AdjustedWeightComponents
FullModelScoreComponents = regenie2_binary_firth_types.FullModelScoreComponents
NullFirthFitResult = regenie2_binary_firth_types.NullFirthFitResult
NullFirthComponents = regenie2_binary_firth_types.NullFirthComponents
NullFirthNewtonRaphsonState = regenie2_binary_firth_types.NullFirthNewtonRaphsonState
NullFirthLineSearchState = regenie2_binary_firth_types.NullFirthLineSearchState
NullFirthLineSearchResult = regenie2_binary_firth_types.NullFirthLineSearchResult
FirthVariantResult = regenie2_binary_firth_types.FirthVariantResult
ScalarFirthComponents = regenie2_binary_firth_types.ScalarFirthComponents
ScalarPseudoFirthState = regenie2_binary_firth_types.ScalarPseudoFirthState
ScalarPseudoLogisticState = regenie2_binary_firth_types.ScalarPseudoLogisticState
ScalarNewtonRaphsonState = regenie2_binary_firth_types.ScalarNewtonRaphsonState
ScalarLineSearchState = regenie2_binary_firth_types.ScalarLineSearchState
ScalarFirthAttemptResult = regenie2_binary_firth_types.ScalarFirthAttemptResult
ApproximateFirthCandidateInputs = regenie2_binary_firth_types.ApproximateFirthCandidateInputs


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
    return regenie2_binary_score.compute_positive_variance_mask(variance, reference_sum_squares)


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
    return regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
    )


def compute_regenie2_multi_binary_score_test_chunk_variant_major_core(
    chromosome_state: regenie2_binary_types.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Compute batched binary score tests for trait-major states and variant-major genotypes.

    Args:
        chromosome_state: Trait-major chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        correction_plan: Binary fallback/correction policy.

    Returns:
        Trait-major score-test result for the chunk.

    """
    return regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
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
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return compute_regenie2_multi_binary_score_test_chunk_variant_major_core(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=jnp.asarray(genotype_matrix, dtype=jnp.float32).T,
            correction_plan=correction_plan,
        )

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
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return compute_regenie2_multi_binary_score_test_chunk_variant_major_core(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
        )

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
    return regenie2_binary_firth_full.compute_information_components(
        covariate_matrix=covariate_matrix,
        genotype_vector=genotype_vector,
        probability_vector=probability_vector,
    )


def compute_weighted_full_model_information_components(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    weight_vector: jax.Array,
) -> InformationComponents:
    """Compute full-model information blocks for one explicit weight vector."""
    return regenie2_binary_firth_full.compute_weighted_full_model_information_components(
        covariate_matrix=covariate_matrix,
        genotype_vector=genotype_vector,
        weight_vector=weight_vector,
    )


def compute_firth_penalized_log_likelihood_from_cholesky(
    probability_vector: jax.Array,
    phenotype_vector: jax.Array,
    information_cholesky_factor: jax.Array,
) -> jax.Array:
    """Compute Firth-penalized log-likelihood from a Cholesky factor."""
    return regenie2_binary_firth_common.compute_firth_penalized_log_likelihood_from_cholesky(
        probability_vector=probability_vector,
        phenotype_vector=phenotype_vector,
        information_cholesky_factor=information_cholesky_factor,
    )


def compute_firth_convergence_mask(
    *,
    current_penalized_log_likelihood: jax.Array,
    candidate_penalized_log_likelihood: jax.Array,
    coefficient_step: jax.Array,
    adjusted_score: jax.Array,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> jax.Array:
    """Return whether an accepted Firth step satisfies convergence tolerances."""
    return regenie2_binary_firth_common.compute_firth_convergence_mask(
        current_penalized_log_likelihood=current_penalized_log_likelihood,
        candidate_penalized_log_likelihood=candidate_penalized_log_likelihood,
        coefficient_step=coefficient_step,
        adjusted_score=adjusted_score,
        kernel_config=kernel_config,
    )


def run_firth_step_halving(
    *,
    current_coefficients: jax.Array,
    current_penalized_log_likelihood: jax.Array,
    coefficient_step: jax.Array,
    evaluate_penalized_log_likelihood: typing.Callable[[jax.Array], jax.Array],
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> FirthBacktrackingResult:
    """Accept the first bounded Firth step that preserves penalized likelihood."""
    return regenie2_binary_firth_common.run_firth_step_halving(
        current_coefficients=current_coefficients,
        current_penalized_log_likelihood=current_penalized_log_likelihood,
        coefficient_step=coefficient_step,
        evaluate_penalized_log_likelihood=evaluate_penalized_log_likelihood,
        kernel_config=kernel_config,
    )


def map_firth_reason_code_to_failure_code(reason_code: jax.Array) -> jax.Array:
    """Map internal Firth termination reasons to public failure labels."""
    return regenie2_binary_firth_common.map_firth_reason_code_to_failure_code(reason_code)


def map_scalar_pseudo_firth_failure_to_reason_code(failure_code: jax.Array) -> jax.Array:
    """Map REGENIE scalar pseudo-Firth failure states to internal reason codes."""
    return regenie2_binary_firth_scalar.map_scalar_pseudo_firth_failure_to_reason_code(failure_code)


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
    return regenie2_binary_firth_scalar.compute_scalar_firth_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=beta,
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
    return regenie2_binary_firth_scalar.fit_scalar_pseudo_logistic_step(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        active_sample_mask=active_sample_mask,
        offset_vector=offset_vector,
        adjusted_response=adjusted_response,
        initial_score=initial_score,
        initial_genotype_information=initial_genotype_information,
        initial_beta=initial_beta,
        tolerance=tolerance,
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
    return regenie2_binary_firth_scalar.fit_scalar_pseudo_firth(
        deviance_null=deviance_null,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        initial_beta=initial_beta,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
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
    return regenie2_binary_firth_scalar.run_scalar_line_search(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        current_beta=current_beta,
        current_penalized_deviance=current_penalized_deviance,
        initial_step_size=initial_step_size,
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
    return regenie2_binary_firth_scalar.fit_scalar_newton_raphson_firth(
        deviance_null=deviance_null,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        initial_beta=initial_beta,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        maximum_step_size=maximum_step_size,
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
    return regenie2_binary_firth_scalar.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        carrier_sample_mask=carrier_sample_mask,
        sparse_correction=sparse_correction,
        warm_start_beta=warm_start_beta,
        skip_firth=skip_firth,
        null_failed=null_failed,
        kernel_config=kernel_config,
    )


def compute_full_model_adjusted_weight_components(
    full_design_matrix: jax.Array,
    probability_vector: jax.Array,
    information_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> AdjustedWeightComponents:
    """Compute leverage-adjusted Firth weights for one full model."""
    return regenie2_binary_firth_full.compute_full_model_adjusted_weight_components(
        full_design_matrix=full_design_matrix,
        probability_vector=probability_vector,
        information_matrix=information_matrix,
        phenotype_vector=phenotype_vector,
    )


def compute_full_model_adjusted_weight_components_from_parts(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    probability_vector: jax.Array,
    information_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> AdjustedWeightComponents:
    """Compute full-model Firth weights without materializing a full design matrix."""
    return regenie2_binary_firth_full.compute_full_model_adjusted_weight_components_from_parts(
        covariate_matrix=covariate_matrix,
        genotype_vector=genotype_vector,
        probability_vector=probability_vector,
        information_matrix=information_matrix,
        phenotype_vector=phenotype_vector,
    )


def compute_full_model_score_components(
    covariate_matrix: jax.Array,
    genotype_vector: jax.Array,
    score_weight_vector: jax.Array,
) -> FullModelScoreComponents:
    """Compute covariate and genotype score blocks without a full design matrix."""
    return regenie2_binary_firth_full.compute_full_model_score_components(
        covariate_matrix=covariate_matrix,
        genotype_vector=genotype_vector,
        score_weight_vector=score_weight_vector,
    )


def build_full_model_information_matrix(
    *,
    covariate_information_matrix: jax.Array,
    cross_information_vector: jax.Array,
    genotype_information: jax.Array,
) -> jax.Array:
    """Build a full-model information matrix from block components."""
    return regenie2_binary_firth_full.build_full_model_information_matrix(
        covariate_information_matrix=covariate_information_matrix,
        cross_information_vector=cross_information_vector,
        genotype_information=genotype_information,
    )


def compute_covariate_only_adjusted_weight_components(
    covariate_matrix: jax.Array,
    probability_vector: jax.Array,
    information_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> AdjustedWeightComponents:
    """Compute leverage-adjusted Firth weights for the covariate-only null model."""
    return regenie2_binary_firth_full.compute_covariate_only_adjusted_weight_components(
        covariate_matrix=covariate_matrix,
        probability_vector=probability_vector,
        information_matrix=information_matrix,
        phenotype_vector=phenotype_vector,
    )


def compute_null_firth_components(
    *,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    coefficients: jax.Array,
) -> NullFirthComponents:
    """Compute REGENIE null Firth score and deviance quantities."""
    return regenie2_binary_firth_null.compute_null_firth_components(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        loco_offset=loco_offset,
        coefficients=coefficients,
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
    return regenie2_binary_firth_null.run_null_firth_line_search(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        loco_offset=loco_offset,
        current_coefficients=current_coefficients,
        current_deviance=current_deviance,
        coefficient_step=coefficient_step,
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
    return regenie2_binary_firth_null.fit_covariate_only_firth_null_model_once(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        loco_offset=loco_offset,
        initial_coefficients=initial_coefficients,
        maximum_iterations=maximum_iterations,
        maximum_step_size=maximum_step_size,
        tolerance=tolerance,
        check_score_increase=check_score_increase,
    )


def fit_covariate_only_firth_null_model(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = DEFAULT_BINARY_KERNEL_CONFIG,
) -> NullFirthFitResult:
    """Fit the covariate-only Firth null model and return diagnostics."""
    return regenie2_binary_firth_null.fit_covariate_only_firth_null_model(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        loco_offset=loco_offset,
        initial_coefficients=initial_coefficients,
        kernel_config=kernel_config,
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
    return regenie2_binary_firth_full.fit_single_variant_firth_logistic_regression(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        loco_offset=loco_offset,
        initial_coefficients=initial_coefficients,
        skip_firth=skip_firth,
        null_penalized_log_likelihood=null_penalized_log_likelihood,
        kernel_config=kernel_config,
    )


def compute_firth_pre_dispatch_mask_without_mask(
    genotype_matrix_by_variant: jax.Array,
    phenotype_vector: jax.Array,
) -> jax.Array:
    """Identify variants with obvious case-control allele-count separation."""
    return regenie2_binary_firth_batch.compute_firth_pre_dispatch_mask_without_mask(
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        phenotype_vector=phenotype_vector,
    )


def initialize_full_model_coefficients_without_mask(
    covariate_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    phenotype_vector: jax.Array,
) -> jax.Array:
    """Initialize full-model coefficients with a pseudo-response regression."""
    return regenie2_binary_firth_batch.initialize_full_model_coefficients_without_mask(
        covariate_matrix=covariate_matrix,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        phenotype_vector=phenotype_vector,
    )


def residualize_and_scale_genotypes_for_approximate_firth(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
) -> jax.Array:
    """Build REGENIE's approximate-Firth residualized genotype vector."""
    return regenie2_binary_firth_batch.residualize_and_scale_genotypes_for_approximate_firth(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
    )


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
    return regenie2_binary_firth_batch.compute_firth_variantwise(
        covariate_matrix=covariate_matrix,
        null_logistic_coefficients=null_logistic_coefficients,
        null_firth_offset=null_firth_offset,
        phenotype_vector=phenotype_vector,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant=raw_genotype_matrix_by_variant,
        loco_offset=loco_offset,
        initial_coefficients=initial_coefficients,
        skip_firth_mask=skip_firth_mask,
        sparse_correction_mask=sparse_correction_mask,
        null_penalized_log_likelihood=null_penalized_log_likelihood,
        kernel_config=kernel_config,
    )


def build_empty_firth_variant_result(
    batch_size: int,
) -> FirthVariantResult:
    """Build a placeholder Firth result for skipped padded batches."""
    return regenie2_binary_firth_batch.build_empty_firth_variant_result(batch_size)


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
