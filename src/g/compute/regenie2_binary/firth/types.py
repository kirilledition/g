"""JAX pytree containers for binary Firth correction kernels."""

from __future__ import annotations

import enum
import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g import types as g_types


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


def map_firth_reason_code_to_failure_code(reason_code: jax.Array) -> jax.Array:
    """Map internal Firth termination reasons to public failure labels."""
    return jnp.where(
        reason_code == FirthConvergenceReason.MAX_ITERATIONS.value,
        g_types.FirthFailureCode.MAX_ITERATIONS.value,
        jnp.where(
            reason_code == FirthConvergenceReason.INVALID_STATISTIC.value,
            g_types.FirthFailureCode.INVALID_STATISTIC.value,
            jnp.where(
                reason_code == FirthConvergenceReason.NEGATIVE_LRT.value,
                g_types.FirthFailureCode.INVALID_STATISTIC.value,
                jnp.where(
                    (reason_code == FirthConvergenceReason.STEP_HALVING_EXHAUSTED.value)
                    | (reason_code == FirthConvergenceReason.STEP_SIZE_INCREASE.value),
                    g_types.FirthFailureCode.STEP_HALVING.value,
                    jnp.where(
                        (reason_code == FirthConvergenceReason.NUMERICAL_FAILURE.value)
                        | (reason_code == FirthConvergenceReason.NULL_FAILURE.value)
                        | (reason_code == FirthConvergenceReason.PROBABILITY_FAILURE.value),
                        g_types.FirthFailureCode.NUMERICAL.value,
                        g_types.FirthFailureCode.NONE.value,
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)


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
class NullFirthLineSearchLoopCarry:
    """Loop carry for covariate-only null Firth line search.

    Attributes:
        state: Mutable line-search state.
        covariate_matrix: Covariate design matrix.
        phenotype_vector: Binary phenotype values.
        loco_offset: Per-sample LOCO offset.
        current_coefficients: Coefficients at the start of line search.
        current_deviance: Accepted deviance at the start of line search.
        maximum_attempts: Maximum step-halving attempts.
        step_halving_scale: Multiplicative factor for rejected steps.

    """

    state: NullFirthLineSearchState
    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    loco_offset: jax.Array
    current_coefficients: jax.Array
    current_deviance: jax.Array
    maximum_attempts: jax.Array
    step_halving_scale: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthNewtonRaphsonLoopCarry:
    """Loop carry for covariate-only null Firth Newton-Raphson."""

    state: NullFirthNewtonRaphsonState
    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    loco_offset: jax.Array
    maximum_iterations: jax.Array
    maximum_step_size: jax.Array
    tolerance: jax.Array
    line_search_maximum_attempts: jax.Array
    line_search_step_halving_scale: jax.Array
    check_score_increase: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthFallbackParameters:
    """Explicit operands for the null Firth fallback cascade."""

    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    loco_offset: jax.Array
    initial_coefficients: jax.Array
    zero_start_coefficients: jax.Array
    maximum_iterations: jax.Array
    fallback_maximum_iterations: jax.Array
    maximum_step_size: jax.Array
    fallback_maximum_step_size: jax.Array
    tolerance: jax.Array
    line_search_maximum_attempts: jax.Array
    line_search_step_halving_scale: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthFallbackLoopCarry:
    """Loop carry for lazy null Firth fallback attempts."""

    parameters: NullFirthFallbackParameters
    selected_result: NullFirthFitResult
    next_attempt_index: jax.Array


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


def build_empty_firth_variant_result(batch_size: int) -> FirthVariantResult:
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


def flatten_batched_firth_variant_result(result: FirthVariantResult) -> FirthVariantResult:
    """Flatten batched Firth outputs into candidate-lane order."""
    return FirthVariantResult(
        beta=result.beta.reshape((-1,)),
        standard_error=result.standard_error.reshape((-1,)),
        chi_squared=result.chi_squared.reshape((-1,)),
        log10_p_value=result.log10_p_value.reshape((-1,)),
        penalized_log_likelihood=result.penalized_log_likelihood.reshape((-1,)),
        converged_mask=result.converged_mask.reshape((-1,)),
        valid_mask=result.valid_mask.reshape((-1,)),
        iteration_count=result.iteration_count.reshape((-1,)),
        failure_code=result.failure_code.reshape((-1,)),
        convergence_reason_code=result.convergence_reason_code.reshape((-1,)),
        correction_code=result.correction_code.reshape((-1,)),
        sparse_correction_mask=result.sparse_correction_mask.reshape((-1,)),
        pseudo_firth_iteration_count=result.pseudo_firth_iteration_count.reshape((-1,)),
        nr_zero_start_iteration_count=result.nr_zero_start_iteration_count.reshape((-1,)),
        nr_warm_start_iteration_count=result.nr_warm_start_iteration_count.reshape((-1,)),
    )


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
class ScalarApproximateFirthSolverParameters:
    """Scalar approximate-Firth policy values carried through JAX branches."""

    minimum_variance: jax.Array
    tolerance: jax.Array
    maximum_step_size: jax.Array
    pseudo_maximum_iterations: jax.Array
    pseudo_inner_maximum_iterations: jax.Array
    newton_raphson_maximum_iterations: jax.Array
    newton_raphson_zero_start_iterations: jax.Array
    line_search_maximum_attempts: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarPseudoLogisticLoopCarry:
    """Loop carry for one pseudo-response scalar logistic update."""

    state: ScalarPseudoLogisticState
    genotype_vector: jax.Array
    active_sample_mask: jax.Array
    offset_vector: jax.Array
    adjusted_response: jax.Array
    tolerance: jax.Array
    maximum_iterations: jax.Array
    maximum_step_size: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarPseudoFirthLoopCarry:
    """Loop carry for the scalar pseudo-Firth outer iteration."""

    state: ScalarPseudoFirthState
    phenotype_vector: jax.Array
    genotype_vector: jax.Array
    offset_vector: jax.Array
    active_sample_mask: jax.Array
    non_active_deviance: jax.Array
    tolerance: jax.Array
    maximum_iterations: jax.Array
    inner_maximum_iterations: jax.Array
    maximum_step_size: jax.Array
    minimum_variance: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarLineSearchLoopCarry:
    """Loop carry for scalar Newton-Raphson step-halving."""

    state: ScalarLineSearchState
    phenotype_vector: jax.Array
    genotype_vector: jax.Array
    offset_vector: jax.Array
    active_sample_mask: jax.Array
    non_active_deviance: jax.Array
    current_beta: jax.Array
    current_penalized_deviance: jax.Array
    maximum_attempts: jax.Array
    minimum_variance: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarNewtonRaphsonLoopCarry:
    """Loop carry for scalar Newton-Raphson approximate Firth."""

    state: ScalarNewtonRaphsonState
    phenotype_vector: jax.Array
    genotype_vector: jax.Array
    offset_vector: jax.Array
    active_sample_mask: jax.Array
    non_active_deviance: jax.Array
    tolerance: jax.Array
    maximum_iterations: jax.Array
    maximum_step_size: jax.Array
    line_search_maximum_attempts: jax.Array
    minimum_variance: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarApproximateFirthDispatchOperands:
    """Operands for scalar approximate-Firth active/inactive dispatch."""

    phenotype_vector: jax.Array
    genotype_vector: jax.Array
    offset_vector: jax.Array
    active_sample_mask: jax.Array
    deviance_null: jax.Array
    non_active_deviance: jax.Array
    sparse_correction: jax.Array
    warm_start_beta: jax.Array
    skip_firth: jax.Array
    null_failed: jax.Array
    solver_parameters: ScalarApproximateFirthSolverParameters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarApproximateFirthFallbackOperands:
    """Operands for scalar approximate-Firth fallback after pseudo-Firth."""

    dispatch_operands: ScalarApproximateFirthDispatchOperands
    pseudo_result: ScalarFirthAttemptResult


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarApproximateFirthZeroStartOperands:
    """Operands for scalar zero-start selection before warm-start fallback."""

    fallback_operands: ScalarApproximateFirthFallbackOperands
    zero_start_result: ScalarFirthAttemptResult


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarApproximateFirthResultOperands:
    """Operands used to build the final scalar approximate-Firth result."""

    dispatch_operands: ScalarApproximateFirthDispatchOperands
    pseudo_result: ScalarFirthAttemptResult
    zero_start_result: ScalarFirthAttemptResult
    warm_start_result: ScalarFirthAttemptResult
    run_zero_start: jax.Array
    run_warm_start: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarVariantFirthLaneSharedOperands:
    """Shared operands for scalar variant-wise approximate-Firth lanes."""

    phenotype_vector: jax.Array
    offset_vector: jax.Array
    sparse_carrier_dosage_threshold: jax.Array
    null_failed: jax.Array
    solver_parameters: ScalarApproximateFirthSolverParameters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarVariantFirthFixedBatchScanCarry:
    """Scan carry for scalar variant-wise approximate-Firth fixed batches."""

    shared_operands: ScalarVariantFirthLaneSharedOperands
    genotype_batches: jax.Array
    raw_genotype_batches: jax.Array
    active_mask_batches: jax.Array
    sparse_correction_mask_batches: jax.Array
    active_batch_count: jax.Array
    empty_firth_variant_result: FirthVariantResult


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarVariantFirthFixedBatchOperands:
    """Branch operands for one scalar variant-wise fixed batch."""

    carry: ScalarVariantFirthFixedBatchScanCarry
    batch_index: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CompactSparseFirthLaneSharedOperands:
    """Shared operands for compact sparse approximate-Firth lanes."""

    solver_parameters: ScalarApproximateFirthSolverParameters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CompactSparseFirthFixedBatchScanCarry:
    """Scan carry for compact sparse approximate-Firth fixed batches."""

    shared_operands: CompactSparseFirthLaneSharedOperands
    phenotype_batches: jax.Array
    genotype_batches: jax.Array
    offset_batches: jax.Array
    active_carrier_slot_mask_batches: jax.Array
    active_mask_batches: jax.Array
    full_null_deviance_batches: jax.Array
    null_failed_mask_batches: jax.Array
    active_batch_count: jax.Array
    empty_firth_variant_result: FirthVariantResult


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CompactSparseFirthFixedBatchOperands:
    """Branch operands for one compact sparse fixed batch."""

    carry: CompactSparseFirthFixedBatchScanCarry
    batch_index: jax.Array


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "null_firth_offset",
        "phenotype_vector",
        "genotype_matrix_by_variant",
        "raw_genotype_matrix_by_variant",
        "active_mask",
        "sparse_correction_mask",
        "fallback_count",
        "null_penalized_log_likelihood",
        "full_null_deviance",
        "sparse_carrier_dosage_threshold",
        "solver_parameters",
    ),
    meta_fields=("firth_batch_size",),
)
@dataclass(frozen=True)
class ScalarFirthSparseCompactionOperands:
    """Operands for single-trait scalar sparse-compaction dispatch."""

    null_firth_offset: jax.Array
    phenotype_vector: jax.Array
    genotype_matrix_by_variant: jax.Array
    raw_genotype_matrix_by_variant: jax.Array
    active_mask: jax.Array
    sparse_correction_mask: jax.Array
    fallback_count: jax.Array
    firth_batch_size: int
    null_penalized_log_likelihood: jax.Array
    full_null_deviance: jax.Array
    sparse_carrier_dosage_threshold: jax.Array
    solver_parameters: ScalarApproximateFirthSolverParameters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarFirthSparseSplitOperands:
    """Operands for splitting scalar Firth lanes into dense and compact streams."""

    compaction_operands: ScalarFirthSparseCompactionOperands
    carrier_count: jax.Array
    compact_sparse_lane_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarFirthSparseStreamOperands:
    """Operands for one dense or compact scalar Firth stream."""

    split_operands: ScalarFirthSparseSplitOperands
    lane_indices: jax.Array
    active_mask: jax.Array
    active_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthStreamScatterOperands:
    """Operands for scattering stream results back into candidate-lane order."""

    base_result: FirthVariantResult
    lane_indices: jax.Array
    active_mask: jax.Array
    stream_result: FirthVariantResult
