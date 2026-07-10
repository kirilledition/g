"""JAX pytree containers for binary Firth correction kernels."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


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

    """

    coefficients: jax.Array
    penalized_log_likelihood: jax.Array
    converged: jax.Array
    failed: jax.Array
    iteration_count: jax.Array


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
class AdjustedWeightComponents:
    """Intermediate leverage-adjusted weights for Firth updates.

    Attributes:
        adjusted_weight_vector: Adjusted score contribution per sample.
        second_weight_vector: Second-order Hessian weights.

    """

    adjusted_weight_vector: jax.Array
    second_weight_vector: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthFitResult:
    """Result of the covariate-only Firth null fit.

    Attributes:
        coefficients: Final covariate coefficients, or the last attempted coefficients on failure.
        penalized_log_likelihood: Final trusted penalized log-likelihood, or NaN on failure.
        converged: Whether the null fit converged.

    """

    coefficients: jax.Array
    penalized_log_likelihood: jax.Array
    converged: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthComponents:
    """Intermediate quantities for REGENIE-style null Firth Newton-Raphson."""

    information_cholesky_factor: jax.Array
    deviance: jax.Array
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
        valid_mask: Whether corrected statistics are valid.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    valid_mask: jax.Array


def build_empty_firth_variant_result(batch_size: int) -> FirthVariantResult:
    """Build a placeholder Firth result for skipped padded batches."""
    return FirthVariantResult(
        beta=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        standard_error=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        chi_squared=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        log10_p_value=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        valid_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


def flatten_batched_firth_variant_result(result: FirthVariantResult) -> FirthVariantResult:
    """Flatten batched Firth outputs into candidate-lane order."""
    return FirthVariantResult(
        beta=result.beta.reshape((-1,)),
        standard_error=result.standard_error.reshape((-1,)),
        chi_squared=result.chi_squared.reshape((-1,)),
        log10_p_value=result.log10_p_value.reshape((-1,)),
        valid_mask=result.valid_mask.reshape((-1,)),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarFirthComponents:
    """Scalar approximate-Firth quantities for one beta value.

    Attributes:
        probability_vector: Fitted probabilities for the active correction samples.
        genotype_information: Scalar genotype information.
        genotype_information_diagonal: Per-sample contributions to genotype information.
        penalized_deviance: REGENIE approximate penalized deviance.
        score: Scalar modified score.
        valid: Whether probabilities, weights, and information are finite and usable.

    """

    probability_vector: jax.Array
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
    outer_iteration_count: jax.Array
    beta_iteration_14: jax.Array
    converged: jax.Array
    failed: jax.Array


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
    valid: jax.Array


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
class CompactSparseFirthFixedBatchScanCarry:
    """Scan carry for compact sparse approximate-Firth fixed batches."""

    solver_parameters: ScalarApproximateFirthSolverParameters
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
