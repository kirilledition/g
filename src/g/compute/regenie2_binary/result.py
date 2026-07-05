"""Binary result constructors for REGENIE step 2 compute."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2BinaryScoreChunkResult:
    """Score-test association outputs for a REGENIE step 2 binary chunk.

    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        chi_squared: Chi-squared statistics.
        log10_p_value: Negative log10 p-values.
        extra_code: Integer value from `types.BinaryExtraCode` for output rendering.
        valid_mask: Boolean mask for valid statistics.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2BinaryChunkResult:
    """Association outputs for a REGENIE step 2 binary chunk.

    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        chi_squared: Chi-squared statistics.
        log10_p_value: Negative log10 p-values.
        extra_code: Integer value from `types.BinaryExtraCode` for output rendering.
        valid_mask: Boolean mask for valid statistics.
        firth_iteration_count: Number of Firth iterations per variant, or zero for non-Firth rows.
        firth_failure_code: Integer value from `types.FirthFailureCode`, or zero for non-failed rows.
        firth_convergence_reason_code: Internal Firth termination-reason integer.
        firth_correction_code: Integer value from `types.FirthCorrectionCode`.
        firth_sparse_correction_mask: Whether the approximate correction used carrier-only sparse inputs.
        pseudo_firth_iteration_count: Scalar pseudo-Firth iterations per variant.
        nr_zero_start_iteration_count: Scalar Newton-Raphson zero-start iterations per variant.
        nr_warm_start_iteration_count: Scalar Newton-Raphson warm-start iterations per variant.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array
    firth_iteration_count: jax.Array
    firth_failure_code: jax.Array
    firth_convergence_reason_code: jax.Array
    firth_correction_code: jax.Array
    firth_sparse_correction_mask: jax.Array
    pseudo_firth_iteration_count: jax.Array
    nr_zero_start_iteration_count: jax.Array
    nr_warm_start_iteration_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryScoreChunkResult:
    """Trait-major score-test outputs for a multi-trait binary chunk.

    Attributes:
        beta: Estimated effect sizes with shape ``traits x variants``.
        standard_error: Standard errors with shape ``traits x variants``.
        chi_squared: Chi-squared statistics with shape ``traits x variants``.
        log10_p_value: Negative log10 p-values with shape ``traits x variants``.
        extra_code: Integer values from `types.BinaryExtraCode` with shape ``traits x variants``.
        valid_mask: Boolean mask for valid statistics with shape ``traits x variants``.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryChunkResult:
    """Trait-major association outputs for a multi-trait binary chunk.

    Attributes:
        beta: Estimated effect sizes with shape ``traits x variants``.
        standard_error: Standard errors with shape ``traits x variants``.
        chi_squared: Chi-squared statistics with shape ``traits x variants``.
        log10_p_value: Negative log10 p-values with shape ``traits x variants``.
        extra_code: Integer values from `types.BinaryExtraCode` with shape ``traits x variants``.
        valid_mask: Boolean mask for valid statistics with shape ``traits x variants``.
        firth_iteration_count: Firth iteration counts with shape ``traits x variants``.
        firth_failure_code: Values from `types.FirthFailureCode` with shape ``traits x variants``.
        firth_convergence_reason_code: Internal Firth termination-reason integers with shape ``traits x variants``.
        firth_correction_code: Values from `types.FirthCorrectionCode` with shape ``traits x variants``.
        firth_sparse_correction_mask: Sparse carrier-only correction flags with shape ``traits x variants``.
        pseudo_firth_iteration_count: Scalar pseudo-Firth iteration counts with shape ``traits x variants``.
        nr_zero_start_iteration_count: Scalar zero-start NR iteration counts with shape ``traits x variants``.
        nr_warm_start_iteration_count: Scalar warm-start NR iteration counts with shape ``traits x variants``.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array
    firth_iteration_count: jax.Array
    firth_failure_code: jax.Array
    firth_convergence_reason_code: jax.Array
    firth_correction_code: jax.Array
    firth_sparse_correction_mask: jax.Array
    pseudo_firth_iteration_count: jax.Array
    nr_zero_start_iteration_count: jax.Array
    nr_warm_start_iteration_count: jax.Array


type Regenie2AnyBinaryChunkResult = (
    Regenie2BinaryScoreChunkResult
    | Regenie2BinaryChunkResult
    | Regenie2MultiBinaryScoreChunkResult
    | Regenie2MultiBinaryChunkResult
)
type Regenie2BinaryDiagnosticChunkResult = Regenie2BinaryChunkResult | Regenie2MultiBinaryChunkResult


def build_empty_firth_integer_array(extra_code: jax.Array) -> jax.Array:
    """Build an integer Firth diagnostic array for score-test-only results."""
    return jnp.zeros_like(extra_code, dtype=jnp.int32)


def build_empty_firth_boolean_array(extra_code: jax.Array) -> jax.Array:
    """Build a boolean Firth diagnostic array for score-test-only results."""
    return jnp.zeros_like(extra_code, dtype=jnp.bool_)


def expand_score_result_with_empty_firth_diagnostics(
    result: Regenie2BinaryScoreChunkResult,
) -> Regenie2BinaryChunkResult:
    """Add empty Firth diagnostic arrays to a single-trait score result."""
    return Regenie2BinaryChunkResult(
        beta=result.beta,
        standard_error=result.standard_error,
        chi_squared=result.chi_squared,
        log10_p_value=result.log10_p_value,
        extra_code=result.extra_code,
        valid_mask=result.valid_mask,
        firth_iteration_count=build_empty_firth_integer_array(result.extra_code),
        firth_failure_code=build_empty_firth_integer_array(result.extra_code),
        firth_convergence_reason_code=build_empty_firth_integer_array(result.extra_code),
        firth_correction_code=build_empty_firth_integer_array(result.extra_code),
        firth_sparse_correction_mask=build_empty_firth_boolean_array(result.extra_code),
        pseudo_firth_iteration_count=build_empty_firth_integer_array(result.extra_code),
        nr_zero_start_iteration_count=build_empty_firth_integer_array(result.extra_code),
        nr_warm_start_iteration_count=build_empty_firth_integer_array(result.extra_code),
    )


def expand_multi_score_result_with_empty_firth_diagnostics(
    result: Regenie2MultiBinaryScoreChunkResult,
) -> Regenie2MultiBinaryChunkResult:
    """Add empty Firth diagnostic arrays to a multi-trait score result."""
    return Regenie2MultiBinaryChunkResult(
        beta=result.beta,
        standard_error=result.standard_error,
        chi_squared=result.chi_squared,
        log10_p_value=result.log10_p_value,
        extra_code=result.extra_code,
        valid_mask=result.valid_mask,
        firth_iteration_count=build_empty_firth_integer_array(result.extra_code),
        firth_failure_code=build_empty_firth_integer_array(result.extra_code),
        firth_convergence_reason_code=build_empty_firth_integer_array(result.extra_code),
        firth_correction_code=build_empty_firth_integer_array(result.extra_code),
        firth_sparse_correction_mask=build_empty_firth_boolean_array(result.extra_code),
        pseudo_firth_iteration_count=build_empty_firth_integer_array(result.extra_code),
        nr_zero_start_iteration_count=build_empty_firth_integer_array(result.extra_code),
        nr_warm_start_iteration_count=build_empty_firth_integer_array(result.extra_code),
    )


def expand_binary_result_with_empty_firth_diagnostics(
    result: Regenie2AnyBinaryChunkResult,
) -> Regenie2BinaryDiagnosticChunkResult:
    """Return a binary result with every Firth diagnostic array present."""
    if isinstance(result, Regenie2BinaryScoreChunkResult):
        return expand_score_result_with_empty_firth_diagnostics(result)
    if isinstance(result, Regenie2MultiBinaryScoreChunkResult):
        return expand_multi_score_result_with_empty_firth_diagnostics(result)
    return result


def squeeze_single_binary_score_result(
    result: Regenie2MultiBinaryScoreChunkResult,
) -> Regenie2BinaryScoreChunkResult:
    """Remove the trait axis from a single-trait binary score result."""
    return Regenie2BinaryScoreChunkResult(
        beta=result.beta[0],
        standard_error=result.standard_error[0],
        chi_squared=result.chi_squared[0],
        log10_p_value=result.log10_p_value[0],
        extra_code=result.extra_code[0],
        valid_mask=result.valid_mask[0],
    )
