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


def build_empty_firth_integer_array(extra_code: jax.Array) -> jax.Array:
    """Build an integer Firth diagnostic array for score-test-only results."""
    return jnp.zeros_like(extra_code, dtype=jnp.int32)


def build_empty_firth_boolean_array(extra_code: jax.Array) -> jax.Array:
    """Build a boolean Firth diagnostic array for score-test-only results."""
    return jnp.zeros_like(extra_code, dtype=jnp.bool_)


def build_multi_binary_score_test_chunk_result(
    *,
    beta: jax.Array,
    standard_error: jax.Array,
    chi_squared: jax.Array,
    log10_p_value: jax.Array,
    extra_code: jax.Array,
    valid_mask: jax.Array,
) -> Regenie2MultiBinaryScoreChunkResult:
    """Build a multi-trait binary score-test chunk result."""
    return Regenie2MultiBinaryScoreChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        valid_mask=valid_mask,
    )


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


def squeeze_single_binary_chunk_result(
    result: Regenie2MultiBinaryChunkResult,
) -> Regenie2BinaryChunkResult:
    """Remove the trait axis from a single-trait binary result."""
    return Regenie2BinaryChunkResult(
        beta=result.beta[0],
        standard_error=result.standard_error[0],
        chi_squared=result.chi_squared[0],
        log10_p_value=result.log10_p_value[0],
        extra_code=result.extra_code[0],
        valid_mask=result.valid_mask[0],
        firth_iteration_count=result.firth_iteration_count[0],
        firth_failure_code=result.firth_failure_code[0],
        firth_convergence_reason_code=result.firth_convergence_reason_code[0],
        firth_correction_code=result.firth_correction_code[0],
        firth_sparse_correction_mask=result.firth_sparse_correction_mask[0],
        pseudo_firth_iteration_count=result.pseudo_firth_iteration_count[0],
        nr_zero_start_iteration_count=result.nr_zero_start_iteration_count[0],
        nr_warm_start_iteration_count=result.nr_warm_start_iteration_count[0],
    )


def stack_binary_chunk_results(
    results: list[Regenie2BinaryChunkResult],
) -> Regenie2MultiBinaryChunkResult:
    """Stack per-trait binary chunk results into a trait-major result."""
    return Regenie2MultiBinaryChunkResult(
        beta=jnp.stack([result.beta for result in results], axis=0),
        standard_error=jnp.stack([result.standard_error for result in results], axis=0),
        chi_squared=jnp.stack([result.chi_squared for result in results], axis=0),
        log10_p_value=jnp.stack([result.log10_p_value for result in results], axis=0),
        extra_code=jnp.stack([result.extra_code for result in results], axis=0),
        valid_mask=jnp.stack([result.valid_mask for result in results], axis=0),
        firth_iteration_count=jnp.stack([result.firth_iteration_count for result in results], axis=0),
        firth_failure_code=jnp.stack([result.firth_failure_code for result in results], axis=0),
        firth_convergence_reason_code=jnp.stack(
            [result.firth_convergence_reason_code for result in results],
            axis=0,
        ),
        firth_correction_code=jnp.stack([result.firth_correction_code for result in results], axis=0),
        firth_sparse_correction_mask=jnp.stack([result.firth_sparse_correction_mask for result in results], axis=0),
        pseudo_firth_iteration_count=jnp.stack([result.pseudo_firth_iteration_count for result in results], axis=0),
        nr_zero_start_iteration_count=jnp.stack([result.nr_zero_start_iteration_count for result in results], axis=0),
        nr_warm_start_iteration_count=jnp.stack([result.nr_warm_start_iteration_count for result in results], axis=0),
    )
