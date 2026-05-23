"""Binary result constructors for REGENIE step 2 compute."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute.regenie2_binary import types as regenie2_binary_types


def build_empty_firth_integer_array(extra_code: jax.Array) -> jax.Array:
    """Build an integer Firth diagnostic array for score-test-only results."""
    return jnp.zeros_like(extra_code, dtype=jnp.int32)


def build_empty_firth_boolean_array(extra_code: jax.Array) -> jax.Array:
    """Build a boolean Firth diagnostic array for score-test-only results."""
    return jnp.zeros_like(extra_code, dtype=jnp.bool_)


def build_binary_score_test_chunk_result(
    *,
    beta: jax.Array,
    standard_error: jax.Array,
    chi_squared: jax.Array,
    log10_p_value: jax.Array,
    extra_code: jax.Array,
    valid_mask: jax.Array,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Build a single-trait binary score-test chunk result with empty Firth diagnostics."""
    return regenie2_binary_types.Regenie2BinaryChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        valid_mask=valid_mask,
        firth_iteration_count=build_empty_firth_integer_array(extra_code),
        firth_failure_code=build_empty_firth_integer_array(extra_code),
        firth_convergence_reason_code=build_empty_firth_integer_array(extra_code),
        firth_correction_code=build_empty_firth_integer_array(extra_code),
        firth_sparse_correction_mask=build_empty_firth_boolean_array(extra_code),
        pseudo_firth_iteration_count=build_empty_firth_integer_array(extra_code),
        nr_zero_start_iteration_count=build_empty_firth_integer_array(extra_code),
        nr_warm_start_iteration_count=build_empty_firth_integer_array(extra_code),
    )


def build_multi_binary_score_test_chunk_result(
    *,
    beta: jax.Array,
    standard_error: jax.Array,
    chi_squared: jax.Array,
    log10_p_value: jax.Array,
    extra_code: jax.Array,
    valid_mask: jax.Array,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Build a multi-trait binary score-test chunk result with empty Firth diagnostics."""
    return regenie2_binary_types.Regenie2MultiBinaryChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        valid_mask=valid_mask,
        firth_iteration_count=build_empty_firth_integer_array(extra_code),
        firth_failure_code=build_empty_firth_integer_array(extra_code),
        firth_convergence_reason_code=build_empty_firth_integer_array(extra_code),
        firth_correction_code=build_empty_firth_integer_array(extra_code),
        firth_sparse_correction_mask=build_empty_firth_boolean_array(extra_code),
        pseudo_firth_iteration_count=build_empty_firth_integer_array(extra_code),
        nr_zero_start_iteration_count=build_empty_firth_integer_array(extra_code),
        nr_warm_start_iteration_count=build_empty_firth_integer_array(extra_code),
    )


def build_multi_binary_chunk_result(
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Rewrap a batched single-trait binary result as a multi-trait result."""
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


def squeeze_single_binary_chunk_result(
    result: regenie2_binary_types.Regenie2MultiBinaryChunkResult,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Remove the trait axis from a single-trait binary result."""
    return regenie2_binary_types.Regenie2BinaryChunkResult(
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
    results: list[regenie2_binary_types.Regenie2BinaryChunkResult],
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Stack per-trait binary chunk results into a trait-major result."""
    return regenie2_binary_types.Regenie2MultiBinaryChunkResult(
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
