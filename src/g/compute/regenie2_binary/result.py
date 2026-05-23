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
