"""Binary score-test correction-label selection for REGENIE step 2."""

from __future__ import annotations

import math
import typing

import jax
import jax.numpy as jnp

from g import types
from g.compute.regenie2_binary import result as regenie2_binary_result

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types


def validate_runtime_correction_plan(correction_plan: types.BinaryCorrectionPlan) -> None:
    """Validate binary correction methods accepted by runtime compute code."""
    if correction_plan.method in {
        types.BinaryFallbackMethod.SCORE_ONLY,
        types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
    }:
        return
    if correction_plan.method == types.BinaryFallbackMethod.FIRTH:
        message = "Unsupported binary correction method for runtime compute: exact Firth. Use approximate Firth."
        raise ValueError(message)
    if correction_plan.method == types.BinaryFallbackMethod.SPA:
        message = "Unsupported binary correction method for runtime compute: SPA. Omit SPA for score-test-only output."
        raise ValueError(message)
    message = f"Unsupported binary correction method for runtime compute: {correction_plan.method.value}."
    raise ValueError(message)


def build_extra_code(
    log10_p_value: jax.Array,
    valid_mask: jax.Array,
    correction_plan: types.BinaryCorrectionPlan,
) -> jax.Array:
    """Select correction labels from score-test statistics."""
    validate_runtime_correction_plan(correction_plan)
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        candidate_mask = jnp.zeros_like(valid_mask, dtype=jnp.bool_)
        correction_code = types.BinaryExtraCode.SCORE.value
    elif correction_plan.method == types.BinaryFallbackMethod.FIRTH_APPROXIMATE:
        fallback_log10p_threshold = -math.log10(correction_plan.p_threshold)
        candidate_mask = log10_p_value > fallback_log10p_threshold
        correction_code = types.BinaryExtraCode.FIRTH.value
    else:
        message = f"Unsupported binary correction method: {correction_plan.method.value}."
        raise ValueError(message)
    return jnp.where(
        valid_mask,
        jnp.where(candidate_mask, correction_code, types.BinaryExtraCode.SCORE.value),
        types.BinaryExtraCode.TEST_FAIL.value,
    ).astype(jnp.int32)


def merge_firth_variant_result_into_chunk(
    *,
    result: regenie2_binary_result.Regenie2BinaryChunkResult,
    firth_result: regenie2_binary_firth_types.FirthVariantResult,
    active_flat_positions: jax.Array,
    active_fallback_indices: jax.Array,
    genotype_flip_mask: jax.Array,
    firth_se: bool,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Merge active Firth candidate results into a binary chunk result."""
    active_valid_mask = firth_result.valid_mask[active_flat_positions]
    active_firth_beta = jnp.where(
        genotype_flip_mask[active_flat_positions],
        -firth_result.beta[active_flat_positions],
        firth_result.beta[active_flat_positions],
    )
    active_firth_chi_squared = firth_result.chi_squared[active_flat_positions]
    active_firth_standard_error = firth_result.standard_error[active_flat_positions]
    invalid_firth_statistic = jnp.full_like(active_firth_beta, jnp.nan)
    if firth_se:
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
        active_valid_mask,
        types.BinaryExtraCode.FIRTH.value,
        types.BinaryExtraCode.TEST_FAIL.value,
    ).astype(jnp.int32)
    return regenie2_binary_result.Regenie2BinaryChunkResult(
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
