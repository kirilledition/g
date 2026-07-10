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


def build_correction_code(
    log10_p_value: jax.Array,
    valid_mask: jax.Array,
    correction_plan: types.BinaryCorrectionPlan,
) -> jax.Array:
    """Select correction labels from score-test statistics."""
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        candidate_mask = jnp.zeros_like(valid_mask, dtype=jnp.bool_)
        correction_code = types.BinaryCorrectionCode.SCORE_SUCCESS.value
    else:
        fallback_log10p_threshold = -math.log10(correction_plan.p_threshold)
        candidate_mask = log10_p_value > fallback_log10p_threshold
        correction_code = types.BinaryCorrectionCode.FIRTH_SUCCESS.value
    return jnp.where(
        valid_mask,
        jnp.where(candidate_mask, correction_code, types.BinaryCorrectionCode.SCORE_SUCCESS.value),
        types.BinaryCorrectionCode.SCORE_FAILED.value,
    ).astype(jnp.int32)


def merge_firth_variant_result_into_multi_chunk(
    *,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    firth_result: regenie2_binary_firth_types.FirthVariantResult,
    active_flat_positions: jax.Array,
    active_trait_indices: jax.Array,
    active_variant_indices: jax.Array,
    genotype_flip_mask: jax.Array,
    firth_se: bool,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Merge active Firth candidate results into a multi-trait binary chunk result."""
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
    merged_correction_code = jnp.where(
        active_valid_mask,
        types.BinaryCorrectionCode.FIRTH_SUCCESS.value,
        types.BinaryCorrectionCode.FIRTH_FAILED.value,
    ).astype(jnp.int32)
    return regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult(
        beta=result.beta.at[active_trait_indices, active_variant_indices].set(
            jnp.asarray(merged_beta, dtype=result.beta.dtype)
        ),
        standard_error=result.standard_error.at[active_trait_indices, active_variant_indices].set(
            jnp.asarray(merged_standard_error, dtype=result.standard_error.dtype)
        ),
        chi_squared=result.chi_squared.at[active_trait_indices, active_variant_indices].set(
            jnp.asarray(merged_chi_squared, dtype=result.chi_squared.dtype)
        ),
        log10_p_value=result.log10_p_value.at[active_trait_indices, active_variant_indices].set(
            jnp.asarray(merged_log10_p_value, dtype=result.log10_p_value.dtype)
        ),
        correction_code=result.correction_code.at[active_trait_indices, active_variant_indices].set(
            merged_correction_code
        ),
    )
