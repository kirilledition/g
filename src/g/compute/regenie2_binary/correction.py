"""Binary score-test correction-label selection for REGENIE step 2."""

from __future__ import annotations

import math
import typing

import jax
import jax.numpy as jnp

from g import types


def build_extra_code(
    log10_p_value: jax.Array,
    valid_mask: jax.Array,
    correction_plan: types.BinaryCorrectionPlan,
) -> jax.Array:
    """Select correction labels from score-test statistics."""
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        candidate_mask = jnp.zeros_like(valid_mask, dtype=jnp.bool_)
        correction_code = types.BinaryExtraCode.SCORE.value
    elif correction_plan.method == types.BinaryFallbackMethod.FIRTH_APPROXIMATE:
        fallback_log10p_threshold = -math.log10(correction_plan.p_threshold)
        candidate_mask = log10_p_value > fallback_log10p_threshold
        correction_code = types.BinaryExtraCode.FIRTH.value
    elif correction_plan.method == types.BinaryFallbackMethod.FIRTH:
        message = "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx."
        raise NotImplementedError(message)
    elif correction_plan.method == types.BinaryFallbackMethod.SPA:
        message = "SPA fallback is not implemented yet. Omit --spa for score-test-only output."
        raise NotImplementedError(message)
    else:
        typing.assert_never(correction_plan.method)
    return jnp.where(
        valid_mask,
        jnp.where(candidate_mask, correction_code, types.BinaryExtraCode.SCORE.value),
        types.BinaryExtraCode.TEST_FAIL.value,
    ).astype(jnp.int32)
