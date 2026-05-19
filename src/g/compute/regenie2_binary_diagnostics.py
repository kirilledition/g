"""Diagnostic constants and counters for REGENIE step 2 binary results."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from g import types

if typing.TYPE_CHECKING:
    import g.compute.regenie2_binary_types as regenie2_binary_types


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BinaryChunkDiagnostics:
    """Diagnostic counts for one binary association chunk.

    Attributes:
        score_test_candidate_count: Variants selected for any score-test fallback label.
        firth_candidate_count: Variants with a nonzero Firth iteration count.
        firth_iteration_min: Minimum Firth iteration count among attempted candidates.
        firth_iteration_median: Median Firth iteration count among attempted candidates.
        firth_iteration_max: Maximum Firth iteration count among attempted candidates.
        firth_converged_count: Variants that completed Firth correction successfully.
        firth_failed_count: Variants labelled as failed candidate tests.
        firth_numerical_failure_count: Firth candidates that failed numerically.
        firth_max_iteration_failure_count: Firth candidates that hit the iteration limit.
        firth_invalid_statistic_failure_count: Firth candidates with invalid final statistics.
        firth_step_halving_failure_count: Firth candidates that exhausted step-halving attempts.

    """

    score_test_candidate_count: jax.Array
    firth_candidate_count: jax.Array
    firth_iteration_min: jax.Array
    firth_iteration_median: jax.Array
    firth_iteration_max: jax.Array
    firth_converged_count: jax.Array
    firth_failed_count: jax.Array
    firth_numerical_failure_count: jax.Array
    firth_max_iteration_failure_count: jax.Array
    firth_invalid_statistic_failure_count: jax.Array
    firth_step_halving_failure_count: jax.Array


def count_binary_chunk_diagnostics(
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
) -> BinaryChunkDiagnostics:
    """Count diagnostic categories for one binary result chunk."""
    firth_iteration_count = result.firth_iteration_count
    firth_attempt_mask = firth_iteration_count > 0
    firth_candidate_count = jnp.sum(firth_attempt_mask, dtype=jnp.int32)
    finite_iteration_count = jnp.where(firth_attempt_mask, firth_iteration_count, jnp.asarray(0, dtype=jnp.int32))
    sorted_active_iteration_count = jnp.sort(
        jnp.where(firth_attempt_mask, firth_iteration_count, np.iinfo(np.int32).max)
    )
    median_iteration_index = jnp.maximum((firth_candidate_count - 1) // 2, 0)
    return BinaryChunkDiagnostics(
        score_test_candidate_count=jnp.sum(
            (result.extra_code == types.BinaryExtraCode.FIRTH.value)
            | (result.extra_code == types.BinaryExtraCode.SPA.value)
            | (result.extra_code == types.BinaryExtraCode.TEST_FAIL.value),
            dtype=jnp.int32,
        ),
        firth_candidate_count=firth_candidate_count,
        firth_iteration_min=jnp.where(
            firth_candidate_count > 0,
            sorted_active_iteration_count[0],
            jnp.asarray(0, dtype=jnp.int32),
        ),
        firth_iteration_median=jnp.where(
            firth_candidate_count > 0,
            sorted_active_iteration_count[median_iteration_index],
            jnp.asarray(0, dtype=jnp.int32),
        ),
        firth_iteration_max=jnp.max(finite_iteration_count),
        firth_converged_count=jnp.sum(result.extra_code == types.BinaryExtraCode.FIRTH.value, dtype=jnp.int32),
        firth_failed_count=jnp.sum(result.extra_code == types.BinaryExtraCode.TEST_FAIL.value, dtype=jnp.int32),
        firth_numerical_failure_count=jnp.sum(
            result.firth_failure_code == types.FirthFailureCode.NUMERICAL.value,
            dtype=jnp.int32,
        ),
        firth_max_iteration_failure_count=jnp.sum(
            result.firth_failure_code == types.FirthFailureCode.MAX_ITERATIONS.value,
            dtype=jnp.int32,
        ),
        firth_invalid_statistic_failure_count=jnp.sum(
            result.firth_failure_code == types.FirthFailureCode.INVALID_STATISTIC.value,
            dtype=jnp.int32,
        ),
        firth_step_halving_failure_count=jnp.sum(
            result.firth_failure_code == types.FirthFailureCode.STEP_HALVING.value,
            dtype=jnp.int32,
        ),
    )
