"""Diagnostic constants and counters for REGENIE step 2 binary results."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from g import types
from g.compute.regenie2_binary import result as regenie2_binary_result


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BinaryChunkDiagnostics:
    """Diagnostic counts for one binary association chunk.

    Attributes:
        score_only_count: Variants that retained score-test statistics without correction.
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
        pseudo_firth_attempt_count: Candidates that attempted scalar pseudo-Firth.
        pseudo_firth_success_count: Candidates that finished through scalar pseudo-Firth.
        nr_zero_start_attempt_count: Candidates that attempted zero-start Newton-Raphson fallback.
        nr_zero_start_success_count: Candidates that finished through zero-start Newton-Raphson fallback.
        nr_warm_start_attempt_count: Candidates that attempted warm-start Newton-Raphson fallback.
        nr_warm_start_success_count: Candidates that finished through warm-start Newton-Raphson fallback.
        sparse_correction_count: Candidates corrected through carrier-only sparse inputs.
        dense_correction_count: Candidates corrected through dense inputs.

    """

    score_only_count: jax.Array
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
    pseudo_firth_attempt_count: jax.Array
    pseudo_firth_success_count: jax.Array
    nr_zero_start_attempt_count: jax.Array
    nr_zero_start_success_count: jax.Array
    nr_warm_start_attempt_count: jax.Array
    nr_warm_start_success_count: jax.Array
    sparse_correction_count: jax.Array
    dense_correction_count: jax.Array


@dataclass(frozen=True)
class BinaryCorrectionSummaryCounts:
    """Host integer counters needed by aggregate binary correction telemetry.

    Attributes:
        chunk_count: Chunks included in these aggregate counters.
        score_only_count: Variants that retained score-test statistics without correction.
        score_test_candidate_count: Variants selected for any score-test fallback label.
        firth_candidate_count: Variants with a nonzero Firth iteration count.
        firth_converged_count: Variants that completed Firth correction successfully.
        firth_failed_count: Variants labelled as failed candidate tests.
        firth_numerical_failure_count: Firth candidates that failed numerically.
        firth_max_iteration_failure_count: Firth candidates that hit the iteration limit.
        firth_invalid_statistic_failure_count: Firth candidates with invalid final statistics.
        firth_step_halving_failure_count: Firth candidates that exhausted step-halving attempts.
        pseudo_firth_attempt_count: Candidates that attempted scalar pseudo-Firth.
        pseudo_firth_success_count: Candidates that finished through scalar pseudo-Firth.
        nr_zero_start_attempt_count: Candidates that attempted zero-start Newton-Raphson fallback.
        nr_zero_start_success_count: Candidates that finished through zero-start Newton-Raphson fallback.
        nr_warm_start_attempt_count: Candidates that attempted warm-start Newton-Raphson fallback.
        nr_warm_start_success_count: Candidates that finished through warm-start Newton-Raphson fallback.
        sparse_correction_count: Candidates corrected through carrier-only sparse inputs.
        dense_correction_count: Candidates corrected through dense inputs.

    """

    chunk_count: int
    score_only_count: int
    score_test_candidate_count: int
    firth_candidate_count: int
    firth_converged_count: int
    firth_failed_count: int
    firth_numerical_failure_count: int
    firth_max_iteration_failure_count: int
    firth_invalid_statistic_failure_count: int
    firth_step_halving_failure_count: int
    pseudo_firth_attempt_count: int
    pseudo_firth_success_count: int
    nr_zero_start_attempt_count: int
    nr_zero_start_success_count: int
    nr_warm_start_attempt_count: int
    nr_warm_start_success_count: int
    sparse_correction_count: int
    dense_correction_count: int


def count_binary_chunk_diagnostics(
    result: regenie2_binary_result.Regenie2AnyBinaryChunkResult,
) -> BinaryChunkDiagnostics:
    """Count diagnostic categories for one binary result chunk."""
    diagnostic_result = regenie2_binary_result.expand_binary_result_with_empty_firth_diagnostics(result)
    extra_code = jnp.ravel(diagnostic_result.extra_code)
    firth_iteration_count = jnp.ravel(diagnostic_result.firth_iteration_count)
    firth_failure_code = jnp.ravel(diagnostic_result.firth_failure_code)
    firth_correction_code = jnp.ravel(diagnostic_result.firth_correction_code)
    firth_sparse_correction_mask = jnp.ravel(diagnostic_result.firth_sparse_correction_mask)
    pseudo_firth_iteration_count = jnp.ravel(diagnostic_result.pseudo_firth_iteration_count)
    nr_zero_start_iteration_count = jnp.ravel(diagnostic_result.nr_zero_start_iteration_count)
    nr_warm_start_iteration_count = jnp.ravel(diagnostic_result.nr_warm_start_iteration_count)
    firth_attempt_mask = firth_iteration_count > 0
    firth_candidate_count = jnp.sum(firth_attempt_mask, dtype=jnp.int32)
    finite_iteration_count = jnp.where(firth_attempt_mask, firth_iteration_count, jnp.asarray(0, dtype=jnp.int32))
    sorted_active_iteration_count = jnp.sort(
        jnp.where(firth_attempt_mask, firth_iteration_count, np.iinfo(np.int32).max)
    )
    median_iteration_index = jnp.maximum((firth_candidate_count - 1) // 2, 0)
    return BinaryChunkDiagnostics(
        score_only_count=jnp.sum(extra_code == types.BinaryExtraCode.SCORE.value, dtype=jnp.int32),
        score_test_candidate_count=jnp.sum(
            (extra_code == types.BinaryExtraCode.FIRTH.value)
            | (extra_code == types.BinaryExtraCode.SPA.value)
            | (extra_code == types.BinaryExtraCode.TEST_FAIL.value),
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
        firth_converged_count=jnp.sum(
            (extra_code == types.BinaryExtraCode.FIRTH.value) & firth_attempt_mask,
            dtype=jnp.int32,
        ),
        firth_failed_count=jnp.sum(
            (extra_code == types.BinaryExtraCode.TEST_FAIL.value) & firth_attempt_mask,
            dtype=jnp.int32,
        ),
        firth_numerical_failure_count=jnp.sum(
            firth_failure_code == types.FirthFailureCode.NUMERICAL.value,
            dtype=jnp.int32,
        ),
        firth_max_iteration_failure_count=jnp.sum(
            firth_failure_code == types.FirthFailureCode.MAX_ITERATIONS.value,
            dtype=jnp.int32,
        ),
        firth_invalid_statistic_failure_count=jnp.sum(
            firth_failure_code == types.FirthFailureCode.INVALID_STATISTIC.value,
            dtype=jnp.int32,
        ),
        firth_step_halving_failure_count=jnp.sum(
            firth_failure_code == types.FirthFailureCode.STEP_HALVING.value,
            dtype=jnp.int32,
        ),
        pseudo_firth_attempt_count=jnp.sum(pseudo_firth_iteration_count > 0, dtype=jnp.int32),
        pseudo_firth_success_count=jnp.sum(
            firth_correction_code == types.FirthCorrectionCode.PSEUDO_FIRTH.value,
            dtype=jnp.int32,
        ),
        nr_zero_start_attempt_count=jnp.sum(nr_zero_start_iteration_count > 0, dtype=jnp.int32),
        nr_zero_start_success_count=jnp.sum(
            firth_correction_code == types.FirthCorrectionCode.NEWTON_RAPHSON_ZERO_START.value,
            dtype=jnp.int32,
        ),
        nr_warm_start_attempt_count=jnp.sum(nr_warm_start_iteration_count > 0, dtype=jnp.int32),
        nr_warm_start_success_count=jnp.sum(
            firth_correction_code == types.FirthCorrectionCode.NEWTON_RAPHSON_WARM_START.value,
            dtype=jnp.int32,
        ),
        sparse_correction_count=jnp.sum(firth_sparse_correction_mask & firth_attempt_mask, dtype=jnp.int32),
        dense_correction_count=jnp.sum((~firth_sparse_correction_mask) & firth_attempt_mask, dtype=jnp.int32),
    )
