"""Diagnostic constants and counters for REGENIE step 2 binary results."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from g import types

if typing.TYPE_CHECKING:
    import collections.abc

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
    result: (
        regenie2_binary_result.Regenie2BinaryScoreChunkResult
        | regenie2_binary_result.Regenie2BinaryChunkResult
        | regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult
        | regenie2_binary_result.Regenie2MultiBinaryChunkResult
    ),
) -> BinaryChunkDiagnostics:
    """Count diagnostic categories for one binary result chunk."""
    empty_integer_array = jnp.zeros_like(result.extra_code, dtype=jnp.int32)
    empty_boolean_array = jnp.zeros_like(result.extra_code, dtype=jnp.bool_)
    extra_code = jnp.ravel(result.extra_code)
    firth_iteration_count = jnp.ravel(getattr(result, "firth_iteration_count", empty_integer_array))
    firth_failure_code = jnp.ravel(getattr(result, "firth_failure_code", empty_integer_array))
    firth_correction_code = jnp.ravel(getattr(result, "firth_correction_code", empty_integer_array))
    firth_sparse_correction_mask = jnp.ravel(getattr(result, "firth_sparse_correction_mask", empty_boolean_array))
    pseudo_firth_iteration_count = jnp.ravel(getattr(result, "pseudo_firth_iteration_count", empty_integer_array))
    nr_zero_start_iteration_count = jnp.ravel(getattr(result, "nr_zero_start_iteration_count", empty_integer_array))
    nr_warm_start_iteration_count = jnp.ravel(getattr(result, "nr_warm_start_iteration_count", empty_integer_array))
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


def binary_chunk_diagnostics_to_mapping(diagnostics: BinaryChunkDiagnostics) -> dict[str, int | float]:
    """Materialize binary chunk diagnostics as JSON-ready counters."""
    diagnostics_on_host = jax.device_get(diagnostics)
    return {
        "score_only_count": int(diagnostics_on_host.score_only_count),
        "score_test_candidate_count": int(diagnostics_on_host.score_test_candidate_count),
        "firth_candidate_count": int(diagnostics_on_host.firth_candidate_count),
        "firth_iteration_min": int(diagnostics_on_host.firth_iteration_min),
        "firth_iteration_median": float(diagnostics_on_host.firth_iteration_median),
        "firth_iteration_max": int(diagnostics_on_host.firth_iteration_max),
        "firth_converged_count": int(diagnostics_on_host.firth_converged_count),
        "firth_failed_count": int(diagnostics_on_host.firth_failed_count),
        "firth_numerical_failure_count": int(diagnostics_on_host.firth_numerical_failure_count),
        "firth_max_iteration_failure_count": int(diagnostics_on_host.firth_max_iteration_failure_count),
        "firth_invalid_statistic_failure_count": int(diagnostics_on_host.firth_invalid_statistic_failure_count),
        "firth_step_halving_failure_count": int(diagnostics_on_host.firth_step_halving_failure_count),
        "pseudo_firth_attempt_count": int(diagnostics_on_host.pseudo_firth_attempt_count),
        "pseudo_firth_success_count": int(diagnostics_on_host.pseudo_firth_success_count),
        "nr_zero_start_attempt_count": int(diagnostics_on_host.nr_zero_start_attempt_count),
        "nr_zero_start_success_count": int(diagnostics_on_host.nr_zero_start_success_count),
        "nr_warm_start_attempt_count": int(diagnostics_on_host.nr_warm_start_attempt_count),
        "nr_warm_start_success_count": int(diagnostics_on_host.nr_warm_start_success_count),
        "sparse_correction_count": int(diagnostics_on_host.sparse_correction_count),
        "dense_correction_count": int(diagnostics_on_host.dense_correction_count),
    }


def binary_chunk_diagnostics_to_summary_counts(
    diagnostics_batch: collections.abc.Sequence[BinaryChunkDiagnostics],
) -> BinaryCorrectionSummaryCounts:
    """Materialize binary diagnostics as one aggregate summary counter payload."""
    diagnostics_on_host_batch = jax.device_get(tuple(diagnostics_batch))
    return BinaryCorrectionSummaryCounts(
        chunk_count=len(diagnostics_on_host_batch),
        score_only_count=sum(
            int(diagnostics_on_host.score_only_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        score_test_candidate_count=sum(
            int(diagnostics_on_host.score_test_candidate_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        firth_candidate_count=sum(
            int(diagnostics_on_host.firth_candidate_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        firth_converged_count=sum(
            int(diagnostics_on_host.firth_converged_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        firth_failed_count=sum(
            int(diagnostics_on_host.firth_failed_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        firth_numerical_failure_count=sum(
            int(diagnostics_on_host.firth_numerical_failure_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        firth_max_iteration_failure_count=sum(
            int(diagnostics_on_host.firth_max_iteration_failure_count)
            for diagnostics_on_host in diagnostics_on_host_batch
        ),
        firth_invalid_statistic_failure_count=sum(
            int(diagnostics_on_host.firth_invalid_statistic_failure_count)
            for diagnostics_on_host in diagnostics_on_host_batch
        ),
        firth_step_halving_failure_count=sum(
            int(diagnostics_on_host.firth_step_halving_failure_count)
            for diagnostics_on_host in diagnostics_on_host_batch
        ),
        pseudo_firth_attempt_count=sum(
            int(diagnostics_on_host.pseudo_firth_attempt_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        pseudo_firth_success_count=sum(
            int(diagnostics_on_host.pseudo_firth_success_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        nr_zero_start_attempt_count=sum(
            int(diagnostics_on_host.nr_zero_start_attempt_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        nr_zero_start_success_count=sum(
            int(diagnostics_on_host.nr_zero_start_success_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        nr_warm_start_attempt_count=sum(
            int(diagnostics_on_host.nr_warm_start_attempt_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        nr_warm_start_success_count=sum(
            int(diagnostics_on_host.nr_warm_start_success_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        sparse_correction_count=sum(
            int(diagnostics_on_host.sparse_correction_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
        dense_correction_count=sum(
            int(diagnostics_on_host.dense_correction_count) for diagnostics_on_host in diagnostics_on_host_batch
        ),
    )
