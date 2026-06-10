"""Binary callback diagnostics and null-logistic policy helpers."""

from __future__ import annotations

import typing

import jax
import numpy as np

import g.engine.callbacks.shared as shared
from g import types
from g.compute.regenie2_binary import api as regenie2_binary
from g.engine import timing

logger = shared.logger


def block_until_ready(value: typing.Any) -> None:
    """Synchronize a JAX value when it supports readiness blocking."""
    block_until_ready_method = getattr(value, "block_until_ready", None)
    if callable(block_until_ready_method):
        block_until_ready_method()


def enforce_null_logistic_nonconvergence_policy(
    *,
    chromosome: str,
    null_logistic_converged: typing.Any,
    policy: types.NullLogisticNonconvergencePolicy,
    phenotype_names: tuple[str, ...] | None = None,
) -> None:
    """Raise or warn when a binary null-logistic chromosome fit did not converge."""
    convergence_flags = np.asarray(jax.device_get(null_logistic_converged), dtype=np.bool_)
    if convergence_flags.ndim == 0:
        if bool(convergence_flags):
            return
        message = f"Binary null logistic model did not converge for chromosome {chromosome}."
    else:
        failed_trait_indices = tuple(int(index) for index in np.flatnonzero(~convergence_flags))
        if not failed_trait_indices:
            return
        if phenotype_names is None:
            failed_traits = ", ".join(str(index) for index in failed_trait_indices)
        else:
            failed_traits = ", ".join(phenotype_names[index] for index in failed_trait_indices)
        message = f"Binary null logistic model did not converge for chromosome {chromosome}: {failed_traits}."
    if policy == types.NullLogisticNonconvergencePolicy.FAIL:
        raise RuntimeError(message)
    logger.warning("%s Continuing because --null_logistic_nonconvergence_policy=warn.", message)


def record_binary_chunk_diagnostics(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    result: regenie2_binary.Regenie2BinaryScoreChunkResult | regenie2_binary.Regenie2BinaryChunkResult,
) -> None:
    """Record binary candidate and Firth diagnostics for one chunk."""
    if not timing.should_collect_exact_stage_timings(stage_timing_recorder):
        return
    record_binary_chunk_diagnostics_from_count(
        stage_timing_recorder=stage_timing_recorder,
        diagnostics=regenie2_binary.count_binary_chunk_diagnostics(result),
    )


def record_binary_chunk_diagnostics_from_count(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    diagnostics: regenie2_binary.BinaryChunkDiagnostics | None,
) -> None:
    """Record binary diagnostics that were already counted on-device."""
    if diagnostics is None:
        return
    if not timing.should_collect_exact_stage_timings(stage_timing_recorder):
        return
    assert stage_timing_recorder is not None
    diagnostics_on_host = jax.device_get(diagnostics)
    stage_timing_recorder.add_binary_chunk_diagnostics(
        {
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
    )


def collect_binary_chunk_diagnostics_if_needed(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    result: regenie2_binary.Regenie2BinaryScoreChunkResult | regenie2_binary.Regenie2BinaryChunkResult,
) -> regenie2_binary.BinaryChunkDiagnostics | None:
    """Collect binary chunk diagnostics only when exact stage timings are enabled."""
    if not timing.should_collect_exact_stage_timings(stage_timing_recorder):
        return None
    return regenie2_binary.count_binary_chunk_diagnostics(result)


__all__ = [
    "block_until_ready",
    "collect_binary_chunk_diagnostics_if_needed",
    "enforce_null_logistic_nonconvergence_policy",
    "record_binary_chunk_diagnostics",
    "record_binary_chunk_diagnostics_from_count",
]
