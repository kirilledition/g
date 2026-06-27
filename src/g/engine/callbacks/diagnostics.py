"""Binary callback diagnostics and null-logistic policy helpers."""

from __future__ import annotations

import typing

import jax
import numpy as np

import g.engine.callbacks.shared as shared
from g import _core, types
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
    phenotype_names: tuple[str, ...] | None,
) -> None:
    """Raise or warn when a binary null-logistic chromosome fit did not converge."""
    convergence_flags = np.asarray(jax.device_get(null_logistic_converged), dtype=np.bool_)
    native_policy_plan = _core.plan_null_logistic_nonconvergence(
        chromosome=chromosome,
        convergence_flags=tuple(bool(flag) for flag in np.ravel(convergence_flags)),
        scalar_convergence=convergence_flags.ndim == 0,
        phenotype_names=phenotype_names,
        policy=policy.value,
    )
    if native_policy_plan.action == "continue":
        return
    if native_policy_plan.action == "fail":
        message = native_policy_plan.message
        if message is None:
            raise RuntimeError("Native null-logistic nonconvergence fail plan did not include a message.")
        raise RuntimeError(message)
    warning_message = native_policy_plan.warning_message
    if warning_message is None:
        raise RuntimeError("Native null-logistic nonconvergence warning plan did not include a warning message.")
    logger.warning("%s", warning_message)


def record_binary_chunk_diagnostics(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    result: (
        regenie2_binary.Regenie2BinaryScoreChunkResult
        | regenie2_binary.Regenie2BinaryChunkResult
        | regenie2_binary.Regenie2MultiBinaryScoreChunkResult
        | regenie2_binary.Regenie2MultiBinaryChunkResult
    ),
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
    stage_timing_recorder.add_binary_chunk_diagnostics(regenie2_binary.binary_chunk_diagnostics_to_mapping(diagnostics))


def collect_binary_chunk_diagnostics_if_needed(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    result: (
        regenie2_binary.Regenie2BinaryScoreChunkResult
        | regenie2_binary.Regenie2BinaryChunkResult
        | regenie2_binary.Regenie2MultiBinaryScoreChunkResult
        | regenie2_binary.Regenie2MultiBinaryChunkResult
    ),
) -> regenie2_binary.BinaryChunkDiagnostics | None:
    """Collect binary chunk diagnostics for summary telemetry and optional exact timings."""
    del stage_timing_recorder
    return regenie2_binary.count_binary_chunk_diagnostics(result)


__all__ = [
    "block_until_ready",
    "collect_binary_chunk_diagnostics_if_needed",
    "enforce_null_logistic_nonconvergence_policy",
    "record_binary_chunk_diagnostics",
    "record_binary_chunk_diagnostics_from_count",
]
