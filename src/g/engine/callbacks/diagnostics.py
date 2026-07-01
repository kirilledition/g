"""Binary callback diagnostics and null-logistic policy helpers."""

from __future__ import annotations

import typing

import jax
import numpy as np
import numpy.typing as npt

from g import _core, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.engine import timing


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
) -> _core.NativeNullLogisticNonconvergencePlan:
    """Raise or warn when a binary null-logistic chromosome fit did not converge."""
    convergence_flags = np.asarray(jax.device_get(null_logistic_converged), dtype=np.bool_)
    return enforce_host_null_logistic_nonconvergence_policy(
        chromosome=chromosome,
        convergence_flags=convergence_flags,
        policy=policy,
        phenotype_names=phenotype_names,
    )


def record_null_logistic_chromosome_diagnostics(
    *,
    chromosome: str,
    null_logistic_converged: typing.Any,
    null_logistic_iteration_count: typing.Any,
    null_firth_iteration_count: typing.Any | None,
    null_firth_convergence_reason_code: typing.Any | None,
    policy: types.NullLogisticNonconvergencePolicy,
    phenotype_names: tuple[str, ...] | None,
    correction_method: types.BinaryFallbackMethod,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> int:
    """Apply null-logistic policy and record native timing diagnostics."""
    host_value_requests: dict[str, typing.Any] = {"converged": null_logistic_converged}
    if stage_timing_recorder is not None:
        host_value_requests["iteration_count"] = null_logistic_iteration_count
        if phenotype_names is None:
            if null_firth_iteration_count is None or null_firth_convergence_reason_code is None:
                raise RuntimeError("Scalar null-logistic diagnostics require null Firth diagnostic values.")
            host_value_requests["firth_iteration_count"] = null_firth_iteration_count
            host_value_requests["firth_convergence_reason_code"] = null_firth_convergence_reason_code

    host_values = typing.cast("dict[str, object]", jax.device_get(host_value_requests))
    convergence_flags = np.asarray(host_values["converged"], dtype=np.bool_)
    native_policy_plan = enforce_host_null_logistic_nonconvergence_policy(
        chromosome=chromosome,
        convergence_flags=convergence_flags,
        policy=policy,
        phenotype_names=phenotype_names,
    )
    if stage_timing_recorder is not None:
        iteration_counts = np.asarray(host_values["iteration_count"], dtype=np.int64)
        if phenotype_names is None:
            stage_timing_recorder.add_scalar_null_logistic_diagnostics_from_arrays(
                chromosome=chromosome,
                convergence_values=convergence_flags,
                iteration_count_values=iteration_counts,
                firth_iteration_count_values=np.asarray(host_values["firth_iteration_count"], dtype=np.int64),
                firth_convergence_reason_code_values=np.asarray(
                    host_values["firth_convergence_reason_code"],
                    dtype=np.int64,
                ),
                correction_method=correction_method.value,
            )
        else:
            stage_timing_recorder.add_multi_null_logistic_diagnostics_from_arrays(
                chromosome=chromosome,
                convergence_values=convergence_flags,
                iteration_count_values=iteration_counts,
                phenotype_names=phenotype_names,
                correction_method=correction_method.value,
            )
    return native_policy_plan.nonconverged_count


def enforce_host_null_logistic_nonconvergence_policy(
    *,
    chromosome: str,
    convergence_flags: npt.NDArray[np.bool_],
    policy: types.NullLogisticNonconvergencePolicy,
    phenotype_names: tuple[str, ...] | None,
) -> _core.NativeNullLogisticNonconvergencePlan:
    """Raise or warn using already materialized null-logistic convergence flags."""
    native_policy_plan = _core.plan_null_logistic_nonconvergence_from_array(
        chromosome=chromosome,
        convergence_values=convergence_flags,
        phenotype_names=phenotype_names,
        policy=policy.value,
    )
    if native_policy_plan.action == "continue":
        return native_policy_plan
    if native_policy_plan.action == "fail":
        message = native_policy_plan.message
        if message is None:
            raise RuntimeError("Native null-logistic nonconvergence fail plan did not include a message.")
        raise RuntimeError(message)
    warning_message = native_policy_plan.warning_message
    if warning_message is None:
        raise RuntimeError("Native null-logistic nonconvergence warning plan did not include a warning message.")
    _core.record_callback_null_logistic_nonconvergence_warning_diagnostic_event(
        message=warning_message,
        chromosome=chromosome,
        nonconverged_count=native_policy_plan.nonconverged_count,
        phenotype_count=0 if phenotype_names is None else len(phenotype_names),
        policy=policy.value,
        scalar_convergence=native_policy_plan.scalar_convergence,
        total_fit_count=native_policy_plan.total_fit_count,
    )
    return native_policy_plan


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
    "record_null_logistic_chromosome_diagnostics",
]
