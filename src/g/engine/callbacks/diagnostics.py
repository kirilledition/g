"""Binary callback diagnostics and null-logistic policy helpers."""

from __future__ import annotations

import typing

import jax
import numpy as np
import numpy.typing as npt

from g import _core, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.engine.callbacks import events, timing

if typing.TYPE_CHECKING:
    import collections.abc


def block_until_ready(value: typing.Any) -> None:
    """Synchronize a JAX value or pytree."""
    jax.block_until_ready(value)


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
    native_policy_plan = native_callback_diagnostics_policy().plan_null_logistic_nonconvergence_from_array(
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
    events.native_pipeline_diagnostic_policy().record_callback_null_logistic_nonconvergence_warning_diagnostic_event(
        message=warning_message,
        chromosome=chromosome,
        nonconverged_count=native_policy_plan.nonconverged_count,
        phenotype_count=0 if phenotype_names is None else len(phenotype_names),
        policy=policy.value,
        scalar_convergence=native_policy_plan.scalar_convergence,
        total_fit_count=native_policy_plan.total_fit_count,
    )
    return native_policy_plan


def native_callback_diagnostics_policy() -> _core.NativeCallbackDiagnosticsPolicy:
    """Build the native callback diagnostics policy handle."""
    return _core.NativeCallbackDiagnosticsPolicy()


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
    stage_timing_recorder.add_binary_chunk_diagnostics(binary_chunk_diagnostics_to_mapping(diagnostics))


def binary_chunk_diagnostics_to_mapping(diagnostics: regenie2_binary.BinaryChunkDiagnostics) -> dict[str, int | float]:
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
    diagnostics_batch: collections.abc.Sequence[regenie2_binary.BinaryChunkDiagnostics],
) -> regenie2_binary.BinaryCorrectionSummaryCounts:
    """Materialize binary diagnostics as one aggregate summary counter payload."""
    diagnostics_on_host_batch = jax.device_get(tuple(diagnostics_batch))
    return regenie2_binary.BinaryCorrectionSummaryCounts(
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
    "binary_chunk_diagnostics_to_mapping",
    "binary_chunk_diagnostics_to_summary_counts",
    "block_until_ready",
    "collect_binary_chunk_diagnostics_if_needed",
    "enforce_null_logistic_nonconvergence_policy",
    "record_binary_chunk_diagnostics",
    "record_binary_chunk_diagnostics_from_count",
    "record_null_logistic_chromosome_diagnostics",
]
