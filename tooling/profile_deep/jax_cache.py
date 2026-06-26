"""Deep-profile JAX cache helpers."""

from __future__ import annotations

import dataclasses
import os
import statistics
import typing
from pathlib import Path

from tooling.common import jax_cache as tooling_jax_cache
from tooling.profile_deep import models as profile_deep_models

GPU_JAX_CACHE_PARENT_DEFAULT = "/tmp/g-jax-profile-cache"
JAX_LOG_SAMPLE_LINE_LIMIT = 20


def resolve_profile_jax_cache_directory(
    candidate: profile_deep_models.Step2Candidate,
    cache_directory: Path | None,
) -> Path | None:
    """Resolve the actual JAX cache directory used by one profile child."""
    if cache_directory is None:
        return None
    if candidate.device != "gpu":
        return tooling_jax_cache.resolve_cpu_feature_aware_cache_directory(cache_directory)
    job_identifier = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    gpu_cache_parent = os.environ.get("G_PROFILE_GPU_JAX_CACHE_PARENT", GPU_JAX_CACHE_PARENT_DEFAULT)
    return Path(gpu_cache_parent) / job_identifier / cache_directory.name


def collect_jax_cache_snapshot(cache_directory: Path | None) -> profile_deep_models.JaxCacheSnapshot | None:
    """Collect lightweight file-count and byte-size stats for a JAX cache directory."""
    if cache_directory is None:
        return None
    resolved_cache_directory = cache_directory.expanduser()
    if not resolved_cache_directory.exists():
        return profile_deep_models.JaxCacheSnapshot(
            path=str(resolved_cache_directory),
            exists=False,
            file_count=0,
            total_size_bytes=0,
        )
    file_count = 0
    total_size_bytes = 0
    try:
        for cache_path in resolved_cache_directory.rglob("*"):
            if not cache_path.is_file():
                continue
            file_count += 1
            total_size_bytes += cache_path.stat().st_size
    except OSError as error:
        return profile_deep_models.JaxCacheSnapshot(
            path=str(resolved_cache_directory),
            exists=True,
            file_count=file_count,
            total_size_bytes=total_size_bytes,
            error=str(error),
        )
    return profile_deep_models.JaxCacheSnapshot(
        path=str(resolved_cache_directory),
        exists=True,
        file_count=file_count,
        total_size_bytes=total_size_bytes,
    )


def parse_jax_compile_log(log_text: str) -> profile_deep_models.JaxCompileLogSummary:
    """Parse supported JAX compile and persistent-cache log lines from stderr text."""
    compilation_event_count = 0
    persistent_cache_event_count = 0
    persistent_cache_hit_count = 0
    persistent_cache_miss_count = 0
    tracing_cache_miss_count = 0
    cache_miss_explanation_count = 0
    sample_log_lines: list[str] = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        lower_line = line.lower()
        has_compilation_event = (
            "compiling " in lower_line
            or "finished xla compilation" in lower_line
            or "tracing + transforming" in lower_line
            or "lowering " in lower_line
        )
        has_persistent_cache_event = "persistent compilation cache" in lower_line
        has_persistent_cache_hit = has_persistent_cache_event and (
            "cache hit" in lower_line or "cache_hit" in lower_line
        )
        has_persistent_cache_miss = has_persistent_cache_event and (
            "cache miss" in lower_line or "cache_miss" in lower_line or "not found" in lower_line
        )
        has_tracing_cache_miss = "tracing cache miss" in lower_line
        if has_compilation_event:
            compilation_event_count += 1
        if has_persistent_cache_event:
            persistent_cache_event_count += 1
        if has_persistent_cache_hit:
            persistent_cache_hit_count += 1
        if has_persistent_cache_miss:
            persistent_cache_miss_count += 1
        if has_tracing_cache_miss:
            tracing_cache_miss_count += 1
        if (has_persistent_cache_miss or has_tracing_cache_miss) and (
            "because" in lower_line or "explain" in lower_line
        ):
            cache_miss_explanation_count += 1
        if (
            has_compilation_event
            or has_persistent_cache_event
            or has_persistent_cache_hit
            or has_persistent_cache_miss
            or has_tracing_cache_miss
        ) and len(sample_log_lines) < JAX_LOG_SAMPLE_LINE_LIMIT:
            sample_log_lines.append(line[:500])
    return profile_deep_models.JaxCompileLogSummary(
        compilation_event_count=compilation_event_count,
        persistent_cache_event_count=persistent_cache_event_count,
        persistent_cache_hit_count=persistent_cache_hit_count,
        persistent_cache_miss_count=persistent_cache_miss_count,
        tracing_cache_miss_count=tracing_cache_miss_count,
        cache_miss_explanation_count=cache_miss_explanation_count,
        sample_log_lines=sample_log_lines,
    )


def read_jax_compile_log_summary(stderr_log_path: str) -> profile_deep_models.JaxCompileLogSummary:
    """Read a subprocess stderr file and parse JAX compile/cache log counters."""
    log_path = Path(stderr_log_path)
    if not log_path.exists():
        return parse_jax_compile_log("")
    return parse_jax_compile_log(log_path.read_text(encoding="utf-8", errors="replace"))


def snapshot_delta(
    *,
    before_snapshot: profile_deep_models.JaxCacheSnapshot | None,
    after_snapshot: profile_deep_models.JaxCacheSnapshot | None,
    field_name: str,
) -> int | None:
    """Return an integer delta for a cache snapshot field."""
    if before_snapshot is None or after_snapshot is None:
        return None
    before_value = getattr(before_snapshot, field_name)
    after_value = getattr(after_snapshot, field_name)
    if not isinstance(before_value, int) or not isinstance(after_value, int):
        return None
    return after_value - before_value


def build_jax_cache_diagnostics(
    *,
    cache_directory: Path | None,
    child_reported_cache_directory: str | None,
    persistent_cache_used: bool,
    before_snapshot: profile_deep_models.JaxCacheSnapshot | None,
    after_snapshot: profile_deep_models.JaxCacheSnapshot | None,
    stderr_log_path: str,
) -> profile_deep_models.JaxCacheDiagnostics:
    """Build one subprocess JAX cache diagnostic payload."""
    return profile_deep_models.JaxCacheDiagnostics(
        cache_directory=str(cache_directory) if cache_directory is not None else None,
        child_reported_cache_directory=child_reported_cache_directory,
        persistent_cache_used=persistent_cache_used,
        before=before_snapshot,
        after=after_snapshot,
        file_count_delta=snapshot_delta(
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            field_name="file_count",
        ),
        size_bytes_delta=snapshot_delta(
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            field_name="total_size_bytes",
        ),
        compile_log_summary=read_jax_compile_log_summary(stderr_log_path),
    )


def successful_trials_with_jax_diagnostics(
    trials: list[profile_deep_models.TrialResult],
) -> list[profile_deep_models.TrialResult]:
    """Return successful trials that include JAX cache diagnostics."""
    return [
        trial
        for trial in trials
        if trial.status == "success" and trial.wall_time_seconds is not None and trial.jax_cache_diagnostics is not None
    ]


def sum_optional_integer_values(values: typing.Iterable[int | None]) -> int | None:
    """Sum integer values when at least one value is available."""
    observed_values = [value for value in values if value is not None]
    if not observed_values:
        return None
    return sum(observed_values)


def compile_log_summary_for_trial(
    trial: profile_deep_models.TrialResult,
) -> profile_deep_models.JaxCompileLogSummary:
    """Return the parsed JAX compile log summary for a diagnostic trial."""
    if trial.jax_cache_diagnostics is None:
        return parse_jax_compile_log("")
    return trial.jax_cache_diagnostics.compile_log_summary


def build_jax_cold_warm_diagnostics(
    *,
    warmup_trials: list[profile_deep_models.TrialResult],
    trial_results: list[profile_deep_models.TrialResult],
) -> profile_deep_models.JaxColdWarmDiagnostics | None:
    """Build cold-versus-warm JAX diagnostics for one aggregate result."""
    successful_trials = successful_trials_with_jax_diagnostics([*warmup_trials, *trial_results])
    if not successful_trials:
        return None
    successful_diagnostics = [
        trial.jax_cache_diagnostics for trial in successful_trials if trial.jax_cache_diagnostics is not None
    ]
    cold_trial = successful_trials[0]
    warm_trials = successful_trials[1:]
    cold_diagnostics = successful_diagnostics[0]
    warm_wall_times = [trial.wall_time_seconds for trial in warm_trials if trial.wall_time_seconds is not None]
    warm_median_wall_time = statistics.median(warm_wall_times) if warm_wall_times else None
    warm_mean_wall_time = statistics.fmean(warm_wall_times) if warm_wall_times else None
    cold_to_warm_speedup_ratio = None
    if cold_trial.wall_time_seconds is not None and warm_median_wall_time is not None and warm_median_wall_time > 0.0:
        cold_to_warm_speedup_ratio = cold_trial.wall_time_seconds / warm_median_wall_time
    warm_diagnostics = [trial.jax_cache_diagnostics for trial in warm_trials if trial.jax_cache_diagnostics is not None]
    return profile_deep_models.JaxColdWarmDiagnostics(
        cache_directory=cold_diagnostics.cache_directory,
        persistent_cache_used=any(diagnostic.persistent_cache_used for diagnostic in successful_diagnostics),
        cold_trial_name=cold_trial.name,
        cold_wall_time_seconds=cold_trial.wall_time_seconds,
        warm_trial_count=len(warm_trials),
        warm_median_wall_time_seconds=warm_median_wall_time,
        warm_mean_wall_time_seconds=warm_mean_wall_time,
        cold_to_warm_speedup_ratio=cold_to_warm_speedup_ratio,
        cold_cache_file_count_delta=cold_diagnostics.file_count_delta,
        warm_cache_file_count_delta=sum_optional_integer_values(
            diagnostic.file_count_delta for diagnostic in warm_diagnostics
        ),
        cold_cache_size_bytes_delta=cold_diagnostics.size_bytes_delta,
        warm_cache_size_bytes_delta=sum_optional_integer_values(
            diagnostic.size_bytes_delta for diagnostic in warm_diagnostics
        ),
        cold_compilation_event_count=compile_log_summary_for_trial(cold_trial).compilation_event_count,
        warm_compilation_event_count=sum(
            compile_log_summary_for_trial(trial).compilation_event_count for trial in warm_trials
        ),
        cold_cache_hit_count=compile_log_summary_for_trial(cold_trial).persistent_cache_hit_count,
        warm_cache_hit_count=sum(
            compile_log_summary_for_trial(trial).persistent_cache_hit_count for trial in warm_trials
        ),
        cold_cache_miss_count=compile_log_summary_for_trial(cold_trial).persistent_cache_miss_count,
        warm_cache_miss_count=sum(
            compile_log_summary_for_trial(trial).persistent_cache_miss_count for trial in warm_trials
        ),
        cold_tracing_cache_miss_count=compile_log_summary_for_trial(cold_trial).tracing_cache_miss_count,
        warm_tracing_cache_miss_count=sum(
            compile_log_summary_for_trial(trial).tracing_cache_miss_count for trial in warm_trials
        ),
    )


def collect_jax_cache_diagnostics(
    aggregate_results: list[profile_deep_models.AggregateResult],
) -> dict[str, dict[str, object]]:
    """Collect aggregate JAX cache diagnostics keyed by result name."""
    diagnostics: dict[str, dict[str, object]] = {}
    for aggregate_result in aggregate_results:
        if aggregate_result.jax_cold_warm_summary is None:
            continue
        diagnostics[aggregate_result.name] = dataclasses.asdict(aggregate_result.jax_cold_warm_summary)
    return diagnostics
