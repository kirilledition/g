"""Application commands and completed-output evidence for deep profiling."""

from __future__ import annotations

import dataclasses
import json
import sys
import typing
from pathlib import Path

from tooling.benchmark import native_lifecycle
from tooling.profile_deep import commands as profile_deep_commands
from tooling.profile_deep import environment as profile_deep_environment
from tooling.profile_deep import jax_cache as profile_deep_jax_cache
from tooling.profile_deep import models as profile_deep_models


def build_g_step2_child_command(
    *,
    baseline_paths: typing.Any,
    candidate: profile_deep_models.Step2Candidate,
    output_prefix: Path,
    cache_directory: Path | None = None,
    stage_timing_path: Path | None = None,
    trace_directory: Path | None = None,
    memory_profile_path: Path | None = None,
    diagnostic_options: dict[str, object] | None = None,
) -> list[str]:
    """Build one isolated child command for the current native CLI."""
    jax_cache_directory = profile_deep_jax_cache.resolve_profile_jax_cache_directory(candidate, cache_directory)
    return profile_deep_commands.build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=output_prefix,
        cache_directory=jax_cache_directory,
        stage_timing_path=stage_timing_path,
        trace_directory=trace_directory,
        memory_profile_path=memory_profile_path,
        diagnostic_options=diagnostic_options,
    )


def write_inline_python_profile_script(command_arguments: list[str], script_path: Path) -> Path:
    """Write an inline Python command to a script file for external profilers."""
    if len(command_arguments) < 3 or command_arguments[0] != sys.executable or command_arguments[1] != "-c":
        message = "Expected an inline Python child command."
        raise ValueError(message)
    script_path.write_text(command_arguments[2], encoding="utf-8")
    return script_path


def build_deep_profiler_run_paths(
    *,
    profile_directory: Path,
    profile_name: str,
    emit_stage_timings: bool,
) -> profile_deep_models.DeepProfilerRunPaths:
    """Build isolated application paths for one deep profiler implementation."""
    application_output_prefix = profile_directory / profile_name
    stage_timing_path = profile_directory / f"{profile_name}.stage_timings.json" if emit_stage_timings else None
    return profile_deep_models.DeepProfilerRunPaths(
        application_output_prefix=application_output_prefix,
        application_output_run_directory=application_output_prefix.with_name(f"{application_output_prefix.name}.g"),
        stage_timing_path=stage_timing_path,
        profile_script_path=profile_directory / f"{profile_name}_child.py",
    )


def build_deep_profiler_child_command(
    *,
    profile_directory: Path,
    profile_name: str,
    baseline_paths: typing.Any,
    candidate: profile_deep_models.Step2Candidate,
    cache_directory: Path,
    emit_stage_timings: bool,
) -> profile_deep_models.DeepProfilerChildCommand:
    """Build an isolated child command for one deep profiler implementation."""
    run_paths = build_deep_profiler_run_paths(
        profile_directory=profile_directory,
        profile_name=profile_name,
        emit_stage_timings=emit_stage_timings,
    )
    inline_command_arguments = build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=run_paths.application_output_prefix,
        cache_directory=cache_directory,
        stage_timing_path=run_paths.stage_timing_path,
        diagnostic_options={"telemetry": "profile"},
    )
    write_inline_python_profile_script(inline_command_arguments, run_paths.profile_script_path)
    return profile_deep_models.DeepProfilerChildCommand(
        command_arguments=[sys.executable, str(run_paths.profile_script_path)],
        environment_overrides=profile_deep_environment.build_g_trial_environment(
            enable_jax_debug_logging=True,
        ),
        run_paths=run_paths,
    )


@dataclasses.dataclass(frozen=True)
class GTrialApplicationMetadata:
    """Application metadata emitted by an isolated g trial.

    Attributes:
        wall_time_seconds: Wall time measured inside the application child.
        output_row_count: Rows committed by the application run.
        output_path: First readable output artifact produced by the run.
        application_output_run_directory: Concrete per-trait application run directory.
        profile_summary_path: Existing application profile-summary path.
        device_diagnostics: Configured device, association backend, and genotype format.
        child_reported_cache_directory: JAX cache directory reported by the application child.

    """

    wall_time_seconds: float | None
    output_row_count: int | None
    output_path: str | None
    application_output_run_directory: str
    profile_summary_path: str | None
    device_diagnostics: dict[str, typing.Any] | None
    child_reported_cache_directory: str | None


def read_g_trial_child_payload(stdout_log_path: str) -> dict[str, typing.Any] | None:
    """Read the application payload from a direct or profiler-wrapped stdout log."""
    try:
        stdout_lines = Path(stdout_log_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for stdout_line in reversed(stdout_lines):
        try:
            raw_payload: object = json.loads(stdout_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw_payload, dict):
            continue
        payload = {key: value for key, value in raw_payload.items() if isinstance(key, str)}
        if "wall_time_seconds" in payload and "application_output_run_directory" in payload:
            return payload
    return None


def collect_g_trial_application_metadata(
    *,
    stdout_log_path: str,
    expected_output_root: Path,
    require_child_artifacts: bool,
) -> GTrialApplicationMetadata:
    """Recover application metadata from child stdout and its run manifest."""
    child_payload = read_g_trial_child_payload(stdout_log_path)
    if child_payload is None and require_child_artifacts:
        message = f"Successful g trial did not emit an application payload in {stdout_log_path}."
        raise RuntimeError(message)
    child_payload = child_payload or {}
    raw_run_directory = child_payload.get("application_output_run_directory")
    application_run_directory = Path(raw_run_directory) if isinstance(raw_run_directory, str) else expected_output_root

    completed_output: native_lifecycle.CompletedOutputEvidence | None = None
    if (application_run_directory / "run_manifest.json").is_file():
        try:
            completed_output = native_lifecycle.measure_completed_output_run(application_run_directory)
        except OSError, RuntimeError, ValueError, json.JSONDecodeError:
            if require_child_artifacts:
                raise
    if completed_output is None and require_child_artifacts:
        raise RuntimeError(f"Successful g trial has no valid production output at {application_run_directory}.")
    output_row_count = completed_output.row_count if completed_output is not None else None
    output_path = completed_output.parquet_paths[0] if completed_output is not None else None

    raw_profile_summary_path = child_payload.get("profile_summary_path")
    profile_summary_candidates = [
        Path(raw_profile_summary_path) if isinstance(raw_profile_summary_path, str) else None,
        expected_output_root / "logs" / "profile.summary.json",
        application_run_directory.parent / "logs" / "profile.summary.json",
    ]
    profile_summary_path = next(
        (str(path) for path in profile_summary_candidates if path is not None and path.exists()),
        None,
    )

    device_diagnostics: dict[str, typing.Any] | None = None
    if completed_output is not None:
        runtime = completed_output.manifest.get("runtime")
        execution_plan = completed_output.manifest.get("execution_plan")
        runtime_payload = runtime if isinstance(runtime, dict) else {}
        execution_plan_payload = execution_plan if isinstance(execution_plan, dict) else {}
        association_backend = execution_plan_payload.get("association_backend")
        association_backend_payload = association_backend if isinstance(association_backend, dict) else {}
        device_diagnostics = {
            "configured_device": runtime_payload.get("device"),
            "association_backend": association_backend_payload.get("kind"),
            "gpu_genotype_format": association_backend_payload.get("genotype_format"),
        }

    raw_wall_time_seconds = child_payload.get("wall_time_seconds")
    wall_time_seconds = (
        float(raw_wall_time_seconds)
        if isinstance(raw_wall_time_seconds, (int, float)) and not isinstance(raw_wall_time_seconds, bool)
        else None
    )
    raw_cache_directory = child_payload.get("jax_cache_directory")
    child_reported_cache_directory = raw_cache_directory if isinstance(raw_cache_directory, str) else None
    return GTrialApplicationMetadata(
        wall_time_seconds=wall_time_seconds,
        output_row_count=output_row_count,
        output_path=output_path,
        application_output_run_directory=str(application_run_directory),
        profile_summary_path=profile_summary_path,
        device_diagnostics=device_diagnostics,
        child_reported_cache_directory=child_reported_cache_directory,
    )


def attach_deep_profiler_metadata(
    *,
    result: profile_deep_models.TrialResult,
    run_paths: profile_deep_models.DeepProfilerRunPaths,
    profiler_artifact_path: Path | None,
) -> profile_deep_models.TrialResult:
    """Attach profiler artifact and application output metadata to a result."""
    application_metadata = collect_g_trial_application_metadata(
        stdout_log_path=result.stdout_log_path,
        expected_output_root=run_paths.application_output_run_directory,
        require_child_artifacts=False,
    )
    if result.status in {"success", "partial"} and application_metadata.output_row_count is None:
        result = dataclasses.replace(
            result,
            status="failed",
            notes=" ".join(
                note
                for note in (
                    result.notes,
                    "Profiler returned successfully, but its application did not produce valid completed output. "
                    "The retained profiler artifact may contain only a partial run.",
                )
                if note
            ),
        )
    return dataclasses.replace(
        result,
        wall_time_seconds=(
            application_metadata.wall_time_seconds
            if application_metadata.wall_time_seconds is not None
            else result.wall_time_seconds
        ),
        output_row_count=application_metadata.output_row_count,
        output_path=application_metadata.output_path,
        profiler_artifact_path=str(profiler_artifact_path) if profiler_artifact_path is not None else None,
        application_output_prefix=str(run_paths.application_output_prefix),
        application_output_run_directory=application_metadata.application_output_run_directory,
        stage_timing_path=(
            str(run_paths.stage_timing_path)
            if run_paths.stage_timing_path is not None and run_paths.stage_timing_path.exists()
            else None
        ),
        profile_summary_path=application_metadata.profile_summary_path,
        device_diagnostics=application_metadata.device_diagnostics,
    )
