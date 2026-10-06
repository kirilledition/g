"""Logged subprocess execution and profiler failure records."""

from __future__ import annotations

import dataclasses
import logging
import os
import shlex
import signal
import subprocess
import time
import typing

from tooling.profile_deep import models as profile_deep_models

if typing.TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def run_logged_command(
    *,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    command_arguments: list[str],
    environment_overrides: dict[str, str],
    log_directory: Path,
    timeout_seconds: int | None = None,
) -> profile_deep_models.TrialResult:
    """Run one command and persist stdout/stderr logs."""
    log_directory.mkdir(parents=True, exist_ok=True)
    stdout_log_path = log_directory / f"{name}.stdout.log"
    stderr_log_path = log_directory / f"{name}.stderr.log"
    environment = dict(os.environ)
    environment.update(environment_overrides)
    logger.info("Starting %s profiler/workload command", name)
    logger.debug("Command for %s: %s", name, shlex.join(command_arguments))
    if timeout_seconds is not None:
        logger.debug("Timeout for %s set to %.1fs", name, float(timeout_seconds))
    start_time = time.perf_counter()
    command_stdout: str = ""
    command_stderr: str = ""
    status = "success"
    notes: str | None = None
    timeout_reached = False
    process = subprocess.Popen(
        command_arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        command_stdout, command_stderr = process.communicate(timeout=timeout_seconds)
        status = "success" if process.returncode == 0 else "failed"
        if process.returncode != 0:
            notes = command_stderr.strip() or command_stdout.strip()
    except subprocess.TimeoutExpired:
        timeout_reached = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            command_stdout, command_stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            command_stdout, command_stderr = process.communicate()
        notes = f"{name} timed out after {float(timeout_seconds or 0):.3f}s; the profiler process group was terminated."
        status = "failed"
    wall_time_seconds = time.perf_counter() - start_time
    stdout_log_path.write_text(command_stdout, encoding="utf-8")
    stderr_log_path.write_text(command_stderr, encoding="utf-8")
    permission_block_note = permission_blocked_profiler_note(
        stdout=command_stdout,
        stderr=command_stderr,
    )
    if permission_block_note is not None:
        status = "skipped"
        notes = permission_block_note
    if notes is None and not timeout_reached:
        notes = command_stderr.strip() or command_stdout.strip()
    logger.info(
        "Finished %s with status=%s in %.3fs",
        name,
        status,
        wall_time_seconds,
    )
    return profile_deep_models.TrialResult(
        name=name,
        implementation=implementation,
        trait_type=trait_type,
        device=device,
        status=status,
        wall_time_seconds=wall_time_seconds,
        process_wall_time_seconds=wall_time_seconds,
        output_row_count=None,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        command_arguments=command_arguments,
        environment_overrides=environment_overrides,
        notes=notes,
    )


def permission_blocked_profiler_note(*, stdout: str, stderr: str) -> str | None:
    """Return an actionable note for known profiler permission failures."""
    combined_output = f"{stderr}\n{stdout}"
    if "ERR_NVGPUCTRPERM" in combined_output:
        return (
            "Nsight Compute connected to the CUDA process, but the NVIDIA driver restricts GPU performance "
            "counter access to admin users. Ask the cluster administrator to allow non-admin GPU performance "
            "counters on the GPU nodes, or keep using Nsight Systems/JAX traces for CUDA timelines."
        )
    if (
        "perf_event_open" in combined_output
        or "No permission to enable" in combined_output
        or "Access to performance monitoring" in combined_output
        or "perf_event_paranoid setting is" in combined_output
    ):
        return (
            "Linux perf is blocked by this node's perf_event policy. Ask the cluster administrator to lower "
            "perf_event_paranoid for profiling jobs."
        )
    return None


def skipped_profile_result(
    *,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    log_directory: Path,
    notes: str,
) -> profile_deep_models.TrialResult:
    """Build a skipped profiler result and persist the skip reason."""
    log_directory.mkdir(parents=True, exist_ok=True)
    stdout_log_path = log_directory / f"{name}.stdout.log"
    stderr_log_path = log_directory / f"{name}.stderr.log"
    stdout_log_path.write_text("", encoding="utf-8")
    stderr_log_path.write_text(notes + "\n", encoding="utf-8")
    logger.info("Skipping %s: %s", name, notes)
    return profile_deep_models.TrialResult(
        name=name,
        implementation=implementation,
        trait_type=trait_type,
        device=device,
        status="skipped",
        wall_time_seconds=None,
        process_wall_time_seconds=None,
        output_row_count=None,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        command_arguments=[],
        environment_overrides={},
        notes=notes,
    )


def unsupported_aggregate_result(
    *,
    name: str,
    trait_type: str,
    device: str,
    log_directory: Path,
    notes: str,
) -> profile_deep_models.AggregateResult:
    """Build an unsupported aggregate result and persist the reason."""
    trial_result = skipped_profile_result(
        name=f"{name}_unsupported",
        implementation="regenie",
        trait_type=trait_type,
        device=device,
        log_directory=log_directory,
        notes=notes,
    )
    unsupported_trial_result = dataclasses.replace(trial_result, status="unsupported")
    return profile_deep_models.AggregateResult(
        name=name,
        implementation="regenie",
        trait_type=trait_type,
        device=device,
        status="unsupported",
        trial_count=1,
        warmup_count=0,
        median_wall_time_seconds=None,
        mean_wall_time_seconds=None,
        min_wall_time_seconds=None,
        max_wall_time_seconds=None,
        standard_deviation_seconds=None,
        rows_per_second=None,
        trials=[unsupported_trial_result],
    )
