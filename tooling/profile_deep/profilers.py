"""External profiler captures for completed application runs."""

from __future__ import annotations

import dataclasses
import logging
import sys
import typing
from pathlib import Path

from tooling.profile_deep import application as profile_deep_application
from tooling.profile_deep import candidates as profile_deep_candidates
from tooling.profile_deep import config as profile_deep_config
from tooling.profile_deep import environment as profile_deep_environment
from tooling.profile_deep import execution as profile_deep_execution
from tooling.profile_deep import models as profile_deep_models
from tooling.profile_deep import tools as profile_deep_tools
from tooling.profile_deep import trials as profile_deep_trials

logger = logging.getLogger(__name__)


def build_scalene_command_arguments(
    *,
    tool_status: profile_deep_models.ProfilerToolStatus,
    output_path: Path,
    profile_script_path: Path,
) -> list[str]:
    """Build a Scalene command that preserves project dependencies."""
    if tool_status.executable_path == sys.executable:
        return [
            sys.executable,
            "-m",
            "scalene",
            "run",
            "--profile-all",
            "--outfile",
            str(output_path),
            str(profile_script_path),
        ]
    if profile_deep_environment.executable_name(tool_status.executable_path) == "uv":
        return [
            tool_status.executable_path or "uv",
            "run",
            "--no-sync",
            "--with",
            "scalene",
            "scalene",
            "run",
            "--profile-all",
            "--outfile",
            str(output_path),
            str(profile_script_path),
        ]
    return [
        tool_status.executable_path or "scalene",
        "run",
        "--profile-all",
        "--outfile",
        str(output_path),
        str(profile_script_path),
    ]


def build_memray_command_arguments(
    *,
    tool_status: profile_deep_models.ProfilerToolStatus,
    output_path: Path,
    profile_script_path: Path,
) -> list[str]:
    """Build a Memray command that preserves project dependencies."""
    memray_arguments = [
        "-m",
        "memray",
        "run",
        "--force",
        "--aggregate",
        "--native",
        "--output",
        str(output_path),
        str(profile_script_path),
    ]
    if tool_status.executable_path == sys.executable:
        return [sys.executable, *memray_arguments]
    if profile_deep_environment.executable_name(tool_status.executable_path) == "uv":
        return [
            tool_status.executable_path or "uv",
            "run",
            "--no-sync",
            "--with",
            "memray",
            "python",
            *memray_arguments,
        ]
    return [
        tool_status.executable_path or "memray",
        "run",
        "--force",
        "--aggregate",
        "--native",
        "--output",
        str(output_path),
        str(profile_script_path),
    ]


def append_skipped_executable_profile(
    *,
    results: dict[str, typing.Any],
    tool_status: profile_deep_models.ProfilerToolStatus,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    log_directory: Path,
) -> None:
    """Append a skipped profiler result for a missing executable."""
    results["sampling_profiles"].append(
        dataclasses.asdict(
            profile_deep_execution.skipped_profile_result(
                name=name,
                implementation=implementation,
                trait_type=trait_type,
                device=device,
                log_directory=log_directory,
                notes=tool_status.notes,
            )
        )
    )


def append_logged_profile_result(
    *,
    results: dict[str, typing.Any],
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    command_arguments: list[str],
    environment_overrides: dict[str, str],
    log_directory: Path,
    run_paths: profile_deep_models.DeepProfilerRunPaths,
    profiler_artifact_path: Path | None,
    timeout_seconds: int | None = None,
) -> None:
    """Run and append one external profiler result."""
    profile_result = profile_deep_execution.run_logged_command(
        name=name,
        implementation=implementation,
        trait_type=trait_type,
        device=device,
        command_arguments=command_arguments,
        environment_overrides=environment_overrides,
        log_directory=log_directory,
        timeout_seconds=timeout_seconds,
    )
    if (
        implementation == "Nsight Systems"
        and profile_result.status == "success"
        and profile_result.notes is not None
        and "TargetProfilingFailed" in profile_result.notes
    ):
        profile_result = dataclasses.replace(
            profile_result,
            status="partial",
            notes=(
                "Nsight Systems produced a CUDA report and summary tables, but its importer reported "
                "non-fatal target metadata errors for this driver/GPU combination."
            ),
        )
    if (
        profile_result.status in {"success", "partial"}
        and profiler_artifact_path is not None
        and not profiler_artifact_path.exists()
    ):
        profile_result = dataclasses.replace(
            profile_result,
            status="failed",
            notes=f"Profiler exited successfully but did not create {profiler_artifact_path}.",
        )
    results["sampling_profiles"].append(
        dataclasses.asdict(
            profile_deep_application.attach_deep_profiler_metadata(
                result=profile_result,
                run_paths=run_paths,
                profiler_artifact_path=profiler_artifact_path,
            )
        )
    )


def run_deep_profiles(
    *,
    arguments: profile_deep_models.ProfileArguments,
    baseline_paths: typing.Any,
    winners: dict[str, profile_deep_models.AggregateResult],
    output_directory: Path,
    cache_directory: Path,
) -> dict[str, typing.Any]:
    """Run optional profiler commands for representative g winners."""
    profile_directory = output_directory / "deep_profiles"
    profile_directory.mkdir(parents=True, exist_ok=True)
    profiler_tool_status = profile_deep_tools.build_profiler_tool_status(arguments)
    emit_stage_timings = profile_deep_config.should_emit_stage_timings(arguments)
    results: dict[str, typing.Any] = {
        "profiler_tools": profile_deep_tools.serialize_profiler_tool_status(profiler_tool_status),
        "sampling_profiles": [],
    }
    for winner_key, winner in sorted(winners.items()):
        if not winner.trials:
            continue
        candidate = profile_deep_candidates.candidate_from_aggregate_name(winner_key, winner)
        if arguments.enable_jax_trace or arguments.enable_jax_memory_profile:
            trace_directory = profile_directory / f"{winner_key}_jax_trace" if arguments.enable_jax_trace else None
            memory_profile_path = (
                profile_directory / f"{winner_key}_device_memory.prof" if arguments.enable_jax_memory_profile else None
            )
            profiler_artifact_path = trace_directory if trace_directory is not None else memory_profile_path
            logger.info("Running JAX profiler capture for %s", winner_key)
            profile_result = profile_deep_trials.run_g_trial(
                name=f"profile_{winner_key}_jax",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=profile_directory,
                log_directory=output_directory / "logs",
                cache_directory=cache_directory,
                emit_stage_timings=emit_stage_timings,
                trace_directory=trace_directory,
                memory_profile_path=memory_profile_path,
                diagnostic_options={"telemetry": "profile"},
            )
            results["sampling_profiles"].append(
                dataclasses.asdict(
                    dataclasses.replace(
                        profile_result,
                        profiler_artifact_path=(
                            str(profiler_artifact_path) if profiler_artifact_path is not None else None
                        ),
                    )
                )
            )
        if arguments.enable_python_cprofile:
            cprofile_output_path = profile_directory / f"{winner_key}.cprofile"
            cprofile_text_path = profile_directory / f"{winner_key}.cprofile.txt"
            cprofile_child_command = profile_deep_application.build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_cprofile",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                emit_stage_timings=emit_stage_timings,
            )
            cprofile_result = profile_deep_application.attach_deep_profiler_metadata(
                result=profile_deep_execution.run_logged_command(
                    name=f"profile_{winner_key}_cprofile",
                    implementation="cProfile",
                    trait_type=candidate.trait_type,
                    device=candidate.device,
                    command_arguments=[
                        sys.executable,
                        "-m",
                        "cProfile",
                        "-o",
                        str(cprofile_output_path),
                        str(cprofile_child_command.run_paths.profile_script_path),
                    ],
                    environment_overrides=cprofile_child_command.environment_overrides,
                    log_directory=output_directory / "logs",
                ),
                run_paths=cprofile_child_command.run_paths,
                profiler_artifact_path=cprofile_output_path,
            )
            results["sampling_profiles"].append(dataclasses.asdict(cprofile_result))
            if cprofile_result.status == "success":
                cprofile_text_result = profile_deep_environment.command_output(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import pstats, sys; "
                            "pstats.Stats(sys.argv[1]).strip_dirs().sort_stats('cumtime').print_stats(80)"
                        ),
                        str(cprofile_output_path),
                    ]
                )
                cprofile_text_path.write_text(cprofile_text_result["stdout"], encoding="utf-8")
        py_spy_status = profiler_tool_status["py_spy"]
        if arguments.enable_py_spy and py_spy_status.available:
            speedscope_path = profile_directory / f"{winner_key}.speedscope.json"
            py_spy_child_command = profile_deep_application.build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_py_spy",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                emit_stage_timings=emit_stage_timings,
            )
            command_arguments = [
                py_spy_status.executable_path or "py-spy",
                "record",
                "--native",
                "--threads",
                "--rate",
                "10",
                "--format",
                "speedscope",
                "--output",
                str(speedscope_path),
                "--",
                *py_spy_child_command.command_arguments,
            ]
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_py_spy",
                implementation="py-spy",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=command_arguments,
                environment_overrides=py_spy_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=py_spy_child_command.run_paths,
                profiler_artifact_path=speedscope_path,
                timeout_seconds=arguments.py_spy_timeout_seconds,
            )
        elif arguments.enable_py_spy:
            append_skipped_executable_profile(
                results=results,
                tool_status=py_spy_status,
                name=f"profile_{winner_key}_py_spy",
                implementation="py-spy",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
        scalene_status = profiler_tool_status["scalene"]
        if arguments.enable_scalene and scalene_status.available:
            scalene_json_path = profile_directory / f"{winner_key}.scalene.json"
            scalene_child_command = profile_deep_application.build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_scalene",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                emit_stage_timings=emit_stage_timings,
            )
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_scalene",
                implementation="Scalene",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=build_scalene_command_arguments(
                    tool_status=scalene_status,
                    output_path=scalene_json_path,
                    profile_script_path=scalene_child_command.run_paths.profile_script_path,
                ),
                environment_overrides=scalene_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=scalene_child_command.run_paths,
                profiler_artifact_path=scalene_json_path,
                timeout_seconds=arguments.scalene_timeout_seconds,
            )
        elif arguments.enable_scalene:
            append_skipped_executable_profile(
                results=results,
                tool_status=scalene_status,
                name=f"profile_{winner_key}_scalene",
                implementation="Scalene",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
        memray_status = profiler_tool_status["memray"]
        if arguments.enable_memray and memray_status.available:
            memray_output_path = profile_directory / f"{winner_key}.memray.bin"
            memray_child_command = profile_deep_application.build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_memray",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                emit_stage_timings=emit_stage_timings,
            )
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_memray",
                implementation="Memray",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=build_memray_command_arguments(
                    tool_status=memray_status,
                    output_path=memray_output_path,
                    profile_script_path=memray_child_command.run_paths.profile_script_path,
                ),
                environment_overrides=memray_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=memray_child_command.run_paths,
                profiler_artifact_path=memray_output_path,
                timeout_seconds=arguments.memray_timeout_seconds,
            )
        elif arguments.enable_memray:
            append_skipped_executable_profile(
                results=results,
                tool_status=memray_status,
                name=f"profile_{winner_key}_memray",
                implementation="Memray",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
        nsight_systems_status = profiler_tool_status["nsight_systems"]
        if arguments.enable_nsight_systems and nsight_systems_status.available:
            nsight_report_prefix = profile_directory / f"{winner_key}_nsys"
            nsight_report_path = Path(f"{nsight_report_prefix}.nsys-rep")
            nsight_systems_child_command = profile_deep_application.build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_nsys",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                emit_stage_timings=emit_stage_timings,
            )
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_nsys",
                implementation="Nsight Systems",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=[
                    nsight_systems_status.executable_path or "nsys",
                    "profile",
                    "--trace=cuda,cudnn,cublas,osrt,nvtx",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--stats=true",
                    "--force-overwrite=true",
                    "--output",
                    str(nsight_report_prefix),
                    *nsight_systems_child_command.command_arguments,
                ],
                environment_overrides=nsight_systems_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=nsight_systems_child_command.run_paths,
                profiler_artifact_path=nsight_report_path,
                timeout_seconds=arguments.nsight_systems_timeout_seconds,
            )
        elif arguments.enable_nsight_systems:
            append_skipped_executable_profile(
                results=results,
                tool_status=nsight_systems_status,
                name=f"profile_{winner_key}_nsys",
                implementation="Nsight Systems",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
        nsight_compute_status = profiler_tool_status["nsight_compute"]
        if arguments.enable_nsight_compute and nsight_compute_status.available:
            nsight_compute_report_prefix = profile_directory / f"{winner_key}_ncu"
            nsight_compute_report_path = Path(f"{nsight_compute_report_prefix}.ncu-rep")
            nsight_compute_child_command = profile_deep_application.build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_ncu",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                emit_stage_timings=emit_stage_timings,
            )
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_ncu",
                implementation="Nsight Compute",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=[
                    nsight_compute_status.executable_path or "ncu",
                    "--target-processes",
                    "all",
                    "--set",
                    "default",
                    "--export",
                    str(nsight_compute_report_prefix),
                    *nsight_compute_child_command.command_arguments,
                ],
                environment_overrides=nsight_compute_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=nsight_compute_child_command.run_paths,
                profiler_artifact_path=nsight_compute_report_path,
                timeout_seconds=arguments.nsight_compute_timeout_seconds,
            )
        elif arguments.enable_nsight_compute:
            append_skipped_executable_profile(
                results=results,
                tool_status=nsight_compute_status,
                name=f"profile_{winner_key}_ncu",
                implementation="Nsight Compute",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
        perf_status = profiler_tool_status["linux_perf"]
        if arguments.enable_linux_perf and perf_status.available:
            perf_path = profile_directory / f"{winner_key}.perf.data"
            perf_child_command = profile_deep_application.build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_perf",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                emit_stage_timings=emit_stage_timings,
            )
            command_arguments = [
                perf_status.executable_path or "perf",
                "record",
                "--call-graph",
                "dwarf,16384",
                "-o",
                str(perf_path),
                "--",
                *perf_child_command.command_arguments,
            ]
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_perf",
                implementation="perf",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=command_arguments,
                environment_overrides=perf_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=perf_child_command.run_paths,
                profiler_artifact_path=perf_path,
                timeout_seconds=arguments.linux_perf_timeout_seconds,
            )
        elif arguments.enable_linux_perf:
            append_skipped_executable_profile(
                results=results,
                tool_status=perf_status,
                name=f"profile_{winner_key}_perf",
                implementation="perf",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
    return results
