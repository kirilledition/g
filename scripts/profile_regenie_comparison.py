#!/usr/bin/env python3
"""Profile original regenie and g REGENIE runs with unified reporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import typing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import psutil

if typing.TYPE_CHECKING:
    from scripts import benchmark as baseline_benchmark
else:
    try:
        from scripts import benchmark as baseline_benchmark
    except ModuleNotFoundError:
        import benchmark as baseline_benchmark


@dataclass(frozen=True)
class ExternalProfileResult:
    """Portable external-process profile summary."""

    program_name: str
    implementation: str
    trait_type: str
    step: int
    device: str
    status: str
    wall_time_seconds: float | None
    peak_rss_megabytes: float | None
    cpu_user_seconds: float | None
    cpu_system_seconds: float | None
    output_size_bytes: int | None
    stdout_log_path: str | None
    stderr_log_path: str | None
    output_paths: list[str]
    notes: str | None = None


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Unified regenie comparison profiling.")
    parser.add_argument("--include-gpu", action="store_true", help="Run GPU profile for g.")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU profile.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/profiles/regenie_comparison"),
        help="Directory where profile reports and logs are written.",
    )
    parser.add_argument("--sample-interval-seconds", type=float, default=0.05, help="psutil sampling interval.")
    parser.add_argument("--g-variant-limit", type=int, help="Optional variant cap for g profiling runs.")
    parser.add_argument("--g-chunk-size", type=int, default=1024, help="Chunk size for g profiling runs.")
    parser.add_argument("--enable-jax-trace", action="store_true", help="Enable JAX tracing for g profiles.")
    parser.add_argument(
        "--enable-memory-profile",
        action="store_true",
        help="Enable JAX memory profile for g profiles.",
    )
    return parser


def _collect_tree_metrics(process_handle: psutil.Process) -> tuple[float, float, float]:
    total_rss_bytes = 0.0
    total_user_seconds = 0.0
    total_system_seconds = 0.0
    try:
        child_processes = process_handle.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        child_processes = []
    process_list = [process_handle, *child_processes]
    for process in process_list:
        try:
            memory_info = process.memory_info()
            cpu_times = process.cpu_times()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        total_rss_bytes += float(memory_info.rss)
        total_user_seconds += float(cpu_times.user)
        total_system_seconds += float(cpu_times.system)
    return total_rss_bytes, total_user_seconds, total_system_seconds


def run_profiled_subprocess(
    *,
    command_arguments: list[str],
    stdout_log_path: Path,
    stderr_log_path: Path,
    sample_interval_seconds: float,
) -> tuple[bool, float, float, float, float, str | None]:
    """Run one process and sample process-tree RSS/CPU metrics."""
    process = subprocess.Popen(command_arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process_handle = psutil.Process(process.pid)
    start_time = time.perf_counter()
    peak_rss_bytes = 0.0
    first_user_seconds = 0.0
    first_system_seconds = 0.0
    last_user_seconds = 0.0
    last_system_seconds = 0.0
    initialized_samples = False
    while process.poll() is None:
        rss_bytes, user_seconds, system_seconds = _collect_tree_metrics(process_handle)
        peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
        if not initialized_samples:
            first_user_seconds = user_seconds
            first_system_seconds = system_seconds
            initialized_samples = True
        last_user_seconds = user_seconds
        last_system_seconds = system_seconds
        time.sleep(sample_interval_seconds)
    rss_bytes, user_seconds, system_seconds = _collect_tree_metrics(process_handle)
    peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
    if not initialized_samples:
        first_user_seconds = user_seconds
        first_system_seconds = system_seconds
    last_user_seconds = user_seconds
    last_system_seconds = system_seconds
    stdout_text, stderr_text = process.communicate()
    duration_seconds = time.perf_counter() - start_time
    stdout_log_path.write_text(stdout_text)
    stderr_log_path.write_text(stderr_text)
    success = process.returncode == 0
    error_message = None
    if not success:
        error_message = stderr_text.strip() or f"Command exited with code {process.returncode}."
    return (
        success,
        duration_seconds,
        peak_rss_bytes / (1024.0 * 1024.0),
        max(0.0, last_user_seconds - first_user_seconds),
        max(0.0, last_system_seconds - first_system_seconds),
        error_message,
    )


def _total_output_size_bytes(output_paths: list[Path]) -> int | None:
    present_paths = [output_path for output_path in output_paths if output_path.exists()]
    if not present_paths:
        return None
    return int(sum(output_path.stat().st_size for output_path in present_paths))


def _run_external_regenie_profile(
    *,
    program_name: str,
    trait_type: str,
    step: int,
    command_arguments: list[str],
    output_paths: list[Path],
    log_directory: Path,
    sample_interval_seconds: float,
) -> ExternalProfileResult:
    stdout_log_path = log_directory / f"{program_name}.stdout.log"
    stderr_log_path = log_directory / f"{program_name}.stderr.log"
    success, duration_seconds, peak_rss_megabytes, cpu_user_seconds, cpu_system_seconds, error_message = (
        run_profiled_subprocess(
            command_arguments=command_arguments,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            sample_interval_seconds=sample_interval_seconds,
        )
    )
    return ExternalProfileResult(
        program_name=program_name,
        implementation="regenie",
        trait_type=trait_type,
        step=step,
        device="external_cpu",
        status="success" if success else "failed",
        wall_time_seconds=duration_seconds,
        peak_rss_megabytes=peak_rss_megabytes,
        cpu_user_seconds=cpu_user_seconds,
        cpu_system_seconds=cpu_system_seconds,
        output_size_bytes=_total_output_size_bytes(output_paths),
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        output_paths=[str(output_path) for output_path in output_paths if output_path.exists()],
        notes=error_message,
    )


def _run_g_profile(
    *,
    program_name: str,
    device: str,
    baseline_paths: baseline_benchmark.BaselinePaths,
    output_dir: Path,
    variant_limit: int | None,
    chunk_size: int,
    enable_jax_trace: bool,
    enable_memory_profile: bool,
) -> ExternalProfileResult:
    profile_script_path = Path("scripts/profile_regenie2_linear_detailed.py")
    profile_run_directory = output_dir / program_name
    profile_run_directory.mkdir(parents=True, exist_ok=True)
    summary_path = profile_run_directory / "summary.json"
    command_arguments = [
        "uv",
        "run",
        "python",
        str(profile_script_path),
        "--bgen",
        str(baseline_paths.bgen_path),
        "--sample",
        str(baseline_paths.sample_path),
        "--pheno",
        str(baseline_paths.continuous_phenotype_path),
        "--pheno-name",
        "phenotype_continuous",
        "--covar",
        str(baseline_paths.covariate_path),
        "--covar-names",
        "age,sex",
        "--pred",
        str(baseline_paths.regenie_qt_prediction_list_path),
        "--device",
        device,
        "--chunk-size",
        str(chunk_size),
        "--output-dir",
        str(profile_run_directory),
        "--report-name",
        "profile",
        "--json-summary-path",
        str(summary_path),
    ]
    if variant_limit is not None:
        command_arguments.extend(["--variant-limit", str(variant_limit)])
    if enable_jax_trace:
        command_arguments.append("--enable-jax-trace")
    if enable_memory_profile:
        command_arguments.append("--enable-memory-profile")
    stdout_log_path = profile_run_directory / "stdout.log"
    stderr_log_path = profile_run_directory / "stderr.log"
    success, duration_seconds, peak_rss_megabytes, cpu_user_seconds, cpu_system_seconds, error_message = (
        run_profiled_subprocess(
            command_arguments=command_arguments,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            sample_interval_seconds=0.05,
        )
    )
    output_paths: list[Path] = [summary_path]
    if summary_path.exists():
        summary_data = json.loads(summary_path.read_text())
        if summary_data.get("final_parquet_path"):
            output_paths.append(Path(summary_data["final_parquet_path"]))
    return ExternalProfileResult(
        program_name=program_name,
        implementation="g",
        trait_type="quantitative",
        step=2,
        device=device,
        status="success" if success else "failed",
        wall_time_seconds=duration_seconds,
        peak_rss_megabytes=peak_rss_megabytes,
        cpu_user_seconds=cpu_user_seconds,
        cpu_system_seconds=cpu_system_seconds,
        output_size_bytes=_total_output_size_bytes(output_paths),
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        output_paths=[str(output_path) for output_path in output_paths if output_path.exists()],
        notes=error_message,
    )


def _not_implemented_profile(program_name: str, trait_type: str, step: int, device: str) -> ExternalProfileResult:
    return ExternalProfileResult(
        program_name=program_name,
        implementation="g",
        trait_type=trait_type,
        step=step,
        device=device,
        status="not_implemented",
        wall_time_seconds=None,
        peak_rss_megabytes=None,
        cpu_user_seconds=None,
        cpu_system_seconds=None,
        output_size_bytes=None,
        stdout_log_path=None,
        stderr_log_path=None,
        output_paths=[],
        notes="This g workflow is not implemented in the active public surface.",
    )


def _result_by_name(results: list[ExternalProfileResult], program_name: str) -> ExternalProfileResult | None:
    for result in results:
        if result.program_name == program_name:
            return result
    return None


def _runtime_comparison(
    regenie_result: ExternalProfileResult | None,
    g_result: ExternalProfileResult | None,
) -> dict[str, float] | None:
    if regenie_result is None or g_result is None:
        return None
    if (
        regenie_result.status != "success"
        or g_result.status != "success"
        or regenie_result.wall_time_seconds is None
        or g_result.wall_time_seconds is None
    ):
        return None
    return {
        "speedup_ratio": regenie_result.wall_time_seconds / g_result.wall_time_seconds,
        "absolute_delta_seconds": g_result.wall_time_seconds - regenie_result.wall_time_seconds,
    }


def _write_text_summary(report_path: Path, report_data: dict[str, typing.Any]) -> None:
    lines: list[str] = ["REGENIE Comparison Profiling Summary", ""]
    for result in report_data["results"]:
        lines.append(
            f"{result['program_name']}: status={result['status']}, wall={result['wall_time_seconds']}, "
            f"peak_rss_mb={result['peak_rss_megabytes']}, cpu_user={result['cpu_user_seconds']}, "
            f"cpu_system={result['cpu_system_seconds']}",
        )
    lines.append("")
    runtime = report_data.get("comparisons", {})
    lines.append(f"Comparisons: {json.dumps(runtime, indent=2)}")
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run profiling comparison suite."""
    parser = build_argument_parser()
    arguments = parser.parse_args()
    baseline_paths = baseline_benchmark.build_baseline_paths()
    baseline_benchmark.validate_input_files(baseline_paths)
    regenie_executable = baseline_benchmark.resolve_required_executable("REGENIE_BIN", "regenie")
    baseline_benchmark.resolve_required_executable("UV_BIN", "uv")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    log_directory = arguments.output_dir / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    regenie_step1_binary_prefix = baseline_paths.baseline_directory / "regenie_step1"
    regenie_step2_binary_prefix = baseline_paths.baseline_directory / "regenie_step2"
    regenie_step1_quantitative_prefix = baseline_paths.baseline_directory / "regenie_step1_qt"
    regenie_step2_quantitative_prefix = baseline_paths.baseline_directory / "regenie_step2_qt"

    results: list[ExternalProfileResult] = []
    results.append(
        _run_external_regenie_profile(
            program_name="regenie_step1_binary",
            trait_type="binary",
            step=1,
            command_arguments=baseline_benchmark.build_regenie_step1_command(regenie_executable, baseline_paths),
            output_paths=[regenie_step1_binary_prefix.parent / "regenie_step1_pred.list"],
            log_directory=log_directory,
            sample_interval_seconds=arguments.sample_interval_seconds,
        )
    )
    results.append(
        _run_external_regenie_profile(
            program_name="regenie_step2_binary",
            trait_type="binary",
            step=2,
            command_arguments=baseline_benchmark.build_regenie_step2_command(regenie_executable, baseline_paths),
            output_paths=[regenie_step2_binary_prefix.parent / "regenie_step2_phenotype_binary.regenie"],
            log_directory=log_directory,
            sample_interval_seconds=arguments.sample_interval_seconds,
        )
    )
    results.append(
        _run_external_regenie_profile(
            program_name="regenie_step1_quantitative",
            trait_type="quantitative",
            step=1,
            command_arguments=baseline_benchmark.build_regenie_step1_continuous_command(
                regenie_executable,
                baseline_paths,
            ),
            output_paths=[regenie_step1_quantitative_prefix.parent / "regenie_step1_qt_pred.list"],
            log_directory=log_directory,
            sample_interval_seconds=arguments.sample_interval_seconds,
        )
    )
    results.append(
        _run_external_regenie_profile(
            program_name="regenie_step2_quantitative",
            trait_type="quantitative",
            step=2,
            command_arguments=baseline_benchmark.build_regenie_step2_continuous_command(
                regenie_executable,
                baseline_paths,
            ),
            output_paths=[regenie_step2_quantitative_prefix.parent / "regenie_step2_qt_phenotype_continuous.regenie"],
            log_directory=log_directory,
            sample_interval_seconds=arguments.sample_interval_seconds,
        )
    )

    results.append(_not_implemented_profile("g_regenie2_binary_step1", "binary", 1, "cpu"))
    results.append(_not_implemented_profile("g_regenie2_binary_step2", "binary", 2, "cpu"))
    results.append(_not_implemented_profile("g_regenie2_quantitative_step1", "quantitative", 1, "cpu"))

    results.append(
        _run_g_profile(
            program_name="g_regenie2_quantitative_step2_cpu",
            device="cpu",
            baseline_paths=baseline_paths,
            output_dir=arguments.output_dir,
            variant_limit=arguments.g_variant_limit,
            chunk_size=arguments.g_chunk_size,
            enable_jax_trace=arguments.enable_jax_trace,
            enable_memory_profile=arguments.enable_memory_profile,
        )
    )
    if arguments.include_gpu and not arguments.cpu_only:
        results.append(
            _run_g_profile(
                program_name="g_regenie2_quantitative_step2_gpu",
                device="gpu",
                baseline_paths=baseline_paths,
                output_dir=arguments.output_dir,
                variant_limit=arguments.g_variant_limit,
                chunk_size=arguments.g_chunk_size,
                enable_jax_trace=arguments.enable_jax_trace,
                enable_memory_profile=arguments.enable_memory_profile,
            )
        )
    else:
        results.append(
            ExternalProfileResult(
                program_name="g_regenie2_quantitative_step2_gpu",
                implementation="g",
                trait_type="quantitative",
                step=2,
                device="gpu",
                status="not_implemented",
                wall_time_seconds=None,
                peak_rss_megabytes=None,
                cpu_user_seconds=None,
                cpu_system_seconds=None,
                output_size_bytes=None,
                stdout_log_path=None,
                stderr_log_path=None,
                output_paths=[],
                notes="GPU run skipped (enable with --include-gpu).",
            )
        )

    regenie_quantitative = _result_by_name(results, "regenie_step2_quantitative")
    g_cpu = _result_by_name(results, "g_regenie2_quantitative_step2_cpu")
    g_gpu = _result_by_name(results, "g_regenie2_quantitative_step2_gpu")
    report_data: dict[str, typing.Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hardware": asdict(baseline_benchmark.collect_hardware_summary()),
        "results": [asdict(result) for result in results],
        "comparisons": {
            "g_cpu_vs_gpu": _runtime_comparison(g_cpu, g_gpu),
            "regenie_step2_quantitative_vs_g_cpu": _runtime_comparison(regenie_quantitative, g_cpu),
            "regenie_step2_quantitative_vs_g_gpu": _runtime_comparison(regenie_quantitative, g_gpu),
        },
    }

    json_report_path = arguments.output_dir / "profile_report.json"
    text_report_path = arguments.output_dir / "profile_summary.txt"
    json_report_path.write_text(f"{json.dumps(report_data, indent=2)}\n")
    _write_text_summary(text_report_path, report_data)
    print(f"Wrote profiling report: {json_report_path}")
    print(f"Wrote profiling summary: {text_report_path}")


if __name__ == "__main__":
    main()
