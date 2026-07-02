#!/usr/bin/env python3
"""Benchmark native extension build profiles."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import os
import shutil
import time
import typing
from pathlib import Path

import hydra

import tooling.configuration as tooling_configuration
from tooling.common import commands as tooling_commands
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports

if typing.TYPE_CHECKING:
    import collections.abc

    import omegaconf


class BuildProfileLabel(enum.StrEnum):
    """Build profile labels accepted by the benchmark harness."""

    DEV_FAST = "dev-fast"
    DEV_FAST_LLD = "dev-fast-lld"
    DEV_FAST_MOLD = "dev-fast-mold"
    DEV_OPT = "dev-opt"
    RELEASE = "release"
    PERF_THIN_CGU8 = "perf-thin-cgu8"
    PERF_THIN_CGU8_LLD = "perf-thin-cgu8-lld"
    PERF_THIN_CGU8_MOLD = "perf-thin-cgu8-mold"
    PERF_THIN_CGU1 = "perf-thin-cgu1"
    PERF_FAT_CGU1 = "perf-fat-cgu1"
    PERF_O2_THIN_CGU8 = "perf-o2-thin-cgu8"
    PERF_O3_THIN_CGU8 = "perf-o3-thin-cgu8"


@dataclasses.dataclass(frozen=True)
class BuildProfileSpec:
    """Cargo and rustc settings for one benchmark label.

    Attributes:
        label: User-facing profile label.
        cargo_profile: Cargo profile passed to Maturin.
        rustflags: Extra RUSTFLAGS used for the build.

    """

    label: BuildProfileLabel
    cargo_profile: str
    rustflags: str


@dataclasses.dataclass(frozen=True)
class SourceTimestamp:
    """Original source file timestamps captured before a touch build.

    Attributes:
        path: Source file path.
        access_time_nanoseconds: Original access timestamp.
        modification_time_nanoseconds: Original modification timestamp.

    """

    path: Path
    access_time_nanoseconds: int
    modification_time_nanoseconds: int


@dataclasses.dataclass(frozen=True)
class CommandTiming:
    """Timed command execution summary.

    Attributes:
        name: Stable stage name.
        command_arguments: Command argument vector.
        return_code: Process return code.
        duration_seconds: Wall-clock command duration.
        timed_out: Whether the command timed out.
        missing_executable: Whether the executable was missing.
        stdout_log_path: Captured stdout log path.
        stderr_log_path: Captured stderr log path.

    """

    name: str
    command_arguments: tuple[str, ...]
    return_code: int | None
    duration_seconds: float
    timed_out: bool
    missing_executable: bool
    stdout_log_path: str
    stderr_log_path: str


@dataclasses.dataclass(frozen=True)
class IncrementalBuildTiming:
    """Incremental build timing for one touched source file.

    Attributes:
        touched_path: Source path touched before the build.
        command_timing: Timed build command.

    """

    touched_path: str
    command_timing: CommandTiming


@dataclasses.dataclass(frozen=True)
class ProfileBuildReport:
    """Build and runtime measurements for one profile label.

    Attributes:
        label: User-facing profile label.
        cargo_profile: Cargo profile passed to Maturin.
        rustflags: Extra RUSTFLAGS used for the build.
        target_directory: Isolated Cargo target directory.
        clean_build: Clean build timing.
        incremental_builds: Incremental touch-build timings.
        extension_size_bytes: Largest observed native extension artifact size.
        import_timing: Optional native import timing command.
        smoke_timing: Optional smoke command timing.
        bgen_reader_timing: Optional BGEN reader smoke timing.
        gpu_smoke_timing: Optional GPU smoke timing.

    """

    label: str
    cargo_profile: str
    rustflags: str
    target_directory: str
    clean_build: CommandTiming
    incremental_builds: list[IncrementalBuildTiming]
    extension_size_bytes: int | None
    import_timing: CommandTiming | None
    smoke_timing: CommandTiming | None
    bgen_reader_timing: CommandTiming | None
    gpu_smoke_timing: CommandTiming | None


@dataclasses.dataclass(frozen=True)
class RustBuildProfilesReport:
    """Complete Rust build profile benchmark report.

    Attributes:
        schema_version: Report schema version.
        generated_at_utc: UTC timestamp when the report was written.
        repository_root: Repository root path.
        reports: Per-profile reports.

    """

    schema_version: int
    generated_at_utc: str
    repository_root: str
    reports: list[ProfileBuildReport]


@dataclasses.dataclass(frozen=True)
class RustBuildProfilesArguments:
    """Resolved benchmark harness arguments.

    Attributes:
        labels: Profile labels to benchmark.
        output_parent: Parent directory for timestamped reports and logs.
        clean_build: Whether to remove each profile target directory before the first build.
        incremental_touch_paths: Source files touched for incremental rebuild measurements.
        build_timeout_seconds: Optional build command timeout.
        runtime_timeout_seconds: Optional runtime command timeout.
        run_import_timing: Whether to time importing the native extension.
        run_smoke_command: Whether to run the generic smoke command.
        smoke_command: Generic smoke command argument vector.
        run_bgen_reader_smoke: Whether to run the BGEN reader smoke command.
        bgen_reader_command: BGEN reader smoke command argument vector.
        run_gpu_smoke: Whether to run the GPU smoke command.
        gpu_smoke_command: GPU smoke command argument vector.

    """

    labels: tuple[BuildProfileLabel, ...]
    output_parent: Path
    clean_build: bool
    incremental_touch_paths: tuple[Path, ...]
    build_timeout_seconds: float | None
    runtime_timeout_seconds: float | None
    run_import_timing: bool
    run_smoke_command: bool
    smoke_command: tuple[str, ...]
    run_bgen_reader_smoke: bool
    bgen_reader_command: tuple[str, ...]
    run_gpu_smoke: bool
    gpu_smoke_command: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RuntimeCommandReports:
    """Runtime command timings for one build profile.

    Attributes:
        import_timing: Optional native import timing command.
        smoke_timing: Optional smoke command timing.
        bgen_reader_timing: Optional BGEN reader smoke timing.
        gpu_smoke_timing: Optional GPU smoke timing.

    """

    import_timing: CommandTiming | None
    smoke_timing: CommandTiming | None
    bgen_reader_timing: CommandTiming | None
    gpu_smoke_timing: CommandTiming | None


PROFILE_SPECS: dict[BuildProfileLabel, BuildProfileSpec] = {
    BuildProfileLabel.DEV_FAST: BuildProfileSpec(
        label=BuildProfileLabel.DEV_FAST,
        cargo_profile="dev-fast",
        rustflags="",
    ),
    BuildProfileLabel.DEV_FAST_LLD: BuildProfileSpec(
        label=BuildProfileLabel.DEV_FAST_LLD,
        cargo_profile="dev-fast",
        rustflags="-C link-arg=-fuse-ld=lld",
    ),
    BuildProfileLabel.DEV_FAST_MOLD: BuildProfileSpec(
        label=BuildProfileLabel.DEV_FAST_MOLD,
        cargo_profile="dev-fast",
        rustflags="-C link-arg=-fuse-ld=mold",
    ),
    BuildProfileLabel.DEV_OPT: BuildProfileSpec(
        label=BuildProfileLabel.DEV_OPT,
        cargo_profile="dev-opt",
        rustflags="",
    ),
    BuildProfileLabel.RELEASE: BuildProfileSpec(
        label=BuildProfileLabel.RELEASE,
        cargo_profile="release",
        rustflags="",
    ),
    BuildProfileLabel.PERF_THIN_CGU8: BuildProfileSpec(
        label=BuildProfileLabel.PERF_THIN_CGU8,
        cargo_profile="perf",
        rustflags="-C target-cpu=native",
    ),
    BuildProfileLabel.PERF_THIN_CGU8_LLD: BuildProfileSpec(
        label=BuildProfileLabel.PERF_THIN_CGU8_LLD,
        cargo_profile="perf",
        rustflags="-C target-cpu=native -C link-arg=-fuse-ld=lld",
    ),
    BuildProfileLabel.PERF_THIN_CGU8_MOLD: BuildProfileSpec(
        label=BuildProfileLabel.PERF_THIN_CGU8_MOLD,
        cargo_profile="perf",
        rustflags="-C target-cpu=native -C link-arg=-fuse-ld=mold",
    ),
    BuildProfileLabel.PERF_THIN_CGU1: BuildProfileSpec(
        label=BuildProfileLabel.PERF_THIN_CGU1,
        cargo_profile="perf-thin-cgu1",
        rustflags="-C target-cpu=native",
    ),
    BuildProfileLabel.PERF_FAT_CGU1: BuildProfileSpec(
        label=BuildProfileLabel.PERF_FAT_CGU1,
        cargo_profile="perf-max",
        rustflags="-C target-cpu=native",
    ),
    BuildProfileLabel.PERF_O2_THIN_CGU8: BuildProfileSpec(
        label=BuildProfileLabel.PERF_O2_THIN_CGU8,
        cargo_profile="perf-o2",
        rustflags="-C target-cpu=native",
    ),
    BuildProfileLabel.PERF_O3_THIN_CGU8: BuildProfileSpec(
        label=BuildProfileLabel.PERF_O3_THIN_CGU8,
        cargo_profile="perf",
        rustflags="-C target-cpu=native",
    ),
}
IMPORT_TIMING_PROGRAM = (
    "import time; start_time = time.perf_counter(); import g._core; print(time.perf_counter() - start_time)"
)
CARGO_BUILD_JOBS_ENVIRONMENT_VARIABLE = "CARGO_BUILD_JOBS"


def parse_profile_labels(raw_labels: typing.Any) -> tuple[BuildProfileLabel, ...]:
    """Parse profile labels from Hydra scalar or sequence values.

    Args:
        raw_labels: Hydra-resolved label value.

    Returns:
        Parsed profile labels.

    """
    if isinstance(raw_labels, str):
        values = [value.strip() for value in raw_labels.split(",") if value.strip()]
    else:
        values = [str(value) for value in typing.cast("collections.abc.Sequence[typing.Any]", raw_labels)]
    return tuple(BuildProfileLabel(value) for value in values)


def parse_command_arguments(raw_values: typing.Any) -> tuple[str, ...]:
    """Parse a command argument vector from Hydra values.

    Args:
        raw_values: Hydra-resolved list or string value.

    Returns:
        Command argument tuple.

    """
    if raw_values is None:
        return ()
    if isinstance(raw_values, str):
        return tuple(raw_values.split())
    return tuple(str(value) for value in typing.cast("collections.abc.Sequence[typing.Any]", raw_values))


def build_arguments_from_config(config: omegaconf.DictConfig) -> RustBuildProfilesArguments:
    """Build benchmark arguments from a composed Hydra config.

    Args:
        config: Composed Hydra configuration.

    Returns:
        Resolved benchmark arguments.

    """
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return RustBuildProfilesArguments(
        labels=parse_profile_labels(tool_values["labels"]),
        output_parent=Path(str(tool_values["output_parent"])),
        clean_build=bool(tool_values["clean_build"]),
        incremental_touch_paths=tuple(Path(str(path)) for path in tool_values["incremental_touch_paths"]),
        build_timeout_seconds=tooling_hydra_arguments.float_or_none(tool_values["build_timeout_seconds"]),
        runtime_timeout_seconds=tooling_hydra_arguments.float_or_none(tool_values["runtime_timeout_seconds"]),
        run_import_timing=tooling_hydra_arguments.boolean_value(tool_values["run_import_timing"]),
        run_smoke_command=tooling_hydra_arguments.boolean_value(tool_values["run_smoke_command"]),
        smoke_command=parse_command_arguments(tool_values["smoke_command"]),
        run_bgen_reader_smoke=tooling_hydra_arguments.boolean_value(tool_values["run_bgen_reader_smoke"]),
        bgen_reader_command=parse_command_arguments(tool_values["bgen_reader_command"]),
        run_gpu_smoke=tooling_hydra_arguments.boolean_value(tool_values["run_gpu_smoke"]),
        gpu_smoke_command=parse_command_arguments(tool_values["gpu_smoke_command"]),
    )


def build_arguments_from_overrides(
    overrides: typing.Sequence[str] | None = None,
) -> RustBuildProfilesArguments:
    """Compose the Rust build profile config and return resolved arguments.

    Args:
        overrides: Optional Hydra override list.

    Returns:
        Resolved benchmark arguments.

    """
    config = tooling_configuration.compose_config(config_name="rust_build_profiles", overrides=overrides)
    return build_arguments_from_config(config)


def utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def report_timestamp() -> str:
    """Return an ISO-8601 UTC report timestamp."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def resolve_output_directory(arguments: RustBuildProfilesArguments, repository_root: Path) -> Path:
    """Resolve a timestamped output directory for one benchmark run.

    Args:
        arguments: Resolved harness arguments.
        repository_root: Repository root.

    Returns:
        Output directory path.

    """
    output_parent = tooling_paths.resolve_repo_relative_path(arguments.output_parent, repository_root)
    return output_parent / utc_timestamp()


def build_environment(spec: BuildProfileSpec, target_directory: Path) -> dict[str, str]:
    """Build command environment overrides for one profile.

    Args:
        spec: Build profile spec.
        target_directory: Isolated Cargo target directory.

    Returns:
        Environment overrides.

    """
    environment = {"CARGO_TARGET_DIR": str(target_directory)}
    if spec.rustflags:
        existing_rustflags = os.environ.get("RUSTFLAGS", "")
        environment["RUSTFLAGS"] = " ".join(value for value in (existing_rustflags, spec.rustflags) if value)
    return environment


def maturin_develop_job_arguments() -> tuple[str, ...]:
    """Return the explicit Maturin job-count arguments for the configured build."""
    job_count = os.environ.get(CARGO_BUILD_JOBS_ENVIRONMENT_VARIABLE)
    if job_count is None or not job_count.strip():
        return ()
    return ("-j", job_count.strip())


def maturin_develop_command(spec: BuildProfileSpec) -> tuple[str, ...]:
    """Return the Maturin develop command for one profile.

    Args:
        spec: Build profile spec.

    Returns:
        Command argument tuple.

    """
    return (
        "uv",
        "run",
        "--no-sync",
        "maturin",
        "develop",
        *maturin_develop_job_arguments(),
        "--profile",
        spec.cargo_profile,
        "--uv",
    )


def run_timed_command(
    *,
    name: str,
    command_arguments: tuple[str, ...],
    repository_root: Path,
    environment: dict[str, str],
    timeout_seconds: float | None,
    log_directory: Path,
) -> CommandTiming:
    """Run one command and capture timing plus logs.

    Args:
        name: Stable stage name.
        command_arguments: Command argument vector.
        repository_root: Command working directory.
        environment: Environment overrides.
        timeout_seconds: Optional command timeout.
        log_directory: Directory for captured stdout/stderr.

    Returns:
        Command timing report.

    """
    stdout_path = log_directory / f"{name}.stdout.log"
    stderr_path = log_directory / f"{name}.stderr.log"
    command_spec = tooling_commands.build_command_spec(
        command_arguments,
        cwd=repository_root,
        env=environment,
        timeout_seconds=timeout_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stream=True,
        sensitive_env_keys=("RUSTFLAGS",),
    )
    start_time = time.perf_counter()
    result = tooling_commands.run_command(command_spec)
    duration_seconds = time.perf_counter() - start_time
    return CommandTiming(
        name=name,
        command_arguments=result.args,
        return_code=result.return_code,
        duration_seconds=duration_seconds,
        timed_out=result.timed_out,
        missing_executable=result.missing_executable,
        stdout_log_path=str(stdout_path),
        stderr_log_path=str(stderr_path),
    )


def capture_timestamp(path: Path) -> SourceTimestamp:
    """Capture one source file timestamp.

    Args:
        path: Source file path.

    Returns:
        Timestamp snapshot.

    """
    stat_result = path.stat()
    return SourceTimestamp(
        path=path,
        access_time_nanoseconds=stat_result.st_atime_ns,
        modification_time_nanoseconds=stat_result.st_mtime_ns,
    )


def restore_timestamp(timestamp: SourceTimestamp) -> None:
    """Restore one source file timestamp.

    Args:
        timestamp: Timestamp snapshot.

    """
    os.utime(
        timestamp.path,
        ns=(timestamp.access_time_nanoseconds, timestamp.modification_time_nanoseconds),
    )


def touch_source_path(path: Path) -> SourceTimestamp:
    """Touch a source path and return its original timestamp.

    Args:
        path: Source file path.

    Returns:
        Original timestamp.

    """
    timestamp = capture_timestamp(path)
    os.utime(path, ns=(timestamp.access_time_nanoseconds, time.time_ns()))
    return timestamp


def find_extension_size(repository_root: Path, target_directory: Path) -> int | None:
    """Return the largest observed native extension artifact size.

    Args:
        repository_root: Repository root.
        target_directory: Isolated Cargo target directory.

    Returns:
        Largest artifact size, or None when no extension artifact is found.

    """
    candidate_paths = list((repository_root / "src" / "g").glob("_core*.so"))
    candidate_paths.extend(target_directory.rglob("*_core*.so"))
    candidate_paths.extend(target_directory.rglob("lib_core*.so"))
    sizes = [candidate_path.stat().st_size for candidate_path in candidate_paths if candidate_path.is_file()]
    if not sizes:
        return None
    return max(sizes)


def build_runtime_command_reports(
    *,
    arguments: RustBuildProfilesArguments,
    repository_root: Path,
    environment: dict[str, str],
    log_directory: Path,
) -> RuntimeCommandReports:
    """Run configured runtime smoke commands.

    Args:
        arguments: Resolved harness arguments.
        repository_root: Repository root.
        environment: Environment overrides.
        log_directory: Directory for captured logs.

    Returns:
        Runtime command timing reports.

    """
    import_timing = None
    smoke_timing = None
    bgen_reader_timing = None
    gpu_smoke_timing = None
    if arguments.run_import_timing:
        import_timing = run_timed_command(
            name="import",
            command_arguments=(
                "uv",
                "run",
                "--no-sync",
                "python",
                "-c",
                IMPORT_TIMING_PROGRAM,
            ),
            repository_root=repository_root,
            environment=environment,
            timeout_seconds=arguments.runtime_timeout_seconds,
            log_directory=log_directory,
        )
    if arguments.run_smoke_command and arguments.smoke_command:
        smoke_timing = run_timed_command(
            name="smoke",
            command_arguments=arguments.smoke_command,
            repository_root=repository_root,
            environment=environment,
            timeout_seconds=arguments.runtime_timeout_seconds,
            log_directory=log_directory,
        )
    if arguments.run_bgen_reader_smoke and arguments.bgen_reader_command:
        bgen_reader_timing = run_timed_command(
            name="bgen-reader",
            command_arguments=arguments.bgen_reader_command,
            repository_root=repository_root,
            environment=environment,
            timeout_seconds=arguments.runtime_timeout_seconds,
            log_directory=log_directory,
        )
    if arguments.run_gpu_smoke and arguments.gpu_smoke_command:
        gpu_smoke_timing = run_timed_command(
            name="gpu-smoke",
            command_arguments=arguments.gpu_smoke_command,
            repository_root=repository_root,
            environment=environment,
            timeout_seconds=arguments.runtime_timeout_seconds,
            log_directory=log_directory,
        )
    return RuntimeCommandReports(
        import_timing=import_timing,
        smoke_timing=smoke_timing,
        bgen_reader_timing=bgen_reader_timing,
        gpu_smoke_timing=gpu_smoke_timing,
    )


def benchmark_profile(
    *,
    arguments: RustBuildProfilesArguments,
    spec: BuildProfileSpec,
    repository_root: Path,
    output_directory: Path,
) -> ProfileBuildReport:
    """Benchmark one build profile.

    Args:
        arguments: Resolved harness arguments.
        spec: Build profile spec.
        repository_root: Repository root.
        output_directory: Report output directory.

    Returns:
        Per-profile build report.

    """
    target_directory = repository_root / "target" / "rust-build-profiles" / spec.label.value
    log_directory = output_directory / "logs" / spec.label.value
    environment = build_environment(spec, target_directory)
    if arguments.clean_build and target_directory.exists():
        shutil.rmtree(target_directory)
    clean_build = run_timed_command(
        name="clean-build",
        command_arguments=maturin_develop_command(spec),
        repository_root=repository_root,
        environment=environment,
        timeout_seconds=arguments.build_timeout_seconds,
        log_directory=log_directory,
    )
    incremental_builds: list[IncrementalBuildTiming] = []
    if clean_build.return_code == 0:
        for touch_path in arguments.incremental_touch_paths:
            source_path = tooling_paths.resolve_repo_relative_path(touch_path, repository_root)
            source_timestamp = touch_source_path(source_path)
            try:
                command_timing = run_timed_command(
                    name=f"incremental-{source_path.stem}",
                    command_arguments=maturin_develop_command(spec),
                    repository_root=repository_root,
                    environment=environment,
                    timeout_seconds=arguments.build_timeout_seconds,
                    log_directory=log_directory,
                )
            finally:
                restore_timestamp(source_timestamp)
            incremental_builds.append(
                IncrementalBuildTiming(
                    touched_path=str(source_path.relative_to(repository_root)),
                    command_timing=command_timing,
                )
            )
    import_timing = None
    smoke_timing = None
    bgen_reader_timing = None
    gpu_smoke_timing = None
    extension_size_bytes = None
    if clean_build.return_code == 0:
        extension_size_bytes = find_extension_size(repository_root, target_directory)
        runtime_command_reports = build_runtime_command_reports(
            arguments=arguments,
            repository_root=repository_root,
            environment=environment,
            log_directory=log_directory,
        )
        import_timing = runtime_command_reports.import_timing
        smoke_timing = runtime_command_reports.smoke_timing
        bgen_reader_timing = runtime_command_reports.bgen_reader_timing
        gpu_smoke_timing = runtime_command_reports.gpu_smoke_timing
    return ProfileBuildReport(
        label=spec.label.value,
        cargo_profile=spec.cargo_profile,
        rustflags=spec.rustflags,
        target_directory=str(target_directory),
        clean_build=clean_build,
        incremental_builds=incremental_builds,
        extension_size_bytes=extension_size_bytes,
        import_timing=import_timing,
        smoke_timing=smoke_timing,
        bgen_reader_timing=bgen_reader_timing,
        gpu_smoke_timing=gpu_smoke_timing,
    )


def command_status_text(command_timing: CommandTiming | None) -> str:
    """Format command status for Markdown.

    Args:
        command_timing: Optional command timing.

    Returns:
        Compact status text.

    """
    if command_timing is None:
        return "skipped"
    if command_timing.return_code == 0:
        return f"{command_timing.duration_seconds:.3f}s"
    if command_timing.timed_out:
        return "timeout"
    if command_timing.missing_executable:
        return "missing"
    return f"failed({command_timing.return_code})"


def build_markdown_report(report: RustBuildProfilesReport) -> str:
    """Build a Markdown summary table.

    Args:
        report: Complete build profile report.

    Returns:
        Markdown report text.

    """
    lines = [
        "# Rust Build Profile Benchmark",
        "",
        f"Generated at: `{report.generated_at_utc}`",
        "",
        "| label | cargo profile | rustflags | clean | import | smoke | size bytes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for profile_report in report.reports:
        size_text = str(profile_report.extension_size_bytes) if profile_report.extension_size_bytes is not None else ""
        rustflags_text = f"`{profile_report.rustflags}`" if profile_report.rustflags else ""
        lines.append(
            "| "
            f"{profile_report.label} | "
            f"{profile_report.cargo_profile} | "
            f"{rustflags_text} | "
            f"{command_status_text(profile_report.clean_build)} | "
            f"{command_status_text(profile_report.import_timing)} | "
            f"{command_status_text(profile_report.smoke_timing)} | "
            f"{size_text} |"
        )
    return "\n".join(lines) + "\n"


def run_tool(arguments: RustBuildProfilesArguments) -> None:
    """Run the build profile benchmark harness.

    Args:
        arguments: Resolved harness arguments.

    """
    repository_root = tooling_paths.find_repository_root()
    output_directory = resolve_output_directory(arguments, repository_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    reports = [
        benchmark_profile(
            arguments=arguments,
            spec=PROFILE_SPECS[label],
            repository_root=repository_root,
            output_directory=output_directory,
        )
        for label in arguments.labels
    ]
    report = RustBuildProfilesReport(
        schema_version=1,
        generated_at_utc=report_timestamp(),
        repository_root=str(repository_root),
        reports=reports,
    )
    json_summary_path = output_directory / "summary.json"
    markdown_summary_path = output_directory / "summary.md"
    tooling_reports.write_json_report(json_summary_path, report, sort_keys=True)
    tooling_reports.write_markdown_report(markdown_summary_path, build_markdown_report(report))
    print(f"Wrote JSON summary: {json_summary_path}")
    print(f"Wrote Markdown summary: {markdown_summary_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="rust_build_profiles")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the build profile benchmark CLI through Hydra."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run the build profile benchmark CLI."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
