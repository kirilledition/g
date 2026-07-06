"""Runner-local run-event and telemetry helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types

if typing.TYPE_CHECKING:
    from g.interface import config


@dataclass(frozen=True)
class TelemetryPaths:
    """Resolved telemetry output paths for one run.

    Attributes:
        log_dir: Directory containing telemetry streams.
        stream_file: Unified JSONL event stream.
        profile_summary_json: Optional aggregate profile summary path.
        stage_timings_json: Optional detailed synchronized stage timings path.

    """

    log_dir: Path | None
    stream_file: Path | None
    profile_summary_json: Path | None
    stage_timings_json: Path | None


class TelemetrySession:
    """Run-scoped structured telemetry writer."""

    def __init__(
        self,
        *,
        mode: types.TelemetryMode,
        paths: TelemetryPaths,
        progress_interval_seconds: float,
        progress_interval_chunks: int,
        queue_size: int,
        lossy: bool,
        trace_event_cap: int,
        run_id: str | None,
    ) -> None:
        """Initialize a run telemetry session."""
        self.mode = mode
        self.paths = paths
        self.native_session_handle = _core.NativeTelemetryRunSession(
            telemetry_mode=mode.value,
            stream_file=None if paths.stream_file is None else str(paths.stream_file),
            progress_interval_seconds=progress_interval_seconds,
            progress_interval_chunks=progress_interval_chunks,
            queue_size=queue_size,
            lossy=lossy,
            trace_event_cap=trace_event_cap,
            run_id=run_id,
        )

    @property
    def enabled(self) -> bool:
        """Return whether this session writes telemetry."""
        return self.native_session_handle.enabled

    @property
    def profile_enabled(self) -> bool:
        """Return whether profiling-grade telemetry is enabled."""
        return self.native_session_handle.profile_enabled

    @property
    def run_id(self) -> str:
        """Return the native run identifier."""
        return self.native_session_handle.run_id

    @property
    def native_telemetry_session(self) -> _core.NativeTelemetryRunSession | None:
        """Return the native session handle when a writer is configured."""
        if not self.native_session_handle.has_native_telemetry_session:
            return None
        return self.native_session_handle


@dataclass(frozen=True)
class RunArtifacts:
    """Immutable pointers to generated output files.

    Attributes:
        output_run_directory: Chunked output run directory.
        final_dataset: Parquet dataset directory for part-based output.
        final_parquet: Finalized Parquet output path.
        final_regenie: Finalized REGENIE-compatible text output path.
        effective_config: Written effective TOML config path.
        phenotype_artifacts: Per-phenotype artifacts for multi-phenotype runs.
        phenotype_name: Phenotype column represented by this artifact set.
        association_mode: Statistical association engine used by the run.
        phenotype_count: Number of phenotypes included in the run.
        run_id: Diagnostics run identifier when telemetry created one.

    """

    output_run_directory: Path | None
    final_dataset: Path | None
    final_parquet: Path | None
    final_regenie: Path | None
    effective_config: Path | None
    phenotype_artifacts: tuple[RunArtifacts, ...]
    phenotype_name: str | None
    association_mode: types.AssociationMode | None
    phenotype_count: int | None
    run_id: str | None


def build_telemetry_session(regenie_config: config.RegenieConfig) -> TelemetrySession:
    """Build the run telemetry session for one runner invocation."""
    diagnostics_config = regenie_config.g_diagnostics
    return TelemetrySession(
        mode=diagnostics_config.telemetry,
        paths=resolve_telemetry_paths(regenie_config),
        progress_interval_seconds=diagnostics_config.progress_interval_seconds,
        progress_interval_chunks=diagnostics_config.progress_interval_chunks,
        queue_size=diagnostics_config.log_queue_size,
        lossy=diagnostics_config.log_lossy,
        trace_event_cap=diagnostics_config.trace_event_cap,
        run_id=None,
    )


def resolve_telemetry_paths(regenie_config: config.RegenieConfig) -> TelemetryPaths:
    """Resolve diagnostics paths using documented log-dir defaults."""
    diagnostics_config = regenie_config.g_diagnostics
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    output_run_directory = regenie_config.g_output.output_run_directory
    native_paths = _core.resolve_telemetry_paths(
        str(output_prefix),
        None if output_run_directory is None else str(output_run_directory),
        diagnostics_config.telemetry.value,
        None if diagnostics_config.log_dir is None else str(diagnostics_config.log_dir),
        None if diagnostics_config.log_file is None else str(diagnostics_config.log_file),
        None if diagnostics_config.trace_file is None else str(diagnostics_config.trace_file),
        None if diagnostics_config.profile_summary_json is None else str(diagnostics_config.profile_summary_json),
        None if diagnostics_config.stage_timings_json is None else str(diagnostics_config.stage_timings_json),
    )
    return TelemetryPaths(
        log_dir=optional_path_from_native_value(native_paths.log_dir),
        stream_file=optional_path_from_native_value(native_paths.stream_file),
        profile_summary_json=optional_path_from_native_value(native_paths.profile_summary_json),
        stage_timings_json=optional_path_from_native_value(native_paths.stage_timings_json),
    )


def run_artifacts_from_native_artifacts(native_artifacts: _core.NativeRunArtifacts) -> RunArtifacts:
    """Adapt a native artifact tree to the public Python dataclass."""
    association_mode_payload = native_artifacts.association_mode
    return RunArtifacts(
        output_run_directory=optional_path_from_native_value(native_artifacts.output_run_directory),
        final_dataset=optional_path_from_native_value(native_artifacts.final_dataset),
        final_parquet=optional_path_from_native_value(native_artifacts.final_parquet),
        final_regenie=optional_path_from_native_value(native_artifacts.final_regenie),
        effective_config=optional_path_from_native_value(native_artifacts.effective_config),
        phenotype_artifacts=tuple(
            run_artifacts_from_native_artifacts(phenotype_artifact)
            for phenotype_artifact in native_artifacts.phenotype_artifacts
        ),
        phenotype_name=native_artifacts.phenotype_name,
        association_mode=None if association_mode_payload is None else types.AssociationMode(association_mode_payload),
        phenotype_count=native_artifacts.phenotype_count,
        run_id=native_artifacts.run_id,
    )


def optional_path_from_native_value(path_payload: object) -> Path | None:
    """Adapt an optional native path string."""
    if path_payload is None:
        return None
    return Path(typing.cast("str", path_payload))
