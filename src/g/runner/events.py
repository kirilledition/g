"""Runner-owned run-event and telemetry helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types

if typing.TYPE_CHECKING:
    from g.interface import config
    from g.runner import lifecycle

TelemetryCounterValue = bool | float | int | None
TelemetryWriterCounters = dict[str, TelemetryCounterValue]
TelemetryCloseMetadata = dict[str, TelemetryWriterCounters]


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

    @property
    def close_metadata(self) -> TelemetryCloseMetadata | None:
        """Return close metadata captured by the native telemetry handle."""
        metadata = self.native_session_handle.close_metadata()
        if metadata is None:
            return None
        return typing.cast("TelemetryCloseMetadata", dict(metadata))

    def close(self) -> TelemetryCloseMetadata | None:
        """Flush buffered telemetry resources."""
        metadata = self.native_session_handle.finish_close_metadata()
        if metadata is None:
            return None
        return typing.cast("TelemetryCloseMetadata", dict(metadata))


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


@dataclass(frozen=True)
class RunArtifactPayload:
    """Structured payload for one phenotype's user-visible artifacts.

    Attributes:
        phenotype_name: Phenotype column represented by these artifacts.
        output_run_directory: Chunked output run directory.
        final_dataset: Parquet dataset directory for part-based output.
        final_parquet: Finalized Parquet output path.
        final_regenie: Finalized REGENIE-compatible text output path.
        effective_config: Written effective TOML config path.

    """

    phenotype_name: str | None
    output_run_directory: Path | None
    final_dataset: Path | None
    final_parquet: Path | None
    final_regenie: Path | None
    effective_config: Path | None


@dataclass(frozen=True)
class RunCompletedEvent:
    """Canonical completion event for telemetry and terminal rendering.

    Attributes:
        run_id: Diagnostics run identifier when available.
        association_mode: Statistical association engine used by the run.
        phenotype_count: Number of phenotypes included in the run.
        artifacts: User-visible artifacts produced by the run.

    """

    run_id: str | None
    association_mode: types.AssociationMode | None
    phenotype_count: int | None
    artifacts: tuple[RunArtifactPayload, ...]


@dataclass(frozen=True)
class RunInterruptedEvent:
    """Canonical graceful-interruption event.

    Attributes:
        signal_number: POSIX signal number.
        signal_name: POSIX signal name.
        exit_code: Conventional process exit code for the signal.
        flushed_for_resume: Whether committed output was flushed for resume.

    """

    signal_number: int
    signal_name: str
    exit_code: int
    flushed_for_resume: bool


@dataclass(frozen=True)
class RunFailedEvent:
    """Canonical non-graceful failure event.

    Attributes:
        error_type: Exception class name.
        error_message: Exception message.

    """

    error_type: str
    error_message: str


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


def close_telemetry_session(telemetry_session: TelemetrySession | None) -> None:
    """Flush native telemetry teardown hooks and preserve close failures."""
    native_telemetry_close_policy().close_telemetry_session_with_event(telemetry_session)


def resolve_output_run_root(regenie_config: config.RegenieConfig) -> Path:
    """Return the output root used for run-start telemetry."""
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    output_run_directory = regenie_config.g_output.output_run_directory
    telemetry_policy = native_telemetry_session_policy(regenie_config)
    return Path(
        telemetry_policy.resolve_output_run_root_value(
            str(output_prefix),
            None if output_run_directory is None else str(output_run_directory),
        )
    )


def resolve_telemetry_paths(regenie_config: config.RegenieConfig) -> TelemetryPaths:
    """Resolve diagnostics paths using documented log-dir defaults."""
    diagnostics_config = regenie_config.g_diagnostics
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    output_run_directory = regenie_config.g_output.output_run_directory
    telemetry_policy = native_telemetry_session_policy(regenie_config)
    return telemetry_paths_from_native_payload(
        telemetry_policy.resolve_paths_payload(
            str(output_prefix),
            None if output_run_directory is None else str(output_run_directory),
            None if diagnostics_config.log_dir is None else str(diagnostics_config.log_dir),
            None if diagnostics_config.log_file is None else str(diagnostics_config.log_file),
            None if diagnostics_config.trace_file is None else str(diagnostics_config.trace_file),
            None if diagnostics_config.profile_summary_json is None else str(diagnostics_config.profile_summary_json),
            None if diagnostics_config.stage_timings_json is None else str(diagnostics_config.stage_timings_json),
        )
    )


def native_telemetry_session_policy(regenie_config: config.RegenieConfig) -> _core.NativeTelemetrySessionPolicy:
    """Build the native telemetry session policy for a run config."""
    diagnostics_config = regenie_config.g_diagnostics
    return _core.NativeTelemetrySessionPolicy(diagnostics_config.telemetry.value, diagnostics_config.trace_event_cap)


def native_telemetry_close_policy() -> _core.NativeTelemetryClosePolicy:
    """Build the native telemetry close policy handle."""
    return _core.NativeTelemetryClosePolicy()


def telemetry_paths_from_native_payload(payload: object) -> TelemetryPaths:
    """Adapt a native telemetry path payload to the public Python dataclass."""
    telemetry_paths_payload = native_mapping_payload(payload)
    return TelemetryPaths(
        log_dir=optional_path_from_native_payload(telemetry_paths_payload["log_dir"]),
        stream_file=optional_path_from_native_payload(telemetry_paths_payload["stream_file"]),
        profile_summary_json=optional_path_from_native_payload(telemetry_paths_payload["profile_summary_json"]),
        stage_timings_json=optional_path_from_native_payload(telemetry_paths_payload["stage_timings_json"]),
    )


def native_run_event_telemetry_policy() -> _core.NativeRunEventTelemetryPolicy:
    """Build the native run-event telemetry policy handle."""
    return _core.NativeRunEventTelemetryPolicy()


def native_runner_diagnostic_policy() -> _core.NativeRunnerDiagnosticPolicy:
    """Build the native runner diagnostic policy handle."""
    return _core.NativeRunnerDiagnosticPolicy()


def native_pipeline_diagnostic_policy() -> _core.NativePipelineDiagnosticPolicy:
    """Build the native pipeline diagnostic policy handle."""
    return _core.NativePipelineDiagnosticPolicy()


def native_dispatch_diagnostic_policy() -> _core.NativeDispatchDiagnosticPolicy:
    """Build the native-dispatch diagnostic policy handle."""
    return _core.NativeDispatchDiagnosticPolicy()


def native_run_event_payload_policy() -> _core.NativeRunEventPayloadPolicy:
    """Build the native run-event payload policy handle."""
    return _core.NativeRunEventPayloadPolicy()


def build_run_interrupted_event(shutdown_request: lifecycle.GracefulShutdownRequested) -> RunInterruptedEvent:
    """Build a structured interruption event from a graceful shutdown request."""
    return run_interrupted_event_from_native_payload(
        native_run_event_payload_policy().build_run_interrupted_event_payload(shutdown_request)
    )


def build_run_failed_event(error: Exception) -> RunFailedEvent:
    """Build a structured failure event from an exception."""
    return run_failed_event_from_native_payload(native_run_event_payload_policy().build_run_failed_event_payload(error))


def attach_run_metadata(
    artifacts: RunArtifacts,
    *,
    run_id: str | None,
    association_mode: types.AssociationMode,
    phenotype_count: int,
) -> RunArtifacts:
    """Attach lifecycle metadata to returned run artifacts."""
    return run_artifacts_from_native_payload(
        native_run_event_payload_policy().attach_run_metadata_payload(
            artifacts,
            run_id,
            association_mode.value,
            phenotype_count,
        )
    )


def build_run_completed_event(artifacts: RunArtifacts) -> RunCompletedEvent:
    """Build a structured completion event from run artifacts."""
    return run_completed_event_from_native_payload(
        native_run_event_payload_policy().build_run_completed_event_payload(artifacts)
    )


def render_run_interrupted_lines(interrupted_event: RunInterruptedEvent) -> tuple[str, ...]:
    """Render graceful interruption lines for CLI output."""
    return tuple(native_run_event_payload_policy().render_run_interrupted_lines(interrupted_event))


def render_run_failed_lines(failed_event: RunFailedEvent) -> tuple[str, ...]:
    """Render run failure lines for CLI output."""
    return tuple(native_run_event_payload_policy().render_run_failed_lines(failed_event))


def render_run_completed_lines(completed_event: RunCompletedEvent) -> tuple[str, ...]:
    """Render run completion lines for CLI output."""
    return tuple(native_run_event_payload_policy().render_run_completed_lines(completed_event))


def run_artifacts_from_native_payload(payload: object) -> RunArtifacts:
    """Adapt a native artifact tree payload to the public Python dataclass."""
    artifacts_payload = native_mapping_payload(payload)
    association_mode_payload = typing.cast("str | None", artifacts_payload["association_mode"])
    return RunArtifacts(
        output_run_directory=optional_path_from_native_payload(artifacts_payload["output_run_directory"]),
        final_dataset=optional_path_from_native_payload(artifacts_payload["final_dataset"]),
        final_parquet=optional_path_from_native_payload(artifacts_payload["final_parquet"]),
        final_regenie=optional_path_from_native_payload(artifacts_payload["final_regenie"]),
        effective_config=optional_path_from_native_payload(artifacts_payload["effective_config"]),
        phenotype_artifacts=tuple(
            run_artifacts_from_native_payload(phenotype_artifact_payload)
            for phenotype_artifact_payload in typing.cast(
                "typing.Sequence[object]",
                artifacts_payload["phenotype_artifacts"],
            )
        ),
        phenotype_name=typing.cast("str | None", artifacts_payload["phenotype_name"]),
        association_mode=None if association_mode_payload is None else types.AssociationMode(association_mode_payload),
        phenotype_count=typing.cast("int | None", artifacts_payload["phenotype_count"]),
        run_id=typing.cast("str | None", artifacts_payload["run_id"]),
    )


def run_completed_event_from_native_payload(payload: object) -> RunCompletedEvent:
    """Adapt a native completed-run event payload to the public Python dataclass."""
    event_payload = native_mapping_payload(payload)
    association_mode_payload = typing.cast("str | None", event_payload["association_mode"])
    return RunCompletedEvent(
        run_id=typing.cast("str | None", event_payload["run_id"]),
        association_mode=None if association_mode_payload is None else types.AssociationMode(association_mode_payload),
        phenotype_count=typing.cast("int | None", event_payload["phenotype_count"]),
        artifacts=tuple(
            run_artifact_payload_from_native_payload(artifact_payload)
            for artifact_payload in typing.cast("typing.Sequence[object]", event_payload["artifacts"])
        ),
    )


def run_artifact_payload_from_native_payload(payload: object) -> RunArtifactPayload:
    """Adapt one native completed-run artifact payload."""
    artifact_payload = native_mapping_payload(payload)
    return RunArtifactPayload(
        phenotype_name=typing.cast("str | None", artifact_payload["phenotype_name"]),
        output_run_directory=optional_path_from_native_payload(artifact_payload["output_run_directory"]),
        final_dataset=optional_path_from_native_payload(artifact_payload["final_dataset"]),
        final_parquet=optional_path_from_native_payload(artifact_payload["final_parquet"]),
        final_regenie=optional_path_from_native_payload(artifact_payload["final_regenie"]),
        effective_config=optional_path_from_native_payload(artifact_payload["effective_config"]),
    )


def run_interrupted_event_from_native_payload(payload: object) -> RunInterruptedEvent:
    """Adapt a native interrupted-run event payload."""
    event_payload = native_mapping_payload(payload)
    return RunInterruptedEvent(
        signal_number=typing.cast("int", event_payload["signal_number"]),
        signal_name=typing.cast("str", event_payload["signal_name"]),
        exit_code=typing.cast("int", event_payload["exit_code"]),
        flushed_for_resume=typing.cast("bool", event_payload["flushed_for_resume"]),
    )


def run_failed_event_from_native_payload(payload: object) -> RunFailedEvent:
    """Adapt a native failed-run event payload."""
    event_payload = native_mapping_payload(payload)
    return RunFailedEvent(
        error_type=typing.cast("str", event_payload["error_type"]),
        error_message=typing.cast("str", event_payload["error_message"]),
    )


def optional_path_from_native_payload(path_payload: object) -> Path | None:
    """Adapt an optional native path string."""
    if path_payload is None:
        return None
    return Path(typing.cast("str", path_payload))


def native_mapping_payload(payload: object) -> dict[str, typing.Any]:
    """Adapt a native mapping payload to a mutable Python dictionary."""
    return dict(typing.cast("typing.Mapping[str, typing.Any]", payload))
