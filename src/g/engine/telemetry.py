"""Structured run telemetry helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types

TelemetryCounterValue = bool | float | int | None
TelemetryWriterCounters = dict[str, TelemetryCounterValue]
TelemetryCloseMetadata = dict[str, TelemetryWriterCounters]

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


def resolve_output_run_root(regenie_config: config.RegenieConfig) -> Path:
    """Resolve the shared output run root for telemetry defaults."""
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
    """Resolve diagnostics paths using documented log_dir defaults."""
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


def optional_path_from_native_payload(path_payload: object) -> Path | None:
    """Adapt an optional native path string to a Python path."""
    if path_payload is None:
        return None
    return Path(typing.cast("str", path_payload))


def native_mapping_payload(payload: object) -> dict[str, typing.Any]:
    """Adapt a native mapping payload to a mutable Python dictionary."""
    return dict(typing.cast("typing.Mapping[str, typing.Any]", payload))


def build_telemetry_session(regenie_config: config.RegenieConfig) -> TelemetrySession:
    """Build a telemetry session from normalized diagnostics config."""
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
