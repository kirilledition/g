"""Structured run telemetry helpers."""

from __future__ import annotations

import contextlib
import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types

TelemetryCounterValue = bool | float | int | None
TelemetryWriterCounters = dict[str, TelemetryCounterValue]
TelemetryCloseMetadata = dict[str, TelemetryWriterCounters]

if typing.TYPE_CHECKING:
    from g.interface import config


class TelemetryCloseableSession(typing.Protocol):
    """Telemetry session shape accepted by the close helper."""

    def log_event(self, event: str, level: str, **fields: object) -> None:
        """Write one telemetry event."""

    def writer_counters(self) -> object:
        """Return writer counters for close telemetry."""

    def close(self) -> object:
        """Close the telemetry resources."""


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
        self.run_id = run_id or _core.generate_telemetry_run_id_value()
        self.native_session_policy = _core.NativeTelemetrySessionPolicy(mode.value, trace_event_cap)
        self.native_progress_throttle = _core.NativeTelemetryProgressThrottle(
            progress_interval_seconds,
            progress_interval_chunks,
        )
        self.native_telemetry_session = (
            _core.NativeTelemetrySession(
                str(paths.stream_file),
                queue_size=queue_size,
                lossy=lossy,
                event_cap=self.native_session_policy.event_cap,
            )
            if self.enabled and paths.stream_file is not None
            else None
        )

    @property
    def enabled(self) -> bool:
        """Return whether this session writes telemetry."""
        return self.native_session_policy.enabled

    @property
    def profile_enabled(self) -> bool:
        """Return whether profiling-grade telemetry is enabled."""
        return self.native_session_policy.profile_enabled

    @property
    def close_metadata(self) -> TelemetryCloseMetadata | None:
        """Return close metadata captured by the native telemetry handle."""
        if self.native_telemetry_session is None:
            return None
        metadata = self.native_telemetry_session.close_metadata()
        if metadata is None:
            return None
        return typing.cast("TelemetryCloseMetadata", dict(metadata))

    def log_event(self, event: str, level: str, **fields: object) -> None:
        """Write one structured lifecycle or profile event."""
        emission_plan = _core.plan_telemetry_event_emission(
            telemetry_enabled=self.enabled,
            has_native_telemetry_session=self.native_telemetry_session is not None,
        )
        if not emission_plan.should_emit:
            return
        native_telemetry_session = typing.cast("_core.NativeTelemetrySession", self.native_telemetry_session)
        native_telemetry_session.emit_current_event(
            self.run_id,
            event,
            level,
            fields,
        )

    def log_progress(self, *, processed_chunk_count: int, **fields: object) -> None:
        """Write throttled progress telemetry."""
        should_emit_progress = self.enabled and self.should_emit_progress(processed_chunk_count)
        emission_plan = _core.plan_telemetry_progress_emission(
            telemetry_enabled=self.enabled,
            has_native_telemetry_session=self.native_telemetry_session is not None,
            should_emit_progress=should_emit_progress,
        )
        if not emission_plan.should_emit:
            return
        native_telemetry_session = typing.cast("_core.NativeTelemetrySession", self.native_telemetry_session)
        progress_fields = {"processed_chunk_count": processed_chunk_count, **fields}
        native_telemetry_session.emit_current_event(
            self.run_id,
            emission_plan.event_name,
            emission_plan.level,
            progress_fields,
        )

    def should_emit_progress(self, processed_chunk_count: int) -> bool:
        """Return whether a progress event should be emitted now."""
        return self.native_progress_throttle.should_emit_progress(processed_chunk_count)

    def build_event_payload(self, *, event: str, level: str, **fields: object) -> dict[str, object]:
        """Build a schema-versioned telemetry event payload."""
        return dict(
            _core.build_current_telemetry_event_payload(
                self.run_id,
                event,
                level,
                fields,
            )
        )

    def write_json_line(self, payload: dict[str, object]) -> None:
        """Append one JSON line when the destination path is configured."""
        if self.native_telemetry_session is None:
            return
        self.native_telemetry_session.emit_payload(payload)

    def writer_counters(self) -> TelemetryWriterCounters:
        """Return the current native telemetry writer counters."""
        if self.native_telemetry_session is None:
            return build_empty_writer_counters()
        return typing.cast("TelemetryWriterCounters", dict(self.native_telemetry_session.counters()))

    def close(self) -> TelemetryCloseMetadata | None:
        """Flush buffered telemetry resources."""
        if self.native_telemetry_session is None:
            return None
        writer_counters = typing.cast("TelemetryWriterCounters", dict(self.native_telemetry_session.finish()))
        return {"writer_counters": writer_counters}

    def close_with_event(self) -> TelemetryCloseMetadata | None:
        """Emit the close event and flush buffered telemetry resources."""
        if self.native_telemetry_session is None:
            return None
        writer_counters = typing.cast(
            "TelemetryWriterCounters",
            dict(self.native_telemetry_session.finish_with_current_close_event(self.run_id)),
        )
        return {"writer_counters": writer_counters}


def format_timestamp(timestamp_seconds: float) -> str:
    """Format a Unix timestamp as an RFC 3339 UTC timestamp."""
    return _core.format_telemetry_timestamp_value(timestamp_seconds)


def resolve_output_run_root(regenie_config: config.RegenieConfig) -> Path:
    """Resolve the shared output run root for telemetry defaults."""
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    output_run_directory = regenie_config.g_output.output_run_directory
    return Path(
        _core.resolve_telemetry_output_run_root_value(
            str(output_prefix),
            None if output_run_directory is None else str(output_run_directory),
        )
    )


def resolve_telemetry_paths(regenie_config: config.RegenieConfig) -> TelemetryPaths:
    """Resolve diagnostics paths using documented log_dir defaults."""
    diagnostics_config = regenie_config.g_diagnostics
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    output_run_directory = regenie_config.g_output.output_run_directory
    return telemetry_paths_from_native_payload(
        _core.resolve_telemetry_paths_payload(
            str(output_prefix),
            None if output_run_directory is None else str(output_run_directory),
            diagnostics_config.telemetry.value,
            None if diagnostics_config.log_dir is None else str(diagnostics_config.log_dir),
            None if diagnostics_config.log_file is None else str(diagnostics_config.log_file),
            None if diagnostics_config.trace_file is None else str(diagnostics_config.trace_file),
            None if diagnostics_config.profile_summary_json is None else str(diagnostics_config.profile_summary_json),
            None if diagnostics_config.stage_timings_json is None else str(diagnostics_config.stage_timings_json),
        )
    )


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


def resolve_telemetry_stream_file(
    *,
    telemetry_mode: types.TelemetryMode,
    log_dir: Path | None,
    log_file: Path | None,
    trace_file: Path | None,
) -> Path | None:
    """Resolve the unified telemetry stream file."""
    stream_file = _core.resolve_telemetry_stream_file_value(
        telemetry_mode.value,
        None if log_dir is None else str(log_dir),
        None if log_file is None else str(log_file),
        None if trace_file is None else str(trace_file),
    )
    return None if stream_file is None else Path(stream_file)


def paths_refer_to_same_file(first_path: Path, second_path: Path) -> bool:
    """Return whether two paths resolve to the same filesystem target."""
    return _core.paths_refer_to_same_file_value(str(first_path), str(second_path))


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


def build_empty_writer_counters() -> TelemetryWriterCounters:
    """Return a zeroed telemetry writer counter snapshot."""
    return typing.cast(
        "TelemetryWriterCounters",
        native_mapping_payload(_core.build_empty_telemetry_writer_counters_payload()),
    )


def close_telemetry_session(telemetry_session: TelemetryCloseableSession | None) -> None:
    """Flush telemetry teardown hooks and preserve close failures."""
    if telemetry_session is None:
        return
    if isinstance(telemetry_session, TelemetrySession):
        telemetry_session.close_with_event()
        return
    with contextlib.suppress(Exception):
        telemetry_session.log_event(
            "telemetry_session_closed",
            level="debug",
            writer_counters=telemetry_session.writer_counters(),
        )
    telemetry_session.close()
