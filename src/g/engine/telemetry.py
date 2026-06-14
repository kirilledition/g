"""Structured run telemetry helpers."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import typing
import uuid
from dataclasses import dataclass

from g import _core, types

TELEMETRY_SCHEMA_VERSION = 1

TelemetryCounterValue = bool | float | int | None
TelemetryWriterCounters = dict[str, TelemetryCounterValue]
TelemetryCloseMetadata = dict[str, TelemetryWriterCounters]

if typing.TYPE_CHECKING:
    from pathlib import Path

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
        self.progress_interval_seconds = progress_interval_seconds
        self.progress_interval_chunks = progress_interval_chunks
        self.run_id = run_id or uuid.uuid4().hex
        self.lock = threading.Lock()
        self.last_progress_time = 0.0
        self.last_progress_chunk_count = 0
        self.close_metadata: TelemetryCloseMetadata | None = None
        native_event_cap = trace_event_cap if mode == types.TelemetryMode.TRACE and trace_event_cap > 0 else None
        self.native_telemetry_session = (
            _core.NativeTelemetrySession(
                str(paths.stream_file),
                queue_size=queue_size,
                lossy=lossy,
                event_cap=native_event_cap,
            )
            if self.enabled and paths.stream_file is not None
            else None
        )

    @property
    def enabled(self) -> bool:
        """Return whether this session writes telemetry."""
        return self.mode != types.TelemetryMode.OFF

    @property
    def profile_enabled(self) -> bool:
        """Return whether profiling-grade telemetry is enabled."""
        return self.mode in {types.TelemetryMode.PROFILE, types.TelemetryMode.TRACE}

    def log_event(self, event: str, level: str, **fields: object) -> None:
        """Write one structured lifecycle or profile event."""
        if not self.enabled:
            return
        payload = self.build_event_payload(event=event, level=level, **fields)
        self.write_json_line(payload)

    def log_progress(self, *, processed_chunk_count: int, **fields: object) -> None:
        """Write throttled progress telemetry."""
        if self.mode == types.TelemetryMode.OFF:
            return
        if not self.should_emit_progress(processed_chunk_count):
            return
        payload = self.build_event_payload(
            event="progress_tick",
            level="info",
            processed_chunk_count=processed_chunk_count,
            **fields,
        )
        self.write_json_line(payload)

    def should_emit_progress(self, processed_chunk_count: int) -> bool:
        """Return whether a progress event should be emitted now."""
        current_time = time.monotonic()
        with self.lock:
            elapsed_seconds = current_time - self.last_progress_time
            elapsed_chunks = processed_chunk_count - self.last_progress_chunk_count
            if (
                self.last_progress_time > 0.0
                and elapsed_seconds < self.progress_interval_seconds
                and elapsed_chunks < self.progress_interval_chunks
            ):
                return False
            self.last_progress_time = current_time
            self.last_progress_chunk_count = processed_chunk_count
            return True

    def build_event_payload(self, *, event: str, level: str, **fields: object) -> dict[str, object]:
        """Build a schema-versioned telemetry event payload."""
        payload: dict[str, object] = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "ts": format_timestamp(time.time()),
            "level": level.upper(),
            "source": "python",
            "target": "g.engine.telemetry",
            "event": event,
            "pid": os.getpid(),
            "thread_name": threading.current_thread().name,
        }
        payload.update({key: value for key, value in fields.items() if value is not None})
        return payload

    def write_json_line(self, payload: dict[str, object]) -> None:
        """Append one JSON line when the destination path is configured."""
        if self.native_telemetry_session is None:
            return
        line = f"{json.dumps(payload, sort_keys=True, default=str)}\n"
        self.native_telemetry_session.emit_json_line(line)

    def writer_counters(self) -> TelemetryWriterCounters:
        """Return the current native telemetry writer counters."""
        if self.native_telemetry_session is None:
            return build_empty_writer_counters()
        return typing.cast("TelemetryWriterCounters", dict(self.native_telemetry_session.counters()))

    def close(self) -> TelemetryCloseMetadata | None:
        """Flush buffered telemetry resources."""
        if self.native_telemetry_session is None:
            self.close_metadata = None
            return None
        writer_counters = typing.cast("TelemetryWriterCounters", dict(self.native_telemetry_session.finish()))
        self.close_metadata = {"writer_counters": writer_counters}
        return self.close_metadata


def format_timestamp(timestamp_seconds: float) -> str:
    """Format a Unix timestamp as an RFC 3339 UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(timestamp_seconds)) + (
        f".{int(timestamp_seconds % 1 * 1_000_000):06d}Z"
    )


def resolve_output_run_root(regenie_config: config.RegenieConfig) -> Path:
    """Resolve the shared output run root for telemetry defaults."""
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    return regenie_config.g_output.output_run_directory or output_prefix.with_name(f"{output_prefix.name}.g")


def resolve_telemetry_paths(regenie_config: config.RegenieConfig) -> TelemetryPaths:
    """Resolve diagnostics paths using documented log_dir defaults."""
    diagnostics_config = regenie_config.g_diagnostics
    log_dir = diagnostics_config.log_dir
    if log_dir is None and diagnostics_config.telemetry != types.TelemetryMode.OFF:
        log_dir = resolve_output_run_root(regenie_config) / "logs"
    stream_file = resolve_telemetry_stream_file(
        telemetry_mode=diagnostics_config.telemetry,
        log_dir=log_dir,
        log_file=diagnostics_config.log_file,
        trace_file=diagnostics_config.trace_file,
    )
    profile_summary_json = diagnostics_config.profile_summary_json
    if (
        profile_summary_json is None
        and log_dir is not None
        and diagnostics_config.telemetry
        in {
            types.TelemetryMode.PROFILE,
            types.TelemetryMode.TRACE,
        }
    ):
        profile_summary_json = log_dir / "profile.summary.json"
    return TelemetryPaths(
        log_dir=log_dir,
        stream_file=stream_file,
        profile_summary_json=profile_summary_json,
        stage_timings_json=diagnostics_config.stage_timings_json,
    )


def resolve_telemetry_stream_file(
    *,
    telemetry_mode: types.TelemetryMode,
    log_dir: Path | None,
    log_file: Path | None,
    trace_file: Path | None,
) -> Path | None:
    """Resolve the unified telemetry stream file."""
    if telemetry_mode == types.TelemetryMode.OFF:
        return None
    if log_file is not None and trace_file is not None and not paths_refer_to_same_file(log_file, trace_file):
        message = "log_file and trace_file both configure the unified telemetry stream; use one path."
        raise ValueError(message)
    if log_file is not None:
        return log_file
    if trace_file is not None:
        return trace_file
    if log_dir is None:
        return None
    return log_dir / "events.jsonl"


def paths_refer_to_same_file(first_path: Path, second_path: Path) -> bool:
    """Return whether two paths resolve to the same filesystem target."""
    return first_path.resolve(strict=False) == second_path.resolve(strict=False)


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
    return {
        "accepted_event_count": 0,
        "written_event_count": 0,
        "dropped_event_count": 0,
        "cap_dropped_event_count": 0,
        "queue_dropped_event_count": 0,
        "event_cap_exceeded": False,
        "lossy": True,
        "event_cap": None,
        "finish_flush_duration_seconds": None,
    }


def close_telemetry_session(telemetry_session: TelemetrySession | None) -> None:
    """Flush telemetry teardown hooks and preserve close failures."""
    if telemetry_session is None:
        return
    with contextlib.suppress(Exception):
        telemetry_session.log_event(
            "telemetry_session_closed",
            level="debug",
            writer_counters=telemetry_session.writer_counters(),
        )
    telemetry_session.close()
