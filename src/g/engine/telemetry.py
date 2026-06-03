"""Structured run telemetry helpers."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
import typing
import uuid
from dataclasses import dataclass

from g import types

TELEMETRY_SCHEMA_VERSION = 1

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.interface import config


@dataclass(frozen=True)
class TelemetryPaths:
    """Resolved telemetry output paths for one run.

    Attributes:
        log_dir: Directory containing telemetry streams.
        event_file: Stable lifecycle and profiling event stream.
        progress_file: Low-volume progress event stream.
        trace_file: Optional high-volume native trace stream.
        profile_summary_json: Optional aggregate profile summary path.
        stage_timings_json: Optional detailed synchronized stage timings path.

    """

    log_dir: Path | None
    event_file: Path | None
    progress_file: Path | None
    trace_file: Path | None
    profile_summary_json: Path | None
    stage_timings_json: Path | None


@dataclass(frozen=True)
class JsonLineWriteRequest:
    """Buffered JSONL write request.

    Attributes:
        path: Destination JSONL file path.
        line: Serialized JSON line with trailing newline.

    """

    path: Path
    line: str


class BufferedJsonLineWriter:
    """Session-scoped buffered JSONL writer."""

    def __init__(self, *, queue_capacity: int = 8192) -> None:
        """Initialize and start the background writer."""
        self.write_queue: queue.Queue[JsonLineWriteRequest | None] = queue.Queue(maxsize=queue_capacity)
        self.lock = threading.Lock()
        self.closed = False
        self.error: Exception | None = None
        self.created_directories: set[Path] = set()
        self.output_files: dict[Path, typing.TextIO] = {}
        self.thread = threading.Thread(target=self.run, name="g-telemetry-jsonl-writer", daemon=True)
        self.thread.start()

    def write(self, path: Path, line: str) -> None:
        """Queue one JSONL write."""
        self.raise_background_error()
        with self.lock:
            if self.closed:
                return
            self.write_queue.put(JsonLineWriteRequest(path=path, line=line))

    def close(self) -> None:
        """Flush queued writes and close all open files."""
        with self.lock:
            if self.closed:
                self.raise_background_error()
                return
            self.closed = True
            self.write_queue.put(None)
        self.thread.join()
        self.raise_background_error()

    def run(self) -> None:
        """Drain the write queue until close is requested."""
        try:
            while True:
                write_request = self.write_queue.get()
                try:
                    if write_request is None:
                        return
                    output_file = self.get_output_file(write_request.path)
                    output_file.write(write_request.line)
                finally:
                    self.write_queue.task_done()
        except OSError as exception:
            self.error = exception
        finally:
            self.close_output_files()

    def get_output_file(self, path: Path) -> typing.TextIO:
        """Return an open output file for the destination path."""
        output_file = self.output_files.get(path)
        if output_file is not None:
            return output_file
        if path.parent not in self.created_directories:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.created_directories.add(path.parent)
        output_file = path.open("a", encoding="utf-8")
        self.output_files[path] = output_file
        return output_file

    def close_output_files(self) -> None:
        """Flush and close all open output files."""
        for output_file in self.output_files.values():
            try:
                output_file.flush()
                output_file.close()
            except OSError as exception:
                if self.error is None:
                    self.error = exception
        self.output_files.clear()

    def raise_background_error(self) -> None:
        """Raise any exception captured by the background writer."""
        if self.error is None:
            return
        message = "Buffered telemetry writer failed."
        raise RuntimeError(message) from self.error


class TelemetrySession:
    """Run-scoped structured telemetry writer."""

    def __init__(
        self,
        *,
        mode: types.TelemetryMode,
        paths: TelemetryPaths,
        progress_interval_seconds: float,
        progress_interval_chunks: int,
        run_id: str | None = None,
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
        self.buffered_json_line_writer = (
            BufferedJsonLineWriter()
            if self.mode
            in {
                types.TelemetryMode.PROFILE,
                types.TelemetryMode.TRACE,
            }
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

    def log_event(self, event: str, level: str = "info", **fields: object) -> None:
        """Write one structured lifecycle or profile event."""
        if not self.enabled:
            return
        payload = self.build_event_payload(event=event, level=level, **fields)
        self.write_json_line(self.paths.event_file, payload)

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
        self.write_json_line(self.paths.progress_file, payload)

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

    def write_json_line(self, path: Path | None, payload: dict[str, object]) -> None:
        """Append one JSON line when the destination path is configured."""
        if path is None:
            return
        line = f"{json.dumps(payload, sort_keys=True, default=str)}\n"
        if self.buffered_json_line_writer is not None:
            self.buffered_json_line_writer.write(path, line)
            return
        with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output_file:
                output_file.write(line)

    def close(self) -> None:
        """Flush buffered telemetry resources."""
        if self.buffered_json_line_writer is None:
            return
        self.buffered_json_line_writer.close()


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
    """Resolve diagnostics paths using documented log-dir defaults."""
    diagnostics_config = regenie_config.g_diagnostics
    log_dir = diagnostics_config.log_dir
    if log_dir is None and diagnostics_config.telemetry != types.TelemetryMode.OFF:
        log_dir = resolve_output_run_root(regenie_config) / "logs"
    event_file = diagnostics_config.log_file
    if event_file is None and log_dir is not None and diagnostics_config.telemetry != types.TelemetryMode.OFF:
        event_file = log_dir / "events.jsonl"
    progress_file = None
    if log_dir is not None and diagnostics_config.telemetry != types.TelemetryMode.OFF:
        progress_file = log_dir / "progress.jsonl"
    trace_file = diagnostics_config.trace_file
    if trace_file is None and log_dir is not None and diagnostics_config.telemetry == types.TelemetryMode.TRACE:
        trace_file = log_dir / "trace.jsonl"
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
    stage_timings_json = diagnostics_config.stage_timings_json
    if (
        stage_timings_json is None
        and log_dir is not None
        and diagnostics_config.telemetry
        in {
            types.TelemetryMode.PROFILE,
            types.TelemetryMode.TRACE,
        }
    ):
        stage_timings_json = log_dir / "stage-timings.json"
    return TelemetryPaths(
        log_dir=log_dir,
        event_file=event_file,
        progress_file=progress_file,
        trace_file=trace_file,
        profile_summary_json=profile_summary_json,
        stage_timings_json=stage_timings_json,
    )


def build_telemetry_session(regenie_config: config.RegenieConfig) -> TelemetrySession:
    """Build a telemetry session from normalized diagnostics config."""
    diagnostics_config = regenie_config.g_diagnostics
    return TelemetrySession(
        mode=diagnostics_config.telemetry,
        paths=resolve_telemetry_paths(regenie_config),
        progress_interval_seconds=diagnostics_config.progress_interval_seconds,
        progress_interval_chunks=diagnostics_config.progress_interval_chunks,
    )


def close_telemetry_session(telemetry_session: TelemetrySession | None) -> None:
    """Flush best-effort telemetry teardown hooks."""
    if telemetry_session is None:
        return
    with contextlib.suppress(Exception):
        telemetry_session.log_event("telemetry_session_closed", level="debug")
    with contextlib.suppress(Exception):
        telemetry_session.close()
