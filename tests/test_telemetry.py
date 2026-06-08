from __future__ import annotations

import json
import typing

import pytest

from g import types
from g.engine import telemetry
from g.interface import config

if typing.TYPE_CHECKING:
    from pathlib import Path


def test_resolve_telemetry_paths_defaults_to_output_run_logs() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
        }
    )

    telemetry_paths = telemetry.resolve_telemetry_paths(regenie_config)
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    expected_log_dir = output_prefix.with_name("output.g") / "logs"

    assert telemetry_paths.log_dir == expected_log_dir
    assert telemetry_paths.stream_file == expected_log_dir / "events.jsonl"
    assert telemetry_paths.profile_summary_json is None
    assert telemetry_paths.stage_timings_json is None


def test_trace_telemetry_paths_default_profile_summary_without_exact_stage_timings() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "g-telemetry": "trace",
            "g-log-dir": "telemetry",
        }
    )

    telemetry_paths = telemetry.resolve_telemetry_paths(regenie_config)
    log_dir = typing.cast("Path", regenie_config.g_diagnostics.log_dir)

    assert telemetry_paths.stream_file == log_dir / "events.jsonl"
    assert telemetry_paths.profile_summary_json == log_dir / "profile.summary.json"
    assert telemetry_paths.stage_timings_json is None


def test_explicit_stage_timings_path_enables_exact_stage_output() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "g-telemetry": "profile",
            "g-log-dir": "telemetry",
            "g-stage-timings-json": "exact/stage-timings.json",
        }
    )

    telemetry_paths = telemetry.resolve_telemetry_paths(regenie_config)

    assert telemetry_paths.stage_timings_json == regenie_config.g_diagnostics.stage_timings_json


def test_telemetry_stream_uses_log_file_or_trace_file_alias() -> None:
    log_file_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "g-log-file": "logs/events.jsonl",
        }
    )
    trace_file_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "g-trace-file": "logs/trace-events.jsonl",
        }
    )

    assert telemetry.resolve_telemetry_paths(log_file_config).stream_file == log_file_config.g_diagnostics.log_file
    assert (
        telemetry.resolve_telemetry_paths(trace_file_config).stream_file == trace_file_config.g_diagnostics.trace_file
    )


def test_telemetry_stream_rejects_different_log_and_trace_files() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "g-log-file": "logs/events.jsonl",
            "g-trace-file": "logs/trace-events.jsonl",
        }
    )

    with pytest.raises(ValueError, match="g-log-file and g-trace-file both configure"):
        telemetry.resolve_telemetry_paths(regenie_config)


def test_telemetry_session_writes_schema_events_and_throttled_progress(tmp_path: Path) -> None:
    telemetry_paths = telemetry.TelemetryPaths(
        log_dir=tmp_path,
        stream_file=tmp_path / "events.jsonl",
        profile_summary_json=None,
        stage_timings_json=None,
    )
    telemetry_session = telemetry.TelemetrySession(
        mode=types.TelemetryMode.PROGRESS,
        paths=telemetry_paths,
        progress_interval_seconds=999.0,
        progress_interval_chunks=10,
        run_id="run-1",
    )

    telemetry_session.log_event("run_started", association_mode="regenie2_linear")
    telemetry_session.log_progress(processed_chunk_count=1, chromosome="22")
    telemetry_session.log_progress(processed_chunk_count=2, chromosome="22")
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    event_payload = event_payloads[0]
    progress_payloads = [event_payload for event_payload in event_payloads if event_payload["event"] == "progress_tick"]
    assert event_payload["schema_version"] == 1
    assert event_payload["run_id"] == "run-1"
    assert event_payload["event"] == "run_started"
    assert len(progress_payloads) == 1
    assert progress_payloads[0]["event"] == "progress_tick"
    assert progress_payloads[0]["processed_chunk_count"] == 1
    assert progress_payloads[0]["chromosome"] == "22"


def test_profile_telemetry_flushes_buffered_events_on_close(tmp_path: Path) -> None:
    telemetry_paths = telemetry.TelemetryPaths(
        log_dir=tmp_path,
        stream_file=tmp_path / "events.jsonl",
        profile_summary_json=None,
        stage_timings_json=None,
    )
    telemetry_session = telemetry.TelemetrySession(
        mode=types.TelemetryMode.PROFILE,
        paths=telemetry_paths,
        progress_interval_seconds=999.0,
        progress_interval_chunks=10,
        run_id="run-1",
    )

    for chunk_index in range(20):
        telemetry_session.log_event("chunk_profile", chunk_index=chunk_index)
    telemetry.close_telemetry_session(telemetry_session)
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads[:-1]] == ["chunk_profile"] * 20
    assert event_payloads[-1]["event"] == "telemetry_session_closed"
    assert event_payloads[0]["chunk_index"] == 0
    assert event_payloads[19]["chunk_index"] == 19


def test_trace_telemetry_event_cap_fails_without_lossy_mode(tmp_path: Path) -> None:
    telemetry_paths = telemetry.TelemetryPaths(
        log_dir=tmp_path,
        stream_file=tmp_path / "events.jsonl",
        profile_summary_json=None,
        stage_timings_json=None,
    )
    telemetry_session = telemetry.TelemetrySession(
        mode=types.TelemetryMode.TRACE,
        paths=telemetry_paths,
        progress_interval_seconds=999.0,
        progress_interval_chunks=10,
        lossy=False,
        trace_event_cap=2,
        run_id="run-1",
    )

    telemetry_session.log_event("first_trace_event")
    telemetry_session.log_event("second_trace_event")
    with pytest.raises(RuntimeError, match="Trace telemetry event cap exceeded at 2 events"):
        telemetry_session.log_event("third_trace_event")
    with pytest.raises(RuntimeError, match="Trace telemetry event cap exceeded at 2 events"):
        telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads] == [
        "first_trace_event",
        "second_trace_event",
    ]


def test_trace_telemetry_event_cap_drops_with_lossy_mode(tmp_path: Path) -> None:
    telemetry_paths = telemetry.TelemetryPaths(
        log_dir=tmp_path,
        stream_file=tmp_path / "events.jsonl",
        profile_summary_json=None,
        stage_timings_json=None,
    )
    telemetry_session = telemetry.TelemetrySession(
        mode=types.TelemetryMode.TRACE,
        paths=telemetry_paths,
        progress_interval_seconds=999.0,
        progress_interval_chunks=10,
        lossy=True,
        trace_event_cap=2,
        run_id="run-1",
    )

    for event_index in range(5):
        telemetry_session.log_event("trace_event", event_index=event_index)
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event_index"] for event_payload in event_payloads] == [0, 1]


@pytest.mark.parametrize("telemetry_mode", [types.TelemetryMode.PROGRESS, types.TelemetryMode.PROFILE])
def test_non_trace_telemetry_ignores_trace_event_cap(tmp_path: Path, telemetry_mode: types.TelemetryMode) -> None:
    telemetry_paths = telemetry.TelemetryPaths(
        log_dir=tmp_path,
        stream_file=tmp_path / "events.jsonl",
        profile_summary_json=None,
        stage_timings_json=None,
    )
    telemetry_session = telemetry.TelemetrySession(
        mode=telemetry_mode,
        paths=telemetry_paths,
        progress_interval_seconds=999.0,
        progress_interval_chunks=10,
        lossy=False,
        trace_event_cap=1,
        run_id="run-1",
    )

    for event_index in range(3):
        telemetry_session.log_event("non_trace_event", event_index=event_index)
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event_index"] for event_payload in event_payloads] == [0, 1, 2]


def test_log_file_replaces_default_telemetry_stream() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "g-log-dir": "telemetry",
            "g-log-file": "telemetry/rust.jsonl",
        }
    )

    telemetry_paths = telemetry.resolve_telemetry_paths(regenie_config)
    log_dir = typing.cast("Path", regenie_config.g_diagnostics.log_dir)

    assert telemetry_paths.stream_file == log_dir / "rust.jsonl"
    assert regenie_config.g_diagnostics.log_file == log_dir / "rust.jsonl"
