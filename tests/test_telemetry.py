from __future__ import annotations

import json
import typing

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
    assert telemetry_paths.event_file == expected_log_dir / "python.events.jsonl"
    assert telemetry_paths.progress_file == expected_log_dir / "progress.jsonl"
    assert telemetry_paths.trace_file is None
    assert telemetry_paths.profile_summary_json is None
    assert telemetry_paths.stage_timings_json is None


def test_trace_telemetry_paths_default_profile_and_trace_outputs() -> None:
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

    assert telemetry_paths.event_file == log_dir / "python.events.jsonl"
    assert telemetry_paths.trace_file == log_dir / "rust.events.jsonl"
    assert telemetry_paths.profile_summary_json == log_dir / "profile.summary.json"
    assert telemetry_paths.stage_timings_json == log_dir / "stage-timings.json"


def test_telemetry_session_writes_schema_events_and_throttled_progress(tmp_path: Path) -> None:
    telemetry_paths = telemetry.TelemetryPaths(
        log_dir=tmp_path,
        event_file=tmp_path / "python.events.jsonl",
        progress_file=tmp_path / "progress.jsonl",
        trace_file=None,
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

    assert telemetry_paths.event_file is not None
    assert telemetry_paths.progress_file is not None
    event_payload = json.loads(telemetry_paths.event_file.read_text(encoding="utf-8").splitlines()[0])
    progress_payloads = [
        json.loads(line) for line in telemetry_paths.progress_file.read_text(encoding="utf-8").splitlines()
    ]
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
        event_file=tmp_path / "events.jsonl",
        progress_file=tmp_path / "progress.jsonl",
        trace_file=None,
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

    assert telemetry_paths.event_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.event_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads[:-1]] == ["chunk_profile"] * 20
    assert event_payloads[-1]["event"] == "telemetry_session_closed"
    assert event_payloads[0]["chunk_index"] == 0
    assert event_payloads[19]["chunk_index"] == 19


def test_log_file_does_not_replace_python_telemetry_event_stream() -> None:
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

    assert telemetry_paths.event_file == log_dir / "python.events.jsonl"
    assert regenie_config.g_diagnostics.log_file == log_dir / "rust.jsonl"
