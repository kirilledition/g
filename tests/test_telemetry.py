from __future__ import annotations

import json
import os
import typing
import unittest.mock

import pytest

from g import _core, types
from g.engine import telemetry
from g.interface import config
from g.runner import runtime as runner_runtime

if typing.TYPE_CHECKING:
    from pathlib import Path


class NativeLoggingPolicyCore:
    """Fake-core mixin that keeps logging policy resolution native."""

    def build_logging_runtime_policy_payload(self, *arguments: object) -> dict[str, object]:
        """Delegate logging policy payload construction to the native helper."""
        native_build_logging_policy = typing.cast(
            "typing.Callable[..., dict[str, object]]",
            _core.build_logging_runtime_policy_payload,
        )
        return native_build_logging_policy(*arguments)


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
            "telemetry": "trace",
            "log_dir": "telemetry",
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
            "telemetry": "profile",
            "log_dir": "telemetry",
            "stage_timings_json": "exact/stage-timings.json",
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
            "log_file": "logs/events.jsonl",
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
            "trace_file": "logs/trace-events.jsonl",
        }
    )

    assert telemetry.resolve_telemetry_paths(log_file_config).stream_file == log_file_config.g_diagnostics.log_file
    assert (
        telemetry.resolve_telemetry_paths(trace_file_config).stream_file == trace_file_config.g_diagnostics.trace_file
    )


def test_initialize_logging_uses_log_filter_for_profile_unified_stream(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeCoreModule(NativeLoggingPolicyCore):
        def initialize_logging(self, **keyword_arguments: object) -> bool:
            calls.append(keyword_arguments)
            return True

    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "telemetry": "profile",
            "log_filter": "g=info",
            "trace_filter": "g.native.bgen=trace",
            "log_file": str(tmp_path / "events.jsonl"),
        }
    )
    telemetry_paths = telemetry.resolve_telemetry_paths(regenie_config)

    with (
        unittest.mock.patch(
            "g.runner.runtime.PROCESS_RUNTIME_STATE",
            runner_runtime.build_process_runtime_state(None, None, None),
        ),
        unittest.mock.patch("g.runner.runtime._core", FakeCoreModule()),
    ):
        runner_runtime.initialize_logging(regenie_config.g_diagnostics, telemetry_paths)

    assert calls[0]["trace_file"] == str(telemetry_paths.stream_file)
    assert calls[0]["trace_filter"] == "g=info"
    assert calls[0]["trace_event_cap"] is None


def test_initialize_logging_uses_trace_filter_for_trace_unified_stream(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeCoreModule(NativeLoggingPolicyCore):
        def initialize_logging(self, **keyword_arguments: object) -> bool:
            calls.append(keyword_arguments)
            return True

    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "telemetry": "trace",
            "log_filter": "g=debug",
            "trace_filter": "g.native.bgen=trace,g.output=debug",
            "trace_event_cap": 17,
            "log_file": str(tmp_path / "events.jsonl"),
        }
    )
    telemetry_paths = telemetry.resolve_telemetry_paths(regenie_config)

    with (
        unittest.mock.patch(
            "g.runner.runtime.PROCESS_RUNTIME_STATE",
            runner_runtime.build_process_runtime_state(None, None, None),
        ),
        unittest.mock.patch("g.runner.runtime._core", FakeCoreModule()),
    ):
        runner_runtime.initialize_logging(regenie_config.g_diagnostics, telemetry_paths)

    assert calls[0]["trace_file"] == str(telemetry_paths.stream_file)
    assert calls[0]["trace_filter"] == "g.native.bgen=trace,g.output=debug"
    assert calls[0]["trace_event_cap"] == 17


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
            "log_file": "logs/events.jsonl",
            "trace_file": "logs/trace-events.jsonl",
        }
    )

    with pytest.raises(ValueError, match="log_file and trace_file both configure"):
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
        queue_size=1024,
        lossy=True,
        trace_event_cap=0,
        run_id="run-1",
    )

    artifact_path = tmp_path / "artifact.txt"
    telemetry_session.log_event(
        "run_started",
        level="info",
        association_mode="regenie2_linear",
        artifact_path=artifact_path,
    )
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
    assert event_payload["artifact_path"] == str(artifact_path)
    assert event_payload["pid"] == os.getpid()
    assert event_payload["thread_name"] == "MainThread"
    assert len(progress_payloads) == 1
    assert progress_payloads[0]["event"] == "progress_tick"
    assert progress_payloads[0]["processed_chunk_count"] == 1
    assert progress_payloads[0]["chromosome"] == "22"


def test_telemetry_session_generates_native_run_id(tmp_path: Path) -> None:
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
        queue_size=1024,
        lossy=True,
        trace_event_cap=0,
        run_id=None,
    )

    assert len(telemetry_session.run_id) == 32
    assert telemetry_session.run_id == telemetry_session.run_id.lower()
    int(telemetry_session.run_id, 16)

    telemetry_session.log_event("run_started", level="info")
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payload = json.loads(telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()[0])
    assert event_payload["run_id"] == telemetry_session.run_id


def test_telemetry_session_builds_current_event_payload_natively(tmp_path: Path) -> None:
    telemetry_paths = telemetry.TelemetryPaths(
        log_dir=tmp_path,
        stream_file=None,
        profile_summary_json=None,
        stage_timings_json=None,
    )
    telemetry_session = telemetry.TelemetrySession(
        mode=types.TelemetryMode.OFF,
        paths=telemetry_paths,
        progress_interval_seconds=999.0,
        progress_interval_chunks=10,
        queue_size=1024,
        lossy=True,
        trace_event_cap=0,
        run_id="run-1",
    )

    event_payload = telemetry_session.build_event_payload(
        event="payload_built",
        level="debug",
        kept_field="value",
        omitted_field=None,
    )

    assert event_payload["schema_version"] == 1
    assert event_payload["run_id"] == "run-1"
    assert event_payload["event"] == "payload_built"
    assert event_payload["level"] == "DEBUG"
    assert event_payload["pid"] == os.getpid()
    assert event_payload["thread_name"] == "MainThread"
    assert event_payload["kept_field"] == "value"
    assert "omitted_field" not in event_payload


def test_telemetry_session_uses_native_policy_payload(tmp_path: Path) -> None:
    telemetry_paths = telemetry.TelemetryPaths(
        log_dir=tmp_path,
        stream_file=tmp_path / "events.jsonl",
        profile_summary_json=None,
        stage_timings_json=None,
    )

    off_session = telemetry.TelemetrySession(
        mode=types.TelemetryMode.OFF,
        paths=telemetry_paths,
        progress_interval_seconds=999.0,
        progress_interval_chunks=10,
        queue_size=1024,
        lossy=True,
        trace_event_cap=10,
        run_id="run-1",
    )
    profile_session = telemetry.TelemetrySession(
        mode=types.TelemetryMode.PROFILE,
        paths=telemetry_paths,
        progress_interval_seconds=999.0,
        progress_interval_chunks=10,
        queue_size=1024,
        lossy=True,
        trace_event_cap=10,
        run_id="run-2",
    )
    profile_session.close()
    trace_policy = _core.NativeTelemetrySessionPolicy("trace", 10)
    disabled_trace_cap_policy = _core.NativeTelemetrySessionPolicy("trace", 0)
    event_emission_plan = _core.plan_telemetry_event_emission(
        telemetry_enabled=True,
        has_native_telemetry_session=True,
    )
    disabled_event_emission_plan = _core.plan_telemetry_event_emission(
        telemetry_enabled=True,
        has_native_telemetry_session=False,
    )
    progress_emission_plan = _core.plan_telemetry_progress_emission(
        telemetry_enabled=True,
        has_native_telemetry_session=True,
        should_emit_progress=True,
    )

    assert dict(_core.resolve_telemetry_session_policy_payload("trace", 10)) == {
        "enabled": True,
        "profile_enabled": True,
        "event_cap": 10,
    }
    assert dict(_core.resolve_telemetry_session_policy_payload("trace", 0)) == {
        "enabled": True,
        "profile_enabled": True,
        "event_cap": None,
    }
    assert trace_policy.enabled
    assert trace_policy.profile_enabled
    assert trace_policy.event_cap == 10
    assert disabled_trace_cap_policy.enabled
    assert disabled_trace_cap_policy.profile_enabled
    assert disabled_trace_cap_policy.event_cap is None
    assert event_emission_plan.should_emit is True
    assert disabled_event_emission_plan.should_emit is False
    assert progress_emission_plan.should_emit is True
    assert progress_emission_plan.event_name == "progress_tick"
    assert progress_emission_plan.level == "info"
    native_close_plan = _core.plan_telemetry_close(
        has_telemetry_session=True,
        is_native_telemetry_session=True,
    )
    legacy_close_plan = _core.plan_telemetry_close(
        has_telemetry_session=True,
        is_native_telemetry_session=False,
    )
    disabled_close_plan = _core.plan_telemetry_close(
        has_telemetry_session=False,
        is_native_telemetry_session=False,
    )
    assert isinstance(native_close_plan, _core.NativeTelemetryClosePlan)
    assert native_close_plan.should_close is True
    assert native_close_plan.use_native_close_with_event is True
    assert native_close_plan.should_emit_legacy_close_event is False
    assert native_close_plan.legacy_close_event_name == "telemetry_session_closed"
    assert native_close_plan.legacy_close_event_level == "debug"
    assert legacy_close_plan.should_close is True
    assert legacy_close_plan.use_native_close_with_event is False
    assert legacy_close_plan.should_emit_legacy_close_event is True
    assert disabled_close_plan.should_close is False
    assert isinstance(off_session.native_session_handle, _core.NativeTelemetryRunSession)
    assert not off_session.enabled
    assert not off_session.profile_enabled
    assert off_session.native_session_policy.event_cap is None
    assert off_session.native_telemetry_session is None
    assert isinstance(profile_session.native_telemetry_session, _core.NativeTelemetryRunSession)
    assert profile_session.enabled
    assert profile_session.profile_enabled


def test_native_telemetry_run_session_owns_progress_emission(tmp_path: Path) -> None:
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
        queue_size=1024,
        lossy=True,
        trace_event_cap=0,
        run_id="run-1",
    )

    assert telemetry_session.native_session_handle.has_native_telemetry_session
    telemetry_session.log_progress(processed_chunk_count=1, chromosome="22")
    telemetry_session.log_progress(processed_chunk_count=2, chromosome="22")
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payload = json.loads(telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()[0])
    assert event_payload["event"] == "progress_tick"
    assert event_payload["level"] == "INFO"
    assert event_payload["processed_chunk_count"] == 1
    assert event_payload["chromosome"] == "22"


def test_close_telemetry_session_uses_native_close_plan_for_legacy_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCloseableSession:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []
            self.closed = False

        def log_event(self, event: str, level: str, **fields: object) -> None:
            self.events.append({"event": event, "level": level, "fields": fields})

        def writer_counters(self) -> object:
            return {"written_event_count": 3}

        def close(self) -> object:
            self.closed = True
            return None

    def plan_telemetry_close(
        *,
        has_telemetry_session: bool,
        is_native_telemetry_session: bool,
    ) -> object:
        assert has_telemetry_session is True
        assert is_native_telemetry_session is False
        return unittest.mock.Mock(
            should_close=True,
            use_native_close_with_event=False,
            should_emit_legacy_close_event=True,
            legacy_close_event_name="native_close",
            legacy_close_event_level="trace",
        )

    fake_session = FakeCloseableSession()
    monkeypatch.setattr(telemetry._core, "plan_telemetry_close", plan_telemetry_close)

    telemetry.close_telemetry_session(fake_session)

    assert fake_session.events == [
        {
            "event": "native_close",
            "level": "trace",
            "fields": {"writer_counters": {"written_event_count": 3}},
        }
    ]
    assert fake_session.closed is True


def test_telemetry_progress_throttle_emits_after_chunk_interval(tmp_path: Path) -> None:
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
        progress_interval_chunks=2,
        queue_size=1024,
        lossy=True,
        trace_event_cap=0,
        run_id="run-1",
    )

    telemetry_session.log_progress(processed_chunk_count=1, chromosome="22")
    telemetry_session.log_progress(processed_chunk_count=2, chromosome="22")
    telemetry_session.log_progress(processed_chunk_count=3, chromosome="22")
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    progress_payloads = [event_payload for event_payload in event_payloads if event_payload["event"] == "progress_tick"]
    assert [event_payload["processed_chunk_count"] for event_payload in progress_payloads] == [1, 3]


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
        queue_size=1024,
        lossy=True,
        trace_event_cap=0,
        run_id="run-1",
    )

    for chunk_index in range(20):
        telemetry_session.log_event("chunk_profile", level="info", chunk_index=chunk_index)
    assert telemetry_session.close_metadata is None
    telemetry.close_telemetry_session(telemetry_session)
    assert telemetry_session.close_metadata is not None
    assert telemetry_session.close_metadata["writer_counters"]["written_event_count"] == 21
    assert telemetry_session.close_metadata["writer_counters"]["dropped_event_count"] == 0
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads[:-1]] == ["chunk_profile"] * 20
    assert event_payloads[-1]["event"] == "telemetry_session_closed"
    assert event_payloads[-1]["writer_counters"]["written_event_count"] == 20
    assert event_payloads[-1]["writer_counters"]["dropped_event_count"] == 0
    assert event_payloads[0]["chunk_index"] == 0
    assert event_payloads[19]["chunk_index"] == 19


def test_telemetry_close_returns_writer_counters(tmp_path: Path) -> None:
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
        queue_size=1024,
        lossy=True,
        trace_event_cap=0,
        run_id="run-1",
    )

    telemetry_session.log_event("first_profile_event", level="info")
    telemetry_session.log_event("second_profile_event", level="info")
    assert telemetry_session.close_metadata is None
    close_metadata = telemetry_session.close()

    assert close_metadata is not None
    assert telemetry_session.close_metadata == close_metadata
    writer_counters = typing.cast("dict[str, object]", close_metadata["writer_counters"])
    assert writer_counters["accepted_event_count"] == 2
    assert writer_counters["written_event_count"] == 2
    assert writer_counters["dropped_event_count"] == 0
    assert writer_counters["cap_dropped_event_count"] == 0
    assert writer_counters["queue_dropped_event_count"] == 0
    assert writer_counters["event_cap_exceeded"] is False
    assert writer_counters["lossy"] is True
    assert writer_counters["event_cap"] is None
    assert isinstance(writer_counters["finish_flush_duration_seconds"], float)


def test_native_telemetry_finish_close_metadata_returns_writer_counters(tmp_path: Path) -> None:
    native_session = _core.NativeTelemetrySession(str(tmp_path / "events.jsonl"), queue_size=1024, lossy=True)

    native_session.emit_current_event("run-1", "first_profile_event", "info", {})
    close_metadata = native_session.finish_close_metadata()

    assert native_session.close_metadata() == close_metadata
    writer_counters = typing.cast("dict[str, object]", close_metadata["writer_counters"])
    assert writer_counters["accepted_event_count"] == 1
    assert writer_counters["written_event_count"] == 1
    assert writer_counters["dropped_event_count"] == 0


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
        queue_size=1024,
        lossy=False,
        trace_event_cap=2,
        run_id="run-1",
    )

    telemetry_session.log_event("first_trace_event", level="info")
    telemetry_session.log_event("second_trace_event", level="info")
    with pytest.raises(RuntimeError, match="Trace telemetry event cap exceeded at 2 events"):
        telemetry_session.log_event("third_trace_event", level="info")
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
        queue_size=1024,
        lossy=True,
        trace_event_cap=2,
        run_id="run-1",
    )

    for event_index in range(5):
        telemetry_session.log_event("trace_event", level="info", event_index=event_index)
    close_metadata = telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event_index"] for event_payload in event_payloads] == [0, 1]
    assert close_metadata is not None
    writer_counters = close_metadata["writer_counters"]
    assert writer_counters["accepted_event_count"] == 2
    assert writer_counters["written_event_count"] == 2
    assert writer_counters["dropped_event_count"] == 3
    assert writer_counters["cap_dropped_event_count"] == 3
    assert writer_counters["queue_dropped_event_count"] == 0
    assert writer_counters["event_cap_exceeded"] is True
    assert writer_counters["lossy"] is True
    assert writer_counters["event_cap"] == 2


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
        queue_size=1024,
        lossy=False,
        trace_event_cap=1,
        run_id="run-1",
    )

    for event_index in range(3):
        telemetry_session.log_event("non_trace_event", level="info", event_index=event_index)
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
            "log_dir": "telemetry",
            "log_file": "telemetry/rust.jsonl",
        }
    )

    telemetry_paths = telemetry.resolve_telemetry_paths(regenie_config)
    log_dir = typing.cast("Path", regenie_config.g_diagnostics.log_dir)

    assert telemetry_paths.stream_file == log_dir / "rust.jsonl"
    assert regenie_config.g_diagnostics.log_file == log_dir / "rust.jsonl"
