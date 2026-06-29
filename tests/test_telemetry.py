from __future__ import annotations

import json
import os
import typing
import unittest.mock

import pytest

from g import _core, types
from g.engine import run_events, telemetry
from g.interface import config
from g.jax_runtime import models as jax_runtime_models
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


def test_native_telemetry_run_session_owns_run_lifecycle_event_emission(tmp_path: Path) -> None:
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
    completed_event = run_events.RunCompletedEvent(
        run_id="run-1",
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        phenotype_count=1,
        artifacts=(
            run_events.RunArtifactPayload(
                phenotype_name="trait",
                output_run_directory=tmp_path / "run",
                final_dataset=None,
                final_parquet=None,
                final_regenie=tmp_path / "results.regenie",
                effective_config=tmp_path / "config.toml",
            ),
        ),
    )
    interrupted_event = run_events.RunInterruptedEvent(
        signal_number=2,
        signal_name="SIGINT",
        exit_code=130,
        flushed_for_resume=True,
    )
    failed_event = run_events.RunFailedEvent(error_type="RuntimeError", error_message="boom")

    telemetry_session.log_run_started(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        trait_type=types.RegenieTraitType.QUANTITATIVE,
        phenotype_count=1,
        output_run_root=tmp_path / "output.g",
    )
    telemetry_session.log_execution_plan_prepared(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        trait_type=types.RegenieTraitType.QUANTITATIVE,
        phenotype_count=1,
        chunk_size=128,
        variant_limit=None,
        device=types.Device.GPU,
    )
    telemetry_session.log_run_completed(completed_event)
    telemetry_session.log_run_interrupted(interrupted_event)
    telemetry_session.log_run_failed(failed_event)
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads] == [
        "run_started",
        "execution_plan_prepared",
        "run_completed",
        "run_failed",
        "run_failed",
    ]
    assert [event_payload["level"] for event_payload in event_payloads] == ["INFO", "INFO", "INFO", "WARN", "ERROR"]
    assert event_payloads[0]["output_run_root"] == str(tmp_path / "output.g")
    assert event_payloads[1]["chunk_size"] == 128
    assert event_payloads[1]["device"] == "gpu"
    assert "variant_limit" not in event_payloads[1]
    assert event_payloads[2]["association_mode"] == "regenie2_linear"
    assert event_payloads[2]["phenotype_count"] == 1
    assert event_payloads[3]["failure_kind"] == "graceful_shutdown"
    assert event_payloads[3]["signal_name"] == "SIGINT"
    assert event_payloads[4]["failure_kind"] == "exception"
    assert event_payloads[4]["error_message"] == "boom"


def test_native_telemetry_run_session_owns_writer_lifecycle_event_emission(tmp_path: Path) -> None:
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

    telemetry_session.log_effective_config_written(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        phenotype="height",
        effective_config=tmp_path / "height" / "effective_config.toml",
        output_run_directory=tmp_path / "height",
    )
    telemetry_session.log_writer_finished(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        phenotype="height",
        final_output_path=tmp_path / "height.parquet",
    )
    telemetry_session.log_multi_writer_finished(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        phenotype_count=2,
        final_output_paths=(tmp_path / "case_status.parquet", None),
    )
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads] == [
        "effective_config_written",
        "writer_finished",
        "writer_finished",
    ]
    assert [event_payload["level"] for event_payload in event_payloads] == ["INFO", "INFO", "INFO"]
    assert event_payloads[0]["association_mode"] == "regenie2_linear"
    assert event_payloads[0]["phenotype"] == "height"
    assert event_payloads[0]["effective_config"] == str(tmp_path / "height" / "effective_config.toml")
    assert event_payloads[0]["output_run_directory"] == str(tmp_path / "height")
    assert event_payloads[1]["phenotype"] == "height"
    assert event_payloads[1]["final_output_path"] == str(tmp_path / "height.parquet")
    assert event_payloads[2]["association_mode"] == "regenie2_binary"
    assert event_payloads[2]["phenotype_count"] == 2
    assert event_payloads[2]["final_output_paths"] == [str(tmp_path / "case_status.parquet"), None]


def test_native_telemetry_run_session_owns_preflight_event_emission(tmp_path: Path) -> None:
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

    telemetry_session.log_single_trait_preflight_completed(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        phenotype="height",
        sample_count=2504,
        covariate_count=3,
        chromosome_count=22,
    )
    telemetry_session.log_multi_phenotype_preflight_completed(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        phenotype_count=4,
        sample_count=2504,
    )
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads] == [
        "preflight_completed",
        "preflight_completed",
    ]
    assert [event_payload["level"] for event_payload in event_payloads] == ["INFO", "INFO"]
    assert event_payloads[0]["association_mode"] == "regenie2_linear"
    assert event_payloads[0]["phenotype"] == "height"
    assert event_payloads[0]["sample_count"] == 2504
    assert event_payloads[0]["covariate_count"] == 3
    assert event_payloads[0]["chromosome_count"] == 22
    assert event_payloads[1]["association_mode"] == "regenie2_binary"
    assert event_payloads[1]["phenotype_count"] == 4
    assert event_payloads[1]["sample_count"] == 2504


def test_native_telemetry_run_session_owns_pipeline_setup_event_emission(tmp_path: Path) -> None:
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

    telemetry_session.log_sample_alignment_completed(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        phenotype="height",
        phenotype_count=None,
        sample_count=2504,
        covariate_count=3,
        phenotype_group_count=None,
    )
    telemetry_session.log_sample_alignment_completed(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        phenotype=None,
        phenotype_count=4,
        sample_count=None,
        covariate_count=None,
        phenotype_group_count=2,
    )
    telemetry_session.log_prediction_source_loaded(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        phenotype="height",
        phenotype_count=None,
    )
    telemetry_session.log_prediction_source_loaded(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        phenotype=None,
        phenotype_count=4,
    )
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads] == [
        "sample_alignment_completed",
        "sample_alignment_completed",
        "prediction_source_loaded",
        "prediction_source_loaded",
    ]
    assert [event_payload["level"] for event_payload in event_payloads] == ["INFO", "INFO", "INFO", "INFO"]
    assert event_payloads[0]["phenotype"] == "height"
    assert event_payloads[0]["sample_count"] == 2504
    assert event_payloads[0]["covariate_count"] == 3
    assert "phenotype_count" not in event_payloads[0]
    assert event_payloads[1]["phenotype_count"] == 4
    assert event_payloads[1]["phenotype_group_count"] == 2
    assert "phenotype" not in event_payloads[1]
    assert "sample_count" not in event_payloads[1]
    assert event_payloads[2]["phenotype"] == "height"
    assert "phenotype_count" not in event_payloads[2]
    assert event_payloads[3]["phenotype_count"] == 4
    assert "phenotype" not in event_payloads[3]


def test_native_telemetry_run_session_owns_multi_phenotype_sample_summary(tmp_path: Path) -> None:
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

    telemetry_session.log_multi_phenotype_sample_summary(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        sample_counts=(3, 2),
        sample_set_fingerprints=("sample-a", "sample-b"),
        phenotype_group_count=2,
    )
    telemetry_session.log_multi_phenotype_sample_summary(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
        sample_counts=(2504, 2504),
        sample_set_fingerprints=("shared", "shared"),
        phenotype_group_count=1,
    )
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads] == [
        "multi_phenotype_sample_summary",
        "multi_phenotype_sample_summary",
    ]
    assert [event_payload["level"] for event_payload in event_payloads] == ["INFO", "INFO"]
    assert event_payloads[0]["association_mode"] == "regenie2_linear"
    assert event_payloads[0]["multi_phenotype_sample_mode"] == "per-phenotype"
    assert event_payloads[0]["phenotype_count"] == 2
    assert event_payloads[0]["phenotype_group_count"] == 2
    assert event_payloads[0]["sample_counts"] == [3, 2]
    assert event_payloads[0]["sample_counts_differ"] is True
    assert event_payloads[0]["shared_sample_set"] is False
    assert event_payloads[1]["association_mode"] == "regenie2_binary"
    assert event_payloads[1]["multi_phenotype_sample_mode"] == "complete-case"
    assert event_payloads[1]["sample_counts"] == [2504, 2504]
    assert event_payloads[1]["sample_counts_differ"] is False
    assert event_payloads[1]["shared_sample_set"] is True


def test_native_telemetry_run_session_owns_gpu_genotype_format_resolution(tmp_path: Path) -> None:
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

    telemetry_session.log_gpu_genotype_format_resolved(
        requested_gpu_genotype_format=types.GpuGenotypeFormat.AUTO,
        resolved_gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
        resolution_reason="trusted_validation_passed",
        fallback_error=None,
    )
    telemetry_session.log_gpu_genotype_format_resolved(
        requested_gpu_genotype_format=types.GpuGenotypeFormat.AUTO,
        resolved_gpu_genotype_format=types.GpuGenotypeFormat.DOSAGE,
        resolution_reason="trusted_validation_failed",
        fallback_error="packed8 incompatible",
    )
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads] == [
        "gpu_genotype_format_resolved",
        "gpu_genotype_format_resolved",
    ]
    assert [event_payload["level"] for event_payload in event_payloads] == ["INFO", "INFO"]
    assert event_payloads[0]["requested_gpu_genotype_format"] == "auto"
    assert event_payloads[0]["resolved_gpu_genotype_format"] == "packed8"
    assert event_payloads[0]["resolution_reason"] == "trusted_validation_passed"
    assert "fallback_error" not in event_payloads[0]
    assert event_payloads[1]["requested_gpu_genotype_format"] == "auto"
    assert event_payloads[1]["resolved_gpu_genotype_format"] == "dosage"
    assert event_payloads[1]["resolution_reason"] == "trusted_validation_failed"
    assert event_payloads[1]["fallback_error"] == "packed8 incompatible"


def test_native_telemetry_run_session_owns_engine_opening_events(tmp_path: Path) -> None:
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

    telemetry_session.log_association_backend_selected(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        association_backend_kind=types.AssociationBackendKind.JAX_PACKED8,
        device=types.Device.GPU,
        genotype_format=types.GpuGenotypeFormat.PACKED8,
        phenotype="height",
        phenotype_count=None,
    )
    telemetry_session.log_bgen_engine_opened(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        association_backend_kind=types.AssociationBackendKind.JAX_DOSAGE,
        sample_count=2504,
        variant_count=12345,
        phenotype=None,
        phenotype_count=3,
    )
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads] == [
        "association_backend_selected",
        "bgen_engine_opened",
    ]
    assert [event_payload["level"] for event_payload in event_payloads] == ["INFO", "INFO"]
    assert event_payloads[0]["association_mode"] == "regenie2_linear"
    assert event_payloads[0]["association_backend_kind"] == "jax_packed8"
    assert event_payloads[0]["device"] == "gpu"
    assert event_payloads[0]["genotype_format"] == "packed8"
    assert event_payloads[0]["phenotype"] == "height"
    assert "phenotype_count" not in event_payloads[0]
    assert event_payloads[1]["association_mode"] == "regenie2_binary"
    assert event_payloads[1]["association_backend_kind"] == "jax_dosage"
    assert event_payloads[1]["sample_count"] == 2504
    assert event_payloads[1]["variant_count"] == 12345
    assert event_payloads[1]["phenotype_count"] == 3
    assert "phenotype" not in event_payloads[1]


def test_native_telemetry_run_session_owns_callback_progress_events(tmp_path: Path) -> None:
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
    progress_state = _core.NativeCallbackProgressState()

    progress_update = progress_state.record_processed_chunk(_core.build_callback_chunk_identity("chr1", 0, 8))
    telemetry_session.log_callback_progress_event(progress_update.telemetry_plan.events[0])
    progress_completion = progress_state.finish_progress()
    assert progress_completion is not None
    telemetry_session.log_callback_progress_event(progress_completion.telemetry_event)
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payloads = [json.loads(line) for line in telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()]
    assert [event_payload["event"] for event_payload in event_payloads] == [
        "chromosome_started",
        "chromosome_completed",
    ]
    assert [event_payload["level"] for event_payload in event_payloads] == ["INFO", "INFO"]
    assert event_payloads[0]["chromosome"] == "chr1"
    assert event_payloads[0]["processed_chunk_count"] == 1
    assert event_payloads[1]["chromosome"] == "chr1"
    assert event_payloads[1]["processed_chunk_count"] == 1


def test_native_telemetry_run_session_owns_binary_correction_summary(tmp_path: Path) -> None:
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
    summary = _core.NativeBinaryCorrectionSummary()
    summary.add_diagnostics_totals(
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
    )
    summary.add_null_model_failure_count(20)

    telemetry_session.log_binary_correction_summary(summary.summary_payload())
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payload = json.loads(telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()[0])
    assert event_payload["event"] == "binary_correction_summary"
    assert event_payload["level"] == "INFO"
    assert event_payload["chunk_count"] == 2
    assert event_payload["score_only_count"] == 3
    assert event_payload["score_test_candidate_count"] == 4
    assert event_payload["firth_attempted_count"] == 5
    assert event_payload["firth_success_count"] == 6
    assert event_payload["firth_failed_count"] == 7
    assert event_payload["null_model_failure_count"] == 20


def test_native_telemetry_run_session_owns_jax_runtime_diagnostic_event(tmp_path: Path) -> None:
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
    diagnostic_event = jax_runtime_models.JaxRuntimeDiagnosticEvent(
        event_name="jax_native_diagnostic",
        level=jax_runtime_models.JaxRuntimeDiagnosticLevel.INFO,
        message="JAX diagnostic",
        fields=(
            jax_runtime_models.JaxRuntimeDiagnosticField(name="platform", value="cuda"),
            jax_runtime_models.JaxRuntimeDiagnosticField(name="persistent_cache_enabled", value=True),
            jax_runtime_models.JaxRuntimeDiagnosticField(name="cache_entries", value=7),
        ),
    )

    telemetry_session.log_jax_runtime_diagnostic_event(diagnostic_event, telemetry_level="trace")
    telemetry_session.close()

    assert telemetry_paths.stream_file is not None
    event_payload = json.loads(telemetry_paths.stream_file.read_text(encoding="utf-8").splitlines()[0])
    assert event_payload["event"] == "jax_native_diagnostic"
    assert event_payload["level"] == "TRACE"
    assert event_payload["platform"] == "cuda"
    assert event_payload["persistent_cache_enabled"] is True
    assert event_payload["cache_entries"] == 7


def test_close_telemetry_session_uses_close_with_event_contract() -> None:
    class FakeCloseableSession:
        def __init__(self) -> None:
            self.closed = False
            self.close_metadata: dict[str, object] | None = None

        def close_with_event(self) -> object:
            self.closed = True
            self.close_metadata = {"writer_counters": {"written_event_count": 3}}
            return self.close_metadata

    fake_session = FakeCloseableSession()

    telemetry.close_telemetry_session(fake_session)

    assert fake_session.closed is True
    assert fake_session.close_metadata == {"writer_counters": {"written_event_count": 3}}


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
