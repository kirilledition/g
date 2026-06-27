from __future__ import annotations

import json
import typing

from g.engine import timing

if typing.TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_stage_timing_recorder_accumulates_and_snapshots_independent_state() -> None:
    recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    native_profile = {"variant_decode_count": 4}
    binary_diagnostics: dict[str, int | float] = {"firth_candidate_count": 2}
    null_diagnostics = {"chromosome": "22", "iteration_count": 5}
    chunk_identity = timing.ChunkTimingIdentity(
        chunk_identifier=64,
        chromosome="22",
        variant_start_index=64,
        variant_stop_index=96,
        variant_count=32,
    )

    recorder.add_stage_duration("native_engine_delivery", 1.25)
    recorder.add_stage_duration("native_engine_delivery", 0.75)
    recorder.add_chunk_stage_duration(
        chunk_identity=chunk_identity,
        stage_name="python_callback",
        duration_seconds=0.5,
    )
    recorder.set_native_bgen_profile(native_profile)
    recorder.add_binary_chunk_diagnostics(binary_diagnostics)
    recorder.add_null_logistic_diagnostics(null_diagnostics)
    snapshot = recorder.snapshot()

    native_profile["variant_decode_count"] = 99
    binary_diagnostics["firth_candidate_count"] = 99
    null_diagnostics["chromosome"] = "1"
    recorder.add_stage_duration("native_engine_delivery", 3.0)

    assert snapshot.stage_totals_seconds == {"native_engine_delivery": 2.0, "python_callback": 0.5}
    assert snapshot.stage_counts == {"native_engine_delivery": 2, "python_callback": 1}
    assert snapshot.chunk_stage_timings == (
        timing.ChunkStageTimingSnapshot(
            chunk_identifier=64,
            chromosome="22",
            variant_start_index=64,
            variant_stop_index=96,
            variant_count=32,
            stage_name="python_callback",
            duration_seconds=0.5,
        ),
    )
    assert snapshot.native_bgen_profile == {"variant_decode_count": 4}
    assert timing.serialize_binary_chunk_diagnostics(snapshot.binary_chunk_diagnostics) == (
        {"firth_candidate_count": 2},
    )
    assert timing.serialize_null_logistic_diagnostics(snapshot.null_logistic_diagnostics) == (
        {"chromosome": "22", "iteration_count": 5},
    )
    assert snapshot.queue_backpressure == ()
    assert snapshot.transfer_metadata == ()


def test_stage_timing_recorder_aggregates_queue_backpressure() -> None:
    recorder = timing.StageTimingRecorder(exact_stage_timings=False)

    recorder.add_queue_backpressure_observation(
        queue_name="result_queue",
        operation_name="put",
        queue_depth=1,
        queue_capacity=2,
        elapsed_seconds=0.25,
        blocked_seconds=0.0,
    )
    recorder.add_queue_backpressure_observation(
        queue_name="result_queue",
        operation_name="put",
        queue_depth=2,
        queue_capacity=2,
        elapsed_seconds=0.5,
        blocked_seconds=0.5,
    )

    assert recorder.snapshot().queue_backpressure == (
        timing.QueueBackpressureSnapshot(
            queue_name="result_queue",
            operation_name="put",
            observation_count=2,
            max_depth=2,
            max_capacity=2,
            total_elapsed_seconds=0.75,
            total_blocked_seconds=0.5,
        ),
    )


def test_stage_timing_recorder_aggregates_transfer_metadata() -> None:
    recorder = timing.StageTimingRecorder(exact_stage_timings=False)

    recorder.add_stage_duration("host_to_device_transfer", 2.0)
    recorder.add_transfer_metadata(
        transfer_name="host_to_device_transfer",
        array_role="genotype_matrix",
        dtype_name="float32",
        ndim=2,
        byte_count=64,
        element_count=16,
    )
    recorder.add_transfer_metadata(
        transfer_name="host_to_device_transfer",
        array_role="genotype_matrix",
        dtype_name="float32",
        ndim=2,
        byte_count=32,
        element_count=8,
    )

    snapshot = recorder.snapshot()

    assert snapshot.transfer_metadata == (
        timing.TransferMetadataSnapshot(
            transfer_name="host_to_device_transfer",
            array_role="genotype_matrix",
            dtype_name="float32",
            ndim=2,
            observation_count=2,
            total_bytes=96,
            max_bytes=64,
            total_elements=24,
        ),
    )
    assert timing.build_derived_metrics(snapshot) == {"host_to_device_transfer_bytes_per_second": 48.0}


def test_build_stage_timing_recorder_is_opt_in(tmp_path: Path) -> None:
    assert timing.build_stage_timing_recorder(None, force=False) is None
    aggregate_recorder = timing.build_stage_timing_recorder(None, force=True)
    exact_recorder = timing.build_stage_timing_recorder(tmp_path / "timings.json", force=False)

    assert isinstance(aggregate_recorder, timing.StageTimingRecorder)
    assert isinstance(exact_recorder, timing.StageTimingRecorder)
    assert not aggregate_recorder.exact_stage_timings
    assert exact_recorder.exact_stage_timings
    assert not aggregate_recorder.should_collect_exact_stage_timings()
    assert exact_recorder.should_collect_exact_stage_timings()
    assert not timing.should_collect_exact_stage_timings(aggregate_recorder)
    assert timing.should_collect_exact_stage_timings(exact_recorder)


def test_write_stage_timing_snapshot_noops_without_recorder_or_path(tmp_path: Path) -> None:
    output_path = tmp_path / "missing" / "timings.json"
    recorder = timing.StageTimingRecorder(exact_stage_timings=False)

    timing.write_stage_timing_snapshot(None, output_path)
    timing.write_stage_timing_snapshot(recorder, None)

    assert not output_path.exists()


def test_write_stage_timing_snapshot_persists_payload_and_derived_metrics(tmp_path: Path) -> None:
    output_path = tmp_path / "diagnostics" / "timings.json"
    recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    recorder.add_stage_duration("native_engine_delivery", 2.0)
    recorder.add_stage_duration("output_write", 4.0)
    recorder.add_stage_duration("jax_compute", 1.0)
    recorder.add_chunk_stage_duration(
        chunk_identity=timing.ChunkTimingIdentity(
            chunk_identifier=0,
            chromosome="22",
            variant_start_index=0,
            variant_stop_index=8,
            variant_count=8,
        ),
        stage_name="python_callback",
        duration_seconds=0.25,
    )
    recorder.set_native_bgen_profile({"variant_decode_count": 8, "selected_sample_count": 10})
    recorder.add_binary_chunk_diagnostics({"score_test_candidate_count": 1})
    recorder.add_null_logistic_diagnostics({"chromosome": "22"})

    timing.write_stage_timing_snapshot(recorder, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["stage_totals_seconds"]["native_engine_delivery"] == 2.0
    assert payload["stage_counts"]["native_engine_delivery"] == 1
    assert payload["chunk_stage_timings"] == [
        {
            "chunk_identifier": 0,
            "chromosome": "22",
            "variant_start_index": 0,
            "variant_stop_index": 8,
            "variant_count": 8,
            "stage_name": "python_callback",
            "duration_seconds": 0.25,
        }
    ]
    assert payload["binary_chunk_diagnostics"] == [{"score_test_candidate_count": 1}]
    assert payload["null_logistic_diagnostics"] == [{"chromosome": "22"}]
    assert payload["queue_backpressure"] == []
    assert payload["transfer_metadata"] == []
    assert payload["derived_metrics"] == {
        "native_variant_decode_per_second": 4.0,
        "output_variant_rows_per_second": 2.0,
        "jax_variant_compute_per_second": 8.0,
        "native_dosage_values_per_second": 40.0,
    }


def test_write_profile_summary_persists_aggregate_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "logs" / "profile.summary.json"
    recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    recorder.add_stage_duration("native_engine_delivery", 2.0)
    recorder.set_native_bgen_profile({"variant_decode_count": 8})
    recorder.add_chunk_stage_duration(
        chunk_identity=timing.ChunkTimingIdentity(
            chunk_identifier=0,
            chromosome="22",
            variant_start_index=0,
            variant_stop_index=8,
            variant_count=8,
        ),
        stage_name="output_write",
        duration_seconds=0.5,
    )
    recorder.add_binary_chunk_diagnostics(
        {
            "score_test_candidate_count": 2,
            "firth_candidate_count": 1,
            "firth_iteration_min": 4,
            "firth_iteration_max": 8,
        }
    )

    timing.write_profile_summary(recorder, output_path, run_id="run-1")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["run_id"] == "run-1"
    assert payload["chunk_stage_summary"] == {"output_write": {"total_seconds": 0.5, "count": 1}}
    assert payload["queue_backpressure"] == []
    assert payload["transfer_metadata"] == []
    assert payload["binary_chunk_summary"]["chunk_count"] == 1
    assert payload["binary_chunk_summary"]["score_test_candidate_count_total"] == 2
    assert payload["binary_chunk_summary"]["firth_iteration_max"] == 8


def test_build_derived_metrics_omits_zero_denominator_values() -> None:
    snapshot = timing.StageTimingSnapshot(
        stage_totals_seconds={"native_engine_delivery": 0.0, "output_write": 2.0},
        stage_counts={"native_engine_delivery": 1},
        chunk_stage_timings=(),
        native_bgen_profile={"variant_decode_count": 0, "selected_sample_count": 10},
        binary_chunk_diagnostics=(),
        null_logistic_diagnostics=(),
        queue_backpressure=(),
        transfer_metadata=(),
    )

    assert timing.build_derived_metrics(snapshot) == {}


def test_record_stage_duration_uses_elapsed_perf_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    monkeypatch.setattr(timing.time, "perf_counter", lambda: 12.5)

    timing.record_stage_duration(recorder, "jax_compute", 10.0)
    timing.record_stage_duration(None, "ignored", 10.0)

    snapshot = recorder.snapshot()
    assert snapshot.stage_totals_seconds == {"jax_compute": 2.5}
    assert snapshot.stage_counts == {"jax_compute": 1}


def test_record_chunk_stage_duration_uses_elapsed_perf_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    chunk_identity = timing.ChunkTimingIdentity(
        chunk_identifier=5,
        chromosome="22",
        variant_start_index=5,
        variant_stop_index=7,
        variant_count=2,
    )
    monkeypatch.setattr(timing.time, "perf_counter", lambda: 12.5)

    timing.record_chunk_stage_duration(
        recorder,
        chunk_identity=chunk_identity,
        stage_name="output_write",
        start_time=10.0,
    )
    timing.record_chunk_stage_duration(
        None,
        chunk_identity=chunk_identity,
        stage_name="ignored",
        start_time=10.0,
    )

    snapshot = recorder.snapshot()
    assert snapshot.stage_totals_seconds == {"output_write": 2.5}
    assert snapshot.stage_counts == {"output_write": 1}
    assert snapshot.chunk_stage_timings == (
        timing.ChunkStageTimingSnapshot(
            chunk_identifier=5,
            chromosome="22",
            variant_start_index=5,
            variant_stop_index=7,
            variant_count=2,
            stage_name="output_write",
            duration_seconds=2.5,
        ),
    )
