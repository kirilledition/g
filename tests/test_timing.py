from __future__ import annotations

import json
import typing

from g.engine import timing

if typing.TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_stage_timing_recorder_accumulates_and_snapshots_independent_state() -> None:
    recorder = timing.StageTimingRecorder()
    native_profile = {"variant_decode_count": 4}
    binary_diagnostics: dict[str, int | float] = {"firth_candidate_count": 2}
    null_diagnostics = {"chromosome": "22", "null_logistic_iteration_count": 5}

    recorder.add_stage_duration("native_engine_delivery", 1.25)
    recorder.add_stage_duration("native_engine_delivery", 0.75)
    recorder.set_native_bgen_profile(native_profile)
    recorder.add_binary_chunk_diagnostics(binary_diagnostics)
    recorder.add_null_logistic_diagnostics(null_diagnostics)
    snapshot = recorder.snapshot()

    native_profile["variant_decode_count"] = 99
    binary_diagnostics["firth_candidate_count"] = 99
    null_diagnostics["chromosome"] = "1"
    recorder.add_stage_duration("native_engine_delivery", 3.0)

    assert snapshot.stage_totals_seconds == {"native_engine_delivery": 2.0}
    assert snapshot.stage_counts == {"native_engine_delivery": 2}
    assert snapshot.native_bgen_profile == {"variant_decode_count": 4}
    assert snapshot.binary_chunk_diagnostics == ({"firth_candidate_count": 2},)
    assert snapshot.null_logistic_diagnostics == ({"chromosome": "22", "null_logistic_iteration_count": 5},)


def test_build_stage_timing_recorder_is_opt_in(tmp_path: Path) -> None:
    assert timing.build_stage_timing_recorder(None) is None
    assert isinstance(timing.build_stage_timing_recorder(tmp_path / "timings.json"), timing.StageTimingRecorder)


def test_write_stage_timing_snapshot_noops_without_recorder_or_path(tmp_path: Path) -> None:
    output_path = tmp_path / "missing" / "timings.json"
    recorder = timing.StageTimingRecorder()

    timing.write_stage_timing_snapshot(None, output_path)
    timing.write_stage_timing_snapshot(recorder, None)

    assert not output_path.exists()


def test_write_stage_timing_snapshot_persists_payload_and_derived_metrics(tmp_path: Path) -> None:
    output_path = tmp_path / "diagnostics" / "timings.json"
    recorder = timing.StageTimingRecorder()
    recorder.add_stage_duration("native_engine_delivery", 2.0)
    recorder.add_stage_duration("output_write", 4.0)
    recorder.add_stage_duration("jax_compute", 1.0)
    recorder.set_native_bgen_profile({"variant_decode_count": 8, "selected_sample_count": 10})
    recorder.add_binary_chunk_diagnostics({"score_test_candidate_count": 1})
    recorder.add_null_logistic_diagnostics({"chromosome": "22"})

    timing.write_stage_timing_snapshot(recorder, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["stage_totals_seconds"]["native_engine_delivery"] == 2.0
    assert payload["stage_counts"]["native_engine_delivery"] == 1
    assert payload["binary_chunk_diagnostics"] == [{"score_test_candidate_count": 1}]
    assert payload["null_logistic_diagnostics"] == [{"chromosome": "22"}]
    assert payload["derived_metrics"] == {
        "native_variant_decode_per_second": 4.0,
        "output_variant_rows_per_second": 2.0,
        "jax_variant_compute_per_second": 8.0,
        "native_dosage_values_per_second": 40.0,
    }


def test_build_derived_metrics_omits_zero_denominator_values() -> None:
    snapshot = timing.StageTimingSnapshot(
        stage_totals_seconds={"native_engine_delivery": 0.0, "output_write": 2.0},
        stage_counts={"native_engine_delivery": 1},
        native_bgen_profile={"variant_decode_count": 0, "selected_sample_count": 10},
        binary_chunk_diagnostics=(),
        null_logistic_diagnostics=(),
    )

    assert timing.build_derived_metrics(snapshot) == {}


def test_record_stage_duration_uses_elapsed_perf_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = timing.StageTimingRecorder()
    monkeypatch.setattr(timing.time, "perf_counter", lambda: 12.5)

    timing.record_stage_duration(recorder, "jax_compute", 10.0)
    timing.record_stage_duration(None, "ignored", 10.0)

    snapshot = recorder.snapshot()
    assert snapshot.stage_totals_seconds == {"jax_compute": 2.5}
    assert snapshot.stage_counts == {"jax_compute": 1}
