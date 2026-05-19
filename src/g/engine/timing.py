"""Timing diagnostics for REGENIE step 2 engine runs."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageTimingSnapshot:
    """Diagnostic stage timing snapshot for one native REGENIE step 2 run.

    Attributes:
        stage_totals_seconds: Total wall time per measured stage.
        stage_counts: Number of observations per measured stage.
        native_bgen_profile: Native BGEN profile counters from the run engine.
        binary_chunk_diagnostics: Binary score/Firth diagnostics per processed chunk.
        null_logistic_diagnostics: Binary null logistic fit diagnostics per chromosome.

    """

    stage_totals_seconds: dict[str, float]
    stage_counts: dict[str, int]
    native_bgen_profile: dict[str, int]
    binary_chunk_diagnostics: tuple[dict[str, int | float], ...]
    null_logistic_diagnostics: tuple[dict[str, int | str], ...]


class StageTimingRecorder:
    """Thread-safe diagnostic wall-time collector for profiling harnesses."""

    def __init__(self) -> None:
        """Initialize empty stage timing state."""
        self.stage_totals_seconds: dict[str, float] = {}
        self.stage_counts: dict[str, int] = {}
        self.native_bgen_profile: dict[str, int] = {}
        self.binary_chunk_diagnostics: list[dict[str, int | float]] = []
        self.null_logistic_diagnostics: list[dict[str, int | str]] = []
        self.lock = threading.Lock()

    def add_stage_duration(self, stage_name: str, duration_seconds: float) -> None:
        """Accumulate one measured duration."""
        with self.lock:
            self.stage_totals_seconds[stage_name] = self.stage_totals_seconds.get(stage_name, 0.0) + duration_seconds
            self.stage_counts[stage_name] = self.stage_counts.get(stage_name, 0) + 1

    def set_native_bgen_profile(self, profile_snapshot: dict[str, int]) -> None:
        """Store native BGEN profiling counters."""
        with self.lock:
            self.native_bgen_profile = dict(profile_snapshot)

    def add_binary_chunk_diagnostics(self, diagnostics: dict[str, int | float]) -> None:
        """Store diagnostic counters for one binary chunk."""
        with self.lock:
            self.binary_chunk_diagnostics.append(dict(diagnostics))

    def add_null_logistic_diagnostics(self, diagnostics: dict[str, int | str]) -> None:
        """Store null logistic fit diagnostics for one chromosome."""
        with self.lock:
            self.null_logistic_diagnostics.append(dict(diagnostics))

    def snapshot(self) -> StageTimingSnapshot:
        """Return an immutable copy of the current timings."""
        with self.lock:
            return StageTimingSnapshot(
                stage_totals_seconds=dict(self.stage_totals_seconds),
                stage_counts=dict(self.stage_counts),
                native_bgen_profile=dict(self.native_bgen_profile),
                binary_chunk_diagnostics=tuple(dict(diagnostics) for diagnostics in self.binary_chunk_diagnostics),
                null_logistic_diagnostics=tuple(dict(diagnostics) for diagnostics in self.null_logistic_diagnostics),
            )


def build_stage_timing_recorder_from_environment() -> StageTimingRecorder | None:
    """Create a diagnostic stage recorder when requested by the profiling harness."""
    if not os.environ.get("G_REGENIE2_STAGE_TIMINGS_JSON"):
        return None
    return StageTimingRecorder()


def write_stage_timing_snapshot_from_environment(stage_timing_recorder: StageTimingRecorder | None) -> None:
    """Persist diagnostic stage timings when the profiling harness requests them."""
    if stage_timing_recorder is None:
        return
    stage_timing_path = os.environ.get("G_REGENIE2_STAGE_TIMINGS_JSON")
    if not stage_timing_path:
        return
    snapshot = stage_timing_recorder.snapshot()
    payload = {
        "stage_totals_seconds": snapshot.stage_totals_seconds,
        "stage_counts": snapshot.stage_counts,
        "native_bgen_profile": snapshot.native_bgen_profile,
        "binary_chunk_diagnostics": snapshot.binary_chunk_diagnostics,
        "null_logistic_diagnostics": snapshot.null_logistic_diagnostics,
        "derived_metrics": build_derived_metrics(snapshot),
    }
    Path(stage_timing_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stage_timing_path).write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def build_derived_metrics(snapshot: StageTimingSnapshot) -> dict[str, float]:
    """Build throughput metrics from raw timing counters."""
    derived_metrics: dict[str, float] = {}
    variant_decode_count = float(snapshot.native_bgen_profile.get("variant_decode_count", 0))
    native_delivery_seconds = snapshot.stage_totals_seconds.get("native_engine_delivery", 0.0)
    if variant_decode_count > 0.0 and native_delivery_seconds > 0.0:
        derived_metrics["native_variant_decode_per_second"] = variant_decode_count / native_delivery_seconds
    output_write_seconds = snapshot.stage_totals_seconds.get("output_write", 0.0)
    if variant_decode_count > 0.0 and output_write_seconds > 0.0:
        derived_metrics["output_variant_rows_per_second"] = variant_decode_count / output_write_seconds
    jax_compute_seconds = snapshot.stage_totals_seconds.get("jax_compute", 0.0)
    if variant_decode_count > 0.0 and jax_compute_seconds > 0.0:
        derived_metrics["jax_variant_compute_per_second"] = variant_decode_count / jax_compute_seconds
    selected_sample_count = float(snapshot.native_bgen_profile.get("selected_sample_count", 0))
    if variant_decode_count > 0.0 and selected_sample_count > 0.0 and native_delivery_seconds > 0.0:
        derived_metrics["native_dosage_values_per_second"] = (
            variant_decode_count * selected_sample_count / native_delivery_seconds
        )
    return derived_metrics


def record_stage_duration(
    stage_timing_recorder: StageTimingRecorder | None,
    stage_name: str,
    start_time: float,
) -> None:
    """Record elapsed wall time for a stage when diagnostics are active."""
    if stage_timing_recorder is None:
        return
    stage_timing_recorder.add_stage_duration(stage_name, time.perf_counter() - start_time)
