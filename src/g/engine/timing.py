"""Timing diagnostics for REGENIE step 2 engine runs."""

from __future__ import annotations

import json
import threading
import time
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import pathlib


@dataclass(frozen=True)
class ChunkTimingIdentity:
    """Stable identity fields for one native genotype chunk.

    Attributes:
        chunk_identifier: Native chunk identifier.
        chromosome: Chromosome label for the chunk.
        variant_start_index: Inclusive variant start index.
        variant_stop_index: Exclusive variant stop index.
        variant_count: Number of variants in the chunk.

    """

    chunk_identifier: int
    chromosome: str
    variant_start_index: int
    variant_stop_index: int
    variant_count: int


@dataclass(frozen=True)
class ChunkStageTimingSnapshot:
    """Timing observation for one stage of one native genotype chunk.

    Attributes:
        chunk_identifier: Native chunk identifier.
        chromosome: Chromosome label for the chunk.
        variant_start_index: Inclusive variant start index.
        variant_stop_index: Exclusive variant stop index.
        variant_count: Number of variants in the chunk.
        stage_name: Timed stage name.
        duration_seconds: Wall-clock duration for this observation.

    """

    chunk_identifier: int
    chromosome: str
    variant_start_index: int
    variant_stop_index: int
    variant_count: int
    stage_name: str
    duration_seconds: float


@dataclass(frozen=True)
class StageTimingSnapshot:
    """Diagnostic stage timing snapshot for one native REGENIE step 2 run.

    Attributes:
        stage_totals_seconds: Total wall time per measured stage.
        stage_counts: Number of observations per measured stage.
        chunk_stage_timings: Timing observations keyed to individual chunks.
        native_bgen_profile: Native BGEN profile counters from the run engine.
        binary_chunk_diagnostics: Binary score/Firth diagnostics per processed chunk.
        null_logistic_diagnostics: Binary null logistic fit diagnostics per chromosome.
        queue_backpressure: Queue and bounded-resource backpressure summaries.
        transfer_metadata: Host/device transfer metadata summaries.

    """

    stage_totals_seconds: dict[str, float]
    stage_counts: dict[str, int]
    chunk_stage_timings: tuple[ChunkStageTimingSnapshot, ...]
    native_bgen_profile: dict[str, int]
    binary_chunk_diagnostics: tuple[dict[str, int | float], ...]
    null_logistic_diagnostics: tuple[dict[str, int | str], ...]
    queue_backpressure: tuple[QueueBackpressureSnapshot, ...]
    transfer_metadata: tuple[TransferMetadataSnapshot, ...]


@dataclass(frozen=True)
class QueueBackpressureSnapshot:
    """Aggregate queue or bounded-resource pressure metadata.

    Attributes:
        queue_name: Queue or resource being observed.
        operation_name: Operation that produced this observation.
        observation_count: Number of observations included in the aggregate.
        max_depth: Highest observed queue depth or resource occupancy.
        max_capacity: Configured queue or resource capacity.
        total_elapsed_seconds: Total elapsed time spent in the operation.
        total_blocked_seconds: Elapsed time known to be producer/consumer blocking.

    """

    queue_name: str
    operation_name: str
    observation_count: int
    max_depth: int
    max_capacity: int
    total_elapsed_seconds: float
    total_blocked_seconds: float


@dataclass(frozen=True)
class TransferMetadataSnapshot:
    """Aggregate metadata for one host/device transfer class.

    Attributes:
        transfer_name: Timed transfer stage name.
        array_role: Logical role for the transferred array.
        dtype_name: Data type name for the transferred array.
        ndim: Number of array dimensions.
        observation_count: Number of transfers included in the aggregate.
        total_bytes: Total estimated bytes transferred.
        max_bytes: Largest estimated single-transfer byte count.
        total_elements: Total element count across observations.

    """

    transfer_name: str
    array_role: str
    dtype_name: str
    ndim: int
    observation_count: int
    total_bytes: int
    max_bytes: int
    total_elements: int


@dataclass
class QueueBackpressureAccumulator:
    """Mutable queue/backpressure aggregate held behind the recorder lock."""

    observation_count: int = 0
    max_depth: int = 0
    max_capacity: int = 0
    total_elapsed_seconds: float = 0.0
    total_blocked_seconds: float = 0.0

    def add_observation(
        self,
        *,
        queue_depth: int,
        queue_capacity: int,
        elapsed_seconds: float,
        blocked_seconds: float,
    ) -> None:
        """Add one queue/backpressure observation."""
        self.observation_count += 1
        self.max_depth = max(self.max_depth, queue_depth)
        self.max_capacity = max(self.max_capacity, queue_capacity)
        self.total_elapsed_seconds += elapsed_seconds
        self.total_blocked_seconds += blocked_seconds


@dataclass
class TransferMetadataAccumulator:
    """Mutable transfer metadata aggregate held behind the recorder lock."""

    observation_count: int = 0
    total_bytes: int = 0
    max_bytes: int = 0
    total_elements: int = 0

    def add_observation(self, *, byte_count: int, element_count: int) -> None:
        """Add one transfer metadata observation."""
        self.observation_count += 1
        self.total_bytes += byte_count
        self.max_bytes = max(self.max_bytes, byte_count)
        self.total_elements += element_count


class StageTimingRecorder:
    """Thread-safe diagnostic wall-time collector for profiling harnesses."""

    def __init__(self, *, exact_stage_timings: bool = False) -> None:
        """Initialize empty stage timing state."""
        self.exact_stage_timings = exact_stage_timings
        self.stage_totals_seconds: dict[str, float] = {}
        self.stage_counts: dict[str, int] = {}
        self.chunk_stage_timings: list[ChunkStageTimingSnapshot] = []
        self.native_bgen_profile: dict[str, int] = {}
        self.binary_chunk_diagnostics: list[dict[str, int | float]] = []
        self.null_logistic_diagnostics: list[dict[str, int | str]] = []
        self.queue_backpressure: dict[tuple[str, str], QueueBackpressureAccumulator] = {}
        self.transfer_metadata: dict[tuple[str, str, str, int], TransferMetadataAccumulator] = {}
        self.lock = threading.Lock()

    def add_stage_duration_unlocked(self, stage_name: str, duration_seconds: float) -> None:
        """Accumulate one measured duration while the caller holds the lock."""
        self.stage_totals_seconds[stage_name] = self.stage_totals_seconds.get(stage_name, 0.0) + duration_seconds
        self.stage_counts[stage_name] = self.stage_counts.get(stage_name, 0) + 1

    def add_stage_duration(self, stage_name: str, duration_seconds: float) -> None:
        """Accumulate one measured duration."""
        with self.lock:
            self.add_stage_duration_unlocked(stage_name, duration_seconds)

    def add_chunk_stage_duration(
        self,
        *,
        chunk_identity: ChunkTimingIdentity,
        stage_name: str,
        duration_seconds: float,
    ) -> None:
        """Accumulate one measured duration and attach it to a native chunk."""
        with self.lock:
            self.add_stage_duration_unlocked(stage_name, duration_seconds)
            self.chunk_stage_timings.append(
                ChunkStageTimingSnapshot(
                    chunk_identifier=chunk_identity.chunk_identifier,
                    chromosome=chunk_identity.chromosome,
                    variant_start_index=chunk_identity.variant_start_index,
                    variant_stop_index=chunk_identity.variant_stop_index,
                    variant_count=chunk_identity.variant_count,
                    stage_name=stage_name,
                    duration_seconds=duration_seconds,
                )
            )

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

    def add_queue_backpressure_observation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        queue_depth: int,
        queue_capacity: int,
        elapsed_seconds: float = 0.0,
        blocked_seconds: float = 0.0,
    ) -> None:
        """Store one queue or bounded-resource pressure observation."""
        with self.lock:
            key = (queue_name, operation_name)
            accumulator = self.queue_backpressure.setdefault(key, QueueBackpressureAccumulator())
            accumulator.add_observation(
                queue_depth=queue_depth,
                queue_capacity=queue_capacity,
                elapsed_seconds=elapsed_seconds,
                blocked_seconds=blocked_seconds,
            )

    def add_transfer_metadata(
        self,
        *,
        transfer_name: str,
        array_role: str,
        dtype_name: str,
        ndim: int,
        byte_count: int,
        element_count: int,
    ) -> None:
        """Store metadata for one host/device transfer observation."""
        with self.lock:
            key = (transfer_name, array_role, dtype_name, ndim)
            accumulator = self.transfer_metadata.setdefault(key, TransferMetadataAccumulator())
            accumulator.add_observation(byte_count=byte_count, element_count=element_count)

    def snapshot(self) -> StageTimingSnapshot:
        """Return an immutable copy of the current timings."""
        with self.lock:
            return StageTimingSnapshot(
                stage_totals_seconds=dict(self.stage_totals_seconds),
                stage_counts=dict(self.stage_counts),
                chunk_stage_timings=tuple(self.chunk_stage_timings),
                native_bgen_profile=dict(self.native_bgen_profile),
                binary_chunk_diagnostics=tuple(dict(diagnostics) for diagnostics in self.binary_chunk_diagnostics),
                null_logistic_diagnostics=tuple(dict(diagnostics) for diagnostics in self.null_logistic_diagnostics),
                queue_backpressure=tuple(
                    QueueBackpressureSnapshot(
                        queue_name=queue_name,
                        operation_name=operation_name,
                        observation_count=accumulator.observation_count,
                        max_depth=accumulator.max_depth,
                        max_capacity=accumulator.max_capacity,
                        total_elapsed_seconds=accumulator.total_elapsed_seconds,
                        total_blocked_seconds=accumulator.total_blocked_seconds,
                    )
                    for (queue_name, operation_name), accumulator in sorted(self.queue_backpressure.items())
                ),
                transfer_metadata=tuple(
                    TransferMetadataSnapshot(
                        transfer_name=transfer_name,
                        array_role=array_role,
                        dtype_name=dtype_name,
                        ndim=ndim,
                        observation_count=accumulator.observation_count,
                        total_bytes=accumulator.total_bytes,
                        max_bytes=accumulator.max_bytes,
                        total_elements=accumulator.total_elements,
                    )
                    for (transfer_name, array_role, dtype_name, ndim), accumulator in sorted(
                        self.transfer_metadata.items()
                    )
                ),
            )


def build_stage_timing_recorder(
    stage_timing_path: pathlib.Path | None = None,
    *,
    force: bool = False,
) -> StageTimingRecorder | None:
    """Create a diagnostic stage recorder when requested."""
    if stage_timing_path is None and not force:
        return None
    return StageTimingRecorder(exact_stage_timings=stage_timing_path is not None)


def should_collect_exact_stage_timings(stage_timing_recorder: StageTimingRecorder | None) -> bool:
    """Return whether timing should force synchronized exact stage measurements."""
    return stage_timing_recorder is not None and stage_timing_recorder.exact_stage_timings


def write_stage_timing_snapshot(
    stage_timing_recorder: StageTimingRecorder | None,
    stage_timing_path: pathlib.Path | None = None,
) -> None:
    """Persist diagnostic stage timings when requested."""
    if stage_timing_recorder is None:
        return
    if stage_timing_path is None:
        return
    snapshot = stage_timing_recorder.snapshot()
    payload = {
        "stage_totals_seconds": snapshot.stage_totals_seconds,
        "stage_counts": snapshot.stage_counts,
        "chunk_stage_timings": serialize_chunk_stage_timings(snapshot.chunk_stage_timings),
        "native_bgen_profile": snapshot.native_bgen_profile,
        "binary_chunk_diagnostics": snapshot.binary_chunk_diagnostics,
        "null_logistic_diagnostics": snapshot.null_logistic_diagnostics,
        "queue_backpressure": serialize_queue_backpressure(snapshot.queue_backpressure),
        "transfer_metadata": serialize_transfer_metadata(snapshot.transfer_metadata),
        "derived_metrics": build_derived_metrics(snapshot),
    }
    stage_timing_path.parent.mkdir(parents=True, exist_ok=True)
    stage_timing_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def write_profile_summary(
    stage_timing_recorder: StageTimingRecorder | None,
    profile_summary_path: pathlib.Path | None = None,
    *,
    run_id: str | None = None,
) -> None:
    """Persist aggregate profile summary metrics when requested."""
    if stage_timing_recorder is None:
        return
    if profile_summary_path is None:
        return
    snapshot = stage_timing_recorder.snapshot()
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "stage_totals_seconds": snapshot.stage_totals_seconds,
        "stage_counts": snapshot.stage_counts,
        "native_bgen_profile": snapshot.native_bgen_profile,
        "derived_metrics": build_derived_metrics(snapshot),
        "chunk_stage_summary": build_chunk_stage_summary(snapshot.chunk_stage_timings),
        "binary_chunk_summary": build_binary_chunk_summary(snapshot.binary_chunk_diagnostics),
        "queue_backpressure": serialize_queue_backpressure(snapshot.queue_backpressure),
        "transfer_metadata": serialize_transfer_metadata(snapshot.transfer_metadata),
        "null_logistic_summary": {
            "chromosome_count": len(snapshot.null_logistic_diagnostics),
        },
    }
    profile_summary_path.parent.mkdir(parents=True, exist_ok=True)
    profile_summary_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def serialize_chunk_stage_timings(
    chunk_stage_timings: tuple[ChunkStageTimingSnapshot, ...],
) -> tuple[dict[str, int | float | str], ...]:
    """Serialize chunk timing observations to JSON-compatible dictionaries."""
    return tuple(
        {
            "chunk_identifier": chunk_stage_timing.chunk_identifier,
            "chromosome": chunk_stage_timing.chromosome,
            "variant_start_index": chunk_stage_timing.variant_start_index,
            "variant_stop_index": chunk_stage_timing.variant_stop_index,
            "variant_count": chunk_stage_timing.variant_count,
            "stage_name": chunk_stage_timing.stage_name,
            "duration_seconds": chunk_stage_timing.duration_seconds,
        }
        for chunk_stage_timing in chunk_stage_timings
    )


def serialize_queue_backpressure(
    queue_backpressure: tuple[QueueBackpressureSnapshot, ...],
) -> tuple[dict[str, int | float | str], ...]:
    """Serialize queue/backpressure observations to JSON-compatible dictionaries."""
    return tuple(
        {
            "queue_name": queue_snapshot.queue_name,
            "operation_name": queue_snapshot.operation_name,
            "observation_count": queue_snapshot.observation_count,
            "max_depth": queue_snapshot.max_depth,
            "max_capacity": queue_snapshot.max_capacity,
            "total_elapsed_seconds": queue_snapshot.total_elapsed_seconds,
            "total_blocked_seconds": queue_snapshot.total_blocked_seconds,
        }
        for queue_snapshot in queue_backpressure
    )


def serialize_transfer_metadata(
    transfer_metadata: tuple[TransferMetadataSnapshot, ...],
) -> tuple[dict[str, int | str], ...]:
    """Serialize transfer metadata observations to JSON-compatible dictionaries."""
    return tuple(
        {
            "transfer_name": transfer_snapshot.transfer_name,
            "array_role": transfer_snapshot.array_role,
            "dtype_name": transfer_snapshot.dtype_name,
            "ndim": transfer_snapshot.ndim,
            "observation_count": transfer_snapshot.observation_count,
            "total_bytes": transfer_snapshot.total_bytes,
            "max_bytes": transfer_snapshot.max_bytes,
            "total_elements": transfer_snapshot.total_elements,
        }
        for transfer_snapshot in transfer_metadata
    )


def build_chunk_stage_summary(
    chunk_stage_timings: tuple[ChunkStageTimingSnapshot, ...],
) -> dict[str, dict[str, float | int]]:
    """Summarize per-chunk timing observations by stage."""
    summary: dict[str, dict[str, float | int]] = {}
    for chunk_stage_timing in chunk_stage_timings:
        stage_summary = summary.setdefault(
            chunk_stage_timing.stage_name,
            {
                "total_seconds": 0.0,
                "count": 0,
            },
        )
        stage_summary["total_seconds"] = float(stage_summary["total_seconds"]) + chunk_stage_timing.duration_seconds
        stage_summary["count"] = int(stage_summary["count"]) + 1
    return summary


def build_binary_chunk_summary(binary_chunk_diagnostics: tuple[dict[str, int | float], ...]) -> dict[str, int | float]:
    """Build aggregate binary chunk diagnostic counters."""
    if not binary_chunk_diagnostics:
        return {"chunk_count": 0}
    summary: dict[str, int | float] = {"chunk_count": len(binary_chunk_diagnostics)}
    sum_keys = (
        "score_test_candidate_count",
        "firth_candidate_count",
        "firth_converged_count",
        "firth_failed_count",
        "firth_numerical_failure_count",
        "firth_max_iteration_failure_count",
        "firth_invalid_statistic_failure_count",
        "firth_step_halving_failure_count",
    )
    for key in sum_keys:
        summary[f"{key}_total"] = sum(float(diagnostics.get(key, 0.0)) for diagnostics in binary_chunk_diagnostics)
    summary["firth_iteration_min"] = min(
        float(diagnostics.get("firth_iteration_min", 0.0)) for diagnostics in binary_chunk_diagnostics
    )
    summary["firth_iteration_max"] = max(
        float(diagnostics.get("firth_iteration_max", 0.0)) for diagnostics in binary_chunk_diagnostics
    )
    return summary


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
    transfer_byte_totals: dict[str, int] = {}
    for transfer_snapshot in snapshot.transfer_metadata:
        transfer_byte_totals[transfer_snapshot.transfer_name] = (
            transfer_byte_totals.get(transfer_snapshot.transfer_name, 0) + transfer_snapshot.total_bytes
        )
    for transfer_name, byte_count in transfer_byte_totals.items():
        transfer_seconds = snapshot.stage_totals_seconds.get(transfer_name, 0.0)
        if byte_count > 0 and transfer_seconds > 0.0:
            derived_metrics[f"{transfer_name}_bytes_per_second"] = float(byte_count) / transfer_seconds
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


def record_chunk_stage_duration(
    stage_timing_recorder: StageTimingRecorder | None,
    *,
    chunk_identity: ChunkTimingIdentity,
    stage_name: str,
    start_time: float,
) -> None:
    """Record elapsed wall time for a chunk-specific stage."""
    if stage_timing_recorder is None:
        return
    stage_timing_recorder.add_chunk_stage_duration(
        chunk_identity=chunk_identity,
        stage_name=stage_name,
        duration_seconds=time.perf_counter() - start_time,
    )
