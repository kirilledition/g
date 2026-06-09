from __future__ import annotations

import typing

import numpy as np
import pytest

from g.engine import callbacks, native_dispatch


class FailingLifecycleCallbackRunner(callbacks.NativeBgenCallbackRunner):
    """Callback runner that raises from dosage work to test worker error propagation."""

    def __init__(self) -> None:
        """Initialize a failing callback runner."""
        super().__init__(worker_name="failing-lifecycle-callback")

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: object,
        genotype_matrix: object,
        chunk_stats: object,
    ) -> None:
        """Always fail when sample-major dosage work is processed."""
        del variant_metadata, genotype_matrix, chunk_stats
        message = "forced dosageload failure"
        raise RuntimeError(message)

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: object,
        genotype_matrix_by_variant: object,
        chunk_stats: object,
    ) -> None:
        """Ignore variant-major dosage work in this callback runner."""
        del variant_metadata, genotype_matrix_by_variant, chunk_stats

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: object,
        packed_probability_pairs_by_variant: object,
        chunk_stats: object,
    ) -> None:
        """Ignore packed8 dosage work in this callback runner."""
        del variant_metadata, packed_probability_pairs_by_variant, chunk_stats


class ProgressTrackingTelemetrySession:
    """Telemetry session double that captures event and progress history."""

    def __init__(self) -> None:
        """Initialize telemetry capture state."""
        self.logged_events: list[tuple[str, dict[str, typing.Any]]] = []
        self.logged_progress: list[dict[str, typing.Any]] = []

    def log_event(self, event_name: str, **kwargs: typing.Any) -> None:
        """Record a telemetry event call."""
        self.logged_events.append((event_name, kwargs))

    def log_progress(self, **kwargs: typing.Any) -> None:
        """Record a progress callback call."""
        self.logged_progress.append(kwargs)


class ProgressTrackingCallbackRunner(callbacks.NativeBgenCallbackRunner):
    """Lifecycle-only callback runner that tracks progress and telemetry hooks."""

    def __init__(
        self,
        telemetry_session: typing.Any,
    ) -> None:
        """Initialize the tracking callback runner."""
        super().__init__(worker_name="progress-tracking-callback", telemetry_session=telemetry_session)

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: object,
        genotype_matrix: object,
        chunk_stats: object,
    ) -> None:
        """Ignore sample-major work and let telemetry handle chunk lifecycle."""
        del variant_metadata, genotype_matrix, chunk_stats

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: object,
        genotype_matrix_by_variant: object,
        chunk_stats: object,
    ) -> None:
        """Ignore variant-major work and keep callback lifecycle simple."""
        del variant_metadata, genotype_matrix_by_variant, chunk_stats

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: object,
        packed_probability_pairs_by_variant: object,
        chunk_stats: object,
    ) -> None:
        """Ignore packed8 work and keep callback lifecycle simple."""
        del variant_metadata, packed_probability_pairs_by_variant, chunk_stats


class ChunkMetadata:
    """Minimal chunk metadata for queue-driven callback tests."""

    def __init__(self, chromosome: str, variant_start: int, variant_stop: int) -> None:
        """Initialize small metadata tuple fields."""
        self.chromosome = (chromosome,)
        self.variant_start_index = variant_start
        self.variant_stop_index = variant_stop


class LifecycleCallbackRunner(callbacks.NativeBgenCallbackRunner):
    """Minimal concrete callback runner for worker lifecycle tests."""

    def __init__(self) -> None:
        """Initialize a lifecycle-only callback runner."""
        super().__init__(worker_name="lifecycle-callback")

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: object,
        genotype_matrix: object,
        chunk_stats: object,
    ) -> None:
        """Ignore sample-major chunks."""
        del variant_metadata, genotype_matrix, chunk_stats

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: object,
        genotype_matrix_by_variant: object,
        chunk_stats: object,
    ) -> None:
        """Ignore variant-major chunks."""
        del variant_metadata, genotype_matrix_by_variant, chunk_stats

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: object,
        packed_probability_pairs_by_variant: object,
        chunk_stats: object,
    ) -> None:
        """Ignore packed8 chunks."""
        del variant_metadata, packed_probability_pairs_by_variant, chunk_stats


class StartTrackingCallback:
    """Callback double that records explicit lifecycle ordering."""

    def __init__(self) -> None:
        """Initialize lifecycle state."""
        self.events: list[str] = []

    def start(self) -> None:
        """Record callback worker startup."""
        self.events.append("start")

    def finish(self) -> None:
        """Record callback drain."""
        self.events.append("finish")

    def abort(self) -> None:
        """Record callback abort."""
        self.events.append("abort")


class StartCheckingRunEngine:
    """Native engine double that checks callback startup before delivery."""

    def __init__(self) -> None:
        """Initialize run state."""
        self.reset_profile_count = 0

    def reset_profile(self) -> None:
        """Record native profile reset."""
        self.reset_profile_count += 1

    def profile_snapshot(self) -> dict[str, typing.Any]:
        """Return an empty profile snapshot."""
        return {}

    def run_bgen_variant_major_dosage_buffered_chunks(
        self,
        sample_indices: np.ndarray,
        callback: StartTrackingCallback,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        """Assert callback startup happens before native delivery."""
        del sample_indices, committed_chunk_identifiers
        callback.events.append("engine")
        return 0


class RunInput:
    """Minimal native dispatch run input."""

    sample_indices = np.asarray([0, 1], dtype=np.int64)


class WriterSession:
    """Minimal writer session double."""

    def finish(self) -> None:
        """Finish without final Parquet output."""


def test_native_callback_runner_start_is_explicit_and_idempotent() -> None:
    """Ensure callback construction no longer starts worker threads."""
    callback = LifecycleCallbackRunner()

    assert callback.worker_threads_started is False
    assert not callback.worker_thread.is_alive()
    assert not callback.result_worker_thread.is_alive()
    callback.finish()

    callback.start()
    callback.start()

    assert callback.worker_threads_started is True
    assert callback.worker_thread.is_alive()
    assert callback.result_worker_thread.is_alive()

    callback.finish()

    assert not callback.worker_thread.is_alive()
    assert not callback.result_worker_thread.is_alive()


def test_native_dispatch_starts_callback_before_engine_delivery() -> None:
    """Ensure native dispatch owns callback startup before Rust delivery."""
    engine = StartCheckingRunEngine()
    callback = StartTrackingCallback()

    native_dispatch.run_bgen_engine_with_callback(
        engine=typing.cast("typing.Any", engine),
        run_input=typing.cast("native_dispatch.NativeBgenRunInput", RunInput()),
        committed_chunk_identifiers={2},
        writer_session=WriterSession(),
        callback=callback,
        stage_timing_recorder=None,
    )

    assert callback.events == ["start", "engine", "finish"]


def test_native_callback_runner_finish_propagates_worker_error() -> None:
    """Surface worker thread failures as runtime errors during finish."""
    callback = FailingLifecycleCallbackRunner()
    callback.compute_preprocessed_dosage_chunk(
        metadata=ChunkMetadata("chr1", 0, 1),
        genotype_matrix=np.asarray([[0.0]], dtype=np.float32),
        chunk_stats=typing.cast("typing.Any", np.asarray([0], dtype=np.float32)),
    )

    with pytest.raises(RuntimeError, match="native pipeline callback worker failed"):
        callback.finish()


def test_native_callback_runner_records_progress_and_chromosome_events() -> None:
    """Record progression events for a full callback lifecycle."""
    telemetry_session = ProgressTrackingTelemetrySession()
    callback = ProgressTrackingCallbackRunner(telemetry_session=telemetry_session)
    callback.compute_preprocessed_dosage_chunk(
        metadata=ChunkMetadata("chr1", 0, 1),
        genotype_matrix=np.asarray([[0.0]], dtype=np.float32),
        chunk_stats=typing.cast("typing.Any", np.asarray([0], dtype=np.float32)),
    )
    callback.compute_preprocessed_dosage_chunk(
        metadata=ChunkMetadata("chr2", 1, 2),
        genotype_matrix=np.asarray([[0.0]], dtype=np.float32),
        chunk_stats=typing.cast("typing.Any", np.asarray([0], dtype=np.float32)),
    )
    callback.finish()

    assert telemetry_session.logged_events == [
        ("chromosome_started", {"chromosome": "chr1", "processed_chunk_count": 1}),
        ("chromosome_completed", {"chromosome": "chr1", "processed_chunk_count": 1}),
        ("chromosome_started", {"chromosome": "chr2", "processed_chunk_count": 2}),
        ("chromosome_completed", {"chromosome": "chr2", "processed_chunk_count": 2}),
    ]
    assert len(telemetry_session.logged_progress) == 2
    assert telemetry_session.logged_progress[0]["chromosome"] == "chr1"
    assert telemetry_session.logged_progress[1]["chromosome"] == "chr2"
