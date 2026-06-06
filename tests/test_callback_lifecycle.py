from __future__ import annotations

import typing

import numpy as np

from g.engine import callbacks, native_dispatch


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
        run_input=RunInput(),
        committed_chunk_identifiers={2},
        writer_session=WriterSession(),
        callback=callback,
        stage_timing_recorder=None,
    )

    assert callback.events == ["start", "engine", "finish"]
