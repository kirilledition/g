from __future__ import annotations

import time
import typing

import jax.numpy as jnp
import numpy as np
import pytest

from g import types
from g.compute.regenie2_binary import diagnostics as regenie2_binary_diagnostics
from g.engine import timing
from g.engine.callbacks import runtime as callback_runtime
from g.engine.callbacks import transfers as callback_transfers
from g.engine.callbacks import writers as callback_writers
from g.engine.native_dispatch import delivery as native_dispatch_delivery
from g.engine.native_dispatch import models as native_dispatch_models


class FailingLifecycleCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
    """Callback runner that raises from dosage work to test worker error propagation."""

    def __init__(self) -> None:
        """Initialize a failing callback runner."""
        super().__init__(
            worker_name="failing-lifecycle-callback",
            staging_depth=1,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
            stage_timing_recorder=None,
            telemetry_session=None,
            output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
        )

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
        self.logged_events: list[tuple[str, str, dict[str, typing.Any]]] = []
        self.logged_progress: list[dict[str, typing.Any]] = []

    def log_event(self, event_name: str, level: str, **kwargs: typing.Any) -> None:
        """Record a telemetry event call."""
        self.logged_events.append((event_name, level, kwargs))

    def log_callback_progress_event(
        self,
        progress_event: callback_runtime._core.NativeCallbackProgressTelemetryEvent,
    ) -> None:
        """Record a native callback progress event call."""
        self.log_event(
            progress_event.event_name,
            progress_event.level,
            chromosome=progress_event.chromosome,
            processed_chunk_count=progress_event.processed_chunk_count,
        )

    def log_binary_correction_summary(self, summary_payload: dict[str, int]) -> None:
        """Record a native binary correction summary event call."""
        self.log_event("binary_correction_summary", "info", **summary_payload)

    def log_progress(self, **kwargs: typing.Any) -> None:
        """Record a progress callback call."""
        self.logged_progress.append(kwargs)


class ProgressTrackingCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
    """Lifecycle-only callback runner that tracks progress and telemetry hooks."""

    def __init__(
        self,
        telemetry_session: typing.Any,
    ) -> None:
        """Initialize the tracking callback runner."""
        super().__init__(
            worker_name="progress-tracking-callback",
            staging_depth=1,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
            stage_timing_recorder=None,
            telemetry_session=telemetry_session,
            output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
        )

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


class LifecycleCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
    """Minimal concrete callback runner for worker lifecycle tests."""

    def __init__(self) -> None:
        """Initialize a lifecycle-only callback runner."""
        super().__init__(
            worker_name="lifecycle-callback",
            staging_depth=1,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
            stage_timing_recorder=None,
            telemetry_session=None,
            output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
        )

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


class FailingPerfCounterClock:
    """Clock double that fails when the default path collects profiling timings."""

    @staticmethod
    def perf_counter() -> float:
        """Fail when a default-path code path attempts wall-time profiling."""
        message = "perf_counter should not be called without a timing recorder"
        raise AssertionError(message)

    @staticmethod
    def monotonic() -> float:
        """Delegate monotonic time for worker shutdown loops."""
        return time.monotonic()


class CapturingWriterSession:
    """Writer double that captures one native result chunk."""

    def __init__(self) -> None:
        """Initialize captured chunk storage."""
        self.written_chunks: list[dict[str, typing.Any]] = []

    def write_regenie2_native_chunk(
        self,
        *,
        metadata: object,
        chunk_stats: object,
        beta: object,
        standard_error: object,
        chi_squared: object,
        log10_p_value: object,
        extra_code: object,
    ) -> None:
        """Capture written arrays for assertion."""
        self.written_chunks.append(
            {
                "metadata": metadata,
                "chunk_stats": chunk_stats,
                "beta": beta,
                "standard_error": standard_error,
                "chi_squared": chi_squared,
                "log10_p_value": log10_p_value,
                "extra_code": extra_code,
            }
        )


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
        callback_batch_size: int = 1,
    ) -> int:
        """Assert callback startup happens before native delivery."""
        del sample_indices, committed_chunk_identifiers, callback_batch_size
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

    native_dispatch_delivery.run_bgen_engine_with_callback(
        engine=typing.cast("typing.Any", engine),
        run_input=typing.cast("native_dispatch_models.NativeBgenRunInput", RunInput()),
        committed_chunk_identifiers={2},
        writer_session=WriterSession(),
        callback=callback,
        stage_timing_recorder=None,
        variant_major_packed8_probability_pairs=False,
        stage_timing_snapshot_writer=timing.write_stage_timing_snapshot,
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


def test_native_callback_runner_default_queue_path_does_not_collect_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not collect queue or callback timings unless a recorder is configured."""
    monkeypatch.setattr(callback_runtime, "time", FailingPerfCounterClock)
    callback = LifecycleCallbackRunner()

    callback.compute_preprocessed_dosage_chunk(
        metadata=ChunkMetadata("chr1", 0, 1),
        genotype_matrix=np.asarray([[0.0]], dtype=np.float32),
        chunk_stats=typing.cast("typing.Any", np.asarray([0], dtype=np.float32)),
    )
    callback.finish()

    assert callback.processed_chunk_count == 1


def test_default_transfer_path_does_not_block_for_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not synchronize host-to-device transfer timings without a recorder."""

    def fail_block_until_ready(result_ready_value: object) -> None:
        del result_ready_value
        message = "block_until_ready should not be called without exact timings"
        raise AssertionError(message)

    monkeypatch.setattr(callback_transfers, "time", FailingPerfCounterClock)
    monkeypatch.setattr(callback_transfers, "block_until_ready", fail_block_until_ready)
    source_array = np.asarray([[1.0, 2.0]], dtype=np.float32)

    device_array = callback_transfers.put_chunk_array_on_device(
        source_array,
        stage_timing_recorder=None,
        chunk_metadata=ChunkMetadata("chr1", 0, 1),
        array_role="test",
    )

    np.testing.assert_array_equal(np.asarray(device_array), source_array)


def test_transfer_metadata_uses_native_shape_policy() -> None:
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)

    callback_transfers.record_transfer_metadata_for_array(
        stage_timing_recorder=stage_timing_recorder,
        transfer_name="host_to_device_transfer",
        array_role="genotype_matrix",
        array=np.zeros((2, 3), dtype=np.float32),
    )

    assert stage_timing_recorder.snapshot().transfer_metadata == (
        timing.TransferMetadataSnapshot(
            transfer_name="host_to_device_transfer",
            array_role="genotype_matrix",
            dtype_name="float32",
            ndim=2,
            observation_count=1,
            total_bytes=24,
            max_bytes=24,
            total_elements=6,
        ),
    )


def test_default_writer_path_preserves_values_without_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Materialize output values unchanged without default timing probes."""
    monkeypatch.setattr(callback_writers, "time", FailingPerfCounterClock)
    writer_session = CapturingWriterSession()

    callback_writers.write_regenie2_native_chunk_with_optional_timing(
        writer_session=writer_session,
        metadata=typing.cast("typing.Any", ChunkMetadata("chr1", 0, 2)),
        chunk_stats=typing.cast("typing.Any", object()),
        beta=jnp.asarray([1.25, -2.5], dtype=jnp.float32),
        standard_error=jnp.asarray([0.5, 1.5], dtype=jnp.float32),
        chi_squared=jnp.asarray([4.0, 9.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([2.0, 3.0], dtype=jnp.float32),
        extra_code=None,
        stage_timing_recorder=None,
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
    )

    written_chunk = writer_session.written_chunks[0]
    np.testing.assert_array_equal(written_chunk["beta"], np.asarray([1.25, -2.5], dtype=np.float32))
    np.testing.assert_array_equal(written_chunk["standard_error"], np.asarray([0.5, 1.5], dtype=np.float32))
    np.testing.assert_array_equal(written_chunk["chi_squared"], np.asarray([4.0, 9.0], dtype=np.float32))
    np.testing.assert_array_equal(written_chunk["log10_p_value"], np.asarray([2.0, 3.0], dtype=np.float32))
    assert written_chunk["extra_code"] is None


def test_native_callback_runner_records_progress_and_chromosome_events() -> None:
    """Record progression events for a full callback lifecycle."""
    telemetry_session = ProgressTrackingTelemetrySession()
    callback = ProgressTrackingCallbackRunner(telemetry_session=telemetry_session)

    def fail_finish_progress_state() -> typing.NoReturn:
        message = "native runtime resources should finish progress during worker lifecycle"
        raise AssertionError(message)

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
    typing.cast("typing.Any", callback).finish_progress_state = fail_finish_progress_state
    callback.finish()

    assert telemetry_session.logged_events == [
        ("chromosome_started", "info", {"chromosome": "chr1", "processed_chunk_count": 1}),
        ("chromosome_completed", "info", {"chromosome": "chr1", "processed_chunk_count": 1}),
        ("chromosome_started", "info", {"chromosome": "chr2", "processed_chunk_count": 2}),
        ("chromosome_completed", "info", {"chromosome": "chr2", "processed_chunk_count": 2}),
    ]
    assert len(telemetry_session.logged_progress) == 2
    assert telemetry_session.logged_progress[0] == {
        "processed_chunk_count": 1,
        "chromosome": "chr1",
        "chunk_identifier": 0,
        "variant_start_index": 0,
        "variant_stop_index": 1,
        "variant_count": 1,
    }
    assert telemetry_session.logged_progress[1] == {
        "processed_chunk_count": 2,
        "chromosome": "chr2",
        "chunk_identifier": 1,
        "variant_start_index": 1,
        "variant_stop_index": 2,
        "variant_count": 1,
    }


def test_native_callback_runner_emits_binary_correction_summary() -> None:
    telemetry_session = ProgressTrackingTelemetrySession()
    callback = ProgressTrackingCallbackRunner(telemetry_session=telemetry_session)
    diagnostics = regenie2_binary_diagnostics.BinaryChunkDiagnostics(
        score_only_count=jnp.asarray(3, dtype=jnp.int32),
        score_test_candidate_count=jnp.asarray(2, dtype=jnp.int32),
        firth_candidate_count=jnp.asarray(2, dtype=jnp.int32),
        firth_iteration_min=jnp.asarray(1, dtype=jnp.int32),
        firth_iteration_median=jnp.asarray(2, dtype=jnp.float32),
        firth_iteration_max=jnp.asarray(3, dtype=jnp.int32),
        firth_converged_count=jnp.asarray(1, dtype=jnp.int32),
        firth_failed_count=jnp.asarray(1, dtype=jnp.int32),
        firth_numerical_failure_count=jnp.asarray(0, dtype=jnp.int32),
        firth_max_iteration_failure_count=jnp.asarray(1, dtype=jnp.int32),
        firth_invalid_statistic_failure_count=jnp.asarray(0, dtype=jnp.int32),
        firth_step_halving_failure_count=jnp.asarray(0, dtype=jnp.int32),
        pseudo_firth_attempt_count=jnp.asarray(1, dtype=jnp.int32),
        pseudo_firth_success_count=jnp.asarray(1, dtype=jnp.int32),
        nr_zero_start_attempt_count=jnp.asarray(1, dtype=jnp.int32),
        nr_zero_start_success_count=jnp.asarray(0, dtype=jnp.int32),
        nr_warm_start_attempt_count=jnp.asarray(0, dtype=jnp.int32),
        nr_warm_start_success_count=jnp.asarray(0, dtype=jnp.int32),
        sparse_correction_count=jnp.asarray(1, dtype=jnp.int32),
        dense_correction_count=jnp.asarray(1, dtype=jnp.int32),
    )

    callback.record_binary_correction_diagnostics(diagnostics)
    callback.record_binary_null_model_failure_count(2)

    def fail_emit_binary_correction_summary() -> typing.NoReturn:
        message = "native runtime resources should return the pending diagnostics flush decision during worker finish"
        raise AssertionError(message)

    typing.cast("typing.Any", callback).emit_binary_correction_summary = fail_emit_binary_correction_summary
    callback.finish()

    assert telemetry_session.logged_events == [
        (
            "binary_correction_summary",
            "info",
            {
                "chunk_count": 1,
                "score_only_count": 3,
                "score_test_candidate_count": 2,
                "firth_attempted_count": 2,
                "firth_success_count": 1,
                "firth_failed_count": 1,
                "firth_numerical_failure_count": 0,
                "firth_max_iteration_failure_count": 1,
                "firth_invalid_statistic_failure_count": 0,
                "firth_step_halving_failure_count": 0,
                "pseudo_firth_attempt_count": 1,
                "pseudo_firth_success_count": 1,
                "nr_zero_start_attempt_count": 1,
                "nr_zero_start_success_count": 0,
                "nr_warm_start_attempt_count": 0,
                "nr_warm_start_success_count": 0,
                "sparse_correction_count": 1,
                "dense_correction_count": 1,
                "null_model_failure_count": 2,
            },
        )
    ]


def test_native_callback_runner_uses_finish_summary_payload_without_pending_diagnostics() -> None:
    """Emit native finish summary payload without Python summary re-planning."""
    telemetry_session = ProgressTrackingTelemetrySession()
    callback = ProgressTrackingCallbackRunner(telemetry_session=telemetry_session)

    def fail_emit_binary_correction_summary() -> typing.NoReturn:
        message = "native runtime resources should return complete summary payloads during worker finish"
        raise AssertionError(message)

    callback.record_binary_null_model_failure_count(2)
    typing.cast("typing.Any", callback).emit_binary_correction_summary = fail_emit_binary_correction_summary
    callback.finish()

    assert telemetry_session.logged_events == [
        (
            "binary_correction_summary",
            "info",
            {
                "chunk_count": 0,
                "score_only_count": 0,
                "score_test_candidate_count": 0,
                "firth_attempted_count": 0,
                "firth_success_count": 0,
                "firth_failed_count": 0,
                "firth_numerical_failure_count": 0,
                "firth_max_iteration_failure_count": 0,
                "firth_invalid_statistic_failure_count": 0,
                "firth_step_halving_failure_count": 0,
                "pseudo_firth_attempt_count": 0,
                "pseudo_firth_success_count": 0,
                "nr_zero_start_attempt_count": 0,
                "nr_zero_start_success_count": 0,
                "nr_warm_start_attempt_count": 0,
                "nr_warm_start_success_count": 0,
                "sparse_correction_count": 0,
                "dense_correction_count": 0,
                "null_model_failure_count": 2,
            },
        )
    ]


def test_binary_correction_summary_skips_materialization_without_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not device-get binary diagnostics when no telemetry session consumes them."""

    def fail_binary_chunk_diagnostics_to_summary_counts(binary_chunk_diagnostics: object) -> object:
        del binary_chunk_diagnostics
        message = "binary diagnostics should not materialize without telemetry"
        raise AssertionError(message)

    callback = LifecycleCallbackRunner()
    diagnostics = regenie2_binary_diagnostics.BinaryChunkDiagnostics(
        score_only_count=jnp.asarray(3, dtype=jnp.int32),
        score_test_candidate_count=jnp.asarray(2, dtype=jnp.int32),
        firth_candidate_count=jnp.asarray(2, dtype=jnp.int32),
        firth_iteration_min=jnp.asarray(1, dtype=jnp.int32),
        firth_iteration_median=jnp.asarray(2, dtype=jnp.float32),
        firth_iteration_max=jnp.asarray(3, dtype=jnp.int32),
        firth_converged_count=jnp.asarray(1, dtype=jnp.int32),
        firth_failed_count=jnp.asarray(1, dtype=jnp.int32),
        firth_numerical_failure_count=jnp.asarray(0, dtype=jnp.int32),
        firth_max_iteration_failure_count=jnp.asarray(1, dtype=jnp.int32),
        firth_invalid_statistic_failure_count=jnp.asarray(0, dtype=jnp.int32),
        firth_step_halving_failure_count=jnp.asarray(0, dtype=jnp.int32),
        pseudo_firth_attempt_count=jnp.asarray(1, dtype=jnp.int32),
        pseudo_firth_success_count=jnp.asarray(1, dtype=jnp.int32),
        nr_zero_start_attempt_count=jnp.asarray(1, dtype=jnp.int32),
        nr_zero_start_success_count=jnp.asarray(0, dtype=jnp.int32),
        nr_warm_start_attempt_count=jnp.asarray(0, dtype=jnp.int32),
        nr_warm_start_success_count=jnp.asarray(0, dtype=jnp.int32),
        sparse_correction_count=jnp.asarray(1, dtype=jnp.int32),
        dense_correction_count=jnp.asarray(1, dtype=jnp.int32),
    )
    monkeypatch.setattr(
        callback_runtime,
        "binary_chunk_diagnostics_to_summary_counts",
        fail_binary_chunk_diagnostics_to_summary_counts,
    )

    callback.record_binary_correction_diagnostics(diagnostics)

    assert callback.binary_correction_summary_chunk_count == 0
