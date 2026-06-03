"""Native BGEN callback helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
import typing
from dataclasses import dataclass

import jax
import numpy as np
import numpy.typing as npt

from g import _core, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import api as regenie2_linear
from g.engine import telemetry, timing

RESULT_WORKER_JOIN_TIMEOUT_SECONDS = 60.0
logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import collections.abc


class NativeBgenWorkerShutdownError(RuntimeError):
    """Raised when a native callback worker does not stop cleanly."""

    def __init__(self, *, worker_name: str, timeout_seconds: float) -> None:
        """Initialize a worker shutdown error."""
        self.worker_name = worker_name
        self.timeout_seconds = timeout_seconds
        message = f"native pipeline worker {worker_name!r} did not stop within {timeout_seconds:.1f} seconds"
        super().__init__(message)


@dataclass(frozen=True)
class PreprocessedDosageChunkWorkItem:
    """One native-preprocessed dosage chunk staged for asynchronous JAX compute."""

    metadata: typing.Any
    genotype_matrix: npt.NDArray[np.float32]
    chunk_stats: _core.ChunkStats


@dataclass(frozen=True)
class PreprocessedVariantMajorDosageChunkWorkItem:
    """One native-preprocessed variant-major dosage chunk staged for JAX compute."""

    metadata: typing.Any
    genotype_matrix_by_variant: npt.NDArray[np.float32]
    chunk_stats: _core.ChunkStats


@dataclass(frozen=True)
class Regenie2ResultWriteWorkItem:
    """One computed REGENIE result awaiting host materialization and output writing."""

    metadata: _core.VariantMetadata
    chunk_stats: _core.ChunkStats
    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array | None
    host_dosage_buffer: npt.NDArray[np.float32] | None
    release_in_flight_slot: bool


@dataclass(frozen=True)
class Regenie2MultiResultWriteWorkItem:
    """One computed multi-trait REGENIE result awaiting materialization and writing."""

    metadata: _core.VariantMetadata
    chunk_stats: _core.ChunkStats
    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array | None
    host_dosage_buffer: npt.NDArray[np.float32] | None
    release_in_flight_slot: bool


class NativeBgenRunInputProtocol(typing.Protocol):
    """Run input fields required by callback compute initialization."""

    phenotype_vector: jax.Array
    covariate_matrix: jax.Array


class NativeBgenMultiRunInputProtocol(typing.Protocol):
    """Run input fields required by multi-phenotype callbacks."""

    phenotype_names: tuple[str, ...]
    sample_indices: npt.NDArray[np.int64]
    phenotype_matrix: jax.Array
    covariate_matrix: jax.Array


class RegeniePredictionSourceProtocol(typing.Protocol):
    """Native prediction source interface used by the JAX callbacks."""

    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]:
        """Return already-aligned LOCO predictions for one chromosome."""
        ...


class MultiRegeniePredictionSourceProtocol(typing.Protocol):
    """Prediction source interface used by multi-phenotype callbacks."""

    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]:
        """Return trait-major aligned LOCO predictions for one chromosome."""
        ...


def block_until_ready(value: typing.Any) -> None:
    """Synchronize a JAX value when it supports readiness blocking."""
    block_until_ready_method = getattr(value, "block_until_ready", None)
    if callable(block_until_ready_method):
        block_until_ready_method()


def enforce_null_logistic_nonconvergence_policy(
    *,
    chromosome: str,
    null_logistic_converged: typing.Any,
    policy: types.NullLogisticNonconvergencePolicy,
    phenotype_names: tuple[str, ...] | None = None,
) -> None:
    """Raise or warn when a binary null-logistic chromosome fit did not converge."""
    convergence_flags = np.asarray(jax.device_get(null_logistic_converged), dtype=np.bool_)
    if convergence_flags.ndim == 0:
        if bool(convergence_flags):
            return
        message = f"Binary null logistic model did not converge for chromosome {chromosome}."
    else:
        failed_trait_indices = tuple(int(index) for index in np.flatnonzero(~convergence_flags))
        if not failed_trait_indices:
            return
        if phenotype_names is None:
            failed_traits = ", ".join(str(index) for index in failed_trait_indices)
        else:
            failed_traits = ", ".join(phenotype_names[index] for index in failed_trait_indices)
        message = f"Binary null logistic model did not converge for chromosome {chromosome}: {failed_traits}."
    if policy == types.NullLogisticNonconvergencePolicy.FAIL:
        raise RuntimeError(message)
    logger.warning("%s Continuing because --g-null-logistic-nonconvergence=warn.", message)


def record_binary_chunk_diagnostics(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    result: regenie2_binary.Regenie2BinaryScoreChunkResult | regenie2_binary.Regenie2BinaryChunkResult,
) -> None:
    """Record binary candidate and Firth diagnostics for one chunk."""
    if stage_timing_recorder is None:
        return
    diagnostics = jax.device_get(regenie2_binary.count_binary_chunk_diagnostics(result))
    stage_timing_recorder.add_binary_chunk_diagnostics(
        {
            "score_test_candidate_count": int(diagnostics.score_test_candidate_count),
            "firth_candidate_count": int(diagnostics.firth_candidate_count),
            "firth_iteration_min": int(diagnostics.firth_iteration_min),
            "firth_iteration_median": float(diagnostics.firth_iteration_median),
            "firth_iteration_max": int(diagnostics.firth_iteration_max),
            "firth_converged_count": int(diagnostics.firth_converged_count),
            "firth_failed_count": int(diagnostics.firth_failed_count),
            "firth_numerical_failure_count": int(diagnostics.firth_numerical_failure_count),
            "firth_max_iteration_failure_count": int(diagnostics.firth_max_iteration_failure_count),
            "firth_invalid_statistic_failure_count": int(diagnostics.firth_invalid_statistic_failure_count),
            "firth_step_halving_failure_count": int(diagnostics.firth_step_halving_failure_count),
        }
    )


def put_genotype_matrix_on_device(
    genotype_matrix: jax.Array | npt.NDArray[np.float32],
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> jax.Array:
    """Transfer a genotype chunk to the active JAX device with optional timing."""
    start_time = time.perf_counter()
    genotype_device_array = jax.device_put(genotype_matrix)
    if stage_timing_recorder is not None:
        block_until_ready(genotype_device_array)
    timing.record_stage_duration(stage_timing_recorder, "host_to_device_transfer", start_time)
    return genotype_device_array


def block_compute_result_for_timing(
    *,
    result_ready_value: jax.Array,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    start_time: float,
) -> None:
    """Synchronize chunk compute only when detailed stage timings are enabled."""
    if stage_timing_recorder is not None:
        block_until_ready(result_ready_value)
    timing.record_stage_duration(stage_timing_recorder, "jax_compute", start_time)


def cast_statistic_array_for_native_writer(array: object) -> npt.NDArray[np.float32]:
    """Cast computed statistics to the public native writer schema dtype."""
    return np.asarray(array, dtype=np.float32)


def write_regenie2_native_chunk_with_optional_timing(
    *,
    writer_session: typing.Any,
    metadata: _core.VariantMetadata,
    chunk_stats: _core.ChunkStats,
    beta: jax.Array,
    standard_error: jax.Array,
    chi_squared: jax.Array,
    log10_p_value: jax.Array,
    extra_code: jax.Array | None,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> None:
    """Write one native-metadata REGENIE chunk while timing JAX result materialization."""
    materialization_start_time = time.perf_counter()
    host_values = jax.device_get(
        {
            "beta": beta,
            "standard_error": standard_error,
            "chi_squared": chi_squared,
            "log10_p_value": log10_p_value,
            "extra_code": extra_code,
        }
    )
    timing.record_stage_duration(stage_timing_recorder, "device_to_host_materialization", materialization_start_time)

    write_start_time = time.perf_counter()
    writer_session.write_regenie2_native_chunk(
        metadata=metadata,
        chunk_stats=chunk_stats,
        beta=cast_statistic_array_for_native_writer(host_values["beta"]),
        standard_error=cast_statistic_array_for_native_writer(host_values["standard_error"]),
        chi_squared=cast_statistic_array_for_native_writer(host_values["chi_squared"]),
        log10_p_value=cast_statistic_array_for_native_writer(host_values["log10_p_value"]),
        extra_code=host_values["extra_code"],
    )
    timing.record_stage_duration(stage_timing_recorder, "output_write", write_start_time)
    timing.record_stage_duration(stage_timing_recorder, "single_trait_output_write", write_start_time)


def write_regenie2_multi_native_chunk_with_optional_timing(
    *,
    writer_sessions: tuple[typing.Any, ...],
    committed_chunk_identifier_sets: tuple[set[int], ...],
    metadata: _core.VariantMetadata,
    chunk_stats: _core.ChunkStats,
    beta: jax.Array,
    standard_error: jax.Array,
    chi_squared: jax.Array,
    log10_p_value: jax.Array,
    extra_code: jax.Array | None,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> None:
    """Materialize one multi-trait result once and write missing per-trait slices."""
    materialization_start_time = time.perf_counter()
    host_values = jax.device_get(
        {
            "beta": beta,
            "standard_error": standard_error,
            "chi_squared": chi_squared,
            "log10_p_value": log10_p_value,
            "extra_code": extra_code,
        }
    )
    timing.record_stage_duration(stage_timing_recorder, "device_to_host_materialization", materialization_start_time)

    chunk_identifier = int(metadata.variant_start_index)
    write_start_time = time.perf_counter()
    active_trait_indices = tuple(
        trait_index
        for trait_index, _writer_session in enumerate(writer_sessions)
        if chunk_identifier not in committed_chunk_identifier_sets[trait_index]
    )
    if all(isinstance(writer_session, _core.OutputWriterSession) for writer_session in writer_sessions):
        _core.write_regenie2_multi_native_chunk(
            writer_sessions=list(writer_sessions),
            active_trait_indices=list(active_trait_indices),
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=cast_statistic_array_for_native_writer(host_values["beta"]),
            standard_error=cast_statistic_array_for_native_writer(host_values["standard_error"]),
            chi_squared=cast_statistic_array_for_native_writer(host_values["chi_squared"]),
            log10_p_value=cast_statistic_array_for_native_writer(host_values["log10_p_value"]),
            extra_code=host_values["extra_code"],
        )
        timing.record_stage_duration(stage_timing_recorder, "output_write", write_start_time)
        timing.record_stage_duration(stage_timing_recorder, "multi_trait_output_write_total", write_start_time)
        return
    for trait_index, writer_session in enumerate(writer_sessions):
        if trait_index not in active_trait_indices:
            continue
        per_trait_write_start_time = time.perf_counter()
        extra_code_slice = None
        if host_values["extra_code"] is not None:
            extra_code_slice = host_values["extra_code"][trait_index]
        writer_session.write_regenie2_native_chunk(
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=cast_statistic_array_for_native_writer(host_values["beta"][trait_index]),
            standard_error=cast_statistic_array_for_native_writer(host_values["standard_error"][trait_index]),
            chi_squared=cast_statistic_array_for_native_writer(host_values["chi_squared"][trait_index]),
            log10_p_value=cast_statistic_array_for_native_writer(host_values["log10_p_value"][trait_index]),
            extra_code=extra_code_slice,
        )
        timing.record_stage_duration(
            stage_timing_recorder, "multi_trait_output_write_per_trait", per_trait_write_start_time
        )
    timing.record_stage_duration(stage_timing_recorder, "output_write", write_start_time)
    timing.record_stage_duration(stage_timing_recorder, "multi_trait_output_write_total", write_start_time)


def get_metadata_chromosome(metadata: typing.Any) -> str:
    """Return the first chromosome label from native or Python metadata."""
    return str(metadata.chromosome[0])


class NativeBgenCallbackRunner:
    """Reusable callback lifecycle for native BGEN chunk delivery."""

    def __init__(
        self,
        *,
        worker_name: str,
        staging_depth: int = 1,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize shared native callback state."""
        if staging_depth <= 0:
            message = "staging_depth must be positive."
            raise ValueError(message)
        self.processed_chunk_count = 0
        self.stage_timing_recorder = stage_timing_recorder
        self.telemetry_session = telemetry_session
        self.current_progress_chromosome: str | None = None
        self.dosage_queue_depth = staging_depth
        self.result_queue_depth = staging_depth
        self.result_in_flight_limit = self.result_queue_depth + 1
        self.dosage_buffer_limit = self.dosage_queue_depth + 1
        self.dosage_queue: queue.Queue[
            PreprocessedDosageChunkWorkItem | PreprocessedVariantMajorDosageChunkWorkItem | None
        ] = queue.Queue(maxsize=self.dosage_queue_depth)
        self.result_queue: queue.Queue[Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem | None] = (
            queue.Queue(maxsize=self.result_queue_depth)
        )
        self.result_in_flight_slots = threading.BoundedSemaphore(self.result_in_flight_limit)
        self.free_dosage_buffers: queue.Queue[npt.NDArray[np.float32]] = queue.Queue(maxsize=self.dosage_buffer_limit)
        self.dosage_buffer_count = 0
        self.worker_error: BaseException | None = None
        self.result_worker_error: BaseException | None = None
        self.worker_thread = threading.Thread(
            target=self.consume_dosage_chunks,
            name=worker_name,
            daemon=True,
        )
        self.result_worker_thread = threading.Thread(
            target=self.consume_result_write_items,
            name=f"{worker_name}-writer",
            daemon=True,
        )
        self.result_worker_thread.start()
        self.worker_thread.start()

    def record_stage_duration(self, stage_name: str, start_time: float) -> None:
        """Record a nested callback stage using this runner's timing recorder."""
        timing.record_stage_duration(self.stage_timing_recorder, stage_name, start_time)

    def get_stage_duration_recorder(self) -> collections.abc.Callable[[str, float], None] | None:
        """Return an optional nested stage recorder for lower-level compute helpers."""
        if self.stage_timing_recorder is None:
            return None
        return self.record_stage_duration

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed chunk and write it."""
        raise NotImplementedError

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed variant-major chunk and write it."""
        raise NotImplementedError

    def compute_preprocessed_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed dosage chunk for JAX association."""
        self.put_dosage_work_item(
            PreprocessedDosageChunkWorkItem(
                metadata=metadata,
                genotype_matrix=genotype_matrix,
                chunk_stats=chunk_stats,
            )
        )

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed variant-major dosage chunk for JAX association."""
        self.put_dosage_work_item(
            PreprocessedVariantMajorDosageChunkWorkItem(
                metadata=metadata,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                chunk_stats=chunk_stats,
            )
        )

    def consume_dosage_chunks(self) -> None:
        """Consume queued dosage chunks and run JAX work in order."""
        try:
            while True:
                work_item = self.dosage_queue.get()
                if work_item is None:
                    return
                if isinstance(work_item, PreprocessedVariantMajorDosageChunkWorkItem):
                    self.compute_preprocessed_variant_major_chunk(
                        variant_metadata=work_item.metadata,
                        genotype_matrix_by_variant=work_item.genotype_matrix_by_variant,
                        chunk_stats=work_item.chunk_stats,
                    )
                    self.processed_chunk_count += 1
                    self.record_progress(work_item.metadata)
                    continue
                if isinstance(work_item, PreprocessedDosageChunkWorkItem):
                    self.compute_preprocessed_chunk(
                        variant_metadata=work_item.metadata,
                        genotype_matrix=work_item.genotype_matrix,
                        chunk_stats=work_item.chunk_stats,
                    )
                    self.processed_chunk_count += 1
                    self.record_progress(work_item.metadata)
                    continue
        except Exception as error:  # noqa: BLE001
            self.worker_error = error

    def record_progress(self, metadata: typing.Any) -> None:
        """Record throttled progress after one chunk is processed."""
        if self.telemetry_session is None:
            return
        chromosome = get_metadata_chromosome(metadata)
        if chromosome != self.current_progress_chromosome:
            if self.current_progress_chromosome is not None:
                self.telemetry_session.log_event(
                    "chromosome_completed",
                    chromosome=self.current_progress_chromosome,
                    processed_chunk_count=self.processed_chunk_count - 1,
                )
            self.current_progress_chromosome = chromosome
            self.telemetry_session.log_event(
                "chromosome_started",
                chromosome=chromosome,
                processed_chunk_count=self.processed_chunk_count,
            )
        variant_start_index = int(metadata.variant_start_index)
        variant_stop_index = int(metadata.variant_stop_index)
        self.telemetry_session.log_progress(
            processed_chunk_count=self.processed_chunk_count,
            chromosome=chromosome,
            chunk_identifier=variant_start_index,
            variant_start_index=variant_start_index,
            variant_stop_index=variant_stop_index,
            variant_count=variant_stop_index - variant_start_index,
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed JAX results and write them in order."""
        try:
            while True:
                work_item = self.result_queue.get()
                if work_item is None:
                    return
                try:
                    write_regenie2_native_chunk_with_optional_timing(
                        writer_session=typing.cast("typing.Any", self).writer_session,
                        metadata=work_item.metadata,
                        chunk_stats=work_item.chunk_stats,
                        beta=work_item.beta,
                        standard_error=work_item.standard_error,
                        chi_squared=work_item.chi_squared,
                        log10_p_value=work_item.log10_p_value,
                        extra_code=work_item.extra_code,
                        stage_timing_recorder=self.stage_timing_recorder,
                    )
                finally:
                    self.release_result_work_item_buffer(work_item)
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def put_dosage_work_item(
        self,
        work_item: PreprocessedDosageChunkWorkItem | PreprocessedVariantMajorDosageChunkWorkItem | None,
    ) -> None:
        """Put work into the bounded worker queue while surfacing worker errors."""
        while True:
            self.raise_worker_error_if_present()
            put_start_time = time.perf_counter()
            try:
                self.dosage_queue.put(work_item, timeout=0.1)
                timing.record_stage_duration(self.stage_timing_recorder, "callback_queue_put", put_start_time)
                return
            except queue.Full:
                timing.record_stage_duration(
                    self.stage_timing_recorder, "callback_queue_producer_blocking", put_start_time
                )
                continue

    def raise_worker_error_if_present(self) -> None:
        """Raise an asynchronous worker failure on the producer thread."""
        if self.worker_error is not None:
            message = f"native pipeline callback worker failed: {self.worker_error}"
            raise RuntimeError(message) from self.worker_error
        if self.result_worker_error is not None:
            message = f"native pipeline result writer worker failed: {self.result_worker_error}"
            raise RuntimeError(message) from self.result_worker_error

    def put_result_write_item(
        self,
        work_item: Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem | None,
    ) -> None:
        """Put a computed result into the bounded materialization/write queue."""
        while True:
            self.raise_worker_error_if_present()
            put_start_time = time.perf_counter()
            try:
                self.result_queue.put(work_item, timeout=0.1)
                timing.record_stage_duration(self.stage_timing_recorder, "result_queue_put", put_start_time)
                return
            except queue.Full:
                timing.record_stage_duration(
                    self.stage_timing_recorder, "result_queue_producer_blocking", put_start_time
                )
                continue

    def acquire_result_in_flight_slot(self) -> None:
        """Reserve capacity for one chunk of pending GPU result work."""
        while True:
            self.raise_worker_error_if_present()
            acquire_start_time = time.perf_counter()
            if self.result_in_flight_slots.acquire(timeout=0.1):
                timing.record_stage_duration(
                    self.stage_timing_recorder, "result_in_flight_slot_acquire", acquire_start_time
                )
                return
            timing.record_stage_duration(
                self.stage_timing_recorder,
                "result_in_flight_producer_blocking",
                acquire_start_time,
            )

    def release_result_in_flight_slot(self) -> None:
        """Release capacity for one completed chunk of GPU result work."""
        self.result_in_flight_slots.release()

    def finish(self) -> None:
        """Wait until all queued JAX work has been written."""
        self.put_dosage_work_item(None)
        self.worker_thread.join()
        self.stop_result_worker()
        self.join_result_worker()
        self.raise_worker_error_if_present()
        if self.telemetry_session is not None and self.current_progress_chromosome is not None:
            self.telemetry_session.log_event(
                "chromosome_completed",
                chromosome=self.current_progress_chromosome,
                processed_chunk_count=self.processed_chunk_count,
            )
            self.current_progress_chromosome = None

    def abort(self) -> None:
        """Stop the worker after an upstream failure."""
        with contextlib.suppress(queue.Full):
            self.dosage_queue.put_nowait(None)
        with contextlib.suppress(queue.Full):
            self.result_queue.put_nowait(None)

    def stop_result_worker(self) -> None:
        """Signal the result worker to exit after queued results drain."""
        stop_deadline = time.monotonic() + RESULT_WORKER_JOIN_TIMEOUT_SECONDS
        while time.monotonic() < stop_deadline:
            if self.result_worker_error is not None:
                return
            if not self.result_worker_thread.is_alive():
                return
            timeout_seconds = max(0.0, min(0.1, stop_deadline - time.monotonic()))
            try:
                self.result_queue.put(None, timeout=timeout_seconds)
                return
            except queue.Full:
                continue
        raise NativeBgenWorkerShutdownError(
            worker_name=self.result_worker_thread.name,
            timeout_seconds=RESULT_WORKER_JOIN_TIMEOUT_SECONDS,
        )

    def join_result_worker(self) -> None:
        """Join the result writer worker with a bounded shutdown wait."""
        self.result_worker_thread.join(timeout=RESULT_WORKER_JOIN_TIMEOUT_SECONDS)
        if self.result_worker_thread.is_alive():
            raise NativeBgenWorkerShutdownError(
                worker_name=self.result_worker_thread.name,
                timeout_seconds=RESULT_WORKER_JOIN_TIMEOUT_SECONDS,
            )

    def acquire_dosage_buffer(self, sample_count: int, variant_count: int) -> npt.NDArray[np.float32]:
        """Return a reusable host dosage buffer for Rust to fill."""
        expected_shape = (sample_count, variant_count)
        return self.acquire_dosage_buffer_with_shape(expected_shape)

    def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> npt.NDArray[np.float32]:
        """Return a reusable host variant-major dosage buffer for Rust to fill."""
        expected_shape = (variant_count, sample_count)
        return self.acquire_dosage_buffer_with_shape(expected_shape)

    def acquire_dosage_buffer_with_shape(self, expected_shape: tuple[int, int]) -> npt.NDArray[np.float32]:
        """Return a reusable host dosage buffer with the requested shape."""
        while True:
            self.raise_worker_error_if_present()
            with contextlib.suppress(queue.Empty):
                dosage_buffer = self.free_dosage_buffers.get_nowait()
                if dosage_buffer.shape == expected_shape:
                    return dosage_buffer
                return np.empty(expected_shape, dtype=np.float32, order="C")
            if self.dosage_buffer_count < self.dosage_buffer_limit:
                self.dosage_buffer_count += 1
                return np.empty(expected_shape, dtype=np.float32, order="C")
            with contextlib.suppress(queue.Empty):
                dosage_buffer = self.free_dosage_buffers.get(timeout=0.1)
                if dosage_buffer.shape == expected_shape:
                    return dosage_buffer
                return np.empty(expected_shape, dtype=np.float32, order="C")

    def release_dosage_buffer(self, dosage_buffer: npt.NDArray[np.float32]) -> None:
        """Return a processed host dosage buffer to the reusable pool."""
        with contextlib.suppress(queue.Full):
            self.free_dosage_buffers.put_nowait(dosage_buffer)

    def release_numpy_dosage_buffer(self, dosage_buffer: jax.Array | npt.NDArray[np.float32]) -> None:
        """Return a NumPy host dosage buffer to the pool after device transfer."""
        if isinstance(dosage_buffer, np.ndarray):
            self.release_dosage_buffer(typing.cast("npt.NDArray[np.float32]", dosage_buffer))

    def get_releasable_dosage_buffer(
        self,
        dosage_buffer: jax.Array | npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32] | None:
        """Return a host dosage buffer reference when it belongs to the reusable pool."""
        if isinstance(dosage_buffer, np.ndarray):
            return typing.cast("npt.NDArray[np.float32]", dosage_buffer)
        return None

    def release_result_work_item_buffer(
        self,
        work_item: Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem,
    ) -> None:
        """Release resources after a dependent JAX result is materialized."""
        if work_item.host_dosage_buffer is not None:
            self.release_dosage_buffer(work_item.host_dosage_buffer)
        if work_item.release_in_flight_slot:
            self.release_result_in_flight_slot()


class LinearRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback used by the native BGEN pipeline for quantitative traits."""

    def __init__(
        self,
        run_input: NativeBgenRunInputProtocol,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        staging_depth: int = 1,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_session = writer_session
        self.regenie_state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=run_input.covariate_matrix,
            phenotype_vector=run_input.phenotype_vector,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_linear.Regenie2LinearChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-linear-callback",
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed chunk and enqueue its result for writing."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix)
        self.acquire_result_in_flight_slot()
        try:
            result = self.compute_linear_result(variant_metadata=variant_metadata, genotype_matrix=genotype_matrix)
            self.put_result_write_item(
                Regenie2ResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=None,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=True,
                )
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one variant-major chunk and enqueue its result for writing."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            result = self.compute_linear_variant_major_result(
                variant_metadata=variant_metadata,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                chunk_stats=chunk_stats,
            )
            self.put_result_write_item(
                Regenie2ResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=None,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=True,
                )
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def enqueue_linear_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: regenie2_linear.Regenie2LinearChunkResult,
        host_dosage_buffer: npt.NDArray[np.float32] | None = None,
        release_in_flight_slot: bool = False,
    ) -> None:
        """Enqueue a linear result for materialization and writing."""
        self.put_result_write_item(
            Regenie2ResultWriteWorkItem(
                metadata=variant_metadata,
                chunk_stats=chunk_stats,
                beta=result.beta,
                standard_error=result.standard_error,
                chi_squared=result.chi_squared,
                log10_p_value=result.log10_p_value,
                extra_code=None,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=release_in_flight_slot,
            )
        )

    def compute_linear_variant_major_result(
        self,
        *,
        variant_metadata: typing.Any,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> regenie2_linear.Regenie2LinearChunkResult:
        """Compute quantitative REGENIE step 2 statistics for a variant-major chunk."""
        self.prepare_chromosome_state(variant_metadata)
        assert self.current_chromosome_state is not None

        genotype_device_array = put_genotype_matrix_on_device(genotype_matrix_by_variant, self.stage_timing_recorder)
        genotype_dosage_sum = jax.device_put(chunk_stats.dosage_sum)
        genotype_observation_count = jax.device_put(chunk_stats.observation_count)
        genotype_imputed_dosage_square_sum = jax.device_put(chunk_stats.imputed_dosage_square_sum)
        compute_start_time = time.perf_counter()
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
            chromosome_state=self.current_chromosome_state,
            genotype_matrix_by_variant=genotype_device_array,
            genotype_dosage_sum=genotype_dosage_sum,
            genotype_observation_count=genotype_observation_count,
            genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
        )
        block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
        )
        return result

    def prepare_chromosome_state(self, variant_metadata: typing.Any) -> None:
        """Prepare cached linear chromosome state for the metadata chromosome."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_predictions = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(
            self.regenie_state,
            loco_predictions,
        )
        chromosome_ready_value = getattr(
            self.current_chromosome_state,
            "adjusted_residual",
            self.current_chromosome_state,
        )
        block_until_ready(chromosome_ready_value)
        timing.record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
        self.current_chromosome = chromosome

    def compute_linear_result(
        self,
        *,
        variant_metadata: typing.Any,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
    ) -> regenie2_linear.Regenie2LinearChunkResult:
        """Compute quantitative REGENIE step 2 statistics for one chunk."""
        self.prepare_chromosome_state(variant_metadata)
        assert self.current_chromosome_state is not None

        genotype_device_array = put_genotype_matrix_on_device(genotype_matrix, self.stage_timing_recorder)
        compute_start_time = time.perf_counter()
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=self.current_chromosome_state,
            genotype_matrix=genotype_device_array,
        )
        block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
        )
        return result


class MultiLinearRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback for quantitative multi-phenotype REGENIE step 2."""

    def __init__(
        self,
        run_input: NativeBgenMultiRunInputProtocol,
        prediction_source: MultiRegeniePredictionSourceProtocol,
        writer_sessions: tuple[typing.Any, ...],
        committed_chunk_identifier_sets: tuple[set[int], ...],
        staging_depth: int = 1,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_sessions = writer_sessions
        self.committed_chunk_identifier_sets = committed_chunk_identifier_sets
        self.regenie_state = regenie2_linear.prepare_regenie2_multi_linear_state(
            covariate_matrix=run_input.covariate_matrix,
            phenotype_matrix=run_input.phenotype_matrix,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_linear.Regenie2MultiLinearChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-multi-linear-callback",
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed multi-trait JAX results and write each trait in order."""
        try:
            while True:
                work_item = self.result_queue.get()
                if work_item is None:
                    return
                multi_work_item = typing.cast("Regenie2MultiResultWriteWorkItem", work_item)
                try:
                    write_regenie2_multi_native_chunk_with_optional_timing(
                        writer_sessions=self.writer_sessions,
                        committed_chunk_identifier_sets=self.committed_chunk_identifier_sets,
                        metadata=multi_work_item.metadata,
                        chunk_stats=multi_work_item.chunk_stats,
                        beta=multi_work_item.beta,
                        standard_error=multi_work_item.standard_error,
                        chi_squared=multi_work_item.chi_squared,
                        log10_p_value=multi_work_item.log10_p_value,
                        extra_code=multi_work_item.extra_code,
                        stage_timing_recorder=self.stage_timing_recorder,
                    )
                finally:
                    self.release_result_work_item_buffer(multi_work_item)
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one sample-major Rust-preprocessed chunk and enqueue multi-trait results."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            assert self.current_chromosome_state is not None
            genotype_device_array = put_genotype_matrix_on_device(genotype_matrix, self.stage_timing_recorder)
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state(
                chromosome_state=self.current_chromosome_state,
                genotype_matrix=genotype_device_array,
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one variant-major Rust-preprocessed chunk and enqueue multi-trait results."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            assert self.current_chromosome_state is not None
            genotype_device_array = put_genotype_matrix_on_device(
                genotype_matrix_by_variant,
                self.stage_timing_recorder,
            )
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major(
                chromosome_state=self.current_chromosome_state,
                genotype_matrix_by_variant=genotype_device_array,
                genotype_dosage_sum=jax.device_put(chunk_stats.dosage_sum),
                genotype_observation_count=jax.device_put(chunk_stats.observation_count),
                genotype_imputed_dosage_square_sum=jax.device_put(chunk_stats.imputed_dosage_square_sum),
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def prepare_chromosome_state(self, variant_metadata: typing.Any) -> None:
        """Prepare cached multi-linear chromosome state for the metadata chromosome."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_predictions = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_linear.prepare_regenie2_multi_linear_chromosome_state(
            self.regenie_state,
            loco_predictions,
        )
        block_until_ready(self.current_chromosome_state.adjusted_residual_matrix)
        timing.record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
        self.current_chromosome = chromosome

    def enqueue_multi_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: regenie2_linear.Regenie2MultiLinearChunkResult,
        host_dosage_buffer: npt.NDArray[np.float32] | None = None,
        release_in_flight_slot: bool = False,
    ) -> None:
        """Enqueue a multi-linear result for materialization and writing."""
        self.put_result_write_item(
            typing.cast(
                "Regenie2ResultWriteWorkItem",
                Regenie2MultiResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=None,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=release_in_flight_slot,
                ),
            )
        )


class BinaryRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback used by the native BGEN pipeline for binary traits."""

    def __init__(
        self,
        run_input: NativeBgenRunInputProtocol,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        correction_plan: types.BinaryCorrectionPlan,
        kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
        null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
            types.NullLogisticNonconvergencePolicy.FAIL
        ),
        staging_depth: int = 1,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_session = writer_session
        self.correction_plan = correction_plan
        self.kernel_config = kernel_config
        self.null_logistic_nonconvergence_policy = null_logistic_nonconvergence_policy
        self.regenie_state = regenie2_binary.prepare_regenie2_binary_state(
            covariate_matrix=run_input.covariate_matrix,
            phenotype_vector=run_input.phenotype_vector,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_binary.Regenie2BinaryChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-binary-callback",
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed chunk and enqueue its result for writing."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix)
        self.acquire_result_in_flight_slot()
        try:
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else jax.device_put(chunk_stats.is_rare_sparse_firth_candidate)
            )
            result = self.compute_binary_result(
                variant_metadata=variant_metadata,
                genotype_matrix=genotype_matrix,
                sparse_candidate_mask=sparse_candidate_mask,
            )
            self.enqueue_binary_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def enqueue_binary_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: (regenie2_binary.Regenie2BinaryScoreChunkResult | regenie2_binary.Regenie2BinaryChunkResult),
        host_dosage_buffer: npt.NDArray[np.float32] | None = None,
        release_in_flight_slot: bool = False,
    ) -> None:
        """Enqueue a binary result for materialization and writing."""
        self.put_result_write_item(
            Regenie2ResultWriteWorkItem(
                metadata=variant_metadata,
                chunk_stats=chunk_stats,
                beta=result.beta,
                standard_error=result.standard_error,
                chi_squared=result.chi_squared,
                log10_p_value=result.log10_p_value,
                extra_code=result.extra_code,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=release_in_flight_slot,
            )
        )

    def prepare_chromosome_state(self, variant_metadata: typing.Any) -> None:
        """Prepare cached binary chromosome state for the metadata chromosome."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_offset = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
            state=self.regenie_state,
            loco_offset=loco_offset,
            correction_plan=self.correction_plan,
            kernel_config=self.kernel_config,
        )
        chromosome_ready_value = getattr(
            self.current_chromosome_state,
            "score_residual",
            self.current_chromosome_state,
        )
        block_until_ready(chromosome_ready_value)
        enforce_null_logistic_nonconvergence_policy(
            chromosome=chromosome,
            null_logistic_converged=self.current_chromosome_state.null_logistic_converged,
            policy=self.null_logistic_nonconvergence_policy,
        )
        if self.stage_timing_recorder is not None:
            self.stage_timing_recorder.add_null_logistic_diagnostics(
                {
                    "chromosome": chromosome,
                    "iteration_count": int(jax.device_get(self.current_chromosome_state.null_logistic_iteration_count)),
                    "converged": int(jax.device_get(self.current_chromosome_state.null_logistic_converged)),
                    "firth_iteration_count": int(
                        jax.device_get(self.current_chromosome_state.null_firth_iteration_count)
                    ),
                    "firth_convergence_reason_code": int(
                        jax.device_get(self.current_chromosome_state.null_firth_convergence_reason_code)
                    ),
                    "correction_method": self.correction_plan.method.value,
                }
            )
        timing.record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
        self.current_chromosome = chromosome

    def compute_binary_result(
        self,
        *,
        variant_metadata: typing.Any,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        sparse_candidate_mask: jax.Array | None = None,
    ) -> regenie2_binary.Regenie2BinaryScoreChunkResult | regenie2_binary.Regenie2BinaryChunkResult:
        """Compute binary REGENIE step 2 statistics for one chunk."""
        self.prepare_chromosome_state(variant_metadata)
        assert self.current_chromosome_state is not None

        genotype_device_array = put_genotype_matrix_on_device(genotype_matrix, self.stage_timing_recorder)
        compute_start_time = time.perf_counter()
        result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=self.current_chromosome_state,
            genotype_matrix=genotype_device_array,
            correction_plan=self.correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=self.kernel_config,
            stage_duration_recorder=self.get_stage_duration_recorder(),
        )
        block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
        )
        record_binary_chunk_diagnostics(stage_timing_recorder=self.stage_timing_recorder, result=result)
        return result

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one variant-major chunk and enqueue its result for writing."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            assert self.current_chromosome_state is not None

            genotype_device_array = put_genotype_matrix_on_device(
                genotype_matrix_by_variant,
                self.stage_timing_recorder,
            )
            compute_start_time = time.perf_counter()
            if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                result = regenie2_binary.compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major(
                    chromosome_state=self.current_chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=self.correction_plan,
                    kernel_config=self.kernel_config,
                )
            else:
                result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
                    chromosome_state=self.current_chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=self.correction_plan,
                    sparse_candidate_mask=jax.device_put(chunk_stats.is_rare_sparse_firth_candidate),
                    kernel_config=self.kernel_config,
                    stage_duration_recorder=self.get_stage_duration_recorder(),
                )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
            )
            record_binary_chunk_diagnostics(stage_timing_recorder=self.stage_timing_recorder, result=result)
            self.enqueue_binary_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise


class MultiBinaryRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback for binary multi-phenotype REGENIE step 2."""

    def __init__(
        self,
        run_input: NativeBgenMultiRunInputProtocol,
        prediction_source: MultiRegeniePredictionSourceProtocol,
        writer_sessions: tuple[typing.Any, ...],
        committed_chunk_identifier_sets: tuple[set[int], ...],
        correction_plan: types.BinaryCorrectionPlan,
        kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
        null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
            types.NullLogisticNonconvergencePolicy.FAIL
        ),
        staging_depth: int = 1,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_sessions = writer_sessions
        self.committed_chunk_identifier_sets = committed_chunk_identifier_sets
        self.correction_plan = correction_plan
        self.kernel_config = kernel_config
        self.null_logistic_nonconvergence_policy = null_logistic_nonconvergence_policy
        self.regenie_state = regenie2_binary.prepare_regenie2_multi_binary_state(
            covariate_matrix=run_input.covariate_matrix,
            phenotype_matrix=run_input.phenotype_matrix,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_binary.Regenie2MultiBinaryChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-multi-binary-callback",
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed multi-trait JAX results and write each trait in order."""
        try:
            while True:
                work_item = self.result_queue.get()
                if work_item is None:
                    return
                multi_work_item = typing.cast("Regenie2MultiResultWriteWorkItem", work_item)
                try:
                    write_regenie2_multi_native_chunk_with_optional_timing(
                        writer_sessions=self.writer_sessions,
                        committed_chunk_identifier_sets=self.committed_chunk_identifier_sets,
                        metadata=multi_work_item.metadata,
                        chunk_stats=multi_work_item.chunk_stats,
                        beta=multi_work_item.beta,
                        standard_error=multi_work_item.standard_error,
                        chi_squared=multi_work_item.chi_squared,
                        log10_p_value=multi_work_item.log10_p_value,
                        extra_code=multi_work_item.extra_code,
                        stage_timing_recorder=self.stage_timing_recorder,
                    )
                finally:
                    self.release_result_work_item_buffer(multi_work_item)
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one sample-major Rust-preprocessed chunk and enqueue multi-trait results."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            assert self.current_chromosome_state is not None
            genotype_device_array = put_genotype_matrix_on_device(genotype_matrix, self.stage_timing_recorder)
            compute_start_time = time.perf_counter()
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else jax.device_put(chunk_stats.is_rare_sparse_firth_candidate)
            )
            result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state(
                chromosome_state=self.current_chromosome_state,
                genotype_matrix=genotype_device_array,
                correction_plan=self.correction_plan,
                sparse_candidate_mask=sparse_candidate_mask,
                kernel_config=self.kernel_config,
                stage_duration_recorder=self.get_stage_duration_recorder(),
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one variant-major Rust-preprocessed chunk and enqueue multi-trait results."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            assert self.current_chromosome_state is not None
            genotype_device_array = put_genotype_matrix_on_device(
                genotype_matrix_by_variant,
                self.stage_timing_recorder,
            )
            compute_start_time = time.perf_counter()
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else jax.device_put(chunk_stats.is_rare_sparse_firth_candidate)
            )
            result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
                chromosome_state=self.current_chromosome_state,
                genotype_matrix_by_variant=genotype_device_array,
                correction_plan=self.correction_plan,
                sparse_candidate_mask=sparse_candidate_mask,
                kernel_config=self.kernel_config,
                stage_duration_recorder=self.get_stage_duration_recorder(),
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def prepare_chromosome_state(self, variant_metadata: typing.Any) -> None:
        """Prepare cached multi-binary chromosome state for the metadata chromosome."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_offset = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_binary.prepare_regenie2_multi_binary_chromosome_state(
            self.regenie_state,
            loco_offset,
            self.correction_plan,
            self.kernel_config,
        )
        block_until_ready(self.current_chromosome_state.score_residual)
        enforce_null_logistic_nonconvergence_policy(
            chromosome=chromosome,
            null_logistic_converged=self.current_chromosome_state.null_logistic_converged,
            policy=self.null_logistic_nonconvergence_policy,
            phenotype_names=self.run_input.phenotype_names,
        )
        if self.stage_timing_recorder is not None:
            iteration_counts = jax.device_get(self.current_chromosome_state.null_logistic_iteration_count)
            convergence_flags = jax.device_get(self.current_chromosome_state.null_logistic_converged)
            for trait_index, phenotype_name in enumerate(self.run_input.phenotype_names):
                self.stage_timing_recorder.add_null_logistic_diagnostics(
                    {
                        "chromosome": chromosome,
                        "phenotype": phenotype_name,
                        "iteration_count": int(iteration_counts[trait_index]),
                        "converged": int(convergence_flags[trait_index]),
                        "correction_method": self.correction_plan.method.value,
                    }
                )
        timing.record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
        self.current_chromosome = chromosome

    def enqueue_multi_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: regenie2_binary.Regenie2MultiBinaryScoreChunkResult | regenie2_binary.Regenie2MultiBinaryChunkResult,
        host_dosage_buffer: npt.NDArray[np.float32] | None = None,
        release_in_flight_slot: bool = False,
    ) -> None:
        """Enqueue a multi-binary result for materialization and writing."""
        self.put_result_write_item(
            typing.cast(
                "Regenie2ResultWriteWorkItem",
                Regenie2MultiResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=result.extra_code,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=release_in_flight_slot,
                ),
            )
        )
