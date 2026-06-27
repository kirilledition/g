"""Core callback lifecycle and bounded-queue runtime used by REGENIE callbacks."""

from __future__ import annotations

import abc
import contextlib
import queue
import threading
import time
import typing

import numpy as np
import numpy.typing as npt

import g.engine.callbacks.diagnostics as diagnostics
import g.engine.callbacks.shared as shared
import g.engine.callbacks.transfers as transfers
import g.engine.callbacks.writers as writers
from g import _core, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.engine import telemetry, timing

if typing.TYPE_CHECKING:
    import collections.abc

    import jax

DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS = shared.DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS
RESULT_WORKER_JOIN_TIMEOUT_SECONDS = shared.RESULT_WORKER_JOIN_TIMEOUT_SECONDS
GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS = shared.GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS
GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS = shared.GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS
WORKER_ABORT_STOP_TIMEOUT_SECONDS = shared.WORKER_ABORT_STOP_TIMEOUT_SECONDS
CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS = _core.resolve_callback_worker_backpressure_poll_timeout_seconds()
HostGenotypeBuffer = shared.HostGenotypeBuffer
PreprocessedDosageChunkWorkItem = shared.PreprocessedDosageChunkWorkItem
PreprocessedVariantMajorDosageChunkBatchWorkItem = shared.PreprocessedVariantMajorDosageChunkBatchWorkItem
PreprocessedVariantMajorDosageChunkWorkItem = shared.PreprocessedVariantMajorDosageChunkWorkItem
PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem = (
    shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
)
Regenie2ResultWriteWorkItem = shared.Regenie2ResultWriteWorkItem
Regenie2MultiResultWriteWorkItem = shared.Regenie2MultiResultWriteWorkItem
NativeBgenWorkerShutdownError = shared.NativeBgenWorkerShutdownError
record_stage_duration_with_optional_chunk = transfers.record_stage_duration_with_optional_chunk
build_native_callback_chunk_identity = transfers.build_native_callback_chunk_identity
write_regenie2_native_chunk_with_optional_timing = writers.write_regenie2_native_chunk_with_optional_timing
materialize_regenie2_native_chunk_with_optional_timing = writers.materialize_regenie2_native_chunk_with_optional_timing
write_materialized_regenie2_native_chunk_with_optional_timing = (
    writers.write_materialized_regenie2_native_chunk_with_optional_timing
)
record_binary_chunk_diagnostics_from_count = diagnostics.record_binary_chunk_diagnostics_from_count
binary_chunk_diagnostics_to_summary_counts = regenie2_binary.binary_chunk_diagnostics_to_summary_counts


def require_current_chromosome_state[ChromosomeStateType](
    chromosome_state: ChromosomeStateType | None,
    *,
    chromosome: str | None,
) -> ChromosomeStateType:
    """Return a prepared chromosome state or fail with an explicit runtime error."""
    if chromosome_state is not None:
        return chromosome_state
    if chromosome is None:
        message = "Chromosome state was not prepared before chunk computation."
    else:
        message = f"Chromosome state for {chromosome!r} was not prepared before chunk computation."
    raise RuntimeError(message)


class NativeBgenCallbackRunner(abc.ABC):
    """Reusable callback lifecycle for native BGEN chunk delivery."""

    def __init__(
        self,
        *,
        worker_name: str,
        staging_depth: int,
        native_callback_batch_size: int,
        result_in_flight_limit: int | None,
        dosage_buffer_limit: int | None,
        stage_timing_recorder: timing.StageTimingRecorder | None,
        telemetry_session: telemetry.TelemetrySession | None,
        output_statistic_dtype: types.FloatingPointDtype,
    ) -> None:
        """Initialize shared native callback state."""
        queue_limits = _core.resolve_native_callback_queue_limits(
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
        )
        self.progress_state = _core.NativeCallbackProgressState()
        self.stage_timing_recorder = stage_timing_recorder
        self.telemetry_session = telemetry_session
        self.output_statistic_dtype = output_statistic_dtype
        self.dosage_queue_depth = queue_limits.dosage_queue_depth
        self.result_queue_depth = queue_limits.result_queue_depth
        self.result_in_flight_limit = queue_limits.result_in_flight_limit
        self.dosage_buffer_limit = queue_limits.dosage_buffer_limit
        self.native_callback_batch_size = native_callback_batch_size
        self.result_in_flight_slot_lock = threading.Lock()
        self.result_in_flight_slot_state = _core.NativeResultInFlightSlotState(self.result_in_flight_limit)
        self.dosage_queue: queue.Queue[
            PreprocessedDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkBatchWorkItem
            | PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
            | None
        ] = queue.Queue(maxsize=self.dosage_queue_depth)
        self.result_queue: queue.Queue[Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem | None] = (
            queue.Queue(maxsize=self.result_queue_depth)
        )
        self.result_in_flight_slots = threading.BoundedSemaphore(self.result_in_flight_limit)
        self.free_dosage_buffers: queue.Queue[HostGenotypeBuffer] = queue.Queue(maxsize=self.dosage_buffer_limit)
        self.dosage_buffer_pool = _core.NativeDosageBufferPoolState(self.dosage_buffer_limit)
        self.worker_error: BaseException | None = None
        self.result_worker_error: BaseException | None = None
        self.binary_correction_summary = _core.NativeBinaryCorrectionSummary()
        self.binary_correction_pending_diagnostics: list[regenie2_binary.BinaryChunkDiagnostics] = []
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
        self.worker_start_lock = threading.Lock()
        self.worker_lifecycle_state = _core.NativeCallbackWorkerLifecycleState()

    @property
    def processed_chunk_count(self) -> int:
        """Return the native processed chunk count."""
        return self.progress_state.processed_chunk_count

    @property
    def current_progress_chromosome(self) -> str | None:
        """Return the native active progress chromosome."""
        return self.progress_state.current_progress_chromosome

    @property
    def binary_correction_summary_chunk_count(self) -> int:
        """Return the number of chunks included in binary correction summary telemetry."""
        return self.binary_correction_summary.chunk_count + len(self.binary_correction_pending_diagnostics)

    def start(self) -> None:
        """Start asynchronous callback workers after owner setup is complete."""
        with self.worker_start_lock:
            if self.worker_threads_started:
                return
            self.result_worker_thread.start()
            self.worker_thread.start()
            self.worker_lifecycle_state.mark_started()

    @property
    def worker_threads_started(self) -> bool:
        """Return whether callback worker threads have been started."""
        return self.worker_lifecycle_state.has_started

    def worker_threads_have_started(self) -> bool:
        """Return whether callback worker threads have been started."""
        return self.worker_threads_started

    @property
    def dosage_buffer_count(self) -> int:
        """Return the native dosage-buffer pool allocation count."""
        return self.dosage_buffer_pool.allocated_count

    @property
    def dosage_buffer_identifiers(self) -> set[int]:
        """Return the native dosage-buffer pool ownership identifiers."""
        return set(self.dosage_buffer_pool.buffer_identifiers)

    @property
    def result_in_flight_slot_count(self) -> int:
        """Return the native result in-flight occupied slot count."""
        return self.result_in_flight_slot_state.occupied_count

    def record_stage_duration(self, stage_name: str, start_time: float) -> None:
        """Record a nested callback stage using this runner's timing recorder."""
        timing.record_stage_duration(self.stage_timing_recorder, stage_name, start_time)

    def record_chunk_stage_duration(self, metadata: typing.Any, stage_name: str, start_time: float) -> None:
        """Record a nested callback stage for a specific native chunk."""
        record_stage_duration_with_optional_chunk(
            stage_timing_recorder=self.stage_timing_recorder,
            stage_name=stage_name,
            start_time=start_time,
            chunk_metadata=metadata,
        )

    def record_chunk_stage_elapsed_duration(
        self,
        metadata: typing.Any,
        stage_name: str,
        elapsed_seconds: float,
    ) -> None:
        """Record an already-measured chunk stage duration."""
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_chunk_stage_duration(
            chunk_identity=transfers.build_chunk_timing_identity(metadata),
            stage_name=stage_name,
            duration_seconds=elapsed_seconds,
        )

    def record_work_item_stage_elapsed_duration(
        self,
        work_item: (
            PreprocessedDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkBatchWorkItem
            | PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
        ),
        stage_name: str,
        elapsed_seconds: float,
    ) -> None:
        """Record a stage duration across one queued work item."""
        if isinstance(work_item, PreprocessedVariantMajorDosageChunkBatchWorkItem):
            duration_per_chunk = elapsed_seconds / len(work_item.work_items)
            for chunk_work_item in work_item.work_items:
                self.record_chunk_stage_elapsed_duration(chunk_work_item.metadata, stage_name, duration_per_chunk)
            return
        self.record_chunk_stage_elapsed_duration(work_item.metadata, stage_name, elapsed_seconds)

    def get_stage_duration_recorder(self) -> collections.abc.Callable[[str, float], None] | None:
        """Return an optional nested stage recorder for lower-level compute helpers."""
        if self.stage_timing_recorder is None:
            return None
        return self.record_stage_duration

    def record_queue_operation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        observed_queue: queue.Queue[typing.Any],
        elapsed_seconds: float,
        blocked: bool,
    ) -> None:
        """Record aggregate queue depth and wait metadata."""
        if self.stage_timing_recorder is None:
            return
        observation_plan = _core.plan_callback_queue_operation_observation(
            queue_name=queue_name,
            operation_name=operation_name,
            elapsed_seconds=elapsed_seconds,
            blocked=blocked,
        )
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=observation_plan.queue_name,
            operation_name=observation_plan.operation_name,
            queue_depth=observed_queue.qsize(),
            queue_capacity=observed_queue.maxsize,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=observation_plan.blocked_seconds,
        )

    def record_queue_stage_duration(
        self,
        *,
        queue_name: str,
        operation_name: str,
        observed_queue: queue.Queue[typing.Any],
        start_time: float,
        blocked: bool,
    ) -> None:
        """Record a queue stage duration plus aggregate pressure metadata."""
        elapsed_seconds = time.perf_counter() - start_time
        if self.stage_timing_recorder is None:
            return
        observation_plan = _core.plan_callback_queue_stage_observation(
            queue_name=queue_name,
            operation_name=operation_name,
            elapsed_seconds=elapsed_seconds,
            blocked=blocked,
        )
        self.stage_timing_recorder.add_stage_duration(observation_plan.stage_name, elapsed_seconds)
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=observation_plan.queue_name,
            operation_name=observation_plan.operation_name,
            queue_depth=observed_queue.qsize(),
            queue_capacity=observed_queue.maxsize,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=observation_plan.blocked_seconds,
        )

    def record_bounded_resource_operation(
        self,
        *,
        resource_name: str,
        operation_name: str,
        current_depth: int,
        capacity: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> None:
        """Record aggregate bounded-resource occupancy metadata."""
        if self.stage_timing_recorder is None:
            return
        observation_plan = _core.plan_callback_queue_operation_observation(
            queue_name=resource_name,
            operation_name=operation_name,
            elapsed_seconds=elapsed_seconds,
            blocked=blocked,
        )
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=observation_plan.queue_name,
            operation_name=observation_plan.operation_name,
            queue_depth=current_depth,
            queue_capacity=capacity,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=observation_plan.blocked_seconds,
        )

    def record_bounded_resource_stage_duration(
        self,
        *,
        resource_name: str,
        operation_name: str,
        current_depth: int,
        capacity: int,
        start_time: float,
        blocked: bool,
    ) -> None:
        """Record a bounded-resource stage duration plus pressure metadata."""
        elapsed_seconds = time.perf_counter() - start_time
        if self.stage_timing_recorder is None:
            return
        observation_plan = _core.plan_callback_queue_stage_observation(
            queue_name=resource_name,
            operation_name=operation_name,
            elapsed_seconds=elapsed_seconds,
            blocked=blocked,
        )
        self.stage_timing_recorder.add_stage_duration(observation_plan.stage_name, elapsed_seconds)
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=observation_plan.queue_name,
            operation_name=observation_plan.operation_name,
            queue_depth=current_depth,
            queue_capacity=capacity,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=observation_plan.blocked_seconds,
        )

    @abc.abstractmethod
    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed chunk and write it."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed variant-major chunk and write it."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: jax.Array | npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed packed8 chunk and write it."""
        raise NotImplementedError

    def compute_preprocessed_dosage_chunk(
        self,
        metadata: typing.Any,
        genotype_matrix: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed dosage chunk for JAX association."""
        if self.stage_timing_recorder is None:
            self.put_dosage_work_item(
                PreprocessedDosageChunkWorkItem(
                    metadata=metadata,
                    genotype_matrix=genotype_matrix,
                    chunk_stats=chunk_stats,
                )
            )
            return
        native_delivery_start_time = time.perf_counter()
        try:
            self.put_dosage_work_item(
                PreprocessedDosageChunkWorkItem(
                    metadata=metadata,
                    genotype_matrix=genotype_matrix,
                    chunk_stats=chunk_stats,
                )
            )
        finally:
            self.record_chunk_stage_duration(metadata, "native_delivery", native_delivery_start_time)

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: typing.Any,
        genotype_matrix_by_variant: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed variant-major dosage chunk for JAX association."""
        if self.stage_timing_recorder is None:
            self.put_dosage_work_item(
                PreprocessedVariantMajorDosageChunkWorkItem(
                    metadata=metadata,
                    genotype_matrix_by_variant=genotype_matrix_by_variant,
                    chunk_stats=chunk_stats,
                )
            )
            return
        native_delivery_start_time = time.perf_counter()
        try:
            self.put_dosage_work_item(
                PreprocessedVariantMajorDosageChunkWorkItem(
                    metadata=metadata,
                    genotype_matrix_by_variant=genotype_matrix_by_variant,
                    chunk_stats=chunk_stats,
                )
            )
        finally:
            self.record_chunk_stage_duration(metadata, "native_delivery", native_delivery_start_time)

    def compute_preprocessed_variant_major_dosage_chunk_batch(
        self,
        metadata_batch: collections.abc.Sequence[typing.Any],
        genotype_matrix_by_variant_batch: collections.abc.Sequence[npt.NDArray[np.float32]],
        chunk_stats_batch: collections.abc.Sequence[_core.ChunkStats],
    ) -> None:
        """Enqueue a native batch of variant-major dosage chunks for JAX association."""
        batch_handoff_plan = _core.plan_variant_major_dosage_batch_handoff(
            metadata_count=len(metadata_batch),
            genotype_matrix_by_variant_count=len(genotype_matrix_by_variant_batch),
            chunk_stats_count=len(chunk_stats_batch),
        )
        work_items = tuple(
            PreprocessedVariantMajorDosageChunkWorkItem(
                metadata=metadata,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                chunk_stats=chunk_stats,
            )
            for metadata, genotype_matrix_by_variant, chunk_stats in zip(
                metadata_batch,
                genotype_matrix_by_variant_batch,
                chunk_stats_batch,
                strict=True,
            )
        )
        if len(work_items) != batch_handoff_plan.chunk_count:
            message = "Native variant-major dosage batch handoff plan disagrees with prepared work items."
            raise RuntimeError(message)
        work_item = PreprocessedVariantMajorDosageChunkBatchWorkItem(work_items=work_items)
        if self.stage_timing_recorder is None:
            self.put_dosage_work_item(work_item)
            return
        native_delivery_start_time = time.perf_counter()
        try:
            self.put_dosage_work_item(work_item)
        finally:
            elapsed_seconds = time.perf_counter() - native_delivery_start_time
            self.record_work_item_stage_elapsed_duration(work_item, "native_delivery", elapsed_seconds)

    def compute_preprocessed_variant_major_packed8_probability_pair_chunk(
        self,
        metadata: typing.Any,
        packed_probability_pairs_by_variant: npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed packed8 chunk for JAX association."""
        if self.stage_timing_recorder is None:
            self.put_dosage_work_item(
                PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem(
                    metadata=metadata,
                    packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                    chunk_stats=chunk_stats,
                )
            )
            return
        native_delivery_start_time = time.perf_counter()
        try:
            self.put_dosage_work_item(
                PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem(
                    metadata=metadata,
                    packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                    chunk_stats=chunk_stats,
                )
            )
        finally:
            self.record_chunk_stage_duration(metadata, "native_delivery", native_delivery_start_time)

    def consume_dosage_chunks(self) -> None:
        """Consume queued dosage chunks and run JAX work in order."""
        try:
            if self.stage_timing_recorder is None:
                self.consume_dosage_chunks_without_timing()
                return
            while True:
                get_start_time = time.perf_counter()
                work_item = self.dosage_queue.get()
                if work_item is None:
                    return
                self.record_queue_stage_duration(
                    queue_name="dosage_queue",
                    operation_name="consumer_wait",
                    observed_queue=self.dosage_queue,
                    start_time=get_start_time,
                    blocked=True,
                )
                python_callback_start_time = time.perf_counter()
                try:
                    self.process_dosage_work_item(work_item)
                finally:
                    elapsed_seconds = time.perf_counter() - python_callback_start_time
                    self.record_work_item_stage_elapsed_duration(work_item, "python_callback", elapsed_seconds)
        except Exception as error:  # noqa: BLE001
            self.worker_error = error

    def consume_dosage_chunks_without_timing(self) -> None:
        """Consume queued dosage chunks without diagnostic timing overhead."""
        while True:
            work_item = self.dosage_queue.get()
            if work_item is None:
                return
            self.process_dosage_work_item(work_item)

    def process_dosage_work_item(
        self,
        work_item: (
            PreprocessedDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkBatchWorkItem
            | PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
        ),
    ) -> None:
        """Run one preprocessed dosage work item."""
        if isinstance(work_item, PreprocessedVariantMajorDosageChunkBatchWorkItem):
            for chunk_work_item in work_item.work_items:
                self.process_dosage_work_item(chunk_work_item)
            return
        if isinstance(work_item, PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem):
            self.compute_preprocessed_variant_major_packed8_chunk(
                variant_metadata=work_item.metadata,
                packed_probability_pairs_by_variant=work_item.packed_probability_pairs_by_variant,
                chunk_stats=work_item.chunk_stats,
            )
        elif isinstance(work_item, PreprocessedVariantMajorDosageChunkWorkItem):
            self.compute_preprocessed_variant_major_chunk(
                variant_metadata=work_item.metadata,
                genotype_matrix_by_variant=work_item.genotype_matrix_by_variant,
                chunk_stats=work_item.chunk_stats,
            )
        elif isinstance(work_item, PreprocessedDosageChunkWorkItem):
            self.compute_preprocessed_chunk(
                variant_metadata=work_item.metadata,
                genotype_matrix=work_item.genotype_matrix,
                chunk_stats=work_item.chunk_stats,
            )
        else:
            message = f"Unsupported preprocessed dosage work item: {type(work_item).__name__}"
            raise TypeError(message)
        self.record_progress(work_item.metadata)

    def record_progress(self, metadata: typing.Any) -> None:
        """Record throttled progress after one chunk is processed."""
        if self.telemetry_session is None:
            self.progress_state.record_processed_chunk_without_progress()
            return
        progress_update = self.progress_state.record_processed_chunk(build_native_callback_chunk_identity(metadata))
        if progress_update.completed_chromosome is not None:
            completed_processed_chunk_count = progress_update.completed_processed_chunk_count
            if completed_processed_chunk_count is None:
                message = "Native callback progress completion missing processed chunk count."
                raise RuntimeError(message)
            self.telemetry_session.log_event(
                "chromosome_completed",
                level="info",
                chromosome=progress_update.completed_chromosome,
                processed_chunk_count=completed_processed_chunk_count,
            )
        if progress_update.started_chromosome is not None:
            self.telemetry_session.log_event(
                "chromosome_started",
                level="info",
                chromosome=progress_update.started_chromosome,
                processed_chunk_count=progress_update.processed_chunk_count,
            )
        chunk_identity = progress_update.chunk_identity
        self.telemetry_session.log_progress(
            processed_chunk_count=progress_update.processed_chunk_count,
            chromosome=chunk_identity.chromosome,
            chunk_identifier=chunk_identity.chunk_identifier,
            variant_start_index=chunk_identity.variant_start_index,
            variant_stop_index=chunk_identity.variant_stop_index,
            variant_count=chunk_identity.variant_count,
        )

    def complete_progress(self) -> None:
        """Emit the native final progress completion event when telemetry consumed chunks."""
        progress_completion = self.progress_state.finish_progress()
        if self.telemetry_session is None or progress_completion is None:
            return
        self.telemetry_session.log_event(
            "chromosome_completed",
            level="info",
            chromosome=progress_completion.chromosome,
            processed_chunk_count=progress_completion.processed_chunk_count,
        )

    def record_binary_null_model_failure_count(self, failure_count: int) -> None:
        """Accumulate binary null-model failures for run-level telemetry."""
        self.binary_correction_summary.add_null_model_failure_count(failure_count)

    def record_binary_correction_diagnostics(
        self,
        binary_chunk_diagnostics: regenie2_binary.BinaryChunkDiagnostics | None,
    ) -> None:
        """Accumulate binary correction diagnostics for run-level telemetry."""
        if binary_chunk_diagnostics is None:
            return
        if self.telemetry_session is None:
            return
        self.binary_correction_pending_diagnostics.append(binary_chunk_diagnostics)

    def flush_binary_correction_diagnostics(self) -> None:
        """Materialize pending binary diagnostics and accumulate them into native summary counters."""
        if not self.binary_correction_pending_diagnostics:
            return
        pending_diagnostics = tuple(self.binary_correction_pending_diagnostics)
        diagnostics_counts = binary_chunk_diagnostics_to_summary_counts(pending_diagnostics)
        self.binary_correction_pending_diagnostics.clear()
        self.add_binary_correction_summary_counts(diagnostics_counts)

    def add_binary_correction_summary_counts(
        self,
        diagnostics_counts: regenie2_binary.BinaryCorrectionSummaryCounts,
    ) -> None:
        """Accumulate one host-materialized binary diagnostics summary."""
        self.binary_correction_summary.add_diagnostics_totals(
            diagnostics_counts.chunk_count,
            diagnostics_counts.score_only_count,
            diagnostics_counts.score_test_candidate_count,
            diagnostics_counts.firth_candidate_count,
            diagnostics_counts.firth_converged_count,
            diagnostics_counts.firth_failed_count,
            diagnostics_counts.firth_numerical_failure_count,
            diagnostics_counts.firth_max_iteration_failure_count,
            diagnostics_counts.firth_invalid_statistic_failure_count,
            diagnostics_counts.firth_step_halving_failure_count,
            diagnostics_counts.pseudo_firth_attempt_count,
            diagnostics_counts.pseudo_firth_success_count,
            diagnostics_counts.nr_zero_start_attempt_count,
            diagnostics_counts.nr_zero_start_success_count,
            diagnostics_counts.nr_warm_start_attempt_count,
            diagnostics_counts.nr_warm_start_success_count,
            diagnostics_counts.sparse_correction_count,
            diagnostics_counts.dense_correction_count,
        )

    def emit_binary_correction_summary(self) -> None:
        """Emit aggregate binary correction diagnostics when a binary run produced them."""
        if self.telemetry_session is None:
            return
        self.flush_binary_correction_diagnostics()
        if not self.binary_correction_summary.should_emit():
            return
        summary_payload = typing.cast("dict[str, object]", self.binary_correction_summary.summary_payload())
        self.telemetry_session.log_event(
            "binary_correction_summary",
            level="info",
            **summary_payload,
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed JAX results and write them in order."""
        try:
            if self.stage_timing_recorder is None:
                self.consume_result_write_items_without_timing()
                return
            while True:
                get_start_time = time.perf_counter()
                work_item = self.result_queue.get()
                if work_item is None:
                    self.flush_binary_correction_diagnostics()
                    return
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="consumer_wait",
                    observed_queue=self.result_queue,
                    start_time=get_start_time,
                    blocked=True,
                )
                self.process_result_write_item(work_item)
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def consume_result_write_items_without_timing(self) -> None:
        """Consume result write items without diagnostic queue timing overhead."""
        while True:
            work_item = self.result_queue.get()
            if work_item is None:
                self.flush_binary_correction_diagnostics()
                return
            self.process_result_write_item(work_item)

    def process_result_write_item(
        self,
        work_item: Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem,
    ) -> None:
        """Materialize and write one computed result work item."""
        host_dosage_buffer_released = False
        try:
            materialized_chunk = materialize_regenie2_native_chunk_with_optional_timing(
                metadata=work_item.metadata,
                beta=work_item.beta,
                standard_error=work_item.standard_error,
                chi_squared=work_item.chi_squared,
                log10_p_value=work_item.log10_p_value,
                extra_code=work_item.extra_code,
                stage_timing_recorder=self.stage_timing_recorder,
                output_statistic_dtype=self.output_statistic_dtype,
            )
            self.release_result_work_item_host_buffer(work_item)
            host_dosage_buffer_released = True
            write_materialized_regenie2_native_chunk_with_optional_timing(
                writer_session=typing.cast("typing.Any", self).writer_session,
                metadata=work_item.metadata,
                chunk_stats=work_item.chunk_stats,
                materialized_chunk=materialized_chunk,
                stage_timing_recorder=self.stage_timing_recorder,
                output_statistic_dtype=self.output_statistic_dtype,
            )
            record_binary_chunk_diagnostics_from_count(
                stage_timing_recorder=self.stage_timing_recorder,
                diagnostics=work_item.binary_chunk_diagnostics,
            )
            self.record_binary_correction_diagnostics(work_item.binary_chunk_diagnostics)
        finally:
            if not host_dosage_buffer_released:
                self.release_result_work_item_host_buffer(work_item)
            self.release_result_work_item_in_flight_slot(work_item)

    def put_dosage_work_item(
        self,
        work_item: (
            PreprocessedDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkBatchWorkItem
            | PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
            | None
        ),
    ) -> None:
        """Put work into the bounded worker queue while surfacing worker errors."""
        self.start()
        if self.stage_timing_recorder is None:
            while True:
                self.raise_worker_error_if_present()
                try:
                    self.dosage_queue.put(work_item, timeout=CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS)
                    return
                except queue.Full:
                    continue
        while True:
            self.raise_worker_error_if_present()
            put_start_time = time.perf_counter()
            try:
                self.dosage_queue.put(work_item, timeout=CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS)
                self.record_queue_stage_duration(
                    queue_name="dosage_queue",
                    operation_name="put",
                    observed_queue=self.dosage_queue,
                    start_time=put_start_time,
                    blocked=False,
                )
                return
            except queue.Full:
                self.record_queue_stage_duration(
                    queue_name="dosage_queue",
                    operation_name="producer_blocking",
                    observed_queue=self.dosage_queue,
                    start_time=put_start_time,
                    blocked=True,
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
        self.start()
        if self.stage_timing_recorder is None:
            while True:
                self.raise_worker_error_if_present()
                try:
                    self.result_queue.put(work_item, timeout=CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS)
                    return
                except queue.Full:
                    continue
        while True:
            self.raise_worker_error_if_present()
            put_start_time = time.perf_counter()
            try:
                self.result_queue.put(work_item, timeout=CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS)
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="put",
                    observed_queue=self.result_queue,
                    start_time=put_start_time,
                    blocked=False,
                )
                return
            except queue.Full:
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="producer_blocking",
                    observed_queue=self.result_queue,
                    start_time=put_start_time,
                    blocked=True,
                )
                continue

    def acquire_result_in_flight_slot(self) -> None:
        """Reserve capacity for one chunk of pending GPU result work."""
        if self.stage_timing_recorder is None:
            while True:
                self.raise_worker_error_if_present()
                if self.result_in_flight_slots.acquire(timeout=CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS):
                    with self.result_in_flight_slot_lock:
                        if not self.result_in_flight_slot_state.acquire_slot():
                            self.result_in_flight_slots.release()
                            message = "Native result in-flight slot state has no available slot."
                            raise RuntimeError(message)
                    return
        while True:
            self.raise_worker_error_if_present()
            acquire_start_time = time.perf_counter()
            if self.result_in_flight_slots.acquire(timeout=CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS):
                with self.result_in_flight_slot_lock:
                    if not self.result_in_flight_slot_state.acquire_slot():
                        self.result_in_flight_slots.release()
                        message = "Native result in-flight slot state has no available slot."
                        raise RuntimeError(message)
                    current_depth = self.result_in_flight_slot_state.occupied_count
                self.record_bounded_resource_stage_duration(
                    resource_name="result_in_flight_slots",
                    operation_name="acquire",
                    current_depth=current_depth,
                    capacity=self.result_in_flight_limit,
                    start_time=acquire_start_time,
                    blocked=False,
                )
                return
            with self.result_in_flight_slot_lock:
                current_depth = self.result_in_flight_slot_state.occupied_count
            self.record_bounded_resource_stage_duration(
                resource_name="result_in_flight_slots",
                operation_name="producer_blocking",
                current_depth=current_depth,
                capacity=self.result_in_flight_limit,
                start_time=acquire_start_time,
                blocked=True,
            )

    def release_result_in_flight_slot(self) -> None:
        """Release capacity for one completed chunk of GPU result work."""
        self.result_in_flight_slots.release()
        with self.result_in_flight_slot_lock:
            self.result_in_flight_slot_state.release_slot()
            current_depth = self.result_in_flight_slot_state.occupied_count
        self.record_bounded_resource_operation(
            resource_name="result_in_flight_slots",
            operation_name="release",
            current_depth=current_depth,
            capacity=self.result_in_flight_limit,
            elapsed_seconds=0.0,
            blocked=False,
        )

    def finish(self) -> None:
        """Wait until all queued JAX work has been written."""
        self.stop_dosage_worker(timeout_seconds=None)
        self.join_dosage_worker(timeout_seconds=GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS)
        self.stop_result_worker(timeout_seconds=None)
        self.join_result_worker(timeout_seconds=GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS)
        self.raise_worker_error_if_present()
        self.complete_progress()
        self.emit_binary_correction_summary()

    def abort(self) -> None:
        """Stop the worker after an upstream failure."""
        with contextlib.suppress(NativeBgenWorkerShutdownError):
            self.stop_dosage_worker(timeout_seconds=WORKER_ABORT_STOP_TIMEOUT_SECONDS)
        with contextlib.suppress(NativeBgenWorkerShutdownError):
            self.stop_result_worker(timeout_seconds=WORKER_ABORT_STOP_TIMEOUT_SECONDS)

    def stop_dosage_worker(self, timeout_seconds: float | None) -> None:
        """Signal the dosage worker to exit after queued dosage chunks drain."""
        effective_timeout_seconds = DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        if not _core.should_attempt_callback_worker_stop(
            has_started=self.worker_threads_have_started(),
            has_worker_error=self.worker_error is not None,
            is_worker_alive=self.worker_thread.is_alive(),
        ):
            return
        stop_deadline = time.monotonic() + effective_timeout_seconds
        while time.monotonic() < stop_deadline:
            if not _core.should_attempt_callback_worker_stop(
                has_started=self.worker_threads_have_started(),
                has_worker_error=self.worker_error is not None,
                is_worker_alive=self.worker_thread.is_alive(),
            ):
                return
            current_timeout_seconds = _core.resolve_callback_worker_stop_poll_timeout_seconds(
                stop_deadline - time.monotonic()
            )
            try:
                self.dosage_queue.put(None, timeout=current_timeout_seconds)
                return
            except queue.Full:
                continue
        raise NativeBgenWorkerShutdownError(
            worker_name=self.worker_thread.name,
            timeout_seconds=effective_timeout_seconds,
        )

    def join_dosage_worker(self, timeout_seconds: float | None) -> None:
        """Join the dosage worker with a bounded shutdown wait."""
        if not self.worker_threads_have_started():
            return
        effective_timeout_seconds = DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        self.worker_thread.join(timeout=effective_timeout_seconds)
        if self.worker_thread.is_alive():
            raise NativeBgenWorkerShutdownError(
                worker_name=self.worker_thread.name,
                timeout_seconds=effective_timeout_seconds,
            )

    def stop_result_worker(self, timeout_seconds: float | None) -> None:
        """Signal the result worker to exit after queued results drain."""
        effective_timeout_seconds = RESULT_WORKER_JOIN_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        if not _core.should_attempt_callback_worker_stop(
            has_started=self.worker_threads_have_started(),
            has_worker_error=self.result_worker_error is not None,
            is_worker_alive=self.result_worker_thread.is_alive(),
        ):
            return
        stop_deadline = time.monotonic() + effective_timeout_seconds
        while time.monotonic() < stop_deadline:
            if not _core.should_attempt_callback_worker_stop(
                has_started=self.worker_threads_have_started(),
                has_worker_error=self.result_worker_error is not None,
                is_worker_alive=self.result_worker_thread.is_alive(),
            ):
                return
            current_timeout_seconds = _core.resolve_callback_worker_stop_poll_timeout_seconds(
                stop_deadline - time.monotonic()
            )
            try:
                self.result_queue.put(None, timeout=current_timeout_seconds)
                return
            except queue.Full:
                continue
        raise NativeBgenWorkerShutdownError(
            worker_name=self.result_worker_thread.name,
            timeout_seconds=effective_timeout_seconds,
        )

    def join_result_worker(self, timeout_seconds: float | None) -> None:
        """Join the result writer worker with a bounded shutdown wait."""
        if not self.worker_threads_have_started():
            return
        effective_timeout_seconds = RESULT_WORKER_JOIN_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        self.result_worker_thread.join(timeout=effective_timeout_seconds)
        if self.result_worker_thread.is_alive():
            raise NativeBgenWorkerShutdownError(
                worker_name=self.result_worker_thread.name,
                timeout_seconds=effective_timeout_seconds,
            )

    def acquire_dosage_buffer(self, sample_count: int, variant_count: int) -> npt.NDArray[np.float32]:
        """Return a reusable host dosage buffer for Rust to fill."""
        expected_shape = (sample_count, variant_count)
        return typing.cast(
            "npt.NDArray[np.float32]",
            self.acquire_dosage_buffer_with_shape(expected_shape, np.float32),
        )

    def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> npt.NDArray[np.float32]:
        """Return a reusable host variant-major dosage buffer for Rust to fill."""
        expected_shape = (variant_count, sample_count)
        return typing.cast(
            "npt.NDArray[np.float32]",
            self.acquire_dosage_buffer_with_shape(expected_shape, np.float32),
        )

    def acquire_variant_major_packed8_probability_pair_buffer(
        self,
        variant_count: int,
        sample_count: int,
    ) -> npt.NDArray[np.uint8]:
        """Return a reusable host variant-major packed8 probability-pair buffer."""
        expected_shape = (variant_count, sample_count, 2)
        return typing.cast(
            "npt.NDArray[np.uint8]",
            self.acquire_dosage_buffer_with_shape(expected_shape, np.uint8),
        )

    @staticmethod
    def _dosage_buffer_owner(dosage_buffer: HostGenotypeBuffer) -> HostGenotypeBuffer:
        """Return the base owner array for a dosage buffer view."""
        dosage_buffer_owner = dosage_buffer
        dosage_buffer_base = dosage_buffer_owner.base
        while isinstance(dosage_buffer_base, np.ndarray):
            dosage_buffer_owner = dosage_buffer_base
            dosage_buffer_base = dosage_buffer_owner.base
        return dosage_buffer_owner

    @staticmethod
    def _acquire_reused_dosage_buffer(
        dosage_buffer: HostGenotypeBuffer,
        expected_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> HostGenotypeBuffer | None:
        """Return a reused buffer if dtype/shape constraints are met, else None."""
        if dosage_buffer.dtype != dtype:
            return None
        reuse_plan = _core.plan_dosage_buffer_reuse(
            buffered_shape=dosage_buffer.shape,
            expected_shape=expected_shape,
        )
        if reuse_plan is None:
            return None
        if not reuse_plan.requires_slice:
            return dosage_buffer
        slices = tuple(slice(0, dimension_size) for dimension_size in reuse_plan.slice_dimensions)
        return dosage_buffer[slices]

    def acquire_dosage_buffer_with_shape(
        self,
        expected_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> HostGenotypeBuffer:
        """Return a reusable host dosage buffer with the requested shape."""
        while True:
            self.raise_worker_error_if_present()
            with contextlib.suppress(queue.Empty):
                dosage_buffer = self.free_dosage_buffers.get_nowait()
                reused_dosage_buffer = self._acquire_reused_dosage_buffer(
                    dosage_buffer,
                    expected_shape=expected_shape,
                    dtype=dtype,
                )
                if reused_dosage_buffer is not None:
                    self.record_queue_operation(
                        queue_name="dosage_buffer_pool",
                        operation_name="reuse",
                        observed_queue=self.free_dosage_buffers,
                        elapsed_seconds=0.0,
                        blocked=False,
                    )
                    return reused_dosage_buffer
                self.discard_dosage_buffer_slot(dosage_buffer)
                if self.dosage_buffer_pool.has_available_slot():
                    return self.allocate_dosage_buffer_with_shape(expected_shape, dtype)
                continue
            if self.dosage_buffer_pool.has_available_slot():
                return self.allocate_dosage_buffer_with_shape(expected_shape, dtype)
            with contextlib.suppress(queue.Empty):
                if self.stage_timing_recorder is None:
                    dosage_buffer = self.free_dosage_buffers.get(
                        timeout=CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS
                    )
                else:
                    buffer_wait_start_time = time.perf_counter()
                    dosage_buffer = self.free_dosage_buffers.get(
                        timeout=CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS
                    )
                    self.record_queue_stage_duration(
                        queue_name="dosage_buffer_pool",
                        operation_name="consumer_wait",
                        observed_queue=self.free_dosage_buffers,
                        start_time=buffer_wait_start_time,
                        blocked=True,
                    )
                reused_dosage_buffer = self._acquire_reused_dosage_buffer(
                    dosage_buffer,
                    expected_shape=expected_shape,
                    dtype=dtype,
                )
                if reused_dosage_buffer is not None:
                    self.record_queue_operation(
                        queue_name="dosage_buffer_pool",
                        operation_name="reuse",
                        observed_queue=self.free_dosage_buffers,
                        elapsed_seconds=0.0,
                        blocked=False,
                    )
                    return reused_dosage_buffer
                self.discard_dosage_buffer_slot(dosage_buffer)
                if self.dosage_buffer_pool.has_available_slot():
                    return self.allocate_dosage_buffer_with_shape(expected_shape, dtype)

    def release_dosage_buffer(self, dosage_buffer: HostGenotypeBuffer) -> None:
        """Return a processed host dosage buffer to the reusable pool."""
        dosage_buffer_owner = self._dosage_buffer_owner(dosage_buffer)
        if not self.dosage_buffer_pool.owns_buffer(id(dosage_buffer_owner)):
            return
        try:
            self.free_dosage_buffers.put_nowait(dosage_buffer_owner)
            self.record_queue_operation(
                queue_name="dosage_buffer_pool",
                operation_name="return",
                observed_queue=self.free_dosage_buffers,
                elapsed_seconds=0.0,
                blocked=False,
            )
        except queue.Full:
            self.record_queue_operation(
                queue_name="dosage_buffer_pool",
                operation_name="return_full",
                observed_queue=self.free_dosage_buffers,
                elapsed_seconds=0.0,
                blocked=False,
            )
            self.discard_dosage_buffer_slot(dosage_buffer)

    def allocate_dosage_buffer_with_shape(
        self,
        expected_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> HostGenotypeBuffer:
        """Allocate and register one host genotype buffer slot."""
        dosage_buffer = typing.cast("HostGenotypeBuffer", np.empty(expected_shape, dtype=dtype, order="C"))
        if not self.dosage_buffer_pool.register_buffer(id(dosage_buffer)):
            message = "Native dosage-buffer pool has no available slot for allocation."
            raise RuntimeError(message)
        self.record_queue_operation(
            queue_name="dosage_buffer_pool",
            operation_name="allocate",
            observed_queue=self.free_dosage_buffers,
            elapsed_seconds=0.0,
            blocked=False,
        )
        return dosage_buffer

    def discard_dosage_buffer_slot(self, dosage_buffer: HostGenotypeBuffer) -> None:
        """Remove one discarded host genotype buffer slot from pool accounting."""
        dosage_buffer_identifier = id(dosage_buffer)
        if not self.dosage_buffer_pool.discard_buffer(dosage_buffer_identifier):
            return
        self.record_queue_operation(
            queue_name="dosage_buffer_pool",
            operation_name="discard",
            observed_queue=self.free_dosage_buffers,
            elapsed_seconds=0.0,
            blocked=False,
        )

    def release_numpy_dosage_buffer(self, dosage_buffer: jax.Array | HostGenotypeBuffer) -> None:
        """Return a NumPy host dosage buffer to the pool after device transfer."""
        if isinstance(dosage_buffer, np.ndarray):
            self.release_dosage_buffer(typing.cast("HostGenotypeBuffer", dosage_buffer))

    def get_releasable_dosage_buffer(
        self,
        dosage_buffer: jax.Array | HostGenotypeBuffer,
    ) -> HostGenotypeBuffer | None:
        """Return a host dosage buffer reference when it belongs to the reusable pool."""
        if isinstance(dosage_buffer, np.ndarray):
            host_dosage_buffer = typing.cast("HostGenotypeBuffer", dosage_buffer)
            dosage_buffer_owner = self._dosage_buffer_owner(host_dosage_buffer)
            if self.dosage_buffer_pool.owns_buffer(id(dosage_buffer_owner)):
                return dosage_buffer_owner
        return None

    def release_result_work_item_buffer(
        self,
        work_item: Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem,
    ) -> None:
        """Release resources after a dependent JAX result is materialized."""
        self.release_result_work_item_host_buffer(work_item)
        self.release_result_work_item_in_flight_slot(work_item)

    def release_result_work_item_host_buffer(
        self,
        work_item: Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem,
    ) -> None:
        """Release the host genotype buffer associated with one result."""
        if work_item.host_dosage_buffer is not None:
            self.release_dosage_buffer(work_item.host_dosage_buffer)

    def release_result_work_item_in_flight_slot(
        self,
        work_item: Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem,
    ) -> None:
        """Release the in-flight result slot associated with one result."""
        if work_item.release_in_flight_slot:
            self.release_result_in_flight_slot()


__all__ = [
    "DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
    "GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
    "GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
    "RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
    "WORKER_ABORT_STOP_TIMEOUT_SECONDS",
    "NativeBgenCallbackRunner",
    "require_current_chromosome_state",
]
