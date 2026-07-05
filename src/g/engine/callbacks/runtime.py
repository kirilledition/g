"""Core callback lifecycle and bounded-queue runtime used by REGENIE callbacks."""

from __future__ import annotations

import abc
import enum
import time
import typing

import numpy as np
import numpy.typing as npt

import g.engine.callbacks.diagnostics as diagnostics
import g.engine.callbacks.shared as shared
import g.engine.callbacks.transfers as transfers
import g.engine.callbacks.writers as writers
from g import _core, types
from g.engine import timing as engine_timing

if typing.TYPE_CHECKING:
    import collections.abc

    import jax

    from g.compute.regenie2_binary import api as regenie2_binary
    from g.runner import events

type PreprocessedDosageWorkItem = (
    shared.PreprocessedDosageChunkWorkItem
    | shared.PreprocessedVariantMajorDosageChunkWorkItem
    | shared.PreprocessedVariantMajorDosageChunkBatchWorkItem
    | shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
)
type QueuedPreprocessedDosageWorkItem = PreprocessedDosageWorkItem | None
type QueuedResultWriteWorkItem = shared.Regenie2ResultWriteWorkItem | shared.Regenie2MultiResultWriteWorkItem | None
DOSAGE_BUFFER_POOL_REUSE_OPERATION = "reuse"
DOSAGE_BUFFER_POOL_RETURN_OPERATION = "return"
DOSAGE_BUFFER_POOL_ALLOCATE_OPERATION = "allocate"
DOSAGE_BUFFER_POOL_DISCARD_OPERATION = "discard"
DOSAGE_BUFFER_POOL_CONSUMER_WAIT_OPERATION = "consumer_wait"


class ResultWriteItemKind(enum.StrEnum):
    """Native result-write work-item kind values."""

    SINGLE_RESULT = "single_result"
    MULTI_RESULT = "multi_result"
    STOP_SIGNAL = "stop_signal"


class DosageWorkItemKind(enum.StrEnum):
    """Native dosage work-item kind values."""

    SAMPLE_MAJOR_DOSAGE = "sample_major_dosage"
    VARIANT_MAJOR_DOSAGE = "variant_major_dosage"
    VARIANT_MAJOR_DOSAGE_BATCH = "variant_major_dosage_batch"
    VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR = "variant_major_packed8_probability_pair"
    STOP_SIGNAL = "stop_signal"


def classify_result_write_item(
    work_item: QueuedResultWriteWorkItem,
) -> ResultWriteItemKind:
    """Classify one result write item for native scheduler dispatch."""
    if work_item is None:
        return ResultWriteItemKind.STOP_SIGNAL
    if isinstance(work_item, shared.Regenie2MultiResultWriteWorkItem):
        return ResultWriteItemKind.MULTI_RESULT
    if isinstance(work_item, shared.Regenie2ResultWriteWorkItem):
        return ResultWriteItemKind.SINGLE_RESULT
    message = f"Unsupported result write work item type: {type(work_item).__name__}"
    raise TypeError(message)


def classify_dosage_work_item(
    work_item: QueuedPreprocessedDosageWorkItem,
) -> DosageWorkItemKind:
    """Classify one dosage work item for native scheduler dispatch."""
    if work_item is None:
        return DosageWorkItemKind.STOP_SIGNAL
    if isinstance(work_item, shared.PreprocessedVariantMajorDosageChunkBatchWorkItem):
        return DosageWorkItemKind.VARIANT_MAJOR_DOSAGE_BATCH
    if isinstance(work_item, shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem):
        return DosageWorkItemKind.VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR
    if isinstance(work_item, shared.PreprocessedVariantMajorDosageChunkWorkItem):
        return DosageWorkItemKind.VARIANT_MAJOR_DOSAGE
    if isinstance(work_item, shared.PreprocessedDosageChunkWorkItem):
        return DosageWorkItemKind.SAMPLE_MAJOR_DOSAGE
    message = f"Unsupported preprocessed dosage work item type: {type(work_item).__name__}"
    raise TypeError(message)


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
        expected_result_work_item_kind: ResultWriteItemKind,
        flush_binary_correction_diagnostics_on_result_stop: bool,
        result_in_flight_limit: int | None,
        dosage_buffer_limit: int | None,
        stage_timing_recorder: engine_timing.StageTimingRecorder | None,
        telemetry_session: events.TelemetrySession | None,
        output_statistic_dtype: types.FloatingPointDtype,
    ) -> None:
        """Initialize shared native callback state."""
        self.callback_runtime_resources = _core.NativeCallbackRuntimeResources(
            worker_name=worker_name,
            dosage_worker_target=self.consume_dosage_chunks,
            result_worker_target=self.consume_result_write_items,
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            expected_result_work_item_kind=expected_result_work_item_kind.value,
            has_telemetry_session=telemetry_session is not None,
            flush_binary_correction_diagnostics_on_result_stop=flush_binary_correction_diagnostics_on_result_stop,
            has_stage_timing_recorder=stage_timing_recorder is not None,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
        )
        self.expected_result_work_item_kind = expected_result_work_item_kind
        self.flush_binary_correction_diagnostics_on_result_stop = flush_binary_correction_diagnostics_on_result_stop
        self.stage_timing_recorder = stage_timing_recorder
        self.telemetry_session = telemetry_session
        self.output_statistic_dtype = output_statistic_dtype
        self.worker_error_cause: BaseException | None = None
        self.result_worker_error_cause: BaseException | None = None
        self.binary_correction_pending_diagnostics: list[regenie2_binary.BinaryChunkDiagnostics] = []

    @property
    def processed_chunk_count(self) -> int:
        """Return the native processed chunk count."""
        return self.callback_runtime_resources.processed_chunk_count

    @property
    def current_progress_chromosome(self) -> str | None:
        """Return the native active progress chromosome."""
        return self.callback_runtime_resources.current_progress_chromosome

    @property
    def binary_correction_summary_chunk_count(self) -> int:
        """Return the number of chunks included in binary correction summary telemetry."""
        return self.callback_runtime_resources.binary_correction_chunk_count_with_pending_diagnostics(
            self.binary_correction_pending_diagnostics,
        )

    def start(self) -> None:
        """Start asynchronous callback workers after owner setup is complete."""
        start_attempt_plan = self.callback_runtime_resources.start_workers()
        if start_attempt_plan.has_start_error:
            error_message = start_attempt_plan.error_message
            if error_message is None:
                error_message = "Native callback worker lifecycle failed to mark workers started."
            raise RuntimeError(error_message)

    @property
    def worker_threads_started(self) -> bool:
        """Return whether callback worker threads have been started."""
        return self.callback_runtime_resources.has_started

    @property
    def dosage_worker_name(self) -> str:
        """Return the native dosage worker name."""
        return self.callback_runtime_resources.dosage_worker_name

    @property
    def result_worker_name(self) -> str:
        """Return the native result worker name."""
        return self.callback_runtime_resources.result_worker_name

    @property
    def dosage_worker_is_alive(self) -> bool:
        """Return whether the native dosage worker thread is alive."""
        return self.callback_runtime_resources.dosage_worker_is_alive

    @property
    def result_worker_is_alive(self) -> bool:
        """Return whether the native result worker thread is alive."""
        return self.callback_runtime_resources.result_worker_is_alive

    @property
    def native_callback_batch_size(self) -> int:
        """Return the native callback batch size."""
        return self.callback_runtime_resources.native_callback_batch_size

    @property
    def dosage_queue_depth(self) -> int:
        """Return the native dosage queue depth."""
        return self.callback_runtime_resources.dosage_queue_depth

    @property
    def result_queue_depth(self) -> int:
        """Return the native result queue depth."""
        return self.callback_runtime_resources.result_queue_depth

    @property
    def result_in_flight_limit(self) -> int:
        """Return the native result in-flight limit."""
        return self.callback_runtime_resources.result_in_flight_limit

    @property
    def dosage_buffer_limit(self) -> int:
        """Return the native dosage buffer limit."""
        return self.callback_runtime_resources.dosage_buffer_limit

    @property
    def worker_error(self) -> BaseException | None:
        """Return the Python dosage worker exception cause."""
        return self.worker_error_cause

    @worker_error.setter
    def worker_error(self, error: BaseException | None) -> None:
        """Update dosage worker failure state in native scheduler state."""
        self.worker_error_cause = error
        error_message = None if error is None else str(error)
        self.callback_runtime_resources.update_dosage_worker_error(error_message)

    @property
    def result_worker_error(self) -> BaseException | None:
        """Return the Python result worker exception cause."""
        return self.result_worker_error_cause

    @result_worker_error.setter
    def result_worker_error(self, error: BaseException | None) -> None:
        """Update result worker failure state in native scheduler state."""
        self.result_worker_error_cause = error
        error_message = None if error is None else str(error)
        self.callback_runtime_resources.update_result_worker_error(error_message)

    @property
    def dosage_buffer_count(self) -> int:
        """Return the native dosage-buffer pool allocation count."""
        return self.callback_runtime_resources.dosage_buffer_allocated_count

    @property
    def dosage_buffer_identifiers(self) -> set[int]:
        """Return the native dosage-buffer pool ownership identifiers."""
        return set(self.callback_runtime_resources.dosage_buffer_identifiers)

    @property
    def free_dosage_buffer_count(self) -> int:
        """Return the number of host buffers waiting for reuse."""
        return self.callback_runtime_resources.free_dosage_buffer_count

    @property
    def result_in_flight_slot_count(self) -> int:
        """Return the native result in-flight occupied slot count."""
        return self.callback_runtime_resources.result_in_flight_occupied_count

    @property
    def result_queue_count(self) -> int:
        """Return the native result-queue occupancy count."""
        return self.callback_runtime_resources.result_queue_occupied_count

    @property
    def dosage_queue_count(self) -> int:
        """Return the native dosage-queue occupancy count."""
        return self.callback_runtime_resources.dosage_queue_occupied_count

    def record_processed_chunk(
        self,
        chunk_identity: _core.NativeCallbackChunkIdentity,
    ) -> _core.NativeCallbackProgressUpdate:
        """Record native progress through the active runtime owner."""
        return self.callback_runtime_resources.record_processed_chunk(chunk_identity)

    def record_processed_chunk_without_progress(self) -> None:
        """Record native progress without telemetry through the active runtime owner."""
        self.callback_runtime_resources.record_processed_chunk_without_progress()

    def finish_progress_state(self) -> _core.NativeCallbackProgressCompletion | None:
        """Finish native progress through the active runtime owner."""
        return self.callback_runtime_resources.finish_progress()

    def record_stage_duration(self, stage_name: str, start_time: float) -> None:
        """Record a nested callback stage using this runner's timing recorder."""
        engine_timing.record_stage_duration(self.stage_timing_recorder, stage_name, start_time)

    def record_chunk_stage_duration(self, metadata: typing.Any, stage_name: str, start_time: float) -> None:
        """Record a nested callback stage for a specific native chunk."""
        transfers.record_stage_duration_with_optional_chunk(
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
            shared.PreprocessedDosageChunkWorkItem
            | shared.PreprocessedVariantMajorDosageChunkWorkItem
            | shared.PreprocessedVariantMajorDosageChunkBatchWorkItem
            | shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
        ),
        stage_name: str,
        elapsed_seconds: float,
    ) -> None:
        """Record a stage duration across one queued work item."""
        attribution = self.callback_runtime_resources.plan_dosage_work_item_stage_duration_attribution_for_object(
            work_item,
            elapsed_seconds,
        )
        chunk_metadata_items = attribution.metadata_items
        stage_duration_plan = attribution.stage_duration_plan
        if stage_duration_plan.chunk_count != len(chunk_metadata_items):
            message = "Native dosage work stage duration plan disagrees with the work item chunk count."
            raise RuntimeError(message)
        for chunk_metadata in chunk_metadata_items:
            self.record_chunk_stage_elapsed_duration(
                chunk_metadata,
                stage_name,
                stage_duration_plan.duration_per_chunk,
            )

    def plan_dosage_work_item_stage_duration(
        self,
        *,
        dosage_work_item_kind: str,
        chunk_count: int,
        elapsed_seconds: float,
    ) -> _core.NativeDosageWorkItemStageDurationPlan:
        """Plan dosage work-item stage duration attribution through the active native owner."""
        return self.callback_runtime_resources.plan_dosage_work_item_stage_duration(
            dosage_work_item_kind,
            chunk_count,
            elapsed_seconds,
        )

    def get_stage_duration_recorder(self) -> collections.abc.Callable[[str, float], None] | None:
        """Return an optional nested stage recorder for lower-level compute helpers."""
        if self.stage_timing_recorder is None:
            return None
        return self.record_stage_duration

    def record_bounded_resource_operation(
        self,
        *,
        resource_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> None:
        """Record aggregate bounded-resource occupancy metadata."""
        if self.stage_timing_recorder is None:
            return
        observation = self.plan_current_queue_backpressure_observation(
            queue_name=resource_name,
            operation_name=operation_name,
            elapsed_seconds=elapsed_seconds,
            blocked=blocked,
        )
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=observation.queue_name,
            operation_name=observation.operation_name,
            queue_depth=observation.queue_depth,
            queue_capacity=observation.queue_capacity,
            elapsed_seconds=observation.elapsed_seconds,
            blocked_seconds=observation.blocked_seconds,
        )

    def record_queue_backpressure_observation(
        self,
        observation: _core.NativeCallbackQueueBackpressureObservation | None,
    ) -> None:
        """Record a native queue backpressure observation."""
        if observation is None:
            return
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=observation.queue_name,
            operation_name=observation.operation_name,
            queue_depth=observation.queue_depth,
            queue_capacity=observation.queue_capacity,
            elapsed_seconds=observation.elapsed_seconds,
            blocked_seconds=observation.blocked_seconds,
        )

    def record_queue_stage_backpressure_observation(
        self,
        observation: _core.NativeCallbackQueueStageBackpressureObservation | None,
    ) -> None:
        """Record a native queue stage and backpressure observation."""
        if observation is None:
            return
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_stage_duration(observation.stage_name, observation.elapsed_seconds)
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=observation.queue_name,
            operation_name=observation.operation_name,
            queue_depth=observation.queue_depth,
            queue_capacity=observation.queue_capacity,
            elapsed_seconds=observation.elapsed_seconds,
            blocked_seconds=observation.blocked_seconds,
        )

    def record_dosage_buffer_pool_operation(
        self,
        *,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> None:
        """Record dosage-buffer pool occupancy metadata."""
        if self.stage_timing_recorder is None:
            return
        observation = self.plan_dosage_buffer_pool_backpressure_observation(
            operation_name=operation_name,
            free_buffer_count=free_buffer_count,
            elapsed_seconds=elapsed_seconds,
            blocked=blocked,
        )
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=observation.queue_name,
            operation_name=observation.operation_name,
            queue_depth=observation.queue_depth,
            queue_capacity=observation.queue_capacity,
            elapsed_seconds=observation.elapsed_seconds,
            blocked_seconds=observation.blocked_seconds,
        )

    def record_bounded_resource_stage_duration(
        self,
        *,
        resource_name: str,
        operation_name: str,
        start_time: float,
        blocked: bool,
    ) -> None:
        """Record a bounded-resource stage duration plus pressure metadata."""
        if self.stage_timing_recorder is None:
            return
        elapsed_seconds = time.perf_counter() - start_time
        observation = self.plan_current_queue_stage_backpressure_observation(
            queue_name=resource_name,
            operation_name=operation_name,
            elapsed_seconds=elapsed_seconds,
            blocked=blocked,
        )
        self.stage_timing_recorder.add_stage_duration(observation.stage_name, observation.elapsed_seconds)
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=observation.queue_name,
            operation_name=observation.operation_name,
            queue_depth=observation.queue_depth,
            queue_capacity=observation.queue_capacity,
            elapsed_seconds=observation.elapsed_seconds,
            blocked_seconds=observation.blocked_seconds,
        )

    def plan_current_queue_backpressure_observation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> _core.NativeCallbackQueueBackpressureObservation:
        """Plan bounded-resource backpressure observation through the active native owner."""
        return self.callback_runtime_resources.plan_current_queue_backpressure_observation(
            queue_name,
            operation_name,
            elapsed_seconds,
            blocked,
        )

    def plan_current_queue_stage_backpressure_observation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> _core.NativeCallbackQueueStageBackpressureObservation:
        """Plan bounded-resource stage backpressure observation through the active native owner."""
        return self.callback_runtime_resources.plan_current_queue_stage_backpressure_observation(
            queue_name,
            operation_name,
            elapsed_seconds,
            blocked,
        )

    def record_dosage_buffer_pool_stage_duration(
        self,
        *,
        operation_name: str,
        free_buffer_count: int,
        start_time: float,
        blocked: bool,
    ) -> None:
        """Record dosage-buffer pool stage duration plus pressure metadata."""
        if self.stage_timing_recorder is None:
            return
        elapsed_seconds = time.perf_counter() - start_time
        observation = self.plan_dosage_buffer_pool_stage_backpressure_observation(
            operation_name=operation_name,
            free_buffer_count=free_buffer_count,
            elapsed_seconds=elapsed_seconds,
            blocked=blocked,
        )
        self.stage_timing_recorder.add_stage_duration(observation.stage_name, observation.elapsed_seconds)
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=observation.queue_name,
            operation_name=observation.operation_name,
            queue_depth=observation.queue_depth,
            queue_capacity=observation.queue_capacity,
            elapsed_seconds=observation.elapsed_seconds,
            blocked_seconds=observation.blocked_seconds,
        )

    def plan_dosage_buffer_pool_backpressure_observation(
        self,
        *,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> _core.NativeCallbackQueueBackpressureObservation:
        """Plan dosage-buffer pool backpressure observation through the active native owner."""
        return self.callback_runtime_resources.plan_dosage_buffer_pool_backpressure_observation(
            operation_name,
            free_buffer_count,
            elapsed_seconds,
            blocked,
        )

    def plan_dosage_buffer_pool_stage_backpressure_observation(
        self,
        *,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> _core.NativeCallbackQueueStageBackpressureObservation:
        """Plan dosage-buffer pool stage observation through the active native owner."""
        return self.callback_runtime_resources.plan_dosage_buffer_pool_stage_backpressure_observation(
            operation_name,
            free_buffer_count,
            elapsed_seconds,
            blocked,
        )

    def record_dosage_buffer_pool_reuse_operation(self, *, free_buffer_count: int) -> None:
        """Record native dosage-buffer reuse accounting."""
        if self.stage_timing_recorder is None:
            return
        self.record_dosage_buffer_pool_operation(
            operation_name=DOSAGE_BUFFER_POOL_REUSE_OPERATION,
            free_buffer_count=free_buffer_count,
            elapsed_seconds=0.0,
            blocked=False,
        )

    def record_dosage_buffer_pool_return_operation(self, *, free_buffer_count: int) -> None:
        """Record native dosage-buffer return accounting."""
        if self.stage_timing_recorder is None:
            return
        self.record_dosage_buffer_pool_operation(
            operation_name=DOSAGE_BUFFER_POOL_RETURN_OPERATION,
            free_buffer_count=free_buffer_count,
            elapsed_seconds=0.0,
            blocked=False,
        )

    def record_dosage_buffer_pool_allocate_operation(self, *, free_buffer_count: int) -> None:
        """Record native dosage-buffer allocation accounting."""
        if self.stage_timing_recorder is None:
            return
        self.record_dosage_buffer_pool_operation(
            operation_name=DOSAGE_BUFFER_POOL_ALLOCATE_OPERATION,
            free_buffer_count=free_buffer_count,
            elapsed_seconds=0.0,
            blocked=False,
        )

    def record_dosage_buffer_pool_discard_operation(self, *, free_buffer_count: int) -> None:
        """Record native dosage-buffer discard accounting."""
        if self.stage_timing_recorder is None:
            return
        self.record_dosage_buffer_pool_operation(
            operation_name=DOSAGE_BUFFER_POOL_DISCARD_OPERATION,
            free_buffer_count=free_buffer_count,
            elapsed_seconds=0.0,
            blocked=False,
        )

    def record_dosage_buffer_pool_consumer_wait_stage_duration(
        self,
        *,
        free_buffer_count: int,
        start_time: float,
    ) -> None:
        """Record native dosage-buffer consumer wait accounting."""
        if self.stage_timing_recorder is None:
            return
        self.record_dosage_buffer_pool_stage_duration(
            operation_name=DOSAGE_BUFFER_POOL_CONSUMER_WAIT_OPERATION,
            free_buffer_count=free_buffer_count,
            start_time=start_time,
            blocked=True,
        )

    def record_dosage_buffer_pool_operation_outcome(
        self,
        operation_outcome: _core.NativeCallbackResourceOperationOutcome,
    ) -> None:
        """Record a native dosage-buffer pool operation outcome."""
        self.record_queue_backpressure_observation(operation_outcome.dosage_buffer_pool_backpressure_observation)

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

    def plan_dosage_work_handoff(self, *, chunk_count: int) -> _core.NativeDosageWorkHandoffPlan:
        """Plan a dosage work-item handoff through the active native owner."""
        return self.callback_runtime_resources.plan_dosage_work_handoff(chunk_count)

    def plan_dosage_work_handoff_for_object(
        self,
        work_item: PreprocessedDosageWorkItem,
    ) -> _core.NativeDosageWorkHandoffPlan:
        """Plan a dosage work-item handoff from the queued object."""
        return self.callback_runtime_resources.plan_dosage_work_handoff_for_object(work_item)

    def put_dosage_work_item_with_native_delivery_timing(
        self,
        work_item: PreprocessedDosageWorkItem,
    ) -> None:
        """Put dosage work and attribute native-delivery timing through native policy."""
        if self.stage_timing_recorder is None:
            self.put_dosage_work_item(work_item)
            return
        native_delivery_start_time = time.perf_counter()
        try:
            self.put_dosage_work_item(work_item)
        finally:
            elapsed_seconds = time.perf_counter() - native_delivery_start_time
            self.record_work_item_stage_elapsed_duration(work_item, "native_delivery", elapsed_seconds)

    def plan_variant_major_dosage_batch_handoff(
        self,
        *,
        metadata_count: int,
        genotype_matrix_by_variant_count: int,
        chunk_stats_count: int,
    ) -> _core.NativeVariantMajorDosageBatchHandoffPlan:
        """Plan a variant-major batch handoff through the active native owner."""
        return self.callback_runtime_resources.plan_variant_major_dosage_batch_handoff(
            metadata_count=metadata_count,
            genotype_matrix_by_variant_count=genotype_matrix_by_variant_count,
            chunk_stats_count=chunk_stats_count,
        )

    def plan_variant_major_dosage_batch_handoff_for_sequences(
        self,
        metadata_batch: collections.abc.Sequence[typing.Any],
        genotype_matrix_by_variant_batch: collections.abc.Sequence[npt.NDArray[np.float32]],
        chunk_stats_batch: collections.abc.Sequence[_core.ChunkStats],
    ) -> _core.NativeVariantMajorDosageBatchHandoffPlan:
        """Plan a variant-major batch handoff from the input sequences."""
        return self.callback_runtime_resources.plan_variant_major_dosage_batch_handoff_for_sequences(
            metadata_batch,
            genotype_matrix_by_variant_batch,
            chunk_stats_batch,
        )

    def compute_preprocessed_dosage_chunk(
        self,
        metadata: typing.Any,
        genotype_matrix: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed dosage chunk for JAX association."""
        work_item = shared.PreprocessedDosageChunkWorkItem(
            metadata=metadata,
            genotype_matrix=genotype_matrix,
            chunk_stats=chunk_stats,
        )
        handoff_plan = self.plan_dosage_work_handoff_for_object(work_item)
        if handoff_plan.chunk_count != 1:
            message = "Native dosage work handoff plan disagrees with a single dosage work item."
            raise RuntimeError(message)
        self.put_dosage_work_item_with_native_delivery_timing(work_item)

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: typing.Any,
        genotype_matrix_by_variant: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed variant-major dosage chunk for JAX association."""
        work_item = shared.PreprocessedVariantMajorDosageChunkWorkItem(
            metadata=metadata,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            chunk_stats=chunk_stats,
        )
        handoff_plan = self.plan_dosage_work_handoff_for_object(work_item)
        if handoff_plan.chunk_count != 1:
            message = "Native dosage work handoff plan disagrees with a single variant-major dosage work item."
            raise RuntimeError(message)
        self.put_dosage_work_item_with_native_delivery_timing(work_item)

    def compute_preprocessed_variant_major_dosage_chunk_batch(
        self,
        metadata_batch: collections.abc.Sequence[typing.Any],
        genotype_matrix_by_variant_batch: collections.abc.Sequence[npt.NDArray[np.float32]],
        chunk_stats_batch: collections.abc.Sequence[_core.ChunkStats],
    ) -> None:
        """Enqueue a native batch of variant-major dosage chunks for JAX association."""
        batch_handoff_plan = self.plan_variant_major_dosage_batch_handoff_for_sequences(
            metadata_batch,
            genotype_matrix_by_variant_batch,
            chunk_stats_batch,
        )
        work_items = tuple(
            shared.PreprocessedVariantMajorDosageChunkWorkItem(
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
        work_item = shared.PreprocessedVariantMajorDosageChunkBatchWorkItem(work_items=work_items)
        dosage_handoff_plan = self.plan_dosage_work_handoff_for_object(work_item)
        if dosage_handoff_plan.chunk_count != batch_handoff_plan.chunk_count:
            message = "Native dosage work handoff plan disagrees with prepared batch work items."
            raise RuntimeError(message)
        self.put_dosage_work_item_with_native_delivery_timing(work_item)

    def compute_preprocessed_variant_major_packed8_probability_pair_chunk(
        self,
        metadata: typing.Any,
        packed_probability_pairs_by_variant: npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed packed8 chunk for JAX association."""
        work_item = shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem(
            metadata=metadata,
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            chunk_stats=chunk_stats,
        )
        handoff_plan = self.plan_dosage_work_handoff_for_object(work_item)
        if handoff_plan.chunk_count != 1:
            message = "Native dosage work handoff plan disagrees with a single packed8 dosage work item."
            raise RuntimeError(message)
        self.put_dosage_work_item_with_native_delivery_timing(work_item)

    def consume_dosage_chunks(self) -> None:
        """Consume queued dosage chunks and run JAX work in order."""
        try:
            self.consume_dosage_chunks_with_native_runtime_resources()
        except Exception as error:  # noqa: BLE001
            self.worker_error = error

    def consume_dosage_chunks_with_native_runtime_resources(self) -> None:
        """Consume queued native dosage chunks with optional timing observations."""
        runtime_resources = self.callback_runtime_resources
        while True:
            work_item_outcome = runtime_resources.get_next_dosage_work_item_outcome()
            work_item = typing.cast("QueuedPreprocessedDosageWorkItem", work_item_outcome.item)
            stage_backpressure_observation = work_item_outcome.stage_backpressure_observation
            if work_item_outcome.should_stop:
                return
            self.apply_dosage_work_item_dispatch_outcome(work_item_outcome)
            dosage_work_item = typing.cast(
                "PreprocessedDosageWorkItem",
                work_item,
            )
            if stage_backpressure_observation is None:
                self.process_dosage_work_item_with_dispatch_outcome(dosage_work_item, work_item_outcome)
                continue
            self.record_queue_stage_backpressure_observation(stage_backpressure_observation)
            python_callback_start_time = time.perf_counter()
            try:
                self.process_dosage_work_item_with_dispatch_outcome(dosage_work_item, work_item_outcome)
            finally:
                elapsed_seconds = time.perf_counter() - python_callback_start_time
                self.record_work_item_stage_elapsed_duration(dosage_work_item, "python_callback", elapsed_seconds)

    def process_dosage_work_item_with_dispatch_outcome(
        self,
        work_item: (
            shared.PreprocessedDosageChunkWorkItem
            | shared.PreprocessedVariantMajorDosageChunkWorkItem
            | shared.PreprocessedVariantMajorDosageChunkBatchWorkItem
            | shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
        ),
        dispatch_outcome: _core.NativeCallbackQueueOperationOutcome,
    ) -> None:
        """Run one preprocessed dosage work item using native dispatch outcome."""
        if dispatch_outcome.should_process_variant_major_dosage_batch:
            if not isinstance(work_item, shared.PreprocessedVariantMajorDosageChunkBatchWorkItem):
                message = "Native dosage work dispatch plan selected a mismatched variant-major batch item."
                raise RuntimeError(message)
            self.process_variant_major_dosage_batch_work_item(work_item)
            return
        if dispatch_outcome.should_process_variant_major_packed8_probability_pair:
            if not isinstance(work_item, shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem):
                message = "Native dosage work dispatch plan selected a mismatched packed8 item."
                raise RuntimeError(message)
            self.compute_preprocessed_variant_major_packed8_chunk(
                variant_metadata=work_item.metadata,
                packed_probability_pairs_by_variant=work_item.packed_probability_pairs_by_variant,
                chunk_stats=work_item.chunk_stats,
            )
        elif dispatch_outcome.should_process_variant_major_dosage:
            if not isinstance(work_item, shared.PreprocessedVariantMajorDosageChunkWorkItem):
                message = "Native dosage work dispatch plan selected a mismatched variant-major item."
                raise RuntimeError(message)
            self.compute_preprocessed_variant_major_chunk(
                variant_metadata=work_item.metadata,
                genotype_matrix_by_variant=work_item.genotype_matrix_by_variant,
                chunk_stats=work_item.chunk_stats,
            )
        elif dispatch_outcome.should_process_sample_major_dosage:
            if not isinstance(work_item, shared.PreprocessedDosageChunkWorkItem):
                message = "Native dosage work dispatch plan selected a mismatched sample-major item."
                raise RuntimeError(message)
            self.compute_preprocessed_chunk(
                variant_metadata=work_item.metadata,
                genotype_matrix=work_item.genotype_matrix,
                chunk_stats=work_item.chunk_stats,
            )
        else:
            message = "Native dosage work dispatch plan did not select a processing path."
            raise RuntimeError(message)
        self.record_progress(work_item.metadata)

    def process_variant_major_dosage_batch_work_item(
        self,
        work_item: shared.PreprocessedVariantMajorDosageChunkBatchWorkItem,
    ) -> None:
        """Run one native-planned variant-major dosage batch work item."""
        for chunk_work_item in work_item.work_items:
            self.compute_preprocessed_variant_major_chunk(
                variant_metadata=chunk_work_item.metadata,
                genotype_matrix_by_variant=chunk_work_item.genotype_matrix_by_variant,
                chunk_stats=chunk_work_item.chunk_stats,
            )
            self.record_progress(chunk_work_item.metadata)

    def apply_dosage_work_item_dispatch_outcome(
        self,
        dispatch_outcome: _core.NativeCallbackQueueOperationOutcome,
    ) -> None:
        """Raise native dosage work dispatch errors before processing."""
        if not dispatch_outcome.has_dispatch_error:
            return
        error_message = dispatch_outcome.dispatch_error_message
        if error_message is None:
            message = "Native dosage work dispatch plan omitted the error message."
            raise RuntimeError(message)
        raise RuntimeError(error_message)

    def record_progress(self, metadata: typing.Any) -> None:
        """Record throttled progress after one chunk is processed."""
        progress_update = self.callback_runtime_resources.record_progress_for_metadata(metadata)
        _core.emit_callback_progress_update_telemetry(
            self.telemetry_session, progress_update
        )

    def complete_progress(self) -> None:
        """Emit the native final progress completion event when telemetry consumed chunks."""
        progress_completion = self.finish_progress_state()
        _core.emit_callback_progress_completion_telemetry(
            self.telemetry_session,
            progress_completion,
        )

    def record_binary_null_model_failure_count(self, failure_count: int) -> None:
        """Accumulate binary null-model failures for run-level telemetry."""
        self.callback_runtime_resources.add_binary_null_model_failure_count(failure_count)

    def record_binary_correction_diagnostics(
        self,
        binary_chunk_diagnostics: regenie2_binary.BinaryChunkDiagnostics | None,
    ) -> None:
        """Accumulate binary correction diagnostics for run-level telemetry."""
        diagnostics_record_plan = self.callback_runtime_resources.plan_binary_correction_diagnostics_record_for_object(
            binary_chunk_diagnostics,
        )
        if not diagnostics_record_plan.should_record:
            return
        if binary_chunk_diagnostics is None:
            message = "Native binary correction diagnostics record plan selected a missing diagnostics payload."
            raise RuntimeError(message)
        self.binary_correction_pending_diagnostics.append(binary_chunk_diagnostics)

    def flush_binary_correction_diagnostics(self) -> None:
        """Materialize pending binary diagnostics and accumulate them into native summary counters."""
        summary_emit_plan = self.callback_runtime_resources.plan_binary_correction_summary_emit_for_pending_diagnostics(
            self.binary_correction_pending_diagnostics,
        )
        if not summary_emit_plan.should_flush_pending_diagnostics:
            return
        self.materialize_binary_correction_pending_diagnostics()

    def materialize_binary_correction_pending_diagnostics(self) -> None:
        """Materialize pending binary diagnostics into native summary counters."""
        pending_diagnostics = tuple(self.binary_correction_pending_diagnostics)
        if not pending_diagnostics:
            return
        diagnostics_counts = diagnostics.binary_chunk_diagnostics_to_summary_counts(pending_diagnostics)
        self.binary_correction_pending_diagnostics.clear()
        self.add_binary_correction_summary_counts(diagnostics_counts)

    def add_binary_correction_summary_counts(
        self,
        diagnostics_counts: regenie2_binary.BinaryCorrectionSummaryCounts,
    ) -> None:
        """Accumulate one host-materialized binary diagnostics summary."""
        self.callback_runtime_resources.add_binary_correction_diagnostics_totals(
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
        summary_emit_plan = self.callback_runtime_resources.plan_binary_correction_summary_emit_for_pending_diagnostics(
            self.binary_correction_pending_diagnostics,
        )
        if summary_emit_plan.should_flush_pending_diagnostics:
            self.flush_binary_correction_diagnostics()
        if not summary_emit_plan.should_emit_summary:
            return
        summary_payload = self.callback_runtime_resources.binary_correction_summary_payload()
        _core.emit_binary_correction_summary_telemetry(
            self.telemetry_session,
            summary_payload,
            "Native binary correction summary emit plan selected a missing telemetry session.",
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed JAX results and write them in order."""
        try:
            self.consume_result_write_items_with_native_runtime_resources()
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def consume_result_write_items_with_native_runtime_resources(self) -> None:
        """Consume queued native result write items with optional timing observations."""
        runtime_resources = self.callback_runtime_resources
        while True:
            work_item_outcome = runtime_resources.get_next_result_write_item_outcome()
            work_item = typing.cast("QueuedResultWriteWorkItem", work_item_outcome.item)
            if self.apply_result_write_drain_completion_outcome(work_item_outcome):
                return
            self.record_queue_stage_backpressure_observation(work_item_outcome.stage_backpressure_observation)
            self.apply_result_write_item_dispatch_outcome(work_item_outcome)
            if work_item_outcome.should_process_result_write_item:
                result_work_item = typing.cast("shared.Regenie2ResultWriteWorkItem", work_item)
                self.process_result_write_item(result_work_item)
                continue
            message = "Native result write dispatch plan did not select a single-result processing path."
            raise RuntimeError(message)

    def consume_multi_result_write_items_with_native_runtime_resources(self) -> None:
        """Consume queued native multi-result write items with optional timing observations."""
        runtime_resources = self.callback_runtime_resources
        while True:
            work_item_outcome = runtime_resources.get_next_result_write_item_outcome()
            work_item = typing.cast("QueuedResultWriteWorkItem", work_item_outcome.item)
            if self.apply_result_write_drain_completion_outcome(work_item_outcome):
                return
            self.record_queue_stage_backpressure_observation(work_item_outcome.stage_backpressure_observation)
            self.apply_result_write_item_dispatch_outcome(work_item_outcome)
            if work_item_outcome.should_process_multi_result_write_item:
                multi_work_item = typing.cast("shared.Regenie2MultiResultWriteWorkItem", work_item)
                self.process_multi_result_write_item(multi_work_item)
                continue
            message = "Native result write dispatch plan did not select a multi-result processing path."
            raise RuntimeError(message)

    def process_result_write_item(
        self,
        work_item: shared.Regenie2ResultWriteWorkItem,
    ) -> None:
        """Materialize and write one computed result work item."""
        host_dosage_buffer_released = False
        try:
            materialized_chunk = writers.materialize_regenie2_native_chunk_with_optional_timing(
                metadata=work_item.metadata,
                beta=work_item.beta,
                standard_error=work_item.standard_error,
                chi_squared=work_item.chi_squared,
                log10_p_value=work_item.log10_p_value,
                extra_code=work_item.extra_code,
                stage_timing_recorder=self.stage_timing_recorder,
                output_statistic_dtype=self.output_statistic_dtype,
            )
            host_dosage_buffer_released = self.release_result_work_item_host_buffer(work_item)
            writers.write_materialized_regenie2_native_chunk_with_optional_timing(
                writer_session=typing.cast("typing.Any", self).writer_session,
                metadata=work_item.metadata,
                chunk_stats=work_item.chunk_stats,
                materialized_chunk=materialized_chunk,
                stage_timing_recorder=self.stage_timing_recorder,
                output_statistic_dtype=self.output_statistic_dtype,
            )
            diagnostics.record_binary_chunk_diagnostics_from_count(
                stage_timing_recorder=self.stage_timing_recorder,
                diagnostics=work_item.binary_chunk_diagnostics,
            )
            self.record_binary_correction_diagnostics(work_item.binary_chunk_diagnostics)
        finally:
            self.release_result_work_item_final_resources(
                work_item,
                host_dosage_buffer_released=host_dosage_buffer_released,
            )

    def process_multi_result_write_item(self, multi_work_item: shared.Regenie2MultiResultWriteWorkItem) -> None:
        """Materialize and write one multi-result work item."""
        del multi_work_item
        raise NotImplementedError

    def put_dosage_work_item(
        self,
        work_item: QueuedPreprocessedDosageWorkItem,
    ) -> None:
        """Put work into the bounded worker queue while surfacing worker errors."""
        self.start()
        put_outcome = self.callback_runtime_resources.put_dosage_work_item_until_accepted_outcome(work_item)
        self.raise_worker_error_from_plan(put_outcome.worker_error_raise_plan)
        self.record_queue_stage_backpressure_observation(put_outcome.stage_backpressure_observation)

    def raise_worker_error_if_present(self) -> None:
        """Raise an asynchronous worker failure on the producer thread."""
        error_raise_plan = self.callback_runtime_resources.plan_worker_error_raise()
        self.raise_worker_error_from_plan(error_raise_plan)

    def raise_worker_error_from_plan(
        self,
        error_raise_plan: _core.NativeCallbackWorkerErrorRaisePlan | None,
    ) -> None:
        """Raise an asynchronous worker failure selected by a native plan."""
        if error_raise_plan is None:
            return
        if not error_raise_plan.should_raise:
            return
        error_message = error_raise_plan.error_message
        if error_message is None:
            message = "Native callback worker error raise plan omitted the error message."
            raise RuntimeError(message)
        if error_raise_plan.raise_dosage_worker_error:
            raise RuntimeError(error_message) from self.worker_error_cause
        if error_raise_plan.raise_result_worker_error:
            raise RuntimeError(error_message) from self.result_worker_error_cause
        message = "Native callback worker error raise plan did not select a worker."
        raise RuntimeError(message)

    def put_result_write_item(
        self,
        work_item: QueuedResultWriteWorkItem,
    ) -> None:
        """Put a computed result into the bounded materialization/write queue."""
        self.start()
        put_outcome = self.callback_runtime_resources.put_result_write_item_until_accepted_outcome(work_item)
        self.raise_worker_error_from_plan(put_outcome.worker_error_raise_plan)
        self.record_queue_stage_backpressure_observation(put_outcome.stage_backpressure_observation)

    def acquire_result_in_flight_slot(self) -> None:
        """Reserve capacity for one chunk of pending GPU result work."""
        acquire_outcome = self.callback_runtime_resources.acquire_result_in_flight_slot_until_available_outcome()
        self.raise_worker_error_from_plan(acquire_outcome.worker_error_raise_plan)
        self.record_queue_stage_backpressure_observation(acquire_outcome.stage_backpressure_observation)

    def release_result_in_flight_slot(self) -> None:
        """Release capacity for one completed chunk of GPU result work."""
        release_outcome = self.callback_runtime_resources.release_result_in_flight_slot_outcome()
        self.record_queue_backpressure_observation(release_outcome.backpressure_observation)

    def finish(self) -> None:
        """Wait until all queued JAX work has been written."""
        finish_result = self.callback_runtime_resources.finish_worker_lifecycle_for_pending_diagnostics(
            self.binary_correction_pending_diagnostics,
        )
        if finish_result.has_shutdown_timeout:
            worker_name = finish_result.shutdown_worker_name
            timeout_seconds = finish_result.shutdown_timeout_seconds
            if worker_name is None or timeout_seconds is None:
                message = "Native callback worker finish result omitted shutdown timeout details."
                raise RuntimeError(message)
            raise shared.NativeBgenWorkerShutdownError(
                worker_name=worker_name,
                timeout_seconds=timeout_seconds,
            )
        if finish_result.raise_worker_error:
            self.raise_worker_error_if_present()
        progress_completion_event = finish_result.progress_completion_event
        _core.emit_callback_progress_event_telemetry(
            self.telemetry_session,
            progress_completion_event,
            "Native callback worker finish result selected a missing telemetry session.",
        )
        if finish_result.emit_binary_correction_summary:
            summary_payload = finish_result.binary_correction_summary_payload
            if summary_payload is None and finish_result.flush_binary_correction_pending_diagnostics:
                self.materialize_binary_correction_pending_diagnostics()
                summary_payload = self.callback_runtime_resources.binary_correction_summary_payload()
            _core.emit_binary_correction_summary_telemetry(
                self.telemetry_session,
                summary_payload,
                "Native callback worker finish result selected a missing telemetry session.",
            )

    def abort(self) -> None:
        """Stop the worker after an upstream failure."""
        self.callback_runtime_resources.abort_worker_lifecycle()

    def stop_dosage_worker(self, timeout_seconds: float | None) -> None:
        """Signal the dosage worker to exit after queued dosage chunks drain."""
        timeout_result = self.callback_runtime_resources.stop_dosage_worker(timeout_seconds)
        if timeout_result is None:
            return
        raise shared.NativeBgenWorkerShutdownError(
            worker_name=self.dosage_worker_name,
            timeout_seconds=timeout_result,
        )

    def join_dosage_worker(self, timeout_seconds: float | None) -> None:
        """Join the dosage worker with a bounded shutdown wait."""
        timeout_result = self.callback_runtime_resources.join_dosage_worker(timeout_seconds)
        if timeout_result is None:
            return
        raise shared.NativeBgenWorkerShutdownError(
            worker_name=self.dosage_worker_name,
            timeout_seconds=timeout_result,
        )

    def stop_result_worker(self, timeout_seconds: float | None) -> None:
        """Signal the result worker to exit after queued results drain."""
        timeout_result = self.callback_runtime_resources.stop_result_worker(timeout_seconds)
        if timeout_result is None:
            return
        raise shared.NativeBgenWorkerShutdownError(
            worker_name=self.result_worker_name,
            timeout_seconds=timeout_result,
        )

    def join_result_worker(self, timeout_seconds: float | None) -> None:
        """Join the result writer worker with a bounded shutdown wait."""
        timeout_result = self.callback_runtime_resources.join_result_worker(timeout_seconds)
        if timeout_result is None:
            return
        raise shared.NativeBgenWorkerShutdownError(
            worker_name=self.result_worker_name,
            timeout_seconds=timeout_result,
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

    def acquire_dosage_buffer_from_native_resources(
        self,
        expected_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> shared.HostGenotypeBuffer:
        """Acquire a host dosage buffer using native resource-owner waits and storage."""
        acquire_outcome = self.callback_runtime_resources.acquire_reusable_dosage_buffer_or_allocate_outcome(
            expected_shape,
            dtype,
        )
        self.raise_worker_error_from_plan(acquire_outcome.worker_error_raise_plan)
        self.record_queue_stage_backpressure_observation(acquire_outcome.stage_backpressure_observation)
        self.record_dosage_buffer_pool_operation_outcome(acquire_outcome)
        if acquire_outcome.should_allocate:
            return self.allocate_dosage_buffer_with_shape(expected_shape, dtype)
        if acquire_outcome.dosage_buffer is not None:
            return typing.cast("shared.HostGenotypeBuffer", acquire_outcome.dosage_buffer)
        message = "Native callback runtime returned no reusable dosage buffer and did not request allocation."
        raise RuntimeError(message)

    def acquire_dosage_buffer_with_shape(
        self,
        expected_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> shared.HostGenotypeBuffer:
        """Return a reusable host dosage buffer with the requested shape."""
        return self.acquire_dosage_buffer_from_native_resources(expected_shape, dtype)

    def release_dosage_buffer(self, dosage_buffer: shared.HostGenotypeBuffer) -> None:
        """Return a processed host dosage buffer to the reusable pool."""
        operation_outcome = self.callback_runtime_resources.release_dosage_buffer_outcome(
            dosage_buffer,
        )
        self.record_dosage_buffer_pool_operation_outcome(operation_outcome)

    def allocate_dosage_buffer_with_shape(
        self,
        expected_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> shared.HostGenotypeBuffer:
        """Allocate and register one host genotype buffer slot."""
        dosage_buffer = typing.cast("shared.HostGenotypeBuffer", np.empty(expected_shape, dtype=dtype, order="C"))
        operation_outcome = self.callback_runtime_resources.register_dosage_buffer_outcome(dosage_buffer)
        self.record_dosage_buffer_pool_operation_outcome(operation_outcome)
        return dosage_buffer

    def discard_dosage_buffer_slot(self, dosage_buffer: shared.HostGenotypeBuffer) -> None:
        """Remove one discarded host genotype buffer slot from pool accounting."""
        operation_outcome = self.callback_runtime_resources.discard_dosage_buffer_outcome(dosage_buffer)
        self.record_dosage_buffer_pool_operation_outcome(operation_outcome)

    def release_numpy_dosage_buffer(self, dosage_buffer: jax.Array | shared.HostGenotypeBuffer) -> None:
        """Return a NumPy host dosage buffer to the pool after device transfer."""
        operation_outcome = self.callback_runtime_resources.release_numpy_dosage_buffer_outcome(
            dosage_buffer,
        )
        self.record_dosage_buffer_pool_operation_outcome(operation_outcome)

    def get_releasable_dosage_buffer(
        self,
        dosage_buffer: jax.Array | shared.HostGenotypeBuffer,
    ) -> shared.HostGenotypeBuffer | None:
        """Return a host dosage buffer reference when it belongs to the reusable pool."""
        return typing.cast(
            "shared.HostGenotypeBuffer | None",
            self.callback_runtime_resources.get_releasable_dosage_buffer_owner(dosage_buffer),
        )

    def record_result_work_item_resource_release_result(
        self,
        release_outcome: _core.NativeCallbackResourceOperationOutcome,
    ) -> None:
        """Record Python-side telemetry from native result work item resource cleanup."""
        self.record_queue_backpressure_observation(
            release_outcome.dosage_buffer_pool_backpressure_observation,
        )
        self.record_queue_backpressure_observation(
            release_outcome.result_in_flight_backpressure_observation,
        )

    def release_result_work_item_buffer(
        self,
        work_item: shared.Regenie2ResultWriteWorkItem | shared.Regenie2MultiResultWriteWorkItem,
    ) -> None:
        """Release resources after a dependent JAX result is materialized."""
        host_dosage_buffer_released = self.release_result_work_item_host_buffer(work_item)
        self.release_result_work_item_final_resources(
            work_item,
            host_dosage_buffer_released=host_dosage_buffer_released,
        )

    def apply_result_write_drain_completion_outcome(
        self,
        drain_outcome: _core.NativeCallbackQueueOperationOutcome,
    ) -> bool:
        """Apply native result write drain outcome side effects."""
        if drain_outcome.should_flush_binary_correction_diagnostics:
            self.flush_binary_correction_diagnostics()
        return drain_outcome.should_stop

    def apply_result_write_item_dispatch_outcome(
        self,
        dispatch_outcome: _core.NativeCallbackQueueOperationOutcome,
    ) -> None:
        """Raise native result write dispatch errors before processing."""
        if not dispatch_outcome.has_dispatch_error:
            return
        error_message = dispatch_outcome.dispatch_error_message
        if error_message is None:
            message = "Native result write dispatch plan omitted the error message."
            raise RuntimeError(message)
        raise RuntimeError(error_message)

    def release_result_work_item_host_buffer(
        self,
        work_item: shared.Regenie2ResultWriteWorkItem | shared.Regenie2MultiResultWriteWorkItem,
    ) -> bool:
        """Release the host genotype buffer associated with one result."""
        release_outcome = self.callback_runtime_resources.release_result_work_item_pre_write_resources_outcome(
            work_item,
        )
        self.record_result_work_item_resource_release_result(release_outcome)
        return release_outcome.released_host_buffer

    def release_result_work_item_final_resources(
        self,
        work_item: shared.Regenie2ResultWriteWorkItem | shared.Regenie2MultiResultWriteWorkItem,
        *,
        host_dosage_buffer_released: bool,
    ) -> None:
        """Release final resources associated with one result work item."""
        release_outcome = self.callback_runtime_resources.release_result_work_item_final_resources_outcome(
            work_item,
            host_dosage_buffer_released,
        )
        self.record_result_work_item_resource_release_result(release_outcome)

    def release_result_work_item_in_flight_slot(
        self,
        work_item: shared.Regenie2ResultWriteWorkItem | shared.Regenie2MultiResultWriteWorkItem,
    ) -> None:
        """Release the in-flight result slot associated with one result."""
        release_outcome = self.callback_runtime_resources.release_result_work_item_in_flight_slot_outcome(
            work_item,
        )
        self.record_result_work_item_resource_release_result(release_outcome)
